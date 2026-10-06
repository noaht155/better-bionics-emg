"""The existing programs, run inside the web app on the recorder's armband stream.

Each tool runs in its own thread and keeps a small state dict for the browser. The logic lives in the
original scripts (check_connection.report, calibrate.compute, haptics.output_duties, esp32_link.self_test,
ml.gesture_model's train and LivePredictor, ml.ringnet's NetPredictor), this only feeds them the shared stream
instead of opening the armband again. Network training and evaluation run as their own process (NetworkJob).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import calibrate
import check_connection
from esp32_link import Esp32, self_test
from haptics import OUTPUTS, UPDATE_HZ, output_duties

# The ml package sits next to emg-reading/ in the repo
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ml import gesture_model  # noqa: E402

# Lines of a network job's output kept for the page
JOB_LINES = 40

CHECK_S = 5.0
COUNTDOWN_S = 3


class Tool:
    def __init__(self):
        self.state = {"running": False, "error": None}
        self._thread = None

    def _launch(self, target, *args):
        if self.state["running"]:
            raise ValueError("already running")
        self.state = {"running": True, "error": None}
        self._thread = threading.Thread(target=self._wrap, args=(target, *args), daemon=True)
        self._thread.start()

    def join(self, timeout):
        if self._thread is not None:
            self._thread.join(timeout)

    def _wrap(self, target, *args):
        try:
            target(*args)
        # Anything that goes wrong in the thread ends up on the page instead of a traceback nobody sees
        except (Exception, SystemExit) as e:
            self.state["error"] = str(e) or type(e).__name__
        finally:
            self.state["running"] = False


class ConnectionCheck(Tool):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def start(self):
        self._launch(self._run)

    def _run(self):
        self.state.update(until=time.time() + CHECK_S, total=CHECK_S)
        data = self.recorder.capture(CHECK_S)
        band = self.recorder.band
        self.state["lines"] = [f"streaming from {band.source} for {CHECK_S:g} s"] + check_connection.report(
            data, band.rows, band.rate, CHECK_S, battery=band.source == "armband")


class Calibration(Tool):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def start(self, seconds, car):
        if self.recorder.session is not None:
            raise ValueError("stop the recording first")
        self._launch(self._run, seconds, car)

    def _phase(self, prompt, seconds):
        # Same steps as calibrate.record: count down, record, drop the late reaction at the start
        for n in range(COUNTDOWN_S, 0, -1):
            self.state.update(prompt=prompt, countdown=n, until=None)
            time.sleep(1)
        total = calibrate.REACTION_S + seconds
        self.state.update(countdown=None, until=time.time() + total, total=total)
        data = self.recorder.capture(calibrate.REACTION_S + seconds)[self.recorder.band.emg_rows]
        return data[:, int(calibrate.REACTION_S * self.recorder.band.rate):]

    def _run(self, seconds, car):
        relaxed = self._phase("RELAX your arm and keep it still", seconds)
        squeezed = self._phase("SQUEEZE a firm fist and hold it", seconds)
        self.state.update(prompt=None, until=None)
        if relaxed.shape[1] == 0 or squeezed.shape[1] == 0:
            raise SystemExit("no data received, check the connection")
        band = self.recorder.band
        result, lines = calibrate.compute(relaxed, squeezed, band.rate, band.channels, car)
        calibrate.save(result, car)
        self.state["lines"] = lines + [f"saved {calibrate.CAL_FILE.name}"]


class Haptics(Tool):
    """haptics.py's loop. It shares the ESP32 with the output test, only one of them can hold the port."""

    def __init__(self, recorder, esp_lock):
        super().__init__()
        self.recorder = recorder
        self.esp_lock = esp_lock
        self._stop = threading.Event()

    def start(self, port, wifi, session=None):
        """With session, the levels come from that recorded session (calibrate.from_session), else from
        calibration.json."""
        if session is not None:
            levels, _ = calibrate.from_session(session)
            ranges, car = {ch: (v["rest"], v["max"]) for ch, v in levels.items()}, True
            source = f"levels from session {Path(session).name}"
        else:
            try:
                loaded = calibrate.load_calibration()
            except SystemExit as e:
                raise ValueError(str(e))
            if loaded is None:
                raise ValueError("no calibration yet, run the calibration first")
            ranges, car = loaded
            source = "levels from calibration.json"
        missing = [ch for ch in self.recorder.band.channels if ch not in ranges]
        if missing:
            raise ValueError(f"calibration has no entry for channels {missing}, calibrate again")
        if not self.esp_lock.acquire(blocking=False):
            raise ValueError("the ESP32 is busy with the output test")
        self._stop.clear()
        try:
            self._launch(self._run, port, wifi, ranges, car, source)
        except ValueError:
            self.esp_lock.release()
            raise

    def stop(self):
        self._stop.set()

    def _run(self, port, wifi, ranges, car, source):
        band = self.recorder.band
        self.state.update(duties=[0] * len(OUTPUTS), lost=[False] * len(band.channels), car=car, watchdog=0,
                          outputs=[list(chs) for chs in OUTPUTS], source=source)
        try:
            with Esp32(port, wifi) as esp:
                self.state["link"] = esp.link.name
                self._log_session(True, esp.link.name)
                next_tick = time.perf_counter()
                while not self._stop.is_set():
                    raw, filtered = self.recorder.window(self.recorder.env_len)
                    if raw.shape[1] == self.recorder.env_len:
                        duties, lost = output_duties(raw, filtered, band.channels, ranges, car)
                        esp.set_outputs(duties)
                        self.state.update(duties=duties, lost=lost.tolist())
                    if esp.watchdog_fired:
                        self.state["watchdog"] += 1
                        esp.watchdog_fired = False
                    # If a slow reply made this update late, don't try to catch up with a burst
                    next_tick = max(next_tick + 1 / UPDATE_HZ, time.perf_counter())
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
        finally:
            self._log_session(False)
            self.esp_lock.release()

    def _log_session(self, on, link=None):
        # Motor vibration might show up in the EMG, so recordings note when the haptics ran
        session = self.recorder.session
        if session is not None:
            session.add_event("haptics", on=on, link=link)


class OutputTest(Tool):
    def __init__(self, esp_lock):
        super().__init__()
        self.esp_lock = esp_lock

    def start(self, port, wifi):
        if not self.esp_lock.acquire(blocking=False):
            raise ValueError("stop the haptics first")
        try:
            self._launch(self._run, port, wifi)
        except ValueError:
            self.esp_lock.release()
            raise

    def _run(self, port, wifi):
        self.state["lines"] = []
        try:
            with Esp32(port, wifi) as board:
                self_test(board, self.state["lines"].append)
        finally:
            self.esp_lock.release()


class Predictor(Tool):
    """Runs the active gesture model on the live stream, one prediction per window step."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self._stop = threading.Event()
        # Kept here so it stays the same when another model is picked
        self.threshold = gesture_model.THRESHOLD
        self._live = None

    def set_threshold(self, value):
        self.threshold = value
        if self._live is not None:
            self._live.threshold = value
        self.state["threshold"] = value

    def use(self, path):
        if path.suffix == ".pt":
            # torch is only installed on the training desktop, the rest of the app runs without it
            import torch
            from ml import ringnet
            torch.set_num_threads(1)
            model = ringnet.load_live(path)
        else:
            model = gesture_model.load(path)
        band = self.recorder.band
        if model["channels"] != band.channels or model["rate"] != band.rate:
            raise ValueError("this model was trained on other EMG channels or another sample rate")
        self.stop()
        self.join(2)
        self._stop.clear()
        self._launch(self._run, model, path.name)
        return model

    def stop(self):
        self._stop.set()

    def _run(self, model, name):
        if model.get("kind") == "network":
            from ml.ringnet import NetPredictor
            live = NetPredictor(model, self.threshold, gesture_model.UNSURE)
        else:
            live = gesture_model.LivePredictor(model)
        live.threshold = self.threshold
        self._live = live
        self.state.update(model=name, classes=model["classes"], label=None, threshold=self.threshold)
        period = model["step_ms"] / 1000
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            filtered = self.recorder.window(live.win)[1]
            if filtered.shape[1] == live.win:
                accel = self.recorder.accel_window(live.win) if model["options"]["accel"] else None
                label, raw, probs = live.predict(filtered, accel)
                # Only networks predict finger angles, LDA models leave these None
                self.state.update(label=label, raw=raw, probs=probs, angles=getattr(live, "angles", None),
                                  effort_angles=getattr(live, "effort_angles", None))
            next_tick = max(next_tick + period, time.perf_counter())
            time.sleep(max(0.0, next_tick - time.perf_counter()))


class Trainer(Tool):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def start(self, folders, options):
        self._launch(self._run, folders, options)

    def _run(self, folders, options):
        self.state["step"] = f"loading {len(folders)} sessions and evaluating"
        model, result = gesture_model.train(folders, options)
        path = gesture_model.save(model)
        self.state.update(step=None, model=path.name, lines=[
            f"{model['windows']} labelled windows from {len(model['sessions'])} sessions, "
            f"classes {', '.join(model['classes'])}"] + gesture_model.report(result) + [f"saved {path.name}"])
        self.predictor.use(path)


class NetworkJob(Tool):
    """Runs one of the network scripts (ml.train_ringnet, ml.networks) as a separate process at low priority, so
    training can't stall the armband and camera loops even mid-recording. Its output lines and the result file it
    writes are passed to the page."""

    def __init__(self):
        super().__init__()
        self._proc = None

    def start(self, job, args):
        self._launch(self._run, job, args)

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self, job, args):
        fd, out = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        lines = deque(maxlen=JOB_LINES)
        self.state.update(job=job, lines=[], result=None)
        try:
            self._proc = subprocess.Popen([sys.executable, "-u", "-m", *args, "--json", out], cwd=REPO,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in self._proc.stdout:
                # Library warnings would push the useful lines off the page
                if line.strip() and "Warning" not in line:
                    lines.append(line.rstrip())
                    self.state["lines"] = list(lines)
            if self._proc.wait() != 0:
                raise RuntimeError(lines[-1] if lines else f"stopped (exit code {self._proc.returncode})")
            text = Path(out).read_text()
            self.state["result"] = json.loads(text) if text else None
        finally:
            Path(out).unlink(missing_ok=True)


class Tools:
    def __init__(self, recorder):
        esp_lock = threading.Lock()
        self.check = ConnectionCheck(recorder)
        self.calibration = Calibration(recorder)
        self.haptics = Haptics(recorder, esp_lock)
        self.test = OutputTest(esp_lock)
        self.predictor = Predictor(recorder)
        self.trainer = Trainer(self.predictor)
        self.network = NetworkJob()

    def snapshot(self):
        now = time.time()
        out = {}
        for name in ("check", "calibration", "haptics", "test", "predictor", "trainer", "network"):
            state = dict(getattr(self, name).state)
            if state.get("until"):
                state["left"] = max(0.0, state["until"] - now)
            out[name] = state
        return out
