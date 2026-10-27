"""The existing programs, run inside the web app on the recorder's armband stream.

Each tool runs in its own thread and keeps a small state dict for the browser. The logic lives in the
original scripts (check_connection.report, calibrate.compute, haptics.output_duties, esp32_link.self_test,
ml.gesture_model's train and LivePredictor, ml.ringnet's NetPredictor, control.Controller), this only feeds them the
shared stream instead of opening the armband again. Network training and evaluation run as their own process
(NetworkJob).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import calibrate
import check_connection
import delay
import devices
import hub
import hum_test
import mac_wifi
from control import DEFAULT_RULES, DEFAULTS, STILL_S, Controller, check_rules, movement, split_rules, still_limit
from cursor import DEFAULTS as CURSOR_DEFAULTS, LEVEL_SCALE as CURSOR_LEVEL_SCALE, Cursor
from esp32_link import Esp32, self_test
from gestures import GESTURE_SET, GESTURES, LIBRARY
from session import DATA_DIR, load_session, segments
from haptics import OUTPUTS, UPDATE_HZ, channel_levels, effort, output_duties

# The ml package sits next to emg-reading/ in the repo
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ml import NETWORK_DIR, gesture_model  # noqa: E402

# Lines of a network job's output kept for the page
# Output lines kept for the page's log, the newest are the ones shown first
JOB_LINES = 200
# What the network scripts print (train_ringnet.epoch_printer and the fold loop), read for the Train tab's bar
FOLD_LINE = re.compile(r"fold (\d+)/(\d+)")
EPOCH_LINE = re.compile(r"epoch (\d+)/(\d+)(?:.*about (.+) left)?")

CHECK_S = 5.0
COUNTDOWN_S = 3
WIFI_SCRIPT = Path(__file__).with_name("connect_armband.ps1")
WIFI_CHECK_S = 2.0
# One profile line of `netsh wlan show profiles`, e.g. "    All User Profile     : Mindrove_ARB_66bb04"
WIN_PROFILE_LINE = re.compile(rf"(?m)^\s*[\w ]+Profile\s*:\s*({mac_wifi.ARMBAND_PREFIX}.*?)\s*$")
# Random gaps between mode switches, so a switch never lines up with the band's packet rhythm
DELAY_GAP_S = (0.3, 0.6)
# The page's clock and this computer's agree to a few ms when the page runs here
DELAY_CLOCK_S = 0.05
DELAY_FLASH_TIMEOUT_S = 120
# The Control tab's simulated model steps as often as the models do (50 ms window step)
SIM_STEP_S = 0.05
# How often the header's sync line asks the hub, and how long it waits for an answer. The Mac has one WiFi, so the hub
# is out of reach while it records on the armband's network
SYNC_CHECK_S = 30
SYNC_BUSY_CHECK_S = 2
SYNC_TIMEOUT_S = 5
# How often a quick calibration on the hub checks on the hub's job
HUB_POLL_S = 1.0


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


class ArmbandWifi(Tool):
    """Joins the armband's WiFi and then reopens the armband stream, since the band needs a new start request after
    that. On Windows through connect_armband.ps1 (forget and rejoin, DHCP, back to the static IP; asks for admin
    rights), on a Mac through mac_wifi."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def _check(self):
        if self.recorder.band.source != "armband":
            raise ValueError(f"the app reads from {self.recorder.band.source}, not the armband")
        if self.recorder.session is not None:
            raise ValueError("a session is recording, the stream would drop")

    def start(self, dhcp_s=5):
        self._check()
        if sys.platform == "darwin":
            self._launch(self._run_mac)
        elif sys.platform == "win32":
            self._launch(self._run, dhcp_s)
        else:
            raise ValueError("only works on Windows and macOS")

    def _run_mac(self):
        dev = mac_wifi.device()
        band = self.recorder.band
        lines = []

        def say(line):
            lines.append(line)
            self.state["lines"] = list(lines)
        now = mac_wifi.current(dev)
        try:
            if now and now.startswith(mac_wifi.ARMBAND_PREFIX):
                say(f"Already on {now}")
            else:
                found = mac_wifi.armbands()
                if found is None:
                    raise RuntimeError("this macOS can't scan for networks from the app: join the Mindrove_ARB_ "
                                       "network from the WiFi menu (password #mindrove), then press again")
                if not found:
                    raise RuntimeError("no Mindrove_ARB_ network in range, is the armband switched on?")
                say(f"Found {', '.join(found)}")
                say("Closing the armband stream before the network changes")
                band.close()
                say(f"Joining {found[0]}" + (f", leaving {now} (no hub until the Mac is back on it)" if now else ""))
                mac_wifi.join(dev, found[0], mac_wifi.ARMBAND_PASSWORD)
            say(f"Address {mac_wifi.wait_for_address(dev)}")
            self._reopen(lines)
        finally:
            self._reconnect_if_lost()

    def _reopen(self, lines):
        band = self.recorder.band
        self.state["lines"] = lines = lines + ["Reopening the armband stream"]
        band.open()
        before = self.recorder.received
        time.sleep(WIFI_CHECK_S)
        got = self.recorder.received - before
        if not got:
            raise RuntimeError(f"stream reopened but no samples in {WIFI_CHECK_S:g} s, switch the armband off and "
                               "on and try again")
        self.state["lines"] = lines + [f"Streaming: {got} samples in {WIFI_CHECK_S:g} s "
                                       f"(about {band.rate * WIFI_CHECK_S:.0f} expected)"]

    def _reconnect_if_lost(self):
        # A failed run still tries to get the stream back; if the band can't be reached the warning bar says so
        band = self.recorder.band
        if not band.connected:
            try:
                band.open()
            except Exception:
                pass

    def _run(self, dhcp_s):
        fd, log = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        band = self.recorder.band
        first = ["Closing the armband stream before the network changes",
                 f"Starting {WIFI_SCRIPT.name}, DHCP {dhcp_s} s"]
        self.state["lines"] = first
        band.close()
        try:
            proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                     str(WIFI_SCRIPT), "-DhcpSeconds", str(dhcp_s), "-Log", log],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while proc.poll() is None:
                time.sleep(0.5)
                self.state["lines"] = first + _read_lines(log)
            lines = first + _read_lines(log) + [line for line in proc.stdout.read().splitlines() if line]
            self.state["lines"] = lines
            if proc.returncode != 0:
                raise RuntimeError(lines[-1] if len(lines) > 1 else f"the script failed (exit code {proc.returncode})")
            self._reopen(lines)
        finally:
            Path(log).unlink(missing_ok=True)
            self._reconnect_if_lost()


def forget_armband_wifi():
    """Forgets the armband's network when the app closes, so this computer doesn't join it by itself later and take
    the band while another computer records. Connect armband WiFi joins it again. Returns what it did."""
    if sys.platform == "darwin":
        return mac_wifi.forget(mac_wifi.device())
    if sys.platform != "win32":
        return []
    out = subprocess.run(["netsh", "wlan", "show", "profiles"], capture_output=True, text=True).stdout
    done = []
    # Deleting a profile needs no admin rights, unlike the address changes in connect_armband.ps1
    for name in WIN_PROFILE_LINE.findall(out):
        gone = subprocess.run(["netsh", "wlan", "delete", "profile", f"name={name}"], capture_output=True).returncode == 0
        done.append(f"{'forgot' if gone else 'could not forget'} {name}")
    return done


class RoundTrip(Tool):
    """Delay test (System > Dev), armband half: switches the band between TEST and EEG mode at random gaps and times
    each switch's return in the EMG stamps (delay.round_trips). Ends in EEG mode whatever happens. Results go to the
    shared measured dict until they are saved to the setup (Tools.save_delay)."""

    def __init__(self, recorder, haptics, measured):
        super().__init__()
        self.recorder = recorder
        self.haptics = haptics
        self.measured = measured

    def start(self, switches):
        if self.recorder.band.source != "armband":
            raise ValueError(f"the app reads from {self.recorder.band.source}, not the armband")
        if not self.recorder.band.connected:
            raise ValueError("armband not connected")
        if self.recorder.session is not None:
            raise ValueError("a session is recording")
        if self.haptics.state["running"]:
            raise ValueError("stop the haptics first, the test signal would drive the outputs")
        self._launch(self._run, switches)

    def _run(self, count):
        band = self.recorder.band
        gaps = np.random.uniform(DELAY_GAP_S[0], DELAY_GAP_S[1], count)
        total = gaps.sum() + 1.0
        self.state.update(until=time.time() + total, total=total)
        captured = {}
        capture = threading.Thread(target=lambda: captured.update(data=self.recorder.capture(total)), daemon=True)
        capture.start()
        sent = []
        try:
            time.sleep(0.5)
            for i, gap in enumerate(gaps):
                sent.append(band.set_mode("TEST" if i % 2 == 0 else "EEG"))
                time.sleep(gap)
        finally:
            band.set_mode("EEG")
        capture.join()
        trips = delay.round_trips(captured["data"], band.rows, sent)
        if len(trips) < count // 2:
            raise RuntimeError(f"only {len(trips)} of {count} switches came back, is the armband streaming?")
        lo, hi = np.percentile(trips, [10, 90]) * 1000
        line = (f"Round trip {np.median(trips) * 1000:.1f} ms median over {len(trips)} of {count} switches "
                f"(10 to 90 %: {lo:.1f} to {hi:.1f} ms)")
        self.measured["round_trip"] = (float(np.median(trips)), line)
        self.state["lines"] = [line] + _delay_summary(self.measured)


class ScreenFlash(Tool):
    """Delay test (System > Dev), camera half: logs the brightness of every camera frame while the page flashes the
    screen, then times each flash in the frames (delay.screen_delays). The page sends its flash times when done."""

    def __init__(self, recorder, measured):
        super().__init__()
        self.recorder = recorder
        self.measured = measured
        self._done = threading.Event()
        self._switches = None

    def start(self, page_ms):
        camera = self.recorder.camera
        if camera is None or not camera.ok:
            raise ValueError("no camera")
        if self.recorder.session is not None:
            raise ValueError("a session is recording")
        # The flash times come from the page's clock, they only mean something on the computer with the camera
        if abs(page_ms / 1000 - time.time()) > DELAY_CLOCK_S:
            raise ValueError("open this page on the computer the camera is plugged into")
        self._done.clear()
        self._switches = None
        self._launch(self._run, camera)

    def finish(self, switches):
        if not self.state["running"]:
            raise ValueError("the flash test isn't running")
        self._switches = switches
        self._done.set()

    def _run(self, camera):
        camera.probe(True)
        try:
            finished = self._done.wait(DELAY_FLASH_TIMEOUT_S)
        finally:
            log = camera.probe(False)
        if not finished or not self._switches:
            raise RuntimeError("the page stopped before the flashes were done")
        on, off = delay.screen_delays(log, self._switches)
        both = np.concatenate([on, off])
        if len(both) < len(self._switches) // 2:
            raise RuntimeError(f"only {len(both)} of {len(self._switches)} flashes seen, hold the mirror so the "
                               "camera sees the screen")
        camera_s = delay.camera_delay(on, off)
        line = (f"Camera + screen {camera_s * 1000:.1f} ms over {len(both)} of {len(self._switches)} flashes (to light "
                f"{_ms(on)}, to dark {_ms(off)}, the mean of the two), {camera.fps:.0f} fps")
        self.measured["camera"] = (camera_s, line)
        self.state["lines"] = [line] + _delay_summary(self.measured)


def _ms(delays):
    return f"{np.median(delays) * 1000:.0f} ms" if len(delays) else "none seen"


def _delay_summary(measured):
    if "camera" not in measured or "round_trip" not in measured:
        return ["Run the other half too for the label shift"]
    cam, trip = measured["camera"][0], measured["round_trip"][0]
    return [f"Label shift {(cam - trip) * 1000:.0f} to {cam * 1000:.0f} ms, middle "
            f"{delay.label_shift(cam, trip) * 1000:.0f} ms"]


def _read_lines(path):
    try:
        return [line for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line]
    except OSError:
        return []


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


class HumTest(Tool):
    """Hum test (System > Dev): one block of the desktop vs MacBook comparison, recorded from the running app and
    saved like hum_test.py record, then the tables over every block recorded on this computer."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self._stop = threading.Event()

    def start(self, label, seconds):
        if label not in dict(hum_test.BLOCKS):
            raise ValueError(f"unknown block {label}")
        if self.recorder.session is not None:
            raise ValueError("a session is recording")
        if not self.recorder.band.connected:
            raise ValueError("armband not connected")
        self._stop.clear()
        self._launch(self._run, label, seconds)

    def stop(self):
        self._stop.set()

    def _run(self, label, seconds):
        for n in range(COUNTDOWN_S, 0, -1):
            self.state.update(prompt=f"{dict(hum_test.BLOCKS)[label]}, arm relaxed and still", countdown=n)
            if self._stop.wait(1):
                break
        self.state.update(countdown=None, until=time.time() + seconds, total=seconds)
        band = self.recorder.band
        if not self._stop.is_set():
            board = self.recorder.capture(seconds, self._stop)
        if self._stop.is_set():
            self.state.update(prompt=None, until=None, lines=[f"{label} cancelled, nothing saved"])
            return
        if board.shape[1] < seconds * band.rate / 2:
            raise SystemExit(f"only {board.shape[1]} samples arrived, check the connection")
        replaced = (hum_test.OUT_DIR / f"{label}.npy").exists()
        dropped = hum_test.save(label, board, band.rows, band.rate, band.source)
        self.state.update(prompt=None, until=None)
        self.state["lines"] = ([f"{label}: {board.shape[1]} samples, {dropped} dropped"
                                + (", replaced the earlier one" if replaced else "")]
                               + hum_test.table(hum_test.load(hum_test.saved())))


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
            ranges, car = calibrate.session_ranges(session, self.recorder.band.channels), True
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
        # The Control tab's simulated model: the class it is sure of (None for rest), how sure, effort, arm moving
        self.sim = {"label": None, "sure": 1.0, "effort": 0.0, "moving": False}

    def set_threshold(self, value):
        self.threshold = value
        if self._live is not None:
            self._live.threshold = value
        self.state["threshold"] = value

    def use(self, path, control=None):
        """control: a Control to step with every prediction (the Control tab), None for the prediction alone."""
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
        self._launch(self._run, model, path.name, control)
        return model

    def simulate(self, classes, control):
        """Runs control on inputs from the page instead of a model and the band (set_sim), at the networks' step."""
        self.stop()
        self.join(2)
        self._stop.clear()
        self.sim.update(label=None, effort=0.0, moving=False)
        self._launch(self._simulate, list(classes), control)

    def set_sim(self, inputs):
        if inputs.get("label") is not None and inputs["label"] not in self.state.get("classes", []):
            raise ValueError(f"the simulated model has no class {inputs['label']}")
        self.sim.update(inputs)

    def stop(self):
        self._stop.set()

    def _simulate(self, classes, control):
        rest = "rest" if "rest" in classes else classes[0]
        self.state.update(model="simulated model", classes=classes, label=None, threshold=self.threshold,
                          control=True, simulated=True)
        period = SIM_STEP_S
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            label = self.sim["label"] or rest
            sure = min(max(self.sim["sure"], 0.0), 1.0)
            others = (1 - sure) / (len(classes) - 1) if len(classes) > 1 else 0.0
            probs = {c: sure if c == label else others for c in classes}
            self.state.update(label=label if sure >= self.threshold else gesture_model.UNSURE, raw=label, probs=probs)
            control.advance(probs, self.sim["effort"], period, None, self.sim["moving"])
            next_tick = max(next_tick + period, time.perf_counter())
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def _run(self, model, name, control):
        if model.get("kind") == "network":
            from ml.ringnet import NetPredictor
            live = NetPredictor(model, self.threshold, gesture_model.UNSURE)
        else:
            live = gesture_model.LivePredictor(model)
        live.threshold = self.threshold
        self._live = live
        self.state.update(model=name, classes=model["classes"], label=None, threshold=self.threshold,
                          control=control is not None)
        period = model["step_ms"] / 1000
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            emg, filtered = self.recorder.window(live.win)
            if filtered.shape[1] == live.win:
                accel = self.recorder.accel_window(live.win) if model["options"]["accel"] else None
                label, raw, probs = live.predict(filtered, accel)
                # Only networks predict finger angles, LDA models leave these None
                self.state.update(label=label, raw=raw, probs=probs, angles=getattr(live, "angles", None),
                                  effort_angles=getattr(live, "effort_angles", None))
                if control is not None:
                    control.step(emg, filtered, probs, period, getattr(live, "distance", None))
            next_tick = max(next_tick + period, time.perf_counter())
            time.sleep(max(0.0, next_tick - time.perf_counter()))


# Every pose a rule can pick: the grips and the protocol C library
POSES = {**{k: e["angles"] for k, e in LIBRARY.items()}, **{g: GESTURES[g][1] for g in GESTURE_SET}}


def sign_map(rules, classes):
    """Grip per class for the modes that run on grip probabilities (lock, blend, trigger): a class held to pick a grip
    or to open stands for that grip, any other class not in the grip set counts as rest."""
    rules = split_rules(rules)
    # The relaxed list last, so a trigger that picks a grip there stands for it even if it also opens while holding
    picked = {r["trigger"]: r["action"] for r in rules["holding"] + rules["relaxed"] if r["how"] == "held"
              and (r["action"] in GESTURE_SET or r["action"] == "open")}
    return {c: picked.get(c, c if c in GESTURE_SET else "rest") for c in classes}


class Control:
    """control.Controller on the predictor's output, stepped inside the predictor's loop so the grip probabilities
    and the effort come from the same window. Parameters and the mapping (the rules) can change while it runs. Rules
    mode gets the model's own classes; the other modes get grip probabilities, a sign model's added up per grip
    through sign_map."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.params = dict(DEFAULTS, mode="rules")
        self.rules = split_rules(DEFAULT_RULES)
        self.state = {"params": self.params, "rules": self.rules}
        self._controller = None
        self._ranges = None
        self._still = float("inf")
        self._map = None

    def prepare(self, session):
        """Effort levels from the session the model trained on, the same levels the haptics take, and from its held
        cues how much the accelerometer moves with the arm still (trigger mode's still_gate)."""
        self._ranges = calibrate.session_ranges(session, self.recorder.band.channels)
        data = load_session(session)
        self._still = still_limit(data["accel"], [(s["sample0"], s["sample1"]) for s in segments(data["events"])
                                                  if s["kind"] == "hold" and s["sample1"] and not s["bad"]],
                                  data["meta"]["rate"])
        self._controller = Controller(self.params, GESTURE_SET, POSES)
        self._controller.set_rules(self.rules)
        self._map = None
        self.state = {"params": self.params, "rules": self.rules, "levels_from": Path(session).name, "step": 0,
                      "state": "idle", "grip": None, "closure": 0.0, "effort": 0.0, "angles": None, "starts": 0,
                      "releases": 0, "opens": 0, "locks": 0, "unlocks": 0}

    def set_rules(self, rules):
        check_rules(rules, POSES)
        self.rules = split_rules(rules)
        self.state["rules"] = self.rules
        self._map = None
        if self._controller is not None:
            self._controller.set_rules(self.rules)

    def set_params(self, params):
        self.params.update(params)
        if self._controller is not None:
            self._controller.params.update(params)

    def prepare_sim(self):
        """No session: effort and arm movement come from the page (Predictor.simulate)."""
        self._ranges = None
        self._still = float("inf")
        self._controller = Controller(self.params, GESTURE_SET, POSES)
        self._controller.set_rules(self.rules)
        self._map = None
        self.state = {"params": self.params, "rules": self.rules, "levels_from": None, "step": 0,
                      "state": "idle", "grip": None, "closure": 0.0, "effort": 0.0, "angles": None, "starts": 0,
                      "releases": 0, "opens": 0, "locks": 0, "unlocks": 0}

    def reset_counts(self):
        self.state.update(starts=0, releases=0, opens=0, locks=0, unlocks=0)

    def step(self, emg, filtered, probs, dt, distance=None):
        n = self.recorder.env_len
        band = self.recorder.band
        e = effort(emg[:, -n:], filtered[:, -n:], band.channels, self._ranges, True)
        moving = movement(self.recorder.accel_window(int(STILL_S * band.rate))) > self._still
        self.advance(probs, e, dt, distance, moving)

    def advance(self, probs, e, dt, distance=None, moving=False):
        if self.params["mode"] != "rules" and not set(probs) <= set(GESTURE_SET):
            if self._map is None:
                self._map = sign_map(self.rules, probs)
            grips = dict.fromkeys(GESTURE_SET, 0.0)
            for c, v in probs.items():
                grips[self._map[c]] += v
            probs = grips
        out = self._controller.step(probs, e, dt, distance, moving)
        s = self.state
        s.update(step=s["step"] + 1, state=out["state"], grip=out["grip"], closure=round(out["closure"], 3),
                 effort=round(e, 3), angles=[round(a, 1) for a in out["angles"]],
                 distance=None if distance is None else round(distance, 2), pulses=out.get("pulses"),
                 locked=out.get("locked", False), moving=moving)
        if out["event"]:
            s[out["event"] + "s"] += 1


class CursorTask(Tool):
    """cursor.Cursor on the live channel levels, from a recorded session on the cursor's own scale (LEVEL_SCALE). Each target run is
    logged to data/cursor/ (settings, then one line per trial with its path), so practice can be compared day to
    day. Parameters can change while it runs."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.params = dict(CURSOR_DEFAULTS)
        self.state["params"] = self.params
        self._stop = threading.Event()
        self._cursor = None
        self._log = None
        self._trials = []

    def start(self, session):
        ranges = calibrate.session_ranges(session, self.recorder.band.channels, CURSOR_LEVEL_SCALE)
        self.stop()
        self.join(2)
        self._stop.clear()
        self._launch(self._run, ranges, Path(session).name)

    def stop(self):
        self._stop.set()

    def set_params(self, params):
        self.params.update(params)

    def targets(self, on):
        if self._cursor is None or not self.state["running"]:
            raise ValueError("start the cursor first")
        self._cursor.reset_task()
        self._cursor.targets_on = on
        self._trials = []
        self._log = None
        if on:
            folder = DATA_DIR / "cursor"
            folder.mkdir(parents=True, exist_ok=True)
            self._log = folder / f"{time.strftime('%Y-%m-%d_%H%M%S')}.jsonl"
            self._log.write_text(json.dumps({"params": self.params, "levels_from": self.state["levels_from"],
                                             "channels": self.recorder.band.channels}) + "\n")

    def _run(self, ranges, levels_from):
        band = self.recorder.band
        self._cursor = Cursor(self.params)
        self._trials = []
        self.state.update(params=self.params, levels_from=levels_from)
        dt = 1 / UPDATE_HZ
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            raw, filtered = self.recorder.window(self.recorder.env_len)
            if raw.shape[1] == self.recorder.env_len:
                levels, lost = channel_levels(raw, filtered, band.channels, ranges, True)
                levels = [levels[ch] for ch in band.channels]
                trial = self._cursor.step(levels, dt)
                if trial is not None:
                    trial["at"] = time.strftime("%H:%M:%S")
                    self._trials.append({k: v for k, v in trial.items() if k != "path"})
                    if self._log is not None:
                        with self._log.open("a") as f:
                            f.write(json.dumps(trial) + "\n")
                c = self._cursor
                self.state.update(x=round(c.x, 3), y=round(c.y, 3), levels=[round(v, 3) for v in levels],
                                  lost=lost.tolist(), targets_on=c.targets_on, phase=c.phase, target=c.target,
                                  angles=c.angles(), dwell=round(c.dwell, 2), t=round(c.t, 2),
                                  trials=self._trials[-200:], log=self._log.name if self._log else None)
            next_tick = max(next_tick + dt, time.perf_counter())
            time.sleep(max(0.0, next_tick - time.perf_counter()))


class Trainer(Tool):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def start(self, folders, options, use=True):
        """use: make the new model the live one. Not for a job a device started on the hub, the hub's armband
        isn't the one that device's wearer has on."""
        self._launch(self._run, folders, options, use)

    def _run(self, folders, options, use):
        self.state.update(step=f"loading {len(folders)} sessions", log=[], share=0.0)

        def log(text, share):
            self.state.update(step=text, log=self.state["log"] + [text], share=share)

        model, result = gesture_model.train(folders, options, log=log)
        path = gesture_model.save(model)
        self.state.update(step=None, model=path.name, lines=[
            f"{model['windows']} labelled windows from {len(model['sessions'])} sessions, "
            f"classes {', '.join(model['classes'])}"] + gesture_model.report(result) + [f"saved {path.name}"])
        if use:
            self.predictor.use(path)


class NetworkJob(Tool):
    """Runs one of the network scripts (ml.train_ringnet, ml.networks) as a separate process at low priority, so
    training can't stall the armband and camera loops even mid-recording. Its output lines and the result file it
    writes are passed to the page."""

    def __init__(self):
        super().__init__()
        self._proc = None
        self._stopping = False

    def start(self, job, args):
        self._launch(self._run, job, args)

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._stopping = True
            self._proc.terminate()

    def _run(self, job, args):
        fd, out = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        lines = deque(maxlen=JOB_LINES)
        self._stopping = False
        # The page shows how long it has run and how long since the last line, so a stalled job can be told apart
        # from a slow one
        now = time.time()
        self.state.update(job=job, lines=[], result=None, started=now, last_line=now, progress={"share": None})
        try:
            self._proc = subprocess.Popen([sys.executable, "-u", "-m", *args, "--json", out], cwd=REPO,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in self._proc.stdout:
                # Library warnings would push the useful lines off the page
                if line.strip() and "Warning" not in line:
                    lines.append(line.rstrip())
                    self.state.update(lines=list(lines), last_line=time.time(),
                                      progress=job_progress(line, self.state["progress"]))
            if self._proc.wait() != 0:
                # A stopped job's last line is just where it got to, not an error
                if self._stopping:
                    raise RuntimeError("stopped")
                raise RuntimeError(lines[-1] if lines else f"failed (exit code {self._proc.returncode})")
            text = Path(out).read_text()
            self.state["result"] = json.loads(text) if text else None
        finally:
            Path(out).unlink(missing_ok=True)


def job_progress(line, progress):
    """progress updated from one output line of a network job: {"fold": (k, n), "epoch": (k, n), "left": text}.
    share is how much of the job is done, from the epochs and, when it runs folds, the fold. None until the first
    epoch, the page shows a moving bar then."""
    progress = dict(progress)
    if m := FOLD_LINE.search(line):
        progress.update(fold=(int(m[1]), int(m[2])), epoch=None, left=None)
    elif m := EPOCH_LINE.search(line):
        progress.update(epoch=(int(m[1]), int(m[2])), left=m[3])
    epoch, fold = progress.get("epoch"), progress.get("fold")
    share = epoch[0] / epoch[1] if epoch else None
    if fold:
        share = (fold[0] - 1 + (share or 0)) / fold[1]
    progress["share"] = share
    return progress


def model_path(name):
    """Where a model file of this name goes, refused unless it is a plain LDA or network file name."""
    if Path(name).name != name or Path(name).suffix not in (".joblib", ".pt"):
        raise ValueError(f"not a model name: {name}")
    return (NETWORK_DIR if name.endswith(".pt") else gesture_model.MODEL_DIR) / name


class Sync(Tool):
    """System > Sync on a device: sends this computer's ended sessions to the hub, and downloads the models picked
    there together with the session each one takes its levels from (hub.py)."""

    def send(self):
        self._launch(self._send)

    def fetch(self, names, control_files):
        """control_files: {"mappings": path, "presets": path}, swapped both ways after the models."""
        self._launch(self._fetch, names, control_files)

    def _counter(self, total):
        done = 0

        def step(n):
            nonlocal done
            done += n
            self.state["share"] = min(1.0, done / total) if total else 1.0
        return step

    def _log(self, line):
        self.state["lines"] = self.state["lines"] + [line]

    def _send(self):
        client = hub.client()
        self.state.update(step="asking the hub which sessions it has", share=0.0, lines=[])
        theirs = client.request("GET", "/api/hub/sessions")
        mine = hub.ended_sessions(DATA_DIR)
        todo = [name for name in mine if name not in theirs]
        step = self._counter(sum(sum(mine[name].values()) for name in todo))
        for i, name in enumerate(todo):
            self.state["step"] = f"sending {name} ({i + 1} of {len(todo)})"
            client.send_session(DATA_DIR / name, step)
            self._log(f"sent {name}")
        for name in mine:
            if name in theirs and theirs[name] != mine[name]:
                self._log(f"left out {name}: the hub has a different session with that name")
        if not todo:
            self._log("nothing to send, the hub has every session")
        self.state.update(step=None, share=1.0)

    @staticmethod
    def free_models():
        """Deletes the models here that the hub holds byte for byte, for a device when its app closes, so it never
        runs an old copy. Models only this computer has stay. Returns what it did, one line each."""
        lines = []
        for m in hub.client().request("GET", "/api/hub/models"):
            path = model_path(m["name"])
            if path.exists() and hub.sha256(path) == m["sha256"]:
                path.unlink()
                lines.append(f"deleted the model {m['name']} here, the hub has it")
        return lines

    @staticmethod
    def free_space(keep):
        """Deletes the sessions here that the hub and its backup both hold byte for byte, for a device when its app
        closes. keep: sessions its LDA models take their levels from; the ones its downloaded networks need are
        added from the hub's list. Returns what it did, one line each."""
        client = hub.client()
        keep = set(keep) | {m["levels"] for m in client.request("GET", "/api/hub/models")
                            if m["levels"] and model_path(m["name"]).exists()}
        mine = hub.ended_sessions(DATA_DIR)
        theirs = client.request("GET", "/api/hub/sessions")
        candidates = [name for name in mine if theirs.get(name) == mine[name] and name not in keep]
        backups = client.request("GET", "/api/hub/backups")
        if not backups["folder_set"]:
            return ["kept every session here, the hub has no backup folder set"] if candidates else []
        lines = []
        for name in candidates:
            if name not in backups["sessions"]:
                lines.append(f"kept {name} here, the hub hasn't backed it up yet")
                continue
            here = hub.hashes(DATA_DIR / name)
            if here != client.request("GET", f"/api/hub/sessions/{name}/hashes") or here != backups["sessions"][name]:
                lines.append(f"kept {name} here, the copies differ")
                continue
            try:
                hub.remove_session(DATA_DIR / name)
            except ValueError as e:
                lines.append(str(e))
                continue
            lines.append(f"deleted {name} here, the hub and its backup have it")
        return lines

    def _fetch(self, names, control_files):
        client = hub.client()
        self.state.update(step="asking the hub for its models", share=0.0, lines=[])
        listing = {m["name"]: m for m in client.request("GET", "/api/hub/models")}
        models = []
        for name in names:
            if name not in listing:
                raise ValueError(f"the hub has no model {name}")
            path = model_path(name)
            if not path.exists():
                models.append(listing[name])
            elif hub.sha256(path) != listing[name]["sha256"]:
                self._log(f"kept this computer's {name}, the hub's file with that name differs")
        sessions = sorted({m["levels"] for m in models if m["levels"] and not (DATA_DIR / m["levels"]).exists()})
        sizes = client.request("GET", "/api/hub/sessions") if sessions else {}
        step = self._counter(sum(m["size"] for m in models) + sum(sum(sizes[s].values()) for s in sessions))
        for m in models:
            self.state["step"] = f"downloading {m['name']}"
            client.download({"kind": "model", "name": m["name"]}, model_path(m["name"]), m["size"], m["sha256"], step)
            self._log(f"downloaded {m['name']}")
        for name in sessions:
            hub.check_name(name)
            self.state["step"] = f"downloading the session {name}, a model's levels come from it"
            client.download_session(name, DATA_DIR, step)
            self._log(f"downloaded the session {name}")
        self.state["step"] = "swapping mappings and profiles with the hub"
        mine = {kind: json.loads(path.read_text()) if path.exists() else {} for kind, path in control_files.items()}
        answer = client.request("POST", "/api/hub/control", body=mine)
        kept = answer["kept"] + [n for kind, path in control_files.items() for n in hub.merge_named(path, answer[kind])]
        for name in sorted(set(kept)):
            self._log(f"two different mappings or profiles called {name}, each computer kept its own")
        self._log("mappings and profiles swapped")
        self.state.update(step=None, share=1.0)


class HubCalibration(Tool):
    """A device's quick calibration, run on the hub's GPU: sends the session if the hub doesn't have it yet (through
    Sync, so it never goes up twice at once with auto send), starts the calibration there, follows the hub's job
    and downloads the calibrated network. Its state has the same fields the page reads from NetworkJob."""

    def __init__(self, sync):
        super().__init__()
        self.sync = sync

    def start(self, name):
        self._launch(self._run, name)

    def _wait_for_sync(self):
        while self.sync.state["running"]:
            time.sleep(HUB_POLL_S)

    def _run(self, name):
        now = time.time()
        self.state.update(job="calibrate", lines=[], result=None, started=now, last_line=now, progress={"share": None})
        client = hub.client()
        mine = hub.ended_sessions(DATA_DIR).get(name)
        if mine is None:
            raise ValueError(f"{name} hasn't ended on this computer")
        self._wait_for_sync()
        theirs = client.request("GET", "/api/hub/sessions").get(name)
        if theirs is None:
            self.state["lines"] = [f"Sending {name} to the hub"]
            try:
                self.sync.send()
            except ValueError:
                pass
            # Sync sets running before send() returns, so the wait can't miss it
            self._wait_for_sync()
            theirs = client.request("GET", "/api/hub/sessions").get(name)
            if theirs is None:
                raise ValueError(f"couldn't send {name} to the hub: {self.sync.state.get('error') or 'not there'}")
        if theirs != mine:
            raise ValueError(f"the hub has a different session called {name}")
        network = client.request("POST", "/api/network/quick", body={"session": name})["network"]
        self.state["network"] = network
        while True:
            time.sleep(HUB_POLL_S)
            job = client.request("GET", "/api/hub/jobs")["network"]
            self.state.update(lines=job.get("lines", []), progress=job.get("progress") or {"share": None},
                              last_line=time.time())
            if not job["running"]:
                break
        if job.get("error"):
            raise RuntimeError(f"on the hub: {job['error']}")
        result = job.get("result") or {}
        saved = result.get("saved")
        if not saved:
            raise RuntimeError("calibration on the hub finished without saving a network")
        listing = {m["name"]: m for m in client.request("GET", "/api/hub/models")}
        if saved not in listing:
            raise RuntimeError(f"the hub doesn't list {saved}")
        m = listing[saved]
        self.state["lines"] = self.state["lines"] + [f"downloading {saved}"]
        client.download({"kind": "model", "name": saved}, model_path(saved), m["size"], m["sha256"], lambda n: None)
        self.state["result"] = result


class Backup:
    """Copies every ended session the backup folder doesn't have yet (hub.back_up), in the background whenever a
    session is saved here or received from a device. Asked again while copying, it goes round once more."""

    def __init__(self):
        self.state = {"running": False, "error": None, "step": None, "last": None}
        self._wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def request(self):
        self._wake.set()

    def _loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            self.state.update(running=True, error=None)
            try:
                self._run()
            except Exception as e:
                self.state["error"] = str(e) or type(e).__name__
            finally:
                self.state.update(running=False, step=None)

    def _run(self):
        dest = hub.backup_folder()
        if dest is None:
            return
        done = hub.backed_up(dest)
        todo = [name for name in hub.ended_sessions(DATA_DIR) if name not in done]
        failed = []
        for i, name in enumerate(todo):
            self.state["step"] = f"backing up {name} ({i + 1} of {len(todo)})"
            try:
                hub.back_up(DATA_DIR / name, dest)
            except (OSError, ValueError) as e:
                failed.append(str(e))
        self.state["last"] = time.time()
        if failed:
            raise ValueError("; ".join(failed))


class SyncWatch:
    """The header's sync line. On a device: how many ended sessions the hub doesn't have yet, and with auto send on
    (hub.json "auto_send") it sends them, never while a session or practice run is going. On the hub: how many
    sessions its backup doesn't have yet. Checked every SYNC_CHECK_S and whenever request() is called."""

    def __init__(self, recorder, sync, backup):
        self.recorder = recorder
        self.sync = sync
        self.backup = backup
        self.state = {"role": None}
        self._wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def request(self):
        self._wake.set()

    def _loop(self):
        while True:
            try:
                self.state = self._check()
            except Exception as e:
                self.state = dict(self.state, ok=False, text="sync check failed", detail=str(e) or type(e).__name__)
            busy = self.sync.state["running"] or self.backup.state["running"]
            self._wake.wait(SYNC_BUSY_CHECK_S if busy else SYNC_CHECK_S)
            self._wake.clear()

    def _check(self):
        now = time.time()
        if self.recorder.hub:
            out = {"role": "hub", "checked": now}
            try:
                dest = hub.backup_folder()
            except ValueError as e:
                return dict(out, ok=False, text="backup folder missing", detail=str(e))
            if dest is None:
                return dict(out, ok=False, text="no backup folder", detail="set one on System > Sync")
            done = hub.backed_up(dest)
            missing = len([name for name in hub.ended_sessions(DATA_DIR) if name not in done])
            if self.backup.state["running"]:
                return dict(out, ok=False, text="backing up", detail=self.backup.state.get("step"))
            if self.backup.state.get("error"):
                return dict(out, ok=False, text="backup failed", detail=self.backup.state["error"])
            return dict(out, ok=not missing, text=f"{missing} to back up" if missing else "backed up",
                        detail=f"{len(done)} sessions in {dest}")
        config = hub.load()
        if not config.get("url"):
            return {"role": None}
        out = {"role": "device", "checked": now, "auto": bool(config.get("auto_send"))}
        if self.sync.state["running"]:
            return dict(out, ok=False, text="sending", detail=self.sync.state.get("step"))
        mine = hub.ended_sessions(DATA_DIR)
        try:
            theirs = hub.client().request("GET", "/api/hub/sessions", timeout=SYNC_TIMEOUT_S)
        except hub.HubError as e:
            return dict(out, ok=False, text="hub out of reach" if e.status == 502 else "hub refused", detail=str(e))
        todo = [name for name in mine if name not in theirs]
        if todo and out["auto"] and self.recorder.session is None:
            self.sync.send()
            return dict(out, ok=False, text="sending", detail=f"{len(todo)} new sessions")
        if todo and self.sync.state.get("error"):
            return dict(out, ok=False, text="send failed", detail=self.sync.state["error"])
        detail = f"{len(mine)} sessions here" + (", sends by itself" if out["auto"] else ", send them on System > Sync")
        return dict(out, ok=not todo, text=f"{len(todo)} to send" if todo else "synced", detail=detail)


class Tools:
    def __init__(self, recorder):
        esp_lock = threading.Lock()
        self.check = ConnectionCheck(recorder)
        self.wifi = ArmbandWifi(recorder)
        self.calibration = Calibration(recorder)
        self.hum = HumTest(recorder)
        self.haptics = Haptics(recorder, esp_lock)
        self.test = OutputTest(esp_lock)
        self.predictor = Predictor(recorder)
        self.control = Control(recorder)
        self.cursor = CursorTask(recorder)
        self.trainer = Trainer(self.predictor)
        self.network = NetworkJob()
        self.sync = Sync()
        self.hub_calibration = HubCalibration(self.sync)
        self.backup = Backup()
        self.sync_watch = SyncWatch(recorder, self.sync, self.backup)
        self.recorder = recorder
        # The delay test's two halves, (seconds, result line) each, until saved to the setup
        self.measured = {}
        self.round_trip = RoundTrip(recorder, self.haptics, self.measured)
        self.flash = ScreenFlash(recorder, self.measured)

    def save_delay(self):
        """Writes the measured label shift into the current setup's entry in devices.json."""
        setup = self.recorder.setup
        if setup is None:
            raise ValueError("pick the device first")
        if "camera" not in self.measured or "round_trip" not in self.measured:
            raise ValueError("measure both the round trip and the camera first")
        cam, trip = self.measured["camera"][0], self.measured["round_trip"][0]
        note = f"screen flash {cam * 1000:.1f} ms (camera + screen), armband round trip {trip * 1000:.1f} ms"
        self.recorder.setup = devices.save_shift(setup["id"], delay.label_shift(cam, trip), note)
        return self.recorder.setup

    def snapshot(self):
        now = time.time()
        out = {}
        for name in ("check", "wifi", "calibration", "haptics", "test", "predictor", "trainer", "network", "control",
                     "cursor", "sync", "backup", "sync_watch", "round_trip", "flash", "hum",
                     "hub_calibration"):
            state = dict(getattr(self, name).state)
            if state.get("until"):
                state["left"] = max(0.0, state["until"] - now)
            # Worked out here, a tablet's clock can differ from this computer's
            if state.get("started"):
                state["running_s"] = now - state["started"]
                state["quiet_s"] = now - state["last_line"]
            out[name] = state
        return out
