"""Runs the armband and camera, the live views and the cue sequence of a recording session.

One thread reads the armband every TICK_S, filters the EMG for the live views, runs the quality checks and
advances the cues. The camera has its own thread and hands each frame to the session as it arrives.
"""
import json
import threading
import time

import numpy as np

from calibrate import CAL_FILE
from gestures import session_plan
from processing import ENVELOPE_MS, FILTER_VERSION, StreamFilter, envelope
from quality import SignalQuality
from session import SessionWriter

TICK_S = 0.02
# Hand counts as out of view after this long without a detection
HAND_LOST_S = 0.5
CAMERA_SLOW_FPS = 15
# Mark bad within this long of a rest cue starting still means the gesture before it
BAD_GRACE_S = 1.5
# EMG history kept for the signal view and the haptics
HISTORY_S = 4


class Recorder:
    def __init__(self, band, camera=None):
        self.band = band
        self.camera = camera
        self.quality = SignalQuality(band.channels, band.rate)
        self.stream = StreamFilter(len(band.channels), band.rate)
        self.env_len = int(ENVELOPE_MS / 1000 * band.rate)
        self.history = HISTORY_S * band.rate
        self.raw = np.zeros((len(band.channels), 0))
        self.filtered = np.zeros((len(band.channels), 0))
        self.received = 0
        self._captures = []
        self.env = [0.0] * len(band.channels)
        self.accel = None
        self.hand_seen = 0.0
        self.session = None
        self.plan = []
        self.index = -1
        self.cue_start = 0.0
        self.advanced = False
        self.paused = False
        self.last_summary = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if camera is not None:
            camera.on_frame = self._on_frame

    def start(self):
        self._thread.start()

    def stop(self):
        with self._lock:
            if self.session is not None:
                self._finish(completed=False)
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        rows = self.band.rows
        while not self._stop.is_set():
            data = self.band.read_all()
            with self._lock:
                if data.shape[1]:
                    if self.session is not None:
                        self.session.add_board(data)
                    emg = data[self.band.emg_rows]
                    battery = data[rows["battery"], -1] if self.band.source == "armband" else None
                    self.quality.update(emg, data[rows["package"]], battery)
                    self.raw = np.hstack([self.raw, emg])[:, -self.history:]
                    self.filtered = np.hstack([self.filtered, self.stream.process(emg)])[:, -self.history:]
                    self.received += emg.shape[1]
                    self.env = [envelope(y) for y in self.filtered[:, -self.env_len:]]
                    self.accel = data[rows["accel"], -1].tolist()
                for capture in self._captures:
                    capture["chunks"].append(data)
                    if time.time() >= capture["until"]:
                        capture["done"].set()
                self._captures = [c for c in self._captures if not c["done"].is_set()]
                self._advance()
            time.sleep(TICK_S)

    def capture(self, seconds):
        """Blocks for seconds and returns every board row that arrived meanwhile, like calibrate.py's record."""
        capture = {"chunks": [], "until": time.time() + seconds, "done": threading.Event()}
        with self._lock:
            self._captures.append(capture)
        capture["done"].wait(seconds + 2)
        with self._lock:
            if capture in self._captures:
                self._captures.remove(capture)
        chunks = capture["chunks"] or [np.zeros((self.band.rows["num_rows"], 0))]
        return np.hstack(chunks)

    def window(self, samples):
        """Newest raw and filtered EMG, live channels x up to samples."""
        with self._lock:
            return self.raw[:, -samples:].copy(), self.filtered[:, -samples:].copy()

    def samples_since(self, count):
        """Raw and filtered EMG that arrived after the first count samples, and the new count."""
        with self._lock:
            n = min(self.received - count, self.raw.shape[1])
            if n <= 0:
                return self.received, None, None
            return self.received, self.raw[:, -n:].copy(), self.filtered[:, -n:].copy()

    def _on_frame(self, record):
        if record[2]:
            self.hand_seen = record[0]
        session = self.session
        if session is not None:
            session.add_frame(record)

    # Session control, all called with the lock held or from the web server through command()

    def start_session(self, settings):
        with self._lock:
            if self.session is not None:
                raise ValueError("a session is already running")
            plan = session_plan(settings["postures"], settings["reps"], settings["hold_s"], settings["rest_s"],
                                settings["free_s"])
            if self.camera is not None:
                self.camera.hand = settings["tracked_hand"]
            meta = {"settings": settings, "plan": plan, "rate": self.band.rate, "source": self.band.source,
                    "rows": self.band.rows, "channels": self.band.channels, "filter_version": FILTER_VERSION,
                    "calibration": json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else None,
                    "camera": self._camera_meta(), "camera_delay_s": None}
            self.session = SessionWriter(meta, settings["subject"])
            self.plan = plan
            self.paused = False
            self.last_summary = None
            self._go_to(0)
            return str(self.session.folder)

    def command(self, action):
        with self._lock:
            if self.session is None:
                raise ValueError("no session running")
            cue = self.plan[self.index]
            if action == "continue" and (cue["kind"] == "break" or self.paused):
                if self.paused:
                    self.paused = False
                    self.session.add_event("resume")
                    self._go_to(self.index)
                else:
                    self._go_to(self.index + 1)
            elif action == "pause" and not self.paused and cue["kind"] != "break":
                self.paused = True
                self.session.add_event("pause")
            elif action == "bad" and cue["kind"] != "break":
                target = self.index
                if (cue["label"] == "rest" and self.advanced and not self.paused
                        and time.time() - self.cue_start < BAD_GRACE_S):
                    target = self.index - 1
                self.session.add_event("bad", index=target)
            elif action == "stop":
                self._finish(completed=False)
            else:
                raise ValueError(f"can't {action} now")

    def _go_to(self, index):
        if index >= len(self.plan):
            self._finish(completed=True)
            return
        # False when a cue is redone after a pause, then the cue before it is long past
        self.advanced = index > self.index >= 0
        self.index = index
        self.cue_start = time.time()
        self.session.add_event("cue", index=index, cue=self.plan[index])

    def _advance(self):
        if self.session is None or self.paused:
            return
        seconds = self.plan[self.index]["seconds"]
        if seconds is not None and time.time() - self.cue_start >= seconds:
            self._go_to(self.index + 1)

    def _finish(self, completed):
        session = self.session
        self.session = None
        session.add_event("stop", completed=completed)
        summary = {"completed": completed, "dropped_samples": self.quality.dropped_total}
        session.close(**summary)
        seconds = session.meta["ended"] - session.meta["started"]
        detected = self._detected_share(session)
        self.last_summary = dict(summary, folder=str(session.folder), seconds=seconds, samples=session.samples,
                                 frames=session.frames, hand_detected=detected)
        self.index = -1

    @staticmethod
    def _detected_share(session):
        path = session.folder / "camera.f64"
        width = len(session.meta["camera_columns"])
        cam = np.fromfile(path, dtype=np.float64)
        cam = cam[:len(cam) // width * width].reshape(-1, width)
        return float(cam[:, 2].mean()) if len(cam) else None

    def _camera_meta(self):
        if self.camera is None:
            return None
        return {"source": str(self.camera.source), "size": self.camera.size, "fps": round(self.camera.fps, 1)}

    # What the browser gets

    def warnings(self):
        out = self.quality.warnings()
        if self.camera is None:
            out.append(("camera", "No camera, joint angles aren't recorded"))
        elif not self.camera.ok:
            out.append(("camera", self.camera.error or "Camera starting"))
        else:
            if time.time() - self.hand_seen > HAND_LOST_S:
                out.append(("hand", f"{self.camera.hand.capitalize()} hand out of view"))
            if 0 < self.camera.fps < CAMERA_SLOW_FPS:
                out.append(("camera_slow", f"Camera at {self.camera.fps:.0f} fps"))
        return [{"code": c, "text": t} for c, t in out]

    def snapshot(self):
        with self._lock:
            state = {"env": [round(e, 1) for e in self.env], "accel": self.accel,
                     "hum": [round(float(h), 1) for h in self.quality.hum_uv()],
                     "lost": self.quality.lost.tolist(), "warnings": self.warnings(),
                     "hand": None, "camera_fps": round(self.camera.fps, 1) if self.camera else None,
                     "session": None, "summary": self.last_summary}
            track = self.camera.latest if self.camera is not None else None
            if track is not None and track[2] and time.time() - track[0] < HAND_LOST_S:
                state["hand"] = {"angles": np.round(track[130:], 1).tolist(),
                                 "image": np.round(track[4:67].reshape(21, 3)[:, :2], 4).tolist()}
            if self.session is not None:
                cue = self.plan[self.index]
                upcoming = next((c for c in self.plan[self.index + 1:] if c["label"] != "rest"), None)
                if upcoming is not None and upcoming["kind"] == "break":
                    upcoming = None
                state["session"] = {
                    "folder": str(self.session.folder), "index": self.index, "count": len(self.plan), "cue": cue,
                    "elapsed": time.time() - self.session.meta["started"], "paused": self.paused,
                    "cue_elapsed": time.time() - self.cue_start, "next": upcoming,
                    "samples": self.session.samples, "frames": self.session.frames}
            return state
