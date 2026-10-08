"""Runs the armband and camera, the live views and the cue sequence of a recording session.

One thread reads the armband every TICK_S, filters the EMG for the live views, runs the quality checks and
advances the cues. The camera has its own thread and hands each frame to the session as it arrives.
"""
import json
import threading
import time
import traceback

import numpy as np

from calibrate import CAL_FILE
from gestures import plan_for
from processing import ENVELOPE_MS, FILTER_VERSION, StreamFilter, envelope
from quality import HUM_FAST_OF, HUM_WARN_UV, SignalQuality
from session import PracticeWriter, SessionWriter

TICK_S = 0.02
# Hand counts as out of view after this long without a detection
HAND_LOST_S = 0.5
# The webcam runs at 30 fps, a drop to 15 means frames are being lost somewhere
CAMERA_SLOW_FPS = 25
# Mark bad within this long of a rest cue starting still means the gesture before it
BAD_GRACE_S = 1.5
# Cues whose data is used for training. Hum on any channel during one marks it bad and pauses for a redo
HUM_REDO_KINDS = ("hold", "finger", "free", "move")
# EMG history kept for the signal view and the haptics
HISTORY_S = 4
# How long a failed armband read stays in the warning bar
ERROR_SHOWN_S = 10
# Fraction of the picture at each edge where the wrist is too close to the border to track reliably
EDGE_MARGIN = 0.12


def _folder_text(session):
    return str(session.folder) if session.folder is not None else "practice run, nothing saved"


class Recorder:
    def __init__(self, band, camera=None):
        self.band = band
        self.camera = camera
        # Called with the folder of every completed session once it is fully written (camera delay included)
        self.on_saved = None
        self.quality = SignalQuality(band.channels, band.rate)
        self.stream = StreamFilter(len(band.channels), band.rate)
        self.env_len = int(ENVELOPE_MS / 1000 * band.rate)
        self.history = HISTORY_S * band.rate
        self.raw = np.zeros((len(band.channels), 0))
        self.filtered = np.zeros((len(band.channels), 0))
        self.accel_history = np.zeros((3, 0))
        self.received = 0
        self._captures = []
        self.env = [0.0] * len(band.channels)
        self.accel = None
        self.hand_seen = 0.0
        self.wrist = None
        self.session = None
        self.plan = []
        self.index = -1
        self.cue_start = 0.0
        self.advanced = False
        self.paused = False
        # Set when hum paused the session: which channels, and the first cue to redo
        self.hum_pause = None
        self.last_summary = None
        self.loop_error = None
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
        while not self._stop.is_set():
            # An error used to end this thread for good, which looks exactly like the armband disconnecting.
            # Keep going instead, a WiFi drop can recover, and show what went wrong
            try:
                self._tick()
            except Exception as e:
                self.loop_error = (time.time(), f"{type(e).__name__}: {e}")
                traceback.print_exc()
            time.sleep(TICK_S)

    def _tick(self):
        rows = self.band.rows
        data = self.band.read_all()
        with self._lock:
            if data.shape[1]:
                if self.session is not None:
                    self.session.add_board(data)
                emg = data[self.band.emg_rows]
                battery = data[rows["battery"], -1] if self.band.source == "armband" else None
                self.quality.update(emg, data[rows["package"]], battery)
                self._check_hum()
                self.raw = np.hstack([self.raw, emg])[:, -self.history:]
                self.filtered = np.hstack([self.filtered, self.stream.process(emg)])[:, -self.history:]
                self.accel_history = np.hstack([self.accel_history, data[rows["accel"]]])[:, -self.history:]
                self.received += emg.shape[1]
                self.env = [envelope(y) for y in self.filtered[:, -self.env_len:]]
                self.accel = data[rows["accel"], -1].tolist()
            for capture in self._captures:
                capture["chunks"].append(data)
                if time.time() >= capture["until"]:
                    capture["done"].set()
            self._captures = [c for c in self._captures if not c["done"].is_set()]
            self._advance()

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

    def accel_window(self, samples):
        """Accelerometer rows for the same newest samples as window()."""
        with self._lock:
            return self.accel_history[:, -samples:].copy()

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
            # Wrist position in the image, 0 to 1
            self.wrist = (record[4], record[5])
        session = self.session
        if session is not None:
            session.add_frame(record)

    # Session control, all called with the lock held or from the web server through command()

    def start_session(self, settings):
        with self._lock:
            if self.session is not None:
                raise ValueError("a session is already running")
            plan = plan_for(settings)
            if self.camera is not None:
                self.camera.hand = settings["tracked_hand"]
            meta = {"settings": settings, "plan": plan, "rate": self.band.rate, "source": self.band.source,
                    "rows": self.band.rows, "channels": self.band.channels, "filter_version": FILTER_VERSION,
                    "calibration": json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else None,
                    "camera": self._camera_meta()}
            writer = PracticeWriter if settings.get("practice") else SessionWriter
            self.session = writer(meta, settings["subject"])
            self.plan = plan
            self.paused = False
            self.hum_pause = None
            self.last_summary = None
            # The quality check counts drops since the app started, the session only wants its own
            self.dropped_at_start = self.quality.dropped_total
            self._go_to(0)
            return _folder_text(self.session)

    def command(self, action):
        with self._lock:
            if self.session is None:
                raise ValueError("no session running")
            cue = self.plan[self.index]
            if action == "continue" and (cue["kind"] == "break" or self.paused):
                if self.paused:
                    self.paused = False
                    self.session.add_event("resume")
                    redo = self.hum_pause["redo"] if self.hum_pause else self.index
                    self.hum_pause = None
                    # Old estimates would stop it again straight away, a pad that is still bad shows up again
                    # within HUM_FAST_OF seconds
                    self.quality.reset_hum()
                    self._go_to(redo)
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

    def _check_hum(self):
        if self.session is None or self.paused or self.plan[self.index]["kind"] not in HUM_REDO_KINDS:
            return
        hum = self.quality.hum_recent()
        if hum is None or not (hum > HUM_WARN_UV).any():
            return
        channels = [ch for ch, h in zip(self.quality.channels, hum) if h > HUM_WARN_UV]
        bad = [self.index]
        # The check looks back HUM_FAST_OF seconds, so hum found early in a cue may have started in the one before
        if (self.advanced and time.time() - self.cue_start < HUM_FAST_OF
                and self.plan[self.index - 1]["kind"] in HUM_REDO_KINDS):
            bad.insert(0, self.index - 1)
        self.paused = True
        self.session.add_event("pause", reason="hum")
        for index in bad:
            self.session.add_event("bad", index=index, reason="hum", channels=channels)
        self.hum_pause = {"channels": channels, "uv": [round(float(h)) for h in hum[hum > HUM_WARN_UV]],
                          "redo": bad[0]}

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
        summary = {"completed": completed, "dropped_samples": self.quality.dropped_total - self.dropped_at_start}
        session.close(**summary)
        seconds = session.meta["ended"] - session.meta["started"]
        detected = session.detected / session.frames if session.frames else None
        practice = session.folder is None
        self.last_summary = dict(summary, folder=_folder_text(session), practice=practice, seconds=seconds,
                                 samples=session.samples, frames=session.frames, hand_detected=detected)
        self.index = -1
        if completed and not practice:
            self._saved(session.folder)

    def _saved(self, folder):
        if self.on_saved is None:
            return
        try:
            self.on_saved(folder)
        except Exception as e:
            print(f"after saving {folder.name}: {e}")

    def _camera_meta(self):
        if self.camera is None:
            return None
        return {"source": str(self.camera.source), "size": self.camera.size, "fps": round(self.camera.fps, 1)}

    # What the browser gets

    def warnings(self):
        out = self.quality.warnings()
        if self.loop_error is not None and time.time() - self.loop_error[0] < ERROR_SHOWN_S:
            out.append(("loop_error", f"Reading the armband failed: {self.loop_error[1]}"))
        if self.camera is None:
            out.append(("camera", "No camera, joint angles aren't recorded"))
        elif not self.camera.ok:
            out.append(("camera", self.camera.error or "Camera starting"))
        else:
            edge = self._wrist_edge()
            hand = f"{self.camera.hand.capitalize()} hand"
            if time.time() - self.hand_seen > HAND_LOST_S:
                out.append(("hand", f"{hand} lost at the {edge} edge of the picture, move the camera so the "
                                    "whole hand is well inside" if edge else f"{hand} out of view"))
            elif edge:
                out.append(("hand_edge", f"{hand} near the {edge} edge of the picture, tracking may drop"))
            if 0 < self.camera.fps < CAMERA_SLOW_FPS:
                out.append(("camera_slow", f"Camera at {self.camera.fps:.0f} fps"))
        return [{"code": c, "text": t} for c, t in out]

    def _wrist_edge(self):
        """Which edge of the picture the wrist was last seen near, as the mirrored preview shows it, or None.
        The tracker finds the palm first, so a hand with its wrist on the edge gets lost even with the fingers
        still in view (2026-10-01, lost every time the wrist passed about 95 % of the height)."""
        if self.wrist is None:
            return None
        x, y = self.wrist
        if y > 1 - EDGE_MARGIN:
            return "bottom"
        if y < EDGE_MARGIN:
            return "top"
        # The preview is mirrored, so the left of the camera image shows on the right
        if x < EDGE_MARGIN:
            return "right"
        if x > 1 - EDGE_MARGIN:
            return "left"
        return None

    def snapshot(self):
        with self._lock:
            state = {"env": [round(e, 1) for e in self.env], "accel": self.accel,
                     "hum": [round(float(h), 1) for h in self.quality.hum_uv()],
                     "lost": self.quality.lost.tolist(), "warnings": self.warnings(),
                     "battery": self.quality.battery_status(),
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
                    "folder": _folder_text(self.session), "practice": self.session.folder is None,
                    "index": self.index, "count": len(self.plan), "cue": cue,
                    "elapsed": time.time() - self.session.meta["started"], "paused": self.paused,
                    "hum_pause": self.hum_pause,
                    "cue_elapsed": time.time() - self.cue_start, "next": upcoming,
                    "samples": self.session.samples, "frames": self.session.frames}
            return state
