"""Session folders: everything recorded in one sitting, written as it arrives so a crash loses nothing.

data/<date>_<time>_<subject>/
    session.json    metadata, settings, cue plan, row layout; summary added when the session ends
    board.f64       every armband row (EMG, accelerometer, package number, timestamp...), float64,
                    one record of num_rows values per sample
    camera.f64      one record of CAMERA_COLUMNS per camera frame, float64
    events.jsonl    cue changes and button presses, one JSON object per line

All times are time.time() on the recording PC. The armband's timestamp row is that same clock at the moment
the samples arrived, so EMG and camera share one clock. Use load_session() to read a folder back.
"""
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from hand_angles import JOINTS

DATA_DIR = Path(__file__).with_name("data")

CAMERA_COLUMNS = (["t", "frame", "detected", "score"]
                  + [f"img_{i}_{a}" for i in range(21) for a in "xyz"]
                  + [f"world_{i}_{a}" for i in range(21) for a in "xyz"]
                  + JOINTS)


def app_version():
    """Git commit of the code that made the recording, with + if there were uncommitted changes."""
    try:
        here = Path(__file__).parent
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                               text=True).stdout.strip()
        return commit + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class SessionWriter:
    def __init__(self, meta, subject):
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        self.folder = DATA_DIR / f"{stamp}_{subject}"
        self.folder.mkdir(parents=True)
        self.meta = dict(meta, subject=subject, started=time.time(), app_version=app_version(),
                         camera_columns=CAMERA_COLUMNS)
        self._lock = threading.Lock()
        self._board = open(self.folder / "board.f64", "ab")
        self._camera = open(self.folder / "camera.f64", "ab")
        self._events = open(self.folder / "events.jsonl", "a")
        self.samples = 0
        self.frames = 0
        self._write_meta()

    def _write_meta(self):
        (self.folder / "session.json").write_text(json.dumps(self.meta, indent=1) + "\n")

    def add_board(self, data):
        """data: all board rows x new samples."""
        with self._lock:
            self._board.write(np.ascontiguousarray(data.T, dtype=np.float64).tobytes())
            self.samples += data.shape[1]

    def add_frame(self, record):
        with self._lock:
            # The camera thread can deliver one last frame after the session was closed
            if self._camera.closed:
                return
            self._camera.write(np.asarray(record, dtype=np.float64).tobytes())
            self.frames += 1

    def add_event(self, kind, **fields):
        """Logs an event with the time and the number of EMG samples recorded so far."""
        with self._lock:
            event = {"t": time.time(), "sample": self.samples, "kind": kind, **fields}
            self._events.write(json.dumps(event) + "\n")
            self._events.flush()
        return event

    def close(self, **summary):
        with self._lock:
            for f in (self._board, self._camera, self._events):
                f.close()
            self.meta.update(summary, ended=time.time(), samples=self.samples, frames=self.frames)
            self._write_meta()


def _read_records(path, width):
    raw = np.fromfile(path, dtype=np.float64) if path.exists() else np.zeros(0)
    # A crash can leave half a record at the end
    return raw[:len(raw) // width * width].reshape(-1, width)


def load_session(folder):
    """Returns a dict with
    meta: session.json
    board: all rows x samples, raw, as get_board_data() returns them
    emg: 8 x samples raw EMG, emg_t: time of each sample (see sample_times)
    accel: 3 x samples
    camera: {column name: array per frame}
    events: list of dicts"""
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    rows = meta["rows"]
    board = _read_records(folder / "board.f64", rows["num_rows"]).T
    cam = _read_records(folder / "camera.f64", len(meta["camera_columns"]))
    events_file = folder / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    return {"meta": meta, "board": board, "emg": board[rows["emg"]], "accel": board[rows["accel"]],
            "emg_t": sample_times(board[rows["package"]], board[rows["timestamp"]]),
            "camera": {c: cam[:, i] for i, c in enumerate(meta["camera_columns"])}, "events": events}


def sample_times(package, arrival):
    """Time each EMG sample was taken. The timestamp row is when the samples reached the PC, which jitters
    by about 10 ms with the WiFi packets, so fit a straight line against the package number. The fit also
    absorbs the armband's own clock rate (499.2 to 499.5 Hz on 2026-09-30, not exactly 500)."""
    if len(package) < 2:
        return arrival.copy()
    a, b = np.polyfit(package, arrival, 1)
    return a * package + b


def segments(events, end_t=None):
    """Cue segments from the event list: dicts with the cue fields plus t0, t1, sample0, sample1 and bad.
    A segment ends at the next cue, pause or stop."""
    out = []
    for e in events:
        if out and out[-1]["t1"] is None and e["kind"] in ("cue", "pause", "stop"):
            out[-1].update(t1=e["t"], sample1=e["sample"])
        if e["kind"] == "cue":
            out.append(dict(e["cue"], index=e["index"], t0=e["t"], sample0=e["sample"], t1=None, sample1=None,
                            bad=False))
        elif e["kind"] == "bad":
            for s in out:
                if s["index"] == e["index"]:
                    s["bad"] = True
    if out and out[-1]["t1"] is None:
        out[-1]["t1"] = end_t
    return out
