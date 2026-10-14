"""Open the armband (or the synthetic board) and read its live EMG channels.
Replay plays back recorded raw data in real time instead."""
import threading
import time
from pathlib import Path

import numpy as np
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
from mindrove.exit_codes import MindRoveError

from channels import live_channels


def board_rows(board_id):
    """Which rows of get_board_data() hold what."""
    d = BoardShim.get_board_descr(board_id)
    return {"emg": d["emg_channels"][:8], "battery": d.get("battery_channel"), "accel": d["accel_channels"],
            "gyro": d["gyro_channels"], "package": d["package_num_channel"], "timestamp": d["timestamp_channel"],
            "num_rows": d["num_rows"]}


class Armband:
    def __init__(self, synthetic=False):
        board_id = BoardIds.SYNTHETIC_BOARD if synthetic else BoardIds.MINDROVE_WIFI_BOARD
        self.source = "synthetic" if synthetic else "armband"
        self.rate = BoardShim.get_sampling_rate(board_id)
        self.rows = board_rows(board_id)
        emg_rows = self.rows["emg"]
        self.channels = live_channels(len(emg_rows))
        self.emg_rows = [emg_rows[ch] for ch in self.channels]
        BoardShim.disable_board_logger()
        self._board = BoardShim(board_id, MindRoveInputParams())
        self._lock = threading.Lock()
        self.connected = False

    def read(self):
        """Samples that arrived since the last call, one row per live channel."""
        return self.read_all()[self.emg_rows]

    def read_all(self):
        """Samples that arrived since the last call, all board rows."""
        with self._lock:
            if not self.connected:
                return np.zeros((self.rows["num_rows"], 0))
            return self._board.get_board_data()

    def restart(self):
        """Close and reopen the stream. After the WiFi was reset the band needs a new start request."""
        with self._lock:
            self._close()
            self._open()

    def _open(self):
        self._board.prepare_session()
        self._board.start_stream()
        self.connected = True

    def _close(self):
        if self.connected:
            self.connected = False
            self._board.stop_stream()
            self._board.release_session()

    def __enter__(self):
        try:
            self._open()
        # The band's WiFi isn't joined yet after it was switched on. Start without it, so the app's Connection tab
        # can join it and open the stream
        except MindRoveError as e:
            print(f"armband not reachable, starting without it: {e}")
        return self

    def __exit__(self, *exc):
        with self._lock:
            self._close()


class Replay(Armband):
    """Plays raw recordings (.npy of all board rows, like data/2026-09-30-filter) in real time, in a loop.
    Package numbers and timestamps are rewritten so the stream looks live."""

    def __init__(self, path):
        path = Path(path)
        files = sorted(path.glob("*.npy")) if path.is_dir() else [path]
        if not files:
            raise SystemExit(f"no .npy recordings in {path}")
        self.source = f"replay {path.name}"
        self.rate = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD)
        self.rows = board_rows(BoardIds.MINDROVE_WIFI_BOARD)
        self.channels = live_channels(len(self.rows["emg"]))
        self.emg_rows = [self.rows["emg"][ch] for ch in self.channels]
        self._data = np.hstack([np.load(f) for f in files])
        self._pos = 0
        self._sent = 0

    def read_all(self):
        now = time.time()
        n = int((now - self._start) * self.rate) - self._sent
        idx = (self._pos + np.arange(max(n, 0))) % self._data.shape[1]
        out = self._data[:, idx].copy()
        out[self.rows["package"]] = self._sent + np.arange(len(idx))
        out[self.rows["timestamp"]] = now - (len(idx) - 1 - np.arange(len(idx))) / self.rate
        self._pos = (self._pos + len(idx)) % self._data.shape[1]
        self._sent += len(idx)
        return out

    def read(self):
        return self.read_all()[self.emg_rows]

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        pass
