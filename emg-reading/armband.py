"""Open the armband (or the synthetic board) and read its live EMG channels.
Replay plays back recorded raw data in real time instead."""
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
from mindrove.exit_codes import MindRoveError

from channels import live_channels

# How often the reader process passes new samples on
READ_S = 0.01
# Opening normally answers at once; a reader that hangs is killed so the app can't hang with it
OPEN_TIMEOUT_S = 15


def board_rows(board_id):
    """Which rows of get_board_data() hold what."""
    d = BoardShim.get_board_descr(board_id)
    return {"emg": d["emg_channels"][:8], "battery": d.get("battery_channel"), "accel": d["accel_channels"],
            "gyro": d["gyro_channels"], "package": d["package_num_channel"], "timestamp": d["timestamp_channel"],
            "num_rows": d["num_rows"]}


class Armband:
    """The armband (or the synthetic board), read by a separate reader process (`python armband.py --reader`).

    The MindRove library has crashed the whole Python process (fast-fail in MindRoveWrapper.dll when the network
    changed under an open stream, heap corruption when the stream of a switched-off band was stopped, 2026-10-14).
    In its own process a crash only ends the reader, and closing kills the reader instead of stopping the stream.
    optional: start without the band when it can't be reached (the app, which can connect it later)."""

    def __init__(self, synthetic=False, optional=False):
        self._board_id = BoardIds.SYNTHETIC_BOARD if synthetic else BoardIds.MINDROVE_WIFI_BOARD
        self.source = "synthetic" if synthetic else "armband"
        self.rate = BoardShim.get_sampling_rate(self._board_id)
        self.rows = board_rows(self._board_id)
        emg_rows = self.rows["emg"]
        self.channels = live_channels(len(emg_rows))
        self.emg_rows = [emg_rows[ch] for ch in self.channels]
        self._optional = optional
        self._proc = None
        self._chunks = []
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._proc is not None and self._proc.poll() is None

    def read(self):
        """Samples that arrived since the last call, one row per live channel."""
        return self.read_all()[self.emg_rows]

    def read_all(self):
        """Samples that arrived since the last call, all board rows."""
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return np.hstack(chunks) if chunks else np.zeros((self.rows["num_rows"], 0))

    def open(self):
        """Start a reader. After the WiFi was reset the band needs a new start request, which a new reader sends."""
        self.close()
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--reader",
                                 "synthetic" if self.source == "synthetic" else "armband"],
                                stdout=subprocess.PIPE, cwd=Path(__file__).parent)
        timer = threading.Timer(OPEN_TIMEOUT_S, proc.kill)
        timer.start()
        status = proc.stdout.readline().decode().strip()
        timer.cancel()
        if status != "ok":
            proc.kill()
            proc.wait()
            raise ConnectionError(f"armband: {status.removeprefix('error ') or 'reader ended before opening'}")
        with self._lock:
            self._proc = proc
        threading.Thread(target=self._receive, args=(proc,), daemon=True).start()

    def close(self):
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            proc.wait()

    def _receive(self, proc):
        rows = self.rows["num_rows"]
        while True:
            head = proc.stdout.read(4)
            if len(head) < 4:
                return
            n = int.from_bytes(head, "little")
            body = proc.stdout.read(rows * n * 8)
            if len(body) < rows * n * 8:
                return
            data = np.frombuffer(body, dtype=np.float64).reshape(rows, n)
            with self._lock:
                if self._proc is proc:
                    self._chunks.append(data)

    def __enter__(self):
        try:
            self.open()
        except ConnectionError as e:
            if not self._optional:
                raise
            # The band's WiFi isn't joined yet after it was switched on: the app starts without it and its
            # Connection tab connects it
            print(f"{e}, starting without it")
        return self

    def __exit__(self, *exc):
        self.close()


def _reader(source):
    """The reader process: opens the board, then writes every new block to stdout as a 4-byte sample count and
    the rows x samples float64 values. Ends when the app stops reading."""
    board_id = BoardIds.SYNTHETIC_BOARD if source == "synthetic" else BoardIds.MINDROVE_WIFI_BOARD
    BoardShim.disable_board_logger()
    board = BoardShim(board_id, MindRoveInputParams())
    out = sys.stdout.buffer
    try:
        board.prepare_session()
        board.start_stream()
    except MindRoveError as e:
        out.write(f"error {e}\n".encode())
        out.flush()
        return
    out.write(b"ok\n")
    out.flush()
    while True:
        data = board.get_board_data()
        if data.shape[1]:
            out.write(data.shape[1].to_bytes(4, "little"))
            out.write(np.ascontiguousarray(data, dtype=np.float64).tobytes())
            out.flush()
        time.sleep(READ_S)


class Replay(Armband):
    """Plays raw recordings (.npy of all board rows, like data/2026-09-30-filter) in real time, in a loop.
    Package numbers and timestamps are rewritten so the stream looks live."""
    connected = True

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

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        pass


if __name__ == "__main__" and sys.argv[1:2] == ["--reader"]:
    try:
        _reader(sys.argv[2])
    except (BrokenPipeError, OSError):
        pass
