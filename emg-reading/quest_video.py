"""The headset's picture: `adb screenrecord` on the Quest over USB, raw H.264, split into one access unit per frame
for the VR tab's WebCodecs decoder and for recording it beside a session. One capture shared by all of them, running
only while someone watches, so nothing runs on the headset otherwise. screenrecord stops itself after 3 minutes, so
it is started again, which shows as a short pause.
"""
import queue
import subprocess
import threading
import time

import quest_link
from hand_angles import JOINTS
from session import DATA_DIR

# Wide enough to read the menu, small enough for USB and the headset's encoder next to the sim
SIZE = "1280x720"
BIT_RATE = 3_000_000
READ_BYTES = 1 << 16
# About 2 s at the headset's frame rate
QUEUE_FRAMES = 120
RETRY_S = 2.0
POLL_S = 0.5
VIDEO_DIR = DATA_DIR / "headset-video"
# NAL unit types: slices carry a frame (5 is a key frame), SPS and PPS come before the first one
SLICE, IDR, SPS = 1, 5, 7


def capture_command(adb):
    return [adb, "exec-out", "screenrecord", "--output-format=h264", f"--size={SIZE}", f"--bit-rate={BIT_RATE}", "-"]


def codec_string(sps):
    """avc1.PPCCLL from an SPS NAL unit (header byte first), as WebCodecs wants it."""
    return "avc1." + "".join(f"{b:02x}" for b in sps[1:4])


class Splitter:
    """Annex B bytes in, (key, codec, access unit) out. Each frame is one slice on the Quest's encoder, so an access unit
    is the NAL units since the last slice up to and including the next one."""

    def __init__(self):
        self.buf = b""
        self.pending = b""
        self.codec = None

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            start = self.buf.find(b"\x00\x00\x01")
            if start < 0:
                break
            end = self.buf.find(b"\x00\x00\x01", start + 3)
            if end < 0:
                break
            # A 4-byte start code leaves its leading zero on the unit before it
            if end > 0 and self.buf[end - 1] == 0:
                end -= 1
            out += self._unit(self.buf[start:end])
            self.buf = self.buf[end:]
        return out

    def _unit(self, unit):
        nal = unit[3:]
        if not nal:
            return []
        kind = nal[0] & 0x1F
        if kind == SPS and len(nal) >= 4:
            self.codec = codec_string(nal)
        self.pending += b"\x00" + unit if unit[:3] == b"\x00\x00\x01" else unit
        if kind not in (SLICE, IDR):
            return []
        frame, self.pending = self.pending, b""
        return [(kind == IDR, self.codec, frame)]


def start_capture():
    """A running screenrecord process, or None without adb. Its stdout is the H.264 stream."""
    adb = quest_link.find_adb()
    if adb is None:
        return None
    return subprocess.Popen(capture_command(adb), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


class Capture(threading.Thread):
    """The one screenrecord, shared by everyone who watches (VR tab pages, a session's recording). Each subscriber
    gets ("codec", str), ("frame", key, bytes), ("error", text) or ("lost", None) through its queue. A new
    subscriber restarts the capture, as screenrecord only sends a key frame when it starts; the others see a short
    pause. A subscriber that falls QUEUE_FRAMES behind gets ("lost", None) and is dropped."""

    def __init__(self):
        super().__init__(daemon=True, name="quest-video")
        self._lock = threading.Lock()
        self._subs = []
        self._wake = threading.Event()
        self._proc = None
        self._restarted = False

    def subscribe(self):
        q = queue.Queue(QUEUE_FRAMES)
        with self._lock:
            self._subs.append(q)
            self._restart()
        self._wake.set()
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)
            if not self._subs:
                self._restart()

    def _restart(self):
        self._restarted = True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()

    def _send(self, msg):
        with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    self._subs.remove(q)
                    with q.mutex:
                        q.queue.clear()
                    q.put_nowait(("lost", None))

    def run(self):
        while True:
            self._wake.wait()
            with self._lock:
                if not self._subs:
                    self._wake.clear()
                    continue
                proc = self._proc = start_capture()
                self._restarted = False
            if proc is None:
                self._send(("error", "no adb on this computer"))
                time.sleep(RETRY_S)
                continue
            splitter, codec, frames = Splitter(), None, 0
            while True:
                data = proc.stdout.read(READ_BYTES)
                if not data:
                    break
                for key, unit_codec, frame in splitter.feed(data):
                    if unit_codec and unit_codec != codec:
                        codec = unit_codec
                        self._send(("codec", codec))
                    if codec:
                        self._send(("frame", key, frame))
                        frames += 1
            proc.wait()
            if frames == 0 and self._subs and not self._restarted:
                self._send(("error", "no picture from the headset: plug the Quest in over USB and put it on"))
                time.sleep(RETRY_S)


CAPTURE = None


def capture():
    """The shared capture, started on first use."""
    global CAPTURE
    if CAPTURE is None:
        CAPTURE = Capture()
        CAPTURE.start()
    return CAPTURE


class SessionVideo(threading.Thread):
    """When turned on (on, off at every start: Noah, 2026-11-26), records the headset's picture while a session
    records with the Quest app connected, into
    data/headset-video/<session name>/, outside the session folder so the hub's session files stay as they are:
    one .h264 per capture (each starts with a key frame) and frames.csv with each frame's arrival time (time.time(),
    about 0.1 s after the headset showed it), to line the picture up with the session's events."""

    def __init__(self, recorder, connected):
        super().__init__(daemon=True, name="session-video")
        self.recorder, self.connected = recorder, connected
        self.folder = None
        self.on = False
        self._session_folder = None
        self._lock = threading.Lock()
        self._hands = None

    def run(self):
        sub = None
        while True:
            session = self.recorder.session
            want = (self.on and session is not None and session.folder is not None and self.connected()) \
                or (sub is not None and session is not None and session.folder == self._session_folder)
            if want and sub is None:
                sub = self._start(session.folder)
            elif not want and sub is not None:
                self._stop(sub)
                sub = None
            if sub is None:
                time.sleep(POLL_S)
                continue
            try:
                msg = sub.get(timeout=POLL_S)
            except queue.Empty:
                continue
            if msg[0] == "codec":
                self._segment()
            elif msg[0] == "frame" and self._file is not None:
                self._file.write(msg[2])
                self._frames.write(f"{self._seg},{self._n},{time.time():.3f},{int(msg[1])}\n")
                self._n += 1
            elif msg[0] == "lost":
                sub = capture().subscribe()

    def _start(self, session_folder):
        self._session_folder = session_folder
        self.folder = VIDEO_DIR / session_folder.name
        self.folder.mkdir(parents=True, exist_ok=True)
        self._frames = open(self.folder / "frames.csv", "a")
        self._frames.write("segment,frame,time,key\n")
        with self._lock:
            self._hands = open(self.folder / "hands.csv", "a")
            self._hands.write("time,hand," + ",".join(JOINTS) + "\n")
        self._file, self._seg, self._n = None, -1, 0
        print(f"recording the headset's picture into {self.folder}")
        return capture().subscribe()

    def _segment(self):
        if self._file is not None:
            self._file.close()
        self._seg += 1
        self._n = 0
        self._file = open(self.folder / f"{self._seg:03d}.h264", "wb")

    def add_hands(self, view):
        """Both real hands' angles from the headset's view, beside the picture: hands.csv with the arrival time,
        which hand (free, or other: the prosthetic side), and its 20 angles; untracked hands are left out."""
        with self._lock:
            if self._hands is None:
                return
            now = time.time()
            for side in ("free", "other"):
                angles = view.get(f"{side}_angles")
                if view.get(f"{side}_tracked") and isinstance(angles, list) and len(angles) == len(JOINTS):
                    self._hands.write(f"{now:.3f},{side}," + ",".join(f"{a:.1f}" for a in angles) + "\n")

    def _stop(self, sub):
        capture().unsubscribe(sub)
        with self._lock:
            self._hands.close()
            self._hands = None
        if self._file is not None:
            self._file.close()
        self._frames.close()
        self._file = self._session_folder = None


SESSION_VIDEO = None


def log_hands(view):
    """Passes a headset view's hand angles to the session recording, when one runs."""
    if SESSION_VIDEO is not None:
        SESSION_VIDEO.add_hands(view)
