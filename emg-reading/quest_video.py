"""The headset's picture: `adb screenrecord` on the Quest over USB, raw H.264, split into one access unit per frame
for the VR tab's WebCodecs decoder. One capture shared by all of them, running
only while someone watches, so nothing runs on the headset otherwise. screenrecord stops itself after 3 minutes, so
it is started again, which shows as a short pause.
"""
import queue
import subprocess
import threading
import time

import quest_link

# Wide enough to read the menu, small enough for USB and the headset's encoder next to the sim
SIZE = "1280x720"
BIT_RATE = 3_000_000
READ_BYTES = 1 << 16
# About 2 s at the headset's frame rate
QUEUE_FRAMES = 120
RETRY_S = 2.0
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
