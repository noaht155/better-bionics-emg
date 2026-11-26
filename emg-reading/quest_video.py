"""The headset's picture for the VR tab: `adb screenrecord` on the Quest over USB, raw H.264, split into one access unit
per frame for the page's WebCodecs decoder. One capture per watching page, started when it connects and stopped when
it leaves, so nothing runs on the headset while nobody watches. screenrecord stops itself after 3 minutes, so it is
started again, which shows as a short pause on the page.
"""
import subprocess

import quest_link

# Wide enough to read the menu, small enough for USB and the headset's encoder next to the sim
SIZE = "1280x720"
BIT_RATE = 4_000_000
READ_BYTES = 1 << 16
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
