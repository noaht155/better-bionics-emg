"""Signal quality checks on the live armband stream: mains hum, pad contact, dropped samples, no data."""
import time
from collections import deque

import numpy as np
from scipy import signal

from processing import ENVELOPE_MS, MAINS_HZ, contact_lost

# 60 Hz line in the raw signal, median of the last 5 one-second estimates. On the 2026-09-30 session good pads
# stayed under 46 uV (strongest during grips, muscle has power at 60 Hz too), the pad losing contact read 120 to 180
HUM_WARN_UV = 80.0
HUM_MEDIAN_OF = 5
NO_DATA_S = 1.0
DROP_WINDOW_S = 10.0
BATTERY_LOW = 15


def line_rms(x, rate, hz=MAINS_HZ[0]):
    """RMS in uV of a mains line above the noise floor around it. x: channels x samples of raw data."""
    x = signal.detrend(x, axis=1)
    w = np.hanning(x.shape[1])
    psd = 2 * np.abs(np.fft.rfft(x * w, axis=1)) ** 2 / (rate * np.sum(w ** 2))
    f = np.fft.rfftfreq(x.shape[1], 1 / rate)
    df = f[1] - f[0]
    peak = (np.abs(f - hz) <= 2)
    near = ((np.abs(f - hz) > 3) & (np.abs(f - hz) <= 10))
    excess = psd[:, peak].sum(axis=1) - np.median(psd[:, near], axis=1) * peak.sum()
    return np.sqrt(np.maximum(excess, 0) * df)


class SignalQuality:
    def __init__(self, channels, rate):
        self.channels = channels
        self.rate = rate
        self.raw = np.zeros((len(channels), 0))
        self.hum = deque(maxlen=HUM_MEDIAN_OF)
        self.since_hum = 0
        self.last_package = None
        self.drops = deque()
        self.dropped_total = 0
        self.last_data = time.time()
        self.lost = np.zeros(len(channels), bool)
        self.battery = None

    def update(self, emg, package, battery=None):
        """emg: live channels x new raw samples, package: their package numbers."""
        now = time.time()
        if emg.shape[1] == 0:
            return
        self.last_data = now
        if battery is not None:
            self.battery = float(battery)
        if self.last_package is not None:
            gaps = np.diff(np.concatenate([[self.last_package], package]))
            # Counter resets and wraps show up as negative or huge steps, those aren't drops
            missing = int(np.sum(np.where((gaps > 1) & (gaps < 10 * self.rate), gaps - 1, 0)))
            if missing:
                self.drops.append((now, missing))
                self.dropped_total += missing
        self.last_package = package[-1]

        self.raw = np.hstack([self.raw, emg])[:, -self.rate:]
        self.lost = contact_lost(self.raw[:, -int(ENVELOPE_MS / 1000 * self.rate):])
        self.since_hum += emg.shape[1]
        if self.since_hum >= self.rate and self.raw.shape[1] == self.rate:
            self.since_hum = 0
            self.hum.append(line_rms(self.raw, self.rate))

    def hum_uv(self):
        return np.median(np.array(self.hum), axis=0) if self.hum else np.zeros(len(self.channels))

    def warnings(self):
        """List of (code, message) for everything wrong right now."""
        now = time.time()
        out = []
        if now - self.last_data > NO_DATA_S:
            out.append(("no_data", f"No EMG for {now - self.last_data:.0f} s, is the armband on and streaming?"))
            return out
        while self.drops and now - self.drops[0][0] > DROP_WINDOW_S:
            self.drops.popleft()
        dropped = sum(n for _, n in self.drops)
        if dropped:
            out.append(("dropped", f"{dropped} EMG samples dropped in the last {DROP_WINDOW_S:.0f} s"))
        if self.lost.any():
            chs = [ch for ch, lost in zip(self.channels, self.lost) if lost]
            out.append(("contact", f"Pad losing contact on ch {', '.join(map(str, chs))}"))
        hum = self.hum_uv()
        if (hum > HUM_WARN_UV).any():
            bad = ", ".join(f"{ch} ({h:.0f} uV)" for ch, h in zip(self.channels, hum) if h > HUM_WARN_UV)
            out.append(("hum", f"Mains hum on ch {bad}, check the pad contact"))
        if self.battery is not None and self.battery < BATTERY_LOW:
            out.append(("battery", f"Armband battery at {self.battery:.0f} %"))
        return out
