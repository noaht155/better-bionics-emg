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
# The redo check during a session needs to react within a cue, so it takes the last 2 one-second estimates and
# needs both over the limit: one noisy second doesn't stop a session, a pad that keeps humming does
HUM_FAST_OF = 2
NO_DATA_S = 1.0
DROP_WINDOW_S = 10.0
BATTERY_LOW = 15
# Drain estimate: one reading every 10 s, a straight line over the last 15 minutes. The armband reports whole
# percent, so it needs a few minutes and an actual drop before the slope means anything. While streaming it went
# from 35 to 29 % in an 11 minute session on 2026-10-01
BATTERY_SAMPLE_S = 10
BATTERY_WINDOW_S = 15 * 60
BATTERY_MIN_SPAN_S = 5 * 60


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
        self.battery_log = deque()

    def update(self, emg, package, battery=None):
        """emg: live channels x new raw samples, package: their package numbers."""
        now = time.time()
        if emg.shape[1] == 0:
            return
        self.last_data = now
        if battery is not None:
            self.battery = float(battery)
            if not self.battery_log or now - self.battery_log[-1][0] >= BATTERY_SAMPLE_S:
                self.battery_log.append((now, self.battery))
            while now - self.battery_log[0][0] > BATTERY_WINDOW_S:
                self.battery_log.popleft()
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

    def hum_recent(self):
        """Lowest of the last HUM_FAST_OF one-second hum estimates per channel, None until there are that many."""
        if len(self.hum) < HUM_FAST_OF:
            return None
        return np.array(self.hum)[-HUM_FAST_OF:].min(axis=0)

    def reset_hum(self):
        self.hum.clear()
        self.since_hum = 0

    def battery_status(self):
        """{"percent", "per_min", "minutes_left"}, the last two None until there is enough history, or None
        without an armband battery reading."""
        if self.battery is None:
            return None
        out = {"percent": self.battery, "per_min": None, "minutes_left": None}
        t = np.array([x[0] for x in self.battery_log])
        pct = np.array([x[1] for x in self.battery_log])
        if len(t) > 2 and t[-1] - t[0] >= BATTERY_MIN_SPAN_S and pct.max() > pct.min():
            drain = -np.polyfit((t - t[0]) / 60, pct, 1)[0]
            if drain > 0:
                out["per_min"] = round(float(drain), 2)
                out["minutes_left"] = round(float(self.battery / drain))
        return out

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
