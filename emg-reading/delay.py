"""Camera and armband delays of a recording setup, from the delay test on System > Dev, without the ESP32.
Measured once per setup and saved in devices.json (devices.save_shift).

Armband round trip: the band is switched between EEG mode (offsets of tens of mV) and TEST mode (a 314 uV square
wave near 0) at logged times; the first sample where most channels jump is the switch arriving back. Same method as
the 2026-10-08 measurement (17 ms on the desktop). The EMG stamps can't lag the true sample time by more than this.

Camera: the page flashes the screen between black and white at logged times, the camera films it in a mirror, and
the first frame past half brightness counts. Frame times are fitted like the labels' (ml.dataset.fitted_frame_times).
The screen's own delay (refresh and panel) is included, so this is an upper bound on the camera delay.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ml.dataset import fitted_frame_times  # noqa: E402

# A mode switch moves every channel by tens of mV, EMG and the test signal stay far below this between samples
STEP_UV = 10_000
STEP_CHANNELS = 5
# A switch, flash or frame further than this from its cause isn't counted
MAX_DELAY_S = 0.3
# Brightness levels (0 to 255) between dark and light frames, below this the camera hardly sees the screen
MIN_CONTRAST = 20


def round_trips(data, rows, switches):
    """data: all board rows. switches: send times from Armband.set_mode. One delay per switch that came back."""
    emg = data[rows["emg"]]
    t = data[rows["timestamp"]]
    jumps = np.flatnonzero((np.abs(np.diff(emg, axis=1)) > STEP_UV).sum(axis=0) >= STEP_CHANNELS) + 1
    out = []
    for sent in switches:
        after = jumps[t[jumps] > sent]
        if after.size and t[after[0]] - sent < MAX_DELAY_S:
            out.append(t[after[0]] - sent)
    return np.array(out)


def screen_delays(log, switches):
    """log: (time, frame number, brightness) rows from Camera.probe. switches: (time, on) of each flash change.
    Returns the delays of the switches to light and to dark separately."""
    if len(log) < 30:
        raise ValueError("the camera sent almost no frames during the test")
    times = fitted_frame_times(log[:, 0], log[:, 1])
    level = log[:, 2]
    dark, light = np.percentile(level, [5, 95])
    if light - dark < MIN_CONTRAST:
        raise ValueError(f"the flashes barely change the picture (brightness {dark:.0f} to {light:.0f}), hold the "
                         "mirror so the screen fills most of the camera's view and dim the room")
    middle = (dark + light) / 2
    on, off = [], []
    for when, lit in switches:
        past = (level > middle) if lit else (level < middle)
        hit = np.flatnonzero(past & (times > when))
        if hit.size and times[hit[0]] - when < MAX_DELAY_S:
            (on if lit else off).append(times[hit[0]] - when)
    return np.array(on), np.array(off)


def camera_delay(on, off):
    """The two directions differ (88 and 15 ms on the MacBook, 47 ms apart for the desktop's LED), so the delays form
    two clusters; the median of all of them lands anywhere in the gap between. The mean of each direction's median
    cancels the difference instead."""
    parts = [np.median(d) for d in (on, off) if len(d)]
    return float(np.mean(parts))


def label_shift(camera_s, round_trip_s):
    """The EMG stamps lag the true sample time by 0 to the round trip, so the shift lies between camera - round trip
    and camera; the middle, like the desktop's 28 ms."""
    return camera_s - round_trip_s / 2
