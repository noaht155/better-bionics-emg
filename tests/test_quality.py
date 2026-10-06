"""Signal quality checks: the hum that makes the recorder redo a cue, dropped samples and pad contact."""
import numpy as np

import ml  # noqa: F401
from quality import HUM_WARN_UV, SignalQuality, line_rms

RATE = 500
CHANNELS = list(range(8))


def raw(seconds, hum_uv=None, seed=0):
    """Raw-like data: DC offset, EMG-like noise, and optionally 60 Hz with the given RMS per channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = 30_000 + rng.normal(0, 20, (len(CHANNELS), len(t)))
    if hum_uv is not None:
        x += np.asarray(hum_uv)[:, None] * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)
    return x


def feed(q, x, start_package=0):
    """In 10-sample packets like the armband."""
    for i in range(0, x.shape[1], 10):
        q.update(x[:, i:i + 10], np.arange(start_package + i, start_package + i + x[:, i:i + 10].shape[1]))


def test_line_rms_measures_the_hum():
    hum = np.array([0, 10, 50, 100, 150, 0, 0, 0])
    measured = line_rms(raw(1, hum), RATE)
    # Noise alone reads a few uV, far under the 80 uV warning
    assert np.all(np.abs(measured - hum) < 0.1 * hum + 5)


def test_emg_noise_alone_is_not_hum():
    """Muscle signal also has power at 60 Hz, only a peak above the neighbouring frequencies counts."""
    assert np.all(line_rms(raw(1), RATE) < 10)


def test_hum_on_one_channel_is_flagged_after_two_seconds():
    q = SignalQuality(CHANNELS, RATE)
    hum = np.zeros(8)
    hum[5] = 150
    feed(q, raw(1, hum))
    assert q.hum_recent() is None
    feed(q, raw(1.2, hum, seed=1), start_package=RATE)
    flagged = q.hum_recent() > HUM_WARN_UV
    assert list(np.flatnonzero(flagged)) == [5]


def test_clean_signal_flags_nothing():
    q = SignalQuality(CHANNELS, RATE)
    feed(q, raw(3))
    assert not (q.hum_recent() > HUM_WARN_UV).any()
    assert q.warnings() == []


def test_dropped_samples_counted_from_package_numbers():
    q = SignalQuality(CHANNELS, RATE)
    x = raw(1)
    q.update(x[:, :10], np.arange(10))
    q.update(x[:, 10:20], np.arange(15, 25))
    assert q.dropped_total == 5
    # A counter reset is not a drop
    q.update(x[:, 20:30], np.arange(10))
    assert q.dropped_total == 5


def test_floating_pad_is_lost_contact():
    q = SignalQuality(CHANNELS, RATE)
    x = raw(1)
    # A floating pad swings over the whole input range
    x[3] = 30_000 + 200_000 * np.sin(2 * np.pi * 2 * np.arange(x.shape[1]) / RATE)
    feed(q, x)
    assert list(np.flatnonzero(q.lost)) == [3]
