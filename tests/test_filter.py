"""The live filter must give exactly what the models were trained on.

    emg-reading/.venv/Scripts/python -m pytest tests
"""
import numpy as np

import ml  # noqa: F401 (puts emg-reading/ on the import path)
from processing import StreamFilter, filter_block

RATE = 500


def fake_raw(seconds=20, channels=8, seed=0):
    """Large drifting DC offset, mains hum and noise bursts, like the raw armband rows."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * RATE) / RATE
    offset = 30_000 + np.cumsum(rng.normal(0, 5, (channels, len(t))), axis=1)
    hum = 3 * np.sin(2 * np.pi * 60 * t)
    bursts = rng.normal(0, 1, (channels, len(t))) * (50 * (np.sin(2 * np.pi * 0.3 * t) > 0.5) + 5)
    return offset + hum + bursts


def test_chunks_match_whole_recording():
    """The recorder filters the stream in packets of 10 samples, training filters whole sessions at once."""
    raw = fake_raw()
    stream = StreamFilter(raw.shape[0], RATE)
    live = np.hstack([stream.process(raw[:, i:i + 10]) for i in range(0, raw.shape[1], 10)])
    np.testing.assert_allclose(live, filter_block(raw, RATE), rtol=0, atol=1e-6)


def test_uneven_chunks():
    raw = fake_raw(seconds=5, seed=1)
    cuts = np.cumsum(np.random.default_rng(2).integers(1, 40, 200))
    cuts = cuts[cuts < raw.shape[1]]
    stream = StreamFilter(raw.shape[0], RATE)
    live = np.hstack([stream.process(part) for part in np.split(raw, cuts, axis=1)])
    np.testing.assert_allclose(live, filter_block(raw, RATE), rtol=0, atol=1e-6)


def test_offset_removed_without_startup_spike():
    """The filter starts in steady state, so the tens of mV of offset don't ring at the start."""
    y = filter_block(fake_raw(seconds=5), RATE)
    assert np.abs(y[:, :RATE]).max() < 500
    assert abs(y[:, RATE:].mean()) < 1


def test_mains_removed():
    t = np.arange(10 * RATE) / RATE
    hum = np.vstack([100 * np.sin(2 * np.pi * f * t) for f in (60, 120, 180)])
    y = filter_block(hum, RATE)[:, 2 * RATE:]
    assert np.sqrt(np.mean(y ** 2)) < 1
