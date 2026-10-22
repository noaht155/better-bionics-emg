"""Haptic levels from a recorded session: the outputs should stay off at rest."""
import numpy as np
import pytest

import ml  # noqa: F401
import calibrate
from haptics import level
from ml import dataset
from processing import common_average, envelope_series, filter_block
from session import load_session, segments

SESSIONS = dataset.good_sessions()


@pytest.mark.skipif(not SESSIONS, reason="no recorded sessions")
def test_session_levels_keep_rest_quiet():
    """Levels from the whole newest session, played back over its own rest cues. On held-out repetitions of 9
    sessions an output was on in 1.7 % of rest windows (worst session 4.2 %), see devlog.md 2026-10-06."""
    folder = SESSIONS[-1]
    result, _ = calibrate.from_session(folder)
    data = load_session(folder)
    channels = data["meta"]["channels"]
    y = common_average(filter_block(data["emg"][channels], 500))
    trim = int(calibrate.REACTION_S * 500)
    rests = [s for s in segments(data["events"]) if s["kind"] == "hold" and s["label"] == "rest" and s["sample1"]
             and not s["bad"] and s["sample1"] - s["sample0"] > trim]
    on = []
    for i, ch in enumerate(channels):
        assert 0 < result[ch]["rest"] < result[ch]["max"]
        env = np.concatenate([envelope_series(y[i, s["sample0"] + trim:s["sample1"]], 500) for s in rests])
        on.append([level(e, result[ch]["rest"], result[ch]["max"]) > 0 for e in env])
    assert np.mean(np.any(on, axis=0)) < 0.05


def test_band_bump_doesnt_lose_channels():
    """A bump trips the contact check for one window on most channels. In a whole session that mustn't drop them
    from the common average (2026-10-21_141651: 7 of 8 dropped, ch 1 averaged with itself to a rest level of 0)."""
    raw = np.random.default_rng(0).normal(0, 10, (8, 500 * 60))
    raw[:7, 1000:1010] += 200_000
    assert calibrate.lost_channels(raw, 500).sum() == 7
    assert calibrate.lost_channels(raw, 500, calibrate.SESSION_LOST_SHARE).sum() == 0
    raw[3, :] += np.tile([0, 200_000], 500 * 30)
    assert list(np.flatnonzero(calibrate.lost_channels(raw, 500, calibrate.SESSION_LOST_SHARE))) == [3]
    with pytest.raises(ValueError, match="no signal"):
        calibrate.levels(1, np.zeros(10), np.ones(10))
