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
