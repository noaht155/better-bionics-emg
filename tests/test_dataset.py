"""Training windows and their labels. The checks on real sessions are skipped where emg-reading/data/ has none."""
import numpy as np
import pytest

import ml  # noqa: F401
from gestures import GESTURE_SET
from ml import dataset, gesture_model
from session import load_session, segments

SESSIONS = dataset.good_sessions()
needs_sessions = pytest.mark.skipif(not SESSIONS, reason="no recorded sessions")


@pytest.fixture(scope="module")
def newest():
    return SESSIONS[-1], dataset.load(SESSIONS[-1])


@needs_sessions
def test_shapes(newest):
    _, d = newest
    n = len(d["grip"])
    assert d["x"].shape == (n, 8, dataset.WINDOW)
    assert d["angles"].shape[0] == n
    assert np.all(np.diff(d["ends"]) > 0)


@needs_sessions
def test_grip_windows_lie_inside_the_trimmed_hold(newest):
    """Every grip label comes from a hold cue that isn't marked bad, at least TRIM_START_S after its start."""
    folder, d = newest
    holds = [s for s in segments(load_session(folder)["events"]) if s["kind"] == "hold" and s["sample1"]]
    trim = int(dataset.TRIM_START_S * dataset.RATE)
    for end, grip in zip(d["ends"], d["grip"]):
        if not grip:
            continue
        seg = next(s for s in holds if s["sample0"] < end <= s["sample1"])
        assert seg["label"] == grip
        assert not seg["bad"]
        assert end - dataset.WINDOW >= seg["sample0"] + trim


@needs_sessions
def test_labels_are_in_range(newest):
    _, d = newest
    assert set(d["grip"]) - {""} <= set(GESTURE_SET)
    held = (d["grip"] != "") & (d["grip"] != "rest")
    assert set(d["rep"][held]) <= {1, 2, 3}


@needs_sessions
@pytest.mark.parametrize("folder", SESSIONS, ids=lambda f: f.name[:17])
def test_no_impossible_angles(folder):
    """Camera frames with a trained joint bent backwards past IMPOSSIBLE_DEG are tracking glitches and are dropped from
    the labels (0.02 to 2.2 % of angle windows per session had one). A window between a good and a dropped frame can't
    get one either, since labels are interpolated only between good frames. Abduction isn't trained and isn't checked:
    it wraps round +-180 degrees on a bent finger."""
    from hand_angles import JOINTS, RELIABLE
    a = dataset.load(folder)["angles"][:, [JOINTS.index(j) for j in RELIABLE]]
    a = a[~np.isnan(a).any(axis=1)]
    assert not (a <= dataset.IMPOSSIBLE_DEG).any()


def test_moving_windows_count_as_much_as_the_rest():
    pytest.importorskip("torch")
    from ml.train_ringnet import MOVING, angle_weights
    rng = np.random.default_rng(0)
    n = 1000
    kind = rng.choice(["hold", "rest", "finger", "free"], n, p=[0.6, 0.2, 0.15, 0.05])
    angles = rng.normal(40, 20, (n, 20))
    angles[rng.random(n) < 0.1] = np.nan
    w = angle_weights({"grip": np.full(n, ""), "kind": kind, "angles": angles})
    known = ~np.isnan(angles).any(axis=1)
    moving = np.isin(kind, MOVING) & known
    assert w[moving].sum() == pytest.approx(w[known & ~moving].sum())
    assert np.all(w[~moving] == 1)


def test_batch_features_match_one_window():
    """LivePredictor computes the LDA features one window at a time, training does it in batches."""
    x = np.random.default_rng(0).normal(0, 30, (20, 8, 100))
    batch = gesture_model.window_features(x, True, True)
    for i in range(len(x)):
        np.testing.assert_allclose(gesture_model.window_features(x[i], True, True), batch[i], rtol=1e-9, atol=1e-9)
