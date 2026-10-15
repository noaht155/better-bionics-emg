"""Control layer state machine (emg-reading/control.py): when a grip starts, stays locked and lets go."""
import numpy as np
import pytest

import ml  # noqa: F401
from control import Controller
from gestures import GESTURE_SET, GESTURES
from hand_angles import JOINTS

POSES = {g: GESTURES[g][1] for g in GESTURE_SET}


def probs(grip, p=0.9):
    out = {g: (1 - p) / (len(GESTURE_SET) - 1) for g in GESTURE_SET}
    out[grip] = p
    return out


# Fixed here so the tests don't move when the tuned defaults change
BASE = {"threshold": 0.8, "vote_n": 3, "start_level": 0.15, "onset_s": 0.6, "release_mode": "either",
        "release_level": 0.08, "release_s": 0.5, "spring_w": 0}


def make(**params):
    return Controller({**BASE, **params}, GESTURE_SET, POSES)


def test_start_needs_vote_n_steps_during_onset():
    c = make(vote_n=3)
    events = [c.step(probs("fist"), 0.5)["event"] for _ in range(3)]
    assert events == [None, None, "start"]
    assert c.state == "active" and c.grip == "fist"
    # A different winner in between starts the count again
    c = make(vote_n=3)
    for grip in ("fist", "fist", "pinch", "fist", "fist"):
        assert c.step(probs(grip), 0.5)["event"] is None
    assert c.step(probs("fist"), 0.5)["event"] == "start"
    # Below the threshold or below the start level nothing counts
    c = make(vote_n=1)
    assert c.step(probs("fist", 0.5), 0.5)["state"] == "idle"
    c = make(vote_n=1)
    assert c.step(probs("fist"), 0.05)["state"] == "idle"


def test_no_start_after_onset_until_effort_drops():
    c = make(vote_n=3, onset_s=0.2)
    for _ in range(5):
        c.step(probs("rest"), 0.5)
    # Effort has been up longer than onset_s, a grip now is too late
    for _ in range(5):
        assert c.step(probs("fist"), 0.5)["state"] == "idle"
    c.step(probs("rest"), 0.0)
    out = [c.step(probs("fist"), 0.5)["event"] for _ in range(3)]
    assert out[-1] == "start"


def test_open_class_start_stays_idle():
    c = make(vote_n=2)
    assert [c.step(probs("open"), 0.5)["event"] for _ in range(4)] == [None, "open", None, None]
    assert c.state == "idle" and c.closure == 0
    # Blocked until effort drops, then a grip can start
    c.step(probs("rest"), 0.0)
    assert [c.step(probs("pinch"), 0.5)["event"] for _ in range(2)] == [None, "start"]


def test_lock_ignores_other_classes():
    c = make(vote_n=1, release_mode="rest")
    c.step(probs("fist"), 0.5)
    for grip in ("pinch", "point", "open", "key"):
        out = c.step(probs(grip), 0.6)
        assert out["grip"] == "fist" and out["state"] == "active"


def test_position_closure_holds_on_relax():
    c = make(vote_n=1, start_level=0.1, gain=2, release_mode="open_class")
    assert c.step(probs("fist"), 0.3)["closure"] == pytest.approx(0.4)
    assert c.step(probs("fist"), 0.5)["closure"] == pytest.approx(0.8)
    assert c.step(probs("rest"), 0.2)["closure"] == pytest.approx(0.8)
    c.params["hold_on_relax"] = False
    assert c.step(probs("rest"), 0.2)["closure"] == pytest.approx(0.2)


def test_speed_closure():
    c = make(vote_n=1, closure_mode="speed", hold_level=0.1, gain=2, release_mode="open_class")
    assert c.step(probs("fist"), 0.5)["closure"] == 0
    assert c.step(probs("fist"), 0.6)["closure"] == pytest.approx(0.05)
    assert c.step(probs("fist"), 0.0)["closure"] == pytest.approx(0.05)


def test_release_modes():
    def started(**params):
        c = make(vote_n=1, release_n=2, release_s=0.2, **params)
        c.step(probs("fist"), 0.5)
        return c

    c = started(release_mode="open_class")
    for _ in range(10):
        assert c.step(probs("rest"), 0.0)["state"] == "active"
    assert [c.step(probs("open"), 0.5)["event"] for _ in range(2)] == [None, "release"]
    assert c.state == "idle" and c.grip is None and c.closure == 0

    c = started(release_mode="rest")
    for _ in range(5):
        assert c.step(probs("open"), 0.5)["state"] == "active"
    events = [c.step(probs("rest"), 0.0)["event"] for _ in range(4)]
    assert events == [None, None, None, "release"]

    c = started(release_mode="either")
    assert [c.step(probs("open"), 0.5)["event"] for _ in range(2)] == [None, "release"]
    c = started(release_mode="either")
    assert [c.step(probs("rest"), 0.0)["event"] for _ in range(4)][-1] == "release"


def test_angles_follow_closure():
    c = make(vote_n=1, gain=1, start_level=0.0, release_mode="open_class")
    out = c.step(probs("fist"), 1.0)
    assert out["angles"] == POSES["fist"]
    c = Controller({"vote_n": 1, "gain": 1, "start_level": 0.0}, GESTURE_SET, POSES)
    out = c.step(probs("fist"), 1.0)
    # The spring only moves part of the way in one step
    j = JOINTS.index("index_pip")
    assert POSES["rest"][j] < out["angles"][j] < POSES["fist"][j]


def test_far_windows_count_as_rest():
    """A window farther than max_distance from every trained grip can't vote for one, whatever its probability."""
    c = make(vote_n=2, max_distance=6.0)
    assert [c.step(probs("fist"), 0.5, distance=9.0)["event"] for _ in range(4)] == [None] * 4
    assert [c.step(probs("fist"), 0.5, distance=3.0)["event"] for _ in range(2)] == [None, "start"]
    off = make(vote_n=2, max_distance=0.0)
    assert [off.step(probs("fist"), 0.5, distance=9.0)["event"] for _ in range(2)] == [None, "start"]


def test_rest_or_class_release():
    """rest_or_class lets go when rest wins release_n steps even while effort stays up, or on low effort."""
    c = make(vote_n=1, release_mode="rest_or_class", release_n=3)
    c.step(probs("fist"), 0.5)
    assert [c.step(probs("rest"), 0.5)["event"] for _ in range(3)] == [None, None, "release"]
    c = make(vote_n=1, release_mode="rest_or_class", release_n=3, release_s=0.5)
    c.step(probs("fist"), 0.5)
    events = [c.step(probs("fist"), 0.0)["event"] for _ in range(10)]
    assert "release" in events
    # A far window counts as rest for the release too
    c = make(vote_n=1, release_mode="rest_or_class", release_n=2, max_distance=6.0)
    c.step(probs("fist"), 0.5, distance=2.0)
    assert [c.step(probs("fist"), 0.5, distance=9.0)["event"] for _ in range(2)] == [None, "release"]


def test_distance_space_inside_and_outside_a_class():
    """gesture_model.distance_space: training windows sit a few spreads from their class centre, a window far from
    every class is many spreads away."""
    import numpy as np
    from ml.gesture_model import distance_space
    rng = np.random.default_rng(0)
    centres = {"rest": np.zeros(6), "fist": np.r_[5.0, np.zeros(5)], "pinch": np.r_[0, 5.0, np.zeros(4)]}
    feats = np.vstack([rng.normal(c, 1.0, (200, 6)) for c in centres.values()])
    labels = np.repeat(list(centres), 200).astype(object)
    space = distance_space(feats, labels)

    def dist(x, cls):
        z = space["lda"].transform(x[None])[0] - space["centres"][cls]
        return float(np.sqrt(z @ space["cov_inv"] @ z))
    inside = np.median([dist(x, "fist") for x in feats[labels == "fist"]])
    outside = dist(np.r_[5.0, 0, 0, 0, 0, 0] + np.r_[0, -12.0, 0, 0, 0, 0], "fist")
    assert inside < 2.5 and outside > 3 * inside


def blend(**params):
    return make(**{"mode": "blend", "smooth_s": 0.25, "lock_s": 0.5, "start_level": 0.02, "gain": 1.5, **params})


def test_blend_closes_with_effort_straight_away():
    """No votes to wait for: the first window with effort already closes the hand."""
    c = blend()
    out = c.step(probs("fist"), 0.5)
    assert out["state"] == "blend" and out["closure"] > 0.5 and out["event"] is None


def test_blend_shape_follows_the_probabilities():
    """Half fist and half pinch closes halfway between their poses; a single flicker only nudges the shape."""
    c = blend(lock_s=0)
    mix = {g: 0.0 for g in GESTURE_SET}
    mix.update(fist=0.5, pinch=0.5)
    for _ in range(40):
        out = c.step(mix, 1.0)
    expected = POSES["rest"] + (0.5 * (np.asarray(POSES["fist"]) + np.asarray(POSES["pinch"])) - POSES["rest"])
    assert np.allclose(out["angles"], expected, atol=1e-3)
    for _ in range(40):
        c.step(probs("fist"), 1.0)
    before = np.asarray(c.step(probs("fist"), 1.0)["angles"])
    flicker = np.asarray(c.step(probs("point"), 1.0)["angles"])
    assert np.abs(flicker - before).max() < 0.3 * np.abs(np.asarray(POSES["point"]) - POSES["fist"]).max()


def test_blend_locks_and_lets_go():
    c = blend()
    events = [c.step(probs("fist"), 0.5)["event"] for _ in range(30)]
    assert "start" in events and c.state == "active" and c.grip == "fist"
    # Locked: another grip winning doesn't change the shape
    assert c.step(probs("pinch"), 0.5)["grip"] == "fist"
    events = [c.step(probs("rest"), 0.0)["event"] for _ in range(40)]
    assert "release" in events and c.state == "blend"
