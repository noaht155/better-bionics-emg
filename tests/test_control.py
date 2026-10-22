"""Control layer state machine (emg-reading/control.py): when a grip starts, stays locked and lets go."""
import numpy as np
import pytest

import ml  # noqa: F401
from control import Controller, movement, still_limit
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
    """No votes to wait for: the hand closes as soon as rest fades from the smoothed probabilities (smooth_s)."""
    c = blend()
    out = c.step(probs("fist"), 0.5)
    assert out["state"] == "blend" and 0 < out["closure"] and out["event"] is None
    for _ in range(7):
        out = c.step(probs("fist"), 0.5)
    assert out["closure"] > 0.5 and out["event"] is None


def test_blend_shape_follows_the_probabilities():
    """Half fist and half pinch closes halfway between their poses; a single flicker only nudges the shape."""
    c = blend(lock_s=0)
    mix = {g: 0.0 for g in GESTURE_SET}
    mix.update(fist=0.5, pinch=0.5)
    for _ in range(40):
        out = c.step(mix, 1.0)
    expected = POSES["rest"] + (0.5 * (np.asarray(POSES["fist"]) + np.asarray(POSES["pinch"])) - POSES["rest"])
    assert np.allclose(out["angles"], expected, atol=0.1)
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


def test_blend_lock_releases_when_the_open_hand_wins():
    """Opening the hand takes effort too, so a locked shape has to let go on the class, not only on low effort."""
    c = blend(release_mode="rest_or_class", release_n=3)
    for _ in range(30):
        c.step(probs("fist"), 0.5)
    assert c.state == "active"
    events = [c.step(probs("open"), 0.5)["event"] for _ in range(3)]
    assert events == [None, None, "release"] and c.state == "blend"


def test_blend_rest_keeps_the_hand_open():
    """Effort with rest winning (or windows too far from every grip) doesn't close the hand."""
    c = blend(lock_s=0, max_distance=6.0)
    for _ in range(40):
        out = c.step(probs("fist"), 1.0, distance=9.0)
    assert out["closure"] < 0.05
    for _ in range(40):
        out = c.step(probs("fist"), 1.0, distance=3.0)
    assert out["closure"] > 0.9


def run(c, *parts):
    """parts: (grip winning, effort, steps). Returns the events in order."""
    events = []
    for grip, effort, steps in parts:
        for _ in range(steps):
            event = c.step(probs(grip), effort)["event"]
            if event:
                events.append(event)
    return events


SHORT = ("fist", 0.5, 4)
PAUSE = ("rest", 0.0, 4)
GAP = ("rest", 0.0, 14)


def test_trigger_clench_patterns_pick_the_grip():
    c = make(mode="trigger")
    assert run(c, ("fist", 0.5, 13)) == [] and c.state == "idle"
    assert run(c, ("fist", 0.5, 1)) == ["start"] and c.grip == "fist"
    for count, grip in ((1, "pinch"), (2, "tripod"), (3, "key")):
        c = make(mode="trigger")
        assert run(c, *[SHORT, PAUSE] * count) == [] and c.pulses == count
        assert run(c, GAP) == ["start"] and c.grip == grip and c.closure == 0
    # Four clenches pick nothing
    c = make(mode="trigger")
    assert run(c, *[SHORT, PAUSE] * 4, GAP) == [] and c.state == "idle" and c.pulses == 0


def test_trigger_point_comes_from_the_classifier():
    c = make(mode="trigger", vote_n=3)
    assert run(c, ("point", 0.5, 3)) == ["start"] and c.grip == "point"


def test_trigger_squeeze_closes_relax_holds_open_opens():
    c = make(mode="trigger")
    run(c, ("fist", 0.5, 14))
    closed = c.closure
    run(c, ("fist", 0.5, 4))
    assert c.closure > closed
    closed = c.closure
    run(c, ("rest", 0.05, 10))
    assert c.closure == closed
    run(c, ("open", 0.5, 2))
    assert c.closure < closed and c.state == "active"


def test_trigger_double_open_flick_goes_back():
    c = make(mode="trigger", release_with="open")
    run(c, ("fist", 0.5, 14))
    # One flick only opens a little, a one window drop inside a flick isn't a second one
    assert run(c, ("open", 0.5, 3), ("rest", 0.5, 1), ("open", 0.5, 2), ("rest", 0.0, 30)) == []
    assert c.state == "active"
    assert run(c, ("open", 0.5, 3), ("rest", 0.0, 4), ("open", 0.5, 3)) == ["release"]
    assert c.state == "idle" and c.closure == 0
    # The flicks didn't count as clenches
    run(c, ("rest", 0.0, 20))
    assert c.state == "idle" and c.pulses == 0


def test_trigger_flicks_too_far_apart_do_nothing():
    c = make(mode="trigger", double_s=0.5, release_with="open")
    run(c, ("fist", 0.5, 14))
    assert run(c, ("open", 0.5, 2), ("rest", 0.0, 12), ("open", 0.5, 2)) == []
    assert c.state == "active"


def test_trigger_double_point_goes_back_and_open_only_opens():
    c = make(mode="trigger")
    run(c, ("fist", 0.5, 14))
    # With point as the release, opening twice only opens the hand
    assert run(c, ("open", 0.5, 3), ("rest", 0.0, 4), ("open", 0.5, 3)) == [] and c.state == "active"
    closed = c.closure
    # A point flick neither closes nor opens
    run(c, ("point", 0.5, 3), ("rest", 0.0, 30))
    assert c.closure == closed and c.state == "active"
    assert run(c, ("point", 0.5, 3), ("rest", 0.0, 4), ("point", 0.5, 3)) == ["release"]
    # The point still held after the release doesn't pick point, it has to drop out first
    assert run(c, ("point", 0.5, 10)) == [] and c.state == "idle" and c.pulses == 0
    assert run(c, ("rest", 0.0, 2), ("point", 0.5, 3)) == ["start"] and c.grip == "point"


def test_trigger_still_gate_ignores_clenches_while_moving():
    for gate, expected in ((True, []), (False, ["start"])):
        c = make(mode="trigger", still_gate=gate)
        events = []
        for grip, effort, steps in (SHORT, PAUSE, GAP):
            for _ in range(steps):
                event = c.step(probs(grip), effort, moving=True)["event"]
                if event:
                    events.append(event)
        assert events == expected
    c = make(mode="trigger", still_gate=True, vote_n=3)
    for _ in range(5):
        assert c.step(probs("point"), 0.5, moving=True)["event"] is None
    # Still again, the same clench counts
    assert run(c, ("rest", 0.0, 2), SHORT, PAUSE, GAP) == ["start"] and c.grip == "pinch"


def test_trigger_pinch_clenches_moves_the_count():
    for count, grip in ((1, None), (2, "pinch"), (3, "tripod"), (4, "key"), (5, None)):
        c = make(mode="trigger", pinch_clenches=2)
        run(c, *[SHORT, PAUSE] * count, GAP)
        assert c.grip == grip


def test_still_limit_and_movement():
    rng = np.random.default_rng(0)
    rate = 500
    still = rng.normal(0, 0.01, (3, 20 * rate))
    limit = still_limit(still, [(0, 10 * rate), (12 * rate, 20 * rate)], rate)
    assert movement(still[:, :250]) < limit
    assert movement(rng.normal(0, 0.1, (3, 250))) > limit
    assert still_limit(still, [], rate) == float("inf")


def rules(**params):
    return make(mode="rules", **params)


def test_rules_default_mapping_picks_like_trigger_mode():
    c = rules()
    assert run(c, ("fist", 0.5, 13)) == [] and c.state == "idle"
    assert run(c, ("fist", 0.5, 1)) == ["start"] and c.grip == "fist"
    for count, grip in ((1, "pinch"), (2, "tripod")):
        c = rules()
        assert run(c, *[SHORT, PAUSE] * count) == [] and c.state == "idle"
        assert run(c, GAP) == ["start"] and c.grip == grip and c.closure == 0
    # 3 is the highest count any clench rule asks for, so it fires without waiting for the pause
    c = rules()
    assert run(c, *[SHORT, PAUSE] * 3) == ["start"] and c.grip == "key"
    c = rules()
    assert run(c, ("point", 0.5, 8)) == ["start"] and c.grip == "point"


def test_rules_hold_squeeze_open_and_release():
    c = rules()
    run(c, ("fist", 0.5, 14))
    closed = c.closure
    run(c, ("rest", 0.5, 4))
    assert c.closure > closed
    closed = c.closure
    run(c, ("rest", 0.05, 10))
    assert c.closure == closed
    run(c, ("open", 0.5, 2))
    assert c.closure < closed and c.state == "active"
    # One point flick does nothing, a one window drop inside a flick isn't a second one, two flicks release
    assert run(c, ("point", 0.5, 3), ("rest", 0.0, 1), ("point", 0.5, 2), ("rest", 0.0, 20)) == []
    assert run(c, ("point", 0.5, 3), ("rest", 0.0, 4), ("point", 0.5, 3)) == ["release"]
    assert c.state == "idle" and c.closure == 0


def test_rules_never_switch_grips_while_holding():
    c = rules()
    run(c, ("fist", 0.5, 14), ("rest", 0.0, 4))
    assert run(c, SHORT, PAUSE, GAP, ("point", 0.5, 10), ("rest", 0.0, 4)) == []
    assert c.grip == "fist"


def test_rules_on_sign_classes_and_library_poses():
    from gestures import LIBRARY
    poses = dict(POSES, cylinder=LIBRARY["cylinder"]["angles"])
    c = Controller({**BASE, "mode": "rules"}, GESTURE_SET, poses)
    c.set_rules({"relaxed": [{"trigger": "little_out", "how": "held", "value": 0.2, "action": "cylinder"}],
                 "holding": [{"trigger": "two_point", "how": "repeated", "value": 2, "action": "release"}]})
    sign = lambda name: {name: 0.9, "rest": 0.1}
    events = [c.step(sign("little_out"), 0.3)["event"] for _ in range(4)]
    assert events.count("start") == 1 and c.grip == "cylinder"
    for name, steps in (("rest", 4), ("two_point", 3), ("rest", 4), ("two_point", 3)):
        events = [c.step(sign(name), 0.0)["event"] for _ in range(steps)]
    assert "release" in events and c.state == "idle"


def test_rules_close_action_closes_while_its_trigger_stays_on():
    c = rules()
    c.set_rules({"relaxed": [{"trigger": "clench", "how": "held", "value": 0.7, "action": "fist"}],
                 "holding": [{"trigger": "point", "how": "held", "value": 0.0, "action": "close"}]})
    run(c, ("fist", 0.5, 14), ("rest", 0.0, 4))
    closed = c.closure
    # Held 0 s acts on the first step: the hand closes while point stays on, even with no effort above the hold level
    run(c, ("point", 0.05, 1))
    assert c.closure > closed
    closed = c.closure
    run(c, ("point", 0.05, 5))
    assert c.closure > closed
    closed = c.closure
    run(c, ("rest", 0.05, 5))
    assert c.closure == closed


def test_rules_clench_with_a_rule_class_on_is_that_class():
    c = rules()
    # Opening is a rule's class, so the effort that comes with it is not a clench
    assert run(c, ("open", 0.5, 4), PAUSE, GAP) == [] and c.state == "idle"


def test_rules_still_gate():
    c = rules(still_gate=True)
    for _ in range(20):
        assert c.step(probs("fist"), 0.5, moving=True)["event"] is None
    assert c.state == "idle"


def test_rules_are_checked():
    from control import check_rules, rules_for
    good = [{"trigger": "clench", "how": "repeated", "value": 2, "action": "pinch"}]
    check_rules(good, POSES)
    for bad in ([dict(good[0], action="nothing")], [dict(good[0], how="twice")], [dict(good[0], value=0)],
                [dict(good[0], value=1.5)], [dict(good[0], how="held", value=9)], good + good,
                [dict(good[0], how="held", value=0.5), dict(good[0], how="held", value=1.0)],
                [dict(good[0], trigger="")]):
        with pytest.raises(ValueError):
            check_rules(bad, POSES)
    # Either list takes any rule
    check_rules({"relaxed": [dict(good[0], action="release")], "holding": good}, POSES)
    triggers = lambda rules: {s: {r["trigger"] for r in rules[s]} for s in rules}
    assert triggers(rules_for(list(GESTURE_SET))) == {"relaxed": {"clench", "point"}, "holding": {"open", "point"}}
    # Sign models get the sign default, no clench; a model without open gets no open rule
    assert triggers(rules_for(["open", "two_point", "little_out", "move_thumb_abd", "notsign", "rest"])) == {
        "relaxed": {"two_point", "little_out", "move_thumb_abd"}, "holding": {"open", "two_point"}}
    assert triggers(rules_for(["two_point", "little_out", "rest"])) == {"relaxed": {"two_point", "little_out"},
                                                                        "holding": {"two_point"}}


def test_rules_saved_as_one_list_split_by_action():
    from control import split_rules
    old = [{"trigger": "point", "how": "held", "value": 0.4, "action": "point"},
           {"trigger": "point", "how": "repeated", "value": 2, "action": "release"},
           {"trigger": "open", "how": "held", "value": 0.0, "action": "open"}]
    rules = split_rules(old)
    assert [r["action"] for r in rules["relaxed"]] == ["point"]
    assert [r["action"] for r in rules["holding"]] == ["release", "open"]


def test_rules_one_trigger_held_in_both_lists():
    c = rules()
    # Open held 0 s while holding must not stop a longer open hold from picking a pose while relaxed
    c.set_rules({"relaxed": [{"trigger": "open", "how": "held", "value": 0.7, "action": "fist"}],
                 "holding": [{"trigger": "open", "how": "held", "value": 0.0, "action": "open"}]})
    assert run(c, ("open", 0.05, 10)) == [] and c.state == "idle"
    assert run(c, ("open", 0.05, 5)) == ["start"] and c.grip == "fist"
    run(c, ("rest", 0.5, 10))
    closed = c.closure
    run(c, ("open", 0.05, 1))
    assert c.closure < closed and c.state == "active"


def test_rules_pose_while_holding_switches_at_the_same_closure():
    c = rules()
    c.set_rules({"relaxed": [{"trigger": "point", "how": "held", "value": 0.4, "action": "point"}],
                 "holding": [{"trigger": "pinch", "how": "held", "value": 0.2, "action": "tripod"}]})
    run(c, ("point", 0.5, 8), ("rest", 0.5, 10))
    closed = c.closure
    assert c.grip == "point" and closed > 0
    assert run(c, ("pinch", 0.0, 4)) == ["start"] and c.grip == "tripod" and c.closure == closed


def test_rules_higher_in_the_list_wins():
    for first, second in (("pinch", "key"), ("key", "pinch")):
        c = rules()
        c.set_rules({"relaxed": [{"trigger": "point", "how": "held", "value": 0.0, "action": first},
                                 {"trigger": "open", "how": "held", "value": 0.0, "action": second}],
                     "holding": []})
        c.step({"open": 0.9, "point": 0.9}, 0.0)
        assert c.grip == first


def test_rules_lock_freezes_the_hand_until_unlocked():
    holding = [{"trigger": "open", "how": "held", "value": 0.0, "action": "open"},
               {"trigger": "point", "how": "repeated", "value": 2, "action": "release"},
               {"trigger": "pinch", "how": "held", "value": 0.2, "action": "tripod"},
               {"trigger": "key", "how": "held", "value": 0.2, "action": "lock"}]
    c = rules()
    c.set_rules({"relaxed": [{"trigger": "clench", "how": "held", "value": 0.7, "action": "fist"}], "holding": holding})
    run(c, ("fist", 0.5, 14), ("rest", 0.0, 4))
    assert run(c, ("key", 0.0, 4)) == ["lock"] and c.locked
    closed = c.closure
    # Squeezing, opening, a pose and the release pattern all do nothing while locked
    assert run(c, ("rest", 0.0, 4), ("rest", 0.6, 10), ("open", 0.5, 6), ("pinch", 0.0, 6), ("rest", 0.0, 4),
               ("point", 0.5, 3), ("rest", 0.0, 4), ("point", 0.5, 3), ("rest", 0.0, 4)) == []
    assert c.grip == "fist" and c.closure == closed and c.state == "active"
    # The same trigger unlocks, then squeezing closes again
    assert run(c, ("key", 0.0, 4)) == ["unlock"] and not c.locked
    run(c, ("rest", 0.0, 4), ("rest", 0.6, 4))
    assert c.closure > closed
    # A separate unlock rule
    c.set_rules({"relaxed": [{"trigger": "clench", "how": "held", "value": 0.7, "action": "fist"}],
                 "holding": [{"trigger": "key", "how": "held", "value": 0.2, "action": "lock"},
                             {"trigger": "point", "how": "held", "value": 0.2, "action": "unlock"}]})
    assert run(c, ("point", 0.0, 4), ("rest", 0.0, 4), ("key", 0.0, 4), ("rest", 0.0, 4)) == ["lock"]
    assert run(c, ("point", 0.0, 4)) == ["unlock"] and not c.locked


def test_simulated_model_drives_the_control_layer():
    import time
    import tools
    control = tools.Control(None)
    control.set_params({"mode": "blend"})
    control.prepare_sim()
    predictor = tools.Predictor(None)
    predictor.simulate(GESTURE_SET, control)
    try:
        with pytest.raises(ValueError):
            predictor.set_sim({"label": "not_a_class"})
        predictor.set_sim({"label": "fist", "sure": 0.95, "effort": 0.8})
        time.sleep(0.6)
        assert predictor.state["probs"]["fist"] == 0.95 and predictor.state["simulated"]
        assert control.state["grip"] == "fist" and control.state["closure"] > 0
        assert control.state["effort"] == 0.8 and control.state["step"] > 5
    finally:
        predictor.stop()
        predictor.join(2)
    assert not predictor.state["running"]
