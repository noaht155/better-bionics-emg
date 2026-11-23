"""Haptic levels from a recorded session: the outputs should stay off at rest."""
import numpy as np
import pytest

import ml  # noqa: F401
import calibrate
from haptics import (FADE, GAP, GRIP_CODES, OUTPUTS, SHORT, STEP_MS, SWEEP, TICK_OUTPUT, buzz, level, output_duties,
                     pattern, scaled_duty)
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


def test_effort_floor():
    """Above 0 the duty starts at the floor, 0 stays off, full effort is 100 % whatever the floor."""
    assert scaled_duty(0.0, 35) == 0 and scaled_duty(1.0, 35) == 100 and scaled_duty(0.01, 35) == 36
    assert scaled_duty(0.5, 0) == 50
    raw = np.zeros((8, 75))
    filtered = np.vstack([np.full(75, v) for v in (0, 10, 20, 30, 40, 50, 60, 70)])
    ranges = {ch: (0.0, 100.0) for ch in range(8)}
    plain, _ = output_duties(raw, filtered, list(range(8)), ranges, False)
    floored, _ = output_duties(raw, filtered, list(range(8)), ranges, False, 30)
    assert plain[:3] == [0, 10, 20] and floored[:3] == [0, 37, 44]


def test_selection_patterns_tell_events_apart():
    """Every grip has its own pattern on all motors, release fades, lock and unlock sweep opposite ways, and a tick
    only uses one motor so it can't be taken for a pick."""
    picks = {g: pattern("start", 100, g) for g in [*GRIP_CODES, "some_library_pose"]}
    assert len({str(p) for p in picks.values()}) == len(picks)
    for steps in picks.values():
        assert all(len(d) == len(OUTPUTS) and len(set(d)) == 1 for d in steps)
        assert steps[0][0] == 100 and steps[-1][0] == 100
    release = [d[0] for d in pattern("release", 80)]
    assert release[0] == 80 and release == sorted(release, reverse=True) and release[-1] > 0
    lock, unlock = pattern("lock"), pattern("unlock")
    assert [d.index(100) for d in lock] == list(range(len(OUTPUTS))) and lock == unlock[::-1]
    tick = pattern("tick", 60)
    assert all(d[TICK_OUTPUT] == 60 and sum(d) == 60 for d in tick)


def test_tick_never_cuts_a_pick():
    import threading
    import time
    import tools
    h = tools.Haptics(None, threading.Lock())
    h.cue("start", "tripod")
    h.cue("tick")
    assert h._cue[2] == "start" and h._pattern_step(h._cue) == [100] * len(OUTPUTS)
    # Once the pick has played, a tick plays, and any event cuts a tick
    h._cue = (h._cue[0], time.perf_counter() - 10, "start")
    assert h._pattern_step(h._cue) is None
    h.cue("tick")
    assert h._cue[2] == "tick"
    h.cue("release")
    assert h._cue[2] == "release"


def test_rule_haptic_replaces_the_event_pattern():
    """A mapping rule's own pattern: a pulse code on every motor, fade and sweep borrow release and lock, none is
    silent and auto keeps the event's pattern."""
    assert pattern("start", 100, "pinch", "auto") == pattern("start", 100, "pinch")
    assert pattern("release", 100, None, "LL") == pattern("start", 100, "point")
    assert pattern("start", 70, "fist", "fade") == pattern("release", 70)
    assert pattern("release", 100, None, "sweep") == pattern("lock", 100)
    assert pattern("start", 100, "fist", "none") == []
    assert len(pattern("start", 100, "fist", "SS")) == 2 * SHORT + GAP


def test_haptic_settings_play_as_set():
    """Buzz pulses play their on and off times in steps, the last off is dropped; strength is % of the profile's;
    a fade runs from the start down to its end %; a sweep goes one motor at a time, back reverses it, rounds repeat
    it; default keeps the event's own pattern (lock and unlock sweep opposite ways)."""
    steps = pattern("start", 80, None, {"pattern": "buzz", "pulses": [[300, 100], [100, 200]], "strength": 50})
    assert [d[0] for d in steps] == [40] * (300 // STEP_MS) + [0] * (100 // STEP_MS) + [40] * (100 // STEP_MS)
    fade = [d[0] for d in pattern("release", 100, None, dict(FADE, ms=500, end=0))]
    assert len(fade) == 500 // STEP_MS and fade[0] == 100 and fade[-1] == 0
    back = pattern("lock", 100, None, dict(SWEEP, direction="back", step_ms=2 * STEP_MS, rounds=2))
    assert [d.index(100) for d in back] == [i for i in range(len(OUTPUTS) - 1, -1, -1) for _ in range(2)] * 2
    assert pattern("unlock", 100, None, {"pattern": "default"}) == pattern("unlock", 100)
    assert pattern("start", 100, "pinch", buzz("S")) == pattern("start", 100, "pinch")


def test_control_haptics_layer_patterns_over_the_contact(monkeypatch):
    """The Control tab's combined mode with a fake ESP32: off while the hand touches nothing; touching, every output
    vibrates at the force's duty (half of full force: 50 % on a normal surface) and drops to the surface's off % of it
    between pulses, starting on a pulse; a slippery surface pulses faster than a rough one; a selection pattern plays over it; a lost link is
    connected again."""
    import threading
    import time
    from types import SimpleNamespace
    import tools
    from esp32_link import Esp32Error
    from haptics import HAPTIC_DEFAULTS, SURFACE_DEFAULTS, surface_profile

    rates = [SURFACE_DEFAULTS[s]["hz"] for s in ("slippery", "smooth", "normal", "rough")]
    assert rates == sorted(rates, reverse=True)
    assert surface_profile({}, None) == SURFACE_DEFAULTS["normal"]
    assert surface_profile({"surfaces": {"rough": {"hz": 1, "off": 0, "strength": 50}}}, "rough")["strength"] == 50
    sent, fail = [], []

    class FakeEsp:
        watchdog_fired = False
        link = SimpleNamespace(name="fake")

        def __init__(self, port, wifi):
            if fail:
                fail.pop()
                raise Esp32Error("no reply")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def set_outputs(self, duties):
            sent.append(duties)

    monkeypatch.setattr(tools, "Esp32", FakeEsp)
    recorder = SimpleNamespace(band=SimpleNamespace(channels=list(range(8))), session=None)

    def run(touch, seconds=1.2, cue=False):
        sent.clear()
        h = tools.Haptics(recorder, threading.Lock())
        h.touch = lambda: touch
        h.start(None, None, settings=dict(HAPTIC_DEFAULTS))
        if cue:
            time.sleep(0.2)
            h.cue("start", "fist")
        time.sleep(seconds)
        assert h.state["touch"]["surface"] == (touch[1] if touch[0] > 0 else None)
        h.stop()
        h.join(3)
        firsts = [d[0] for d in sent]
        assert all(len(set(d)) == 1 for d in sent)
        return firsts, sum(a < b for a, b in zip(firsts, firsts[1:]))

    firsts, _ = run((0.0, None))
    assert set(firsts) == {0}
    full = HAPTIC_DEFAULTS["haptic_full_force"]
    firsts, _ = run((full / 2, "normal"))
    assert firsts[0] == 50 and set(firsts) == {25, 50}
    _, fast = run((full / 2, "slippery"))
    _, slow = run((full / 2, "rough"))
    assert fast > slow
    firsts, _ = run((full / 2, "normal"), cue=True)
    assert 100 in firsts
    fail.append(True)
    firsts, _ = run((0.0, None), seconds=1.6)
    assert firsts and set(firsts) == {0}
