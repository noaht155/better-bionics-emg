"""The delay test's analysis (delay.py) on made-up streams with known delays."""
import numpy as np

import ml  # noqa: F401
import delay

RATE = 500
ROWS = {"emg": list(range(8)), "timestamp": 8}


def test_round_trip_found_for_each_switch():
    t = 1000 + np.arange(20 * RATE) / RATE
    data = np.zeros((9, t.size))
    data[8] = t
    sent = [1002.0, 1004.3, 1007.1, 1011.6]
    eeg = True
    offsets = 30_000 + 5_000 * np.arange(8)[:, None]
    level = np.ones(t.size, dtype=bool)
    for s in sent:
        level[t >= s + 0.017] = not eeg
        eeg = not eeg
    data[:8] = np.where(level, offsets, 100 * np.sign(np.sin(2 * np.pi * 2 * t)))
    trips = delay.round_trips(data, ROWS, sent)
    assert len(trips) == len(sent)
    assert np.all(np.abs(trips - 0.017) <= 1 / RATE)


def test_round_trip_ignores_emg_and_lost_switches():
    t = 1000 + np.arange(5 * RATE) / RATE
    data = np.zeros((9, t.size))
    data[8] = t
    data[:8] = 30_000 + np.random.default_rng(0).normal(0, 500, (8, t.size))
    assert delay.round_trips(data, ROWS, [1001.0, 1003.0]).size == 0


def test_screen_delay_from_frames():
    rng = np.random.default_rng(1)
    frame = np.arange(40 * 30)
    taken = 2000 + frame / 30
    camera_s = 0.06
    switches, when, lit = [], 2001.0, False
    while when < 2038:
        lit = not lit
        switches.append((when, lit))
        when += rng.uniform(0.4, 0.8)
    times = np.array([s[0] for s in switches])
    state = np.array([s[1] for s in switches])
    # A frame shows the screen as it was when it was taken, and arrives camera_s later, sometimes a frame late
    shown = np.array([state[times <= x][-1] if (times <= x).any() else False for x in taken])
    arrived = taken + camera_s + np.where(rng.random(frame.size) < 0.3, 1 / 30, 0)
    log = np.column_stack([arrived, frame, np.where(shown, 200, 15) + rng.normal(0, 3, frame.size)])
    on, off = delay.screen_delays(log, switches)
    both = np.concatenate([on, off])
    assert len(both) >= len(switches) - 1
    # The first frame past the change is up to one frame after it
    assert camera_s - 0.005 < np.median(both) < camera_s + 1 / 30 + 0.005


def test_screen_delay_refuses_unseen_flashes():
    frame = np.arange(300)
    log = np.column_stack([2000 + frame / 30, frame, 100 + np.random.default_rng(2).normal(0, 2, frame.size)])
    try:
        delay.screen_delays(log, [(2001.0 + i, i % 2 == 0) for i in range(8)])
    except ValueError as e:
        assert "barely change" in str(e)
    else:
        raise AssertionError("no error for a camera that doesn't see the flashes")


def test_camera_delay_between_two_directions():
    # Light and dark far apart, like the MacBook: the result is their mean, not a point in the gap
    on = np.full(30, 0.088) + np.linspace(-0.004, 0.004, 30)
    off = np.full(30, 0.015) + np.linspace(-0.004, 0.004, 30)
    assert abs(delay.camera_delay(on, off) - 0.0515) < 1e-6
    assert abs(delay.camera_delay(on, np.array([])) - 0.088) < 1e-6
