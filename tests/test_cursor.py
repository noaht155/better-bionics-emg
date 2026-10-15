"""Abstract control cursor (emg-reading/cursor.py): the fixed map and the target task."""
import math

import ml  # noqa: F401
from cursor import DEFAULTS, MAX_R, Cursor

DT = 0.05


def toward(cursor, deg):
    """Levels that pull the cursor towards deg: the channel nearest that direction at 0.5."""
    i = round((deg - cursor.params["rotation_deg"]) / 45) % 8
    return [0.5 if k == i else 0.0 for k in range(8)]


def test_map_points_each_channel_its_way():
    c = Cursor(dict(DEFAULTS, gain=1.0))
    assert c.map([0.0] * 8) == (0.0, 0.0)
    for i in range(8):
        x, y = c.map([0.5 if k == i else 0.0 for k in range(8)])
        assert math.isclose(math.degrees(math.atan2(y, x)) % 360, 45 * i, abs_tol=1e-6)
        assert math.isclose(math.hypot(x, y), 0.5)


def test_even_activity_stays_in_the_middle():
    x, y = Cursor(dict(DEFAULTS)).map([0.8] * 8)
    assert math.hypot(x, y) < 1e-9


def test_rotation_mirror_and_limit():
    c = Cursor(dict(DEFAULTS, gain=1.0, rotation_deg=90.0))
    x, y = c.map([0.5] + [0.0] * 7)
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 0.5)
    c.params["flip"] = True
    x, y = c.map([0.0, 0.5] + [0.0] * 6)
    assert math.isclose(math.degrees(math.atan2(y, x)) % 360, 45)
    c.params.update(rotation_deg=0.0, gain=10.0)
    assert math.isclose(math.hypot(*c.map([1.0] + [0.0] * 7)), MAX_R)


def test_hit_after_home_and_dwell():
    c = Cursor(dict(DEFAULTS))
    c.targets_on = True
    steps = 0
    while c.phase == "home":
        assert c.step([0.0] * 8, DT) is None
        steps += 1
    assert steps * DT >= DEFAULTS["home_s"]
    angle = c.angles()[c.target]
    trial = None
    while trial is None:
        trial = c.step(toward(c, angle), DT)
    assert trial["hit"] and trial["angle"] == angle % 360 and trial["entries"] == 1
    assert trial["time_s"] >= DEFAULTS["dwell_s"]
    assert len(trial["path"][0]) == 3 + 8
    assert c.phase == "home"


def test_miss_on_timeout_and_every_target_once_per_block():
    c = Cursor(dict(DEFAULTS))
    c.targets_on = True
    seen = []
    for _ in range(int(DEFAULTS["targets"])):
        while c.phase == "home":
            c.step([0.0] * 8, DT)
        seen.append(c.target)
        trial = None
        while trial is None:
            trial = c.step([0.0] * 8, DT)
        assert not trial["hit"] and math.isclose(trial["time_s"], DEFAULTS["timeout_s"], abs_tol=DT)
    assert sorted(seen) == list(range(int(DEFAULTS["targets"])))


def test_leaving_the_target_resets_the_dwell():
    c = Cursor(dict(DEFAULTS, smooth_s=0.0))
    c.targets_on = True
    while c.phase == "home":
        c.step([0.0] * 8, DT)
    inside = toward(c, c.angles()[c.target])
    for _ in range(int(DEFAULTS["dwell_s"] / DT) - 2):
        assert c.step(inside, DT) is None
    c.step([0.0] * 8, DT)
    assert c.dwell == 0.0
    c.step(inside, DT)
    assert c.entries == 2
