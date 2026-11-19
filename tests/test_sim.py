"""3D contact simulation (emg-reading/sim.py): the hand matches the drawn one, and grip force follows effort."""
import re
from pathlib import Path

import numpy as np
import pytest

import ml  # noqa: F401

mujoco = pytest.importorskip("mujoco")
import sim  # noqa: E402
from gestures import GESTURES, _pose  # noqa: E402
from hand_angles import JOINTS  # noqa: E402

HAND_JS = Path(__file__).resolve().parents[1] / "emg-reading" / "web" / "hand.js"
# Thumb opposed, fingers half closed: wraps the can with the hand beside it
WRAP = _pose((10, 60, 20, 20), (50, 60, 30), (50, 60, 30), (50, 60, 30), (50, 60, 30))


def test_geometry_matches_hand_js():
    text = HAND_JS.read_text()
    for name, (base, splay, lengths) in sim.FINGER_GEOMETRY.items():
        m = re.search(rf"{name}: {{ base: \[([-\d, ]+)\], dir: (-?\d+), len: \[([\d, ]+)\]", text)
        assert m, name
        assert [int(v) for v in m.group(1).split(",")] == base
        assert int(m.group(2)) == splay
        assert [int(v) for v in m.group(3).split(",")] == lengths
    m = re.search(r"THUMB_GEOMETRY = { base: \[([-\d, ]+)\], len: \[([\d, ]+)\]", text)
    assert [int(v) for v in m.group(1).split(",")] == sim.THUMB_BASE
    assert [int(v) for v in m.group(2).split(",")] == sim.THUMB_LENGTHS


def test_motors_in_joint_order():
    w = sim.World()
    names = [mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(w.model.nu)]
    assert names == JOINTS


def test_hand_reaches_the_pose_in_free_air():
    w = sim.World()
    w.set_hand([0, 0, 0.3], [1, 0, 0, 0])
    target = np.array(GESTURES["fist"][1])
    for _ in range(50):
        w.step(0.02, target, 1.0)
    reached = np.degrees(w.data.qpos[w._qadr])
    assert np.abs(reached - target).max() < 5
    assert w.grip_force() == 0


def test_scene_and_poses_line_up():
    w = sim.World({"object": "ball"})
    scene, poses = w.scene(), w.poses()
    assert len(scene["geoms"]) == len(poses["pos"]) == len(poses["mat"])
    assert {g["role"] for g in scene["geoms"]} == {"table", "hand", "object"}
    palm = next(g for g in scene["geoms"] if g["type"] == "mesh")
    assert len(palm["vertices"]) > 10 and len(palm["faces"]) > 10


def lift(effort, assist, mass_scale=1.0):
    """Hand beside the can, thumb up, wraps it in 1 s and lifts 10 cm. Returns how high the can ended up (m) and
    the grip force."""
    w = sim.World({"object": "can", "assist": assist, "mass_scale": mass_scale})
    # Columns: the hand's x (thumb), y (fingers), z (palm) in the world
    rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    base = np.array([-0.11, -0.12 - 0.033 - 0.012, 0.066])
    w.set_hand(base, q)
    w.data.qpos[0:7] = [*base, *q]
    mujoco.mj_forward(w.model, w.data)
    rest, grip = np.array(GESTURES["open"][1]), np.array(WRAP)
    for k in range(180):
        t = k * 0.02
        closure = min(max((t - 0.2) / 1.0, 0), 1)
        w.set_hand(base + [0, 0, 0.1 * min(max((t - 1.6) / 1.0, 0), 1)], q)
        w.step(0.02, rest + closure * (grip - rest), effort)
    assert w.blowups == 0
    return w.readout()["object_height"] - sim.OBJECTS["can"][3], w.grip_force()


def test_assist_lifts_and_grip_force_follows_effort():
    h_light, f_light = lift(0.3, True)
    h_full, f_full = lift(1.0, True)
    assert h_light > 0.08 and h_full > 0.08
    assert f_full > 2 * f_light


def test_grip_too_light_for_the_weight_drops_it():
    h, _ = lift(0.3, True, mass_scale=4.0)
    assert h < 0.02
