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
# Thumb opposed, fingers curled more at the middle joint: wraps the can with the hand beside it. The best of 108
# poses, placements, frictions and stiffnesses tried on 2026-11-19
WRAP = _pose((10, 60, 20, 20), (40, 80, 48), (40, 80, 48), (40, 80, 48), (40, 80, 48))


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
    names = [mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(len(JOINTS))]
    assert names == ["p_" + j for j in JOINTS]


def test_hand_reaches_the_pose_in_free_air():
    w = sim.World()
    w.set_hand([0, 0, 0.3], [1, 0, 0, 0])
    target = np.array(GESTURES["fist"][1])
    for _ in range(50):
        w.step(0.02, target, 1.0)
    assert np.abs(w.joint_angles() - target).max() < 5
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
    w.set_hand(base, q, teleport=True)
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
    assert f_full > 1.5 * f_light


def test_grip_too_light_for_the_weight_drops_it():
    h, _ = lift(0.3, True, mass_scale=3.0)
    assert h < 0.02


def test_pure_physics_holds_from_light_effort_on():
    for effort in (0.2, 0.6, 1.0):
        h, f = lift(effort, False)
        assert h > 0.05, effort
    h, _ = lift(0.05, False)
    assert h < 0.02


def test_heavier_object_needs_more_effort():
    assert lift(0.2, False)[0] > 0.05
    assert lift(0.2, False, mass_scale=3.0)[0] < 0.02
    assert lift(1.0, False, mass_scale=3.0)[0] > 0.05


def test_grip_force_follows_effort_without_the_assist():
    forces = [lift(e, False)[1] for e in (0.2, 0.6, 1.0)]
    assert forces[0] < forces[1] < forces[2]


# Hand frames, columns the hand's x (thumb), y (fingers), z (palm) in the world: from above, fingers away from the
# viewer, and from above tilted back 40 degrees, for a small object on the table
TOP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
TILT = np.radians(-40)
TOP_TILTED = TOP @ np.array([[1, 0, 0], [0, np.cos(TILT), -np.sin(TILT)], [0, np.sin(TILT), np.cos(TILT)]])
SIDE = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)


def grasp_centre(pose, fingers, palm):
    """Mean of the fingertips (and the palm) in the hand frame with the pose closed in free air."""
    m = mujoco.MjModel.from_xml_string(sim.ghost_xml(False))
    d = mujoco.MjData(m)
    d.qpos[[m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]] = np.radians(pose)
    mujoco.mj_kinematics(m, d)
    points = [np.array([-0.005, 0.06, 0.012])] if palm else []
    for f in fingers:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{f}_dist")
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
        joint = d.xpos[m.geom_bodyid[g]]
        points.append(max((d.geom_xpos[g] + axis, d.geom_xpos[g] - axis),
                          key=lambda end: np.linalg.norm(end - joint)))
    return np.mean(points, 0)


def grasp(obj, pose, rot, fingers, palm, effort, offset):
    """The hand comes down on the object with the grip's contact centre on it, closes and lifts 10 cm. Returns how
    high the object ended up (m)."""
    w = sim.World({"object": obj})
    half = sim.OBJECTS[obj][3]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    base = np.array([0, -0.12, half]) - rot @ grasp_centre(pose, fingers, palm) + rot @ np.array(offset)
    back = -rot[:, 2] * 0.02
    w.set_hand(base + back, q, teleport=True)
    rest, grip = np.array(GESTURES["open"][1]), np.array(pose)
    for k in range(200):
        t = k * 0.02
        approach = min(t / 0.3, 1)
        closure = min(max((t - 0.4) / 1.0, 0), 1)
        w.set_hand(base + back * (1 - approach) + [0, 0, 0.1 * min(max((t - 1.8) / 1.0, 0), 1)], q)
        w.step(0.02, rest + closure * (grip - rest), effort)
    assert w.blowups == 0
    return w.readout()["object_height"] - half


@pytest.mark.parametrize("obj, grip, rot, fingers, palm, offset", [
    ("ball", WRAP, SIDE, sim.FINGERS, True, (0, 0.01, 0)),
    ("block", GESTURES["tripod"][1], TOP, ["thumb", "index", "middle"], False, (0, 0, 0.01)),
    ("pen", GESTURES["pinch"][1], TOP, ["thumb", "index"], False, (0, -0.01, 0)),
    ("small cube", GESTURES["tripod"][1], TOP_TILTED, ["thumb", "index", "middle"], False, (0, 0, 0)),
])
def test_other_objects_hold_from_medium_effort(obj, grip, rot, fingers, palm, offset):
    for effort in (0.6, 1.0):
        assert grasp(obj, grip, rot, fingers, palm, effort, offset) > 0.05, (obj, effort)


def test_mass_is_capped():
    w = sim.World({"mass_scale": 10.0})
    assert w.weight() == pytest.approx(sim.OBJECTS["can"][1] * sim.MAX_MASS_SCALE * sim.GRAVITY)


def test_contact_runs_on_the_control_layer_and_gives_the_force_to_the_haptics():
    import time
    from types import SimpleNamespace
    import tools
    predictor = SimpleNamespace(state={"running": True, "control": True})
    control = SimpleNamespace(state={"angles": GESTURES["fist"][1], "effort": 0.5, "locked": False, "grip": "fist"})
    contact = tools.Contact(predictor, control)
    assert contact.grip_force() == 0.0
    contact.start()
    try:
        time.sleep(0.5)
        frame = contact.frame()
        assert frame["running"] and frame["control"]["grip"] == "fist"
        assert len(frame["pos"]) == len(contact.scene()["geoms"])
        reached = contact.world.joint_angles()
        assert reached[JOINTS.index("index_pip")] > 50
        version = contact.version
        contact.set_params({"object": "ball"})
        assert contact.version == version + 1 and contact.scene()["object"] == "ball"
    finally:
        contact.stop()
        contact.join(2)
    assert contact.grip_force() == 0.0
