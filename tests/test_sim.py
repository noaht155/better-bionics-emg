"""3D contact simulation (emg-reading/sim.py): the hand matches the drawn one, and grip force follows effort."""
import re
from pathlib import Path

import numpy as np
import pytest

import ml  # noqa: F401

mujoco = pytest.importorskip("mujoco")
import sim  # noqa: E402
from gestures import GESTURES  # noqa: E402
from hand_angles import JOINTS  # noqa: E402

HAND_JS = Path(__file__).resolve().parents[1] / "emg-reading" / "web" / "hand.js"
WRAP, SIDE = sim.WRAP, sim.SIDE


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
    # The palm and the thumb's base stop the pinky and index a little short (6 and 9 degrees)
    assert np.abs(w.joint_angles() - target).max() < 12
    assert w.grip_force() == 0


def test_scene_and_poses_line_up():
    w = sim.World({"object": "ball"})
    scene, poses = w.scene(), w.poses()
    assert len(scene["geoms"]) == len(poses["pos"]) == len(poses["mat"])
    assert {g["role"] for g in scene["geoms"]} == {"table", "rail", "hand", "object"}
    assert [g["surface"] for g in scene["geoms"] if g["role"] == "rail" and g["type"] == "cylinder"] == list(sim.SURFACES)
    assert next(g for g in scene["geoms"] if g["role"] == "object")["surface"] == sim.SURFACE_OF["ball"]
    palm = next(g for g in scene["geoms"] if g["type"] == "mesh")
    assert len(palm["vertices"]) > 10 and len(palm["faces"]) > 10


def test_sliding_along_the_rail_reads_each_surface():
    """The hand wrapped round the fixed rail from above and slid along it reads its sections' surfaces in order,
    slippery to rough, without blowing up; Ready on the rail closes on its slippery end."""
    w = sim.World()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, sim.TOP.flatten())
    centre = sim.grasp_centre(WRAP, sim.FINGERS, True)
    at = lambda x: np.array([x, sim.RAIL_Y, sim.RAIL_HEIGHT]) - sim.TOP @ centre
    w.set_hand(at(-0.15) + [0, 0, 0.03], q, teleport=True)
    for k in range(80):
        w.set_hand(at(-0.15), q)
        w.step(0.02, GESTURES["open"][1] if k < 30 else WRAP, 0.3)
    seen = []
    for i in range(150):
        w.set_hand(at(-0.15 + 0.3 * i / 149), q)
        w.step(0.02, WRAP, 0.3)
        force, surface = w.touch()
        if force > 1 and (not seen or seen[-1] != surface):
            seen.append(surface)
    assert seen == list(sim.SURFACES) and w.blowups == 0
    w.place_hand("rail")
    for k in range(90):
        w.step(0.02, GESTURES["open"][1] if k < 30 else WRAP, 0.4)
    force, surface = w.touch()
    assert force > 1 and surface == "slippery"


def lift(effort, assist, mass_scale=1.0):
    """Hand beside the can, thumb up, wraps it in 1 s and lifts 10 cm. Returns how high the can ended up (m) and
    the grip force."""
    w = lift_world(effort, assist, mass_scale)
    return w.readout()["object_height"] - sim.OBJECTS["can"][3], w.grip_force()


def lift_world(effort, assist, mass_scale=1.0):
    """lift()'s world once it's done."""
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
    return w


def test_assist_lifts_and_grip_force_follows_effort():
    h_light, f_light = lift(0.3, True)
    h_full, f_full = lift(1.0, True)
    assert h_light > 0.08 and h_full > 0.08
    assert f_full > 1.5 * f_light


def test_touch_reads_the_surface_pressed_on():
    """Pressing the open hand down on the desk reads force on the desk's surface; the lifted can, held clear of the
    desk, reads the can's."""
    w = sim.World()
    w.set_hand([0.2, -0.3, 0.05], sim.HAND_QUAT, teleport=True)
    assert w.touch() == (0.0, None)
    for _ in range(25):
        w.set_hand([0.2, -0.3, -0.02], sim.HAND_QUAT)
        w.step(0.02, GESTURES["open"][1], 0.5)
    force, surface = w.touch()
    assert force > 1 and surface == sim.SURFACE_OF["desk"]
    w = lift_world(1.0, False)
    force, surface = w.touch()
    assert force > 1 and surface == sim.SURFACE_OF["can"]
    assert set(sim.OBJECTS) <= set(sim.SURFACE_OF) and set(sim.SURFACE_OF.values()) <= set(sim.SURFACES)


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


def grasp(obj, effort):
    """The hand starts ready on the object (place_hand), comes down with the grip's contact centre on it, closes and
    lifts 10 cm. Returns how high the object ended up (m)."""
    w = sim.World({"object": obj})
    half = sim.OBJECTS[obj][3]
    w.place_hand("object")
    assert w.touch() == (0.0, None)
    for _ in range(30):
        w.step(0.02, GESTURES["open"][1], 0.0)
    assert w.touch()[0] < 0.2
    base, q, back = w.grasp_target()
    rest, grip = np.array(GESTURES["open"][1]), np.array(sim.GRASPS[obj][1])
    for k in range(200):
        t = k * 0.02
        approach = min(t / 0.3, 1)
        closure = min(max((t - 0.4) / 1.0, 0), 1)
        w.set_hand(base + back * (1 - approach) + [0, 0, 0.1 * min(max((t - 1.8) / 1.0, 0), 1)], q)
        w.step(0.02, rest + closure * (grip - rest), effort)
    assert w.blowups == 0
    return w.readout()["object_height"] - half


@pytest.mark.parametrize("obj", ["ball", "block", "pen", "small cube"])
def test_other_objects_hold_from_medium_effort(obj):
    for effort in (0.6, 1.0):
        assert grasp(obj, effort) > 0.05, (obj, effort)


def test_hand_goes_back_to_its_start():
    w = sim.World()
    w.place_hand("object")
    w.place_hand("start")
    pos, quat = w.hand_pose()
    assert pos == pytest.approx(sim.HAND_START) and quat == pytest.approx(sim.HAND_QUAT)


def test_mass_is_capped():
    w = sim.World({"mass_scale": 10.0})
    assert w.weight() == pytest.approx(sim.OBJECTS["can"][1] * sim.MAX_MASS_SCALE * sim.GRAVITY)


def test_contact_runs_on_the_control_layer_and_gives_the_touch_to_the_haptics():
    import time
    from types import SimpleNamespace
    import tools
    predictor = SimpleNamespace(state={"running": True, "control": True})
    control = SimpleNamespace(state={"angles": GESTURES["fist"][1], "effort": 0.5, "locked": False, "grip": "fist"})
    contact = tools.Contact(predictor, control)
    assert contact.touch() == (0.0, None)
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
    assert contact.touch() == (0.0, None)


def test_test_touch_grips_the_rail_section_with_the_pressure():
    """3D contact's test touch: the hand comes down on the rail's section of the surface and grips it harder the
    higher the pressure; a new surface slides it along to that section; off hands it back."""
    import time
    from types import SimpleNamespace
    import tools
    predictor = SimpleNamespace(state={"running": False})
    control = SimpleNamespace(state={})
    contact = tools.Contact(predictor, control)
    contact.start()
    try:
        forces = []
        for pressure in (0.2, 1.0):
            contact.set_test(None)
            contact.set_test((pressure, "smooth"))
            time.sleep(1.5)
            force, surface = contact.touch()
            assert surface == "smooth"
            forces.append(force)
        assert forces[1] > 1.5 * forces[0]
        contact.set_test((0.5, "rough"))
        time.sleep(3.0)
        assert contact.touch()[1] == "rough" and contact.world.blowups == 0
        assert contact.frame()["test"]
        contact.set_test(None)
        assert not contact.frame()["test"]
    finally:
        contact.stop()
        contact.join(2)
