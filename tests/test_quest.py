"""Laptop side of the Quest app: the model it gets, what /ws/quest sends and takes, and the adb reverse link."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ml  # noqa: F401

mujoco = pytest.importorskip("mujoco")
import quest_link  # noqa: E402
import sim  # noqa: E402
import tools  # noqa: E402
from gestures import GESTURES  # noqa: E402
from hand_angles import JOINTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quest-app" / "native"))
import export_model  # noqa: E402


def test_scene_has_both_hands_the_pool_and_the_force_rule():
    for left in (True, False):
        m = mujoco.MjModel.from_xml_string(export_model.scene_xml(left))
        assert m.nu == 40 and m.nmocap == 2
        names = [m.actuator(i).name for i in range(m.nu)]
        assert names == [f"{p}{j}" for p in export_model.HANDS for j in JOINTS]
        for i, kind in enumerate(export_model.pool()):
            assert m.geom(f"obj{i}").size == pytest.approx(mujoco.MjModel.from_xml_string(sim.model_xml(kind))
                                                             .geom("object").size)

        def numeric(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
            return m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i] + m.numeric_size[i]]

        assert numeric("kp")[0] == sim.KP and numeric("kp_press")[0] == sim.KP_PRESS
        assert numeric("arm") == pytest.approx(sim._arms())
        # The prosthetic hand is the left one when asked: its thumb is on the other side of the palm (+x in the hand)
        thumb_x = m.body("p_thumb_meta").pos[0]
        assert (thumb_x < 0) == left


def test_left_hand_mirrors_the_right():
    xml = (f'<mujoco><default><default class="hand"><geom/></default></default><asset>{sim.palm_mesh_xml("r_")}'
           f'{sim.palm_mesh_xml("l_", True)}</asset><worldbody><body>{sim.hand_xml("r_")}</body>'
           f'<body pos="1 0 0">{sim.hand_xml("l_", True)}</body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for j in JOINTS:
        lo, hi = m.joint("r_" + j).range
        d.qpos[m.joint("r_" + j).qposadr[0]] = d.qpos[m.joint("l_" + j).qposadr[0]] = rng.uniform(lo, hi) * 0.6
    mujoco.mj_kinematics(m, d)
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith("r_"):
            right, left = d.geom_xpos[g], d.geom_xpos[m.geom("l_" + name[2:]).id] - [1, 0, 0]
            assert right * [-1, 1, 1] == pytest.approx(left, abs=1e-9)


def contact():
    predictor = SimpleNamespace(state={"running": True, "control": True})
    control = SimpleNamespace(state={"angles": GESTURES["fist"][1], "effort": 0.5, "locked": True, "grip": "fist"})
    return tools.Contact(predictor, control)


def test_quest_gets_the_control_output_and_shared_settings():
    c = contact()
    frame = c.quest_frame()
    assert frame["angles"] == [float(a) for a in GESTURES["fist"][1]] and frame["effort"] == 0.5
    assert frame["locked"] and frame["driven"] and frame["params"] == sim.SIM_DEFAULTS
    c.set_params({"max_force": 20.0})
    after = c.quest_frame()
    assert after["params_seq"] == frame["params_seq"] + 1 and after["params"]["max_force"] == 20.0
    c.predictor.state["running"] = False
    assert not c.quest_frame()["driven"]


def test_quest_grip_force_drives_the_haptics_while_fresh(monkeypatch):
    c = contact()
    assert c.grip_force() == 0.0
    c.set_quest_readout({"grip_force": 12.5})
    assert c.grip_force() == 12.5
    monkeypatch.setattr(tools, "QUEST_FRESH_S", 0.0)
    assert c.grip_force() == 0.0
    monkeypatch.undo()
    c.quest_gone()
    assert c.quest_readout() is None


class FakeAdb:
    def __init__(self, devices, mapped=None):
        self.devices, self.mapped, self.calls = devices, mapped, []

    def __call__(self, *args):
        self.calls.append(args)
        if args == ("devices",):
            out = "List of devices attached\n" + "".join(f"{s}\t{state}\n" for s, state in self.devices)
        elif args[2:] == ("reverse", "--list"):
            out = f"UsbFfs tcp:8000 tcp:{self.mapped}\n" if self.mapped else ""
        else:
            self.mapped = int(args[-1].removeprefix("tcp:"))
            out = ""
        return SimpleNamespace(stdout=out, returncode=0)


def test_link_maps_the_quest_once_and_leaves_another_live_copy(monkeypatch):
    link = quest_link.Reverse(8000, "adb")
    adb = FakeAdb([("Q1", "device"), ("Q2", "unauthorized")])
    monkeypatch.setattr(link, "_run", adb)
    link.check()
    assert adb.mapped == 8000
    assert not any(a[:1] == ("-s",) and a[1] == "Q2" for a in adb.calls)
    adb.calls.clear()
    link.check()
    assert not any("tcp:8000" in a and len(a) == 5 for a in adb.calls)

    # A test copy on 8010 doesn't take it while 8000 answers, and does once it's gone
    copy = quest_link.Reverse(8010, "adb")
    monkeypatch.setattr(copy, "_run", adb)
    monkeypatch.setattr(quest_link, "_answers", lambda port: True)
    copy.check()
    assert adb.mapped == 8000
    monkeypatch.setattr(quest_link, "_answers", lambda port: False)
    copy.check()
    assert adb.mapped == 8010
