"""Laptop side of the Quest app: the model it gets, what /ws/quest sends and takes, and the adb reverse link."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml  # noqa: F401

mujoco = pytest.importorskip("mujoco")
import quest_link  # noqa: E402
import sim  # noqa: E402
import tools  # noqa: E402
from gestures import GESTURES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quest-app" / "native"))
from export_model import quest_xml  # noqa: E402


def test_exported_model_carries_the_force_rule():
    for obj in sim.OBJECTS:
        m = mujoco.MjModel.from_xml_string(quest_xml(obj))
        world = sim.World({"object": obj})
        assert m.ngeom == world.model.ngeom and m.nu == 20

        def numeric(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
            return m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i] + m.numeric_size[i]]

        assert numeric("kp")[0] == sim.KP and numeric("kp_press")[0] == sim.KP_PRESS
        assert numeric("object_half")[0] == sim.OBJECTS[obj][3]
        assert numeric("arm") == pytest.approx(sim._arms())


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
