"""Laptop side of the Quest app: the model it gets, what /ws/quest sends and takes, and the adb reverse link."""
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


def test_scene_has_both_hands_the_pool_and_the_force_rule():
    for left in (True, False):
        m = mujoco.MjModel.from_xml_string(sim.scene_xml(left))
        assert m.nu == 40 and m.nmocap == 2
        names = [m.actuator(i).name for i in range(m.nu)]
        assert names == [f"{p}{j}" for p in sim.HANDS for j in JOINTS]
        assert [m.body(f"obj{i}").mass[0] for i in range(len(sim.pool()))] == pytest.approx(
            [sim.OBJECTS[kind][1] for kind in sim.pool()])

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


def test_quest_touch_drives_the_haptics_while_fresh(monkeypatch):
    c = contact()
    assert c.touch() == (0.0, None)
    c.set_quest_readout({"grip_force": 12.5, "touch_force": 14.0, "touch_surface": sim.SURFACES.index("normal")})
    assert c.touch() == (14.0, "normal")
    # The laptop's test touch wins over the headset (its sim isn't running here, so it reads nothing)
    c.set_test(dict(pressure=0.5, pos=1, slide=False))
    assert c.touch() == (0.0, None) and c.touch_friction() is None
    c.set_test(None)
    assert c.touch() == (14.0, "normal")
    monkeypatch.setattr(tools, "QUEST_FRESH_S", 0.0)
    assert c.touch() == (0.0, None)
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
        elif args[2:] == ("shell", "dumpsys", "battery"):
            out = "Current Battery Service state:\n  AC powered: false\n  status: 3\n  level: 12\n  temperature: 331\n"
        elif args[2] == "shell":
            out = ""
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


def test_headset_gets_what_the_motors_play():
    import app
    off = SimpleNamespace(state={"running": False})
    assert app.headset_haptics_state(off, False) == {"on": False}
    effort_mode = SimpleNamespace(state={"running": True, "duties": [0, 45, 30]})
    assert app.headset_haptics_state(effort_mode, False) == {"on": True, "mode": "effort", "duty": 45.0}
    surface = SimpleNamespace(state={"running": True, "duties": [60] * 7,
                                     "touch": {"force": 8.2, "surface": "rough", "hz": 12.0, "friction": None}})
    state = app.headset_haptics_state(surface, True)
    assert state["surface"] == "rough" and state["friction"] == -1.0 and state["test"]
    assert None not in state.values()


def test_headset_train_follows_its_own_calibration():
    """Train in VR: offered after a completed quick session only, and the result shown is the job the headset
    started (seen running first), not one already finished on the Train tab."""
    import app
    quick = {"quick": True, "completed": True, "practice": False, "session": "2026-11-24_100000_f"}
    assert app.headset_cal_session(dict(quick, completed=False)) is None
    assert app.headset_cal_session(quick) == quick["session"]
    job = SimpleNamespace(state={"running": False, "result": {"saved": "old.pt"}})
    tools_ = SimpleNamespace(network=job)
    cal = {"session": quick["session"], "job": "network", "seen": False, "started": 0.0, "error": ""}
    state = app.headset_train_state(cal, quick, tools_)
    assert state["running"] and not state["saved"]
    job.state = {"running": True}
    assert app.headset_train_state(cal, quick, tools_)["running"]
    job.state = {"running": False, "result": {"saved": "new.pt", "scores": {"balanced": 0.82}}}
    state = app.headset_train_state(cal, quick, tools_)
    assert state["saved"] == "new.pt" and state["status"].startswith("82 % balanced")
    assert None not in state.values()
    # Another session recorded since: nothing to show for it yet
    assert not app.headset_train_state(cal, dict(quick, session="later"), tools_)["saved"]


def test_headset_shows_held_wrist_bends():
    """A held wrist bend has a target for the ghost hands (the rest pose, bent at the wrist), a sign has no bend."""
    import app

    def state(label):
        session = {"paused": False, "hum_pause": None, "index": 0, "count": 3, "next": None, "cue_elapsed": 1.0,
                   "marked_bad": False, "cue": {"label": label, "kind": "nogrip", "text": "X", "seconds": 5}}
        return app.headset_record_state(SimpleNamespace(snapshot=lambda: {"session": session}))

    bend = state("hold_wrist_ext")
    assert bend["has_target"] and bend["wrist_cue"] and bend["wrist"] == -55.0
    sign = state("two_point")
    assert sign["has_target"] and not sign["wrist_cue"] and sign["wrist"] == 0.0


def test_headset_calibrations_are_the_newest_calibrated_networks(tmp_path, monkeypatch):
    """The headset's Train section can pick an older calibration: calibrated networks only, newest saved first."""
    import os
    import app
    for i, name in enumerate(["a_signs_cal_2026-11-20_100000.pt", "base_signs.pt", "a_signs_cal_2026-11-24_141429.pt"]):
        (tmp_path / name).write_bytes(b"")
        os.utime(tmp_path / name, (i, i))
    monkeypatch.setattr(app, "NETWORK_DIR", tmp_path)
    assert app.headset_calibrations() == ["a_signs_cal_2026-11-24_141429.pt", "a_signs_cal_2026-11-20_100000.pt"]


def test_video_splits_into_frames_with_the_codec():
    import quest_video
    sps = b"\x00\x00\x00\x01\x67\x64\x00\x2a\xac"
    pps = b"\x00\x00\x00\x01\x68\xee\x06"
    idr = b"\x00\x00\x00\x01\x65\xb8\x40\x00\x01\x02"
    delta = b"\x00\x00\x01\x41\x9a\x00\x00\x02"
    stream = sps + pps + idr + delta + delta + sps
    splitter = quest_video.Splitter()
    frames = []
    # Fed in small pieces, as adb hands them over, start codes split across reads
    for i in range(0, len(stream), 3):
        frames += splitter.feed(stream[i:i + 3])
    assert [key for key, _, _ in frames] == [True, False, False]
    assert all(codec == "avc1.64002a" for _, codec, _ in frames)
    assert frames[0][2] == sps + pps + idr
    # Each frame decodes on its own: a 4-byte start code in front of every unit
    assert frames[1][2] == b"\x00" + delta


def test_link_reads_the_battery_warns_and_closes_the_app_it_links(monkeypatch):
    monkeypatch.setattr(quest_link, "_answers", lambda port: True)
    link = quest_link.Reverse(8000, "adb")
    adb = FakeAdb([("Q1", "device")])
    monkeypatch.setattr(link, "_run", adb)
    link.check()
    link.check()
    assert link.headset == {"level": 12, "temp_c": 33.1, "charging": False}
    assert link.warning()[0] == "quest_battery"
    # Read every BATTERY_S, not every poll
    assert sum(a[2:] == ("shell", "dumpsys", "battery") for a in adb.calls) == 1
    link.headset = {"level": 12, "temp_c": 46.0, "charging": True}
    assert link.warning()[0] == "quest_hot"
    link.headset["temp_c"] = 30.0
    assert link.warning() is None

    # A test copy the headset isn't linked to leaves the app running when it stops
    copy = quest_link.Reverse(8010, "adb")
    monkeypatch.setattr(copy, "_run", adb)
    copy.check()
    adb.calls.clear()
    copy.stop()
    assert not adb.calls
    link.stop()
    assert adb.calls == [("-s", "Q1", "shell", "am", "force-stop", quest_link.QUEST_APP)]
