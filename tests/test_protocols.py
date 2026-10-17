"""Session plans of the recording protocols, and protocol C, D, E and F sessions going through the training code."""
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

import ml  # noqa: F401
import gestures
from gestures import (E_NOT_SIGNS, E_SIGNS, F_SIGNS, GESTURE_SET, GESTURES, LIBRARY, NO_GRIP, plan_for,
                      protocol_b_plan, protocol_c_plan, protocol_d_plan, protocol_e_plan, protocol_f_plan, session_plan)
from hand_angles import JOINTS
from processing import FILTER_VERSION
from ml import dataset, gesture_model
from recorder import Recorder
from session import CAMERA_COLUMNS, load_session, segments

GRIPS = [g for g in GESTURE_SET if g != "rest"]


def test_c_plan_holds_the_selection_once_per_round():
    picked = ["fist", "trigger", "drum", "key"]
    plan = protocol_c_plan(picked, ["table", "raised"], rounds=3, seed=1)
    custom = [c for c in plan if c["kind"] == "custom"]
    assert Counter(c["label"] for c in custom) == {m: 6 for m in picked}
    for posture in ("table", "raised"):
        own = [c["label"] for c in custom if c["posture"] == posture]
        for r in range(3):
            assert sorted(own[4 * r:4 * r + 4]) == sorted(picked)
    # Calibration: every grip twice per posture, as plain holds
    holds = Counter((c["posture"], c["label"]) for c in plan if c["kind"] == "hold" and c["label"] != "rest")
    assert holds == {(p, g): 2 for p in ("table", "raised") for g in GRIPS}
    assert {c["kind"] for c in plan} == {"break", "hold", "custom"}
    # Every movement has rest after it
    for a, b in zip(plan, plan[1:]):
        if a["kind"] == "custom":
            assert (b["kind"], b["label"]) == ("hold", "rest")


def test_c_plan_rounds_are_shuffled_and_seeded():
    picked = list(LIBRARY)[:8]
    a = protocol_c_plan(picked, rounds=4, seed=3)
    assert a == protocol_c_plan(picked, rounds=4, seed=3)
    orders = {tuple(c["label"] for c in a if c["kind"] == "custom")[8 * r:8 * r + 8] for r in range(4)}
    assert len(orders) > 1


def test_c_plan_refuses_an_empty_or_unknown_selection():
    with pytest.raises(ValueError):
        protocol_c_plan([])
    with pytest.raises(ValueError):
        protocol_c_plan(["fist", "not_a_movement"])


def test_plan_for_c_uses_its_own_fields():
    settings = {"protocol": "C", "postures": ["table"], "reps": 3, "hold_s": 4, "rest_s": 2, "rounds": 6,
                "move_s": 5, "moves": ["pinch", "snap"], "c_rounds": 2, "c_move_s": 7}
    plan = plan_for(settings, seed=0)
    custom = [c for c in plan if c["kind"] == "custom"]
    assert len(custom) == 4 and all(c["seconds"] == 7 for c in custom)
    assert plan == protocol_c_plan(["pinch", "snap"], ["table"], 2, 7, 2, 4, 0)


def test_a_and_b_plans_unchanged():
    """Protocols A and B against their plans before protocol C was added (seed 5), stored as a hash so the test
    doesn't need the old code."""
    a = session_plan(["table", "forward", "raised"], 3, 4.0, 3.0, 0, 5)
    b = protocol_b_plan(["table"], seed=5)
    assert plan_for({"postures": ["table", "forward", "raised"], "reps": 3, "hold_s": 4.0, "rest_s": 3.0,
                     "moves": ["fist"], "c_rounds": 2}, 5) == a
    assert plan_for({"protocol": "B", "postures": ["table"], "reps": 3, "hold_s": 4.0, "rest_s": 3.0, "rounds": 6,
                     "move_s": 5.0, "moves": ["fist"]}, 5) == b
    assert _digest(a) == A_DIGEST
    assert _digest(b) == B_DIGEST


def _digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]


A_DIGEST = "2ac60ff5c24b0391"
B_DIGEST = "140a47d1d85c8b6a"


def test_library_entries_are_complete():
    for key, m in LIBRARY.items():
        assert m["category"] in gestures.CATEGORIES, key
        assert m["text"].count("\n") == 1 and m["name"] and m["sources"], key
        assert len(m["angles"]) == len(JOINTS), key
        # Reused keys keep the angles they already had
        assert GESTURES[key][1] == m["angles"], key
        if "keys" in m:
            times = [t for t, _ in m["keys"]]
            assert times[0] == 0 and times[-1] == 1 and times == sorted(times), key
            # Start relaxed; the end is held to the end of the cue and relaxes in the rest after
            assert m["keys"][0][1] == GESTURES["rest"][1], key


def _fake_session(folder, plan, settings):
    """A completed session of this plan with noise for EMG and the relaxed hand on camera, in the recorder's format."""
    rate = 500
    rows = {"emg": list(range(8)), "accel": [8, 9, 10], "package": 11, "timestamp": 12, "num_rows": 13}
    events, sample, t0 = [], 0, 1.7e9
    for i, cue in enumerate(plan):
        events.append({"t": t0 + sample / rate, "sample": sample, "kind": "cue", "index": i, "cue": cue})
        sample += int((cue["seconds"] or 1) * rate)
    events.append({"t": t0 + sample / rate, "sample": sample, "kind": "stop"})
    rng = np.random.default_rng(0)
    board = np.zeros((13, sample))
    board[:8] = rng.normal(0, 5, (8, sample))
    board[11] = np.arange(sample)
    board[12] = t0 + np.arange(sample) / rate
    n_frames = int(sample / rate * 30)
    cam = np.zeros((n_frames, len(CAMERA_COLUMNS)))
    cam[:, 0] = t0 + np.arange(n_frames) / 30
    cam[:, 1] = np.arange(n_frames)
    cam[:, 2] = 1
    cam[:, -len(JOINTS):] = GESTURES["rest"][1]
    folder.mkdir()
    meta = {"settings": {"postures": ["table"], "band_arm": "left", "port_facing": "shoulder", **settings},
            "plan": plan, "rate": rate, "rows": rows, "channels": list(range(8)), "camera_columns": CAMERA_COLUMNS,
            "completed": True, "subject": "test"}
    (folder / "session.json").write_text(json.dumps(meta))
    (folder / "board.f64").write_bytes(np.ascontiguousarray(board.T).tobytes())
    (folder / "camera.f64").write_bytes(cam.tobytes())
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_c_session_trains_without_leaking_into_protocol_a(tmp_path):
    """Library cues named like grips (fist, pinch) get no grip label in the network windows or the LDA windows;
    only the calibration holds do."""
    folder = tmp_path / "2026-10-14_120000_test"
    plan = protocol_c_plan(["fist", "pinch", "trigger"], rounds=1, move_s=3, rest_s=1.5, hold_s=2.5, seed=0)
    _fake_session(folder, plan, {"protocol": "C", "moves": ["fist", "pinch", "trigger"]})
    assert dataset.good_sessions(tmp_path) == [folder]
    d = dataset.build(folder)
    custom = d["kind"] == "custom"
    assert custom.any() and (d["grip"][custom] == "").all()
    assert set(d["grip"][d["grip"] != ""]) == set(GESTURE_SET)
    assert set(d["kind"]) <= {"hold", "custom", "break"}

    data = load_session(folder)
    _, ends, labels, _ = gesture_model.session_windows(data, 500, {"log": True, "accel": False})
    for seg in segments(data["events"]):
        if seg["kind"] == "custom":
            inside = (ends > seg["sample0"]) & (ends <= seg["sample1"])
            assert all(lab is None for lab in labels[inside])


def test_d_plan_holds_every_activity_once_per_round():
    plan = protocol_d_plan(["forward", "table"], rounds=2, activity_s=20, seed=1)
    nogrip = [c for c in plan if c["kind"] == "nogrip"]
    assert Counter(c["label"] for c in nogrip) == {a: 2 for a in NO_GRIP}
    assert all(c["seconds"] == 20 and c["posture"] == "forward" for c in nogrip)
    # Calibration: every grip twice, as plain holds, in the first posture only
    holds = Counter((c["posture"], c["label"]) for c in plan if c["kind"] == "hold" and c["label"] != "rest")
    assert holds == {("forward", g): 2 for g in GRIPS}
    for a, b in zip(plan, plan[1:]):
        if a["kind"] == "nogrip":
            assert (b["kind"], b["label"]) == ("hold", "rest")
    settings = {"protocol": "D", "postures": ["table"], "reps": 3, "hold_s": 4, "rest_s": 2, "d_rounds": 1,
                "d_activity_s": 30}
    assert plan_for(settings, seed=0) == protocol_d_plan(["table"], 1, 30, 2, 4, 0)


def test_d_session_gets_no_grip_or_rest_labels(tmp_path):
    """Activities that aren't grips must not train any class, rest included: the wrist and arm move there."""
    folder = tmp_path / "2026-10-14_130000_test"
    plan = protocol_d_plan(rounds=1, activity_s=3, rest_s=1.5, hold_s=2.5, seed=0)
    _fake_session(folder, plan, {"protocol": "D"})
    d = dataset.build(folder)
    nogrip = d["kind"] == "nogrip"
    assert nogrip.any() and (d["grip"][nogrip] == "").all()
    assert set(d["grip"][d["grip"] != ""]) == set(GESTURE_SET)

    data = load_session(folder)
    _, ends, labels, _ = gesture_model.session_windows(data, 500, {"log": True, "accel": False})
    for seg in segments(data["events"]):
        if seg["kind"] == "nogrip":
            inside = (ends > seg["sample0"]) & (ends <= seg["sample1"])
            assert inside.any() and all(lab is None for lab in labels[inside])


def test_e_plan_holds_every_sign_once_per_round():
    plan = protocol_e_plan(["table", "forward"], rounds=3, sign_s=5, rest_s=2, seed=1)
    items = [c for c in plan if c["kind"] in ("custom", "nogrip")]
    every = set(E_SIGNS) | set(E_NOT_SIGNS)
    # 3 rounds in the first posture, one fewer in the next
    assert Counter((c["posture"], c["label"]) for c in items) == {
        **{("table", k): 3 for k in every}, **{("forward", k): 2 for k in every}}
    n = len(every)
    table = [c["label"] for c in items if c["posture"] == "table"]
    for r in range(3):
        assert set(table[n * r:n * r + n]) == every
    assert all(c["kind"] == "custom" for c in items if c["label"] in E_SIGNS)
    assert all(c["kind"] == "nogrip" for c in items if c["label"] in E_NOT_SIGNS)
    assert all(c["seconds"] == 5 for c in items)
    assert {c["kind"] for c in plan} == {"break", "hold", "custom", "nogrip"}
    for a, b in zip(plan, plan[1:]):
        if a["kind"] in ("custom", "nogrip"):
            assert (b["kind"], b["label"]) == ("hold", "rest")
    assert all(s in LIBRARY for s in E_SIGNS)
    settings = {"protocol": "E", "postures": ["table"], "reps": 3, "hold_s": 4, "rest_s": 2, "e_rounds": 2,
                "e_sign_s": 7}
    assert plan_for(settings, seed=0) == protocol_e_plan(["table"], 2, 7, 2, 0)


def test_e_session_stays_out_of_grip_training(tmp_path):
    """A screening session has no grip holds, so it isn't a good session for the grip and network training, and its
    signs never get a grip label, even fist, open, point and pinch."""
    folder = tmp_path / "2026-10-16_120000_test"
    plan = protocol_e_plan(["table"], rounds=1, sign_s=3, rest_s=1.5, seed=0)
    _fake_session(folder, plan, {"protocol": "E"})
    assert dataset.good_sessions(tmp_path) == []
    data = load_session(folder)
    _, ends, labels, _ = gesture_model.session_windows(data, 500, {"log": True, "accel": False})
    for seg in segments(data["events"]):
        if seg["kind"] in ("custom", "nogrip"):
            inside = (ends > seg["sample0"]) & (ends <= seg["sample1"])
            assert inside.any() and all(lab is None for lab in labels[inside])


def test_f_plan_holds_the_sign_set_once_per_round_in_every_posture():
    plan = protocol_f_plan(["table", "forward", "raised"], rounds=2, sign_s=5, rest_s=2, seed=1)
    items = [c for c in plan if c["kind"] in ("custom", "nogrip")]
    every = set(F_SIGNS) | set(E_NOT_SIGNS)
    assert Counter((c["posture"], c["label"]) for c in items) == {(p, k): 2 for p in ("table", "forward", "raised")
                                                                    for k in every}
    assert all(c["kind"] == "custom" for c in items if c["label"] in F_SIGNS)
    assert all(c["kind"] == "nogrip" for c in items if c["label"] in E_NOT_SIGNS)
    for a, b in zip(plan, plan[1:]):
        if a["kind"] in ("custom", "nogrip"):
            assert (b["kind"], b["label"]) == ("hold", "rest")
    settings = {"protocol": "F", "postures": ["table"], "reps": 3, "hold_s": 4, "rest_s": 2, "f_rounds": 3,
                "f_sign_s": 7}
    assert plan_for(settings, seed=0) == protocol_f_plan(["table"], 3, 7, 2, 0)


def test_f_session_stays_out_of_grip_training(tmp_path):
    """The sign set's open hand is a sign here, not the protocol A open grip, and the session has no grip holds."""
    folder = tmp_path / "2026-10-16_130000_test"
    _fake_session(folder, protocol_f_plan(["table"], rounds=1, sign_s=3, rest_s=1.5, seed=0), {"protocol": "F"})
    assert dataset.good_sessions(tmp_path) == []
    data = load_session(folder)
    _, ends, labels, _ = gesture_model.session_windows(data, 500, {"log": True, "accel": False})
    for seg in segments(data["events"]):
        if seg["kind"] in ("custom", "nogrip"):
            inside = (ends > seg["sample0"]) & (ends <= seg["sample1"])
            assert inside.any() and all(lab is None for lab in labels[inside])


def test_hand_warning_only_where_the_camera_labels():
    """The hand-out-of-view warning applies to cues the camera labels and the setup break before them, not between
    sessions, not during protocol D's activities or the rest right after one."""
    r = Recorder.__new__(Recorder)
    r.session, r.plan, r.index = None, [], -1
    assert not r._camera_needed()
    r.session = object()
    r.plan = protocol_d_plan(rounds=1, activity_s=20, seed=0)
    needed = []
    for r.index in range(len(r.plan)):
        needed.append(r._camera_needed())
    kinds = [c["kind"] for c in r.plan]
    assert needed[0] and kinds[0] == "break"
    for i, cue in enumerate(r.plan):
        if cue["kind"] == "nogrip" or (cue["label"] == "rest" and kinds[i - 1] == "nogrip"):
            assert not needed[i]
        elif cue["kind"] == "hold":
            assert needed[i]


def test_cue_cut_short_by_a_pause_is_bad():
    """The attempt a pause interrupts is left out, the redo after the resume counts."""
    cue = {"kind": "hold", "label": "fist", "text": "", "seconds": 4, "posture": "table"}
    events = [{"t": 0.0, "sample": 0, "kind": "cue", "index": 0, "cue": cue},
              {"t": 1.5, "sample": 750, "kind": "pause"},
              {"t": 9.0, "sample": 750, "kind": "resume"},
              {"t": 9.0, "sample": 750, "kind": "cue", "index": 0, "cue": cue},
              {"t": 13.0, "sample": 2750, "kind": "stop"}]
    first, redo = segments(events)
    assert first["bad"] and (first["sample0"], first["sample1"]) == (0, 750)
    assert not redo["bad"] and (redo["sample0"], redo["sample1"]) == (750, 2750)


def test_sign_sessions_train_their_signs_and_refuse_a_mix(tmp_path):
    """Train tab: E and F sessions train an LDA on the signs they share, rest and the wrist bends (notsign); grip
    sessions train as before; a mix is refused."""
    e = tmp_path / "2026-10-16_120000_test"
    f = tmp_path / "2026-10-16_130000_test"
    c = tmp_path / "2026-10-14_120000_test"
    _fake_session(e, protocol_e_plan(["table"], rounds=2, sign_s=3, rest_s=1.5, seed=0), {"protocol": "E"})
    _fake_session(f, protocol_f_plan(["table"], rounds=2, sign_s=3, rest_s=1.5, seed=0), {"protocol": "F"})
    _fake_session(c, protocol_c_plan(["fist"], rounds=1, move_s=3, rest_s=1.5, hold_s=2.5, seed=0), {"protocol": "C"})
    for folder in (e, f, c):
        meta = json.loads((folder / "session.json").read_text())
        meta.update(filter_version=FILTER_VERSION, started=1.7e9)
        (folder / "session.json").write_text(json.dumps(meta))
    options = {"log": True, "accel": False, "extended": False, "vote": 5}
    model, _ = gesture_model.train([e, f], options, evaluate_it=False)
    assert set(model["classes"]) == set(F_SIGNS) | {"rest", gesture_model.NOT_SIGN}
    assert model["options"]["signs"]
    model, _ = gesture_model.train([f], options, evaluate_it=False)
    assert set(model["classes"]) == set(F_SIGNS) | {"rest", gesture_model.NOT_SIGN}
    model, _ = gesture_model.train([c], options, evaluate_it=False)
    assert set(model["classes"]) == set(GESTURE_SET) and not model["options"]["signs"]
    with pytest.raises(ValueError):
        gesture_model.train([c, f], options, evaluate_it=False)
