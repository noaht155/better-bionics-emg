"""Recording web app, also runs the existing tools (connection check, live signals, calibration, haptics).
Serves the page in web/ and streams the live state to it.

    python app.py [--synthetic | --replay data/2026-09-30-filter] [--camera 0 | --camera clip.mp4 | --no-camera]
                  [--host 0.0.0.0] [--port 8000] [--hub]

Then open http://localhost:8000. --host 0.0.0.0 makes it reachable from a tablet on the same network. --hub makes
this computer the hub other computers send their sessions to and train on (hub.py): it then listens on every
network, but only answers this computer and devices with a token. The hub only trains, so it starts without the
camera and without connecting the armband (the Connection tab can still connect it).
"""
import argparse
import asyncio
import contextlib
import copy
import csv
import importlib.util
import io
import json
import queue
import socket
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import devices
import hub
import hum_test
import quest_link
import quest_video
from armband import Armband, Replay
from calibrate import load_calibration
from camera import Camera
from control import DEFAULTS as CONTROL_DEFAULTS, check_rules, rules_for, split_rules
from esp32_link import WIFI_IP
from gestures import (C_CLOSE_S, C_MOVE_S, C_ROUNDS, CATEGORIES, D_ACTIVITY_S, D_ROUNDS, E_ROUNDS, E_SIGN_S,
                      F_ROUNDS, F_SIGN_S, F_SIGNS, F_SIGNS_DEFAULT, GESTURE_SET, GESTURES, GRIP_CLOSE_S, LIBRARY, POSTURES, QUICK_REPS,
                      plan_for)
from hand_angles import JOINTS, RELIABLE
from haptics import (FADE, GAP_MS, GRIP_CODES, HAPTIC_DEFAULTS, LONG_MS, OTHER_CODE, SHORT_MS, STEP_MS, SWEEP,
                     haptic_settings)
from processing import FILTER_VERSION, spectrum
from recorder import Recorder
from session import DATA_DIR
from tools import POSES, Tools, forget_armband_wifi, model_path

# The ml package sits next to emg-reading/ in the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml import NETWORK_DIR, dataset, gesture_model  # noqa: E402

WEB_DIR = Path(__file__).with_name("web")
# Saved control parameter sets, next to the models. "default" is always the spec's values and isn't stored
PRESET_FILE = gesture_model.MODEL_DIR.parent / "control_presets.json"
# Saved mappings (rule lists) of the Control tab, apart from the profiles (presets) that also carry one
MAPPING_FILE = gesture_model.MODEL_DIR.parent / "control_mappings.json"
# Surface settings kept from the Control tab while trying them on the motors, one line each (Noah, 2026-11-23)
SURFACE_LOG = DATA_DIR / "surface-tries.jsonl"
SEND_HZ = 20
# The 3D contact page: frames sent per second, and how high above the desk the hand and object can go
CONTACT_HZ = 30
CONTACT_REACH_M = 0.45
# The Quest app's sim gets the control layer's output this often
QUEST_HZ = 60
# How often the recording's cue state is taken for the headset
QUEST_RECORD_HZ = 10
PREVIEW_HZ = 15
LOOPBACK = ("127.0.0.1", "::1")
# What the Train tab may ask the hub for through this app, as paths under /api/
HUB_PROXY = {path.removeprefix("/api/") for path in hub.TRAIN_PATHS} | {"hub/jobs"}


class PlanSettings(BaseModel):
    postures: list[Literal[tuple(POSTURES)]]
    # A: held grips and single fingers. B: the Ninapro-style movements (gestures.protocol_b_plan). C: movements
    # picked from the library (gestures.protocol_c_plan). D: activities that shouldn't start a grip
    # (gestures.protocol_d_plan). E: sign screening (gestures.protocol_e_plan). F: the chosen sign set
    # (gestures.protocol_f_plan)
    protocol: Literal["A", "B", "C", "D", "E", "F"] = "A"
    reps: int = Field(ge=1, le=20)
    hold_s: float = Field(ge=1, le=30)
    rest_s: float = Field(ge=1, le=30)
    free_s: float = Field(0, ge=0, le=600)
    rounds: int = Field(6, ge=1, le=20)
    move_s: float = Field(5, ge=2, le=15)
    moves: list[Literal[tuple(LIBRARY)]] = []
    c_rounds: int = Field(C_ROUNDS, ge=1, le=20)
    c_move_s: float = Field(C_MOVE_S, ge=3, le=15)
    d_rounds: int = Field(D_ROUNDS, ge=1, le=5)
    d_activity_s: float = Field(D_ACTIVITY_S, ge=10, le=120)
    e_rounds: int = Field(E_ROUNDS, ge=1, le=10)
    e_sign_s: float = Field(E_SIGN_S, ge=3, le=15)
    f_rounds: int = Field(F_ROUNDS, ge=1, le=20)
    f_sign_s: float = Field(F_SIGN_S, ge=3, le=15)
    f_signs: list[Literal[tuple(F_SIGNS)]] = Field(default_factory=lambda: list(F_SIGNS_DEFAULT), min_length=1)
    # Quick calibration after putting the band on: protocol A without the single fingers, or F, in every ticked
    # posture (calibrating at the table only lost 7 to 16 points with the arm forward, devlog 2026-10-22)
    quick: bool = False

    # Set here rather than on the form, so the form keeps the full session's values for next time
    @model_validator(mode="after")
    def _quick(self):
        if self.quick:
            self.reps = self.f_rounds = QUICK_REPS
        return self


class SessionSettings(PlanSettings):
    subject: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    band_arm: Literal["right", "left"]
    # Which way round the band is. Turned the other way the electrode order round the arm is mirrored
    port_facing: Literal["shoulder", "hand"]
    tracked_hand: Literal["right", "left"]
    # Band position: distance of the band from the anterior elbow crease, and which side of the forearm the main
    # module sits on (anterior: muscle belly, palm side; posterior: bony side). placement is free notes on top
    crease_distance: float = Field(ge=0, le=500)
    crease_unit: Literal["cm", "mm"]
    module_side: Literal["anterior", "posterior"]
    placement: str = ""
    notes: str = ""
    # The plan preview can be asked for with nothing ticked, a session can't start that way
    postures: list[Literal[tuple(POSTURES)]] = Field(min_length=1)
    # Same run, nothing saved, for trying out the setup
    practice: bool = False
    # Dev tab: only this many cues, kept short, never saved. 0 is a normal session
    dev_cues: int = Field(0, ge=0, le=50)


class HandChoice(BaseModel):
    hand: Literal["right", "left"]


class TrackingChoice(BaseModel):
    on: bool


class WifiSettings(BaseModel):
    # How long the adapter stays on DHCP, 0 skips it (connect_armband.ps1)
    dhcp_s: int = Field(5, ge=0, le=60)


class DeviceChoice(BaseModel):
    id: str = Field(max_length=64)


class RoundTripSettings(BaseModel):
    switches: int = Field(80, ge=10, le=200, multiple_of=2)


class FlashStart(BaseModel):
    # The page's clock, ms since 1970, to check it runs on this computer
    page_ms: float


class FlashTimes(BaseModel):
    # (time.time() seconds, screen turned light) of every flash change, from the page's clock
    # Empty when the page stopped early
    switches: list[tuple[float, bool]] = Field(max_length=1000)


class HumTestSettings(BaseModel):
    label: str
    seconds: float = Field(60, ge=5, le=120)


class CalibrationSettings(BaseModel):
    seconds: float = Field(ge=2, le=30)
    car: bool


class Esp32Link(BaseModel):
    port: str | None = None
    wifi: str | None = None


class TrainSettings(BaseModel):
    sessions: list[str] = Field(min_length=1)
    log: bool = True
    accel: bool = False
    extended: bool = True
    vote: int = Field(ge=1, le=15)
    # Positions left out of training
    exclude: list[str] = []


class ModelChoice(BaseModel):
    name: str
    # ESP32 link for the haptics that start with the model
    port: str | None = None
    wifi: str | None = None


class SimInputs(BaseModel):
    """What the Control tab's simulated model outputs: label None is rest."""
    label: str | None = None
    sure: float = Field(1.0, ge=0, le=1)
    effort: float = Field(0.0, ge=0, le=1)
    moving: bool = False


class Threshold(BaseModel):
    value: float = Field(ge=0, le=1)


class ControlParams(BaseModel):
    """control.Controller's parameters, see control.py."""
    threshold: float = Field(ge=0, le=1)
    vote_n: int = Field(ge=1, le=20)
    start_level: float = Field(ge=0, le=1)
    onset_s: float = Field(ge=0, le=5)
    closure_mode: Literal["position", "speed"]
    gain: float = Field(ge=0, le=20)
    hold_level: float = Field(ge=0, le=1)
    hold_on_relax: bool
    release_mode: Literal["open_class", "rest", "either", "rest_or_class"]
    release_n: int = Field(ge=1, le=20)
    release_level: float = Field(ge=0, le=1)
    release_s: float = Field(ge=0, le=5)
    # Presets saved before the distance check have no value, they get the default
    max_distance: float = Field(CONTROL_DEFAULTS["max_distance"], ge=0, le=50)
    mode: Literal["lock", "blend", "trigger", "rules"] = CONTROL_DEFAULTS["mode"]
    smooth_s: float = Field(CONTROL_DEFAULTS["smooth_s"], ge=0, le=3)
    lock_s: float = Field(CONTROL_DEFAULTS["lock_s"], ge=0, le=5)
    pulse_level: float = Field(CONTROL_DEFAULTS["pulse_level"], ge=0, le=1)
    long_s: float = Field(CONTROL_DEFAULTS["long_s"], ge=0.1, le=3)
    gap_s: float = Field(CONTROL_DEFAULTS["gap_s"], ge=0.1, le=3)
    double_s: float = Field(CONTROL_DEFAULTS["double_s"], ge=0.2, le=3)
    release_with: Literal["point", "open"] = CONTROL_DEFAULTS["release_with"]
    still_gate: bool = CONTROL_DEFAULTS["still_gate"]
    pinch_clenches: int = Field(CONTROL_DEFAULTS["pinch_clenches"], ge=1, le=2)
    spring_w: float = Field(ge=0, le=50)
    # The haptics that run with it, see haptics.py
    haptic_floor: int = Field(HAPTIC_DEFAULTS["haptic_floor"], ge=0, le=90)
    haptic_strength: int = Field(HAPTIC_DEFAULTS["haptic_strength"], ge=10, le=100)
    haptic_full_force: float = Field(HAPTIC_DEFAULTS["haptic_full_force"], ge=1, le=200)
    surfaces: dict[Literal["slippery", "smooth", "normal", "rough"], SurfaceProfile] = Field(
        default_factory=lambda: copy.deepcopy(HAPTIC_DEFAULTS["surfaces"]))


class SurfaceProfile(BaseModel):
    """One surface's pulse rate, see haptics.SURFACE_DEFAULTS. Profiles and tries saved before 2026-11-23 also have
    off and strength, which are dropped."""
    hz: float = Field(ge=0.5, le=10)


class SurfaceTry(BaseModel):
    """A note on one surface's pulses, or on the whole set (surface "all"). Every try keeps the whole set, so a note
    on one surface says what it was compared with."""
    surface: Literal["slippery", "smooth", "normal", "rough", "all"]
    surfaces: dict[Literal["slippery", "smooth", "normal", "rough"], SurfaceProfile]
    profile: str | None = None
    note: str = Field("", max_length=500)


class TryDelete(BaseModel):
    """A recording to delete: its place in the newest first list and its time, so a list that changed meanwhile
    can't delete another one."""
    index: int = Field(ge=0)
    at: str


class HandPlace(BaseModel):
    """The contact hand back to its start, or ready to grasp the object or the rail (sim.World.place_hand)."""
    where: Literal["start", "object", "rail"]


class ContactParams(BaseModel):
    """The contact simulation's settings, see sim.py."""
    object: str | None = None
    mass_scale: float | None = Field(None, ge=0.05, le=3)
    max_force: float | None = Field(None, ge=1, le=60)
    assist: bool | None = None
    rail: Literal["steps", "gradient"] | None = None


class VrSpawn(BaseModel):
    kind: str


class Switch(BaseModel):
    on: bool


class VrMenu(BaseModel):
    # Empty closes it
    page: str = Field("", max_length=40)


class VrPoint(BaseModel):
    panel: Literal["menu", "card", "dock", "tour"]
    page: str = Field("", max_length=40)
    name: str = Field(..., min_length=1, max_length=80)


class ObjectPlace(BaseModel):
    pos: list[float] = Field(min_length=3, max_length=3)


class SimStart(BaseModel):
    """The simulated model's classes, and the ESP32 link for its haptics."""
    classes: list[str]
    port: str | None = None
    wifi: str | None = None


class CursorParams(BaseModel):
    """cursor.Cursor's parameters, see cursor.py."""
    gain: float = Field(ge=0, le=10)
    smooth_s: float = Field(ge=0, le=2)
    rotation_deg: float = Field(ge=-180, le=180)
    flip: bool
    targets: int = Field(ge=2, le=12)
    target_r: float = Field(ge=0.2, le=1)
    target_size: float = Field(ge=0.05, le=0.5)
    home_size: float = Field(ge=0.05, le=0.5)
    home_s: float = Field(ge=0, le=3)
    dwell_s: float = Field(ge=0, le=3)
    timeout_s: float = Field(ge=1, le=30)


class CursorStart(BaseModel):
    session: str


class CursorTargets(BaseModel):
    on: bool


class Rule(BaseModel):
    """One line of a Control tab mapping, see control.py's rules mode."""
    trigger: str = Field(max_length=60)
    how: Literal["held", "repeated"]
    value: float = Field(ge=0, le=10)
    action: str = Field(max_length=60)
    # Settings (haptics.haptic_settings, checked with the rules); rules saved before 2026-11-05 have a name or none
    haptic: dict | str = "auto"


class TestTouch(BaseModel):
    """3D contact's test touch while on: the hand grips the rail at pos (a section number, sim.rail_pose) with the
    pressure as its effort. slide: move there gripping, else back off and over."""
    on: bool
    pressure: float = Field(0.0, ge=0, le=1)
    pos: float = Field(2.0, ge=0, le=3)
    slide: bool = False


class HapticPlay(BaseModel):
    """A rule's haptic settings, played once on the motors from the mapping page."""
    haptic: dict


class RuleLists(BaseModel):
    """A mapping: the rules while relaxed (no grip) and while holding one."""
    relaxed: list[Rule]
    holding: list[Rule]


class ControlPreset(BaseModel):
    """A profile: settings, the mapping in use when it was saved and the model it was built on."""
    name: str = Field(pattern=r"^[A-Za-z0-9 _-]{1,40}$")
    params: ControlParams
    rules: RuleLists | None = None
    model: str | None = None


class ControlMapping(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9 _-]{1,40}$")
    rules: RuleLists
    model: str | None = None
    protocol: Literal["grips", "F"] | None = None


class HubSettings(BaseModel):
    url: str = Field(pattern=r"^https?://[^\s/]+/?$")
    # Empty keeps the token already saved
    token: str = Field("", max_length=200)


class AutoSend(BaseModel):
    on: bool


class BackupSettings(BaseModel):
    folder: str


class ModelNames(BaseModel):
    names: list[str] = Field(min_length=1)


class ControlFiles(BaseModel):
    mappings: dict
    presets: dict


class NetworkJobSettings(BaseModel):
    job: Literal["evaluate", "evaluate_newest", "train", "calibrate", "signs"]
    epochs: int = Field(15, ge=1, le=200)
    network: str | None = None
    session: str | None = None
    reps: list[int] = Field([1, 2], min_length=1)
    # Sessions to train a network on, None for all good sessions; for signs the sign sessions
    sessions: list[str] | None = None
    # Signs left out of a sign network, grip networks always learn all the grips
    exclude: list[str] = []
    # calibrate: score from all repetitions but the last, then save the network calibrated on all of them
    cross: bool = False


class QuickCalibration(BaseModel):
    session: str


def placement_text(settings):
    """Band position for the session table. Sessions before 2026-10-13 only have the free text."""
    parts = []
    if settings.get("crease_distance") is not None:
        parts.append(f"{settings['crease_distance']:g} {settings['crease_unit']} from the elbow crease")
    if settings.get("module_side"):
        parts.append(f"module {settings['module_side']}")
    if settings.get("placement"):
        parts.append(settings["placement"])
    return ", ".join(parts)


def levels_session(model):
    """The session a model's effort and haptic levels come from: the one a calibrated network was calibrated on, or
    an LDA model's or sign network's newest training session. model as gesture_model.load gives it, or
    {"kind": "network", "info": ...} for a network."""
    if model.get("kind") == "network" and model["info"].get("kind") != "signs":
        return model["info"].get("calibrated_on")
    if model.get("kind") == "network":
        return model["info"]["sessions"][-1]
    return (model.get("sessions") or [None])[-1]


class Access:
    """Who may use the app. This computer may use everything. A device with a hub token (hub.py) may use the hub's
    requests and the Train tab's, nothing else: not the armband, the haptics or the camera. Anyone else on the
    network only when the app was started with --host (lan), as before for a tablet. Plain ASGI rather than an http
    middleware so it covers the websockets too."""

    def __init__(self, app, lan):
        self.app = app
        self.lan = lan

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope["path"]
        local = (scope.get("client") or ("",))[0] in LOOPBACK
        authorization = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        device = hub.device_for(authorization) if authorization else None
        if path.startswith("/api/hub/") and device is None:
            return await self.deny(scope, receive, send, 401, "unknown device, add it on the hub with hub.py")
        if not local and device is not None and not (path.startswith("/api/hub/") or path in hub.TRAIN_PATHS):
            return await self.deny(scope, receive, send, 403, "devices can only send and train through the hub")
        if not local and device is None and not self.lan:
            return await self.deny(scope, receive, send, 403, "this app only answers its own computer and devices")
        scope.setdefault("state", {})["device"] = device
        await self.app(scope, receive, send)

    @staticmethod
    async def deny(scope, receive, send, status, detail):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)


def session_list():
    """Recorded sessions, newest first, from their session.json only."""
    out = []
    for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
        meta = json.loads(path.read_text())
        settings = meta.get("settings", {})
        gestures = sum(c["kind"] == "hold" and c["label"] != "rest" for c in meta.get("plan", []))
        sign_session = settings.get("protocol") in ("E", "F")
        signs = sum(c["kind"] == "custom" for c in meta.get("plan", [])) if sign_session else 0
        # What a model trained on it could learn, for leaving positions out on the Train tab
        plan = meta.get("plan", [])
        if sign_session:
            positions = {c["label"] for c in plan if c["kind"] == "custom"}
            positions |= {"notsign"} if any(c["kind"] == "nogrip" for c in plan) else set()
        else:
            positions = {c["label"] for c in plan if c["kind"] == "hold" and c["label"] in GESTURE_SET}
        positions.discard("rest")
        out.append({"name": path.parent.name, "subject": meta.get("subject"), "postures": settings.get("postures"),
                    "band_arm": settings.get("band_arm"), "port_facing": settings.get("port_facing"),
                    "placement": placement_text(settings),
                    "gestures": gestures, "signs": signs, "positions": sorted(positions),
                    "completed": meta.get("completed"), "hum_passed": meta.get("hum_passed", False),
                    "seconds": round(meta["ended"] - meta["started"]) if meta.get("ended") else None,
                    "protocol": settings.get("protocol", "A")})
    return out


def model_list():
    """Saved LDA models for the current filter, newest first. Older ones stay on disk, loading them is refused."""
    out = []
    for path in sorted(gesture_model.MODEL_DIR.glob("*.joblib"), reverse=True):
        model = joblib.load(path)
        if model["filter_version"] != FILTER_VERSION:
            continue
        loso = (model.get("evaluation") or {}).get("loso", {}).get(model["vote"])
        out.append({"name": path.name, "subject": model["subject"], "trained": model["trained"],
                    "sessions": model["sessions"], "classes": model["classes"], "options": model["options"],
                    "accuracy": loso["accuracy"] if loso else None,
                    # Models trained before balanced accuracy was added don't have it
                    "balanced": loso.get("balanced") if loso else None,
                    "rest_false": loso["rest_false"] if loso else None})
    return out


def network_list():
    """Saved networks for the current filter, newest first (networks saved before filter version 3 don't say, they were
    version 2). Empty where torch isn't installed (it's only on the training desktop)."""
    try:
        import torch
    except ImportError:
        return []
    out = []
    for path in sorted(NETWORK_DIR.glob("*.pt"), reverse=True):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["info"].get("filter_version", 2) != FILTER_VERSION:
            continue
        out.append(dict(blob["info"], name=path.name, classes=blob["classes"]))
    return out


def saved_presets():
    """{name: {"params", "rules", "model"}}. Presets saved before profiles were only the params."""
    presets = json.loads(PRESET_FILE.read_text()) if PRESET_FILE.exists() else {}
    presets = {k: v if "params" in v else {"params": v, "rules": None, "model": None} for k, v in presets.items()}
    return {k: dict(v, rules=v["rules"] and split_rules(v["rules"])) for k, v in presets.items()}


def saved_mappings():
    """{name: {"rules", "model", "protocol"}}, rules as the two lists also for mappings saved as one."""
    mappings = json.loads(MAPPING_FILE.read_text()) if MAPPING_FILE.exists() else {}
    return {k: dict(v, rules=split_rules(v["rules"])) for k, v in mappings.items()}


def clip_workspace(pos):
    """Keeps the hand and object over the desk and near it (m)."""
    import sim
    hx, hy, _ = sim.DESK_HALF
    return [float(np.clip(pos[0], -hx, hx)), float(np.clip(pos[1], sim.DESK_CENTRE_Y - hy, sim.DESK_CENTRE_Y + hy)),
            float(np.clip(pos[2], 0.0, CONTACT_REACH_M))]


def refuse(e):
    raise HTTPException(409, str(e))


def headset_session_settings(quick):
    """Settings for a protocol F session started from the headset: the newest real session's setup (subject, band
    arm and placement), which the Record tab's form also starts from."""
    for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
        try:
            settings = json.loads(path.read_text()).get("settings") or {}
        except (OSError, ValueError):
            continue
        if settings.get("practice") or settings.get("dev_cues") or "subject" not in settings:
            continue
        settings = {k: v for k, v in settings.items() if k in SessionSettings.model_fields}
        return SessionSettings(**dict(settings, protocol="F", quick=quick, practice=False, dev_cues=0)).model_dump()
    raise ValueError("record one session on the Record tab first, the headset starts from its setup")


# Held wrist bends the headset shows on its ghost hands, degrees towards the palm, as web/app.js wristFrame draws them
HEADSET_WRIST_DEG = {"hold_wrist_flex": 60.0, "hold_wrist_ext": -55.0}


def headset_record_state(recorder):
    """The running session as the headset shows it: the cue, its target pose, progress and pauses."""
    session = recorder.snapshot()["session"]
    if session is None:
        return {"running": False}
    cue = session["cue"]
    upcoming = session["next"]
    return {"running": True, "paused": session["paused"], "hum": session["hum_pause"] is not None,
            "index": session["index"], "count": session["count"], "label": cue["label"], "kind": cue["kind"],
            "text": cue["text"], "seconds": cue["seconds"] or 0, "elapsed": session["cue_elapsed"],
            "marked_bad": session["marked_bad"], "next": upcoming["text"].splitlines()[0] if upcoming else "",
            "target": [float(a) for a in POSES.get(cue["label"], POSES["rest"])],
            "has_target": cue["label"] in POSES or cue["label"] in HEADSET_WRIST_DEG,
            "wrist": HEADSET_WRIST_DEG.get(cue["label"], 0.0), "wrist_cue": cue["label"] in HEADSET_WRIST_DEG}


def headset_haptics_state(haptics, test_touch):
    """What the motors play, for the headset's Play page: the touch they follow, its pulse rate and the strongest
    duty (%). No nulls, the headset's JSON reader can't take them."""
    state = haptics.state
    if not state.get("running"):
        return {"on": False}
    touch = state.get("touch")
    duties = state.get("duties") or [0]
    if touch is None:
        return {"on": True, "mode": "effort", "duty": float(max(duties))}
    friction = touch.get("friction")
    return {"on": True, "mode": "surface", "force": touch["force"], "surface": touch["surface"] or "",
            "hz": touch["hz"], "friction": -1.0 if friction is None else friction, "duty": float(max(duties)),
            "test": test_touch}


def headset_cal_session(summary):
    """The session the headset can calibrate on: the last one, if it was a completed quick session."""
    quick = summary and summary.get("quick") and summary.get("completed") and not summary.get("practice")
    return summary["session"] if quick else None


def headset_train_state(cal, summary, tools):
    """What the headset's Train section shows, for the calibration in cal (make_app's headset_cal) on the last
    session (summary). No nulls, the headset's JSON reader can't take them."""
    session = headset_cal_session(summary)
    out = {"session": session or "", "running": False, "status": "", "saved": "", "error": cal["error"]}
    if cal["job"] is None or cal["session"] != session:
        return out
    job = getattr(tools, cal["job"]).state
    if job["running"]:
        cal["seen"] = True
        out.update(running=True, status=f"Calibrating, {time.monotonic() - cal['started']:.0f} s")
    elif cal["seen"]:
        result = job.get("result") or {}
        if job.get("error") or not result.get("saved"):
            out["error"] = out["error"] or job.get("error") or "calibration finished without saving a network"
        else:
            scores = result.get("scores") or {}
            score = f"{scores['balanced'] * 100:.0f} % balanced on the last round. " if "balanced" in scores else ""
            out.update(saved=result["saved"], status=f"{score}Saved {result['saved']}")
    else:
        out.update(running=True, status="Starting the calibration")
    return out


def headset_calibrations(limit=6):
    """The newest calibrated networks (by when they were saved) for the headset's Train section to pick from. From
    the file names only, so it can run with every refresh."""
    paths = sorted(NETWORK_DIR.glob("*_cal_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in paths[:limit]]


def session_folder(name):
    """A recorded session's folder by its name, refused unless it is one."""
    folder = DATA_DIR / name
    if Path(name).name != name or name in ("", ".", "..") or not (folder / "session.json").exists():
        refuse("unknown session")
    return folder


def model_file(name):
    """A saved LDA model (.joblib) or network (.pt) by its file name, refused unless it is one."""
    folder = NETWORK_DIR if name.endswith(".pt") else gesture_model.MODEL_DIR
    path = folder / name
    if path.parent != folder or path.suffix not in (".joblib", ".pt") or not path.is_file():
        refuse("unknown model")
    return path


# Epochs for the evaluation that runs by itself after every session, the default from the epoch sweep (devlog 2026-10-06)
AUTO_EVALUATE_EPOCHS = 15


def make_app(recorder, band, camera, tools, lan=True, serving=False):
    """lan: other computers on the network may use the page (--host). serving: started with --hub."""
    app = FastAPI()
    app.add_middleware(Access, lan=lan)

    def auto_evaluate(folder):
        """Every completed session gets an honest score before anything trains on it: a network trained on all the
        other sessions is tested on it, in the background at low priority, and the row goes to evaluations.csv.
        Skipped without torch, without grips in the session, or while another network job runs."""
        if importlib.util.find_spec("torch") is None:
            return
        good = [f.name for f in dataset.good_sessions()]
        if folder.name not in good or len(good) < 2:
            return
        try:
            tools.network.start("evaluate_newest", ["ml.train_ringnet", "--epochs", str(AUTO_EVALUATE_EPOCHS),
                                                     "--held", folder.name])
        except ValueError:
            pass

    def after_saved(folder):
        tools.backup.request()
        tools.sync_watch.request()
        auto_evaluate(folder)

    recorder.on_saved = after_saved
    # Catches up on sessions that ended while the app was off, and on sessions stopped early, which don't call
    # on_saved
    tools.backup.request()

    @app.middleware("http")
    async def revalidate(request, call_next):
        # Without this Firefox kept an old hand.js after an update, the new app.js then failed on every
        # gesture cue and the camera overlay froze. no-cache still caches, but asks the server first
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/web/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    def surfaces():
        """What each object, the desk and the floor are made of, for the haptics settings; None without mujoco."""
        try:
            import sim
        except ImportError:
            return None
        return {"names": list(sim.SURFACES), "of": sim.SURFACE_OF, "friction": sim.SURFACE_FRICTION}

    @app.get("/api/config")
    def config():
        try:
            loaded = load_calibration()
        except SystemExit:
            loaded = None
        calibration = {str(ch): {"rest": r, "max": m} for ch, (r, m) in loaded[0].items()} if loaded else None
        return {"joints": JOINTS, "reliable": RELIABLE, "channels": band.channels, "rate": band.rate,
                "source": band.source, "calibration": calibration,
                "gestures": {name: {"text": text, "angles": angles} for name, (text, angles) in GESTURES.items()},
                "gesture_set": GESTURE_SET, "sign_sets": {"F": F_SIGNS}, "f_signs_default": F_SIGNS_DEFAULT,
                "haptic_codes": {"grips": GRIP_CODES, "other": OTHER_CODE, "short_ms": SHORT_MS, "long_ms": LONG_MS,
                                 "gap_ms": GAP_MS, "step_ms": STEP_MS, "fade": FADE, "sweep": SWEEP}, "library": LIBRARY, "categories": CATEGORIES,
                "cue_timing": {"grip_close_s": GRIP_CLOSE_S, "c_close_s": C_CLOSE_S},
                "postures": POSTURES, "camera": camera is not None, "hum_blocks": hum_test.BLOCKS,
                "tracked_hand": camera.hand if camera else None, "esp32_wifi_ip": WIFI_IP, "platform": sys.platform,
                "fixed_delay_s": dataset.CAMERA_DELAY_S, "surfaces": surfaces()}

    @app.get("/api/devices")
    def device_list():
        """The setups in devices.json for the start page, each marked whether it belongs to this computer."""
        return {"computer": devices.computer_name(), "cameras": devices.cameras(),
                "current": recorder.setup["id"] if recorder.setup else None, "recording": recorder.session is not None,
                "setups": [dict(s, here=devices.here(s)) for s in devices.setups()]}

    @app.post("/api/devices/select")
    def select_device(choice: DeviceChoice):
        setup = devices.find(choice.id)
        if setup is None:
            refuse(f"no setup {choice.id}")
        if not devices.here(setup):
            refuse(f"{setup['name']} is the setup of {setup['computer']}, this is {devices.computer_name()}")
        if recorder.session is not None:
            refuse("a session is recording")
        # A video file (--camera clip.mp4) is for testing and stays
        if camera is not None and isinstance(camera.source, int):
            camera.switch(setup["camera"])
        recorder.setup = setup
        return setup

    @app.post("/api/plan")
    def plan_length(settings: PlanSettings):
        """Length of the session these settings give, from the real plan so the page needn't copy its logic. The
        plan itself is for the cue preview; grip order is shuffled again when the session starts."""
        try:
            plan = plan_for(settings.model_dump())
        except ValueError as e:
            refuse(e)
        return {"seconds": sum(c["seconds"] or 0 for c in plan), "cues": len(plan), "plan": plan}

    @app.post("/api/session/start")
    def start(settings: SessionSettings):
        if tools.calibration.state["running"]:
            refuse("wait for the calibration to finish")
        try:
            return {"folder": recorder.start_session(settings.model_dump())}
        except ValueError as e:
            refuse(e)

    @app.post("/api/session/{action}")
    def command(action: Literal["continue", "pause", "bad", "stop", "allow_hum"]):
        try:
            recorder.command(action)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/check")
    def check():
        try:
            tools.check.start()
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/armband/wifi")
    def armband_wifi(settings: WifiSettings):
        try:
            tools.wifi.start(settings.dhcp_s)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/delay/round-trip")
    def delay_round_trip(settings: RoundTripSettings):
        try:
            tools.round_trip.start(settings.switches)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/delay/flash/start")
    def delay_flash_start(settings: FlashStart):
        try:
            tools.flash.start(settings.page_ms)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/delay/flash/finish")
    def delay_flash_finish(times: FlashTimes):
        try:
            tools.flash.finish(times.switches)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/hum-test")
    def hum_test_start(settings: HumTestSettings):
        try:
            tools.hum.start(settings.label, settings.seconds)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/hum-test/stop")
    def hum_test_stop():
        tools.hum.stop()
        return {"ok": True}

    @app.post("/api/delay/save")
    def delay_save():
        try:
            return tools.save_delay()
        except ValueError as e:
            refuse(e)

    @app.post("/api/calibrate")
    def calibrate(settings: CalibrationSettings):
        try:
            tools.calibration.start(settings.seconds, settings.car)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/haptics/start")
    def haptics_start(link: Esp32Link):
        try:
            tools.haptics.start(link.port, link.wifi)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/haptics/stop")
    def haptics_stop():
        tools.haptics.stop()
        return {"ok": True}

    @app.post("/api/esp32/test")
    def esp32_test(link: Esp32Link):
        try:
            tools.test.start(link.port, link.wifi)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.get("/api/spectrum")
    def spectra(filtered: bool = False):
        raw, filt = recorder.window(recorder.history)
        x = filt if filtered else raw
        if x.shape[1] < band.rate:
            return {"f": [], "mag": []}
        # 1 Hz bins are plenty for a screen and keep the message small
        per_hz = x.shape[1] // band.rate
        out = []
        for row in x:
            f, mag = spectrum(row, band.rate)
            n = len(mag) // per_hz * per_hz
            out.append(np.round(mag[:n].reshape(-1, per_hz).mean(axis=1), 4).tolist())
        return {"f": f[:n].reshape(-1, per_hz).mean(axis=1).round(1).tolist(), "mag": out}

    @app.get("/api/sessions")
    def sessions():
        return session_list()

    @app.post("/api/train")
    def train(settings: TrainSettings, request: Request):
        folders = [session_folder(name) for name in settings.sessions]
        try:
            tools.trainer.start([str(f) for f in folders],
                                {"log": settings.log, "accel": settings.accel, "vote": settings.vote,
                                 "extended": settings.extended, "exclude": settings.exclude},
                                use=request.state.device is None)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.get("/api/models")
    def models():
        return model_list()

    @app.get("/api/networks")
    def networks():
        return network_list()

    @app.post("/api/network/run")
    def run_network(settings: NetworkJobSettings):
        good = [f.name for f in dataset.good_sessions() if not dataset.quick(f)]
        if settings.exclude and settings.job != "signs":
            refuse("leaving out positions works for LDA and sign networks; grip networks always learn all 7 grips")
        if settings.job in ("evaluate", "evaluate_newest") and len(good) < 2:
            refuse("needs at least 2 completed sessions with grips")
        if settings.job == "evaluate":
            args = ["ml.train_ringnet", "--epochs", str(settings.epochs)]
        elif settings.job == "evaluate_newest":
            args = ["ml.train_ringnet", "--epochs", str(settings.epochs), "--held", good[-1]]
        elif settings.job == "train":
            if not good:
                refuse("no completed sessions with grips yet")
            args = ["ml.networks", "train", "--epochs", str(settings.epochs)]
            if settings.sessions is not None:
                if not settings.sessions or any(n not in good for n in settings.sessions):
                    refuse("pick at least one completed session with grips")
                args += ["--sessions", *settings.sessions]
        else:
            name = settings.network or ""
            if Path(name).name != name or not (NETWORK_DIR / name).is_file():
                refuse("unknown network")
            if settings.job == "signs":
                signs = [s["name"] for s in session_list() if s["completed"] and not s["hum_passed"] and s["signs"]]
                if not settings.sessions or any(n not in signs for n in settings.sessions):
                    refuse("pick at least one completed sign session")
                args = ["ml.networks", "signs", name, "--epochs", str(settings.epochs), "--sessions", *settings.sessions]
                if settings.exclude:
                    args += ["--exclude", *settings.exclude]
            else:
                args = ["ml.networks", "calibrate", name, str(session_folder(settings.session or "")),
                        "--reps", *map(str, settings.reps)] + (["--cross"] if settings.cross else [])
        try:
            tools.network.start(settings.job, args)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/network/quick")
    def quick_calibration(choice: QuickCalibration):
        """Calibrates the newest base network on a quick calibration session (end card of the Record tab): the
        general network for grips, the newest sign network for a protocol F session. A device with a hub runs it
        on the hub's GPU and downloads the result (tools.HubCalibration)."""
        folder = session_folder(choice.session)
        if not serving and hub.load().get("url"):
            try:
                tools.hub_calibration.start(folder.name)
            except ValueError as e:
                refuse(e)
            return {"network": None, "job": "hub_calibration"}
        settings = json.loads((folder / "session.json").read_text()).get("settings", {})
        signs = settings.get("protocol") == "F"
        if not signs and settings.get("protocol", "A") != "A":
            refuse("only protocol A and F sessions can be used for a quick calibration")
        kind = "signs" if signs else "general"
        bases = [n for n in network_list() if n.get("kind") == kind]
        if not bases:
            what = "sign network" if signs else "general network"
            refuse(f"no {what} on this computer, train one on the Train tab or download one in System > Sync")
        base = max(bases, key=lambda n: n.get("created", ""))["name"]
        reps = list(range(1, settings.get("f_rounds" if signs else "reps", 2) + 1))
        args = ["ml.networks", "calibrate", base, str(folder), "--reps", *map(str, reps), "--cross"]
        try:
            tools.network.start("calibrate", args)
        except ValueError as e:
            refuse(e)
        return {"network": base, "job": "network"}

    @app.post("/api/network/stop")
    def stop_network():
        tools.network.stop()
        return {"ok": True}

    @app.get("/api/ml-cache")
    def ml_cache():
        files = list(dataset.CACHE.glob("*.npz"))
        return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}

    @app.post("/api/ml-cache/clear")
    def clear_ml_cache():
        if tools.network.state["running"]:
            refuse("a network job is reading the cache, wait for it to finish or stop it")
        return {"freed": dataset.clear_cache()}

    @app.post("/api/model/use")
    def use_model(choice: ModelChoice):
        path = model_file(choice.name)
        try:
            model = tools.predictor.use(path)
        except ValueError as e:
            refuse(e)
        return {"ok": True, "haptics": start_model_haptics(model, choice)}

    @app.post("/api/control/start")
    def control_start(choice: ModelChoice):
        """An LDA model or a calibrated network with the control layer on its output, effort levels from the
        model's newest training session (LDA) or its calibration session (network). Haptics start with it as with
        Use this model."""
        path = model_file(choice.name)
        try:
            if path.suffix == ".pt":
                info = next(n for n in network_list() if n["name"] == choice.name)
                # A sign network takes its levels from its newest training session until it is calibrated
                if not info.get("calibrated_on") and info.get("kind") != "signs":
                    refuse("calibrate the network on a session first, the effort levels come from it")
                session = model_session({"kind": "network", "info": info})
            else:
                session = model_session(gesture_model.load(path))
            if session is None:
                refuse("the model's training session isn't on this computer, the effort levels come from it")
            tools.control.prepare(DATA_DIR / session)
            model = tools.predictor.use(path, tools.control)
        except ValueError as e:
            refuse(e)
        return {"ok": True, "haptics": start_model_haptics(model, choice, tools.control.params)}

    @app.post("/api/control/simulate")
    def control_simulate(sim: SimStart):
        """The control layer without a model or the band: the page plays the model (class, effort, arm moving). The
        Control tab's haptics run with it."""
        if len(sim.classes) < 2:
            refuse("the simulated model needs at least two classes")
        tools.control.prepare_sim()
        try:
            tools.predictor.simulate(sim.classes, tools.control)
        except ValueError as e:
            refuse(e)
        tools.haptics.stop()
        tools.haptics.join(timeout=3)
        try:
            tools.haptics.start(sim.port, sim.wifi, settings=tools.control.params)
        except ValueError as e:
            return {"ok": True, "haptics": f"haptics not started: {e}"}
        tools.haptics.state["with_model"] = True
        return {"ok": True, "haptics": "haptics starting, see the Haptics tab"}

    @app.post("/api/contact/test-touch")
    def contact_test_touch(touch: TestTouch):
        if touch.on and not tools.contact.state["running"]:
            refuse("the contact simulation hasn't started")
        tools.contact.set_test(touch.model_dump(exclude={"on"}) if touch.on else None)
        return {"ok": True}

    @app.post("/api/control/sim")
    def control_sim(inputs: SimInputs):
        try:
            tools.predictor.set_sim(inputs.model_dump())
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/contact/start")
    def contact_start():
        """The 3D contact page: the hand follows the control layer while it runs, at rest otherwise."""
        if tools.contact.state["running"]:
            return tools.contact.scene()
        try:
            tools.contact.start()
        except ImportError:
            refuse("mujoco isn't installed, pip install -r requirements.txt")
        return tools.contact.scene()

    @app.post("/api/contact/stop")
    def contact_stop():
        tools.contact.stop()
        return {"ok": True}

    @app.get("/api/contact/scene")
    def contact_scene():
        scene = tools.contact.scene()
        if scene is None:
            refuse("the contact simulation hasn't started")
        return scene

    @app.post("/api/contact/params")
    def contact_params(params: ContactParams):
        import sim
        changes = params.model_dump(exclude_none=True)
        if "object" in changes and changes["object"] not in sim.OBJECTS:
            refuse(f"no object {changes['object']}")
        tools.contact.set_params(changes)
        return {"ok": True, "version": tools.contact.version}

    @app.post("/api/contact/place")
    def contact_place(place: ObjectPlace):
        tools.contact.place(clip_workspace(place.pos))
        return {"ok": True}

    @app.post("/api/vr/spawn")
    def vr_spawn(spawn: VrSpawn):
        import sim
        if spawn.kind not in sim.OBJECTS:
            refuse(f"no object {spawn.kind}")
        tools.contact.quest_command("spawn", spawn.kind)
        return {"ok": True}

    @app.post("/api/vr/point")
    def vr_point(point: VrPoint):
        """Makes a control on a headset panel pulse, to show the user where to press (App.PointAt)."""
        if "	" in point.page + point.name:
            refuse("no tabs in the names")
        tools.contact.quest_command("point", f"{point.panel}	{point.page}	{point.name}")
        return {"ok": True}

    @app.post("/api/vr/menu")
    def vr_menu(menu: VrMenu):
        """Opens the headset's menu on a page, or closes it (App.OpenMenuAt)."""
        tools.contact.quest_command("menu", menu.page)
        return {"ok": True}

    @app.post("/api/vr/keep-video")
    def vr_keep_video(switch: Switch):
        """Turns keeping the headset's picture and hand angles beside sessions on or off (quest_video.SessionVideo)."""
        if quest_video.SESSION_VIDEO is None:
            refuse("no Quest link in this copy of the app")
        quest_video.SESSION_VIDEO.on = switch.on
        return {"ok": True}

    @app.post("/api/vr/tour")
    def vr_tour():
        """Starts the headset's tour from its first step (Tour.cs)."""
        tools.contact.quest_command("tour")
        return {"ok": True}

    @app.post("/api/vr/clear")
    def vr_clear():
        tools.contact.quest_command("clear")
        return {"ok": True}

    @app.post("/api/contact/reset")
    def contact_reset():
        tools.contact.reset_object()
        return {"ok": True}

    @app.post("/api/contact/hand")
    def contact_hand(place: HandPlace):
        tools.contact.place_hand(place.where)
        return {"ok": True}

    @app.post("/api/control/params")
    def control_params(params: ControlParams):
        tools.control.set_params(params.model_dump())
        return {"ok": True}

    @app.post("/api/control/play_haptic")
    def play_haptic(play: HapticPlay):
        try:
            settings = haptic_settings(play.haptic)
        except ValueError as e:
            refuse(e)
        if not tools.haptics.state["running"] or tools.haptics.state.get("mode") != "combined":
            refuse("plays with the Control tab's haptics: start the model on the Control tab with the motors connected")
        tools.haptics.cue("start", None, settings)
        return {"ok": True}

    @app.post("/api/control/reset")
    def control_reset():
        tools.control.reset_counts()
        return {"ok": True}

    @app.post("/api/control/rules")
    def control_rules(rules: RuleLists):
        try:
            tools.control.set_rules(rules.model_dump())
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/control/default_rules")
    def control_default_rules(classes: list[str]):
        """The default mapping for a model with these classes."""
        return rules_for(classes)

    @app.get("/api/control/mappings")
    def control_mappings():
        return saved_mappings()

    @app.post("/api/control/mappings")
    def save_control_mapping(mapping: ControlMapping):
        try:
            check_rules(mapping.rules.model_dump(), POSES)
        except ValueError as e:
            refuse(e)
        mappings = saved_mappings()
        mappings[mapping.name] = {"rules": mapping.rules.model_dump(), "model": mapping.model,
                                 "protocol": mapping.protocol}
        MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
        MAPPING_FILE.write_text(json.dumps(mappings, indent=2) + "\n")
        return {"ok": True}

    @app.delete("/api/control/mappings/{name}")
    def delete_control_mapping(name: str):
        mappings = saved_mappings()
        if mappings.pop(name, None) is None:
            refuse("no saved mapping with that name")
        MAPPING_FILE.write_text(json.dumps(mappings, indent=2) + "\n")
        return {"ok": True}

    @app.get("/api/control/presets")
    def control_presets():
        return {"default": {"params": dict(CONTROL_DEFAULTS, mode="rules"), "rules": None, "model": None},
                **saved_presets()}

    @app.post("/api/control/presets")
    def save_control_preset(preset: ControlPreset):
        if preset.name == "default":
            refuse("default is fixed, pick another name")
        presets = saved_presets()
        presets[preset.name] = {"params": preset.params.model_dump(), "model": preset.model,
                                "rules": None if preset.rules is None else preset.rules.model_dump()}
        PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRESET_FILE.write_text(json.dumps(presets, indent=2) + "\n")
        return {"ok": True}

    @app.delete("/api/control/presets/{name}")
    def delete_control_preset(name: str):
        presets = saved_presets()
        if presets.pop(name, None) is None:
            refuse("no saved preset with that name")
        PRESET_FILE.write_text(json.dumps(presets, indent=2) + "\n")
        return {"ok": True}

    @app.get("/api/surface-tries")
    def surface_tries():
        """Newest first."""
        if not SURFACE_LOG.exists():
            return []
        return [json.loads(line) for line in reversed(SURFACE_LOG.read_text().splitlines()) if line]

    @app.get("/api/surface-tries.csv")
    def surface_tries_csv():
        """The recorded surface settings as a spreadsheet, oldest first."""
        out = io.StringIO()
        writer = csv.writer(out)
        # Tries before 2026-11-23 have only their own surface's settings, which also varied the duty between pulses
        # and the strength
        names = list(HAPTIC_DEFAULTS["surfaces"])
        writer.writerow(["at", "surface", "hz", *[f"{n}_hz" for n in names], "lowest_duty_pct", "full_force_n",
                         "profile", "note", "between_pulses_pct", "strength_pct"])
        for t in reversed(surface_tries()):
            s = t.get("settings") or {}
            every = t.get("surfaces") or {}
            writer.writerow([t["at"], t["surface"], s.get("hz", ""), *[every.get(n, {}).get("hz", "") for n in names],
                             t["floor"], t["full_force"], t["profile"] or "", t["note"], s.get("off", ""),
                             s.get("strength", "")])
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="surface-tries.csv"'})

    @app.post("/api/surface-tries")
    def add_surface_try(entry: SurfaceTry):
        """The surfaces' settings as they are now, with the haptics settings they play under. settings: the noted
        surface's own, None for the whole set."""
        p = tools.control.params
        e = entry.model_dump()
        line = dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), surface=e["surface"],
                    settings=e["surfaces"].get(e["surface"]), surfaces=e["surfaces"], profile=e["profile"],
                    note=e["note"], floor=p["haptic_floor"], full_force=p["haptic_full_force"])
        SURFACE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SURFACE_LOG.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return {"ok": True}

    @app.post("/api/surface-tries/delete")
    def delete_surface_try(which: TryDelete):
        tries = surface_tries()
        if which.index >= len(tries) or tries[which.index]["at"] != which.at:
            refuse("that recording changed, reload the list")
        del tries[which.index]
        SURFACE_LOG.write_text("".join(json.dumps(t) + "\n" for t in reversed(tries)))
        return {"ok": True}

    @app.post("/api/cursor/start")
    def cursor_start(choice: CursorStart):
        """The cursor with channel levels from a recorded session (the haptics' levels)."""
        try:
            tools.cursor.start(session_folder(choice.session))
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/cursor/stop")
    def cursor_stop():
        tools.cursor.stop()
        return {"ok": True}

    @app.post("/api/cursor/params")
    def cursor_params(params: CursorParams):
        tools.cursor.set_params(params.model_dump())
        return {"ok": True}

    @app.post("/api/cursor/targets")
    def cursor_targets(choice: CursorTargets):
        try:
            tools.cursor.targets(choice.on)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/model/threshold")
    def set_threshold(threshold: Threshold):
        tools.predictor.set_threshold(threshold.value)
        return {"ok": True}

    @app.post("/api/model/stop")
    def stop_model():
        tools.predictor.stop()
        if tools.haptics.state.get("with_model"):
            tools.haptics.stop()
        return {"ok": True}

    def model_session(model):
        """levels_session, None if that session isn't on this computer."""
        session = levels_session(model)
        return session if session and (DATA_DIR / session / "session.json").exists() else None

    def start_model_haptics(model, link, settings=None):
        """Haptics run with every prediction model: the Control tab's (settings), or effort mode with levels from
        model_session. Returns what happened, for the page. The model keeps running when the haptics can't start."""
        session = model_session(model)
        if session is None and settings is None:
            return "haptics not started: the model's calibration session isn't on this computer"
        tools.haptics.stop()
        tools.haptics.join(timeout=3)
        try:
            tools.haptics.start(link.port, link.wifi, None if settings else DATA_DIR / session, settings)
        except ValueError as e:
            return f"haptics not started: {e}"
        tools.haptics.state["with_model"] = True
        if settings:
            return "haptics starting, see the Haptics tab"
        return f"haptics starting with levels from {session}, see the Haptics tab"

    # The hub's side, for devices (hub.py). Access only lets these through with a device token

    @app.get("/api/hub/hello")
    def hub_hello(request: Request):
        return {"device": request.state.device, "hub": socket.gethostname()}

    @app.get("/api/hub/sessions")
    def hub_sessions():
        return hub.ended_sessions()

    @app.get("/api/hub/sessions/{name}/hashes")
    def hub_session_hashes(name: str):
        folder = session_folder(name)
        return {file: hub.sha256(folder / file) for file in hub.file_sizes(folder)}

    @app.get("/api/hub/incoming/{name}")
    def hub_received(name: str):
        try:
            return hub.received(name)
        except ValueError as e:
            refuse(e)

    @app.put("/api/hub/incoming/{name}/{file}")
    async def hub_receive(name: str, file: str, request: Request, offset: int = Query(ge=0)):
        data = await request.body()
        if len(data) > hub.CHUNK:
            refuse("chunk too big")
        try:
            return {"size": hub.receive(name, file, offset, data)}
        except ValueError as e:
            refuse(e)

    @app.delete("/api/hub/incoming/{name}")
    def hub_discard(name: str):
        try:
            hub.discard(name)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/hub/incoming/{name}/done")
    def hub_finish(name: str, hashes: dict[str, str]):
        try:
            folder = hub.finish(name, hashes)
        except ValueError as e:
            refuse(e)
        # Same as a session recorded here: backed up and scored before anything trains on it
        tools.backup.request()
        tools.sync_watch.request()
        auto_evaluate(folder)
        return {"ok": True}

    @app.get("/api/hub/backups")
    def hub_backups():
        """What the backup holds, for a device to check before it deletes a session it sent."""
        try:
            dest = hub.backup_folder()
        except ValueError:
            dest = None
        return {"folder_set": dest is not None, "sessions": hub.backed_up(dest) if dest else {}}

    @app.get("/api/hub/models")
    def hub_models():
        """The models a device can run (LDA, calibrated and sign networks), each with the session its levels come
        from if the hub has it."""
        live = [(m, {"sessions": m["sessions"]}) for m in model_list()]
        live += [(n, {"kind": "network", "info": n}) for n in network_list()
                 if n.get("kind") in ("calibrated", "signs")]
        out = []
        for entry, model in live:
            path = model_file(entry["name"])
            out.append(dict(entry, size=path.stat().st_size, sha256=hub.sha256(path), levels=model_session(model)))
        return out

    @app.get("/api/hub/file")
    def hub_file(kind: Literal["model", "session"], name: str, file: str = "", offset: int = Query(0, ge=0),
                 length: int = Query(hub.CHUNK, ge=1, le=hub.CHUNK)):
        if kind == "model":
            path = model_file(name)
        elif file in hub.SESSION_FILES:
            path = session_folder(name) / file
        else:
            refuse("not a session file")
        with open(path, "rb") as f:
            f.seek(offset)
            return Response(f.read(length), media_type="application/octet-stream")

    def read_named(path):
        return json.loads(path.read_text()) if path.exists() else {}

    @app.post("/api/hub/control")
    def hub_control(theirs: ControlFiles):
        """Takes the device's mappings and profiles the hub doesn't have, answers with all of the hub's."""
        kept = hub.merge_named(MAPPING_FILE, theirs.mappings) + hub.merge_named(PRESET_FILE, theirs.presets)
        return {"mappings": read_named(MAPPING_FILE), "presets": read_named(PRESET_FILE), "kept": kept}

    @app.get("/api/hub/jobs")
    def hub_jobs():
        """The Train tab's jobs, which the hub's own page gets over the websocket."""
        snapshot = tools.snapshot()
        return {"trainer": snapshot["trainer"], "network": snapshot["network"]}

    # A device's side: System > Sync, and the Train tab when it trains on the hub

    @app.get("/api/sync/config")
    def sync_config():
        config = hub.load()
        return {"url": config.get("url"), "token_set": bool(config.get("token")), "serving": serving,
                "devices": hub.devices(), "name": socket.gethostname(), "auto_send": bool(config.get("auto_send"))}

    @app.post("/api/sync/auto")
    def set_auto_send(choice: AutoSend):
        hub.save(dict(hub.load(), auto_send=choice.on))
        tools.sync_watch.request()
        return {"ok": True}

    @app.post("/api/sync/config")
    def set_sync_config(settings: HubSettings):
        """Saved only once the hub answers with it."""
        config = hub.load()
        token = settings.token.strip() or config.get("token")
        if not token:
            refuse("paste the token the hub gave this device")
        try:
            hello = hub.Hub(settings.url, token).request("GET", "/api/hub/hello", timeout=10)
        except ValueError as e:
            refuse(e)
        hub.save(dict(config, url=settings.url.rstrip("/"), token=token))
        tools.sync_watch.request()
        return hello

    @app.get("/api/sync/status")
    def sync_status():
        """What this computer and the hub each have: sessions to send, models to download."""
        try:
            client = hub.client()
            hello = client.request("GET", "/api/hub/hello")
            theirs = client.request("GET", "/api/hub/sessions")
            models = client.request("GET", "/api/hub/models")
        except ValueError as e:
            refuse(e)
        sessions = []
        for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
            folder = path.parent
            if not hub.SESSION_NAME.match(folder.name):
                continue
            sizes = hub.file_sizes(folder)
            if not hub.ended(folder):
                where = "recording"
            elif folder.name not in theirs:
                where = "not sent"
            else:
                where = "on the hub" if theirs[folder.name] == sizes else "name taken on the hub"
            sessions.append({"name": folder.name, "where": where, "bytes": sum(sizes.values())})
        for m in models:
            path = model_path(m["name"])
            m["here"] = "no" if not path.exists() else "yes" if hub.sha256(path) == m["sha256"] else "differs"
            m["levels_here"] = bool(m["levels"]) and (DATA_DIR / m["levels"] / "session.json").exists()
        return {"hub": hello, "sessions": sessions, "models": models}

    @app.get("/api/backup")
    def backup_status():
        folder = hub.load().get("backup")
        try:
            dest = hub.backup_folder()
        except ValueError as e:
            return {"folder": folder, "error": str(e)}
        if dest is None:
            return {"folder": None}
        done = hub.backed_up(dest)
        return {"folder": folder, "sessions": len(done),
                "missing": len([name for name in hub.ended_sessions() if name not in done])}

    @app.post("/api/backup/folder")
    def set_backup_folder(settings: BackupSettings):
        folder = settings.folder.strip()
        if folder and not Path(folder).is_dir():
            refuse(f"{folder} isn't a folder on this computer")
        if folder and Path(folder).resolve().is_relative_to(DATA_DIR.resolve()):
            refuse("the backup can't be inside data/")
        hub.save(dict(hub.load(), backup=folder or None))
        tools.backup.request()
        tools.sync_watch.request()
        return {"ok": True}

    @app.post("/api/backup/run")
    def run_backup():
        tools.backup.request()
        return {"ok": True}

    @app.post("/api/sync/send")
    def sync_send():
        try:
            tools.sync.send()
        except ValueError as e:
            refuse(e)
        tools.sync_watch.request()
        return {"ok": True}

    @app.post("/api/sync/fetch")
    def sync_fetch(choice: ModelNames):
        try:
            tools.sync.fetch(choice.names, {"mappings": MAPPING_FILE, "presets": PRESET_FILE})
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.api_route("/api/hub-proxy/{path:path}", methods=["GET", "POST"])
    async def hub_proxy(path: str, request: Request):
        """The Train tab's requests, sent on to the hub when it trains there."""
        if path not in HUB_PROXY:
            refuse("not a Train tab request")
        raw = await request.body()
        try:
            client = hub.client()
            return await asyncio.to_thread(client.request, request.method, f"/api/{path}",
                                           json.loads(raw) if raw else None)
        except hub.HubError as e:
            raise HTTPException(e.status, str(e))
        except ValueError as e:
            refuse(e)

    @app.post("/api/hand")
    def hand(choice: HandChoice):
        if camera is not None and recorder.session is None:
            camera.hand = choice.hand
        return {"hand": camera.hand if camera else None}

    @app.post("/api/tracking")
    def tracking(choice: TrackingChoice):
        if camera is None:
            refuse("started with --no-camera")
        if recorder.session is not None and choice.on != camera.on:
            refuse("not during a session")
        camera.set_on(choice.on)
        return {"on": camera.on}

    @app.websocket("/ws")
    async def live(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json(dict(recorder.snapshot(), tools=tools.snapshot()))
                await asyncio.sleep(1 / SEND_HZ)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/contact")
    async def contact_frames(ws: WebSocket):
        """The contact simulation's geom poses and readout out, the hand pose the page sets in."""
        await ws.accept()

        async def receive():
            while True:
                msg = await ws.receive_json()
                hand = msg.get("hand") if isinstance(msg, dict) else None
                try:
                    pos, quat = [float(v) for v in hand["pos"]], [float(v) for v in hand["quat"]]
                except (TypeError, KeyError, ValueError):
                    continue
                if len(pos) == 3 and len(quat) == 4 and all(np.isfinite(pos + quat)) and any(quat):
                    tools.contact.set_hand(clip_workspace(pos), quat)

        task = asyncio.create_task(receive())
        try:
            while not task.done():
                await ws.send_json(tools.contact.frame())
                await asyncio.sleep(1 / CONTACT_HZ)
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()

    # Train in VR: the quick calibration the headset started on the session it just recorded, as the Record tab's end
    # card runs it. seen: the job was seen running, so its result is this calibration's
    headset_cal = {"session": None, "job": None, "seen": False, "started": 0.0, "error": ""}

    def headset_device():
        """The headset has no start page: it takes this computer's setup from the newest session recorded here, or
        the only setup this computer has."""
        here = [x for x in devices.setups() if devices.here(x)]
        for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
            try:
                used = (json.loads(path.read_text()).get("setup") or {}).get("id")
            except (OSError, ValueError):
                continue
            if any(x["id"] == used for x in here):
                select_device(DeviceChoice(id=used))
                return
        if len(here) == 1:
            select_device(DeviceChoice(id=here[0]["id"]))
        else:
            raise ValueError("pick the device on the start page first")

    def headset_train(action, name=""):
        """calibrate: the newest base network on the last quick session. control: the control layer on name, one of
        headset_calibrations, or else the network that calibration saved. Errors go to the headset's Train
        section."""
        headset_cal["error"] = ""
        try:
            if action == "calibrate":
                session = headset_cal_session(recorder.last_summary)
                if session is None:
                    raise ValueError("record a quick calibration session first")
                reply = quick_calibration(QuickCalibration(session=session))
                headset_cal.update(session=session, job=reply["job"], seen=False, started=time.monotonic())
            elif action == "control":
                saved = name if name in headset_calibrations() else                     headset_train_state(headset_cal, recorder.last_summary, tools)["saved"]
                if not saved:
                    raise ValueError("calibrate first")
                control_start(ModelChoice(name=saved))
        except HTTPException as e:
            headset_cal["error"] = str(e.detail)
        except ValueError as e:
            headset_cal["error"] = str(e)

    @app.websocket("/ws/quest")
    async def quest(ws: WebSocket):
        """The Quest app over USB (quest_link): the control layer's output and the contact settings out, settings
        changed in the headset and its grip force in."""
        import sim
        await ws.accept()

        async def receive():
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    continue
                if isinstance(msg.get("params"), dict):
                    try:
                        changes = ContactParams(**msg["params"]).model_dump(exclude_none=True)
                    except ValueError:
                        continue
                    if changes.get("object", sim.SIM_DEFAULTS["object"]) in sim.OBJECTS:
                        tools.contact.set_params(changes)
                if isinstance(msg.get("scene"), dict):
                    tools.contact.quest_scene = msg["scene"]
                if isinstance(msg.get("view"), dict):
                    tools.contact.quest_view = msg["view"]
                    quest_video.log_hands(msg["view"])
                record = msg.get("record")
                if isinstance(record, dict):
                    action = record.get("action")
                    try:
                        if action == "start":
                            if tools.calibration.state["running"]:
                                raise ValueError("wait for the calibration to finish")
                            if recorder.setup is None:
                                headset_device()
                            recorder.start_session(headset_session_settings(bool(record.get("quick"))))
                        elif action in ("continue", "pause", "bad", "stop"):
                            recorder.command(action)
                        record_error[0] = ""
                    except (ValueError, HTTPException) as e:
                        record_error[0] = str(getattr(e, "detail", e))
                train = msg.get("train")
                if isinstance(train, dict) and train.get("action") in ("calibrate", "control"):
                    await asyncio.to_thread(headset_train, train["action"], str(train.get("name") or ""))
                readout = msg.get("readout")
                if isinstance(readout, dict) and isinstance(readout.get("grip_force"), (int, float))                         and np.isfinite(readout["grip_force"]):
                    tools.contact.set_quest_readout(readout)
                    echo = readout.get("echo")
                    if isinstance(echo, (int, float)) and echo > 0:
                        rtt = time.monotonic() * 1000 - echo
                        if 0 < rtt < 5000:
                            # Smoothed: each value also holds up to one headset send interval (33 ms) of waiting
                            link_rtt[0] = rtt if link_rtt[0] is None else 0.9 * link_rtt[0] + 0.1 * rtt
                            tools.contact.quest_rtt_ms = link_rtt[0]

        record_error = [""]
        link_rtt = [None]
        record_state, record_at = {"running": False}, 0.0
        train_state = dict(headset_train_state(headset_cal, recorder.last_summary, tools),
                           calibrations=headset_calibrations())
        task = asyncio.create_task(receive())
        try:
            while not task.done():
                frame = tools.contact.quest_frame()
                # The recorder's snapshot is more than the headset needs 60 times a second
                if time.monotonic() - record_at > 1 / QUEST_RECORD_HZ:
                    record_state, record_at = headset_record_state(recorder), time.monotonic()
                    train_state = dict(headset_train_state(headset_cal, recorder.last_summary, tools),
                                       calibrations=headset_calibrations())
                frame["record"] = dict(record_state, error=record_error[0])
                frame["haptics"] = headset_haptics_state(tools.haptics, tools.contact.test is not None)
                frame["train"] = train_state
                frame["sent"] = round(time.monotonic() * 1000, 1)
                await ws.send_json(frame)
                await asyncio.sleep(1 / QUEST_HZ)
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            tools.contact.quest_gone()

    @app.websocket("/ws/vr")
    async def vr_frames(ws: WebSocket):
        """The VR tab: the headset's scene when it changes, then what it shows (poses, head, readout) at
        CONTACT_HZ."""
        await ws.accept()
        sent = None
        try:
            while True:
                contact = tools.contact
                scene = contact.quest_scene
                if scene is not None and scene is not sent:
                    await ws.send_json({"scene": scene})
                    sent = scene
                await ws.send_json({"view": contact.quest_view, "readout": contact.quest_readout(),
                                    "params": contact.quest_frame()["params"],
                                    "headset": quest_link.LINK.headset if quest_link.LINK else None,
                                    "rtt_ms": contact.quest_rtt_ms if contact.quest_view is not None else None,
                                    "keep_video": bool(quest_video.SESSION_VIDEO and quest_video.SESSION_VIDEO.on)})
                await asyncio.sleep(1 / CONTACT_HZ)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/vr-video")
    async def vr_video(ws: WebSocket):
        """The VR tab's Headset video: the Quest's screen over USB (quest_video's shared capture), a text message with
        the codec when it starts or changes, then one binary message per frame: a byte that is 1 for a key frame,
        then the frame."""
        await ws.accept()
        video = quest_video.capture()
        q = video.subscribe()
        try:
            while True:
                try:
                    msg = await asyncio.to_thread(q.get, timeout=1.0)
                except queue.Empty:
                    continue
                if msg[0] == "codec":
                    await ws.send_json({"codec": msg[1]})
                elif msg[0] == "frame":
                    await ws.send_bytes(bytes([1 if msg[1] else 0]) + msg[2])
                elif msg[0] == "error":
                    await ws.send_json({"error": msg[1]})
                else:
                    # Fell behind: closing makes the page connect again, with a key frame
                    await ws.close()
                    return
        except WebSocketDisconnect:
            pass
        finally:
            video.unsubscribe(q)

    @app.websocket("/ws/signals")
    async def signals(ws: WebSocket):
        """New raw and filtered EMG samples as they arrive, for the signal view."""
        await ws.accept()
        count = max(0, recorder.received - recorder.history)
        try:
            while True:
                count, raw, filt = recorder.samples_since(count)
                if raw is not None:
                    await ws.send_json({"raw": np.round(raw, 1).tolist(), "filtered": np.round(filt, 1).tolist()})
                await asyncio.sleep(1 / SEND_HZ)
        except WebSocketDisconnect:
            pass

    @app.get("/video.mjpg")
    async def video():
        if camera is None:
            raise HTTPException(404, "no camera")

        async def frames():
            while True:
                jpeg = camera.jpeg()
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / PREVIEW_HZ)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
    return app


def free_space(tools):
    """On a device as its app closes: deletes the models the hub has (tools.Sync.free_models), then the sessions the
    hub and its backup both hold that no model left here needs (tools.Sync.free_space). Never mid-use, so nothing
    can disappear from a page that shows it."""
    print("checking which models and sessions the hub has...")
    try:
        lines = tools.sync.free_models()
        lines += tools.sync.free_space({levels_session(m) for m in model_list()} - {None})
    except Exception as e:
        lines = [f"kept every model and session here, the hub couldn't be asked: {e}"]
    for line in lines:
        print(line)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    source.add_argument("--replay", metavar="PATH", help="play back raw .npy recordings instead of the armband")
    cam = parser.add_mutually_exclusive_group()
    cam.add_argument("--camera", default="0", help="camera index or a video file (default 0)")
    cam.add_argument("--no-camera", action="store_true")
    parser.add_argument("--hand", choices=["right", "left"], default="right", help="hand the camera tracks")
    parser.add_argument("--host", help="address to listen on (default 127.0.0.1, 0.0.0.0 with --hub)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hub", action="store_true", help="let devices with a token send sessions and train here")
    parser.add_argument("--no-quest", action="store_true", help="don't link a Quest plugged in over USB")
    args = parser.parse_args()
    # Only an explicit --host opens the page to the network. --hub alone listens everywhere, for the devices
    lan = args.host is not None and args.host not in LOOPBACK + ("localhost",)
    host = args.host or ("0.0.0.0" if args.hub else "127.0.0.1")

    band = Replay(args.replay) if args.replay else Armband(args.synthetic, optional=True)
    camera = None
    if not args.no_camera and not args.hub:
        camera = Camera(int(args.camera) if args.camera.isdigit() else args.camera, args.hand).start()
    # A reader connected later from the Connection tab ends with the app, its output pipe closes
    with contextlib.nullcontext(band) if args.hub else band:
        recorder = Recorder(band, camera, args.hub)
        recorder.start()
        tools = Tools(recorder)
        print(f"EMG from {band.source}, open http://localhost:{args.port}")
        quest = None if args.no_quest or args.hub else quest_link.start(args.port)
        if quest is not None:
            quest_video.SESSION_VIDEO = quest_video.SessionVideo(recorder, lambda: tools.contact.quest_view is not None)
            quest_video.SESSION_VIDEO.start()
        if args.hub:
            print(f"hub for {', '.join(hub.devices()) or 'no devices yet (python hub.py add-device NAME)'}, "
                  f"at http://{socket.gethostname()}:{args.port}")
        try:
            # An open preview stream never ends by itself, without the timeout Ctrl+C waits on it forever and a second
            # Ctrl+C can cut off the cleanup below
            uvicorn.run(make_app(recorder, band, camera, tools, lan, args.hub), host=host, port=args.port,
                        log_level="warning", timeout_graceful_shutdown=2)
        finally:
            # Leaving the haptics thread running would leave the motors on until the ESP32 watchdog fires
            tools.haptics.stop()
            tools.haptics.join(timeout=3)
            if quest is not None:
                quest.stop()
            recorder.stop()
            if camera is not None:
                camera.stop()
            # A test copy on another port closes while the main app may still show those sessions
            if not args.hub and hub.load().get("url") and args.port == parser.get_default("port"):
                free_space(tools)
            if band.source == "armband" and args.port == parser.get_default("port"):
                try:
                    for line in forget_armband_wifi():
                        print(f"armband WiFi: {line}")
                except Exception as e:
                    print(f"armband WiFi not forgotten: {e}")


if __name__ == "__main__":
    main()
