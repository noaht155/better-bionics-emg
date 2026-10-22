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
import importlib.util
import json
import socket
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import devices
import hub
from armband import Armband, Replay
from calibrate import load_calibration
from camera import Camera
from control import DEFAULTS as CONTROL_DEFAULTS, check_rules, rules_for, split_rules
from esp32_link import WIFI_IP
from gestures import (C_CLOSE_S, C_MOVE_S, C_ROUNDS, CATEGORIES, D_ACTIVITY_S, D_ROUNDS, E_ROUNDS, E_SIGN_S,
                      F_ROUNDS, F_SIGN_S, F_SIGNS, GESTURE_SET, GESTURES, GRIP_CLOSE_S, LIBRARY, POSTURES, plan_for)
from hand_angles import JOINTS, RELIABLE
from processing import spectrum
from recorder import Recorder
from session import DATA_DIR
from tools import POSES, Tools, model_path

# The ml package sits next to emg-reading/ in the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml import NETWORK_DIR, dataset, gesture_model  # noqa: E402

WEB_DIR = Path(__file__).with_name("web")
# Saved control parameter sets, next to the models. "default" is always the spec's values and isn't stored
PRESET_FILE = gesture_model.MODEL_DIR.parent / "control_presets.json"
# Saved mappings (rule lists) of the Control tab, apart from the profiles (presets) that also carry one
MAPPING_FILE = gesture_model.MODEL_DIR.parent / "control_mappings.json"
SEND_HZ = 20
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
                    "completed": meta.get("completed"),
                    "seconds": round(meta["ended"] - meta["started"]) if meta.get("ended") else None,
                    "protocol": settings.get("protocol", "A")})
    return out


def model_list():
    out = []
    for path in sorted(gesture_model.MODEL_DIR.glob("*.joblib"), reverse=True):
        model = joblib.load(path)
        loso = (model.get("evaluation") or {}).get("loso", {}).get(model["vote"])
        out.append({"name": path.name, "subject": model["subject"], "trained": model["trained"],
                    "sessions": model["sessions"], "classes": model["classes"], "options": model["options"],
                    "accuracy": loso["accuracy"] if loso else None,
                    # Models trained before balanced accuracy was added don't have it
                    "balanced": loso.get("balanced") if loso else None,
                    "rest_false": loso["rest_false"] if loso else None})
    return out


def network_list():
    """Saved networks, newest first. Empty where torch isn't installed (it's only on the training desktop)."""
    try:
        import torch
    except ImportError:
        return []
    out = []
    for path in sorted(NETWORK_DIR.glob("*.pt"), reverse=True):
        blob = torch.load(path, map_location="cpu", weights_only=True)
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


def refuse(e):
    raise HTTPException(409, str(e))


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
                "gesture_set": GESTURE_SET, "sign_sets": {"F": F_SIGNS}, "library": LIBRARY, "categories": CATEGORIES,
                "cue_timing": {"grip_close_s": GRIP_CLOSE_S, "c_close_s": C_CLOSE_S},
                "postures": POSTURES, "camera": camera is not None,
                "tracked_hand": camera.hand if camera else None, "esp32_wifi_ip": WIFI_IP, "platform": sys.platform,
                "fixed_delay_s": dataset.CAMERA_DELAY_S}

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
    def command(action: Literal["continue", "pause", "bad", "stop"]):
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
        good = [f.name for f in dataset.good_sessions()]
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
                signs = [s["name"] for s in session_list() if s["completed"] and s["signs"]]
                if not settings.sessions or any(n not in signs for n in settings.sessions):
                    refuse("pick at least one completed sign session")
                args = ["ml.networks", "signs", name, "--epochs", str(settings.epochs), "--sessions", *settings.sessions]
                if settings.exclude:
                    args += ["--exclude", *settings.exclude]
            else:
                args = ["ml.networks", "calibrate", name, str(session_folder(settings.session or "")),
                        "--reps", *map(str, settings.reps)]
        try:
            tools.network.start(settings.job, args)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

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
        return {"ok": True, "haptics": start_model_haptics(model, choice)}

    @app.post("/api/control/simulate")
    def control_simulate(classes: list[str]):
        """The control layer without a model or the band: the page plays the model (class, effort, arm moving)."""
        if len(classes) < 2:
            refuse("the simulated model needs at least two classes")
        tools.control.prepare_sim()
        try:
            tools.predictor.simulate(classes, tools.control)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/control/sim")
    def control_sim(inputs: SimInputs):
        try:
            tools.predictor.set_sim(inputs.model_dump())
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/control/params")
    def control_params(params: ControlParams):
        tools.control.set_params(params.model_dump())
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

    def start_model_haptics(model, link):
        """Haptics run with every prediction model, with levels from model_session. Returns what happened, for the
        page. The model keeps running when the haptics can't start."""
        session = model_session(model)
        if session is None:
            return "haptics not started: the model's calibration session isn't on this computer"
        tools.haptics.stop()
        tools.haptics.join(timeout=3)
        try:
            tools.haptics.start(link.port, link.wifi, DATA_DIR / session)
        except ValueError as e:
            return f"haptics not started: {e}"
        tools.haptics.state["with_model"] = True
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
    """On a device as its app closes: deletes the sessions the hub and its backup both hold (tools.Sync.free_space).
    Never mid-use, so a session can't disappear from a page that shows it."""
    print("checking which sessions the hub has backed up...")
    try:
        lines = tools.sync.free_space({levels_session(m) for m in model_list()} - {None})
    except Exception as e:
        lines = [f"kept every session here, the hub couldn't be asked: {e}"]
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
        if args.hub:
            print(f"hub for {', '.join(hub.devices()) or 'no devices yet (python hub.py add-device NAME)'}, "
                  f"at http://{socket.gethostname()}:{args.port}")
        try:
            uvicorn.run(make_app(recorder, band, camera, tools, lan, args.hub), host=host, port=args.port,
                        log_level="warning")
        finally:
            # Leaving the haptics thread running would leave the motors on until the ESP32 watchdog fires
            tools.haptics.stop()
            tools.haptics.join(timeout=3)
            recorder.stop()
            if camera is not None:
                camera.stop()
            # A test copy on another port closes while the main app may still show those sessions
            if not args.hub and hub.load().get("url") and args.port == parser.get_default("port"):
                free_space(tools)


if __name__ == "__main__":
    main()
