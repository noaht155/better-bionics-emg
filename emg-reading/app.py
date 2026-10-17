"""Recording web app, also runs the existing tools (connection check, live signals, calibration, haptics).
Serves the page in web/ and streams the live state to it.

    python app.py [--synthetic | --replay data/2026-09-30-filter] [--camera 0 | --camera clip.mp4 | --no-camera]
                  [--host 0.0.0.0] [--port 8000]

Then open http://localhost:8000. --host 0.0.0.0 makes it reachable from a tablet on the same network.
"""
import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from armband import Armband, Replay
from calibrate import load_calibration
from camera import Camera
from control import DEFAULTS as CONTROL_DEFAULTS
from esp32_link import WIFI_IP
from gestures import (C_CLOSE_S, C_MOVE_S, C_ROUNDS, CATEGORIES, D_ACTIVITY_S, D_ROUNDS, E_ROUNDS, E_SIGN_S,
                      F_ROUNDS, F_SIGN_S, GESTURE_SET, GESTURES, GRIP_CLOSE_S, LIBRARY, POSTURES, plan_for)
from hand_angles import JOINTS, RELIABLE
from processing import spectrum
from recorder import Recorder
from session import DATA_DIR
from tools import Tools

# The ml package sits next to emg-reading/ in the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml import NETWORK_DIR, dataset, gesture_model  # noqa: E402

WEB_DIR = Path(__file__).with_name("web")
# Saved control parameter sets, next to the models. "default" is always the spec's values and isn't stored
PRESET_FILE = gesture_model.MODEL_DIR.parent / "control_presets.json"
SEND_HZ = 20
PREVIEW_HZ = 15


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


class HandChoice(BaseModel):
    hand: Literal["right", "left"]


class WifiSettings(BaseModel):
    # How long the adapter stays on DHCP, 0 skips it (connect_armband.ps1)
    dhcp_s: int = Field(5, ge=0, le=60)


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


class ModelChoice(BaseModel):
    name: str
    # ESP32 link for the haptics that start with the model
    port: str | None = None
    wifi: str | None = None


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
    mode: Literal["lock", "blend", "trigger"] = CONTROL_DEFAULTS["mode"]
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


class ControlPreset(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9 _-]{1,40}$")
    params: ControlParams


class NetworkJobSettings(BaseModel):
    job: Literal["evaluate", "evaluate_newest", "train", "calibrate"]
    epochs: int = Field(15, ge=1, le=200)
    network: str | None = None
    session: str | None = None
    reps: list[int] = Field([1, 2], min_length=1)
    # Sessions to train a network on, None for all good sessions
    sessions: list[str] | None = None


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


def session_list():
    """Recorded sessions, newest first, from their session.json only."""
    out = []
    for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
        meta = json.loads(path.read_text())
        settings = meta.get("settings", {})
        gestures = sum(c["kind"] == "hold" and c["label"] != "rest" for c in meta.get("plan", []))
        signs = sum(c["kind"] == "custom" for c in meta.get("plan", [])) if settings.get("protocol") in ("E", "F") else 0
        out.append({"name": path.parent.name, "subject": meta.get("subject"), "postures": settings.get("postures"),
                    "band_arm": settings.get("band_arm"), "port_facing": settings.get("port_facing"),
                    "placement": placement_text(settings),
                    "gestures": gestures, "signs": signs, "completed": meta.get("completed"),
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
        info = torch.load(path, map_location="cpu", weights_only=True)["info"]
        out.append(dict(info, name=path.name))
    return out


def saved_presets():
    return json.loads(PRESET_FILE.read_text()) if PRESET_FILE.exists() else {}


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


def make_app(recorder, band, camera, tools):
    app = FastAPI()

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

    recorder.on_saved = auto_evaluate

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
                "gesture_set": GESTURE_SET, "library": LIBRARY, "categories": CATEGORIES,
                "cue_timing": {"grip_close_s": GRIP_CLOSE_S, "c_close_s": C_CLOSE_S},
                "postures": POSTURES, "camera": camera is not None,
                "tracked_hand": camera.hand if camera else None, "esp32_wifi_ip": WIFI_IP}

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
    def train(settings: TrainSettings):
        folders = [session_folder(name) for name in settings.sessions]
        try:
            tools.trainer.start([str(f) for f in folders],
                                {"log": settings.log, "accel": settings.accel, "vote": settings.vote,
                                 "extended": settings.extended})
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
                if not info.get("calibrated_on"):
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

    @app.post("/api/control/params")
    def control_params(params: ControlParams):
        tools.control.set_params(params.model_dump())
        return {"ok": True}

    @app.post("/api/control/reset")
    def control_reset():
        tools.control.reset_counts()
        return {"ok": True}

    @app.get("/api/control/presets")
    def control_presets():
        return {"default": CONTROL_DEFAULTS, **saved_presets()}

    @app.post("/api/control/presets")
    def save_control_preset(preset: ControlPreset):
        if preset.name == "default":
            refuse("default is fixed, pick another name")
        presets = saved_presets()
        presets[preset.name] = preset.params.model_dump()
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
        """The session a model's levels come from: the one a calibrated network was calibrated on, or an LDA
        model's newest training session. None if it isn't on this computer."""
        if model.get("kind") == "network":
            session = model["info"].get("calibrated_on")
        else:
            session = (model.get("sessions") or [None])[-1]
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

    @app.post("/api/hand")
    def hand(choice: HandChoice):
        if camera is not None and recorder.session is None:
            camera.hand = choice.hand
        return {"hand": camera.hand if camera else None}

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


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    source.add_argument("--replay", metavar="PATH", help="play back raw .npy recordings instead of the armband")
    cam = parser.add_mutually_exclusive_group()
    cam.add_argument("--camera", default="0", help="camera index or a video file (default 0)")
    cam.add_argument("--no-camera", action="store_true")
    parser.add_argument("--hand", choices=["right", "left"], default="right", help="hand the camera tracks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    band = Replay(args.replay) if args.replay else Armband(args.synthetic, optional=True)
    camera = None
    if not args.no_camera:
        camera = Camera(int(args.camera) if args.camera.isdigit() else args.camera, args.hand).start()
    with band:
        recorder = Recorder(band, camera)
        recorder.start()
        tools = Tools(recorder)
        print(f"EMG from {band.source}, open http://localhost:{args.port}")
        try:
            uvicorn.run(make_app(recorder, band, camera, tools), host=args.host, port=args.port, log_level="warning")
        finally:
            # Leaving the haptics thread running would leave the motors on until the ESP32 watchdog fires
            tools.haptics.stop()
            tools.haptics.join(timeout=3)
            recorder.stop()
            if camera is not None:
                camera.stop()


if __name__ == "__main__":
    main()
