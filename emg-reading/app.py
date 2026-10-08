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
from esp32_link import WIFI_IP
from gestures import GESTURE_SET, GESTURES, POSTURES, plan_for
from hand_angles import JOINTS, RELIABLE
from processing import spectrum
from recorder import Recorder
from session import DATA_DIR
from tools import Tools

# The ml package sits next to emg-reading/ in the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml import NETWORK_DIR, dataset, gesture_model  # noqa: E402

WEB_DIR = Path(__file__).with_name("web")
SEND_HZ = 20
PREVIEW_HZ = 15


class SessionSettings(BaseModel):
    subject: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    band_arm: Literal["right", "left"]
    # Which way round the band is. Turned the other way the electrode order round the arm is mirrored
    port_facing: Literal["shoulder", "hand"]
    tracked_hand: Literal["right", "left"]
    placement: str = ""
    notes: str = ""
    postures: list[Literal[tuple(POSTURES)]] = Field(min_length=1)
    # A: held grips and single fingers. B: the Ninapro-style movements (gestures.protocol_b_plan)
    protocol: Literal["A", "B"] = "A"
    reps: int = Field(ge=1, le=20)
    hold_s: float = Field(ge=1, le=30)
    rest_s: float = Field(ge=1, le=30)
    free_s: float = Field(0, ge=0, le=600)
    rounds: int = Field(6, ge=1, le=20)
    move_s: float = Field(5, ge=2, le=15)
    # Same run, nothing saved, for trying out the setup
    practice: bool = False


class HandChoice(BaseModel):
    hand: Literal["right", "left"]


class CalibrationSettings(BaseModel):
    seconds: float = Field(ge=2, le=30)
    car: bool


class Esp32Link(BaseModel):
    port: str | None = None
    wifi: str | None = None


class PlanSettings(BaseModel):
    postures: list[Literal[tuple(POSTURES)]]
    # A: held grips and single fingers. B: the Ninapro-style movements (gestures.protocol_b_plan)
    protocol: Literal["A", "B"] = "A"
    reps: int = Field(ge=1, le=20)
    hold_s: float = Field(ge=1, le=30)
    rest_s: float = Field(ge=1, le=30)
    free_s: float = Field(0, ge=0, le=600)
    rounds: int = Field(6, ge=1, le=20)
    move_s: float = Field(5, ge=2, le=15)


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


class NetworkJobSettings(BaseModel):
    job: Literal["evaluate", "evaluate_newest", "train", "calibrate"]
    epochs: int = Field(15, ge=1, le=200)
    network: str | None = None
    session: str | None = None
    reps: list[int] = Field([1, 2], min_length=1)
    # Sessions to train a network on, None for all good sessions
    sessions: list[str] | None = None


def session_list():
    """Recorded sessions, newest first, from their session.json only."""
    out = []
    for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
        meta = json.loads(path.read_text())
        settings = meta.get("settings", {})
        gestures = sum(c["kind"] == "hold" and c["label"] != "rest" for c in meta.get("plan", []))
        out.append({"name": path.parent.name, "subject": meta.get("subject"), "postures": settings.get("postures"),
                    "band_arm": settings.get("band_arm"), "port_facing": settings.get("port_facing"),
                    "placement": settings.get("placement"),
                    "gestures": gestures, "completed": meta.get("completed"),
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


def refuse(e):
    raise HTTPException(409, str(e))


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
                "gesture_set": GESTURE_SET,
                "postures": POSTURES, "camera": camera is not None,
                "tracked_hand": camera.hand if camera else None, "esp32_wifi_ip": WIFI_IP}

    @app.post("/api/plan")
    def plan_length(settings: PlanSettings):
        """Length of the session these settings give, from the real plan so the page needn't copy its logic. The
        plan itself is for the cue preview; grip order is shuffled again when the session starts."""
        plan = plan_for(settings.model_dump())
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
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/check")
    def check():
        try:
            tools.check.start()
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
        folders = [DATA_DIR / name for name in settings.sessions]
        if any(not (f / "session.json").exists() for f in folders):
            refuse("unknown session")
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
            session = settings.session or ""
            if Path(session).name != session or not (DATA_DIR / session / "session.json").exists():
                refuse("unknown session")
            args = ["ml.networks", "calibrate", name, str(DATA_DIR / session),
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
        folder = NETWORK_DIR if choice.name.endswith(".pt") else gesture_model.MODEL_DIR
        path = folder / choice.name
        if path.parent != folder or not path.exists():
            refuse("unknown model")
        try:
            model = tools.predictor.use(path)
        except ValueError as e:
            refuse(e)
        return {"ok": True, "haptics": start_model_haptics(model, choice)}

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

    def start_model_haptics(model, link):
        """Haptics run with every prediction model, with levels from the session the model was calibrated on (a
        calibrated network) or its newest training session (LDA). Returns what happened, for the page. The model
        keeps running when the haptics can't start."""
        if model.get("kind") == "network":
            session = model["info"].get("calibrated_on")
        else:
            session = (model.get("sessions") or [None])[-1]
        if not session or not (DATA_DIR / session / "session.json").exists():
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

    band = Replay(args.replay) if args.replay else Armband(args.synthetic)
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
