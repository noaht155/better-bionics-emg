"""Recording web app, also runs the existing tools (connection check, live signals, calibration, haptics).
Serves the page in web/ and streams the live state to it.

    python app.py [--synthetic | --replay data/2026-09-30-filter] [--camera 0 | --camera clip.mp4 | --no-camera]
                  [--host 0.0.0.0] [--port 8000]

Then open http://localhost:8000. --host 0.0.0.0 makes it reachable from a tablet on the same network.
"""
import argparse
import asyncio
import json
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
import gesture_model
from camera import Camera
from esp32_link import WIFI_IP
from gestures import GESTURES, POSTURES
from hand_angles import JOINTS, RELIABLE
from processing import spectrum
from recorder import Recorder
from session import DATA_DIR
from tools import Tools

WEB_DIR = Path(__file__).with_name("web")
SEND_HZ = 20
PREVIEW_HZ = 15


class SessionSettings(BaseModel):
    subject: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    band_arm: Literal["right", "left"]
    tracked_hand: Literal["right", "left"]
    placement: str = ""
    notes: str = ""
    # None gives a short session of only the sync taps, to check the camera delay
    postures: list[Literal[tuple(POSTURES)]]
    reps: int = Field(ge=1, le=20)
    hold_s: float = Field(ge=1, le=30)
    rest_s: float = Field(ge=1, le=30)
    free_s: float = Field(ge=0, le=600)


class HandChoice(BaseModel):
    hand: Literal["right", "left"]


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
    vote: int = Field(ge=1, le=15)


class ModelChoice(BaseModel):
    name: str


def session_list():
    """Recorded sessions, newest first, from their session.json only."""
    out = []
    for path in sorted(DATA_DIR.glob("*/session.json"), reverse=True):
        meta = json.loads(path.read_text())
        settings = meta.get("settings", {})
        gestures = sum(c["kind"] == "hold" and c["label"] != "rest" for c in meta.get("plan", []))
        out.append({"name": path.parent.name, "subject": meta.get("subject"), "postures": settings.get("postures"),
                    "band_arm": settings.get("band_arm"), "placement": settings.get("placement"),
                    "gestures": gestures, "completed": meta.get("completed"),
                    "seconds": round(meta["ended"] - meta["started"]) if meta.get("ended") else None,
                    "camera_delay_s": meta.get("camera_delay_s")})
    return out


def model_list():
    out = []
    for path in sorted(gesture_model.MODEL_DIR.glob("*.joblib"), reverse=True):
        model = joblib.load(path)
        loso = (model.get("evaluation") or {}).get("loso", {}).get(model["vote"])
        out.append({"name": path.name, "subject": model["subject"], "trained": model["trained"],
                    "sessions": model["sessions"], "classes": model["classes"], "options": model["options"],
                    "accuracy": loso["accuracy"] if loso else None,
                    "rest_false": loso["rest_false"] if loso else None})
    return out


def refuse(e):
    raise HTTPException(409, str(e))


def make_app(recorder, band, camera, tools):
    app = FastAPI()

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
                "postures": POSTURES, "camera": camera is not None,
                "tracked_hand": camera.hand if camera else None, "esp32_wifi_ip": WIFI_IP}

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
                                {"log": settings.log, "accel": settings.accel, "vote": settings.vote})
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.get("/api/models")
    def models():
        return model_list()

    @app.post("/api/model/use")
    def use_model(choice: ModelChoice):
        path = gesture_model.MODEL_DIR / choice.name
        if path.parent != gesture_model.MODEL_DIR or not path.exists():
            refuse("unknown model")
        try:
            tools.predictor.use(path)
        except ValueError as e:
            refuse(e)
        return {"ok": True}

    @app.post("/api/model/stop")
    def stop_model():
        tools.predictor.stop()
        return {"ok": True}

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
