"""Recording web app. Serves the page in web/ and streams the live state to it.

    python app.py [--synthetic | --replay data/2026-09-30-filter] [--camera 0 | --camera clip.mp4 | --no-camera]
                  [--host 0.0.0.0] [--port 8000]

Then open http://localhost:8000. --host 0.0.0.0 makes it reachable from a tablet on the same network.
"""
import argparse
import asyncio
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from armband import Armband, Replay
from calibrate import load_calibration
from camera import Camera
from gestures import GESTURES, POSTURES
from hand_angles import JOINTS, RELIABLE
from recorder import Recorder

WEB_DIR = Path(__file__).with_name("web")
SEND_HZ = 20
PREVIEW_HZ = 15


class SessionSettings(BaseModel):
    subject: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    band_arm: Literal["right", "left"]
    tracked_hand: Literal["right", "left"]
    placement: str = ""
    notes: str = ""
    postures: list[Literal[tuple(POSTURES)]] = Field(min_length=1)
    reps: int = Field(ge=1, le=20)
    hold_s: float = Field(ge=1, le=30)
    rest_s: float = Field(ge=1, le=30)
    free_s: float = Field(ge=0, le=600)


class HandChoice(BaseModel):
    hand: Literal["right", "left"]


def make_app(recorder, band, camera):
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
                "tracked_hand": camera.hand if camera else None}

    @app.post("/api/session/start")
    def start(settings: SessionSettings):
        try:
            return {"folder": recorder.start_session(settings.model_dump())}
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/session/{action}")
    def command(action: Literal["continue", "pause", "bad", "stop"]):
        try:
            recorder.command(action)
        except ValueError as e:
            raise HTTPException(409, str(e))
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
                await ws.send_json(recorder.snapshot())
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
        print(f"EMG from {band.source}, open http://localhost:{args.port}")
        try:
            uvicorn.run(make_app(recorder, band, camera), host=args.host, port=args.port, log_level="warning")
        finally:
            recorder.stop()
            if camera is not None:
                camera.stop()


if __name__ == "__main__":
    main()
