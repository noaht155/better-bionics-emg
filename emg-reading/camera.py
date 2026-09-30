"""Webcam capture and hand tracking (MediaPipe hand landmarker) in a background thread.

Each frame is stamped with time.time() as soon as it is read, the same clock as the EMG. What is left is the
camera's own delay, measured afterwards from the sync taps in each session.
"""
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from hand_angles import JOINTS, joint_angles

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")
MODEL_FILE = Path(__file__).with_name("models") / "hand_landmarker.task"

WIDTH, HEIGHT, FPS = 640, 480, 30
JPEG_QUALITY = 70
# Only encode preview images while a browser is watching
PREVIEW_TIMEOUT_S = 2.0


def model_path():
    if not MODEL_FILE.exists():
        print(f"downloading the hand model to {MODEL_FILE}")
        MODEL_FILE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    return str(MODEL_FILE)


def open_capture(source):
    """source: camera index or a video file (for testing)."""
    if isinstance(source, int) and sys.platform == "win32":
        # The default Windows backend takes seconds to open a webcam, DirectShow is quick
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        # Keep only the newest frame, an older buffered one would add delay
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Camera:
    def __init__(self, source=0, hand="right"):
        self.source = source
        self.hand = hand
        self.on_frame = None
        self.ok = False
        self.error = None
        self.fps = 0.0
        self.size = None
        self.latest = None
        self._jpeg = None
        self._preview_until = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def jpeg(self):
        """Newest frame as JPEG for the preview, or None."""
        self._preview_until = time.time() + PREVIEW_TIMEOUT_S
        with self._lock:
            return self._jpeg

    def _run(self):
        try:
            options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path()),
                                                   running_mode=vision.RunningMode.VIDEO, num_hands=2)
            landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            self.error = f"hand model failed to load: {e}"
            return
        cap = open_capture(self.source)
        if not cap.isOpened():
            self.error = f"camera {self.source} not found"
            return
        from_file = not isinstance(self.source, int)
        file_period = 1 / (cap.get(cv2.CAP_PROP_FPS) or FPS)
        self.ok = True
        index = 0
        last_ms = -1
        rate_t = time.time()
        rate_n = 0
        try:
            while not self._stop.is_set():
                got, frame = cap.read()
                t = time.time()
                if not got:
                    if from_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.ok = False
                    self.error = "camera stopped sending frames"
                    break
                self.size = (frame.shape[1], frame.shape[0])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # The landmarker needs strictly increasing timestamps
                ms = max(int(t * 1000), last_ms + 1)
                last_ms = ms
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
                track = self._pick_hand(result, t, index)
                with self._lock:
                    self.latest = track
                if self.on_frame is not None:
                    self.on_frame(track)
                if time.time() < self._preview_until:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        with self._lock:
                            self._jpeg = buf.tobytes()
                index += 1
                rate_n += 1
                if t - rate_t >= 1.0:
                    self.fps = rate_n / (t - rate_t)
                    rate_t, rate_n = t, 0
                if from_file:
                    time.sleep(max(0.0, file_period - (time.time() - t)))
        finally:
            self.ok = False
            cap.release()
            landmarker.close()

    def _pick_hand(self, result, t, index):
        """The tracked hand's landmarks and angles as one record of session.CAMERA_COLUMNS."""
        record = np.full(4 + 63 + 63 + len(JOINTS), np.nan)
        record[:4] = [t, index, 0, 0]
        for handed, img, world in zip(result.handedness, result.hand_landmarks, result.hand_world_landmarks):
            side = handed[0].category_name.lower()
            if side != self.hand:
                continue
            world = np.array([[p.x, p.y, p.z] for p in world])
            record[2:4] = [1, handed[0].score]
            record[4:67] = np.array([[p.x, p.y, p.z] for p in img]).ravel()
            record[67:130] = world.ravel()
            record[130:] = joint_angles(world, side == "right")
            break
        return record
