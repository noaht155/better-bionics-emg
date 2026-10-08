"""Camera frame times on the EMG clock: the straight-line fit and the delay correction in dataset.camera_times."""
import numpy as np

import ml  # noqa: F401
from ml import dataset

FPS = 30


def frames(seconds=60, delay=0.12, seed=0):
    """Frames taken at a steady 30 fps, arriving on the PC late by the camera delay plus up to a frame of jitter,
    in pairs (two close together, then a gap), like Media Foundation."""
    rng = np.random.default_rng(seed)
    frame = np.arange(seconds * FPS)
    taken = 1000.0 + frame / FPS
    paired = np.where(frame % 2 == 0, 1 / FPS, 0.002)
    arrival = taken + delay + paired + rng.uniform(0, 0.008, len(frame))
    return {"t": arrival, "frame": frame}, taken


def test_fit_removes_jitter_and_delay():
    cam, taken = frames(delay=dataset.CAMERA_DELAY_S)
    t = dataset.camera_times(cam, {})
    err = t - taken
    # Within a few ms of when the frame was taken, no frame-to-frame jitter left
    assert np.abs(err).max() < 0.01
    assert np.std(np.diff(t)) < 1e-4


def test_tap_delay_in_session_is_ignored():
    """The taps were biased (96 to 168 ms against 20 to 37 ms measured directly), a stored value must not be used."""
    cam, _ = frames(delay=dataset.CAMERA_DELAY_S)
    assert np.array_equal(dataset.camera_times(cam, {}), dataset.camera_times(cam, {"camera_delay_s": 0.15}))


def test_wrong_delay_shifts_everything():
    """The fit can only remove jitter, the delay itself has to be right: a different real delay moves every label."""
    cam, taken = frames(delay=0.12)
    t = dataset.camera_times(cam, {})
    assert np.median(t - taken) > 0.06
