"""Training windows from every good session, for the networks.

Good means completed (not stopped early) with gesture cues, from any user. Each window is the raw output of the
shared filter, 8 channels x WINDOW samples, ending every STEP samples, with what is known about it:
- grip: the cued grip if the window lies in the trimmed part of a hold cue (same rule as the LDA), else None
- rep: which repetition of that grip in that posture (1 to 3), 0 outside held grips
- posture: the posture of the cue the window ends in
- kind: the kind of that cue (hold, finger, free, sync, break), finger and free are the moving fingers
- angles: the 20 joint angles from the camera at the window's last sample, NaN when the hand wasn't tracked

Each session is processed once and cached in ml/cache/ (ignored by git, it holds personal data).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gestures import GESTURE_SET
from channels import mirrored
from hand_angles import JOINTS, RELIABLE
from ml import low_priority
from processing import FILTER_SETTLE_S, FILTER_VERSION, filter_block
from session import DATA_DIR, load_session, segments

RATE = 500
WINDOW = 100
STEP = 25
TRIM_START_S = 1.0
# Camera frames reach the PC 37 ms after they are taken (LED test 2026-10-07) and EMG samples arrive up to 17 ms
# late (armband round trip 2026-10-08), so the true shift is 20 to 37 ms; the middle is off by at most 9 ms. Used for
# every session: the tap delays in session.json (96 to 168 ms) were biased and are no longer read
CAMERA_DELAY_S = 0.028
# A camera frame further than this from a window's end doesn't label it
MAX_FRAME_GAP_S = 0.1
# A frame with a trained joint bent backwards past this is a tracking glitch (MediaPipe lost or flipped the hand,
# index PIP read down to -175), not a hand position. It is treated like a frame where no hand was found. 0.02 to 2.2 %
# of the angle windows per session had one before this (2026-10-06)
IMPOSSIBLE_DEG = -60
CACHE = Path(__file__).with_name("cache")
# Bump when the windows or labels change, old cache files are then rebuilt
VERSION = f"4-f{FILTER_VERSION}-w{WINDOW}-s{STEP}-{'.'.join(GESTURE_SET)}"


def good_sessions(data_dir=DATA_DIR):
    """Completed sessions with gesture cues, oldest first."""
    out = []
    for path in sorted(Path(data_dir).glob("*/session.json")):
        meta = json.loads(path.read_text())
        has_grips = any(c["kind"] == "hold" and c["label"] in GESTURE_SET and c["label"] != "rest"
                        for c in meta.get("plan", []))
        if meta.get("completed") and has_grips:
            out.append(path.parent)
    return out


def camera_times(cam, meta):
    """Frame times on the EMG clock. Frames arrive in pairs from Media Foundation, so the arrival time jitters by
    up to a frame. The camera itself runs at a steady rate and every grabbed frame has a number, so a straight line
    of arrival time against frame number gives clean times. Then the fixed camera delay is taken off."""
    t, frame = cam["t"], cam["frame"]
    if len(t) < 10:
        return t
    a, b = np.polyfit(frame, t, 1)
    fitted = a * frame + b
    # Arrival is never early, only late, so the line is moved down to the earliest arrivals
    fitted += np.percentile(t - fitted, 5)
    return fitted - CAMERA_DELAY_S


def _repetitions(segs):
    rep_of = {}
    seen = defaultdict(int)
    last = 0
    for g in segs:
        if g["index"] in rep_of:
            continue
        if g["label"] != "rest":
            seen[(g["posture"], g["label"])] += 1
            last = seen[(g["posture"], g["label"])]
        rep_of[g["index"]] = last
    return rep_of


def build(folder):
    """All windows of one session that have a grip label or tracked angles."""
    data = load_session(folder)
    meta = data["meta"]
    y = filter_block(data["emg"][meta["channels"]], RATE).astype(np.float32)
    if mirrored(meta.get("settings", {})):
        y = np.ascontiguousarray(y[::-1])
    ends = np.arange(max(WINDOW, int(FILTER_SETTLE_S * RATE)), y.shape[1] + 1, STEP)

    grip = np.full(len(ends), "", dtype=object)
    rep = np.zeros(len(ends), np.int8)
    posture = np.full(len(ends), "", dtype=object)
    kind = np.full(len(ends), "", dtype=object)
    segs = segments(data["events"])
    rep_of = _repetitions([g for g in segs if g["kind"] == "hold" and g["sample1"]])
    trim = int(TRIM_START_S * RATE)
    for g in segs:
        if g["sample1"] is None:
            continue
        inside = (ends > g["sample0"]) & (ends <= g["sample1"])
        posture[inside] = g["posture"]
        kind[inside] = g["kind"]
        if g["kind"] == "hold" and not g["bad"] and g["label"] in GESTURE_SET:
            held = inside & (ends - WINDOW >= g["sample0"] + trim)
            grip[held] = g["label"]
            rep[held] = rep_of[g["index"]]

    angles = np.full((len(ends), len(JOINTS)), np.nan, np.float32)
    cam = data["camera"]
    if len(cam["t"]):
        t_cam = camera_times(cam, meta)
        tracked = cam["detected"] == 1
        tracked &= ~np.any(np.stack([cam[j] for j in RELIABLE]) <= IMPOSSIBLE_DEG, axis=0)
        t_det = t_cam[tracked]
        a_det = np.stack([cam[j][tracked] for j in JOINTS], axis=1)
        t_end = data["emg_t"][ends - 1]
        if len(t_det) > 1:
            k = np.clip(np.searchsorted(t_det, t_end), 1, len(t_det) - 1)
            t0, t1 = t_det[k - 1], t_det[k]
            near = (t_end - t0 <= MAX_FRAME_GAP_S) & (t1 - t_end <= MAX_FRAME_GAP_S) & (t1 > t0)
            w = np.clip((t_end - t0) / np.maximum(t1 - t0, 1e-6), 0, 1)[:, None]
            interp = a_det[k - 1] * (1 - w) + a_det[k] * w
            angles[near] = interp[near]

    keep = (grip != "") | ~np.isnan(angles).any(axis=1)
    x = np.lib.stride_tricks.sliding_window_view(y, WINDOW, axis=1)[:, ends[keep] - WINDOW].transpose(1, 0, 2)
    return {"x": np.ascontiguousarray(x), "grip": grip[keep].astype(str), "rep": rep[keep],
            "posture": posture[keep].astype(str), "kind": kind[keep].astype(str), "angles": angles[keep], "ends": ends[keep],
            "subject": meta.get("subject", ""), "channels": np.array(meta["channels"])}


def load(folder):
    """Windows of one session, from the cache when it is current."""
    folder = Path(folder)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{folder.name}.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if str(cached["version"]) == VERSION:
            return {k: cached[k] for k in cached.files if k != "version"}
    out = build(folder)
    np.savez(path, version=VERSION, **{k: (v if not isinstance(v, str) else np.array(v)) for k, v in out.items()})
    return out


def load_all(folders=None):
    """{session name: windows} for the given folders, or for every good session."""
    folders = folders if folders is not None else good_sessions()
    return {Path(f).name: load(f) for f in folders}


if __name__ == "__main__":
    low_priority()
    for name, d in load_all().items():
        labelled = d["grip"] != ""
        tracked = ~np.isnan(d["angles"]).any(axis=1)
        print(f"{name}: {len(d['grip'])} windows, {labelled.sum()} with a grip, {tracked.sum()} with angles, "
              f"{(tracked & ~labelled).sum()} with angles only")
    print(f"cache in {CACHE}")
