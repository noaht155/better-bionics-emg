"""Training windows from every good session, for the networks.

Good means completed (not stopped early) with gesture cues, from any user. Each window is the raw output of the
shared filter, 8 channels x WINDOW samples, ending every STEP samples, with what is known about it:
- grip: the cued grip if the window lies in the trimmed part of a hold cue (same rule as the LDA), else None
- rep: which repetition of that grip in that posture (1 to 3), 0 outside held grips
- posture: the posture of the cue the window ends in
- kind: the kind of that cue (hold, finger, free, move, custom, nogrip, sync, break), finger, free, move and custom
  are moving fingers. custom (protocol C) never gets a grip label, even where its key is a grip name. nogrip
  (protocol D) is forearm activity that isn't a grip, never a grip or rest label
- angles: the 20 joint angles from the camera at the window's last sample, NaN when the hand wasn't tracked

Each session is processed once and cached in ml/cache/ (ignored by git, it holds personal data). The cache holds the
filtered signal once and the windows are cut from it on load, storing every window took 4 times the space.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import devices
from gestures import GESTURE_SET, settle_s
from channels import mirrored
from hand_angles import JOINTS, RELIABLE
from ml import low_priority
from processing import FILTER_SETTLE_S, FILTER_VERSION, filter_block
from session import DATA_DIR, load_session, segments

RATE = 500
WINDOW = 100
STEP = 25
# Camera frames reach the PC 37 ms after they are taken (LED test 2026-10-07) and EMG samples arrive up to 17 ms
# late (armband round trip 2026-10-08), so the true shift is 20 to 37 ms; the middle is off by at most 9 ms. Each
# setup has its own measured shift in devices.json; this one stands in for a setup not measured yet. The tap delays
# in session.json (96 to 168 ms) were biased and are no longer read
CAMERA_DELAY_S = 0.028
# A camera frame further than this from a window's end doesn't label it
MAX_FRAME_GAP_S = 0.1
# A frame with a trained joint bent backwards past this is a tracking glitch (MediaPipe lost or flipped the hand,
# index PIP read down to -175), not a hand position. It is treated like a frame where no hand was found. 0.02 to 2.2 %
# of the angle windows per session had one before this (2026-10-06)
IMPOSSIBLE_DEG = -60
CACHE = Path(__file__).with_name("cache")
# A band bump in 2026-10-21_141651 peaked at 194,000 uV, real EMG in every other session stays under 1,850 uV. The
# network trains on raw windows, so 20 such windows broke whole training runs (devlog 2026-10-23)
BUMP_UV = 5000
# Never trained on, so they stay an honest test (ml.evaluate --benchmark): one F session per day from 10-19 to 10-21,
# the last one recorded on the MacBook
BENCHMARK = {"signs": ["2026-10-19_092314_noah", "2026-10-20_105110_noah", "2026-10-21_141651_noah"]}
# Bump when the windows or labels change, old cache files are then rebuilt
VERSION = f"8-f{FILTER_VERSION}-w{WINDOW}-s{STEP}-{'.'.join(GESTURE_SET)}"


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


def without_benchmark(folders, log=print):
    """The folders minus the benchmark sessions, saying which were left out."""
    held = {n for names in BENCHMARK.values() for n in names}
    left = [f for f in folders if Path(f).name in held]
    if left and log:
        log(f"left out benchmark sessions: {', '.join(Path(f).name for f in left)}")
    return [f for f in folders if Path(f).name not in held]


def quiet(y, ends):
    """True for the windows ending at ends that have no sample over BUMP_UV on any channel."""
    loud = np.flatnonzero(np.abs(y).max(axis=0) > BUMP_UV)
    return np.searchsorted(loud, ends - WINDOW) == np.searchsorted(loud, ends)


def quick(folder):
    """A quick calibration session: two repetitions of the grips, too short to hold out for scoring the
    2-repetition calibration. It still trains like any other good session."""
    return json.loads((Path(folder) / "session.json").read_text()).get("settings", {}).get("quick", False)


def label_shift(meta):
    """The camera delay taken off a session's frame times: its setup's measured shift (devices.json)."""
    setup = devices.of_session(meta)
    measured = setup and setup.get("label_shift_s")
    return measured if measured is not None else CAMERA_DELAY_S


def camera_times(cam, meta):
    """Frame times on the EMG clock. Frames arrive in pairs from Media Foundation, so the arrival time jitters by
    up to a frame. The camera itself runs at a steady rate and every grabbed frame has a number, so a straight line
    of arrival time against frame number gives clean times. Then the setup's camera delay is taken off."""
    t, frame = cam["t"], cam["frame"]
    if len(t) < 10:
        return t
    return fitted_frame_times(t, frame) - label_shift(meta)


def fitted_frame_times(t, frame):
    """Arrival times t of grabbed frames numbered frame, as a straight line through the earliest arrivals. The
    delay test (emg-reading/delay.py) uses the same times, so its number is the one the labels need."""
    a, b = np.polyfit(frame, t, 1)
    fitted = a * frame + b
    # Arrival is never early, only late, so the line is moved down to the earliest arrivals
    return fitted + np.percentile(t - fitted, 5)


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
    return _windows(_labelled(folder))


def _labelled(folder):
    """The session's filtered signal y and the labels of the windows that build keeps, ending at ends."""
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
    for g in segs:
        if g["sample1"] is None:
            continue
        inside = (ends > g["sample0"]) & (ends <= g["sample1"])
        posture[inside] = g["posture"]
        kind[inside] = g["kind"]
        if g["kind"] == "hold" and not g["bad"] and g["label"] in GESTURE_SET:
            held = inside & (ends - WINDOW >= g["sample0"] + int(settle_s(g["kind"], g["label"]) * RATE))
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

    keep = ((grip != "") | ~np.isnan(angles).any(axis=1)) & quiet(y, ends)
    return {"y": y, "grip": grip[keep].astype(str), "rep": rep[keep],
            "posture": posture[keep].astype(str), "kind": kind[keep].astype(str), "angles": angles[keep], "ends": ends[keep],
            "subject": meta.get("subject", ""), "channels": np.array(meta["channels"])}


def _windows(stored):
    """The session's windows with x cut from the filtered signal y."""
    out = {k: v for k, v in stored.items() if k != "y"}
    x = np.lib.stride_tricks.sliding_window_view(stored["y"], WINDOW, axis=1)[:, stored["ends"] - WINDOW]
    out["x"] = np.ascontiguousarray(x.transpose(1, 0, 2))
    return out


def load(folder):
    """Windows of one session, from the cache when it is current."""
    folder = Path(folder)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{folder.name}.npz"
    # A setup measured later changes its sessions' angle labels, so the shift is part of the version
    version = f"{VERSION}-d{label_shift(json.loads((folder / 'session.json').read_text())) * 1000:.1f}"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if str(cached["version"]) == version:
            return _windows({k: cached[k] for k in cached.files if k != "version"})
    out = _labelled(folder)
    np.savez(path, version=version, **{k: (v if not isinstance(v, str) else np.array(v)) for k, v in out.items()})
    return _windows(out)


def clear_cache():
    """Deletes every cached session, each is rebuilt the next time it loads. Returns the bytes freed."""
    freed = 0
    for path in CACHE.glob("*.npz"):
        freed += path.stat().st_size
        path.unlink()
    return freed


def prune_cache(data_dir=DATA_DIR):
    """Deletes the cache files of sessions that no longer exist. Returns the bytes freed."""
    freed = 0
    for path in CACHE.glob("*.npz"):
        if not (Path(data_dir) / path.stem / "session.json").exists():
            freed += path.stat().st_size
            path.unlink()
    return freed


def load_all(folders=None, verbose=False):
    """{session name: windows} for the given folders, or for every good session. verbose prints each session as it
    loads, a session that isn't cached yet takes a while."""
    if folders is None:
        prune_cache()
        folders = good_sessions()
    out = {}
    for i, f in enumerate(folders, 1):
        if verbose:
            print(f"loading session {i}/{len(folders)}: {Path(f).name}", flush=True)
        out[Path(f).name] = load(f)
    return out


if __name__ == "__main__":
    low_priority()
    for name, d in load_all().items():
        labelled = d["grip"] != ""
        tracked = ~np.isnan(d["angles"]).any(axis=1)
        print(f"{name}: {len(d['grip'])} windows, {labelled.sum()} with a grip, {tracked.sum()} with angles, "
              f"{(tracked & ~labelled).sum()} with angles only")
    print(f"cache in {CACHE}")
