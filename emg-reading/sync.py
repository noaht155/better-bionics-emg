"""Camera delay from the sync taps at the start and end of each session.

Each tap is a sharp slap of the palm on the table. The armband's accelerometer sees the impact as a big jump
between two readings, the camera sees the hand stop. The camera delay is how much later the frame timestamps
put the stop than the EMG timestamps put the impact. Subtract it from the camera times before lining them up
with the EMG.

    python sync.py data/<session> [--save]

--save writes camera_delay_s into the session's session.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from session import load_session, segments

# People react late to the cue, so look a little past its end
TAP_PAD_S = 0.6
# A slap jumps the reading by several tenths of a g. At rest consecutive readings differ by under 0.01 g
MIN_JUMP_G = 0.05
# Normalised image units per second, a slap moves the palm across a good part of the frame
MIN_SPEED = 0.5
# The hand counts as stopped once its speed falls below this fraction of the slap's peak speed
STOP_FRACTION = 0.25
# After the slap the hand rests on the table. At the top of the lift before it, the hand moves again at once
STILL_STEPS = 2
# Taps agree when their delays are this close. A single tap could be any sudden movement, so at least
# MIN_AGREEING taps have to agree before the delay is trusted
OUTLIER_S = 0.05
MIN_AGREEING = 2
PALM = [0, 5, 9, 13, 17]


def accel_readings(accel, t):
    """The accelerometer updates once per packet of 10 samples (50 Hz) and repeats its value in between.
    Returns the time and value of each new reading, timed at the first sample of its packet."""
    new = np.r_[True, np.any(np.diff(accel, axis=1) != 0, axis=0)]
    return t[new], accel[:, new]


def accel_impact(t, accel, t0, t1):
    """Time of the first jump between readings within [t0, t1] that is at least half the biggest one."""
    sel = (t >= t0) & (t <= t1)
    tt, aa = t[sel], accel[:, sel]
    if len(tt) < 3:
        return None
    jump = np.linalg.norm(np.diff(aa, axis=1), axis=0)
    if jump.max() < MIN_JUMP_G:
        return None
    # The swing down moves the arm too, the impact is the first jump close to the biggest one
    return tt[np.argmax(jump >= 0.5 * jump.max()) + 1]


def camera_stop(cam, t0, t1):
    """Time the palm stopped after its fastest movement within [t0, t1], interpolated between frames."""
    sel = (cam["t"] >= t0) & (cam["t"] <= t1) & (cam["detected"] == 1)
    if sel.sum() < 4:
        return None
    t, frame = cam["t"][sel], cam["frame"][sel]
    img = np.stack([np.stack([cam[f"img_{i}_x"][sel], cam[f"img_{i}_y"][sel]], axis=1) for i in range(21)], axis=1)
    palm = img[:, PALM].mean(axis=1)
    size = np.linalg.norm(img[:, 0] - img[:, 9], axis=1)
    # A slap towards the camera barely moves the palm in the image but changes its apparent size
    pos = np.column_stack([palm, np.median(size) * np.log(size)])
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
    # A missed detection would hide part of the movement, so steps across a gap don't count
    speed[np.diff(frame) != 1] = np.nan
    mid = (t[1:] + t[:-1]) / 2
    if np.all(np.isnan(speed)) or np.nanmax(speed) < MIN_SPEED:
        return None
    # The lift before the slap can be the fastest movement, so try the fast steps in order until one ends in a
    # proper stop
    for peak in np.argsort(-np.nan_to_num(speed, nan=0.0)):
        if not speed[peak] >= MIN_SPEED:
            return None
        level = STOP_FRACTION * speed[peak]
        k = peak
        while k < len(speed) and speed[k] >= level:
            k += 1
        still = speed[k:k + STILL_STEPS]
        if len(still) == STILL_STEPS and not np.isnan(still).any() and (still < level).all():
            break
    else:
        return None
    # Linear interpolation of the crossing between the last fast step and the first slow one
    frac = (speed[k - 1] - level) / (speed[k - 1] - speed[k])
    return mid[k - 1] + frac * (mid[k] - mid[k - 1])


def camera_delay(data):
    """data: from load_session. Returns {"delay_s", "spread_s", "outliers", "taps": [per tap dict]}, delay None
    unless at least MIN_AGREEING taps agree."""
    t_acc, acc = accel_readings(data["accel"], data["emg_t"])
    taps = []
    for seg in segments(data["events"]):
        if seg["kind"] != "sync" or seg["t1"] is None:
            continue
        t0, t1 = seg["t0"], seg["t1"] + TAP_PAD_S
        impact = accel_impact(t_acc, acc, t0, t1)
        stop = camera_stop(data["camera"], t0, t1)
        taps.append({"t": t0, "impact": impact, "stop": stop,
                     "delay": stop - impact if impact is not None and stop is not None else None})
    delays = np.array([tap["delay"] for tap in taps if tap["delay"] is not None])
    # Use the biggest group of taps that agree. A plain median of a few taps can land between two groups
    agree = [np.abs(delays - d) <= OUTLIER_S for d in delays]
    close = delays[max(agree, key=np.sum)] if len(delays) else delays
    if len(close) < MIN_AGREEING:
        return {"delay_s": None, "spread_s": None, "outliers": len(delays) - len(close), "taps": taps}
    return {"delay_s": float(np.median(close)), "spread_s": float(close.max() - close.min()),
            "outliers": len(delays) - len(close), "taps": taps}


def save(folder, result):
    path = Path(folder) / "session.json"
    meta = json.loads(path.read_text())
    meta["camera_delay_s"] = result["delay_s"]
    meta["camera_delay_taps"] = [tap["delay"] for tap in result["taps"]]
    path.write_text(json.dumps(meta, indent=1) + "\n")


def summary(result):
    found = sum(tap["delay"] is not None for tap in result["taps"])
    if result["delay_s"] is None:
        return (f"camera delay: not measured, {found} of {len(result['taps'])} taps found in both streams "
                f"and fewer than {MIN_AGREEING} agree")
    used = found - result["outliers"]
    text = (f"camera delay {1000 * result['delay_s']:.0f} ms from {used} of {len(result['taps'])} taps, "
            f"spread {1000 * result['spread_s']:.0f} ms")
    return text + (f", {result['outliers']} outliers left out" if result["outliers"] else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--save", action="store_true", help="write the delay into session.json")
    args = parser.parse_args()

    result = camera_delay(load_session(args.folder))
    print("tap   impact (accel)   stop (camera)   delay")
    t_start = result["taps"][0]["t"] if result["taps"] else 0
    for i, tap in enumerate(result["taps"]):
        cols = [f"{tap[k] - t_start:8.3f} s" if tap[k] is not None else "  not found" for k in ("impact", "stop")]
        delay = f"{1000 * tap['delay']:5.0f} ms" if tap["delay"] is not None else "    -"
        print(f"{i:3d}   {cols[0]}      {cols[1]}     {delay}")
    print(summary(result))
    if args.save and result["delay_s"] is not None:
        save(args.folder, result)
        print("saved to session.json")


if __name__ == "__main__":
    main()
