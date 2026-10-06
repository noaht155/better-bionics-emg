"""Measure each EMG channel's rest and squeeze levels and save them for haptics.py.

Wear the armband, run this and follow the prompts:
    python calibrate.py [--synthetic] [--seconds 5] [--no-car]

The 10 s recording is for troubleshooting. Normally the levels come from a recorded session (from_session): every
rest cue in every posture and every held grip, minutes of data instead of seconds. The app does that by itself when
a prediction model starts, from the session the model was calibrated on.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from armband import Armband
from processing import (ENVELOPE_MS, FILTER_VERSION, common_average, contact_lost, envelope_series,
                        filter_block)
from session import load_session, segments

CAL_FILE = Path(__file__).with_name("calibration.json")

# Skip the start of each phase, since people react late to the prompt
REACTION_S = 1.0
# At rest the outputs should stay off. The rest level is the 99th percentile of the resting envelope times a margin.
# Simulated on 9 sessions (levels from repetitions 1 and 2, tested on repetition 3): with the old 95th percentile and
# no margin an output was on in 69 % of rest windows; 99th x 1.5 from a session brings that to 1.7 % (worst session
# 4.2 %) while grips still switch an output on in 89 % of their windows. x 1.25 gives 3.4 % and 96 %
REST_PERCENTILE = 99
REST_MARGIN = 1.5
# Max level below the peaks so 100% is reachable
MAX_PERCENTILE = 80
MIN_RATIO = 2.0


def load_calibration():
    """Returns ({channel: (rest, max)} in uV, uses common average), or None if calibrate.py has not been run."""
    if not CAL_FILE.exists():
        return None
    data = json.loads(CAL_FILE.read_text())
    if data.get("filter_version") != FILTER_VERSION:
        raise SystemExit("calibration.json was made with an older filter, run calibrate.py again")
    return {int(ch): (v["rest"], v["max"]) for ch, v in data["channels"].items()}, data["common_average"]


def lost_channels(raw, rate):
    """True for channels that lost pad contact anywhere in the recording."""
    window = int(ENVELOPE_MS / 1000 * rate)
    lost = np.zeros(raw.shape[0], bool)
    for i in range(0, raw.shape[1] - window + 1, window):
        lost |= contact_lost(raw[:, i:i + window])
    return lost


def process(raw, rate, car):
    y = filter_block(raw, rate)
    return common_average(y, ~lost_channels(raw, rate)) if car else y


def record(band, prompt, seconds):
    print(prompt)
    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(1)
    band.read()
    print(f"  recording {seconds:g} s")
    time.sleep(REACTION_S + seconds)
    return band.read()[:, int(REACTION_S * band.rate):]


def compute(relaxed, squeezed, rate, channels, car):
    """relaxed, squeezed: live channels x samples of raw data.
    Returns ({channel: {"rest", "max"}}, report lines to show)."""
    lines = []
    lost = lost_channels(relaxed, rate) | lost_channels(squeezed, rate)
    if lost.any():
        lines.append(f"warning: channels {[ch for ch, l in zip(channels, lost) if l]} lost pad contact, "
                     "check them and run again")
    relaxed_y, squeezed_y = process(relaxed, rate, car), process(squeezed, rate, car)

    result = {}
    lines.append("ch    rest     max   ratio")
    for i, ch in enumerate(channels):
        result[ch], line = levels(ch, envelope_series(relaxed_y[i], rate), envelope_series(squeezed_y[i], rate))
        lines.append(line)
    return result, lines


def levels(ch, rest_env, active_env, low_note="barely changed, check the pad"):
    """Rest and max level of one channel from its envelope at rest and while active. Returns (levels, report line)."""
    rest = float(np.percentile(rest_env, REST_PERCENTILE)) * REST_MARGIN
    max_ = float(np.percentile(active_env, MAX_PERCENTILE))
    note = ""
    if max_ < MIN_RATIO * rest:
        note = f"  {low_note}"
        max_ = MIN_RATIO * rest
    return {"rest": round(rest, 2), "max": round(max_, 2)}, f"{ch}  {rest:6.1f}  {max_:6.1f}  {max_ / rest:5.1f}x{note}"


def from_session(folder, car=True):
    """Levels from a recorded session: rest from every rest cue, max from every held grip, each without its first
    REACTION_S and without cues marked bad. Returns ({channel: {"rest", "max"}}, report lines)."""
    data = load_session(folder)
    channels = data["meta"]["channels"]
    rate = data["meta"]["rate"]
    raw = data["emg"][channels]
    lost = lost_channels(raw, rate)
    y = filter_block(raw, rate)
    if car:
        y = common_average(y, ~lost)
    trim = int(REACTION_S * rate)
    holds = [s for s in segments(data["events"]) if s["kind"] == "hold" and s["sample1"] and not s["bad"]
             and s["sample1"] - s["sample0"] > trim]
    rest = [s for s in holds if s["label"] == "rest"]
    grips = [s for s in holds if s["label"] != "rest"]
    if not rest or not grips:
        raise ValueError(f"{Path(folder).name} has no rest or grip cues to take haptic levels from")
    lines = [f"from {Path(folder).name}: {len(rest)} rest cues, {len(grips)} grips", "ch    rest     max   ratio"]
    result = {}
    for i, ch in enumerate(channels):
        env = lambda group: np.concatenate([envelope_series(y[i, s["sample0"] + trim:s["sample1"]], rate) for s in group])
        # Some channels sit over muscles the grips hardly use, a low ratio there isn't a pad problem
        result[ch], line = levels(ch, env(rest), env(grips), "grips barely above rest")
        lines.append(line)
    return result, lines


def save(result, car):
    CAL_FILE.write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "filter_version": FILTER_VERSION,
                                    "common_average": car, "channels": result}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    parser.add_argument("--seconds", type=float, default=5.0, help="length of each recording")
    parser.add_argument("--no-car", action="store_true", help="don't subtract the common average of the channels")
    args = parser.parse_args()

    with Armband(args.synthetic) as band:
        relaxed = record(band, "RELAX your arm and keep it still", args.seconds)
        squeezed = record(band, "SQUEEZE a firm fist and hold it", args.seconds)
        rate, channels = band.rate, band.channels
    if relaxed.shape[1] == 0 or squeezed.shape[1] == 0:
        raise SystemExit("no data received, run check_connection.py")

    car = not args.no_car
    result, lines = compute(relaxed, squeezed, rate, channels, car)
    print("\n".join(lines))
    save(result, car)
    print(f"saved {CAL_FILE.name}")


if __name__ == "__main__":
    main()
