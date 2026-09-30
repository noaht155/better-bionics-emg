"""Measure each EMG channel's rest and squeeze levels and save them for haptics.py.

Wear the armband, run this and follow the prompts:
    python calibrate.py [--synthetic] [--seconds 5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from armband import Armband
from processing import envelope_series

CAL_FILE = Path(__file__).with_name("calibration.json")

# Skip the start of each phase, since people react late to the prompt
REACTION_S = 1.0
# Rest level sits above almost all resting noise, max level below the peaks so 100% is reachable
REST_PERCENTILE = 95
MAX_PERCENTILE = 80
MIN_RATIO = 2.0


def load_calibration():
    """Returns {channel: (rest, max)} in uV, or None if calibrate.py has not been run."""
    if not CAL_FILE.exists():
        return None
    data = json.loads(CAL_FILE.read_text())
    return {int(ch): (v["rest"], v["max"]) for ch, v in data["channels"].items()}


def record(band, prompt, seconds):
    print(prompt)
    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(1)
    band.read()
    print(f"  recording {seconds:g} s")
    time.sleep(REACTION_S + seconds)
    return band.read()[:, int(REACTION_S * band.rate):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    parser.add_argument("--seconds", type=float, default=5.0, help="length of each recording")
    args = parser.parse_args()

    with Armband(args.synthetic) as band:
        relaxed = record(band, "RELAX your arm and keep it still", args.seconds)
        squeezed = record(band, "SQUEEZE a firm fist and hold it", args.seconds)
        rate, channels = band.rate, band.channels
    if relaxed.shape[1] == 0 or squeezed.shape[1] == 0:
        raise SystemExit("no data received, run check_connection.py")

    result = {}
    print("ch    rest     max   ratio")
    for i, ch in enumerate(channels):
        rest = float(np.percentile(envelope_series(relaxed[i], rate), REST_PERCENTILE))
        max_ = float(np.percentile(envelope_series(squeezed[i], rate), MAX_PERCENTILE))
        note = ""
        if max_ < MIN_RATIO * rest:
            note = "  barely changed, check the pad"
            max_ = MIN_RATIO * rest
        result[ch] = {"rest": round(rest, 2), "max": round(max_, 2)}
        print(f"{ch}  {rest:6.1f}  {max_:6.1f}  {max_ / rest:5.1f}x{note}")

    CAL_FILE.write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "channels": result}, indent=2) + "\n")
    print(f"saved {CAL_FILE.name}")


if __name__ == "__main__":
    main()
