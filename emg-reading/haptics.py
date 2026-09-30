"""Drive the ESP32 outputs from the armband's EMG.

Each EMG channel's envelope is scaled between its rest and max level from
calibrate.py: 0% at or below rest, 100% at or above max, linear in between.
Ctrl+C to stop.

    python haptics.py [--synthetic] [--port COM3] [--rest 8 --max 80]
"""
import argparse
import time

import numpy as np

from armband import Armband
from calibrate import load_calibration
from esp32_link import Esp32
from processing import ENVELOPE_MS, envelope, filter_channel

# EMG channels feeding each output. Channels 6 and 7 sit on the base module and share the last output
OUTPUTS = [(0,), (1,), (2,), (3,), (4,), (5,), (6, 7)]

UPDATE_HZ = 20
HISTORY_S = 1.0
PRINT_EVERY_S = 0.5


def level(env, rest, max_):
    return float(np.clip((env - rest) / (max_ - rest), 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    parser.add_argument("--port", help="ESP32 serial port, auto-detected if omitted")
    parser.add_argument("--rest", type=float, help="envelope in uV that maps to 0%%, for all channels")
    parser.add_argument("--max", type=float, help="envelope in uV that maps to 100%%, for all channels")
    args = parser.parse_args()
    if (args.rest is None) != (args.max is None):
        parser.error("give both --rest and --max, or neither")
    if args.rest is not None and args.max <= args.rest:
        parser.error("--max must be above --rest")

    with Armband(args.synthetic) as band:
        if args.rest is not None:
            ranges = {ch: (args.rest, args.max) for ch in band.channels}
        else:
            ranges = load_calibration()
            if ranges is None:
                raise SystemExit("no calibration.json, run calibrate.py first (or pass --rest and --max)")
            missing = [ch for ch in band.channels if ch not in ranges]
            if missing:
                raise SystemExit(f"calibration.json has no entry for channels {missing}, run calibrate.py again")

        history_len = int(HISTORY_S * band.rate)
        env_len = int(ENVELOPE_MS / 1000 * band.rate)
        history = np.zeros((len(band.channels), 0))

        with Esp32(args.port) as esp:
            print("output <- EMG ch: " + "  ".join(f"{out}<-{'+'.join(map(str, chs))}" for out, chs in enumerate(OUTPUTS)))
            last_print = 0.0
            next_tick = time.perf_counter()
            try:
                while True:
                    history = np.hstack([history, band.read()])[:, -history_len:]
                    # Wait for a full history so the filter start-up transient is well before the envelope window
                    if history.shape[1] == history_len:
                        levels = {}
                        for ch, x in zip(band.channels, history):
                            levels[ch] = level(envelope(filter_channel(x, band.rate)[-env_len:]), *ranges[ch])
                        duties = []
                        for chs in OUTPUTS:
                            live = [levels[ch] for ch in chs if ch in levels]
                            duties.append(int(round(100 * np.mean(live))) if live else 0)
                        for out, duty in enumerate(duties):
                            esp.set(out, duty)
                        if time.perf_counter() - last_print >= PRINT_EVERY_S:
                            last_print = time.perf_counter()
                            print("  ".join(f"out{out} {duty:3d}%" for out, duty in enumerate(duties)))
                        if esp.watchdog_fired:
                            print("warning: ESP32 watchdog fired, outputs were off for a moment")
                            esp.watchdog_fired = False
                    next_tick += 1 / UPDATE_HZ
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
