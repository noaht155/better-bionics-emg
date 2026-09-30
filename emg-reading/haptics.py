"""Drive the ESP32 outputs from the armband's EMG.

Each EMG channel's envelope is scaled between its rest and max level from
calibrate.py: 0% at or below rest, 100% at or above max, linear in between.
A channel whose pad loses contact is set to 0% until it recovers. Ctrl+C to stop.

    python haptics.py [--synthetic] [--port COM3] [--wifi [IP]] [--rest 8 --max 80 [--no-car]]
"""
import argparse
import time

import numpy as np

from armband import Armband
from calibrate import load_calibration
from esp32_link import Esp32, add_link_args
from processing import ENVELOPE_MS, FILTER_SETTLE_S, StreamFilter, common_average, contact_lost, envelope

# EMG channels feeding each output. Channels 6 and 7 sit on the base module and share the last output
OUTPUTS = [(0,), (1,), (2,), (3,), (4,), (5,), (6, 7)]

UPDATE_HZ = 20
PRINT_EVERY_S = 0.5


def level(env, rest, max_):
    return float(np.clip((env - rest) / (max_ - rest), 0.0, 1.0))


def output_duties(raw, filtered, channels, ranges, car):
    """raw, filtered: live channels x one envelope window. Returns (duty per output in %, lost per channel)."""
    lost = contact_lost(raw)
    y = common_average(filtered, ~lost) if car else filtered
    levels = {}
    for i, ch in enumerate(channels):
        levels[ch] = 0.0 if lost[i] else level(envelope(y[i]), *ranges[ch])
    duties = []
    for chs in OUTPUTS:
        live = [levels[ch] for ch in chs if ch in levels]
        duties.append(int(round(100 * np.mean(live))) if live else 0)
    return duties, lost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    add_link_args(parser)
    parser.add_argument("--rest", type=float, help="envelope in uV that maps to 0%%, for all channels")
    parser.add_argument("--max", type=float, help="envelope in uV that maps to 100%%, for all channels")
    parser.add_argument("--no-car", action="store_true", help="with --rest/--max: don't subtract the common average")
    args = parser.parse_args()
    if (args.rest is None) != (args.max is None):
        parser.error("give both --rest and --max, or neither")
    if args.rest is not None and args.max <= args.rest:
        parser.error("--max must be above --rest")

    with Armband(args.synthetic) as band:
        if args.rest is not None:
            ranges = {ch: (args.rest, args.max) for ch in band.channels}
            car = not args.no_car
        else:
            loaded = load_calibration()
            if loaded is None:
                raise SystemExit("no calibration.json, run calibrate.py first (or pass --rest and --max)")
            ranges, car = loaded
            missing = [ch for ch in band.channels if ch not in ranges]
            if missing:
                raise SystemExit(f"calibration.json has no entry for channels {missing}, run calibrate.py again")

        env_len = int(ENVELOPE_MS / 1000 * band.rate)
        settle_len = int(FILTER_SETTLE_S * band.rate)
        stream = StreamFilter(len(band.channels), band.rate)
        raw = np.zeros((len(band.channels), 0))
        filtered = np.zeros((len(band.channels), 0))
        received = 0

        with Esp32(args.port, args.wifi) as esp:
            print(f"ESP32 on {esp.link.name}, common average {'on' if car else 'off'}")
            print("output <- EMG ch: " + "  ".join(f"{out}<-{'+'.join(map(str, chs))}" for out, chs in enumerate(OUTPUTS)))
            last_print = 0.0
            next_tick = time.perf_counter()
            try:
                while True:
                    new = band.read()
                    received += new.shape[1]
                    raw = np.hstack([raw, new])[:, -env_len:]
                    filtered = np.hstack([filtered, stream.process(new)])[:, -env_len:]
                    if received >= settle_len + env_len:
                        duties, lost = output_duties(raw, filtered, band.channels, ranges, car)
                        esp.set_outputs(duties)
                        if time.perf_counter() - last_print >= PRINT_EVERY_S:
                            last_print = time.perf_counter()
                            line = "  ".join(f"out{out} {duty:3d}%" for out, duty in enumerate(duties))
                            if lost.any():
                                line += f"   pad contact lost on ch {[ch for ch, l in zip(band.channels, lost) if l]}"
                            print(line)
                        if esp.watchdog_fired:
                            print("warning: ESP32 watchdog fired, outputs were off for a moment")
                            esp.watchdog_fired = False
                    # If a slow reply made this update late, don't try to catch up with a burst
                    next_tick = max(next_tick + 1 / UPDATE_HZ, time.perf_counter())
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
