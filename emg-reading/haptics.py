"""Drive the ESP32 outputs from the armband's EMG.

Each EMG channel's envelope is scaled between its rest and max level from
calibrate.py: 0% at or below rest, 100% at or above max, linear in between.
A channel whose pad loses contact is set to 0% until it recovers. Ctrl+C to stop.

The Control tab has a second mode, selections: the outputs stay off and only the control layer's discrete
events play a short pattern (pattern()), so the user feels which grip was picked, a release, a lock and each
counted clench or flick without looking at the hand.

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
# The Control tab's haptics settings. floor: lowest duty above 0 in effort mode, 0 until the motors' start duty is
# measured. strength: duty of the selection patterns
HAPTIC_DEFAULTS = {"haptics": "effort", "haptic_floor": 0, "haptic_strength": 100}

# Selection patterns, built from steps of 1 / UPDATE_HZ. Contact and release bursts are what DESC feedback uses
# (Cipriani group); coin motors take about 40 ms to spin up, so a short pulse is two steps and a long one six. The
# grip code is a count of pulses on every motor, since counts are told apart more reliably than sites round the
# forearm (more than about 4 sites per band gets unreliable) and the pads move under the band anyway. It echoes
# trigger mode: one long pulse for fist (the long clench), 1 to 3 short ones for pinch, tripod and key
SHORT, LONG, GAP = 2, 6, 3
GRIP_CODES = {"fist": "L", "pinch": "S", "tripod": "SS", "key": "SSS", "point": "LL"}
# Any other pose a rule can pick
OTHER_CODE = "LS"
RELEASE_STEPS = 12
# A counted clench or flick only ticks the first motor, so it can't be taken for a pick
TICK_OUTPUT = 0


def level(env, rest, max_):
    return float(np.clip((env - rest) / (max_ - rest), 0.0, 1.0))


def scaled_duty(level_, floor):
    """Duty in % for a level 0 to 1. Above 0 it starts at floor, the lowest duty that still turns a motor (coin
    motors don't start under about 30 to 40 %), so the bottom of the effort range isn't lost; 0 stays off."""
    return int(round(floor + (100 - floor) * level_)) if level_ > 0 else 0


def pattern(kind, strength=100, grip=None):
    """Duties per step (one list of len(OUTPUTS) per 1 / UPDATE_HZ) for a control event. kind: "start" (grip picked,
    grip says which), "release", "lock", "unlock" or "tick" (a clench or flick counted towards a pattern)."""
    n = len(OUTPUTS)
    all_on, off = [strength] * n, [0] * n
    if kind == "start":
        steps = []
        for c in GRIP_CODES.get(grip, OTHER_CODE):
            steps += [all_on] * (LONG if c == "L" else SHORT) + [off] * GAP
        return steps[:-GAP]
    if kind == "release":
        # Fades out, so it can't be taken for a long pulse. Coin motors keep turning below their start duty
        return [[int(round(strength * (1 - 0.6 * i / (RELEASE_STEPS - 1))))] * n for i in range(RELEASE_STEPS)]
    if kind in ("lock", "unlock"):
        # One motor at a time round the band, the other way for unlock
        order = range(n) if kind == "lock" else range(n - 1, -1, -1)
        return [[strength if j == i else 0 for j in range(n)] for i in order]
    if kind == "tick":
        return [[strength if j == TICK_OUTPUT else 0 for j in range(n)]] * SHORT
    raise ValueError(f"no pattern for {kind}")


def channel_levels(raw, filtered, channels, ranges, car):
    """raw, filtered: live channels x one envelope window. Returns ({channel: level 0 to 1}, lost per channel)."""
    lost = contact_lost(raw)
    y = common_average(filtered, ~lost) if car else filtered
    levels = {}
    for i, ch in enumerate(channels):
        levels[ch] = 0.0 if lost[i] else level(envelope(y[i]), *ranges[ch])
    return levels, lost


def effort(raw, filtered, channels, ranges, car):
    """How hard the muscles work, 0 to 1: mean level over the channels with pad contact."""
    levels, lost = channel_levels(raw, filtered, channels, ranges, car)
    good = [levels[ch] for ch, l in zip(channels, lost) if not l]
    return float(np.mean(good)) if good else 0.0


def output_duties(raw, filtered, channels, ranges, car, floor=0):
    """raw, filtered: live channels x one envelope window. Returns (duty per output in %, lost per channel)."""
    levels, lost = channel_levels(raw, filtered, channels, ranges, car)
    duties = []
    for chs in OUTPUTS:
        live = [levels[ch] for ch in chs if ch in levels]
        duties.append(scaled_duty(float(np.mean(live)), floor) if live else 0)
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
