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
STEP_MS = 1000 // UPDATE_HZ
# A mapping rule's own haptic, as settings: buzz (pulses of [on ms, off ms] on every motor), fade (ms, end as % of
# the start), sweep (one motor at a time, round or back, step_ms per motor, rounds), none, or default (the event's
# own pattern, for lock rules: one rule locks and unlocks, the sweep runs back for unlock). strength is % of the
# profile's strength. Times play in STEP_MS steps
HAPTIC_PATTERNS = ("buzz", "fade", "sweep", "none", "default")
SHORT_MS, LONG_MS, GAP_MS = SHORT * STEP_MS, LONG * STEP_MS, GAP * STEP_MS
FADE = {"pattern": "fade", "ms": RELEASE_STEPS * STEP_MS, "end": 40, "strength": 100}
SWEEP = {"pattern": "sweep", "direction": "round", "step_ms": STEP_MS, "rounds": 1, "strength": 100}
# A counted clench or flick only ticks the first motor, so it can't be taken for a pick
TICK_OUTPUT = 0


def level(env, rest, max_):
    return float(np.clip((env - rest) / (max_ - rest), 0.0, 1.0))


def scaled_duty(level_, floor):
    """Duty in % for a level 0 to 1. Above 0 it starts at floor, the lowest duty that still turns a motor (coin
    motors don't start under about 30 to 40 %), so the bottom of the effort range isn't lost; 0 stays off."""
    return int(round(floor + (100 - floor) * level_)) if level_ > 0 else 0


def buzz(code):
    """A buzz from a grip code: S a short pulse, L a long one."""
    return {"pattern": "buzz", "pulses": [[LONG_MS if c == "L" else SHORT_MS, GAP_MS] for c in code], "strength": 100}


def rule_haptic(action):
    """The pattern a mapping rule's action played before rules had their own: the grip's code for a pose, the fade
    for release, nothing for open and close (they move the hand, no event), default for lock and unlock."""
    if action in ("lock", "unlock"):
        return {"pattern": "default"}
    if action == "release":
        return dict(FADE)
    if action in ("open", "close"):
        return {"pattern": "none"}
    return buzz(GRIP_CODES.get(action, OTHER_CODE))


def _within(h, key, lo, hi, what):
    v = h.get(key)
    if not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{what} is {lo:g} to {hi:g}")


def haptic_settings(haptic, action=None):
    """A rule's haptic as checked settings. Rules saved before 2026-11-05 have none ("auto": what the action played
    before, rule_haptic) or a name: a grip code like "SS", "fade", "sweep" or "none". Raises ValueError."""
    if haptic in (None, "auto"):
        return rule_haptic(action) if action else {"pattern": "default"}
    if isinstance(haptic, str):
        if haptic in ("none", "fade", "sweep"):
            return {"none": {"pattern": "none"}, "fade": dict(FADE), "sweep": dict(SWEEP)}[haptic]
        if haptic and set(haptic) <= {"S", "L"} and len(haptic) <= 8:
            return buzz(haptic)
        raise ValueError(f"{haptic} is not a haptic pattern")
    h = dict(haptic)
    kind = h.get("pattern")
    if kind not in HAPTIC_PATTERNS:
        raise ValueError(f"{kind} is not a haptic pattern")
    if kind in ("buzz", "fade", "sweep"):
        _within(h, "strength", 10, 100, "haptic strength")
    if kind == "buzz":
        pulses = h.get("pulses")
        if not isinstance(pulses, list) or not 1 <= len(pulses) <= 8:
            raise ValueError("a buzz is 1 to 8 pulses")
        for p in pulses:
            if not (isinstance(p, list) and len(p) == 2 and STEP_MS <= p[0] <= 2000 and 0 <= p[1] <= 2000):
                raise ValueError(f"a pulse is on {STEP_MS} to 2000 ms, off 0 to 2000 ms")
    elif kind == "fade":
        _within(h, "ms", 2 * STEP_MS, 3000, "fade ms")
        _within(h, "end", 0, 100, "fade end %")
    elif kind == "sweep":
        if h.get("direction") not in ("round", "back"):
            raise ValueError("a sweep goes round or back")
        _within(h, "step_ms", STEP_MS, 500, "sweep ms per motor")
        _within(h, "rounds", 1, 5, "sweep rounds")
    return h


def _steps(ms):
    return max(1, int(round(ms / STEP_MS)))


def pattern(kind, strength=100, grip=None, haptic=None):
    """Duties per step (one list of len(OUTPUTS) per 1 / UPDATE_HZ) for a control event. kind: "start" (grip picked,
    grip says which), "release", "lock", "unlock" or "tick" (a clench or flick counted towards a pattern). haptic: the
    rule's own settings (haptic_settings) in place of the event's pattern; None or default keeps the event's."""
    n = len(OUTPUTS)
    h = haptic_settings(haptic) if haptic is not None else {"pattern": "default"}
    if h["pattern"] == "none":
        return []
    if h["pattern"] != "default":
        s = int(round(strength * h["strength"] / 100))
        if h["pattern"] == "buzz":
            steps = []
            for on, off in h["pulses"]:
                steps += [[s] * n] * _steps(on) + [[0] * n] * (_steps(off) if off else 0)
            while steps and not any(steps[-1]):
                steps.pop()
            return steps
        if h["pattern"] == "fade":
            k = _steps(h["ms"])
            return [[int(round(s * (1 - (1 - h["end"] / 100) * i / (k - 1))))] * n for i in range(k)]
        order = list(range(n)) if h["direction"] == "round" else list(range(n - 1, -1, -1))
        return [[s if j == i else 0 for j in range(n)] for i in order * h["rounds"] for _ in range(_steps(h["step_ms"]))]
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
