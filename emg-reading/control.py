"""Control layer between the grip model and the hand: one locked grip and how far the hand closes on it.

Called once per 50 ms window with the grip probabilities and the effort (0 to 1, the envelope scaled between the
calibrated rest and max levels, mean over the good channels), and optionally the window's distance from the winning
grip's calibration windows (gesture_model.LivePredictor.distance). The grip is only chosen while effort rises from rest,
then stays locked until release, so it never switches mid-grasp. Closure follows effort along the path from the
relaxed pose to the grip's pose. Used by the app's Control tab and the offline replay.

mode "blend" (2026-10-15) aims at a hand that feels continuous without decoding fingers: closure follows effort from
the first window, with no votes to wait for, and the hand closes towards a mix of the grip poses weighted by the
grip probabilities, smoothed over smooth_s, so the shape settles into the grip as the classifier firms up and a one
window flicker only nudges it. When one grip keeps at least the threshold for lock_s, the shape locks to it (0 never
locks) until the release mode lets go, the same releases as mode "lock", the vote and lock design above.

mode "trigger" (Noah, 2026-10-15) only asks the classifier for what it does reliably (open hand and point) and picks
the other grips by clench pattern on effort alone, like the triggers of commercial hands. With the hand open: one
clench held past long_s is fist (and keeps closing), one short clench then a pause of gap_s is pinch, two are tripod,
three are key; point wins vote_n steps is point. Then squeezing closes the hand at a speed set by effort, the open hand
opens it, relaxing holds it. Two flicks of release_with (point by default) within double_s go back to choosing; with
open, opening part way and releasing shared one signal (Noah, live). A pulse during which open or the release class
wins is never a clench, and after a release point can't pick a grip until it has dropped out.
Two options against grips picked during arm activity (both off by default): still_gate ignores clenches and point
while the arm moves (accelerometer, see movement), and pinch_clenches 2 moves pinch, tripod and key to 2, 3 and 4
clenches, since most false picks were a single clench. Replayed on the protocol D activities at clench level 0.3:
14.9 false picks per minute, 5.4 with the gate, 6.1 with 2 clenches, 1.0 with both (diagnostics/trigger_gates.py).

mode "rules" (Noah, 2026-10-19) is trigger mode with the wiring as lists the user edits (the mapping): each rule is
{"trigger": "clench" or one of the model's classes, "how": "held" or "repeated", "value": seconds or count,
"action": "close", "open", "release" or a pose}. A clench is effort above pulse_level; a class is on while its probability is
at least the threshold. Held fires once the trigger has been on that long; repeated counts short episodes (shorter
than the trigger's held rule) and fires after gap_s without a new one, or straight away at the highest count any rule
of that trigger asks for. The mapping has one list used while relaxed (no grip) and one while holding a grip, and
either can hold any rule: a pose picks it (while holding it switches to it at the same closure), release goes back
to relaxed, close and open move the hand for as long as their trigger stays on (speed set by effort), and do nothing
while relaxed. Only the list of the current state runs, so a trigger can have a held rule in each, and the higher
rule in a list wins when two fire on one step. Held 0 s acts on the first step the trigger is on. While holding, effort above hold_level also
closes the hand as in trigger mode. A clench during which a rule's class is on belongs to
that class, not to the clench count. Lock (Noah, 2026-10-20) freezes the hand while holding: nothing closes, opens,
switches or releases it, and only the lock rules (lock again to undo) and unlock rules of the holding list run.
"""
import math

import numpy as np

# Best on the replay of 9 protocol A sessions (2026-10-14): the LDA is often wrong on the rise into a grip, so a long
# vote waits that out, and light grips (pinch, key) only reach about 0.15 effort, so the start level sits near zero.
# Arm and wrist movement that isn't a grip (protocol D, 2026-10-15) lands far from every calibrated grip but still
# wins one of them with full confidence: windows farther than max_distance count as rest (0 turns it off), and the
# hand also lets go when rest or the open hand wins release_n steps, not only when effort stays low.
DEFAULTS = {"threshold": 0.7, "vote_n": 8, "start_level": 0.02, "onset_s": 3.0, "closure_mode": "position",
            "gain": 1.5, "hold_level": 0.1, "hold_on_relax": True, "release_mode": "rest_or_class", "release_n": 6,
            "release_level": 0.03, "release_s": 1.5, "max_distance": 6.0, "spring_w": 12,
            "mode": "lock", "smooth_s": 0.25, "lock_s": 0.5,
            "pulse_level": 0.2, "long_s": 0.7, "gap_s": 0.6, "double_s": 1.0, "release_with": "point",
            "still_gate": False, "pinch_clenches": 1}
# Rules mode: a mapping is two lists, the rules used while relaxed (no grip) and while holding one. The grip default is trigger mode's defaults written as rules; sign models get their own (Noah,
# 2026-10-19), since E and F have no clench (rules_for)
DEFAULT_RULES = {"relaxed": [{"trigger": "clench", "how": "held", "value": 0.7, "action": "fist"},
                             {"trigger": "clench", "how": "repeated", "value": 1, "action": "pinch"},
                             {"trigger": "clench", "how": "repeated", "value": 2, "action": "tripod"},
                             {"trigger": "clench", "how": "repeated", "value": 3, "action": "key"},
                             {"trigger": "point", "how": "held", "value": 0.4, "action": "point"}],
                 "holding": [{"trigger": "open", "how": "held", "value": 0.0, "action": "open"},
                             {"trigger": "point", "how": "repeated", "value": 2, "action": "release"}]}
SIGN_RULES = {"relaxed": [{"trigger": "two_point", "how": "held", "value": 0.4, "action": "point"},
                          {"trigger": "little_out", "how": "held", "value": 0.5, "action": "fist"},
                          {"trigger": "move_thumb_abd", "how": "held", "value": 0.5, "action": "pinch"},
                          {"trigger": "move_thumb_abd", "how": "repeated", "value": 2, "action": "tripod"},
                          {"trigger": "little_out", "how": "repeated", "value": 2, "action": "key"}],
              "holding": [{"trigger": "open", "how": "held", "value": 0.0, "action": "open"},
                          {"trigger": "two_point", "how": "repeated", "value": 2, "action": "release"}]}
SECTIONS = ("relaxed", "holding")
RULE_ACTIONS = ("close", "open", "release", "lock", "unlock")
MAX_REPEATS = 6
MAX_HELD_S = 5.0
# Trigger mode: grips by number of short clenches, from pinch_clenches up. Fist is the long clench, point comes from
# the classifier
TRIGGER_GRIPS = ["pinch", "tripod", "key"]
# The arm counts as moving when the accelerometer's spread over the last STILL_S is above the limit taken from a
# calibration session's holds (still_limit)
STILL_S = 0.5
STILL_MARGIN = 1.5
# The release class has to drop out this long between two flicks, so a one window flicker isn't a second flick
FLICK_GAP_S = 0.15
# Same integration step as the drawn hands in web/app.js
SPRING_STEP_S = 0.005
# Sums of dt drift a little, 12 x 0.05 isn't exactly 0.6
EPS = 1e-9


def split_rules(rules):
    """A copy of a mapping as its two lists. Mappings saved as one list (before 2026-10-19) split by their action:
    poses were only picked while relaxed, open, close and release only acted while holding."""
    if isinstance(rules, dict):
        return {s: [dict(r) for r in rules.get(s, [])] for s in SECTIONS}
    return {"relaxed": [dict(r) for r in rules if r["action"] not in RULE_ACTIONS],
            "holding": [dict(r) for r in rules if r["action"] in RULE_ACTIONS]}


def rules_for(classes):
    """The default mapping for a model with these classes, without the rules of classes it doesn't have."""
    signs = {r["trigger"] for s in SECTIONS for r in SIGN_RULES[s]} - {"open"} & set(classes)
    rules = split_rules(SIGN_RULES if signs else DEFAULT_RULES)
    return {s: [r for r in rules[s] if r["trigger"] == "clench" or r["trigger"] in classes] for s in SECTIONS}


def check_rules(rules, poses):
    """Raises ValueError for a rule that can't work: unknown how or action, a value out of range, or two rules in one
    list on the same trigger and pattern (a trigger can have one held rule per list)."""
    rules = split_rules(rules)
    for section in SECTIONS:
        _check_list(rules[section], poses)


def _check_list(rules, poses):
    seen = set()
    for r in rules:
        if not r["trigger"]:
            raise ValueError("every rule needs a trigger")
        if r["how"] not in ("held", "repeated"):
            raise ValueError(f"{r['how']}: a rule is held or repeated")
        if r["action"] not in RULE_ACTIONS and r["action"] not in poses:
            raise ValueError(f"{r['action']} is not a pose or action")
        if r["how"] == "held" and not 0 <= r["value"] <= MAX_HELD_S:
            raise ValueError(f"held time is 0 to {MAX_HELD_S:g} s")
        if r["how"] == "repeated" and (r["value"] != int(r["value"]) or not 1 <= r["value"] <= MAX_REPEATS):
            raise ValueError(f"repeated is 1 to {MAX_REPEATS} times")
        key = (r["trigger"], r["how"], r["value"] if r["how"] == "repeated" else None)
        if key in seen:
            raise ValueError(f"two rules for {r['trigger']} {r['how']}" + (f" {r['value']:g}" if key[2] else ""))
        seen.add(key)


def movement(accel):
    """Spread of the accelerometer over a window: the three axes' standard deviations summed. accel: 3 x samples."""
    return float(np.asarray(accel).std(axis=1).sum())


def still_limit(accel, holds, rate):
    """Movement above which the arm counts as moving: 99th percentile over the held cues of a calibration session,
    times STILL_MARGIN. holds: (first, last) sample of each held cue."""
    n = int(STILL_S * rate)
    step = n // 5
    values = [movement(accel[:, e - n:e]) for a, b in holds for e in range(a + n, b + 1, step)]
    return float(np.percentile(values, 99)) * STILL_MARGIN if values else float("inf")


class Controller:
    def __init__(self, params, grips, poses):
        # Read on every step, so a changed value applies straight away
        self.params = dict(DEFAULTS, **(params or {}))
        self.grips = list(grips)
        self.poses = {g: np.asarray(a, float) for g, a in poses.items()}
        self.relaxed = self.poses["rest"]
        self.rules = split_rules(DEFAULT_RULES)
        self.reset()

    def set_rules(self, rules):
        check_rules(rules, self.poses)
        self.rules = split_rules(rules)
        self.locked = False
        self._reset_rules()

    def _reset_rules(self):
        """Per trigger: time on in the current episode, time since it was last on, short episodes counted, whether
        this episode already did something; stuck triggers wait to go off first."""
        self.run_t, self.off_t, self.counts, self.used, self.stuck = {}, {}, {}, {}, set()

    def reset(self):
        self.state = "idle"
        self.locked = False
        self.grip = None
        self.closure = 0.0
        self.above = False
        self.onset = 0.0
        self.blocked = False
        self.candidate, self.votes = None, 0
        self.open_votes, self.class_votes, self.rest_time = 0, 0, 0.0
        self.smooth = {g: 0.0 for g in self.grips}
        self.smooth["rest"] = 1.0
        self.leader, self.lead_time = None, 0.0
        self.clock, self.flicks, self.flick_was_on, self.flick_off = 0.0, [], False, FLICK_GAP_S
        self.pulse_time, self.pulse_open, self.pulses, self.gap, self.point_votes = 0.0, False, 0, 0.0, 0
        self.point_blocked = False
        self._reset_rules()
        self.x = self.relaxed.copy()
        self.v = np.zeros_like(self.x)

    def step(self, probs, effort, dt=0.05, distance=None, moving=False):
        """moving: the arm moves (movement() above still_limit()), only used by trigger mode's still_gate."""
        p = self.params
        if not isinstance(probs, dict):
            probs = dict(zip(self.grips, np.asarray(probs, float)))
        if p["max_distance"] > 0 and distance is not None and distance > p["max_distance"]:
            probs = {"rest": 1.0}
        if p["mode"] == "blend":
            return self._blend(probs, effort, dt)
        if p["mode"] == "trigger":
            return self._trigger(probs, effort, dt, moving)
        if p["mode"] == "rules":
            return self._rules(probs, effort, dt, moving)
        prob = lambda g: float(probs.get(g, 0.0))
        event = None

        above = effort >= p["start_level"]
        if above and not self.above:
            self.onset = 0.0
        elif above:
            self.onset += dt
        else:
            self.blocked = False
        self.above = above
        if above and self.onset > p["onset_s"] + EPS:
            self.blocked = True

        if self.state == "idle":
            candidate = max((g for g in self.grips if g != "rest"), key=prob)
            if above and not self.blocked and prob(candidate) >= p["threshold"]:
                self.votes = self.votes + 1 if candidate == self.candidate else 1
                self.candidate = candidate
            else:
                self.candidate, self.votes = None, 0
            if self.votes >= p["vote_n"]:
                self.blocked = True
                self.candidate, self.votes = None, 0
                # The open class opens the hand fully, which in idle is where it already is
                if candidate == "open":
                    event = "open"
                else:
                    self.state, self.grip, event = "active", candidate, "start"
                    self.open_votes, self.class_votes, self.rest_time = 0, 0, 0.0
                    self.closure = self._closure(effort, dt) if p["closure_mode"] == "position" else 0.0
        else:
            self.closure = self._closure(effort, dt)
            if self._released(probs, effort, dt):
                self.state, self.grip, self.closure, event = "idle", None, 0.0, "release"
                self.blocked = True
                self.candidate, self.votes = None, 0
        if self.blocked:
            self.candidate, self.votes = None, 0

        return {"state": self.state, "grip": self.grip, "closure": self.closure, "angles": self._spring(dt),
                "event": event}

    def _blend(self, probs, effort, dt):
        p = self.params
        a = dt / (p["smooth_s"] + dt) if p["smooth_s"] > 0 else 1.0
        for g in self.grips:
            self.smooth[g] += a * (float(probs.get(g, 0.0)) - self.smooth[g])
        hand = {g: v for g, v in self.smooth.items() if g != "rest"}
        leader = max(hand, key=hand.get)
        event = None

        if self.state == "active":
            if self._released(probs, effort, dt):
                self.state, self.grip, event = "blend", None, "release"
                self.leader, self.lead_time = None, 0.0
        else:
            self.state = "blend"
            self.lead_time = self.lead_time + dt if leader == self.leader and hand[leader] >= p["threshold"] else 0.0
            self.leader = leader
            if p["lock_s"] > 0 and self.lead_time >= p["lock_s"] - EPS and effort >= p["start_level"]:
                self.state, self.grip, event = "active", leader, "start"
                self.open_votes, self.class_votes, self.rest_time = 0, 0, 0.0

        c = float(np.clip((effort - p["start_level"]) * p["gain"], 0.0, 1.0))
        # Rest is the relaxed pose in the mix, so a window counted as rest (far away, or the arm moving) keeps the
        # hand open even with effort up
        if self.state != "active":
            c *= 1.0 - self.smooth["rest"]
        if self.state == "active" and p["hold_on_relax"]:
            c = max(c, self.closure)
        self.closure = c
        if self.state == "active":
            shape = self.poses[self.grip]
        else:
            total = sum(hand.values())
            shape = (sum(v * self.poses[g] for g, v in hand.items()) / total if total > 0 else self.relaxed)
        target = self.relaxed + c * (shape - self.relaxed)
        shown = self.grip or (leader if hand[leader] > self.smooth["rest"] else None)
        return {"state": self.state, "grip": shown, "closure": self.closure,
                "angles": self._spring(dt, target), "event": event}

    def _trigger(self, probs, effort, dt, moving=False):
        p = self.params
        open_on = float(probs.get("open", 0.0)) >= p["threshold"]
        point_on = float(probs.get("point", 0.0)) >= p["threshold"]
        flick_on = point_on if p["release_with"] == "point" else open_on
        if flick_on and not self.flick_was_on and self.flick_off >= FLICK_GAP_S - EPS:
            self.flicks = [t for t in self.flicks if self.clock - t <= p["double_s"] + EPS] + [self.clock]
        self.flick_off = 0.0 if flick_on else self.flick_off + dt
        self.flick_was_on = flick_on
        self.clock += dt
        event = None

        if self.state == "active":
            if len(self.flicks) >= 2:
                self.state, self.grip, self.closure, event = "idle", None, 0.0, "release"
                self.flicks, self.pulses, self.gap, self.point_votes = [], 0, 0.0, 0
                # The flick still going on mustn't count as a clench or pick point
                self.pulse_time, self.pulse_open = (dt, True) if effort >= p["pulse_level"] else (0.0, False)
                self.point_blocked = point_on
            elif open_on:
                self.closure -= max(effort, p["hold_level"]) * p["gain"] * dt
            elif flick_on:
                pass
            elif effort >= p["hold_level"]:
                self.closure += (effort - p["hold_level"]) * p["gain"] * dt
            self.closure = float(np.clip(self.closure, 0.0, 1.0))
        else:
            up = effort >= p["pulse_level"]
            gated = p["still_gate"] and moving
            self.point_blocked = self.point_blocked and point_on
            point = point_on and up and not self.point_blocked and not gated
            self.point_votes = self.point_votes + 1 if point else 0
            if up:
                self.pulse_time += dt
                self.pulse_open |= open_on or flick_on or gated
                if self.pulse_time >= p["long_s"] - EPS and not self.pulse_open:
                    event = self._choose("fist")
                elif self.point_votes >= p["vote_n"]:
                    event = self._choose("point")
            elif self.pulse_time > 0:
                if not self.pulse_open:
                    self.pulses += 1
                self.pulse_time, self.pulse_open, self.gap = 0.0, False, 0.0
            else:
                self.gap += dt
                if self.pulses and self.gap >= p["gap_s"] - EPS:
                    i = self.pulses - int(p["pinch_clenches"])
                    self.pulses = 0
                    if 0 <= i < len(TRIGGER_GRIPS) and TRIGGER_GRIPS[i] in self.poses:
                        event = self._choose(TRIGGER_GRIPS[i])

        return {"state": self.state, "grip": self.grip, "closure": self.closure, "angles": self._spring(dt),
                "event": event, "pulses": self.pulses}

    @staticmethod
    def _repeats(rules, t, count):
        return [r for r in rules if r["trigger"] == t and r["how"] == "repeated" and r["value"] == count]

    def _rules(self, probs, effort, dt, moving=False):
        p = self.params
        # Only the list of the state the hand is in counts and times; a class counts as on over both lists, so a
        # sign held while holding still isn't taken for squeezing
        rules = self.rules["relaxed" if self.state == "idle" else "holding"]
        if self.locked:
            rules = [r for r in rules if r["action"] in ("lock", "unlock")]
        else:
            rules = [r for r in rules if r["action"] != "unlock"]
        triggers = sorted({r["trigger"] for r in rules})
        on = {t: float(probs.get(t, 0.0)) >= p["threshold"]
              for t in {r["trigger"] for s in SECTIONS for r in self.rules[s]} if t != "clench"}
        class_on = any(on.values())
        on["clench"] = effort >= p["pulse_level"]
        gated = p["still_gate"] and moving and self.state == "idle"
        held = {r["trigger"]: r for r in rules if r["how"] == "held"}
        events = []

        for t in triggers:
            if t in self.stuck:
                if on[t]:
                    continue
                self.stuck.discard(t)
            now = on[t] and not gated
            run, used = self.run_t.get(t, 0.0), self.used.get(t, False)
            if now:
                if run == 0:
                    # Counted as it starts, so the last of a repeat acts straight away; a hold takes it back
                    self.counts[t] = self.counts.get(t, 0) + 1
                    counts = [r["value"] for r in rules if r["trigger"] == t and r["how"] == "repeated"]
                    if counts and self.counts[t] >= max(counts):
                        events += self._repeats(rules, t, self.counts[t])
                        self.counts[t] = 0
                        used = True
                run += dt
                self.off_t[t] = 0.0
                rule = held.get(t)
                if not used and ((t == "clench" and class_on) or (rule and run >= rule["value"] - EPS)):
                    used = True
                    self.counts[t] -= 1
                    if not (t == "clench" and class_on):
                        events.append(rule)
            else:
                self.off_t[t] = self.off_t.get(t, 0.0) + dt
                # A class has to stay off a little to end an episode, so a one window drop isn't a second flick
                if run > 0 and (t == "clench" or self.off_t[t] >= FLICK_GAP_S - EPS):
                    run, used = 0.0, False
                elif run == 0 and self.counts.get(t) and self.off_t[t] >= p["gap_s"] - EPS:
                    events += self._repeats(rules, t, self.counts[t])
                    self.counts[t] = 0
            self.run_t[t], self.used[t] = run, used

        event = None
        # Two rules firing on the same step: the higher one in the list wins
        events.sort(key=lambda r: next(i for i, x in enumerate(rules) if x is r))
        for r in events:
            if r["action"] == "release" and self.state == "active":
                self.state, self.grip, self.closure, event = "idle", None, 0.0, "release"
            elif r["action"] in ("lock", "unlock") and self.state == "active":
                self.locked = r["action"] == "lock" and not self.locked
                event = "lock" if self.locked else "unlock"
            elif r["action"] not in RULE_ACTIONS and r["action"] in self.poses:
                # A pose while holding switches to it at the closure the hand has
                closure = self.closure if self.state == "active" else 0.0
                self._choose(r["action"])
                self.closure, event = closure, "start"
            else:
                continue
            # Whatever is still on carried the pick or release, it has to go off before it counts again
            self.stuck = {t for t in on if on[t]}
            self.run_t, self.off_t, self.counts, self.used = {}, {}, {}, {}
            break

        if self.state == "active" and event is None and not self.locked:
            moving_to = {r["action"] for r in rules if r["action"] in ("open", "close") and on[r["trigger"]]
                         and self.run_t.get(r["trigger"], 0.0) >= r["value"] - EPS}
            speed = max(effort, p["hold_level"]) * p["gain"] * dt
            if "open" in moving_to:
                self.closure -= speed
            elif "close" in moving_to:
                self.closure += speed
            elif class_on:
                pass
            elif effort >= p["hold_level"]:
                self.closure += (effort - p["hold_level"]) * p["gain"] * dt
            self.closure = float(np.clip(self.closure, 0.0, 1.0))
        return {"state": self.state, "grip": self.grip, "closure": self.closure, "angles": self._spring(dt),
                "event": event, "pulses": self.counts.get("clench", 0), "locked": self.locked}

    def _choose(self, grip):
        self.state, self.grip, self.closure = "active", grip, 0.0
        self.flicks, self.pulses, self.point_votes = [], 0, 0
        self.pulse_time, self.pulse_open = 0.0, False
        return "start"

    def _released(self, probs, effort, dt):
        """Counts one step towards each release, true when the release mode's condition is met."""
        p = self.params
        prob = lambda g: float(probs.get(g, 0.0))
        self.open_votes = self.open_votes + 1 if prob("open") >= p["threshold"] else 0
        self.class_votes = self.class_votes + 1 if prob("open") + prob("rest") >= p["threshold"] else 0
        self.rest_time = self.rest_time + dt if effort < p["release_level"] else 0.0
        by_open = self.open_votes >= p["release_n"]
        by_class = self.class_votes >= p["release_n"]
        by_rest = self.rest_time >= p["release_s"] - EPS
        mode = p["release_mode"]
        return ((by_open and mode in ("open_class", "either")) or (by_class and mode == "rest_or_class")
                or (by_rest and mode in ("rest", "either", "rest_or_class")))

    def _closure(self, effort, dt):
        p = self.params
        if p["closure_mode"] == "speed":
            change = (effort - p["hold_level"]) * p["gain"] * dt
            c = self.closure + (max(0.0, change) if p["hold_on_relax"] else change)
        else:
            c = (effort - p["start_level"]) * p["gain"]
            if p["hold_on_relax"]:
                c = max(c, self.closure)
        return float(np.clip(c, 0.0, 1.0))

    def _spring(self, dt, target=None):
        """Critically damped spring towards the target pose, the same as follow() in web/app.js."""
        if target is None:
            pose = self.poses[self.grip] if self.grip else self.relaxed
            target = self.relaxed + self.closure * (pose - self.relaxed)
        w = self.params["spring_w"]
        if w <= 0:
            self.x, self.v = target.copy(), np.zeros_like(target)
            return self.x.tolist()
        n = max(1, math.ceil(dt / SPRING_STEP_S))
        h = dt / n
        for _ in range(n):
            self.v += h * (w * w * (target - self.x) - 2 * w * self.v)
            self.x += h * self.v
        return self.x.tolist()
