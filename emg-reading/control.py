"""Control layer between the grip model and the hand: one locked grip and how far the hand closes on it.

Called once per 50 ms window with the grip probabilities and the effort (0 to 1, the envelope scaled between the
calibrated rest and max levels, mean over the good channels), and optionally the window's distance from the winning
grip's calibration windows (gesture_model.LivePredictor.distance). The grip is only chosen while effort rises from rest,
then stays locked until release, so it never switches mid-grasp. Closure follows effort along the path from the
relaxed pose to the grip's pose. Used by the app's Control tab and the offline replay.
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
            "release_level": 0.03, "release_s": 1.5, "max_distance": 6.0, "spring_w": 12}
# Same integration step as the drawn hands in web/app.js
SPRING_STEP_S = 0.005
# Sums of dt drift a little, 12 x 0.05 isn't exactly 0.6
EPS = 1e-9


class Controller:
    def __init__(self, params, grips, poses):
        # Read on every step, so a changed value applies straight away
        self.params = dict(DEFAULTS, **(params or {}))
        self.grips = list(grips)
        self.poses = {g: np.asarray(a, float) for g, a in poses.items()}
        self.relaxed = self.poses["rest"]
        self.reset()

    def reset(self):
        self.state = "idle"
        self.grip = None
        self.closure = 0.0
        self.above = False
        self.onset = 0.0
        self.blocked = False
        self.candidate, self.votes = None, 0
        self.open_votes, self.class_votes, self.rest_time = 0, 0, 0.0
        self.x = self.relaxed.copy()
        self.v = np.zeros_like(self.x)

    def step(self, probs, effort, dt=0.05, distance=None):
        p = self.params
        if not isinstance(probs, dict):
            probs = dict(zip(self.grips, np.asarray(probs, float)))
        if p["max_distance"] > 0 and distance is not None and distance > p["max_distance"]:
            probs = {"rest": 1.0}
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
            self.open_votes = self.open_votes + 1 if prob("open") >= p["threshold"] else 0
            self.class_votes = self.class_votes + 1 if prob("open") + prob("rest") >= p["threshold"] else 0
            self.rest_time = self.rest_time + dt if effort < p["release_level"] else 0.0
            by_open = self.open_votes >= p["release_n"]
            by_class = self.class_votes >= p["release_n"]
            by_rest = self.rest_time >= p["release_s"] - EPS
            mode = p["release_mode"]
            if ((by_open and mode in ("open_class", "either")) or (by_class and mode == "rest_or_class")
                    or (by_rest and mode in ("rest", "either", "rest_or_class"))):
                self.state, self.grip, self.closure, event = "idle", None, 0.0, "release"
                self.blocked = True
                self.candidate, self.votes = None, 0
        if self.blocked:
            self.candidate, self.votes = None, 0

        return {"state": self.state, "grip": self.grip, "closure": self.closure, "angles": self._spring(dt),
                "event": event}

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

    def _spring(self, dt):
        """Critically damped spring towards the target pose, the same as follow() in web/app.js."""
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
