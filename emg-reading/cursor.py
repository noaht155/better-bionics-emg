"""Abstract control: a cursor steered by muscle activity through a fixed map, with targets to reach. Nothing is
trained, the person learns the map instead (abstract myoelectric control, Dyson, Barnes and Nazarpour). Each
channel's level (envelope between rest and max, the haptics' levels) pulls the cursor towards that electrode's
direction round the ring, so which side of the forearm works sets the direction and how hard sets the distance.
Relaxed sits in the middle. Channels 0 to 5 run in order round the arm (neighbours correlate 0.25 to 0.53 across
grips, all sessions to 2026-10-15); 6 and 7 on the base module don't follow that pattern, the person learns them
like the rest.

Targets come in shuffled blocks, one at each angle. A trial: hold the cursor in the middle for home_s, then a target
shows; it counts as hit once the cursor stays inside for dwell_s, missed after timeout_s.
"""
import math
import random

# Starting point, easy on purpose: on 2026-10-15_140633 the dot wobbled 0.11 within one held grip at 0.3 s smoothing
# (0.16 at 0.15 s) and 0.01 to 0.02 at rest. Natural grips all landed between 130 and 185 deg (pinch near 0), so the
# other directions need new contractions; that is the part to learn
DEFAULTS = {"gain": 1.5, "smooth_s": 0.3, "rotation_deg": 0.0, "flip": False, "targets": 4, "target_r": 0.6,
            "target_size": 0.25, "home_size": 0.15, "home_s": 0.5, "dwell_s": 0.5, "timeout_s": 6.0}
# Channel levels for the cursor (calibrate.levels scale): rest 95th percentile x 1.2, max 95th percentile of the grips.
# The haptics' scale (99th x 1.5, max 80th) keeps outputs off at rest at any cost; on 2026-10-15_140633 it left ch 1 and
# 6 at 0.45 and 0.30 in a grip (90th percentile) while ch 2, 3 and 5 were at 1.0, so the dot leaned to the strong side.
# This scale gives 0.69 to 0.85 on every channel and the dot stays inside the home zone at rest (95th percentile 0.09)
LEVEL_SCALE = (95, 1.2, 95)
# The cursor can go a little past the target ring, not off the screen
MAX_R = 1.2


class Cursor:
    def __init__(self, params):
        self.params = params
        self.x = self.y = 0.0
        self.targets_on = False
        self._block = []
        self.reset_task()

    def reset_task(self):
        self.phase = "home"
        self.target = None
        self.t = 0.0
        self.dwell = 0.0
        self.entries = 0
        self.path = []

    def angles(self):
        n = int(self.params["targets"])
        return [90 + 360 * k / n for k in range(n)]

    def map(self, levels):
        """Where the levels put the cursor, before smoothing. levels: 0 to 1 per channel in ring order."""
        p = self.params
        sign = -1 if p["flip"] else 1
        n = len(levels)
        x = y = 0.0
        for i, level in enumerate(levels):
            a = math.radians(p["rotation_deg"]) + sign * 2 * math.pi * i / n
            x += level * math.cos(a)
            y += level * math.sin(a)
        x, y = x * p["gain"], y * p["gain"]
        r = math.hypot(x, y)
        return (x * MAX_R / r, y * MAX_R / r) if r > MAX_R else (x, y)

    def step(self, levels, dt):
        """Moves the cursor and runs the task. Returns a finished trial (dict) or None."""
        p = self.params
        tx, ty = self.map(levels)
        a = 1.0 if p["smooth_s"] <= 0 else min(1.0, dt / p["smooth_s"])
        self.x += a * (tx - self.x)
        self.y += a * (ty - self.y)
        return self._task(levels, dt) if self.targets_on else None

    def _task(self, levels, dt):
        p = self.params
        self.t += dt
        if self.phase == "home":
            self.t = self.t if math.hypot(self.x, self.y) <= p["home_size"] else 0.0
            if self.t >= p["home_s"]:
                if not self._block:
                    self._block = list(range(int(p["targets"])))
                    random.shuffle(self._block)
                self.target = self._block.pop()
                self.phase, self.t, self.dwell, self.entries, self.path = "go", 0.0, 0.0, 0, []
            return None
        a = math.radians(self.angles()[self.target % int(p["targets"])])
        cx, cy = p["target_r"] * math.cos(a), p["target_r"] * math.sin(a)
        inside = math.hypot(self.x - cx, self.y - cy) <= p["target_size"]
        if inside and self.dwell == 0:
            self.entries += 1
        self.dwell = self.dwell + dt if inside else 0.0
        self.path.append([round(self.t, 2), round(self.x, 3), round(self.y, 3)] + [round(v, 3) for v in levels])
        hit = self.dwell >= p["dwell_s"]
        if not hit and self.t < p["timeout_s"]:
            return None
        trial = {"target": self.target, "angle": round(math.degrees(a) % 360, 1), "hit": hit,
                 "time_s": round(self.t, 2), "entries": self.entries, "path": self.path}
        self.reset_task()
        return trial
