"""Gestures, arm postures and the cue sequence of a recording session.

Target angles are rough poses for the on-screen skeleton and the pose match score, not labels.
The labels for protocol B are the tracked angles.
"""
import random

from hand_angles import FINGERS, JOINTS


def _pose(thumb, index, middle, ring, pinky):
    """thumb: (cmc_flex, cmc_abd, mcp, ip), fingers: (mcp_flex, pip, dip) or (mcp_flex, pip, dip, abd)."""
    angles = dict(zip(JOINTS[:4], thumb))
    for name, f in zip(("index", "middle", "ring", "pinky"), (index, middle, ring, pinky)):
        angles.update({f"{name}_mcp_flex": f[0], f"{name}_pip": f[1], f"{name}_dip": f[2],
                       f"{name}_mcp_abd": f[3] if len(f) > 3 else 0})
    return [float(angles[j]) for j in JOINTS]


RELAXED = (20, 30, 15)
CURLED = (85, 100, 55)
STRAIGHT = (0, 0, 0)
# Thumb angles fitted so the thumb tip lands where it should on the skeleton in web/hand.js
THUMB_OVER = (40, 30, 15, 55)
THUMB_TO_INDEX = (40, 40, 0, 5)

# name: (cue text, target angles)
GESTURES = {
    "rest": ("RELAX\nhand loose", _pose((15, 20, 15, 15), RELAXED, RELAXED, RELAXED, (25, 35, 15))),
    "open": ("OPEN HAND\nfingers straight and spread",
             _pose((-10, 10, 5, 5), (0, 0, 0, 8), STRAIGHT, (0, 0, 0, -5), (0, 0, 0, -10))),
    "fist": ("FIST\nthumb over the fingers", _pose(THUMB_OVER, CURLED, CURLED, CURLED, CURLED)),
    "pinch": ("PINCH\nthumb tip to index tip, others loose",
              _pose(THUMB_TO_INDEX, (45, 55, 25), RELAXED, RELAXED, RELAXED)),
    "point": ("POINT\nindex finger out", _pose(THUMB_OVER, STRAIGHT, CURLED, CURLED, CURLED)),
    "peace": ("PEACE\nindex and middle out", _pose((50, 25, 35, 35), (0, 0, 0, 10), (0, 0, 0, -8), CURLED, CURLED)),
    "thumbs_up": ("THUMBS UP", _pose((0, 20, 0, 0), CURLED, CURLED, CURLED, CURLED)),
    "ok": ("OK\nthumb and index ring, others straight",
           _pose(THUMB_TO_INDEX, (45, 55, 25), (5, 5, 0), (5, 5, 0, -5), (5, 5, 0, -10))),
    "tripod": ("TRIPOD\nthumb, index and middle tips together",
               _pose((40, 40, 15, 10), (55, 60, 30), (55, 60, 30), (40, 50, 20), (40, 55, 20))),
    "key": ("KEY PINCH\nthumb pad on the side of the index", _pose((15, 10, 45, 5), (60, 90, 40), (60, 90, 40),
                                                                  (60, 90, 40), (60, 90, 40))),
    "hook": ("HOOK\nknuckles straight, fingers curled", _pose((0, 15, 5, 5), (0, 90, 60), (0, 90, 60),
                                                             (0, 90, 60), (0, 90, 60))),
}

# The gestures protocol A uses and the sessions cue, chosen on 2026-10-01: the common prosthetic grips. Thumbs up,
# peace and OK are signs rather than grips, hook scored 51 % and was often taken for peace. With these 7 one band position
# scored 87 % (73 % with all 11). The others stay above so older sessions and models still display
GESTURE_SET = ["rest", "open", "fist", "pinch", "tripod", "key", "point"]

POSTURES = {
    "table": "Forearm resting on the table",
    "forward": "Arm held straight forward",
    "raised": "Arm raised, hand above the shoulder",
}

# Gentle gestures barely rose above rest with the arm up on 2026-10-01 and were the ones the model missed.
# Medium rather than maximum effort, the model still has to work on everyday grips
HOLD_TEXT = "hold it firmly and steady, medium effort"
SYNC_TAPS = 3
SYNC_TEXT = "TAP\nlift your hand and slap the table once with your palm, sharply"
FREE_TEXT = "MOVE YOUR FINGERS\nslowly, any way: open, close, one finger at a time"

# Protocol B has to learn each finger on its own. The held grips are 7 fixed shapes, and in free movement the fingers
# mostly moved together (middle and ring correlated 0.87, ring and pinky 0.93 on 2026-10-01). In every posture
FINGER_S = 8.0
WAVE_S = 10.0
FINGER_TEXT = "FLEX YOUR {}\nslowly bend and straighten it 3 times, keep the others still"
WAVE_TEXT = "WAVE YOUR FINGERS\none after another, like drumming, slowly"


def _bent(finger):
    """Open hand with one finger bent, the target shown for that finger's cue."""
    pose = dict(zip(JOINTS, GESTURES["open"][1]))
    if finger == "thumb":
        pose.update(thumb_cmc_flex=45, thumb_cmc_abd=30, thumb_mcp=40, thumb_ip=40)
    else:
        pose.update({f"{finger}_mcp_flex": 60, f"{finger}_pip": 80, f"{finger}_dip": 40})
    return [float(pose[j]) for j in JOINTS]


for _f in FINGERS:
    GESTURES[f"flex_{_f}"] = (FINGER_TEXT.format(_f.upper()), _bent(_f))


def _cue(kind, label, text, seconds, posture):
    return {"kind": kind, "label": label, "text": text, "seconds": seconds, "posture": posture}


def _sync_block():
    # Sharp taps show up in the accelerometer and on camera, so the camera delay can be measured afterwards
    cues = [_cue("break", "setup", "Forearm on the table for the sync taps.\nPress Continue when ready.", None, "table"),
            _cue("hold", "rest", GESTURES["rest"][0], 2, "table")]
    for _ in range(SYNC_TAPS):
        cues += [_cue("sync", "sync_tap", SYNC_TEXT, 2, "table"), _cue("hold", "rest", GESTURES["rest"][0], 2, "table")]
    return cues


def session_plan(postures, reps=3, hold_s=4.0, rest_s=3.0, free_s=30.0, seed=None):
    """Cue list: sync taps, then per posture every gesture of GESTURE_SET reps times in shuffled order with rest in
    between, then free movement, and sync taps again at the end.

    In every posture each finger is also bent on its own, then all of them one after another.

    kind is "hold" (a held gesture or rest, protocol A), "finger" (one finger moving, protocol B), "free"
    (continuous movement), "sync" (a tap) or "break" (waits for Continue). seconds is None for breaks."""
    rng = random.Random(seed)
    moves = [g for g in GESTURE_SET if g != "rest"]
    rest = GESTURES["rest"][0]
    cues = _sync_block()
    for posture in postures:
        cues.append(_cue("break", "setup", f"{POSTURES[posture]}.\nPress Continue when ready.", None, posture))
        cues.append(_cue("hold", "rest", rest, rest_s, posture))
        for _ in range(reps):
            rng.shuffle(moves)
            for g in moves:
                cues += [_cue("hold", g, f"{GESTURES[g][0]}\n{HOLD_TEXT}", hold_s, posture),
                         _cue("hold", "rest", rest, rest_s, posture)]
        for f in FINGERS:
            cues += [_cue("finger", f"flex_{f}", GESTURES[f"flex_{f}"][0], FINGER_S, posture),
                     _cue("hold", "rest", rest, rest_s, posture)]
        cues += [_cue("finger", "wave", WAVE_TEXT, WAVE_S, posture), _cue("hold", "rest", rest, rest_s, posture)]
        if free_s > 0:
            cues += [_cue("free", "free", FREE_TEXT, free_s, posture), _cue("hold", "rest", rest, rest_s, posture)]
    return cues + _sync_block()
