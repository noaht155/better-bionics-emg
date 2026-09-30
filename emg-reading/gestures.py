"""Gestures, arm postures and the cue sequence of a recording session.

Target angles are rough poses for the on-screen skeleton and the pose match score, not labels.
The labels for protocol B are the tracked angles.
"""
import random

from hand_angles import JOINTS


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

POSTURES = {
    "table": "Forearm resting on the table",
    "forward": "Arm held straight forward",
    "raised": "Arm raised, hand above the shoulder",
}

SYNC_TAPS = 3
SYNC_TEXT = "TAP\nslap the table once with your palm, sharply"
FREE_TEXT = "MOVE YOUR FINGERS\nslowly, any way: open, close, one finger at a time"


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
    """Cue list: sync taps, then per posture every gesture reps times in shuffled order with rest in between,
    then free movement, and sync taps again at the end.

    kind is "hold" (a held gesture or rest, protocol A), "free" (continuous movement), "sync" (a tap) or
    "break" (waits for Continue). seconds is None for breaks."""
    rng = random.Random(seed)
    moves = [g for g in GESTURES if g != "rest"]
    rest = GESTURES["rest"][0]
    cues = _sync_block()
    for posture in postures:
        cues.append(_cue("break", "setup", f"{POSTURES[posture]}.\nPress Continue when ready.", None, posture))
        cues.append(_cue("hold", "rest", rest, rest_s, posture))
        for _ in range(reps):
            rng.shuffle(moves)
            for g in moves:
                cues += [_cue("hold", g, GESTURES[g][0], hold_s, posture), _cue("hold", "rest", rest, rest_s, posture)]
        if free_s > 0:
            cues += [_cue("free", "free", FREE_TEXT, free_s, posture), _cue("hold", "rest", rest, rest_s, posture)]
    return cues + _sync_block()
