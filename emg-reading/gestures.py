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
FREE_TEXT = "MOVE YOUR FINGERS\nslowly, any way: open, close, one finger at a time"

# Protocol B has to learn each finger on its own. The held grips are 7 fixed shapes, and in free movement the fingers
# mostly moved together (middle and ring correlated 0.87, ring and pinky 0.93 on 2026-10-01). In every posture
# The pause halfway is there because the grips are all end positions, so a half-bent finger was hardly ever recorded
FINGER_S = 12.0
WAVE_S = 10.0
FINGER_TEXT = ("FLEX YOUR {}\nbend it slowly, stop halfway for a second, bend it fully, then back. 3 times, keep the "
               "others still")
# The thumb cue was done with small movements up to 2026-10-05, thumb MCP then covered only about 25 degrees
THUMB_TEXT = ("FLEX YOUR THUMB\nbend it fully across the palm, stop halfway on the way, then out as far as it goes. "
              "3 times, keep the others still")
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
    GESTURES[f"flex_{_f}"] = (THUMB_TEXT if _f == "thumb" else FINGER_TEXT.format(_f.upper()), _bent(_f))


# Protocol B, after Ninapro DB8 (Krasoulis 2019): its 9 movements, each a slow close and open following the animated
# target hand. Thumb rotation is replaced by abduction, the camera follows it better
MOVE_S = 5.0
MOVE_ROUNDS = 6
MOVE_FOLLOW = "follow the hand: close slowly, then open"
CYLINDER = (60, 70, 40)


def _shape(**joints):
    pose = dict(zip(JOINTS, GESTURES["open"][1]))
    pose.update(joints)
    return [float(pose[j]) for j in JOINTS]


MOVEMENTS = {
    "move_thumb_flex": ("THUMB FLEX\nbend it across the palm and back", _bent("thumb")),
    "move_thumb_abd": ("THUMB OUT AND IN\nmove it away from the index and back",
                       _shape(thumb_cmc_flex=-20, thumb_cmc_abd=60, thumb_mcp=5, thumb_ip=5)),
    "move_index": ("INDEX\nbend it fully and back", _bent("index")),
    "move_middle": ("MIDDLE\nbend it fully and back", _bent("middle")),
    "move_ring_little": ("RING AND LITTLE\nbend both together and back",
                         _shape(ring_mcp_flex=60, ring_pip=80, ring_dip=40, pinky_mcp_flex=60, pinky_pip=80,
                                pinky_dip=40)),
    "move_point": ("POINT\ncurl all but the index and back", GESTURES["point"][1]),
    "move_cylinder": ("CYLINDER GRIP\nclose round a bottle and back", _pose(THUMB_OVER, CYLINDER, CYLINDER, CYLINDER,
                                                                           CYLINDER)),
    "move_key": ("KEY PINCH\nthumb pad to the side of the index and back", GESTURES["key"][1]),
    "move_tripod": ("TRIPOD\nthumb, index and middle tips together and back", GESTURES["tripod"][1]),
}
GESTURES.update(MOVEMENTS)


# Protocol C: movements the person picks from this library, for record and play. Grips from the GRASP taxonomy
# (Feix 2016, numbers below), Cutkosky 1989, Ninapro DB1/DB2 exercises A to C and the grip lists of bebionic,
# i-Limb, TASKA, Ability Hand (Psyonic) and Michelangelo. Hand only, the drawing has no wrist. common marks the grips
# used most in daily life (Bullock 2013 top 10, Vergara 2014 top 5) and the ones every multi-grip hand ships with.
# Entries that already exist above keep their key and angles. partial: stored as the open hand with some joints
# changed, so only those joints move and the rest of the hand stays relaxed. keys: dynamic movements, a list of
# (share of the cue time, angles) that starts and ends at the relaxed hand
def _relaxed(**joints):
    pose = dict(zip(JOINTS, GESTURES["rest"][1]))
    pose.update(joints)
    return [float(pose[j]) for j in JOINTS]


def _finger(name, joints):
    """joints (mcp_flex, pip, dip) or (mcp_flex, pip, dip, abd) as keyword arguments for _relaxed or _shape."""
    out = {f"{name}_mcp_flex": joints[0], f"{name}_pip": joints[1], f"{name}_dip": joints[2]}
    if len(joints) > 3:
        out[f"{name}_mcp_abd"] = joints[3]
    return out


LIBRARY = {}


def _lib(key, name, category, sources, text, angles, common=False, partial=False, keys=None):
    LIBRARY[key] = {"name": name, "category": category, "sources": sources, "text": text,
                    "angles": [float(a) for a in angles], "common": common, "partial": partial}
    if keys:
        LIBRARY[key]["keys"] = [[float(t), [float(a) for a in pose]] for t, pose in keys]


def _cycle(pose_a, pose_b, times, start=0.12, end=0.88):
    """Keyframes that go relaxed, pose_a, then pose_b and back to pose_a `times` times, relaxed again."""
    rest = GESTURES["rest"][1]
    step = (end - start) / (2 * times)
    keys = [(0, rest), (start, pose_a)]
    for i in range(times):
        keys += [(start + (2 * i + 1) * step, pose_b), (start + (2 * i + 2) * step, pose_a)]
    return keys + [(1, rest)]


CATEGORIES = {"power": "Power grips", "precision": "Precision grips", "lateral": "Lateral and flat grips",
              "point": "Point, press and tools", "sign": "Signs and gestures", "finger": "Single fingers",
              "dynamic": "Moving gestures"}

_lib("cylinder", "Cylinder", "power", "Feix 3 medium wrap, Cutkosky, Ninapro C, bebionic power, Ability Hand "
     "cylinder, TASKA general grasp", "CYLINDER GRIP\nclose round a bottle", MOVEMENTS["move_cylinder"][1], True)
_lib("fist", "Fist", "power", "Ninapro B fist, Ability Hand power, Coapt hand close", GESTURES["fist"][0],
     GESTURES["fist"][1], True)
_lib("large_diameter", "Large diameter", "power", "Feix 1, Cutkosky, Ninapro C",
     "LARGE DIAMETER\nwhole hand round a big jar or a thick pole",
     _pose((25, 55, 10, 10), (35, 35, 20), (35, 35, 20), (35, 35, 20), (35, 40, 20)))
_lib("small_diameter", "Small diameter", "power", "Feix 2, Cutkosky, Ninapro C",
     "SMALL DIAMETER\nfingers wrapped tight round a thin handle", _pose((40, 30, 20, 40), (75, 85, 45), (75, 85, 45),
                                                                       (75, 85, 45), (75, 85, 45)))
_lib("light_tool", "Light tool", "power", "Feix 5, Cutkosky, Bullock top 10",
     "LIGHT TOOL\nhandle across the fingers, thumb resting along it",
     _pose((5, 5, 10, 5), (65, 75, 40), (65, 75, 40), (65, 75, 40), (65, 75, 40)), True)
_lib("index_extension", "Index extension", "power", "Feix 17, Ninapro C, Bullock top 10",
     "INDEX ON TOP\nhold a knife, index straight along the back of the blade",
     _pose((40, 30, 15, 40), (20, 15, 5), CYLINDER, CYLINDER, CYLINDER), True)
_lib("power_sphere", "Power sphere", "power", "Feix 11, Cutkosky, Ninapro C, Bullock top 10",
     "POWER SPHERE\nwhole hand round an apple, fingers spread",
     _pose((30, 55, 15, 15), (45, 45, 25, 12), (45, 45, 25, 0), (45, 45, 25, -10), (45, 45, 25, -18)), True)
_lib("three_finger_sphere", "Three finger sphere", "power", "Feix 28, Ninapro C",
     "THREE FINGER SPHERE\nthumb, index and middle round a ball, ring and little tucked in",
     _pose((30, 55, 15, 15), (50, 50, 30, 10), (50, 50, 30, 0), CURLED, CURLED))
_lib("power_disk", "Power disk", "power", "Feix 10, Cutkosky, Ninapro C",
     "POWER DISK\npalm on a jar lid, fingers spread round its edge",
     _pose((20, 40, 10, 10), (30, 30, 15, 14), (30, 30, 15, 0), (30, 30, 15, -12), (30, 30, 15, -24)))
_lib("hook", "Hook", "power", "Feix 15, Cutkosky, Ninapro C, bebionic, Ability Hand, TASKA",
     "HOOK\nknuckles straight, fingers curled, carry a bag", GESTURES["hook"][1], True)
_lib("handshake", "Handshake", "power", "Ability Hand handshake",
     "HANDSHAKE\nhalfway between a key pinch and a fist", _pose((10, 15, 20, 10), (45, 50, 25), (45, 50, 25),
                                                                (45, 50, 25), (45, 50, 25)))
_lib("lateral_power", "Lateral power", "power", "Michelangelo lateral power grip",
     "LATERAL POWER\nfingers round a phone, thumb pressing on the side of the index",
     _pose((15, 10, 40, 10), CYLINDER, CYLINDER, CYLINDER, CYLINDER))

_lib("pinch", "Pinch", "precision", "Feix 24 tip pinch, Feix 9, Cutkosky, Ninapro C, Vergara top 5, bebionic "
     "precision open, Ability Hand, Coapt fine pinch", GESTURES["pinch"][0], GESTURES["pinch"][1], True)
_lib("precision_closed", "Pinch, others closed", "precision", "bebionic precision closed, TASKA pincer",
     "PINCH, OTHERS CLOSED\nthumb tip to index tip, the other fingers in the palm",
     _pose(THUMB_TO_INDEX, (45, 55, 25), CURLED, CURLED, CURLED), True)
_lib("tripod", "Tripod", "precision", "Feix 14, Cutkosky, Ninapro C, Bullock top 10, bebionic, i-Limb, Ability "
     "Hand, TASKA, Michelangelo", GESTURES["tripod"][0], GESTURES["tripod"][1], True)
_lib("tripod_open", "Tripod, others open", "precision", "Ability Hand tripod open",
     "TRIPOD, OTHERS OPEN\nthumb, index and middle tips together, ring and little straight",
     _pose((40, 40, 15, 10), (55, 60, 30), (55, 60, 30), (5, 5, 0, -5), (5, 5, 0, -10)))
_lib("quadpod", "Quadpod", "precision", "Feix 27, Ninapro C",
     "QUADPOD\nthumb, index, middle and ring tips together",
     _pose((40, 45, 15, 10), (55, 60, 30, 0), (55, 60, 30), (55, 60, 30, 4), (45, 55, 25)))
_lib("writing_tripod", "Writing tripod", "precision", "Feix 20, Ninapro C",
     "WRITING TRIPOD\nhold a pen to write", _pose((40, 40, 10, 15), (45, 45, 20), (55, 55, 30), (70, 80, 45),
                                                   (75, 85, 45)))
_lib("precision_disk", "Precision disk", "precision", "Feix 12, Cutkosky, Bullock top 10 (2nd, 17 %)",
     "PRECISION DISK\nfingertips round a jar lid, palm off it",
     _pose((20, 55, 10, 10), (25, 35, 20, 15), (25, 35, 20, 2), (25, 35, 20, -12), (25, 35, 20, -25)), True)
_lib("precision_sphere", "Precision sphere", "precision", "Feix 13, Cutkosky, Ninapro C",
     "PRECISION SPHERE\nfingertips round a ball, palm off it",
     _pose((30, 50, 10, 10), (35, 35, 25, 12), (35, 35, 25, 0), (35, 35, 25, -10), (35, 35, 25, -20)))
_lib("thumb_2_finger", "Thumb and 2 fingers", "precision", "Feix 8 prismatic 2 finger, Cutkosky, Bullock top 10",
     "THUMB AND 2 FINGERS\nthumb pad against the index and middle pads, a thin box",
     _pose((35, 45, 5, 5), (45, 20, 10), (45, 20, 10), RELAXED, RELAXED), True)
_lib("thumb_3_finger", "Thumb and 3 fingers", "precision", "Feix 7 prismatic 3 finger, Cutkosky, Bullock top 10",
     "THUMB AND 3 FINGERS\nthumb pad against the index, middle and ring pads",
     _pose((35, 45, 5, 5), (45, 20, 10), (45, 20, 10), (45, 20, 10), RELAXED), True)
_lib("thumb_4_finger", "Thumb and 4 fingers", "precision", "Feix 6 prismatic 4 finger, Cutkosky, Ninapro C",
     "THUMB AND 4 FINGERS\nthumb pad against all four finger pads, a book",
     _pose((35, 45, 5, 5), (45, 20, 10), (45, 20, 10), (45, 20, 10), (45, 20, 10)))
_lib("parallel_extension", "Parallel extension", "precision", "Feix 22, Ninapro C, Feix 18 extension type",
     "FLAT PINCH\nfingers straight from the knuckles, thumb under them, a plate edge",
     _pose((40, 45, 10, 5), (60, 0, 0), (60, 0, 0), (60, 0, 0), (60, 0, 0)))

_lib("key", "Key pinch", "lateral", "Feix 16 lateral, Cutkosky, Ninapro C, Bullock top 10 (3rd), Vergara top 5, "
     "every multi-grip hand", GESTURES["key"][0], GESTURES["key"][1], True)
_lib("lateral_tripod", "Lateral tripod", "lateral", "Feix 25, Bullock top 10",
     "LATERAL TRIPOD\nthumb pressing against the side of the middle finger, index on top",
     _pose((30, 20, 35, 15), (30, 40, 20), (60, 75, 35), CURLED, CURLED), True)
_lib("finger_adduction", "Between two fingers", "lateral", "Feix 23 adduction grip, Feix 29 stick, Ninapro C stick, "
     "bebionic finger adduction, Michelangelo",
     "BETWEEN TWO FINGERS\na card or a cigarette held between index and middle",
     _pose((15, 20, 15, 15), (15, 15, 5, -6), (15, 15, 5, 4), CURLED, CURLED))
_lib("mouse", "Mouse", "lateral", "bebionic, i-Limb, Ability Hand, TASKA",
     "MOUSE\nhand over a computer mouse, thumb on its side",
     _pose((10, 10, 20, 10), (25, 20, 10), (25, 20, 10), (25, 20, 10), (30, 25, 15)))
_lib("open_palm", "Open palm", "lateral", "bebionic, Michelangelo, TASKA, Ninapro B fingers together",
     "OPEN PALM\nfingers straight and together, thumb alongside, carry a tray",
     _pose((-5, 0, 5, 5), (0, 0, 0, -6), STRAIGHT, (0, 0, 0, 6), (0, 0, 0, 12)))
_lib("open", "Open hand", "lateral", "Ninapro B fingers spread, Coapt hand open", GESTURES["open"][0],
     GESTURES["open"][1], True)

_lib("point", "Point", "point", "Ninapro B pointing index, bebionic finger point, i-Limb, Ability Hand, TASKA pointer",
     GESTURES["point"][0], GESTURES["point"][1], True)
_lib("column", "Column", "point", "bebionic column",
     "COLUMN\nthumb in the palm, fingers closed over it, index firm for pressing buttons",
     _pose((55, 0, 30, 30), (35, 45, 20), CURLED, CURLED, CURLED))
_lib("sleeve", "Sleeve", "point", "Ability Hand sleeve, TASKA don and doff, i-Limb thumb park",
     "SLEEVE\nfingers straight and together, thumb across the palm, through a sleeve",
     _pose((60, 5, 30, 20), (0, 0, 0, -6), STRAIGHT, (0, 0, 0, 6), (0, 0, 0, 12)))
_TRIGGER = _pose((40, 30, 15, 40), (40, 45, 20), CYLINDER, CYLINDER, CYLINDER)
_lib("trigger", "Trigger", "dynamic", "bebionic active index, Ability Hand trigger, TASKA 1 finger trigger",
     "TRIGGER\nhold a spray bottle, squeeze the index three times", _TRIGGER, True,
     keys=_cycle(_TRIGGER, _pose((40, 30, 15, 40), (70, 90, 45), CYLINDER, CYLINDER, CYLINDER), 3))
_TRIGGER2 = _pose((40, 30, 15, 40), (40, 45, 20), (40, 45, 20), CYLINDER, CYLINDER)
_lib("two_finger_trigger", "Two finger trigger", "dynamic", "TASKA 2 finger trigger",
     "TWO FINGER TRIGGER\nsqueeze with index and middle three times", _TRIGGER2,
     keys=_cycle(_TRIGGER2, _pose((40, 30, 15, 40), (70, 90, 45), (70, 90, 45), CYLINDER, CYLINDER), 3))
_TAP = GESTURES["point"][1]
_lib("tap", "Tap", "dynamic", "TASKA keyboard, bebionic finger point",
     "TAP\npoint and tap a key with the index four times", _TAP,
     keys=_cycle(_TAP, _pose(THUMB_OVER, (30, 10, 5), CURLED, CURLED, CURLED), 4))

_lib("thumbs_up", "Thumbs up", "sign", "Ninapro B", "THUMBS UP\nthumb up, the fingers curled",
     GESTURES["thumbs_up"][1])
_lib("peace", "Peace", "sign", "Ninapro B index and middle out, Ability Hand peace", GESTURES["peace"][0],
     GESTURES["peace"][1])
_lib("ok", "OK", "sign", "Ability Hand chuck OK", GESTURES["ok"][0], GESTURES["ok"][1])
_lib("three", "Three", "sign", "Ninapro B ring and little flexed",
     "THREE\nring and little bent, the others straight",
     _pose((-10, 10, 5, 5), (0, 0, 0, 8), STRAIGHT, CURLED, CURLED))
_lib("thumb_to_little", "Thumb to little", "sign", "Ninapro B thumb to the base of the little finger",
     "THUMB TO LITTLE\nthumb tip to the base of the little finger, the others straight",
     _pose((70, 0, 35, 25), STRAIGHT, STRAIGHT, STRAIGHT, STRAIGHT))
_lib("horns", "Horns", "sign", "Ability Hand sign of the horns",
     "HORNS\nindex and little out, thumb over the others", _pose((50, 25, 35, 35), (0, 0, 0, 8), CURLED, CURLED,
                                                                 (0, 0, 0, -10)))
_lib("hang_loose", "Hang loose", "sign", "Ability Hand hang loose",
     "HANG LOOSE\nthumb and little out, the others bent", _pose((-10, 15, 0, 0), CURLED, CURLED, CURLED,
                                                                (0, 0, 0, -12)))
_lib("finger_gun", "L shape", "sign", "common sign",
     "L SHAPE\nthumb up, index out, the others bent", _pose((-10, 15, 0, 0), (0, 0, 0, 8), CURLED, CURLED, CURLED))
_lib("middle_finger", "Middle finger", "sign", "Ability Hand rude point",
     "MIDDLE FINGER\nmiddle out, the others bent", _pose((50, 25, 35, 35), CURLED, STRAIGHT, CURLED, CURLED))

_lib("move_thumb_flex", "Thumb flex", "finger", "Ninapro A, DB8", "THUMB FLEX\nbend it across the palm",
     MOVEMENTS["move_thumb_flex"][1], partial=True)
_lib("move_thumb_abd", "Thumb out", "finger", "Ninapro A, DB8", "THUMB OUT\nmove it away from the index",
     MOVEMENTS["move_thumb_abd"][1], partial=True)
_lib("move_index", "Index flex", "finger", "Ninapro A, DB8", "INDEX\nbend it fully", MOVEMENTS["move_index"][1],
     partial=True)
_lib("move_middle", "Middle flex", "finger", "Ninapro A, DB8", "MIDDLE\nbend it fully", MOVEMENTS["move_middle"][1],
     partial=True)
_lib("flex_ring", "Ring flex", "finger", "Ninapro A", "RING\nbend it fully, the others still",
     GESTURES["flex_ring"][1], partial=True)
_lib("flex_pinky", "Little flex", "finger", "Ninapro A", "LITTLE\nbend it fully, the others still",
     GESTURES["flex_pinky"][1], partial=True)
_lib("move_ring_little", "Ring and little", "finger", "Ninapro DB8", "RING AND LITTLE\nbend both together",
     MOVEMENTS["move_ring_little"][1], partial=True)

_DRUM = [(0, GESTURES["rest"][1])]
for _i in range(8):
    _DRUM.append((0.1 + 0.1 * _i, _relaxed(**_finger(FINGERS[1 + _i % 4], (65, 80, 40)))))
_lib("drum", "Finger drumming", "dynamic", "Ability Hand finger wave", "DRUMMING\nbend the fingers one after "
     "another, twice", _relaxed(**_finger("index", (65, 80, 40))), keys=_DRUM + [(0.9, GESTURES["rest"][1]),
                                                                                (1, GESTURES["rest"][1])])
_BECKON = _pose(THUMB_OVER, STRAIGHT, CURLED, CURLED, CURLED)
_lib("beckon", "Come here", "dynamic", "common sign", "COME HERE\ncurl the index in and out three times", _BECKON,
     keys=_cycle(_BECKON, _pose(THUMB_OVER, (50, 90, 50), CURLED, CURLED, CURLED), 3))
_COUNT3 = _pose((50, 25, 35, 35), (0, 0, 0, 8), STRAIGHT, (0, 0, 0, -5), CURLED)
_FOUR = _pose((60, 15, 30, 20), (0, 0, 0, 8), STRAIGHT, (0, 0, 0, -5), (0, 0, 0, -10))
_lib("count", "Count to five", "dynamic", "common sign", "COUNT TO FIVE\none finger at a time, index first, "
     "thumb last", _FOUR,
     keys=[(0, GESTURES["rest"][1]), (0.12, GESTURES["point"][1]), (0.24, GESTURES["point"][1]),
           (0.3, GESTURES["peace"][1]), (0.42, GESTURES["peace"][1]),
           (0.48, _COUNT3), (0.6, _COUNT3),
           (0.66, _FOUR), (0.74, _FOUR), (0.8, GESTURES["open"][1]), (0.88, GESTURES["open"][1]),
           (1, GESTURES["rest"][1])])
_SCISSORS_OPEN = _pose((-5, 30, 5, 5), (10, 10, 5, 10), (40, 40, 20), CURLED, CURLED)
_SCISSORS_SHUT = _pose((25, 30, 15, 10), (35, 30, 15, 0), (55, 50, 25), CURLED, CURLED)
_lib("scissors", "Scissors", "dynamic", "Feix 19 distal type, Ninapro C cut something",
     "SCISSORS\nopen and close a pair of scissors three times", _SCISSORS_SHUT,
     keys=_cycle(_SCISSORS_OPEN, _SCISSORS_SHUT, 3))
_SNAP_SET = _pose((40, 40, 5, 5), (30, 30, 10), (60, 65, 30), RELAXED, (25, 35, 15))
_SNAP_DONE = _pose((0, 30, 0, 0), (30, 30, 10), CURLED, CURLED, CURLED)
_lib("snap", "Snap", "dynamic", "common sign", "SNAP\nmiddle tip on the thumb, then flick it into the palm, twice",
     _SNAP_SET, keys=[(0, GESTURES["rest"][1]), (0.15, _SNAP_SET), (0.3, _SNAP_SET), (0.34, _SNAP_DONE),
                      (0.5, _SNAP_DONE), (0.62, _SNAP_SET), (0.74, _SNAP_SET), (0.78, _SNAP_DONE),
                      (0.88, _SNAP_DONE), (1, GESTURES["rest"][1])])

# Not drawable without a wrist, listed for the record: Ninapro B wrist flexion, extension, radial and ulnar
# deviation, pronation and supination, wrist extension with a closed fist; Ninapro C turning a screw and opening a
# bottle; waving goodbye
for _k, _v in LIBRARY.items():
    if _k not in GESTURES:
        GESTURES[_k] = (_v["text"], _v["angles"])


# How the cue hand moves (web/app.js cueFrame, sent to the page in the config): a grip closes in GRIP_CLOSE_S, a held
# protocol C movement in C_CLOSE_S, then both hold until the cue ends. The drawn hand follows on a spring (SPRING_W 12
# in web/app.js), which trails a moving target by about 2 / 12 s and settles from a jump in about 0.4 s
GRIP_CLOSE_S = 0.5
C_CLOSE_S = 1.0
SPRING_LAG_S = 2 / 12
SPRING_SETTLE_S = 0.4
# Time to see the drawn hand arrive and match it, on top of the drawn hand reaching the pose
REACTION_S = 0.5


def settle_s(kind, label):
    """Seconds at the start of a held cue that are left out of training: the drawn hand reaching the pose plus the
    reaction time. The rest of the cue, as long as the hold time set for the session, is used. Moving cues have no
    pose to reach and lose nothing."""
    if kind == "hold":
        reach = SPRING_SETTLE_S if label == "rest" else GRIP_CLOSE_S + SPRING_LAG_S
    elif kind == "custom" and label in LIBRARY and "keys" not in LIBRARY[label]:
        reach = C_CLOSE_S + SPRING_LAG_S
    else:
        return 0.0
    return reach + REACTION_S


def _cue(kind, label, text, seconds, posture):
    return {"kind": kind, "label": label, "text": text, "seconds": seconds, "posture": posture}


def session_plan(postures, reps=3, hold_s=4.0, rest_s=3.0, free_s=0.0, seed=None):
    """Cue list: per posture every gesture of GESTURE_SET reps times in shuffled order with rest in between, then
    free movement if free_s is set. No sync taps since 2026-10-08: the camera delay is fixed (ml/dataset.py).

    In every posture each finger is also bent on its own, then all of them one after another.

    kind is "hold" (a held gesture or rest, protocol A), "finger" (one finger moving, protocol B), "free"
    (continuous movement), "move" (protocol_b_plan), "sync" (a tap, older sessions) or "break" (waits for Continue). seconds is None
    for breaks. Free movement is off by default since 2026-10-08, protocol_b_plan records movement instead."""
    rng = random.Random(seed)
    moves = [g for g in GESTURE_SET if g != "rest"]
    rest = GESTURES["rest"][0]
    cues = []
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
    return cues


def protocol_b_plan(postures=("table",), rounds=MOVE_ROUNDS, move_s=MOVE_S, rest_s=3.0, hold_s=4.0, seed=None):
    """Per posture a calibration block (every grip twice, each finger once) and the 9 movements of MOVEMENTS in
    rounds, each movement once per round in shuffled order. Normally one posture, the forearm on the table like
    Ninapro; posture made no difference to the angles (2026-10-06).

    The calibration block keeps every session usable for calibrating a model (it has grips, so it is also a good
    session for training). Ninapro records all repetitions of a movement in a row; rounds spread each movement over
    the whole posture instead, so any few minutes hold all of them. kind "move" is a movement following the animated
    target, it opens and closes once in move_s."""
    rng = random.Random(seed)
    grips = [g for g in GESTURE_SET if g != "rest"]
    moves = list(MOVEMENTS)
    rest = GESTURES["rest"][0]
    cues = []
    for posture in postures:
        cues.append(_cue("break", "setup", f"{POSTURES[posture]}.\nPress Continue when ready.", None, posture))
        cues.append(_cue("hold", "rest", rest, rest_s, posture))
        for _ in range(2):
            rng.shuffle(grips)
            for g in grips:
                cues += [_cue("hold", g, f"{GESTURES[g][0]}\n{HOLD_TEXT}", hold_s, posture),
                         _cue("hold", "rest", rest, rest_s, posture)]
        for f in FINGERS:
            cues += [_cue("finger", f"flex_{f}", GESTURES[f"flex_{f}"][0], FINGER_S, posture),
                     _cue("hold", "rest", rest, rest_s, posture)]
        for _ in range(rounds):
            rng.shuffle(moves)
            for m in moves:
                title, how = MOVEMENTS[m][0].split("\n")
                cues += [_cue("move", m, f"{title}\n{how}, {MOVE_FOLLOW}", move_s, posture),
                         _cue("hold", "rest", rest, rest_s, posture)]
    return cues


C_ROUNDS = 5
C_MOVE_S = 6.0
C_FOLLOW = "follow the hand in and hold"
C_DYNAMIC_FOLLOW = "follow the hand"


def protocol_c_plan(moves, postures=("table",), rounds=C_ROUNDS, move_s=C_MOVE_S, rest_s=3.0, hold_s=4.0, seed=None):
    """Movements the person picked from LIBRARY, for record and play. Per posture a short calibration block (every
    grip twice, the same cues as protocol B's) and then the picked movements in rounds, each once per round in
    shuffled order, with rest in between.

    kind "custom" is a library movement: the target hand closes in C_CLOSE_S and holds to the end of the cue,
    relaxing in the rest after (web/app.js cueFrame); dynamic ones follow their keyframes. Its own kind keeps these
    cues out of the protocol A grip labels even where the key is a grip name (fist, pinch...)."""
    moves = list(dict.fromkeys(moves))
    if not moves:
        raise ValueError("pick at least one movement")
    unknown = [m for m in moves if m not in LIBRARY]
    if unknown:
        raise ValueError(f"unknown movements: {', '.join(unknown)}")
    rng = random.Random(seed)
    grips = [g for g in GESTURE_SET if g != "rest"]
    rest = GESTURES["rest"][0]
    cues = []
    for posture in postures:
        cues.append(_cue("break", "setup", f"{POSTURES[posture]}.\nPress Continue when ready.", None, posture))
        cues.append(_cue("hold", "rest", rest, rest_s, posture))
        for _ in range(2):
            rng.shuffle(grips)
            for g in grips:
                cues += [_cue("hold", g, f"{GESTURES[g][0]}\n{HOLD_TEXT}", hold_s, posture),
                         _cue("hold", "rest", rest, rest_s, posture)]
        for _ in range(rounds):
            rng.shuffle(moves)
            for m in moves:
                title, how = LIBRARY[m]["text"].split("\n")
                follow = C_DYNAMIC_FOLLOW if "keys" in LIBRARY[m] else C_FOLLOW
                cues += [_cue("custom", m, f"{title}\n{how}, {follow}", move_s, posture),
                         _cue("hold", "rest", rest, rest_s, posture)]
    return cues


D_ROUNDS = 1
D_ACTIVITY_S = 40.0
# Forearm muscle activity that isn't a grip: wrist and arm movement, load, fingers moving without grasping. Any grip
# the control layer starts during these is a false start
NO_GRIP = {
    "wrist_flex": "BEND YOUR WRIST\nslowly up and down, hand relaxed",
    "wrist_side": "WRIST SIDE TO SIDE\nslowly, hand relaxed",
    "wrist_turn": "ROTATE YOUR FOREARM\npalm up, then palm down, like turning a doorknob, slowly, hand loose",
    "arm_raise": "RAISE YOUR ARM\nup and down a few times, hand relaxed",
    "arm_out": "HOLD YOUR ARM OUT\nstraight in front, hand relaxed",
    "reach_up": "REACH OVERHEAD\nand back down, a few times, hand relaxed",
    "lean": "LEAN ON YOUR FOREARM\nput weight on it on the table, hand relaxed",
    "push": "PUSH THE TABLE\nflat hand on the table edge, press and let go a few times",
    "shake": "SHAKE YOUR HAND OUT\nloose, like drying it",
    "typing": "TYPE\non the keyboard, normally",
    "drum": "DRUM YOUR FINGERS\nlightly on the table",
    "wave_hello": "WAVE HELLO\nopen hand, from the wrist",
    "talk": "TALK WITH YOUR HANDS\ngesture as if explaining something, no grips",
    "walk": "STAND UP AND WALK\nswing your arms naturally",
}
NO_GRIP_FOLLOW = "don't make a grip"


def protocol_d_plan(postures=("table",), rounds=D_ROUNDS, activity_s=D_ACTIVITY_S, rest_s=3.0, hold_s=4.0, seed=None):
    """False start check: a short calibration block (every grip twice, like protocols B and C) in the first ticked
    posture, then the NO_GRIP activities in rounds, each once per round in shuffled order, with rest in between.

    kind "nogrip" marks a cue where no grip should start. It never gets a grip or rest label; the calibration holds
    are plain holds, so the session can calibrate a model and is scored on its own calibration."""
    rng = random.Random(seed)
    posture = postures[0]
    grips = [g for g in GESTURE_SET if g != "rest"]
    activities = list(NO_GRIP)
    rest = GESTURES["rest"][0]
    cues = [_cue("break", "setup", f"{POSTURES[posture]}.\nPress Continue when ready.", None, posture),
            _cue("hold", "rest", rest, rest_s, posture)]
    for _ in range(2):
        rng.shuffle(grips)
        for g in grips:
            cues += [_cue("hold", g, f"{GESTURES[g][0]}\n{HOLD_TEXT}", hold_s, posture),
                     _cue("hold", "rest", rest, rest_s, posture)]
    for _ in range(rounds):
        rng.shuffle(activities)
        for a in activities:
            cues += [_cue("nogrip", a, f"{NO_GRIP[a]}, {NO_GRIP_FOLLOW}", activity_s, posture),
                     _cue("hold", "rest", rest, rest_s, posture)]
    return cues


def plan_for(settings, seed=None):
    """The cue list for a session's settings, any protocol."""
    if settings.get("protocol", "A") == "D":
        return protocol_d_plan(settings["postures"], settings.get("d_rounds", D_ROUNDS),
                               settings.get("d_activity_s", D_ACTIVITY_S), settings["rest_s"], settings["hold_s"], seed)
    if settings.get("protocol", "A") == "C":
        return protocol_c_plan(settings.get("moves", []), settings["postures"], settings.get("c_rounds", C_ROUNDS),
                               settings.get("c_move_s", C_MOVE_S), settings["rest_s"], settings["hold_s"], seed)
    if settings.get("protocol", "A") == "B":
        return protocol_b_plan(settings["postures"], settings.get("rounds", MOVE_ROUNDS), settings.get("move_s", MOVE_S),
                               settings["rest_s"], settings["hold_s"], seed)
    return session_plan(settings["postures"], settings["reps"], settings["hold_s"], settings["rest_s"],
                        settings.get("free_s", 0), seed)
