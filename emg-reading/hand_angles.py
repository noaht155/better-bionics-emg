"""Joint angles of the hand from MediaPipe's 21 world landmarks (metres, true handedness).

20 angles in degrees, 4 per finger, in the order of JOINTS:
- fingers: base knuckle (MCP) flexion and abduction, middle joint (PIP), tip joint (DIP)
- thumb: base (CMC) flexion and abduction, middle joint (MCP), tip joint (IP)

Flexion is positive towards the palm. Finger abduction is positive towards the thumb.
Thumb CMC flexion moves the thumb across the palm, CMC abduction lifts it out of the palm plane.
web/hand.js rebuilds a hand from these angles with the same definitions, keep the two in step.
"""
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINTS = ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"]
for _f in FINGERS[1:]:
    JOINTS += [f"{_f}_mcp_flex", f"{_f}_mcp_abd", f"{_f}_pip", f"{_f}_dip"]

# First stage of protocol B: the joints one webcam measures well
RELIABLE = ["thumb_mcp"] + [f"{f}_{j}" for f in FINGERS[1:] for j in ("mcp_flex", "pip")]

WRIST = 0
# Landmark indices along each finger, base to tip
CHAINS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
          "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}

# In-palm angle of the resting thumb metacarpal from the wrist-to-middle-knuckle line, towards the thumb
THUMB_REST_DEG = 45.0


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def palm_frame(p, right_hand=True):
    """x towards the thumb side, y from the wrist to the middle knuckle, z out of the palm."""
    y = _unit(p[9] - p[WRIST])
    x = p[5] - p[17]
    x = _unit(x - x.dot(y) * y)
    z = np.cross(x, y)
    # A left hand is a mirror image, the same cross product points out of the back of the hand
    return x, y, z if right_hand else -z


def _elevation(v, z):
    """Angle of v above the palm plane."""
    return np.degrees(np.arctan2(v.dot(z), np.linalg.norm(v - v.dot(z) * z)))


def _bend(a, b, axis):
    """Angle from segment a to segment b, positive when b turns towards the palm around axis."""
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a.dot(b)))
    return ang if np.cross(a, b).dot(axis) >= 0 else -ang


def joint_angles(world, right_hand=True):
    """world: 21 x 3 landmarks. Returns the 20 angles of JOINTS in degrees."""
    p = np.asarray(world, dtype=np.float64)
    x, y, z = palm_frame(p, right_hand)
    out = {}
    for finger in FINGERS[1:]:
        mcp, pip, dip, tip = (p[i] for i in CHAINS[finger])
        meta, prox, mid, dist = mcp - p[WRIST], pip - mcp, dip - pip, tip - dip
        # Finger's own straight-ahead direction in the palm plane, the metacarpals fan out
        ahead = _unit(meta - meta.dot(z) * z)
        thumbwards = _unit(x - x.dot(ahead) * ahead)
        flat = prox - prox.dot(z) * z
        out[f"{finger}_mcp_abd"] = np.degrees(np.arctan2(flat.dot(thumbwards), flat.dot(ahead)))
        out[f"{finger}_mcp_flex"] = _elevation(prox, z) - _elevation(meta, z)
        # Bending towards the palm turns the finger around this axis, on either hand
        axis = np.cross(ahead, z)
        out[f"{finger}_pip"] = _bend(prox, mid, axis)
        out[f"{finger}_dip"] = _bend(mid, dist, axis)

    cmc, mcp, ip, tip = (p[i] for i in CHAINS["thumb"])
    meta, prox, dist = mcp - cmc, ip - mcp, tip - ip
    out["thumb_cmc_abd"] = _elevation(meta, z)
    out["thumb_cmc_flex"] = THUMB_REST_DEG - np.degrees(np.arctan2(meta.dot(x), meta.dot(y)))
    # The thumb bends in its own plane, which isn't tied to the palm, so these two are unsigned
    out["thumb_mcp"] = _bend(meta, prox, np.cross(meta, prox))
    out["thumb_ip"] = _bend(prox, dist, np.cross(prox, dist))
    return np.array([out[j] for j in JOINTS])
