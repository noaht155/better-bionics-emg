"""3D contact simulation for the Control tab: the control layer's 20 joint angles drive a hand in MuJoCo that grasps
objects under gravity, and the grip force on the object drives the motors (haptics mode "force").

Each finger follows the control layer's angles until it touches the object (Noah, 2026-11-19). While it touches, its
bending joints track the angles stiffer (KP_PRESS) with their torque capped by effort, up to max_force at the
fingertip at full effort, like a position controlled hand with current limited motors: a light grip lets a heavy
object slide out, and the grip presses no harder than the pose asks, so the fingertip joints don't curl on past the
pose and roll small objects out. Opening always follows the angles, so a release lets go. Lock holds the effort of
the moment it locked.

Pure physics holds objects since MuJoCo's no-slip pass is on (NOSLIP_ITERATIONS): without it contacts are soft and
the can crept out of a grip pressing several times its weight. Hand friction is HAND_FRICTION, about a silicone
glove: at 1 the can popped out when squeezed harder. The assist (off by default) welds the object to the palm while
the thumb and another finger press on it from both sides hard enough for friction to carry its weight (grip force x
friction at least the weight), for when pure physics can't hold a grasp; a grip too light for the weight still
drops it.

The hand is web/hand.js's geometry (a right hand, the mirror convention of the drawn hands) with the same joint
definitions as hand_angles.py. The page moves the hand through a mocap body the palm is welded to, so the hand
stops at the table instead of passing through it. Coordinates in metres, z up. The hand only collides with itself
where the four fingers meet the palm or the thumb's base (contype bit 4 on those, conaffinity bit 4 on the
fingers), so a fist doesn't sink its fingertips into the palm (Noah, 2026-11-20); fingers pass each other as real
ones slide past, and the thumb passes everything, since it would otherwise stop the index from curling in the fist
pose.

One sim for the PC and the Quest app: this module is the model (geometry, constants, objects, scene_xml) and
quest-app/native/bbsim.c the rule, run on the headset and here through ctypes. World builds bbsim for this computer
against the venv's MuJoCo the first time and again whenever bbsim.c changes (MSVC Build Tools on Windows, cc on a
Mac), into the ignored quest-app/native/build/host/.
"""
import ctypes
import math
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from gestures import GESTURES, _pose
from hand_angles import JOINTS

# Knuckle positions (mm, hand frame: x towards the thumb, y wrist to fingers, z out of the palm), splay of each finger
# (degrees towards the thumb) and segment lengths, the same as web/hand.js; keep the two in step
FINGER_GEOMETRY = {"index": ([22, 88, 0], 8, [40, 24, 19]), "middle": ([2, 92, 0], 0, [45, 28, 20]),
                   "ring": ([-17, 86, 0], -8, [42, 27, 20]), "pinky": ([-33, 76, 0], -16, [33, 19, 18])}
THUMB_BASE, THUMB_LENGTHS = [20, 22, 0], [42, 30, 25]
THUMB_REST_DEG = 45.0
# Palm outline (mm) from the wrist round the knuckles; the palm is the convex hull of it at two thicknesses
PALM_OUTLINE = [(-26, 0), (16, 0), (27, 20), (29, 50), (25, 84), (2, 92), (-17, 86), (-33, 76), (-38, 50), (-34, 18)]
FINGER_RADIUS = 9.0
TAPER = 0.8

# At 2 ms a can 4 times as heavy squeezed against the table blew up, 1 ms held
TIMESTEP = 0.001
# Joint stiffness towards the target angle, N m per rad
KP = 0.6
# Stiffness of a finger touching the object. Pressing with the full effort torque on every bending joint, however
# far past the pose, rolled the small cube and the block out of a pinch and the can out at full effort; 2 held the
# ball, block and pen in every grasp tried from effort 0.6 on (5 and 15 lost more, 2026-11-19)
KP_PRESS = 2.0
# Without the no-slip pass the can slid out of a 20 N grip with friction 2 (2026-11-19); with 10 it lifts from
# effort 0.2. At 1 the can popped out of the hand at effort 0.6 and up, at 2 it holds from 0.2 to 1.0
NOSLIP_ITERATIONS = 10
HAND_FRICTION = 2.0
# The table's friction wins over the hand's (priority), so fingertips slide over it as real ones do: at the hand's 2
# they stuck to it and the index couldn't curl in to pinch the small cube
# Heavier objects blow up against the light finger segments (6x the can, about 150 times a finger segment's mass)
# whatever the damping, armature or finger mass; 3x was clean at every effort
MAX_MASS_SCALE = 3.0
# name: geom attributes, mass (kg), friction, how it starts on the table (half height, m)
OBJECTS = {
    "can": ('type="cylinder" size="0.033 0.061"', 0.35, 0.5, 0.061),
    "ball": ('type="sphere" size="0.034"', 0.058, 0.8, 0.034),
    "block": ('type="box" size="0.025 0.025 0.025"', 0.12, 0.7, 0.025),
    "small cube": ('type="box" size="0.01 0.01 0.01"', 0.01, 0.7, 0.01),
    "pen": ('type="capsule" size="0.0045 0.065" euler="0 90 0"', 0.012, 0.5, 0.0045),
}
# Surfaces the haptics tell apart and the friction each one stands for. Friction is the one rule (Noah, 2026-11-23):
# a thing's colour and pulse rate both come from its friction on this scale, in between two surfaces for frictions
# in between (haptics.friction_rate, the page's colours). Its surface, carried in the scene as each geom's user value
# (its index in SURFACES), is the nearest one
SURFACES = ("slippery", "smooth", "normal", "rough")
SURFACE_FRICTION = {"slippery": 0.15, "smooth": 0.35, "normal": 0.7, "rough": 1.0}
DESK_FRICTION, FLOOR_FRICTION, POST_FRICTION = 0.6, 0.8, 0.7


def surface_at(friction):
    return min(SURFACES, key=lambda s: abs(SURFACE_FRICTION[s] - friction))


SURFACE_OF = {**{k: surface_at(v[2]) for k, v in OBJECTS.items()},
              "desk": surface_at(DESK_FRICTION), "floor": surface_at(FLOOR_FRICTION)}
# The surface rail (Noah, 2026-11-23): a bar fixed over the far side of the desk, slippery on the user's right to
# rough on the left, to grip and slide the hand along to compare them. "steps": one section per surface at its
# friction; "gradient": RAIL_PIECES pieces per section, the friction straight between the sections' middles, so
# what lies between can be felt. Its friction wins over the hand's like the desk's. Radius, section half length,
# height of its axis over the desk top, its y (m)
RAIL_RADIUS, RAIL_HALF, RAIL_HEIGHT, RAIL_Y = 0.015, 0.05, 0.12, -0.38
RAIL_PIECES = 5
SIM_DEFAULTS = {"object": "can", "mass_scale": 1.0, "max_force": 15.0, "assist": False, "rail": "steps"}
# Assist: thumb and another finger each press at least this hard
ASSIST_N = 1.0
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
GRAVITY = 9.81

NATIVE = Path(__file__).resolve().parents[1] / "quest-app" / "native"
SOURCE = NATIVE / "bbsim.c"
HANDS = ("p_", "f_")
# Copies of each object in the pool
POOL = 2
# MuJoCo's native collision code keeps the can from lifting off the box desk in the wrap grasp (it rests at
# 0.0625 m, 0.1352 m with libccd; on a plane 0.1324 m, 2026-11-20), so the scene uses libccd
# Desk half sizes and centre (m): 1 m wide, 70 cm deep, its near edge 20 cm in front of the hand's start
DESK_HALF = (0.5, 0.35, 0.02)
DESK_CENTRE_Y = -0.15
DESK_HEIGHT = 0.74
PARK_X, PARK_Y, PARK_STEP = 2.0, 2.0, 0.25
# The PC's hand starts above the desk, palm down, fingers away from the viewer; its free hand waits off to the side
HAND_START, HAND_QUAT = (0.0, -0.05, 0.25), (0.0, 1.0, 0.0, 0.0)
FREE_HAND_AWAY = (-1.5, 2.0, 0.0)

# How each object is grasped (the tests lift them so), for placing the hand ready on it. Hand frames, columns the
# hand's x (thumb), y (fingers), z (palm) in the world: beside it thumb up, from above fingers away from the viewer,
# and from above tilted back 40 degrees for a small object
SIDE = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
TOP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
_TILT = np.radians(-40)
TOP_TILTED = TOP @ np.array([[1, 0, 0], [0, np.cos(_TILT), -np.sin(_TILT)], [0, np.sin(_TILT), np.cos(_TILT)]])
# Thumb opposed, fingers curled more at the middle joint: wraps the can with the hand beside it. The best of 108
# poses, placements, frictions and stiffnesses tried on 2026-11-19
WRAP = _pose((10, 60, 20, 20), (40, 80, 48), (40, 80, 48), (40, 80, 48), (40, 80, 48))
# object: hand frame, closed pose, the parts that close on it, whether the palm does, and a shift of the grip's
# centre on the object (m, hand frame)
GRASPS = {
    "can": (SIDE, WRAP, FINGERS, True, (-0.01, 0, 0)),
    "ball": (SIDE, WRAP, FINGERS, True, (0, 0.01, 0)),
    "block": (TOP, GESTURES["tripod"][1], ["thumb", "index", "middle"], False, (0, 0, 0.01)),
    "pen": (TOP, GESTURES["pinch"][1], ["thumb", "index"], False, (0, -0.01, 0)),
    "small cube": (TOP_TILTED, GESTURES["tripod"][1], ["thumb", "index", "middle"], False, (0, 0, 0)),
}
# The ready hand waits this far back from the grasp along its palm, and beside an object this much higher too: lower
# down, the open hand beside the ball rested on the desk (1.3 N at 3 cm, 0.2 at 4, none at 5)
READY_GAP_M = 0.02
READY_LIFT_M = 0.05


def _mm(v):
    return " ".join(f"{x / 1000:.4f}" for x in v)


def _quat_z(deg):
    a = math.radians(deg) / 2
    return f"{math.cos(a):.6f} 0 0 {math.sin(a):.6f}"


def _palm_vertices(m=1):
    """The outline at the palm's full thickness, and shrunk a little further out, so the edges come out rounded. m -1
    mirrors it into a left hand."""
    pts = []
    centre = np.mean(PALM_OUTLINE, 0)
    for z, shrink in ((10, 0), (-10, 0), (14, 6), (-13, 6)):
        for p in PALM_OUTLINE:
            d = (np.array(p) - centre) / np.linalg.norm(np.array(p) - centre)
            pts.append((m * (p[0] - d[0] * shrink), p[1] - d[1] * shrink, z))
    return " ".join(f"{v / 1000:.4f}" for p in pts for v in p)


def _finger(name, base, splay, lengths, prefix="", m=1):
    """One finger; prefix goes before every name, m -1 mirrors it into a left hand. A mirror isn't a rotation:
    flexion axes along x stay, the spread axis along z and the splay angle flip, so the same angles mirror the pose."""
    joints = [f'<joint name="{prefix}{name}_mcp_abd" axis="0 0 {-m}" range="-30 30"/>'
              f'<joint name="{prefix}{name}_mcp_flex" axis="1 0 0" range="-20 100"/>',
              f'<joint name="{prefix}{name}_pip" axis="1 0 0" range="0 115"/>',
              f'<joint name="{prefix}{name}_dip" axis="1 0 0" range="-5 90"/>']
    xml = ""
    for i, (segment, length) in enumerate(zip(("prox", "mid", "dist"), lengths)):
        pos = _mm([m * base[0], *base[1:]]) if i == 0 else f"0 {lengths[i - 1] / 1000:.4f} 0"
        quat = f' quat="{_quat_z(-m * splay)}"' if i == 0 else ""
        radius = FINGER_RADIUS * (1 - (1 - TAPER) * i / 2) / 1000
        xml += (f'<body name="{prefix}{name}_{segment}" pos="{pos}"{quat}>{joints[i]}'
                f'<geom name="{prefix}{name}_{segment}" class="hand" conaffinity="6" type="capsule" '
                f'fromto="0 0 0 0 {length / 1000:.4f} 0" size="{radius:.4f}"/>')
    return xml + "</body>" * 3


def _thumb(prefix="", m=1):
    # hand.js bends the thumb towards (-1, 0, 1) in the hand frame, taken at the rest pose into the metacarpal's frame
    bend = np.array([-0.5, 0.0, math.sqrt(0.5)])
    axis = np.array([bend[2], 0.0, -m * bend[0]])
    axis = " ".join(f"{v:.4f}" for v in axis / np.linalg.norm(axis))
    meta, prox, dist = (v / 1000 for v in THUMB_LENGTHS)
    r = FINGER_RADIUS / 1000
    n = prefix
    return (f'<body name="{n}thumb_meta" pos="{_mm([m * THUMB_BASE[0], *THUMB_BASE[1:]])}" '
            f'quat="{_quat_z(-m * THUMB_REST_DEG)}">'
            f'<joint name="{n}thumb_cmc_flex" axis="0 0 {m}" range="-30 80"/>'
            f'<joint name="{n}thumb_cmc_abd" axis="1 0 0" range="-15 80"/>'
            f'<geom name="{n}thumb_meta" class="hand" type="capsule" fromto="0 0 0 0 {meta:.4f} 0" size="0.011"/>'
            f'<geom name="{n}thumb_thenar" class="hand" contype="5" type="ellipsoid" pos="{-m * 0.008:.3f} 0.018 0.006" '
            'size="0.014 0.024 0.009"/>'
            f'<body name="{n}thumb_prox" pos="0 {meta:.4f} 0"><joint name="{n}thumb_mcp" axis="{axis}" range="-10 80"/>'
            f'<geom name="{n}thumb_prox" class="hand" type="capsule" fromto="0 0 0 0 {prox:.4f} 0" size="{r:.4f}"/>'
            f'<body name="{n}thumb_dist" pos="0 {prox:.4f} 0"><joint name="{n}thumb_ip" axis="{axis}" range="-10 90"/>'
            f'<geom name="{n}thumb_dist" class="hand" type="capsule" fromto="0 0 0 0 {dist:.4f} 0" '
            f'size="{r * TAPER:.4f}"/></body></body></body>')


def hand_xml(prefix="", mirror=False):
    """What goes inside a hand's palm body: palm, pad, fingers and thumb, every name starting with prefix. mirror makes
    it a left hand. The palm mesh is the asset named prefix + "palm" (palm_mesh_xml)."""
    m = -1 if mirror else 1
    return (f'<geom name="{prefix}palm" class="hand" contype="5" type="mesh" mesh="{prefix}palm"/>'
            f'<geom name="{prefix}palm_pad" class="hand" contype="5" type="ellipsoid" pos="{-m * 0.024:.3f} 0.035 0.008" '
            'size="0.013 0.03 0.008"/>'
            + "".join(_finger(n, *g, prefix, m) for n, g in FINGER_GEOMETRY.items()) + _thumb(prefix, m))


def palm_mesh_xml(prefix="", mirror=False):
    return f'<mesh name="{prefix}palm" vertex="{_palm_vertices(-1 if mirror else 1)}"/>'


def pool():
    return [kind for kind in OBJECTS for _ in range(POOL)]


def parked(i, height=DESK_HEIGHT):
    """Where object i waits off the desk, resting on the floor."""
    return [PARK_X + PARK_STEP * i, PARK_Y, -height + OBJECTS[pool()[i]][3] + 0.001]


def _arms():
    """Per joint in JOINTS: distance (m) from the joint to the middle of the rest of its finger, so a joint torque
    cap of force x arm gives roughly that force where the finger touches."""
    arms = {}
    for name, (_, _, lengths) in FINGER_GEOMETRY.items():
        for joint, start in (("mcp_abd", 0), ("mcp_flex", 0), ("pip", 1), ("dip", 2)):
            rest = lengths[start:]
            arms[f"{name}_{joint}"] = sum(rest) / 2 / 1000
    arms["thumb_cmc_flex"] = arms["thumb_cmc_abd"] = sum(THUMB_LENGTHS) / 2 / 1000
    arms["thumb_mcp"] = sum(THUMB_LENGTHS[1:]) / 2 / 1000
    arms["thumb_ip"] = THUMB_LENGTHS[2] / 2 / 1000
    return np.array([arms[j] for j in JOINTS])


def grasp_centre(pose, fingers, palm):
    """Mean of the fingertips (and the palm) in the hand frame with the pose closed in free air."""
    m = mujoco.MjModel.from_xml_string(ghost_xml(False))
    d = mujoco.MjData(m)
    d.qpos[[m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]] = np.radians(pose)
    mujoco.mj_kinematics(m, d)
    points = [np.array([-0.005, 0.06, 0.012])] if palm else []
    for f in fingers:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{f}_dist")
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
        joint = d.xpos[m.geom_bodyid[g]]
        points.append(max((d.geom_xpos[g] + axis, d.geom_xpos[g] - axis),
                          key=lambda end: np.linalg.norm(end - joint)))
    return np.mean(points, 0)


def constants():
    values = {"kp": KP, "kp_press": KP_PRESS, "assist_n": ASSIST_N, "max_mass_scale": MAX_MASS_SCALE,
              "gravity": GRAVITY}
    custom = "".join(f'<numeric name="{k}" data="{v!r}"/>' for k, v in values.items())
    return custom + f'<numeric name="arm" data="{" ".join(repr(float(a)) for a in _arms())}"/>'


def _surface(thing):
    return SURFACES.index(SURFACE_OF[thing])


def rail_geoms(kind="steps"):
    """The rail's sections ("steps") or pieces ("gradient") as (x, half length, friction)."""
    if kind == "steps":
        return [((2 * i - len(SURFACES) + 1) * RAIL_HALF, RAIL_HALF, SURFACE_FRICTION[s])
                for i, s in enumerate(SURFACES)]
    half = RAIL_HALF / RAIL_PIECES
    out = []
    for k in range(len(SURFACES) * RAIL_PIECES):
        section = (k + 0.5) / RAIL_PIECES - 0.5
        friction = float(np.interp(section, range(len(SURFACES)), [SURFACE_FRICTION[s] for s in SURFACES]))
        out.append(((2 * k + 1) * half - len(SURFACES) * RAIL_HALF, half, round(friction, 4)))
    return out


def scene_xml(prosthetic_left=True, rail_kind="steps"):
    """Both hands, the desk over the floor and the object pool parked on the floor away from the desk, so showing an
    object only moves it. Names: the prosthetic hand's start with "p_", the free hand's with "f_", objects are obj0,
    obj1, ... The desk top is at z 0 and the user sits on its +y side, so +x is the user's left."""
    mirror = {"p_": prosthetic_left, "f_": not prosthetic_left}
    bodies, welds = "", ""
    for p in HANDS:
        x = 0.15 if mirror[p] else -0.15
        pose = f'pos="{x} 0.1 0.25" xyaxes="1 0 0 0 -1 0"'
        bodies += (f'<body name="{p}target" mocap="true" {pose}/>'
                   f'<body name="{p}palm" {pose}><freejoint/>{hand_xml(p, mirror[p])}</body>\n')
        welds += f'<weld name="{p}hand" body1="{p}palm" body2="{p}target" solref="0.02 1"/>'
    for i, kind in enumerate(pool()):
        shape, mass, friction, _ = OBJECTS[kind]
        pos = " ".join(f"{v:.4f}" for v in parked(i))
        bodies += (f'<body name="obj{i}" pos="{pos}"><freejoint/><geom name="obj{i}" {shape} mass="{mass}" '
                   f'friction="{friction} 0.01 0.001" condim="4" contype="2" conaffinity="3" '
                   f'user="{_surface(kind)}"/></body>\n')
        welds += "".join(f'<weld name="{p}hold{i}" body1="{p}palm" body2="obj{i}" active="false"/>' for p in HANDS)
    motors = "".join(f'<motor name="{p}{j}" joint="{p}{j}"/>' for p in HANDS for j in JOINTS)
    rail = ""
    for i, (x, half, friction) in enumerate(rail_geoms(rail_kind)):
        rail += (f'<geom name="rail{i}" type="cylinder" size="{RAIL_RADIUS} {half}" pos="{x} {RAIL_Y} '
                 f'{RAIL_HEIGHT}" euler="0 90 0" friction="{friction} 0.01 0.001" contype="2" '
                 f'conaffinity="3" priority="1" user="{SURFACES.index(surface_at(friction))}"/>')
    for side in (-1, 1):
        x = side * (len(SURFACES) * RAIL_HALF + RAIL_RADIUS)
        rail += (f'<geom name="rail_post{(side + 1) // 2}" type="box" size="{RAIL_RADIUS} {RAIL_RADIUS} '
                 f'{(RAIL_HEIGHT + RAIL_RADIUS) / 2}" pos="{x} {RAIL_Y} {(RAIL_HEIGHT + RAIL_RADIUS) / 2}" '
                 f'friction="{POST_FRICTION} 0.01 0.001" contype="2" conaffinity="3" priority="1" '
                 f'user="{SURFACES.index(surface_at(POST_FRICTION))}"/>')
    meshes = "".join(palm_mesh_xml(p, mirror[p]) for p in HANDS)
    hx, hy, hz = DESK_HALF
    return f"""<mujoco model="contact">
<option timestep="{TIMESTEP}" integrator="implicitfast" gravity="0 0 -{GRAVITY}" cone="elliptic"
noslip_iterations="{NOSLIP_ITERATIONS}"><flag nativeccd="disable"/></option>
<size nuser_geom="1"/>
<asset>{meshes}</asset>
<default><joint damping="0.02" armature="0.0002"/><default class="hand">
<geom friction="{HAND_FRICTION} 0.01 0.001" condim="4" density="1100" contype="1" conaffinity="2"/></default></default>
<worldbody>
<geom name="desk" type="box" size="{hx} {hy} {hz}" pos="0 {DESK_CENTRE_Y} {-hz}" friction="{DESK_FRICTION} 0.01 0.001"
contype="2" conaffinity="3" priority="1" user="{_surface("desk")}"/>
<geom name="floor" type="plane" size="0 0 1" pos="0 0 {-DESK_HEIGHT}" friction="{FLOOR_FRICTION} 0.01 0.001" contype="2"
conaffinity="3" priority="1" user="{_surface("floor")}"/>
{rail}{bodies}</worldbody>
<equality>{welds}</equality>
<actuator>{motors}</actuator>
<custom>{constants()}</custom></mujoco>"""


def ghost_xml(left):
    """One hand on its own, for showing a pose without physics (bbsim's bb_fk)."""
    motors = "".join(f'<motor joint="g_{j}"/>' for j in JOINTS)
    return (f'<mujoco model="ghost"><default><default class="hand"><geom contype="0" conaffinity="0"/></default>'
            f'</default><asset>{palm_mesh_xml("g_", left)}</asset><worldbody><body name="g_palm">'
            f'{hand_xml("g_", left)}</body></worldbody><actuator>{motors}</actuator></mujoco>')


def _library():
    """Path of bbsim built for this computer against the venv's MuJoCo, building it when bbsim.c is newer."""
    mj = Path(mujoco.__file__).parent
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    out = NATIVE / "build" / "host" / f"bbsim-{mujoco.__version__}{ext}"
    if out.exists() and out.stat().st_mtime >= SOURCE.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Built under another name first, so a second process never loads a half written file
    tmp = out.with_name(f"{out.stem}-{os.getpid()}{ext}")
    if sys.platform == "win32":
        _build_msvc(mj, tmp)
    else:
        lib = next(p for p in mj.glob("libmujoco*") if p.suffix in (".dylib", ".so") or ".so." in p.name)
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-fvisibility=hidden", f"-I{mj / 'include'}", str(SOURCE),
                        str(lib), f"-Wl,-rpath,{mj}", "-lm", "-o", str(tmp)], check=True)
    os.replace(tmp, out)
    return out


def _build_msvc(mj, out):
    """MSVC can't link against a DLL directly: an import library made from mujoco.dll's exports goes in between."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    found = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                           capture_output=True, text=True).stdout.strip() if vswhere.exists() else ""
    if not found:
        raise RuntimeError("the contact sim needs a C compiler: install Visual Studio Build Tools (C++)")
    env = f'"{Path(found) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"}" >nul && '
    work = out.parent / "msvc"
    work.mkdir(exist_ok=True)
    dump = subprocess.run(env + f'dumpbin /nologo /exports "{mj / "mujoco.dll"}"', shell=True, capture_output=True,
                          text=True, check=True).stdout
    names = [line.split()[3] for line in dump.splitlines() if len(line.split()) == 4 and line.split()[0].isdigit()]
    (work / "mujoco.def").write_text("EXPORTS\n" + "\n".join(names) + "\n")
    subprocess.run(env + f'lib /nologo /machine:x64 /def:"{work / "mujoco.def"}" /out:"{work / "mujoco.lib"}" && '
                   f'cl /nologo /O2 /LD /I"{mj / "include"}" "{SOURCE}" "{work / "mujoco.lib"}" '
                   f'/Fo"{work}\\\\" /Fe"{out}" /link /implib:"{work / "bbsim.lib"}"',
                   shell=True, capture_output=True, text=True, check=True)


_lib = None


def _load():
    global _lib
    if _lib is None:
        # On the Mac bbsim only finds libmujoco because importing mujoco already loaded it: the wheel's library names
        # itself inside a mujoco.framework folder the wheel doesn't have
        if sys.platform == "win32":
            os.add_dll_directory(str(Path(mujoco.__file__).parent))
        lib = ctypes.CDLL(str(_library()))
        P, D, I = ctypes.c_void_p, ctypes.c_double, ctypes.c_int
        DP, FP, IP, S = ctypes.POINTER(D), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(I), ctypes.c_char_p
        for name, res, args in [("bb_create", P, [S, I, S, I]), ("bb_free", None, [P]),
                                ("bb_set_params", None, [P, D, D, I]), ("bb_set_input", None, [P, I, DP, D, I, I]),
                                ("bb_set_hand", None, [P, I, DP, DP, I]), ("bb_hand_target", None, [P, I, DP, DP]),
                                ("bb_joint_angles", None, [P, I, DP]), ("bb_place_object", None, [P, I, DP, DP, I]),
                                ("bb_object_pose", None, [P, I, DP, DP]), ("bb_step", None, [P, D]),
                                ("bb_poses", None, [P, FP, FP]), ("bb_readout", None, [P, DP])]:
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        _lib = lib
    return _lib


def _doubles(values):
    return (ctypes.c_double * len(values))(*map(float, values))


# The rail is held from the user's side: palm towards it, fingers up and curled over the top, thumb under it. From
# above (TOP), WRAP closed on its top like a pinch (Noah, 2026-11-23). The palm's middle (hand frame) sits on the
# rail, the hand RAIL_GRIP_DOWN m down the fingers' line so they reach over, and RAIL_GRIP_X along it: sliding, the
# fingers drag behind and read the next section 2 to 3 cm late one way and on time the other
_ROLL = np.radians(-90)
RAIL_ROT = TOP @ np.array([[1, 0, 0], [0, np.cos(_ROLL), -np.sin(_ROLL)], [0, np.sin(_ROLL), np.cos(_ROLL)]])
PALM_MIDDLE = np.array([-0.005, 0.06, 0.012])
RAIL_GRIP_DOWN = 0.03
RAIL_GRIP_X = 0.01


def rail_pose(section, gap=0.0):
    """Hand position and quaternion that grip the rail in WRAP at section (SURFACES index, fractions in between: 1.5
    is the border of smooth and normal), backed off gap m from it."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, RAIL_ROT.flatten())
    x = (2 * section + 1 - len(SURFACES)) * RAIL_HALF + RAIL_GRIP_X
    on = np.array([x, RAIL_Y, RAIL_HEIGHT]) - RAIL_ROT @ PALM_MIDDLE - RAIL_ROT[:, 2] * RAIL_RADIUS
    return on - RAIL_ROT[:, 1] * RAIL_GRIP_DOWN - RAIL_ROT[:, 2] * gap, q


def rail_back(gap):
    """The step that backs the hand off the rail by gap m."""
    return -RAIL_ROT[:, 2] * gap


def rail_section(hand_pos):
    """The section number (as rail_pose takes it) the hand at hand_pos grips."""
    x = hand_pos[0] - rail_pose(0)[0][0] + (1 - len(SURFACES)) * RAIL_HALF
    return (x / RAIL_HALF + len(SURFACES) - 1) / 2


class World:
    """The scene (scene_xml) in bbsim, the code the Quest app runs, seen as one hand and one object: the prosthetic
    hand (a right hand, the mirror convention of the drawn hands) and the chosen object on the desk. The free hand
    waits out of the way and the rest of the pool stays parked. step() advances it; the hand pose, object and settings
    can change between steps. model is the scene loaded in MuJoCo's Python bindings for names and shapes only, the
    state lives in bbsim."""

    def __init__(self, params=None):
        self.params = dict(SIM_DEFAULTS, **(params or {}))
        lib = _load()
        xml = scene_xml(prosthetic_left=False, rail_kind=self.params["rail"])
        self.model = mujoco.MjModel.from_xml_string(xml)
        err = ctypes.create_string_buffer(1000)
        self._w = lib.bb_create(xml.encode(), len(xml.encode()), err, len(err))
        if not self._w:
            raise RuntimeError(err.value.decode())
        self._lib = lib
        self._pool = pool()
        self.object, self._index = None, None
        self.set_hand(HAND_START, HAND_QUAT, teleport=True)
        lib.bb_set_hand(self._w, 1, _doubles(FREE_HAND_AWAY), _doubles([1, 0, 0, 0]), 1)
        self.set_params({})

    def __del__(self):
        if getattr(self, "_w", None):
            self._lib.bb_free(self._w)
            self._w = None

    def _show(self, obj):
        """Parks the shown object and puts obj on the desk in its place."""
        if self._index is not None:
            self._lib.bb_place_object(self._w, self._index, _doubles(parked(self._index)), _doubles([1, 0, 0, 0]), 1)
        self.object, self._index = obj, self._pool.index(obj)
        self.reset_object()
        m = self.model
        geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obj{self._index}")
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "p_palm")
        hand = []
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            while b > 0 and b != palm:
                b = m.body_parentid[b]
            if b == palm:
                hand.append(g)
        rail = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("rail")]
        self._shown = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "desk"), *rail, *hand, geom]

    def set_params(self, params):
        self.params.update(params)
        if self.params["object"] != self.object:
            self._show(self.params["object"])
        self._lib.bb_set_params(self._w, self.params["mass_scale"], self.params["max_force"],
                                int(self.params["assist"]))

    def hand_pose(self):
        pos, quat = _doubles([0] * 3), _doubles([0] * 4)
        self._lib.bb_hand_target(self._w, 0, pos, quat)
        return np.array(pos), np.array(quat)

    def set_hand(self, pos, quat, teleport=False):
        """Moves the hand's target; the palm follows through its weld, or is put there at once with teleport."""
        self._lib.bb_set_hand(self._w, 0, _doubles(pos), _doubles(quat), int(teleport))

    def grasp_target(self):
        """Where the hand grasps the object as it lies now (GRASPS): position, orientation (w x y z) and the step
        back to the ready pose (READY_GAP_M, READY_LIFT_M)."""
        rot, pose, fingers, palm, offset = GRASPS[self.object]
        pos, quat = _doubles([0] * 3), _doubles([0] * 4)
        self._lib.bb_object_pose(self._w, self._index, pos, quat)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, rot.flatten())
        base = np.array(pos) - rot @ grasp_centre(pose, fingers, palm) + rot @ np.array(offset)
        back = -rot[:, 2] * READY_GAP_M + [0, 0, READY_LIFT_M if rot is SIDE else 0.0]
        return base, q, back

    def place_hand(self, where):
        """Puts the hand at its start ("start"), or open just above the slippery end of the rail ("rail"), or ready to
        grasp the object, a little back from it ("object")."""
        if where == "start":
            self.set_hand(HAND_START, HAND_QUAT, teleport=True)
        elif where == "rail":
            self.set_hand(*rail_pose(0, READY_GAP_M), teleport=True)
        else:
            base, q, back = self.grasp_target()
            self.set_hand(base + back, q, teleport=True)

    def joint_angles(self):
        """The hand's joint angles now (degrees, JOINTS order)."""
        out = _doubles([0] * len(JOINTS))
        self._lib.bb_joint_angles(self._w, 0, out)
        return np.array(out)

    def place_object(self, pos):
        """Puts the object at pos (m), upright and still, and lets go of the assist."""
        self._lib.bb_place_object(self._w, self._index, _doubles(pos), _doubles([1, 0, 0, 0]), 0)

    def reset_object(self):
        self.place_object([0.0, -0.12, OBJECTS[self.object][3] + 0.001])

    def step(self, dt, angles, effort, locked=False):
        """Advances dt seconds towards the control layer's angles (degrees, JOINTS order) at effort 0 to 1."""
        self._lib.bb_set_input(self._w, 0, _doubles(angles), float(effort), int(locked), 0)
        self._lib.bb_step(self._w, float(dt))

    def _readout(self):
        out = _doubles([0] * 18)
        self._lib.bb_readout(self._w, out)
        return np.array(out)

    @property
    def finger_force(self):
        return self._readout()[1:6]

    @property
    def palm_force(self):
        return float(self._readout()[6])

    @property
    def time(self):
        return float(self._readout()[12])

    @property
    def blowups(self):
        return int(self._readout()[13])

    def grip_force(self):
        """Total force the hand presses on the object with (N), what the motors follow."""
        return float(self._readout()[0])

    def touch(self):
        """Force the hand presses on everything it touches (N: object, desk) and the surface it presses on most
        (SURFACES), None while it touches nothing. What the haptics play."""
        r = self._readout()
        return float(r[14]), SURFACES[int(r[16])] if r[16] >= 0 else None

    def touch_friction(self):
        """Friction of what the hand presses on, weighted by force (each surface's own), None while it touches
        nothing."""
        r = self._readout()
        return float(r[15]) if r[14] > 0 else None

    def weight(self):
        scale = min(max(float(self.params["mass_scale"]), 0.05), MAX_MASS_SCALE)
        return OBJECTS[self.object][1] * scale * GRAVITY

    def friction(self):
        return OBJECTS[self.object][2]

    def scene(self):
        """The shown geoms, for the page to build: type, size, body role, surface (SURFACES, None on the hand) and,
        for meshes, vertices and faces."""
        m = self.model
        geoms = []
        for g in self._shown:
            kind = mujoco.mjtGeom(m.geom_type[g]).name.removeprefix("mjGEOM_").lower()
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            role = ("table" if name == "desk" else "rail" if name.startswith("rail") else
                    "hand" if name.startswith("p_") else "object")
            entry = {"name": name.removeprefix("p_") if role == "hand" else role, "type": kind, "role": role,
                     "size": m.geom_size[g].round(5).tolist(),
                     "surface": None if role == "hand" else SURFACES[int(m.geom_user[g, 0])],
                     "friction": None if role == "hand" else round(float(m.geom_friction[g, 0]), 4)}
            if kind == "mesh":
                mesh = m.geom_dataid[g]
                v0, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                f0, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
                entry["vertices"] = m.mesh_vert[v0:v0 + nv].round(5).tolist()
                entry["faces"] = m.mesh_face[f0:f0 + nf].tolist()
            geoms.append(entry)
        return {"object": self.object, "objects": list(OBJECTS), "geoms": geoms}

    def poses(self):
        """Position and orientation (3 x 3, row major) of every shown geom, in the order of scene()."""
        n = self.model.ngeom
        pos, mat = (ctypes.c_float * (3 * n))(), (ctypes.c_float * (9 * n))()
        self._lib.bb_poses(self._w, pos, mat)
        pos, mat = np.array(pos).reshape(n, 3)[self._shown], np.array(mat).reshape(n, 9)[self._shown]
        return {"pos": pos.round(5).tolist(), "mat": mat.round(5).tolist()}

    def readout(self):
        r = self._readout()
        pos, quat = _doubles([0] * 3), _doubles([0] * 4)
        self._lib.bb_object_pose(self._w, self._index, pos, quat)
        return {"grip_force": round(float(r[0]), 2), "finger_force": r[1:6].round(2).tolist(),
                "palm_force": round(float(r[6]), 2), "weight": round(self.weight(), 3), "friction": self.friction(),
                "assisted": bool(r[8] >= 0), "object_height": round(pos[2], 4), "time": round(float(r[12]), 3),
                "touch_force": round(float(r[14]), 2), "touch_surface": SURFACES[int(r[16])] if r[16] >= 0 else None}
