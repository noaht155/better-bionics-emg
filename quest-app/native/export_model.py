"""Writes the Quest app's sim scene from sim.py: both hands, a desk over a floor, and a pool of objects parked on the
floor away from the desk, so spawning one only moves it (no model rebuild on the headset). One XML per prosthetic
side (scene_left.xml: the prosthetic hand is the left), with the force rule's constants as custom numerics for
bbsim.c, and scene.json with what the app needs to know about it.
    emg-reading/.venv/Scripts/python quest-app/native/export_model.py [output folder]
Default output: quest-app/Assets/Resources/sim (ignored, the build script runs this).

Names: the prosthetic hand's start with "p_", the free hand's with "f_", objects are obj0, obj1, ... Sim axes as in
sim.py: z up, the desk top at z 0, the user sits on the +y side, so +x is the user's left.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "emg-reading"))
import sim  # noqa: E402
from hand_angles import JOINTS  # noqa: E402
from tools import POSES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "Assets" / "Resources" / "sim"
# Copies of each object in the pool
POOL = 2
# MuJoCo's native collision code keeps the can from lifting off the box desk in the wrap grasp (it rests at
# 0.0625 m, 0.1352 m with libccd; on sim.py's plane 0.1324 m, 2026-11-20), so the scene uses libccd
# Desk half sizes and centre (m): 1 m wide, 70 cm deep, its near edge 20 cm in front of the hand's start
DESK_HALF = (0.5, 0.35, 0.02)
DESK_CENTRE_Y = -0.15
DESK_HEIGHT = 0.74
PARK_X, PARK_Y, PARK_STEP = 2.0, 2.0, 0.25
HANDS = ("p_", "f_")


def pool():
    return [kind for kind in sim.OBJECTS for _ in range(POOL)]


def parked(i, height=DESK_HEIGHT):
    """Where object i waits off the desk, resting on the floor."""
    return [PARK_X + PARK_STEP * i, PARK_Y, -height + sim.OBJECTS[pool()[i]][3] + 0.001]


def constants():
    values = {"kp": sim.KP, "kp_press": sim.KP_PRESS, "assist_n": sim.ASSIST_N,
              "max_mass_scale": sim.MAX_MASS_SCALE, "gravity": sim.GRAVITY}
    custom = "".join(f'<numeric name="{k}" data="{v!r}"/>' for k, v in values.items())
    return custom + f'<numeric name="arm" data="{" ".join(repr(float(a)) for a in sim._arms())}"/>'


def scene_xml(prosthetic_left=True):
    mirror = {"p_": prosthetic_left, "f_": not prosthetic_left}
    bodies, welds = "", ""
    for p in HANDS:
        x = 0.15 if mirror[p] else -0.15
        pose = f'pos="{x} 0.1 0.25" xyaxes="1 0 0 0 -1 0"'
        bodies += (f'<body name="{p}target" mocap="true" {pose}/>'
                   f'<body name="{p}palm" {pose}><freejoint/>{sim.hand_xml(p, mirror[p])}</body>\n')
        welds += f'<weld name="{p}hand" body1="{p}palm" body2="{p}target" solref="0.02 1"/>'
    for i, kind in enumerate(pool()):
        shape, mass, friction, _ = sim.OBJECTS[kind]
        pos = " ".join(f"{v:.4f}" for v in parked(i))
        bodies += (f'<body name="obj{i}" pos="{pos}"><freejoint/><geom name="obj{i}" {shape} mass="{mass}" '
                   f'friction="{friction} 0.01 0.001" condim="4" contype="2" conaffinity="3"/></body>\n')
        welds += "".join(f'<weld name="{p}hold{i}" body1="{p}palm" body2="obj{i}" active="false"/>' for p in HANDS)
    motors = "".join(f'<motor name="{p}{j}" joint="{p}{j}"/>' for p in HANDS for j in JOINTS)
    meshes = "".join(sim.palm_mesh_xml(p, mirror[p]) for p in HANDS)
    hx, hy, hz = DESK_HALF
    return f"""<mujoco model="quest">
<option timestep="{sim.TIMESTEP}" integrator="implicitfast" gravity="0 0 -{sim.GRAVITY}" cone="elliptic"
noslip_iterations="{sim.NOSLIP_ITERATIONS}"><flag nativeccd="disable"/></option>
<asset>{meshes}</asset>
<default><joint damping="0.02" armature="0.0002"/><default class="hand">
<geom friction="{sim.HAND_FRICTION} 0.01 0.001" condim="4" density="1100" contype="1" conaffinity="2"/></default></default>
<worldbody>
<geom name="desk" type="box" size="{hx} {hy} {hz}" pos="0 {DESK_CENTRE_Y} {-hz}" friction="0.6 0.01 0.001"
contype="2" conaffinity="3" priority="1"/>
<geom name="floor" type="plane" size="0 0 1" pos="0 0 {-DESK_HEIGHT}" friction="0.8 0.01 0.001" contype="2"
conaffinity="3" priority="1"/>
{bodies}</worldbody>
<equality>{welds}</equality>
<actuator>{motors}</actuator>
<custom>{constants()}</custom></mujoco>"""


def ghost_xml(left):
    """One hand on its own, the prosthetic side's, for showing a cue's target pose (bbsim's bb_fk)."""
    motors = "".join(f'<motor joint="g_{j}"/>' for j in JOINTS)
    return (f'<mujoco model="ghost"><default><default class="hand"><geom contype="0" conaffinity="0"/></default>'
            f'</default><asset>{sim.palm_mesh_xml("g_", left)}</asset><worldbody><body name="g_palm">'
            f'{sim.hand_xml("g_", left)}</body></worldbody><actuator>{motors}</actuator></mujoco>')


def meta():
    return dict(sim.SIM_DEFAULTS, objects=list(sim.OBJECTS), pool=pool(),
                half=[v[3] for v in sim.OBJECTS.values()], joints=list(JOINTS),
                rest=[float(a) for a in POSES["rest"]], fist=[float(a) for a in POSES["fist"]],
                point=[float(a) for a in POSES["point"]], desk_half=list(DESK_HALF), desk_centre_y=DESK_CENTRE_Y,
                desk_height=DESK_HEIGHT, park=[PARK_X, PARK_Y, PARK_STEP])


def main(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        old.unlink()
    (out / "scene_left.xml").write_text(scene_xml(True))
    (out / "scene_right.xml").write_text(scene_xml(False))
    (out / "ghost_left.xml").write_text(ghost_xml(True))
    (out / "ghost_right.xml").write_text(ghost_xml(False))
    (out / "scene.json").write_text(json.dumps(meta()))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
