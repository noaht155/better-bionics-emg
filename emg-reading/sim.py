"""3D contact simulation for the Control tab: the control layer's 20 joint angles drive a hand in MuJoCo that grasps
objects under gravity, and the grip force on the object drives the motors (haptics mode "force").

Each finger follows the control layer's angles until it touches the object. While it touches and the control layer
still closes it, it presses with a force set by effort instead (Noah, 2026-11-19), up to max_force at the fingertip
at full effort, so a light grip lets a heavy object slide out. Opening always follows the angles, so a release lets
go. Lock holds the effort of the moment it locked. The assist (off by default) welds the object to the palm while
the thumb and another finger press on it from both sides hard enough for friction to carry its weight (grip force x
friction at least the weight), for when pure physics can't hold a grasp; a grip too light for the weight still
drops it.

The hand is web/hand.js's geometry (a right hand, the mirror convention of the drawn hands) with the same joint
definitions as hand_angles.py. The page moves the hand through a mocap body the palm is welded to, so the hand
stops at the table instead of passing through it. Coordinates in metres, z up. The hand doesn't
collide with itself (contype and conaffinity): fingers pass each other as real ones slide past, and the thumb of the
fist pose would otherwise stop the index from curling.
"""
import math

import mujoco
import numpy as np

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

TIMESTEP = 0.002
# Joint stiffness towards the target angle, N m per rad
KP = 0.6
# name: geom attributes, mass (kg), friction, how it starts on the table (half height, m)
OBJECTS = {
    "can": ('type="cylinder" size="0.033 0.061"', 0.35, 0.5, 0.061),
    "ball": ('type="sphere" size="0.034"', 0.058, 0.8, 0.034),
    "block": ('type="box" size="0.025 0.025 0.025"', 0.12, 0.7, 0.025),
    "small cube": ('type="box" size="0.01 0.01 0.01"', 0.01, 0.7, 0.01),
    "pen": ('type="capsule" size="0.0045 0.065" euler="0 90 0"', 0.012, 0.5, 0.0045),
}
SIM_DEFAULTS = {"object": "can", "mass_scale": 1.0, "max_force": 15.0, "full_force": 30.0, "assist": False}
# Assist: thumb and another finger each press at least this hard
ASSIST_N = 1.0
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
GRAVITY = 9.81


def _mm(v):
    return " ".join(f"{x / 1000:.4f}" for x in v)


def _quat_z(deg):
    a = math.radians(deg) / 2
    return f"{math.cos(a):.6f} 0 0 {math.sin(a):.6f}"


def _palm_vertices():
    """The outline at the palm's full thickness, and shrunk a little further out, so the edges come out rounded."""
    pts = []
    centre = np.mean(PALM_OUTLINE, 0)
    for z, shrink in ((10, 0), (-10, 0), (14, 6), (-13, 6)):
        for p in PALM_OUTLINE:
            d = (np.array(p) - centre) / np.linalg.norm(np.array(p) - centre)
            pts.append((p[0] - d[0] * shrink, p[1] - d[1] * shrink, z))
    return " ".join(f"{v / 1000:.4f}" for p in pts for v in p)


def _finger(name, base, splay, lengths):
    joints = [f'<joint name="{name}_mcp_abd" axis="0 0 -1" range="-30 30"/>'
              f'<joint name="{name}_mcp_flex" axis="1 0 0" range="-20 100"/>',
              f'<joint name="{name}_pip" axis="1 0 0" range="0 115"/>',
              f'<joint name="{name}_dip" axis="1 0 0" range="-5 90"/>']
    xml = ""
    for i, (segment, length) in enumerate(zip(("prox", "mid", "dist"), lengths)):
        pos = _mm(base) if i == 0 else f"0 {lengths[i - 1] / 1000:.4f} 0"
        quat = f' quat="{_quat_z(-splay)}"' if i == 0 else ""
        radius = FINGER_RADIUS * (1 - (1 - TAPER) * i / 2) / 1000
        xml += (f'<body name="{name}_{segment}" pos="{pos}"{quat}>{joints[i]}'
                f'<geom name="{name}_{segment}" class="hand" type="capsule" '
                f'fromto="0 0 0 0 {length / 1000:.4f} 0" size="{radius:.4f}"/>')
    return xml + "</body>" * 3


def _thumb():
    # hand.js bends the thumb towards (-1, 0, 1) in the hand frame, taken at the rest pose into the metacarpal's frame
    bend = np.array([-0.5, 0.0, math.sqrt(0.5)])
    axis = np.array([bend[2], 0.0, -bend[0]])
    axis = " ".join(f"{v:.4f}" for v in axis / np.linalg.norm(axis))
    meta, prox, dist = (v / 1000 for v in THUMB_LENGTHS)
    r = FINGER_RADIUS / 1000
    return (f'<body name="thumb_meta" pos="{_mm(THUMB_BASE)}" quat="{_quat_z(-THUMB_REST_DEG)}">'
            '<joint name="thumb_cmc_flex" axis="0 0 1" range="-30 80"/>'
            '<joint name="thumb_cmc_abd" axis="1 0 0" range="-15 80"/>'
            f'<geom name="thumb_meta" class="hand" type="capsule" fromto="0 0 0 0 {meta:.4f} 0" size="0.011"/>'
            '<geom name="thumb_thenar" class="hand" type="ellipsoid" pos="-0.008 0.018 0.006" size="0.014 0.024 0.009"/>'
            f'<body name="thumb_prox" pos="0 {meta:.4f} 0"><joint name="thumb_mcp" axis="{axis}" range="-10 80"/>'
            f'<geom name="thumb_prox" class="hand" type="capsule" fromto="0 0 0 0 {prox:.4f} 0" size="{r:.4f}"/>'
            f'<body name="thumb_dist" pos="0 {prox:.4f} 0"><joint name="thumb_ip" axis="{axis}" range="-10 90"/>'
            f'<geom name="thumb_dist" class="hand" type="capsule" fromto="0 0 0 0 {dist:.4f} 0" '
            f'size="{r * TAPER:.4f}"/></body></body></body>')


def model_xml(obj):
    shape, mass, friction, half = OBJECTS[obj]
    motors = "".join(f'<motor name="{j}" joint="{j}"/>' for j in JOINTS)
    return f"""<mujoco model="contact">
<option timestep="{TIMESTEP}" integrator="implicitfast" gravity="0 0 -{GRAVITY}" cone="elliptic"/>
<asset><mesh name="palm" vertex="{_palm_vertices()}"/></asset>
<default><joint damping="0.02" armature="0.0002"/><default class="hand">
<geom friction="1 0.01 0.001" condim="4" density="1100" contype="1" conaffinity="2"/></default></default>
<worldbody>
<light pos="0 -0.3 0.8" dir="0 0.3 -1" directional="true"/>
<geom name="table" type="plane" size="0.5 0.5 0.01" friction="0.6 0.01 0.001" contype="2" conaffinity="3"/>
<body name="target" mocap="true" pos="0 -0.05 0.15" xyaxes="1 0 0 0 -1 0"/>
<body name="palm" pos="0 -0.05 0.15" xyaxes="1 0 0 0 -1 0"><freejoint/>
<geom name="palm" class="hand" type="mesh" mesh="palm"/>
<geom name="palm_pad" class="hand" type="ellipsoid" pos="-0.024 0.035 0.008" size="0.013 0.03 0.008"/>
{''.join(_finger(n, *g) for n, g in FINGER_GEOMETRY.items())}{_thumb()}</body>
<body name="object" pos="0 -0.12 {half + 0.001:.4f}"><freejoint/>
<geom name="object" {shape} mass="{mass}" friction="{friction} 0.01 0.001" condim="4" contype="2"
conaffinity="3"/></body>
</worldbody>
<equality><weld name="hand" body1="palm" body2="target" solref="0.02 1"/>
<weld name="assist" body1="palm" body2="object" active="false"/></equality>
<actuator>{motors}</actuator></mujoco>"""


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


class World:
    """One hand and one object. step() advances it; the hand pose, object and settings can change between steps."""

    def __init__(self, params=None):
        self.params = dict(SIM_DEFAULTS, **(params or {}))
        self._build(self.params["object"])

    def _build(self, obj):
        self.model = mujoco.MjModel.from_xml_string(model_xml(obj))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.object = obj
        self._qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self._vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self._finger_of_joint = np.array([FINGERS.index(j.split("_")[0]) for j in JOINTS])
        # Spreading the fingers stays on the angles, only bending presses
        self._bends = np.array(["abd" not in j or j == "thumb_cmc_abd" for j in JOINTS])
        self._arm = _arms()
        self._object_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._object_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
        self._palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self._assist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "assist")
        # Finger per hand geom, -1 for the palm
        self._finger_of_geom = {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if m.geom_bodyid[g] not in (0, self._object_body):
                part = name.split("_")[0]
                self._finger_of_geom[g] = FINGERS.index(part) if part in FINGERS else -1
        self._base_mass = m.body_mass[self._object_body]
        self._base_inertia = m.body_inertia[self._object_body].copy()
        self.set_mass(self.params["mass_scale"])
        self.data.qpos[self._qadr] = 0.0
        mujoco.mj_forward(m, self.data)
        self.finger_force = np.zeros(5)
        self.palm_force = 0.0
        self.time = 0.0
        self._lock_effort = None

    def set_params(self, params):
        self.params.update(params)
        if self.params["object"] != self.object:
            hand = self.hand_pose()
            self._build(self.params["object"])
            self.set_hand(*hand)
            self.data.qpos[self.model.jnt_qposadr[0]:self.model.jnt_qposadr[0] + 7] = [*hand[0], *hand[1]]
            mujoco.mj_forward(self.model, self.data)
        self.set_mass(self.params["mass_scale"])
        if not self.params["assist"]:
            self.data.eq_active[self._assist] = 0

    def set_mass(self, scale):
        scale = max(float(scale), 0.05)
        self.model.body_mass[self._object_body] = self._base_mass * scale
        self.model.body_inertia[self._object_body] = self._base_inertia * scale

    def hand_pose(self):
        return self.data.mocap_pos[0].copy(), self.data.mocap_quat[0].copy()

    def set_hand(self, pos, quat):
        q = np.asarray(quat, float)
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = q / (np.linalg.norm(q) or 1.0)

    def place_object(self, pos):
        """Puts the object at pos (m), upright and still, and lets go of the assist."""
        adr = self.model.jnt_qposadr[self.model.body_jntadr[self._object_body]]
        dof = self.model.jnt_dofadr[self.model.body_jntadr[self._object_body]]
        self.data.eq_active[self._assist] = 0
        self.data.qpos[adr:adr + 7] = [*pos, 1, 0, 0, 0]
        self.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def reset_object(self):
        half = OBJECTS[self.object][3]
        self.place_object([0.0, -0.12, half + 0.001])

    def step(self, dt, angles, effort, locked=False):
        """Advances dt seconds towards the control layer's angles (degrees, JOINTS order) at effort 0 to 1."""
        if locked:
            if self._lock_effort is None:
                self._lock_effort = effort
            effort = self._lock_effort
        else:
            self._lock_effort = None
        target = np.radians(np.asarray(angles, float))
        cap_full = self.params["max_force"] * self._arm
        cap_effort = cap_full * float(np.clip(effort, 0.0, 1.0))
        d = self.data
        for _ in range(max(1, round(dt / TIMESTEP))):
            q = d.qpos[self._qadr]
            torque = KP * (target - q)
            touching = self.finger_force[self._finger_of_joint] > 0
            # Pressing on the object: effort sets the force, however far past the surface the target is
            pressing = touching & self._bends & (torque > 0)
            torque = np.where(pressing, cap_effort, np.clip(torque, -cap_full, cap_full))
            d.ctrl[:] = torque
            mujoco.mj_step(self.model, d)
            self._read_forces()
            self._update_assist()
        self.time += dt

    def _read_forces(self):
        f6 = np.zeros(6)
        fingers = np.zeros(5)
        palm = 0.0
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if self._object_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self._object_geom else c.geom1
            finger = self._finger_of_geom.get(other)
            if finger is None:
                continue
            mujoco.mj_contactForce(self.model, d, i, f6)
            if finger < 0:
                palm += f6[0]
            else:
                fingers[finger] += f6[0]
        self.finger_force, self.palm_force = fingers, palm

    def _update_assist(self):
        d = self.data
        f = self.finger_force
        if not self.params["assist"]:
            return
        holds = (f[0] >= ASSIST_N and f[1:].max() >= ASSIST_N
                 and self.grip_force() * self.friction() >= self.weight())
        if holds and not d.eq_active[self._assist]:
            self._weld_here()
        elif not holds and d.eq_active[self._assist]:
            d.eq_active[self._assist] = 0

    def _weld_here(self):
        """Activates the assist weld at the object's current pose relative to the palm."""
        d, m = self.data, self.model
        palm_pos, palm_quat = d.xpos[self._palm], d.xquat[self._palm]
        obj_pos, obj_quat = d.xpos[self._object_body], d.xquat[self._object_body]
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, palm_quat)
        rel_pos, rel_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(rel_pos, obj_pos - palm_pos, inv)
        mujoco.mju_mulQuat(rel_quat, inv, obj_quat)
        # weld data: anchor, relative position, relative orientation, torque scale
        m.eq_data[self._assist, 0:3] = 0
        m.eq_data[self._assist, 3:6] = rel_pos
        m.eq_data[self._assist, 6:10] = rel_quat
        m.eq_data[self._assist, 10] = 1
        d.eq_active[self._assist] = 1

    def grip_force(self):
        """Total force the hand presses on the object with (N), what the motors follow."""
        return float(self.finger_force.sum() + self.palm_force)

    def weight(self):
        return float(self.model.body_mass[self._object_body] * GRAVITY)

    def friction(self):
        return float(self.model.geom_friction[self._object_geom, 0])

    def scene(self):
        """Every geom once, for the page to build: type, size, body role and, for meshes, vertices and faces."""
        m = self.model
        geoms = []
        for g in range(m.ngeom):
            kind = mujoco.mjtGeom(m.geom_type[g]).name.removeprefix("mjGEOM_").lower()
            body = m.geom_bodyid[g]
            role = "table" if body == 0 else "object" if body == self._object_body else "hand"
            entry = {"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "type": kind, "role": role,
                     "size": m.geom_size[g].round(5).tolist()}
            if kind == "mesh":
                mesh = m.geom_dataid[g]
                v0, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                f0, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
                entry["vertices"] = m.mesh_vert[v0:v0 + nv].round(5).tolist()
                entry["faces"] = m.mesh_face[f0:f0 + nf].tolist()
            geoms.append(entry)
        return {"object": self.object, "geoms": geoms}

    def poses(self):
        """Position and orientation (3 x 3, row major) of every geom, in the order of scene()."""
        d = self.data
        return {"pos": d.geom_xpos.round(5).tolist(), "mat": d.geom_xmat.round(5).tolist()}

    def readout(self):
        return {"grip_force": round(self.grip_force(), 2), "finger_force": self.finger_force.round(2).tolist(),
                "palm_force": round(self.palm_force, 2), "weight": round(self.weight(), 3),
                "friction": self.friction(), "assisted": bool(self.data.eq_active[self._assist]),
                "object_height": round(float(self.data.xpos[self._object_body][2]), 4), "time": round(self.time, 3)}
