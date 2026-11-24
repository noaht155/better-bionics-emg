// The contact sim, for the Quest app and for the PC through sim.py's World, with two hands and a pool of objects
// (sim.scene_xml). Each hand follows its own angles with the per-finger force rule sim.py describes. The prosthetic
// hand ("p_") has sim.py's grasp assist; the free hand ("f_") holds whatever it touches while its grab is on, so
// grabbing with a controller is never fiddly. The rule's constants come in the XML as custom numerics. Coordinates
// and units as in sim.py.
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mujoco/mujoco.h>

#define NJ 20
#define NF 5
#define NH 2
#define MAXOBJ 64
#define NSURF 8
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static const char* FINGERS[NF] = {"thumb", "index", "middle", "ring", "pinky"};
static const char* PREFIX[NH] = {"p_", "f_"};

typedef struct {
  int palm, mocap;
  int act[NJ], qadr[NJ], finger_of_joint[NJ], bends[NJ];
  double angles[NJ], effort, lock_effort;
  int locked, was_locked, grab;
  double finger_force[NF], palm_force, force_on[MAXOBJ];
  // Each finger's force on the surface rail: a finger pressing on it presses with effort as on an object
  double rail_force[NF];
  // Force on everything the hand touches (objects, desk, floor), that force times each surface's own friction, and
  // the force per surface kind (each geom's user value, sim.SURFACES), for the haptics
  double touch_force, touch_friction, touch_on[NSURF];
  int held;
  int hold_eq[MAXOBJ];
} Hand;

typedef struct {
  mjModel* m;
  mjData* d;
  double kp, kp_press, assist_n, max_mass_scale, gravity, arm[NJ];
  Hand hand[NH];
  int nobj, obj_body[MAXOBJ], obj_geom[MAXOBJ];
  // Parked objects wait off the desk out of the collisions and held still, so the unused pool costs nothing
  int parked[MAXOBJ];
  double park_qpos[MAXOBJ][7];
  double base_mass[MAXOBJ], base_inertia[MAXOBJ][3];
  double mass_scale;
  // Per geom: the hand it belongs to (-1 none), its finger (-1 palm), the object it is (-1 none)
  int *hand_of_geom, *finger_of_geom, *obj_of_geom, *is_rail;
  int floor_geom;
  double max_force;
  int assist_on, blowups;
  double time;
} World;

static double clip(double x, double lo, double hi) { return x < lo ? lo : x > hi ? hi : x; }

static int finger_index(const char* name) {
  for (int i = 0; i < NF; i++) {
    size_t n = strlen(FINGERS[i]);
    if (strncmp(name, FINGERS[i], n) == 0 && name[n] == '_') return i;
  }
  return -1;
}

static const mjtNum* numeric(const mjModel* m, const char* name, int size) {
  int id = mj_name2id(m, mjOBJ_NUMERIC, name);
  if (id < 0 || m->numeric_size[id] != size) return NULL;
  return m->numeric_data + m->numeric_adr[id];
}

static void fail(char* err, int errlen, const char* msg) {
  strncpy(err, msg, errlen - 1);
  err[errlen - 1] = 0;
}

EXPORT void bb_free(World* w) {
  if (!w) return;
  mj_deleteData(w->d);
  mj_deleteModel(w->m);
  free(w->hand_of_geom);
  free(w->finger_of_geom);
  free(w->obj_of_geom);
  free(w->is_rail);
  free(w);
}

static int setup_hand(World* w, int h) {
  mjModel* m = w->m;
  Hand* hd = &w->hand[h];
  char name[64];
  snprintf(name, sizeof(name), "%spalm", PREFIX[h]);
  hd->palm = mj_name2id(m, mjOBJ_BODY, name);
  snprintf(name, sizeof(name), "%starget", PREFIX[h]);
  int target = mj_name2id(m, mjOBJ_BODY, name);
  if (hd->palm < 0 || target < 0 || m->body_mocapid[target] < 0) return 0;
  hd->mocap = m->body_mocapid[target];
  int j = 0;
  size_t plen = strlen(PREFIX[h]);
  for (int a = 0; a < m->nu && j < NJ; a++) {
    int joint = m->actuator_trnid[2 * a];
    const char* jn = mj_id2name(m, mjOBJ_JOINT, joint);
    if (!jn || strncmp(jn, PREFIX[h], plen)) continue;
    const char* bare = jn + plen;
    hd->act[j] = a;
    hd->qadr[j] = m->jnt_qposadr[joint];
    hd->finger_of_joint[j] = finger_index(bare);
    // Spreading the fingers stays on the angles, only bending presses
    hd->bends[j] = !strstr(bare, "abd") || !strcmp(bare, "thumb_cmc_abd");
    j++;
  }
  if (j != NJ) return 0;
  for (int o = 0; o < w->nobj; o++) {
    snprintf(name, sizeof(name), "%shold%d", PREFIX[h], o);
    hd->hold_eq[o] = mj_name2id(m, mjOBJ_EQUALITY, name);
    if (hd->hold_eq[o] < 0) return 0;
  }
  hd->held = -1;
  return 1;
}

EXPORT World* bb_create(const char* xml, int len, char* err, int errlen) {
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, len);
  mjModel* m = mj_loadXML("model.xml", &vfs, err, errlen);
  mj_deleteVFS(&vfs);
  if (!m) return NULL;
  const char* names[] = {"kp", "kp_press", "assist_n", "max_mass_scale", "gravity", "arm"};
  const mjtNum* v[6];
  for (int i = 0; i < 6; i++) {
    v[i] = numeric(m, names[i], i == 5 ? NJ : 1);
    if (!v[i]) {
      fail(err, errlen, "model has no force rule constants, export it with export_model.py");
      mj_deleteModel(m);
      return NULL;
    }
  }
  World* w = calloc(1, sizeof(World));
  w->m = m;
  w->d = mj_makeData(m);
  w->kp = *v[0];
  w->kp_press = *v[1];
  w->assist_n = *v[2];
  w->max_mass_scale = *v[3];
  w->gravity = *v[4];
  for (int j = 0; j < NJ; j++) w->arm[j] = v[5][j];
  char name[32];
  for (w->nobj = 0; w->nobj < MAXOBJ; w->nobj++) {
    snprintf(name, sizeof(name), "obj%d", w->nobj);
    int body = mj_name2id(m, mjOBJ_BODY, name);
    if (body < 0) break;
    w->obj_body[w->nobj] = body;
    w->obj_geom[w->nobj] = mj_name2id(m, mjOBJ_GEOM, name);
    w->base_mass[w->nobj] = m->body_mass[body];
    w->parked[w->nobj] = 1;
    m->geom_contype[w->obj_geom[w->nobj]] = m->geom_conaffinity[w->obj_geom[w->nobj]] = 0;
    memcpy(w->park_qpos[w->nobj], m->qpos0 + m->jnt_qposadr[m->body_jntadr[body]], 7 * sizeof(double));
    for (int k = 0; k < 3; k++) w->base_inertia[w->nobj][k] = m->body_inertia[3 * body + k];
  }
  for (int h = 0; h < NH; h++) {
    if (!setup_hand(w, h)) {
      fail(err, errlen, "model lacks a hand, its joints or its hold welds");
      bb_free(w);
      return NULL;
    }
  }
  w->hand_of_geom = malloc(sizeof(int) * m->ngeom);
  w->finger_of_geom = malloc(sizeof(int) * m->ngeom);
  w->obj_of_geom = malloc(sizeof(int) * m->ngeom);
  w->is_rail = malloc(sizeof(int) * m->ngeom);
  for (int g = 0; g < m->ngeom; g++) {
    w->hand_of_geom[g] = w->finger_of_geom[g] = w->obj_of_geom[g] = -1;
    for (int o = 0; o < w->nobj; o++)
      if (w->obj_geom[o] == g) w->obj_of_geom[g] = o;
    for (int b = m->geom_bodyid[g]; b > 0; b = m->body_parentid[b])
      for (int h = 0; h < NH; h++)
        if (b == w->hand[h].palm) w->hand_of_geom[g] = h;
    const char* gn = mj_id2name(m, mjOBJ_GEOM, g);
    w->is_rail[g] = gn && !strncmp(gn, "rail", 4);
    if (w->hand_of_geom[g] >= 0 && gn) w->finger_of_geom[g] = finger_index(gn + strlen(PREFIX[w->hand_of_geom[g]]));
  }
  w->floor_geom = mj_name2id(m, mjOBJ_GEOM, "floor");
  w->max_force = 15;
  for (int h = 0; h < NH; h++)
    for (int j = 0; j < NJ; j++) w->d->qpos[w->hand[h].qadr[j]] = 0;
  mj_forward(m, w->d);
  return w;
}

EXPORT void bb_set_params(World* w, double mass_scale, double max_force, int assist) {
  mass_scale = clip(mass_scale, 0.05, w->max_mass_scale);
  if (mass_scale != w->mass_scale) {
    w->mass_scale = mass_scale;
    for (int o = 0; o < w->nobj; o++) {
      w->m->body_mass[w->obj_body[o]] = w->base_mass[o] * mass_scale;
      for (int k = 0; k < 3; k++) w->m->body_inertia[3 * w->obj_body[o] + k] = w->base_inertia[o][k] * mass_scale;
    }
    // The constants MuJoCo derives from the masses at load (constraint weights among them) have to follow, or a 3x
    // can sank into the desk and blew up when a light grip dropped it (2026-11-20). mj_setConst overwrites the
    // state it's given, so it gets a scratch one
    mjData* scratch = mj_makeData(w->m);
    mj_setConst(w->m, scratch);
    mj_deleteData(scratch);
  }
  w->max_force = max_force;
  w->assist_on = assist;
  Hand* p = &w->hand[0];
  if (!assist && p->held >= 0) {
    w->d->eq_active[p->hold_eq[p->held]] = 0;
    p->held = -1;
  }
}

// The hand's angles (degrees, JOINTS order), effort 0 to 1, lock (holds the effort of the moment it locked) and, for
// the free hand, grab
EXPORT void bb_set_input(World* w, int h, const double* angles, double effort, int locked, int grab) {
  Hand* hd = &w->hand[h];
  memcpy(hd->angles, angles, sizeof(hd->angles));
  hd->effort = effort;
  hd->locked = locked;
  hd->grab = grab;
}

// Where the hand's target is (the palm follows it through the weld); teleport puts the palm there at once
EXPORT void bb_set_hand(World* w, int h, const double* pos, const double* quat, int teleport) {
  Hand* hd = &w->hand[h];
  double n = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  if (n == 0) return;
  for (int k = 0; k < 3; k++) w->d->mocap_pos[3 * hd->mocap + k] = pos[k];
  for (int k = 0; k < 4; k++) w->d->mocap_quat[4 * hd->mocap + k] = quat[k] / n;
  if (teleport) {
    int adr = w->m->jnt_qposadr[w->m->body_jntadr[hd->palm]], dof = w->m->jnt_dofadr[w->m->body_jntadr[hd->palm]];
    memcpy(w->d->qpos + adr, w->d->mocap_pos + 3 * hd->mocap, 3 * sizeof(double));
    memcpy(w->d->qpos + adr + 3, w->d->mocap_quat + 4 * hd->mocap, 4 * sizeof(double));
    memset(w->d->qvel + dof, 0, 6 * sizeof(double));
    mj_forward(w->m, w->d);
  }
}

static void release(World* w, int o) {
  for (int h = 0; h < NH; h++)
    if (w->hand[h].held == o) {
      w->d->eq_active[w->hand[h].hold_eq[o]] = 0;
      w->hand[h].held = -1;
    }
}

static void hold_parked(World* w) {
  for (int o = 0; o < w->nobj; o++) {
    if (!w->parked[o]) continue;
    int jnt = w->m->body_jntadr[w->obj_body[o]];
    memcpy(w->d->qpos + w->m->jnt_qposadr[jnt], w->park_qpos[o], 7 * sizeof(double));
    memset(w->d->qvel + w->m->jnt_dofadr[jnt], 0, 6 * sizeof(double));
  }
}

// Puts object o at pos with orientation quat (w x y z), still, and lets go of any hold on it. A parked object is
// out of the collisions and stays where it's put
EXPORT void bb_place_object(World* w, int o, const double* pos, const double* quat, int parked) {
  int jnt = w->m->body_jntadr[w->obj_body[o]];
  int adr = w->m->jnt_qposadr[jnt], dof = w->m->jnt_dofadr[jnt];
  release(w, o);
  memcpy(w->d->qpos + adr, pos, 3 * sizeof(double));
  memcpy(w->d->qpos + adr + 3, quat, 4 * sizeof(double));
  memset(w->d->qvel + dof, 0, 6 * sizeof(double));
  memcpy(w->park_qpos[o], w->d->qpos + adr, 7 * sizeof(double));
  w->parked[o] = parked;
  w->m->geom_contype[w->obj_geom[o]] = parked ? 0 : 2;
  w->m->geom_conaffinity[w->obj_geom[o]] = parked ? 0 : 3;
  mj_forward(w->m, w->d);
}

EXPORT int bb_nobjects(World* w) { return w->nobj; }

// Hand h's target position and orientation (w x y z)
EXPORT void bb_hand_target(World* w, int h, double* pos, double* quat) {
  memcpy(pos, w->d->mocap_pos + 3 * w->hand[h].mocap, 3 * sizeof(double));
  memcpy(quat, w->d->mocap_quat + 4 * w->hand[h].mocap, 4 * sizeof(double));
}

// Hand h's joint angles (degrees, JOINTS order)
EXPORT void bb_joint_angles(World* w, int h, double* out) {
  for (int j = 0; j < NJ; j++) out[j] = w->d->qpos[w->hand[h].qadr[j]] * 180 / M_PI;
}

// Moves the floor to height below the desk top; objects resting on it need placing again
EXPORT void bb_set_floor(World* w, double height) {
  if (w->floor_geom >= 0) w->m->geom_pos[3 * w->floor_geom + 2] = -height;
}

static double grip_force(const Hand* hd) {
  double f = hd->palm_force;
  for (int i = 0; i < NF; i++) f += hd->finger_force[i];
  return f;
}

static void read_forces(World* w) {
  mjtNum f6[6];
  for (int h = 0; h < NH; h++) {
    Hand* hd = &w->hand[h];
    memset(hd->finger_force, 0, sizeof(hd->finger_force));
    memset(hd->rail_force, 0, sizeof(hd->rail_force));
    memset(hd->force_on, 0, sizeof(double) * w->nobj);
    hd->palm_force = hd->touch_force = hd->touch_friction = 0;
    memset(hd->touch_on, 0, sizeof(hd->touch_on));
  }
  for (int i = 0; i < w->d->ncon; i++) {
    mjContact* c = w->d->contact + i;
    int g0 = c->geom[0], g1 = c->geom[1];
    if ((w->hand_of_geom[g0] >= 0) != (w->hand_of_geom[g1] >= 0)) {
      int hand = w->hand_of_geom[g0] >= 0 ? g0 : g1, surface = hand == g0 ? g1 : g0;
      mj_contactForce(w->m, w->d, i, f6);
      Hand* hd = &w->hand[w->hand_of_geom[hand]];
      hd->touch_force += f6[0];
      // The surface's own friction, not the contact's: the hand's high friction would win against every object
      hd->touch_friction += f6[0] * w->m->geom_friction[3 * surface];
      int kind = w->m->nuser_geom > 0 ? (int)w->m->geom_user[surface * w->m->nuser_geom] : -1;
      if (kind >= 0 && kind < NSURF) hd->touch_on[kind] += f6[0];
      int finger = w->finger_of_geom[hand];
      if (w->is_rail[surface] && finger >= 0) hd->rail_force[finger] += f6[0];
    }
    int o = w->obj_of_geom[c->geom[0]], other = c->geom[1];
    if (o < 0) {
      o = w->obj_of_geom[c->geom[1]];
      other = c->geom[0];
    }
    if (o < 0) continue;
    int h = w->hand_of_geom[other];
    if (h < 0) continue;
    mj_contactForce(w->m, w->d, i, f6);
    Hand* hd = &w->hand[h];
    int finger = w->finger_of_geom[other];
    if (finger < 0) hd->palm_force += f6[0];
    else hd->finger_force[finger] += f6[0];
    hd->force_on[o] += f6[0];
  }
}

static int most_touched(const World* w, const Hand* hd) {
  int best = -1;
  for (int o = 0; o < w->nobj; o++)
    if (hd->force_on[o] > 0 && (best < 0 || hd->force_on[o] > hd->force_on[best])) best = o;
  return best;
}

// Activates hand h's hold weld on object o at their current relative pose
static void weld_here(World* w, int h, int o) {
  mjData* d = w->d;
  Hand* hd = &w->hand[h];
  int palm = hd->palm, body = w->obj_body[o];
  mjtNum inv[4], rel_pos[3], rel_quat[4], diff[3];
  mju_negQuat(inv, d->xquat + 4 * palm);
  mju_sub3(diff, d->xpos + 3 * body, d->xpos + 3 * palm);
  mju_rotVecQuat(rel_pos, diff, inv);
  mju_mulQuat(rel_quat, inv, d->xquat + 4 * body);
  // weld data: anchor, relative position, relative orientation, torque scale
  mjtNum* e = w->m->eq_data + mjNEQDATA * hd->hold_eq[o];
  mju_zero3(e);
  mju_copy3(e + 3, rel_pos);
  mju_copy4(e + 6, rel_quat);
  e[10] = 1;
  d->eq_active[hd->hold_eq[o]] = 1;
  hd->held = o;
}

static void update_holds(World* w) {
  // Prosthetic hand: sim.py's assist, thumb and another finger pressing hard enough for friction to carry the weight
  Hand* p = &w->hand[0];
  if (w->assist_on) {
    int o = p->held >= 0 ? p->held : most_touched(w, p);
    int holds = 0;
    if (o >= 0) {
      double most = 0;
      for (int i = 1; i < NF; i++) most = fmax(most, p->finger_force[i]);
      double weight = w->m->body_mass[w->obj_body[o]] * w->gravity;
      holds = p->finger_force[0] >= w->assist_n && most >= w->assist_n &&
              grip_force(p) * w->m->geom_friction[3 * w->obj_geom[o]] >= weight;
    }
    if (holds && p->held < 0) weld_here(w, 0, o);
    else if (!holds && p->held >= 0) {
      w->d->eq_active[p->hold_eq[p->held]] = 0;
      p->held = -1;
    }
  }
  // Free hand: holds what it touches while the grab is on
  Hand* f = &w->hand[1];
  if (f->grab && f->held < 0) {
    int o = most_touched(w, f);
    if (o >= 0) weld_here(w, 1, o);
  } else if (!f->grab && f->held >= 0) {
    w->d->eq_active[f->hold_eq[f->held]] = 0;
    f->held = -1;
  }
}

static void drive(World* w, Hand* hd) {
  double effort = hd->effort;
  if (hd->locked) {
    if (!hd->was_locked) hd->lock_effort = effort;
    effort = hd->lock_effort;
  }
  hd->was_locked = hd->locked;
  effort = clip(effort, 0, 1);
  mjData* d = w->d;
  for (int j = 0; j < NJ; j++) {
    double target = hd->angles[j] * M_PI / 180, q = d->qpos[hd->qadr[j]];
    double cap_full = w->max_force * w->arm[j], cap_effort = cap_full * effort;
    int finger = hd->finger_of_joint[j];
    int pressing = (hd->finger_force[finger] > 0 || hd->rail_force[finger] > 0) && hd->bends[j];
    d->ctrl[hd->act[j]] = pressing ? clip(w->kp_press * (target - q), -cap_full, cap_effort)
                                   : clip(w->kp * (target - q), -cap_full, cap_full);
  }
}

EXPORT void bb_step(World* w, double dt) {
  mjModel* m = w->m;
  mjData* d = w->d;
  long n = lround(dt / m->opt.timestep);
  for (long s = 0; s < (n > 1 ? n : 1); s++) {
    for (int h = 0; h < NH; h++) drive(w, &w->hand[h]);
    double before = d->time;
    mj_step(m, d);
    hold_parked(w);
    if (d->time < before) {
      // MuJoCo resets everything when it goes unstable, the objects back to where the XML parks them: put the
      // palms back on their targets and drop the holds
      w->blowups++;
      for (int h = 0; h < NH; h++) {
        Hand* hd = &w->hand[h];
        hd->held = -1;
        bb_set_hand(w, h, d->mocap_pos + 3 * hd->mocap, d->mocap_quat + 4 * hd->mocap, 1);
      }
    }
    read_forces(w);
    update_holds(w);
  }
  w->time += dt;
}

EXPORT int bb_ngeom(World* w) { return w->m->ngeom; }

// role: 0 desk or floor, 1 object, 2 hand, 3 the surface rail. hand: 0 prosthetic, 1 free. finger: -1 palm. obj:
// the object's index
EXPORT void bb_geom(World* w, int g, int* type, int* role, int* hand, int* finger, int* obj, double* size) {
  *type = w->m->geom_type[g];
  *hand = w->hand_of_geom[g];
  *finger = w->finger_of_geom[g];
  *obj = w->obj_of_geom[g];
  const char* name = mj_id2name(w->m, mjOBJ_GEOM, g);
  *role = *obj >= 0 ? 1 : *hand >= 0 ? 2 : name && !strncmp(name, "rail", 4) ? 3 : 0;
  memcpy(size, w->m->geom_size + 3 * g, 3 * sizeof(double));
}

// A geom's kind of surface (its user value, sim.SURFACES), -1 for none
EXPORT int bb_geom_surface(World* w, int g) {
  return w->m->nuser_geom > 0 ? (int)w->m->geom_user[g * w->m->nuser_geom] : -1;
}

// A geom's sliding friction, which sets its colour (sim.SURFACE_FRICTION)
EXPORT double bb_geom_friction(World* w, int g) {
  return w->m->geom_friction[3 * g];
}

// Vertex and face counts of a mesh geom, 0 for the rest; with buffers, also copies them out
EXPORT int bb_mesh(World* w, int g, float* vert, int* face, int* nface) {
  int mesh = w->m->geom_dataid[g];
  if (w->m->geom_type[g] != mjGEOM_MESH || mesh < 0) return 0;
  int nv = w->m->mesh_vertnum[mesh], nf = w->m->mesh_facenum[mesh];
  *nface = nf;
  if (vert) memcpy(vert, w->m->mesh_vert + 3 * w->m->mesh_vertadr[mesh], 3 * nv * sizeof(float));
  if (face) memcpy(face, w->m->mesh_face + 3 * w->m->mesh_faceadr[mesh], 3 * nf * sizeof(int));
  return nv;
}

// Every geom's position (3) and orientation (3 x 3, row major)
EXPORT void bb_poses(World* w, float* pos, float* mat) {
  for (int i = 0; i < 3 * w->m->ngeom; i++) pos[i] = (float)w->d->geom_xpos[i];
  for (int i = 0; i < 9 * w->m->ngeom; i++) mat[i] = (float)w->d->geom_xmat[i];
}

// Object o's position and orientation (w x y z)
EXPORT void bb_object_pose(World* w, int o, double* pos, double* quat) {
  memcpy(pos, w->d->xpos + 3 * w->obj_body[o], 3 * sizeof(double));
  memcpy(quat, w->d->xquat + 4 * w->obj_body[o], 4 * sizeof(double));
}

// Prosthetic grip force, its 5 finger forces and palm force, the free hand's grip force, the object each hand
// holds (-1 none), the object the prosthetic hand presses on most, its weight, time, blow-ups, then the prosthetic
// hand's force on everything it touches, those surfaces' friction weighted by force (0 while it touches nothing) and
// the kind of surface it presses on most (-1 none)
EXPORT void bb_readout(World* w, double* out) {
  Hand *p = &w->hand[0], *f = &w->hand[1];
  out[0] = grip_force(p);
  for (int i = 0; i < NF; i++) out[1 + i] = p->finger_force[i];
  out[6] = p->palm_force;
  out[7] = grip_force(f);
  out[8] = p->held;
  out[9] = f->held;
  int o = most_touched(w, p);
  out[10] = o;
  out[11] = o >= 0 ? w->m->body_mass[w->obj_body[o]] * w->gravity : 0;
  out[12] = w->time;
  out[13] = w->blowups;
  out[14] = p->touch_force;
  out[15] = p->touch_force > 0 ? p->touch_friction / p->touch_force : 0;
  int kind = -1;
  for (int k = 0; k < NSURF; k++)
    if (p->touch_on[k] > 0 && (kind < 0 || p->touch_on[k] > p->touch_on[kind])) kind = k;
  out[16] = kind;
}

// A hand on its own for showing poses (the recording cue's target): no physics, only where its parts are at given
// angles. The XML is export_model.py's ghost hand, its motors in JOINTS order
typedef struct {
  mjModel* m;
  mjData* d;
  int qadr[NJ];
  // Bodies at the 21 landmarks (MediaPipe order, the wrist is the palm) and each finger's last geom for its tip
  int body[21], tip_geom[NF];
} Fk;

EXPORT Fk* bb_fk_create(const char* xml, int len, char* err, int errlen) {
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, len);
  mjModel* m = mj_loadXML("model.xml", &vfs, err, errlen);
  mj_deleteVFS(&vfs);
  if (!m) return NULL;
  if (m->nu != NJ) {
    fail(err, errlen, "ghost hand needs its 20 motors");
    mj_deleteModel(m);
    return NULL;
  }
  Fk* fk = calloc(1, sizeof(Fk));
  fk->m = m;
  fk->d = mj_makeData(m);
  for (int j = 0; j < NJ; j++) fk->qadr[j] = m->jnt_qposadr[m->actuator_trnid[2 * j]];
  static const char* SEG[2][3] = {{"meta", "prox", "dist"}, {"prox", "mid", "dist"}};
  char name[40];
  fk->body[0] = 1;
  for (int f = 0; f < NF; f++) {
    for (int s = 0; s < 3; s++) {
      snprintf(name, sizeof name, "g_%s_%s", FINGERS[f], SEG[f > 0][s]);
      fk->body[1 + 4 * f + s] = mj_name2id(m, mjOBJ_BODY, name);
    }
    fk->tip_geom[f] = mj_name2id(m, mjOBJ_GEOM, name);
  }
  return fk;
}

EXPORT void bb_fk_free(Fk* fk) {
  if (!fk) return;
  mj_deleteData(fk->d);
  mj_deleteModel(fk->m);
  free(fk);
}

EXPORT int bb_fk_ngeom(Fk* fk) { return fk->m->ngeom; }

EXPORT void bb_fk_geom(Fk* fk, int g, int* type, int* finger, double* size) {
  *type = fk->m->geom_type[g];
  const char* gn = mj_id2name(fk->m, mjOBJ_GEOM, g);
  *finger = gn && strlen(gn) > 2 ? finger_index(gn + 2) : -1;
  memcpy(size, fk->m->geom_size + 3 * g, 3 * sizeof(double));
}

EXPORT int bb_fk_mesh(Fk* fk, int g, float* vert, int* face, int* nface) {
  int mesh = fk->m->geom_dataid[g];
  if (fk->m->geom_type[g] != mjGEOM_MESH || mesh < 0) return 0;
  int nv = fk->m->mesh_vertnum[mesh], nf = fk->m->mesh_facenum[mesh];
  *nface = nf;
  if (vert) memcpy(vert, fk->m->mesh_vert + 3 * fk->m->mesh_vertadr[mesh], 3 * nv * sizeof(float));
  if (face) memcpy(face, fk->m->mesh_face + 3 * fk->m->mesh_faceadr[mesh], 3 * nf * sizeof(int));
  return nv;
}

// Geom positions and orientations (row major) in the palm's frame at these angles (degrees, JOINTS order)
EXPORT void bb_fk(Fk* fk, const double* angles, float* pos, float* mat) {
  for (int j = 0; j < NJ; j++) fk->d->qpos[fk->qadr[j]] = angles[j] * M_PI / 180;
  mj_kinematics(fk->m, fk->d);
  for (int i = 0; i < 3 * fk->m->ngeom; i++) pos[i] = (float)fk->d->geom_xpos[i];
  for (int i = 0; i < 9 * fk->m->ngeom; i++) mat[i] = (float)fk->d->geom_xmat[i];
}

// The 21 landmarks (MediaPipe order, m) in the palm's frame at these angles: joint centres, and the fingertips at the
// far end of each last segment, whose capsule starts at its body's origin
EXPORT void bb_fk_points(Fk* fk, const double* angles, float* out) {
  for (int j = 0; j < NJ; j++) fk->d->qpos[fk->qadr[j]] = angles[j] * M_PI / 180;
  mj_kinematics(fk->m, fk->d);
  for (int f = 0; f < NF; f++) {
    for (int s = 0; s < 3; s++)
      for (int k = 0; k < 3; k++) out[3 * (1 + 4 * f + s) + k] = (float)fk->d->xpos[3 * fk->body[1 + 4 * f + s] + k];
    int g = fk->tip_geom[f], b = fk->body[3 + 4 * f];
    for (int k = 0; k < 3; k++)
      out[3 * (4 + 4 * f) + k] = (float)(2 * fk->d->geom_xpos[3 * g + k] - fk->d->xpos[3 * b + k]);
  }
  for (int k = 0; k < 3; k++) out[k] = (float)fk->d->xpos[3 * fk->body[0] + k];
}
