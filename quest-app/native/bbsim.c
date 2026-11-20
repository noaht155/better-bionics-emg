// sim.py's World in C for the Quest app: one hand and one object, the same per-finger force rule, assist and blow-up
// recovery. The model XML comes from export_model.py, which adds the rule's constants to it as custom numerics, so
// the XML alone carries everything sim.py would. Coordinates and units as in sim.py (metres, z up, degrees in).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mujoco/mujoco.h>

#define NJ 20
#define NF 5
#define EXPORT __attribute__((visibility("default")))

static const char* FINGERS[NF] = {"thumb", "index", "middle", "ring", "pinky"};

typedef struct {
  mjModel* m;
  mjData* d;
  double kp, kp_press, assist_n, max_mass_scale, gravity, half;
  double arm[NJ];
  int qadr[NJ], finger_of_joint[NJ], bends[NJ];
  // -2 not part of the hand, -1 palm, else the finger
  int* finger_of_geom;
  int obj_body, obj_geom, palm, assist;
  double base_mass, base_inertia[3];
  double max_force;
  int assist_on;
  double finger_force[NF], palm_force, time;
  int blowups, was_locked;
  double lock_effort;
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

static void set_mass(World* w, double scale) {
  scale = clip(scale, 0.05, w->max_mass_scale);
  w->m->body_mass[w->obj_body] = w->base_mass * scale;
  for (int k = 0; k < 3; k++) w->m->body_inertia[3 * w->obj_body + k] = w->base_inertia[k] * scale;
}

EXPORT void bb_free(World* w) {
  if (!w) return;
  mj_deleteData(w->d);
  mj_deleteModel(w->m);
  free(w->finger_of_geom);
  free(w);
}

EXPORT World* bb_create(const char* xml, int len, char* err, int errlen) {
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, len);
  mjModel* m = mj_loadXML("model.xml", &vfs, err, errlen);
  mj_deleteVFS(&vfs);
  if (!m) return NULL;
  const char* names[] = {"kp", "kp_press", "assist_n", "max_mass_scale", "gravity", "object_half", "arm"};
  const mjtNum* v[7];
  for (int i = 0; i < 7; i++) {
    v[i] = numeric(m, names[i], i == 6 ? NJ : 1);
    if (!v[i] || m->nu != NJ) {
      strncpy(err, "model has no force rule constants, export it with export_model.py", errlen - 1);
      err[errlen - 1] = 0;
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
  w->half = *v[5];
  for (int j = 0; j < NJ; j++) {
    int joint = m->actuator_trnid[2 * j];
    const char* name = mj_id2name(m, mjOBJ_JOINT, joint);
    w->arm[j] = v[6][j];
    w->qadr[j] = m->jnt_qposadr[joint];
    w->finger_of_joint[j] = finger_index(name);
    // Spreading the fingers stays on the angles, only bending presses
    w->bends[j] = !strstr(name, "abd") || !strcmp(name, "thumb_cmc_abd");
  }
  w->obj_body = mj_name2id(m, mjOBJ_BODY, "object");
  w->obj_geom = mj_name2id(m, mjOBJ_GEOM, "object");
  w->palm = mj_name2id(m, mjOBJ_BODY, "palm");
  w->assist = mj_name2id(m, mjOBJ_EQUALITY, "assist");
  w->finger_of_geom = malloc(sizeof(int) * m->ngeom);
  for (int g = 0; g < m->ngeom; g++) {
    int body = m->geom_bodyid[g];
    const char* name = mj_id2name(m, mjOBJ_GEOM, g);
    w->finger_of_geom[g] = (body == 0 || body == w->obj_body) ? -2 : name ? finger_index(name) : -1;
  }
  w->base_mass = m->body_mass[w->obj_body];
  for (int k = 0; k < 3; k++) w->base_inertia[k] = m->body_inertia[3 * w->obj_body + k];
  w->max_force = 15;
  for (int j = 0; j < NJ; j++) w->d->qpos[w->qadr[j]] = 0;
  mj_forward(m, w->d);
  return w;
}

EXPORT void bb_set_params(World* w, double mass_scale, double max_force, int assist) {
  set_mass(w, mass_scale);
  w->max_force = max_force;
  w->assist_on = assist;
  if (!assist) w->d->eq_active[w->assist] = 0;
}

EXPORT void bb_hand(World* w, double* pos, double* quat) {
  memcpy(pos, w->d->mocap_pos, 3 * sizeof(double));
  memcpy(quat, w->d->mocap_quat, 4 * sizeof(double));
}

// teleport also moves the palm there at once, for a new world taking over the old one's hand
EXPORT void bb_set_hand(World* w, const double* pos, const double* quat, int teleport) {
  double n = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  if (n == 0) return;
  for (int k = 0; k < 3; k++) w->d->mocap_pos[k] = pos[k];
  for (int k = 0; k < 4; k++) w->d->mocap_quat[k] = quat[k] / n;
  if (teleport) {
    memcpy(w->d->qpos, w->d->mocap_pos, 3 * sizeof(double));
    memcpy(w->d->qpos + 3, w->d->mocap_quat, 4 * sizeof(double));
    mj_forward(w->m, w->d);
  }
}

EXPORT void bb_place_object(World* w, const double* pos) {
  int jnt = w->m->body_jntadr[w->obj_body];
  int adr = w->m->jnt_qposadr[jnt], dof = w->m->jnt_dofadr[jnt];
  w->d->eq_active[w->assist] = 0;
  double q[7] = {pos[0], pos[1], pos[2], 1, 0, 0, 0};
  memcpy(w->d->qpos + adr, q, sizeof(q));
  memset(w->d->qvel + dof, 0, 6 * sizeof(double));
  mj_forward(w->m, w->d);
}

EXPORT void bb_reset_object(World* w) {
  double pos[3] = {0, -0.12, w->half + 0.001};
  bb_place_object(w, pos);
}

static double grip_force(const World* w) {
  double f = w->palm_force;
  for (int i = 0; i < NF; i++) f += w->finger_force[i];
  return f;
}

static double weight(const World* w) { return w->m->body_mass[w->obj_body] * w->gravity; }

static void read_forces(World* w) {
  mjtNum f6[6];
  memset(w->finger_force, 0, sizeof(w->finger_force));
  w->palm_force = 0;
  for (int i = 0; i < w->d->ncon; i++) {
    mjContact* c = w->d->contact + i;
    if (c->geom[0] != w->obj_geom && c->geom[1] != w->obj_geom) continue;
    int other = c->geom[0] == w->obj_geom ? c->geom[1] : c->geom[0];
    int finger = w->finger_of_geom[other];
    if (finger == -2) continue;
    mj_contactForce(w->m, w->d, i, f6);
    if (finger < 0) w->palm_force += f6[0];
    else w->finger_force[finger] += f6[0];
  }
}

static void weld_here(World* w) {
  mjData* d = w->d;
  mjtNum inv[4], rel_pos[3], rel_quat[4], diff[3];
  mju_negQuat(inv, d->xquat + 4 * w->palm);
  mju_sub3(diff, d->xpos + 3 * w->obj_body, d->xpos + 3 * w->palm);
  mju_rotVecQuat(rel_pos, diff, inv);
  mju_mulQuat(rel_quat, inv, d->xquat + 4 * w->obj_body);
  // weld data: anchor, relative position, relative orientation, torque scale
  mjtNum* e = w->m->eq_data + mjNEQDATA * w->assist;
  mju_zero3(e);
  mju_copy3(e + 3, rel_pos);
  mju_copy4(e + 6, rel_quat);
  e[10] = 1;
  d->eq_active[w->assist] = 1;
}

static void update_assist(World* w) {
  if (!w->assist_on) return;
  double most = 0;
  for (int i = 1; i < NF; i++) most = fmax(most, w->finger_force[i]);
  int holds = w->finger_force[0] >= w->assist_n && most >= w->assist_n &&
              grip_force(w) * w->m->geom_friction[3 * w->obj_geom] >= weight(w);
  if (holds && !w->d->eq_active[w->assist]) weld_here(w);
  else if (!holds && w->d->eq_active[w->assist]) w->d->eq_active[w->assist] = 0;
}

// Advances dt seconds towards the control layer's angles (degrees, JOINTS order) at effort 0 to 1
EXPORT void bb_step(World* w, double dt, const double* angles, double effort, int locked) {
  if (locked) {
    if (!w->was_locked) w->lock_effort = effort;
    w->was_locked = 1;
    effort = w->lock_effort;
  } else {
    w->was_locked = 0;
  }
  effort = clip(effort, 0, 1);
  mjModel* m = w->m;
  mjData* d = w->d;
  long n = lround(dt / m->opt.timestep);
  for (long s = 0; s < (n > 1 ? n : 1); s++) {
    for (int j = 0; j < NJ; j++) {
      double target = angles[j] * M_PI / 180, q = d->qpos[w->qadr[j]];
      double cap_full = w->max_force * w->arm[j], cap_effort = cap_full * effort;
      int pressing = w->finger_force[w->finger_of_joint[j]] > 0 && w->bends[j];
      d->ctrl[j] = pressing ? clip(w->kp_press * (target - q), -cap_full, cap_effort)
                            : clip(w->kp * (target - q), -cap_full, cap_full);
    }
    double hand_pos[3], hand_quat[4], before = d->time;
    bb_hand(w, hand_pos, hand_quat);
    mj_step(m, d);
    if (d->time < before) {
      // MuJoCo resets everything, its clock too, when it goes unstable: put the hand back and the object on the table
      w->blowups++;
      bb_set_hand(w, hand_pos, hand_quat, 0);
      memcpy(d->qpos, hand_pos, sizeof(hand_pos));
      memcpy(d->qpos + 3, hand_quat, sizeof(hand_quat));
      bb_reset_object(w);
    }
    read_forces(w);
    update_assist(w);
  }
  w->time += dt;
}

EXPORT int bb_ngeom(World* w) { return w->m->ngeom; }

// role: 0 table, 1 object, 2 hand. finger: as finger_of_geom. size: MuJoCo's geom size
EXPORT void bb_geom(World* w, int g, int* type, int* role, int* finger, double* size) {
  int body = w->m->geom_bodyid[g];
  *type = w->m->geom_type[g];
  *role = body == 0 ? 0 : body == w->obj_body ? 1 : 2;
  *finger = w->finger_of_geom[g];
  memcpy(size, w->m->geom_size + 3 * g, 3 * sizeof(double));
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

// grip force, 5 finger forces, palm force, weight, friction, assisted, object height, time, blow-ups
EXPORT void bb_readout(World* w, double* out) {
  out[0] = grip_force(w);
  for (int i = 0; i < NF; i++) out[1 + i] = w->finger_force[i];
  out[6] = w->palm_force;
  out[7] = weight(w);
  out[8] = w->m->geom_friction[3 * w->obj_geom];
  out[9] = w->d->eq_active[w->assist];
  out[10] = w->d->xpos[3 * w->obj_body + 2];
  out[11] = w->time;
  out[12] = w->blowups;
}
