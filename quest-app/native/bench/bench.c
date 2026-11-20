// Closed-loop can grasp on the Quest: sim.py's per-finger force rule in C, fed the per-tick hand pose, angles and
// effort that record.py saved. Prints the speed and where the can ends up, to check against the Python run.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mujoco/mujoco.h>

#define NJ 20

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static double clip(double x, double lo, double hi) { return x < lo ? lo : x > hi ? hi : x; }

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 3;
  double kp, kp_press, timestep, max_force;
  int obj_geom, ngeom, nticks, qadr[NJ], fjoint[NJ], bends[NJ];
  double arm[NJ];
  FILE* r = fopen("rule.txt", "r");
  if (!r) { printf("no rule.txt\n"); return 1; }
  fscanf(r, "%lf %lf %lf %lf %d %d %d", &kp, &kp_press, &timestep, &max_force, &obj_geom, &ngeom, &nticks);
  for (int j = 0; j < NJ; j++) fscanf(r, "%d", &qadr[j]);
  for (int j = 0; j < NJ; j++) fscanf(r, "%d", &fjoint[j]);
  for (int j = 0; j < NJ; j++) fscanf(r, "%d", &bends[j]);
  for (int j = 0; j < NJ; j++) fscanf(r, "%lf", &arm[j]);
  int* fgeom = malloc(sizeof(int) * ngeom);
  for (int g = 0; g < ngeom; g++) fscanf(r, "%d", &fgeom[g]);
  fclose(r);

  int cols = 2 + 7 + NJ;
  double* ticks = malloc(sizeof(double) * cols * nticks);
  FILE* f = fopen("ticks.bin", "rb");
  if (!f) { printf("no ticks.bin\n"); return 1; }
  fread(ticks, sizeof(double), cols * nticks, f);
  fclose(f);

  char err[1000];
  mjModel* m = mj_loadXML("can.xml", NULL, err, sizeof(err));
  if (!m) { printf("load failed: %s\n", err); return 1; }
  int obj = mj_name2id(m, mjOBJ_BODY, "object");

  for (int rep = 0; rep < reps; rep++) {
    mjData* d = mj_makeData(m);
    for (int k = 0; k < 7; k++) d->qpos[k] = ticks[2 + k];
    for (int j = 0; j < NJ; j++) d->qpos[qadr[j]] = 0;
    mj_forward(m, d);
    double finger_force[5] = {0}, f6[6], steps = 0, t0 = now();
    for (int t = 0; t < nticks; t++) {
      double* row = ticks + t * cols;
      double dt = row[0], effort = clip(row[1], 0, 1);
      for (int k = 0; k < 3; k++) d->mocap_pos[k] = row[2 + k];
      double qn = sqrt(row[5] * row[5] + row[6] * row[6] + row[7] * row[7] + row[8] * row[8]);
      for (int k = 0; k < 4; k++) d->mocap_quat[k] = row[5 + k] / qn;
      int n = (int)lround(dt / timestep);
      for (int s = 0; s < (n > 1 ? n : 1); s++) {
        for (int j = 0; j < NJ; j++) {
          double target = row[9 + j] * M_PI / 180, q = d->qpos[qadr[j]];
          double cap_full = max_force * arm[j], cap_effort = cap_full * effort;
          int pressing = finger_force[fjoint[j]] > 0 && bends[j];
          d->ctrl[j] = pressing ? clip(kp_press * (target - q), -cap_full, cap_effort)
                                : clip(kp * (target - q), -cap_full, cap_full);
        }
        mj_step(m, d);
        steps++;
        for (int i = 0; i < 5; i++) finger_force[i] = 0;
        for (int i = 0; i < d->ncon; i++) {
          mjContact* c = d->contact + i;
          if (c->geom[0] != obj_geom && c->geom[1] != obj_geom) continue;
          int other = c->geom[0] == obj_geom ? c->geom[1] : c->geom[0];
          int fi = fgeom[other];
          if (fi < 0) continue;  // palm or not the hand
          mj_contactForce(m, d, i, f6);
          finger_force[fi] += f6[0];
        }
      }
    }
    double el = now() - t0;
    printf("run %d: %.1fx real time, object z %.4f m\n", rep + 1, steps * timestep / el, d->xpos[3 * obj + 2]);
    mj_deleteData(d);
  }
  return 0;
}
