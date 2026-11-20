// Closed-loop can grasp on the Quest through bbsim.c (the app's sim and scene), fed the per-tick hand pose, angles
// and effort that record.py saved, with the free hand parked out of the way. Prints the speed and where the can ends
// up, to check against the Python run.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NJ 20

typedef struct World World;
World* bb_create(const char* xml, int len, char* err, int errlen);
void bb_free(World* w);
void bb_set_params(World* w, double mass_scale, double max_force, int assist);
void bb_set_input(World* w, int h, const double* angles, double effort, int locked, int grab);
void bb_set_hand(World* w, int h, const double* pos, const double* quat, int teleport);
void bb_place_object(World* w, int o, const double* pos, const double* quat, int parked);
void bb_object_pose(World* w, int o, double* pos, double* quat);
void bb_step(World* w, double dt);
void bb_readout(World* w, double* out);

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static char* slurp(const char* path, long* len) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *len = ftell(f);
  rewind(f);
  char* buf = malloc(*len);
  fread(buf, 1, *len, f);
  fclose(f);
  return buf;
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 3;
  double max_force, half;
  int can;
  FILE* r = fopen("rule.txt", "r");
  if (!r) { printf("no rule.txt\n"); return 1; }
  fscanf(r, "%lf %d %lf", &max_force, &can, &half);
  fclose(r);
  long xml_len, ticks_len;
  char* xml = slurp("scene.xml", &xml_len);
  double* ticks = (double*)slurp("ticks.bin", &ticks_len);
  if (!xml || !ticks) { printf("no scene.xml or ticks.bin\n"); return 1; }
  int cols = 2 + 7 + NJ, nticks = ticks_len / sizeof(double) / cols;
  double rest[NJ] = {0}, start[3] = {0, -0.12, half + 0.001}, upright[4] = {1, 0, 0, 0};
  double away[3] = {-0.45, 0.3, 0.45}, down[4] = {0, 0, 0.7071068, 0.7071068};

  for (int rep = 0; rep < reps; rep++) {
    char err[1000];
    World* w = bb_create(xml, xml_len, err, sizeof(err));
    if (!w) { printf("load failed: %s\n", err); return 1; }
    bb_set_params(w, 1.0, max_force, 0);
    bb_place_object(w, can, start, upright, 0);
    bb_set_hand(w, 1, away, down, 1);
    bb_set_input(w, 1, rest, 0, 0, 0);
    bb_set_hand(w, 0, ticks + 2, ticks + 5, 1);
    double t0 = now(), out[17], pos[3], quat[4];
    for (int t = 0; t < nticks; t++) {
      double* row = ticks + t * cols;
      bb_set_hand(w, 0, row + 2, row + 5, 0);
      bb_set_input(w, 0, row + 9, row[1], 0, 0);
      bb_step(w, row[0]);
    }
    double el = now() - t0;
    bb_readout(w, out);
    bb_object_pose(w, can, pos, quat);
    printf("run %d: %.1fx real time, object z %.4f m, blow-ups %d\n", rep + 1, out[12] / el, pos[2], (int)out[13]);
    bb_free(w);
  }
  return 0;
}
