// Closed-loop can grasp on the Quest through bbsim.c (the app's sim), fed the per-tick hand pose, angles and effort
// that record.py saved. Prints the speed and where the can ends up, to check against the Python run.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NJ 20

typedef struct World World;
World* bb_create(const char* xml, int len, char* err, int errlen);
void bb_free(World* w);
void bb_set_params(World* w, double mass_scale, double max_force, int assist);
void bb_set_hand(World* w, const double* pos, const double* quat, int teleport);
void bb_step(World* w, double dt, const double* angles, double effort, int locked);
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
  double max_force;
  FILE* r = fopen("rule.txt", "r");
  if (!r) { printf("no rule.txt\n"); return 1; }
  fscanf(r, "%lf", &max_force);
  fclose(r);
  long xml_len, ticks_len;
  char* xml = slurp("can.xml", &xml_len);
  double* ticks = (double*)slurp("ticks.bin", &ticks_len);
  if (!xml || !ticks) { printf("no can.xml or ticks.bin\n"); return 1; }
  int cols = 2 + 7 + NJ, nticks = ticks_len / sizeof(double) / cols;

  for (int rep = 0; rep < reps; rep++) {
    char err[1000];
    World* w = bb_create(xml, xml_len, err, sizeof(err));
    if (!w) { printf("load failed: %s\n", err); return 1; }
    bb_set_params(w, 1.0, max_force, 0);
    bb_set_hand(w, ticks + 2, ticks + 5, 1);
    double out[13], t0 = now();
    for (int t = 0; t < nticks; t++) {
      double* row = ticks + t * cols;
      bb_set_hand(w, row + 2, row + 5, 0);
      bb_step(w, row[0], row + 9, row[1], 0);
    }
    double el = now() - t0;
    bb_readout(w, out);
    printf("run %d: %.1fx real time, object z %.4f m, blow-ups %d\n", rep + 1, out[11] / el, out[10], (int)out[12]);
    bb_free(w);
  }
  return 0;
}
