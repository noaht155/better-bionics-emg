"""Records the can wrap grasp at full effort for bench.c: the per-tick hand pose, control angles and effort, the
force-rule constants and the model. Run from the repo root:
    emg-reading/.venv/Scripts/python quest-app/native/bench/record.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import ml  # noqa: E402,F401  puts emg-reading on the path
import numpy as np  # noqa: E402
import sim  # noqa: E402
from test_sim import SIDE, WRAP, grasp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "build" / "bench"

ticks, worlds = [], []
step = sim.World.step


def record(self, dt, angles, effort, locked=False):
    if not worlds or worlds[-1] is not self:
        worlds.append(self)
        ticks.clear()
    pos, quat = self.hand_pose()
    ticks.append(np.concatenate([[dt, effort], pos, quat, angles]))
    step(self, dt, angles, effort, locked)


sim.World.step = record
lift = grasp("can", WRAP, SIDE, sim.FINGERS, True, 1.0, (-0.01, 0, 0))
w = worlds[-1]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "can.xml").write_text(sim.model_xml("can"))
np.asarray(ticks, np.float64).tofile(OUT / "ticks.bin")
geom = np.full(w.model.ngeom, -2)
for g, finger in w._finger_of_geom.items():
    geom[g] = finger
lines = [f"{sim.KP} {sim.KP_PRESS} {sim.TIMESTEP} {w.params['max_force']} {w._object_geom} {w.model.ngeom} {len(ticks)}",
         " ".join(map(str, w._qadr)), " ".join(map(str, w._finger_of_joint)),
         " ".join(str(int(b)) for b in w._bends), " ".join(repr(float(a)) for a in w._arm), " ".join(map(str, geom))]
(OUT / "rule.txt").write_text("\n".join(lines) + "\n")
print(f"python: lifted {lift:.3f} m, object z {w.readout()['object_height']:.4f} m, {len(ticks)} ticks")
