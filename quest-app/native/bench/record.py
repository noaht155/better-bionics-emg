"""Records the can wrap grasp at full effort for bench.c: the per-tick hand pose, control angles and effort, the
hand strength, and the app's scene with the right hand as the prosthetic one. Run from the repo root:
    emg-reading/.venv/Scripts/python quest-app/native/bench/record.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "quest-app" / "native"))
import ml  # noqa: E402,F401  puts emg-reading on the path
import numpy as np  # noqa: E402
import sim  # noqa: E402
from export_model import pool, scene_xml  # noqa: E402
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
(OUT / "scene.xml").write_text(scene_xml(prosthetic_left=False))
np.asarray(ticks, np.float64).tofile(OUT / "ticks.bin")
(OUT / "rule.txt").write_text(f"{w.params['max_force']} {pool().index('can')} {sim.OBJECTS['can'][3]}\n")
print(f"python: lifted {lift:.3f} m, object z {w.readout()['object_height']:.4f} m, {len(ticks)} ticks")
