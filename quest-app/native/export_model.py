"""Writes the Quest app's sim scene (sim.scene_xml, the PC runs the same one): one XML per prosthetic side
(scene_left.xml: the prosthetic hand is the left), the ghost hands, and scene.json with what the app needs to know
about it.
    emg-reading/.venv/Scripts/python quest-app/native/export_model.py [output folder]
Default output: quest-app/Assets/Resources/sim (ignored, the build script runs this).
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


def meta():
    return dict(sim.SIM_DEFAULTS, objects=list(sim.OBJECTS), pool=sim.pool(),
                half=[v[3] for v in sim.OBJECTS.values()], joints=list(JOINTS),
                rest=[float(a) for a in POSES["rest"]], fist=[float(a) for a in POSES["fist"]],
                point=[float(a) for a in POSES["point"]], desk_half=list(sim.DESK_HALF),
                desk_centre_y=sim.DESK_CENTRE_Y, desk_height=sim.DESK_HEIGHT,
                park=[sim.PARK_X, sim.PARK_Y, sim.PARK_STEP])


def main(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        old.unlink()
    (out / "scene_left.xml").write_text(sim.scene_xml(True))
    (out / "scene_right.xml").write_text(sim.scene_xml(False))
    (out / "ghost_left.xml").write_text(sim.ghost_xml(True))
    (out / "ghost_right.xml").write_text(sim.ghost_xml(False))
    (out / "scene.json").write_text(json.dumps(meta()))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
