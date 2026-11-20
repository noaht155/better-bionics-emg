"""Writes sim.py's model for the Quest app: one XML per object with the force rule's constants added as custom
numerics (read by bbsim.c), and defaults.json with the object list, sim.SIM_DEFAULTS and the rest pose.
    emg-reading/.venv/Scripts/python quest-app/native/export_model.py [output folder]
Default output: quest-app/Assets/Resources/sim (ignored, the build script runs this).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "emg-reading"))
import sim  # noqa: E402
from tools import POSES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "Assets" / "Resources" / "sim"


def quest_xml(obj):
    """sim.model_xml with the constants bbsim.c needs that the XML doesn't already hold."""
    values = {"kp": sim.KP, "kp_press": sim.KP_PRESS, "assist_n": sim.ASSIST_N,
              "max_mass_scale": sim.MAX_MASS_SCALE, "gravity": sim.GRAVITY, "object_half": sim.OBJECTS[obj][3]}
    custom = "".join(f'<numeric name="{k}" data="{v!r}"/>' for k, v in values.items())
    custom += f'<numeric name="arm" data="{" ".join(repr(float(a)) for a in sim._arms())}"/>'
    return sim.model_xml(obj).replace("</mujoco>", f"<custom>{custom}</custom></mujoco>")


def main(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for obj in sim.OBJECTS:
        (out / f"{obj}.xml").write_text(quest_xml(obj))
    defaults = dict(sim.SIM_DEFAULTS, objects=list(sim.OBJECTS), rest=[float(a) for a in POSES["rest"]])
    (out / "defaults.json").write_text(json.dumps(defaults))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
