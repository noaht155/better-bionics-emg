"""Models trained on the recorded sessions: gesture classification now, the general network later.

The recording side (filter, session files, gesture list) lives in emg-reading/ as plain modules, so they are put on
the import path here. Run scripts from the repo root with the emg-reading venv, for example
    emg-reading/.venv/Scripts/python -m ml.gesture_model emg-reading/data/<session> ...
"""
import sys
from pathlib import Path

EMG_READING = Path(__file__).resolve().parent.parent / "emg-reading"
if str(EMG_READING) not in sys.path:
    sys.path.insert(0, str(EMG_READING))


def low_priority():
    """Run the calling script below normal priority, so training never takes CPU time from a recording (camera
    tracking runs on the CPU). Only on Windows, elsewhere it does nothing."""
    if sys.platform == "win32":
        import ctypes
        below_normal = 0x4000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
