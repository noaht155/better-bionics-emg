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
