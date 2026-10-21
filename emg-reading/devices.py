"""The recording setups (computer + camera) in devices.json, committed so every computer and the hub agree on them.

Each setup has the label shift measured for its camera once (System > Dev, delay test), so it isn't measured again
for every session. A session stores the id of the setup it was recorded on; training looks the shift up here, so a
setup measured later fixes its older sessions too. Sessions from before setups existed are the desktop's.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

DEVICE_FILE = Path(__file__).with_name("devices.json")


def _load():
    return json.loads(DEVICE_FILE.read_text())


def setups():
    return _load()["setups"]


def find(setup_id):
    for s in setups():
        if s["id"] == setup_id:
            return s
    return None


def of_session(meta):
    """The setup a session was recorded on, the default one for sessions from before setups."""
    stored = meta.get("setup") or {}
    return find(stored.get("id") or _load()["default"])


def computer_name():
    """A name that stays the same on every network. A Mac's hostname comes from the network it is on
    (Noahs-MBP.cogeco.local at home), its local name doesn't."""
    if sys.platform == "darwin":
        name = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True).stdout.strip()
        if name:
            return name
    return socket.gethostname()


def here(setup):
    return setup["computer"].lower() == computer_name().lower()


def cameras():
    """Names of the cameras plugged in, on a Mac only; Windows has no simple way to list them."""
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True).stdout
    return [line.strip().rstrip(":") for line in out.splitlines() if line.startswith("    ") and
            not line.startswith("      ") and line.strip().endswith(":")]


def save_shift(setup_id, shift_s, note):
    """Writes a measured label shift into devices.json, to be committed with the measurement's devlog entry."""
    data = _load()
    for s in data["setups"]:
        if s["id"] == setup_id:
            s.update(label_shift_s=round(shift_s, 4), measured=time.strftime("%Y-%m-%d"), note=note)
            DEVICE_FILE.write_text(json.dumps(data, indent=2) + "\n")
            return s
    raise ValueError(f"no setup {setup_id}")
