"""USB link to the Quest app (quest-app/): keeps `adb reverse` on a Quest plugged in, so the headset's localhost:8000
reaches this app's /ws/quest. USB rather than WiFi so the laptop stays on the armband's network and the Quest never
joins it. adb is the one on PATH or Unity's (it comes with Unity's Android support); without one nothing runs.

Another copy of the app (a test copy on another port) only takes the mapping over when the port it points to no
longer answers, so two copies don't keep taking it from each other.

It also reads the headset's battery and temperature for the VR tab, and closes the Quest app when this app stops,
if this copy is the one the headset is linked to: left running, it drains the battery.
"""
import glob
import shutil
import socket
import subprocess
import threading
import time

# The port the Quest app connects to on the headset (Link.cs)
QUEST_PORT = 8000
POLL_S = 2.0
BATTERY_S = 30.0
# Warned about while linked: low and not charging, or hot enough that the Quest starts slowing itself down
LOW_BATTERY = 15
HOT_C = 45
# QuestBuild.Package
QUEST_APP = "com.betterbionics.contact"
# The running link, for the VR tab's battery readout
LINK = None
UNITY_ADB = ["C:/Program Files/Unity/Hub/Editor/*/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe",
             "/Applications/Unity/Hub/Editor/*/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb"]


def find_adb():
    found = shutil.which("adb")
    if found:
        return found
    for pattern in UNITY_ADB:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def _answers(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


class Reverse(threading.Thread):
    def __init__(self, port, adb):
        super().__init__(daemon=True, name="quest-link")
        self.port, self.adb = port, adb
        self._stop = threading.Event()
        self._told = set()
        self._linked = set()
        self._battery_at = {}
        # The linked headset's battery: {"level": %, "temp_c": degrees, "charging": bool}, None before it's read
        self.headset = None

    def _run(self, *args):
        return subprocess.run([self.adb, *args], capture_output=True, text=True, timeout=10)

    def _tell(self, key, text):
        if key not in self._told:
            self._told.add(key)
            print(text)

    def check(self):
        devices = self._run("devices").stdout.splitlines()[1:]
        for line in devices:
            parts = line.split()
            if len(parts) < 2:
                continue
            serial, state = parts[0], parts[1]
            if state == "unauthorized":
                self._tell(("auth", serial), f"Quest {serial} on USB: allow USB debugging in the headset")
                continue
            if state != "device":
                continue
            mapped = None
            for entry in self._run("-s", serial, "reverse", "--list").stdout.splitlines():
                fields = entry.split()
                if len(fields) >= 3 and fields[1] == f"tcp:{QUEST_PORT}":
                    mapped = int(fields[2].removeprefix("tcp:"))
            if mapped == self.port:
                self._linked.add(serial)
                self._read_battery(serial)
                continue
            self._linked.discard(serial)
            if mapped is not None and _answers(mapped):
                continue
            if self._run("-s", serial, "reverse", f"tcp:{QUEST_PORT}", f"tcp:{self.port}").returncode == 0:
                self._told.discard(("auth", serial))
                self._linked.add(serial)
                print(f"Quest {serial} linked over USB")
        if not self._linked:
            self.headset = None

    def _read_battery(self, serial):
        now = time.monotonic()
        if now - self._battery_at.get(serial, -BATTERY_S) < BATTERY_S:
            return
        self._battery_at[serial] = now
        found = parse_battery(self._run("-s", serial, "shell", "dumpsys", "battery").stdout)
        if found is not None:
            self.headset = found

    def warning(self):
        """(code, text) for the warnings bar while the linked headset's battery is low or it runs hot, else None."""
        h = self.headset
        if h is None:
            return None
        if h["level"] <= LOW_BATTERY and not h["charging"]:
            return ("quest_battery", f"Quest battery at {h['level']} %, it will close soon: charge it or plug it into "
                                     "a charger")
        if h["temp_c"] >= HOT_C:
            return ("quest_hot", f"Quest battery at {h['temp_c']:.0f} °C, let it cool before going on")
        return None

    def close_app(self):
        """Closes the Quest app on the headsets linked to this copy."""
        for serial in list(self._linked):
            try:
                self._run("-s", serial, "shell", "am", "force-stop", QUEST_APP)
                print(f"closed the Quest app on {serial}")
            except (OSError, subprocess.SubprocessError):
                pass

    def run(self):
        while not self._stop.wait(POLL_S):
            try:
                self.check()
            except (OSError, subprocess.SubprocessError):
                pass

    def stop(self):
        self._stop.set()
        self.close_app()


def parse_battery(text):
    """{"level", "temp_c", "charging"} from `dumpsys battery`, None when it has no level."""
    fields = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition(":")
        fields[key.strip()] = value.strip()
    try:
        level = int(fields["level"])
        temp_c = int(fields.get("temperature", "0")) / 10
    except (KeyError, ValueError):
        return None
    # BatteryManager status 2 is charging, 5 is full
    return {"level": level, "temp_c": temp_c, "charging": fields.get("status") in ("2", "5")}


def start(port):
    """Starts keeping a plugged-in Quest linked to this app on port, or returns None without adb."""
    adb = find_adb()
    if adb is None:
        return None
    global LINK
    link = LINK = Reverse(port, adb)
    link.start()
    return link
