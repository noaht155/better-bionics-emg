"""USB link to the Quest app (quest-app/): keeps `adb reverse` on a Quest plugged in, so the headset's localhost:8000
reaches this app's /ws/quest. USB rather than WiFi so the laptop stays on the armband's network and the Quest never
joins it. adb is the one on PATH or Unity's (it comes with Unity's Android support); without one nothing runs.

Another copy of the app (a test copy on another port) only takes the mapping over when the port it points to no
longer answers, so two copies don't keep taking it from each other.
"""
import glob
import shutil
import socket
import subprocess
import threading

# The port the Quest app connects to on the headset (Link.cs)
QUEST_PORT = 8000
POLL_S = 2.0
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
            if mapped == self.port or (mapped is not None and _answers(mapped)):
                continue
            if self._run("-s", serial, "reverse", f"tcp:{QUEST_PORT}", f"tcp:{self.port}").returncode == 0:
                self._told.discard(("auth", serial))
                print(f"Quest {serial} linked over USB")

    def run(self):
        while not self._stop.wait(POLL_S):
            try:
                self.check()
            except (OSError, subprocess.SubprocessError):
                pass

    def stop(self):
        self._stop.set()


def start(port):
    """Starts keeping a plugged-in Quest linked to this app on port, or returns None without adb."""
    adb = find_adb()
    if adb is None:
        return None
    link = Reverse(port, adb)
    link.start()
    return link
