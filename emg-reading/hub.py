"""Recording on other computers, training on the desktop.

The training desktop runs the app with --hub and is the hub. Other computers (devices) run the same app with their
own armband and camera, send their finished sessions to the hub, start training jobs on it from their Train tab and
download the models it saves (System > Sync). Everything goes over HTTP on the tailnet, every request to the hub
carries the device's token.

    python hub.py add-device mac        on the hub: makes a token for a device, shown once
    python hub.py remove-device mac     that token stops working straight away
    python hub.py devices

hub.json holds the hashes of the device tokens and the backup folder (on the hub) or the hub's address and this
device's token (on a device). It stays out of git.

The hub copies every ended session into its backup folder (a cloud drive's folder or a USB stick) and checks the copy.
A device deletes a session it sent once the hub and the backup both hold it byte for byte, so there are always two
copies off the device.
"""
import argparse
import hashlib
import hmac
import json
import re
import secrets
import shutil
import socket
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from session import DATA_DIR

HUB_FILE = Path(__file__).with_name("hub.json")
SESSION_FILES = ("session.json", "board.f64", "camera.f64", "events.jsonl")
SESSION_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[A-Za-z0-9_-]{1,32}$")
DEVICE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Files arrive in data/.incoming and are only moved into data/ once every file matches the sender's hash, so a half
# sent session never shows up as a session
INCOMING = ".incoming"
# In the backup folder: {session name: {file: sha256}} of every session copied and checked there
BACKUP_LIST = "backups.json"
# Per request. A dropped connection abroad then only costs the chunk it was on
CHUNK = 4 * 1024 * 1024
# The Train tab's requests, which a device may also send to the hub
TRAIN_PATHS = {"/api/sessions", "/api/models", "/api/networks", "/api/train", "/api/network/run",
               "/api/network/stop"}


def load():
    return json.loads(HUB_FILE.read_text()) if HUB_FILE.exists() else {}


def save(config):
    HUB_FILE.write_text(json.dumps(config, indent=2) + "\n")


def _digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def add_device(name):
    """A new token for the device, replacing any it had. Only its hash is kept, so it can't be shown again."""
    if not DEVICE_NAME.match(name):
        raise ValueError("device names are letters, digits, - and _")
    config = load()
    token = secrets.token_urlsafe(24)
    config.setdefault("devices", {})[name] = _digest(token)
    save(config)
    return token


def remove_device(name):
    config = load()
    if config.get("devices", {}).pop(name, None) is None:
        raise ValueError(f"no device called {name}")
    save(config)


def devices():
    return sorted(load().get("devices", {}))


def device_for(authorization):
    """The device an "Authorization: Bearer <token>" header belongs to, None for a missing or unknown token. Reads
    hub.json every time, so adding or removing a device works without restarting the app."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    digest = _digest(authorization[len("Bearer "):].strip())
    for name, known in load().get("devices", {}).items():
        if hmac.compare_digest(digest, known):
            return name
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def file_sizes(folder):
    return {name: (folder / name).stat().st_size for name in SESSION_FILES if (folder / name).is_file()}


def ended(folder):
    """True once the session has been closed. One that is still recording is never sent."""
    try:
        return "ended" in json.loads((folder / "session.json").read_text())
    except (OSError, ValueError):
        return False


def ended_sessions(data_dir=None):
    """{name: {file: size}} of the ended sessions in data/."""
    return {p.parent.name: file_sizes(p.parent) for p in sorted(Path(data_dir or DATA_DIR).glob("*/session.json"))
            if SESSION_NAME.match(p.parent.name) and ended(p.parent)}


def hashes(folder):
    return {file: sha256(folder / file) for file in file_sizes(folder)}


def remove_session(folder):
    """Deletes a session folder that holds only the session files. Anything else in it was never sent, so then the
    folder is left as it is."""
    extra = sorted(p.name for p in folder.iterdir() if p.name not in SESSION_FILES)
    if extra:
        raise ValueError(f"{folder.name} also holds {', '.join(extra)}, left as it is")
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()


def check_name(name):
    if not SESSION_NAME.match(name):
        raise ValueError("not a session name")


# The hub's side of receiving a session

def received(name):
    """Bytes of each file received so far of a session that is being sent, so a broken send carries on."""
    check_name(name)
    return file_sizes(DATA_DIR / INCOMING / name)


def receive(name, file, offset, data):
    """Appends one chunk. offset has to be where the file ends now, so a chunk sent twice can't go in twice."""
    check_name(name)
    if file not in SESSION_FILES:
        raise ValueError("not a session file")
    if (DATA_DIR / name).exists():
        raise ValueError(f"the hub already has a session called {name}")
    path = DATA_DIR / INCOMING / name / file
    path.parent.mkdir(parents=True, exist_ok=True)
    have = path.stat().st_size if path.exists() else 0
    if offset != have:
        raise ValueError(f"expected offset {have}")
    with open(path, "ab") as f:
        f.write(data)
    return have + len(data)


def discard(name):
    """Drops what was received of a session, for starting it again."""
    check_name(name)
    folder = DATA_DIR / INCOMING / name
    for path in folder.glob("*"):
        path.unlink()
    if folder.is_dir():
        folder.rmdir()


def finish(name, hashes):
    """Moves a received session into data/ once every file matches the sender's hash. A damaged file is dropped, so
    the next send starts that file again. Returns the session's folder."""
    check_name(name)
    folder = DATA_DIR / INCOMING / name
    target = DATA_DIR / name
    if target.exists():
        raise ValueError(f"the hub already has a session called {name}")
    if "session.json" not in hashes or set(hashes) - set(SESSION_FILES):
        raise ValueError("the session's file list is wrong")
    for file, digest in hashes.items():
        path = folder / file
        if not path.is_file():
            raise ValueError(f"{file} hasn't arrived")
        if sha256(path) != digest:
            path.unlink()
            raise ValueError(f"{file} of {name} arrived damaged, send again")
    if not ended(folder):
        raise ValueError(f"{name} hadn't ended")
    for path in folder.glob("*"):
        if path.name not in hashes:
            path.unlink()
    folder.rename(target)
    return target


# The hub's backup

def backup_folder():
    """The folder set as the backup, None when none is set. Refused when it is set but missing, for example a USB
    stick that isn't plugged in."""
    path = load().get("backup")
    if not path:
        return None
    if not Path(path).is_dir():
        raise ValueError(f"the backup folder {path} isn't there")
    return Path(path)


def backed_up(dest):
    """{session name: {file: sha256}} of the sessions in the backup, as checked when each was copied."""
    path = dest / BACKUP_LIST
    listing = json.loads(path.read_text()) if path.exists() else {}
    return {name: h for name, h in listing.items() if (dest / name / "session.json").exists()}


def back_up(folder, dest):
    """Copies one ended session into the backup through dest/.incoming, checks every file against the original and
    only then moves it in and lists it. A session already in the backup is never overwritten."""
    name = folder.name
    check_name(name)
    if not ended(folder):
        raise ValueError(f"{name} hasn't ended")
    original = hashes(folder)
    target = dest / name
    if not target.exists():
        part = dest / INCOMING / name
        part.mkdir(parents=True, exist_ok=True)
        for file, digest in original.items():
            shutil.copyfile(folder / file, part / file)
            if sha256(part / file) != digest:
                raise ValueError(f"{file} of {name} came out different in the backup, try again")
        part.rename(target)
    # Also covers a copy moved in just before the list was written last time
    elif hashes(target) != original:
        raise ValueError(f"the backup has a different {name}, left as it is")
    path = dest / BACKUP_LIST
    listing = json.loads(path.read_text()) if path.exists() else {}
    listing[name] = original
    path.write_text(json.dumps(listing, indent=1) + "\n")


def merge_named(path, theirs):
    """Adds the entries of another computer's mappings or profiles file that this one doesn't have. An entry with the
    same name and other content is left as it is here. Returns the names left alone."""
    mine = json.loads(path.read_text()) if path.exists() else {}
    kept = sorted(name for name in theirs if name in mine and mine[name] != theirs[name])
    new = {name: value for name, value in theirs.items() if name not in mine}
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(mine, **new), indent=2) + "\n")
    return kept


# A device's side

class HubError(ValueError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class Hub:
    """The hub as seen from a device."""

    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.token = token
        self._connection = None

    def request(self, method, path, body=None, params=None, data=None, timeout=30):
        """JSON back, or the bytes for a file. Raises HubError with the hub's message when it refuses."""
        path = path + (f"?{urlencode(params)}" if params else "")
        headers = {"Authorization": f"Bearer {self.token}"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        # The hub only listens on IPv4. A name that also resolves to IPv6 (localhost, tailnet names) cost 2 s per new
        # connection on Windows, 75 s for one session in 4 MB chunks: so IPv4 straight away, and one kept connection
        for attempt in (1, 2):
            if self._connection is None:
                parts = urlsplit(self.url)
                if parts.scheme == "https":
                    self._connection = HTTPSConnection(parts.netloc, timeout=timeout)
                else:
                    host = parts.hostname
                    try:
                        host = socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
                    except OSError:
                        pass
                    self._connection = HTTPConnection(host, parts.port or 80, timeout=timeout)
            try:
                if self._connection.sock is not None:
                    self._connection.sock.settimeout(timeout)
                self._connection.request(method, path, body=data, headers=headers)
                res = self._connection.getresponse()
                raw = res.read()
                break
            # A kept connection the hub has closed fails on first use, the second try opens a new one. A chunk that
            # did arrive is refused on the second try by its offset, never written twice
            except (OSError, HTTPException) as e:
                self._connection.close()
                self._connection = None
                if attempt == 2:
                    raise HubError(502, f"can't reach the hub at {self.url}: {e}") from None
        if res.status >= 400:
            try:
                detail = json.loads(raw).get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise HubError(res.status, detail if isinstance(detail, str) else f"the hub answered {res.status}")
        if res.getheader("content-type", "").startswith("application/octet-stream"):
            return raw
        return json.loads(raw) if raw else None

    def send_session(self, folder, progress):
        """Sends one ended session, carrying on from what the hub already has. progress(n) after every n bytes."""
        name = folder.name
        have = self.request("GET", f"/api/hub/incoming/{name}")
        sizes = file_sizes(folder)
        if any(have.get(file, 0) > size for file, size in sizes.items()):
            # Left from another copy of the session, start again
            self.request("DELETE", f"/api/hub/incoming/{name}")
            have = {}
        hashes = {}
        for file, size in sizes.items():
            offset = have.get(file, 0)
            progress(offset)
            first = file not in have
            with open(folder / file, "rb") as f:
                f.seek(offset)
                # An empty file still has to arrive once
                while first or offset < size:
                    first = False
                    chunk = f.read(CHUNK)
                    offset = self.request("PUT", f"/api/hub/incoming/{name}/{file}", params={"offset": offset},
                                          data=chunk, timeout=120)["size"]
                    progress(len(chunk))
            hashes[file] = sha256(folder / file)
        self.request("POST", f"/api/hub/incoming/{name}/done", body=hashes)

    def download(self, params, dest, size, digest, progress):
        """One file from the hub into dest, through a .part file that a broken download carries on from."""
        part = dest.with_name(dest.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            part.unlink()
            offset = 0
        progress(offset)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "ab") as f:
            while offset < size:
                chunk = self.request("GET", "/api/hub/file", params=dict(params, offset=offset, length=CHUNK),
                                     timeout=120)
                if not chunk:
                    raise ValueError(f"{dest.name} stopped short")
                f.write(chunk)
                offset += len(chunk)
                progress(len(chunk))
        if sha256(part) != digest:
            part.unlink()
            raise ValueError(f"{dest.name} arrived damaged, try again")
        part.replace(dest)

    def download_session(self, name, data_dir, progress):
        """A session from the hub, moved into data/ only once every file is in and checked."""
        hashes = self.request("GET", f"/api/hub/sessions/{quote(name)}/hashes")
        sizes = self.request("GET", "/api/hub/sessions")[name]
        folder = Path(data_dir) / INCOMING / name
        for file, digest in hashes.items():
            self.download({"kind": "session", "name": name, "file": file}, folder / file, sizes[file], digest,
                          progress)
        folder.rename(Path(data_dir) / name)


def client():
    config = load()
    if not config.get("url") or not config.get("token"):
        raise ValueError("set the hub's address and this device's token on System > Sync first")
    return Hub(config["url"], config["token"])


def main():
    parser = argparse.ArgumentParser(description="Devices allowed to use this computer as their hub")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add-device").add_argument("name")
    commands.add_parser("remove-device").add_argument("name")
    commands.add_parser("devices")
    args = parser.parse_args()
    try:
        if args.command == "add-device":
            token = add_device(args.name)
            print(f"Token for {args.name}, paste it on that device's System > Sync. It isn't stored and can't be "
                  f"shown again:\n\n    {token}\n")
        elif args.command == "remove-device":
            remove_device(args.name)
            print(f"{args.name} removed")
        else:
            print("\n".join(devices()) or "no devices yet, add one with add-device")
    except ValueError as e:
        raise SystemExit(e)


if __name__ == "__main__":
    main()
