"""Sessions sent to the hub arrive whole or not at all, and only devices with a token get past the hub's door."""
import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import uvicorn

import ml  # noqa: F401
import app
import hub
import tools
from ml import gesture_model

NAME = "2026-10-20_120000_noah"
# A computer on the tailnet, not this one
REMOTE = ("100.64.0.2", 50000)


class FakeBackup:
    def request(self):
        pass


class FakeJob:
    def __init__(self):
        self.started = []

    def start(self, job, args):
        self.started.append((job, args))


@pytest.fixture
def hub_dir(tmp_path, monkeypatch):
    """The hub's data/ and hub.json in a temporary folder, with one device."""
    data = tmp_path / "hub-data"
    data.mkdir()
    monkeypatch.setattr(hub, "DATA_DIR", data)
    monkeypatch.setattr(app, "DATA_DIR", data)
    monkeypatch.setattr(hub, "HUB_FILE", tmp_path / "hub.json")
    monkeypatch.setattr(hub, "CHUNK", 1000)
    return data


def make_session(folder, ended=True):
    folder.mkdir(parents=True)
    meta = {"subject": "noah", "settings": {}, "plan": []}
    if ended:
        meta["ended"] = 1.0
    (folder / "session.json").write_text(json.dumps(meta))
    (folder / "board.f64").write_bytes(np.random.default_rng(0).random(800).tobytes())
    (folder / "camera.f64").write_bytes(b"")
    (folder / "events.jsonl").write_text('{"kind": "cue"}\n')
    return folder


def hub_app(lan=False):
    jobs = SimpleNamespace(network=FakeJob(), backup=FakeBackup(), sync_watch=FakeBackup(), snapshot=lambda: {"trainer": {}, "network": {}})
    return app.make_app(SimpleNamespace(on_saved=None), None, None, jobs, lan=lan, serving=True)


@pytest.fixture
def server(hub_dir):
    """The hub app on a free port of this computer, for hub.Hub to talk to over real HTTP."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(hub_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    while not running.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    running.should_exit = True
    thread.join(5)


class DroppingHub(hub.Hub):
    """A connection that drops after a number of chunks."""

    def __init__(self, url, token, chunks):
        super().__init__(url, token)
        self.chunks = chunks

    def request(self, method, path, **kwargs):
        if method == "PUT":
            if self.chunks == 0:
                raise hub.HubError(502, "connection dropped")
            self.chunks -= 1
        return super().request(method, path, **kwargs)


def ask(client, path, authorization=None, kind="http"):
    """The status Access answers with for a request from client, 200 if it lets it through."""
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    headers = [(b"authorization", authorization.encode())] if authorization else []
    scope = {"type": kind, "path": path, "client": client, "headers": headers}
    asyncio.run(app.Access(inner, lan=ask.lan)(scope, receive, send))
    return "closed" if sent[0]["type"] == "websocket.close" else sent[0]["status"]


def test_tokens(hub_dir):
    token = hub.add_device("mac")
    assert hub.device_for(f"Bearer {token}") == "mac"
    assert hub.device_for("Bearer wrong") is None
    assert hub.device_for(None) is None
    assert token not in hub.HUB_FILE.read_text()
    hub.remove_device("mac")
    assert hub.device_for(f"Bearer {token}") is None


def test_only_devices_and_this_computer_get_in(hub_dir):
    device = f"Bearer {hub.add_device('mac')}"
    ask.lan = False
    local = ("127.0.0.1", 50000)
    assert ask(REMOTE, "/api/sessions") == 403
    assert ask(REMOTE, "/api/sessions", "Bearer wrong") == 403
    assert ask(REMOTE, "/api/sessions", device) == 200
    assert ask(REMOTE, "/api/hub/hello", device) == 200
    # A device can train on the hub, not drive its armband, haptics or camera
    assert ask(REMOTE, "/api/model/use", device) == 403
    assert ask(REMOTE, "/api/haptics/start", device) == 403
    assert ask(REMOTE, "/ws", device, kind="websocket") == "closed"
    assert ask(REMOTE, "/ws", kind="websocket") == "closed"
    # The hub's own requests want a token even from this computer, everything else is open to it
    assert ask(local, "/api/hub/sessions") == 401
    assert ask(local, "/api/sessions") == 200
    assert ask(local, "/ws", kind="websocket") == 200


def test_tablet_mode_still_open(hub_dir):
    ask.lan = True
    assert ask(REMOTE, "/api/sessions") == 200
    assert ask(REMOTE, "/ws", kind="websocket") == 200
    assert ask(REMOTE, "/api/hub/sessions") == 401


def test_send_carries_on_after_a_dropped_connection(hub_dir, server, tmp_path):
    token = hub.add_device("mac")
    folder = make_session(tmp_path / "device-data" / NAME)
    with pytest.raises(hub.HubError, match="dropped"):
        DroppingHub(server, token, chunks=3).send_session(folder, lambda n: None)
    # Half sent: not a session on the hub yet
    assert not (hub_dir / NAME).exists()
    assert hub.received(NAME)
    client = hub.Hub(server, token)
    sent = []
    client.send_session(folder, sent.append)
    for file in hub.SESSION_FILES:
        assert (hub_dir / NAME / file).read_bytes() == (folder / file).read_bytes()
    assert not (hub_dir / hub.INCOMING / NAME).exists()
    assert sum(sent) == sum(hub.file_sizes(folder).values())
    assert NAME in client.request("GET", "/api/hub/sessions")
    # Never overwritten
    with pytest.raises(hub.HubError, match="already has"):
        client.send_session(folder, lambda n: None)


def test_unknown_token_refused(hub_dir, server):
    with pytest.raises(hub.HubError, match="unknown device"):
        hub.Hub(server, "wrong").request("GET", "/api/hub/hello")


def test_damaged_file_is_dropped(hub_dir, tmp_path):
    folder = make_session(tmp_path / "device-data" / NAME)
    for file in hub.SESSION_FILES:
        hub.receive(NAME, file, 0, (folder / file).read_bytes())
    hashes = {file: hub.sha256(folder / file) for file in hub.SESSION_FILES}
    hashes["board.f64"] = "0" * 64
    with pytest.raises(ValueError, match="damaged"):
        hub.finish(NAME, hashes)
    assert not (hub_dir / NAME).exists()
    assert "board.f64" not in hub.received(NAME)


def test_wrong_offset_and_names_refused(hub_dir):
    hub.receive(NAME, "board.f64", 0, b"abc")
    with pytest.raises(ValueError, match="offset"):
        hub.receive(NAME, "board.f64", 0, b"abc")
    with pytest.raises(ValueError):
        hub.receive("../escape", "board.f64", 0, b"")
    with pytest.raises(ValueError):
        hub.receive(NAME, "../../app.py", 0, b"")


def test_session_still_recording_is_refused(hub_dir, tmp_path):
    folder = make_session(tmp_path / "device-data" / NAME, ended=False)
    for file in hub.SESSION_FILES:
        hub.receive(NAME, file, 0, (folder / file).read_bytes())
    with pytest.raises(ValueError, match="ended"):
        hub.finish(NAME, {file: hub.sha256(folder / file) for file in hub.SESSION_FILES})
    assert hub.ended_sessions() == {}


def test_download_session(hub_dir, server, tmp_path):
    client = hub.Hub(server, hub.add_device("mac"))
    source = make_session(hub_dir / NAME)
    device_data = tmp_path / "device-data"
    client.download_session(NAME, device_data, lambda n: None)
    for file in hub.SESSION_FILES:
        assert (device_data / NAME / file).read_bytes() == (source / file).read_bytes()


def test_merge_keeps_both_sides(tmp_path):
    path = tmp_path / "mappings.json"
    path.write_text(json.dumps({"mine": {"rules": 1}, "both": {"rules": 1}}))
    kept = hub.merge_named(path, {"theirs": {"rules": 2}, "both": {"rules": 2}})
    assert kept == ["both"]
    assert json.loads(path.read_text()) == {"mine": {"rules": 1}, "both": {"rules": 1}, "theirs": {"rules": 2}}


def test_back_up_copies_and_checks(hub_dir, tmp_path):
    dest = tmp_path / "backup"
    dest.mkdir()
    folder = make_session(hub_dir / NAME)
    hub.back_up(folder, dest)
    for file in hub.SESSION_FILES:
        assert (dest / NAME / file).read_bytes() == (folder / file).read_bytes()
    assert hub.backed_up(dest) == {NAME: hub.hashes(folder)}
    assert not (dest / hub.INCOMING / NAME).exists()
    # Again is fine, a different copy of the same name is left alone
    hub.back_up(folder, dest)
    (dest / NAME / "events.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="different"):
        hub.back_up(folder, dest)
    assert (dest / NAME / "events.jsonl").read_text() == "changed"
    with pytest.raises(ValueError, match="ended"):
        hub.back_up(make_session(hub_dir / "2026-10-20_130000_noah", ended=False), dest)


def test_backup_worker_and_missing_folder(hub_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DATA_DIR", hub_dir)
    make_session(hub_dir / NAME)
    make_session(hub_dir / "2026-10-20_130000_noah", ended=False)
    worker = tools.Backup()
    # No folder set: nothing to do
    worker._run()
    hub.save({"backup": str(tmp_path / "unplugged")})
    with pytest.raises(ValueError, match="isn't there"):
        worker._run()
    dest = tmp_path / "backup"
    dest.mkdir()
    hub.save({"backup": str(dest)})
    worker._run()
    assert list(hub.backed_up(dest)) == [NAME]


def test_device_deletes_on_close_only_what_the_hub_and_backup_hold(hub_dir, server, tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(gesture_model, "MODEL_DIR", models)
    monkeypatch.setattr(app, "NETWORK_DIR", models)
    monkeypatch.setattr(tools, "NETWORK_DIR", models)
    device = tmp_path / "device-data"
    monkeypatch.setattr(tools, "DATA_DIR", device)
    token = hub.add_device("mac")
    monkeypatch.setattr(hub, "client", lambda: hub.Hub(server, token))
    names = {k: f"2026-10-20_1{i}0000_noah" for i, k in enumerate(["gone", "pending", "levels", "extra", "differs"])}
    for name in names.values():
        make_session(device / name)
    (device / names["extra"] / "notes.txt").write_text("not sent")
    sync = tools.Sync()

    sync._send()
    # Without a backup folder on the hub nothing is deleted
    assert tools.Sync.free_space(set()) == ["kept every session here, the hub has no backup folder set"]
    assert all((device / name).exists() for name in names.values())

    dest = tmp_path / "backup"
    dest.mkdir()
    config = hub.load()
    hub.save(dict(config, backup=str(dest)))
    for key in ("gone", "levels", "extra", "differs"):
        hub.back_up(hub_dir / names[key], dest)
    # The backup's copy no longer matches what the device has
    listing = json.loads((dest / hub.BACKUP_LIST).read_text())
    listing[names["differs"]]["board.f64"] = "0" * 64
    (dest / hub.BACKUP_LIST).write_text(json.dumps(listing))
    # Sending never deletes, only the app closing does
    sync._send()
    assert all((device / name).exists() for name in names.values())
    tools.Sync.free_space({names["levels"]})
    assert not (device / names["gone"]).exists()
    for key in ("pending", "levels", "extra", "differs"):
        assert (device / names[key] / "session.json").exists(), key
    assert (hub_dir / names["gone"] / "session.json").exists()


def watch(recorder, sync, backup):
    """A SyncWatch without its thread, checked by calling _check."""
    w = tools.SyncWatch.__new__(tools.SyncWatch)
    w.recorder, w.sync, w.backup, w.state = recorder, sync, backup, {"role": None}
    return w


class FakeSync:
    def __init__(self):
        self.state = {"running": False, "error": None}
        self.sends = 0

    def send(self):
        self.sends += 1


def test_sync_line_and_auto_send(hub_dir, server, tmp_path, monkeypatch):
    device = tmp_path / "device-data"
    monkeypatch.setattr(tools, "DATA_DIR", device)
    token = hub.add_device("mac")
    sync = FakeSync()
    recorder = SimpleNamespace(hub=False, session=None)
    w = watch(recorder, sync, SimpleNamespace(state={"running": False}))
    # Hub and device share hub.json here, so the hub still knows the device's token
    hubs = hub.load()
    assert w._check() == {"role": None}
    hub.save(dict(hubs, url=server, token=token))
    make_session(device / NAME)
    assert w._check()["text"] == "1 to send"
    assert sync.sends == 0
    # Auto send, but not while a session records
    hub.save(dict(hubs, url=server, token=token, auto_send=True))
    recorder.session = object()
    assert w._check()["text"] == "1 to send"
    recorder.session = None
    assert w._check()["text"] == "sending" and sync.sends == 1
    make_session(hub_dir / NAME)
    assert w._check()["text"] == "synced"
    hub.save(dict(hubs, url=server, token="wrong"))
    assert w._check()["text"] == "hub refused"
    hub.save(dict(hubs, url="http://127.0.0.1:9", token=token))
    assert w._check()["text"] == "hub out of reach"
    # The hub's line is about its backup
    hub.save(hubs)
    monkeypatch.setattr(tools, "DATA_DIR", hub_dir)
    w.recorder = SimpleNamespace(hub=True, session=None)
    assert w._check()["text"] == "no backup folder"
    dest = tmp_path / "backup"
    dest.mkdir()
    hub.save(dict(hub.load(), backup=str(dest)))
    assert w._check()["text"] == "1 to back up"
    hub.back_up(hub_dir / NAME, dest)
    assert w._check()["text"] == "backed up"
