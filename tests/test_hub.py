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

NAME = "2026-10-20_120000_noah"
# A computer on the tailnet, not this one
REMOTE = ("100.64.0.2", 50000)


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
    tools = SimpleNamespace(network=FakeJob(), snapshot=lambda: {"trainer": {}, "network": {}})
    return app.make_app(SimpleNamespace(on_saved=None), None, None, tools, lan=lan, serving=True)


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
