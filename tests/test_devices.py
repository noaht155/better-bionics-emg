"""Recording setups (devices.json): every session gets the camera delay of the setup it was recorded on."""
import json

import pytest

import ml  # noqa: F401
import devices
from ml import dataset


@pytest.fixture
def setup_file(tmp_path, monkeypatch):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"default": "old", "setups": [
        {"id": "old", "name": "Old", "computer": "PC", "camera": 0, "camera_name": "A", "label_shift_s": 0.03},
        {"id": "new", "name": "New", "computer": "Mac", "camera": 0, "camera_name": "B", "label_shift_s": None},
    ]}))
    monkeypatch.setattr(devices, "DEVICE_FILE", path)
    return path


def test_committed_setups_are_complete():
    data = json.loads(devices.DEVICE_FILE.read_text())
    ids = [s["id"] for s in data["setups"]]
    assert len(ids) == len(set(ids)) and data["default"] in ids
    for s in data["setups"]:
        assert {"id", "name", "icon", "computer", "camera", "camera_name", "label_shift_s", "measured"} <= s.keys()


def test_shift_follows_the_setup(setup_file):
    assert dataset.label_shift({}) == 0.03
    assert dataset.label_shift({"setup": None}) == 0.03
    assert dataset.label_shift({"setup": {"id": "new"}}) == dataset.CAMERA_DELAY_S


def test_measuring_later_fixes_older_sessions(setup_file):
    session = {"setup": {"id": "new", "label_shift_s": None}}
    devices.save_shift("new", 0.0451, "test")
    assert dataset.label_shift(session) == 0.0451
    assert devices.find("new")["measured"]


def test_setup_belongs_to_its_computer(monkeypatch):
    monkeypatch.setattr(devices, "computer_name", lambda: "noahs-macbook-pro")
    assert devices.here({"computer": "Noahs-MacBook-Pro"})
    assert not devices.here({"computer": "DESKTOP-SP90LS2"})
