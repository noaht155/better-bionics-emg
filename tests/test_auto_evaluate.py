"""After every completed session the app starts the honest evaluation of that session by itself."""
from types import SimpleNamespace

import pytest

import ml  # noqa: F401
from ml import dataset

SESSIONS = dataset.good_sessions()


class FakeJob:
    def __init__(self):
        self.started = []

    def start(self, job, args):
        self.started.append((job, args))


@pytest.mark.skipif(len(SESSIONS) < 2, reason="needs 2 recorded sessions")
def test_completed_session_is_evaluated_before_training():
    pytest.importorskip("torch")
    import app
    recorder = SimpleNamespace(on_saved=None)
    tools = SimpleNamespace(network=FakeJob())
    app.make_app(recorder, None, None, tools)
    recorder.on_saved(SESSIONS[-1])
    assert tools.network.started == [("evaluate_newest", ["ml.train_ringnet", "--epochs", "15",
                                                          "--held", SESSIONS[-1].name])]


def test_session_without_grips_is_skipped(tmp_path):
    import app
    recorder = SimpleNamespace(on_saved=None)
    tools = SimpleNamespace(network=FakeJob())
    app.make_app(recorder, None, None, tools)
    recorder.on_saved(tmp_path / "sync-check-only")
    assert tools.network.started == []
