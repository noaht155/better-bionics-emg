"""The ring network: rotation independence, live prediction against the evaluation, saving and loading. Skipped
where torch isn't installed."""
import numpy as np
import pytest

import ml  # noqa: F401
from gestures import GESTURE_SET

torch = pytest.importorskip("torch")
from ml import ringnet  # noqa: E402
from ml.train_ringnet import predict  # noqa: E402

CLASSES = sorted(GESTURE_SET)


@pytest.fixture
def net():
    torch.manual_seed(0)
    return ringnet.RingNet(CLASSES).eval()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(32, 8, 100) * 40


def test_whole_electrode_rotation_changes_nothing(net, x):
    with torch.no_grad():
        base = net(x)
        for k in range(1, 8):
            for a, b in zip(base, net(torch.roll(x, k, dims=1))):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-4)


def test_mirrored_band_does_change_the_output(net, x):
    """Not covered by the ring, see the mirrored band entry in devlog.md. If this ever fails, the network has become
    mirror-proof and the planned channel flip isn't needed."""
    with torch.no_grad():
        diff = (net(x)[0] - net(torch.flip(x, dims=[1]))[0]).abs().max()
    assert diff > 1e-3


def test_transform_starts_as_identity(x):
    torch.testing.assert_close(ringnet.ChannelTransform()(x), x)


def test_live_matches_batch(net, x):
    """NetPredictor on one window gives what train_ringnet.predict gives for the same window."""
    model = {"net": net, "classes": CLASSES, "window": 100, "vote": 1}
    probs, angles, effort = predict(net, x)
    for i in range(5):
        live = ringnet.NetPredictor(model, threshold=0.0, unsure="unsure")
        _, raw, p = live.predict(x[i].numpy())
        np.testing.assert_allclose([p[c] for c in CLASSES], probs[i], atol=1e-3)
        assert raw == CLASSES[int(probs[i].argmax())]
        # The first window isn't smoothed yet
        np.testing.assert_allclose(list(live.angles.values()), angles[i], atol=0.1)
        np.testing.assert_allclose(list(live.effort_angles.values()), effort[i], atol=0.1)


def test_save_and_load(net, x, tmp_path):
    path = tmp_path / "net.pt"
    ringnet.save(net, {"channels": list(range(8))}, path)
    loaded, info = ringnet.load(path)
    assert info["channels"] == list(range(8))
    with torch.no_grad():
        for a, b in zip(net(x), loaded(x)):
            torch.testing.assert_close(a, b)


def test_networks_without_effort_head_still_load(x, tmp_path):
    path = tmp_path / "old.pt"
    ringnet.save(ringnet.RingNet(CLASSES, effort=False).eval(), {}, path)
    loaded, _ = ringnet.load(path)
    assert loaded.effort_head is None
    with torch.no_grad():
        assert loaded(x)[2] is None
