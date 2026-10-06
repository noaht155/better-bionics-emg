"""Calibration may only change the per-person transform (and the grip head when asked), never the shared network."""
import pytest

import ml  # noqa: F401

torch = pytest.importorskip("torch")
from ml import train_ringnet  # noqa: E402
from ml.ringnet import RingNet  # noqa: E402

DEVICE = train_ringnet.DEVICE


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(train_ringnet, "CALIBRATION_STEPS", 20)
    torch.manual_seed(0)
    model = RingNet(train_ringnet.CLASSES).to(DEVICE).eval()
    n = 256
    x = torch.randn(n, 8, 100, device=DEVICE) * 40
    y = torch.randint(0, len(train_ringnet.CLASSES), (n,), device=DEVICE)
    a = torch.randn(n, len(train_ringnet.ANGLE_JOINTS), device=DEVICE) * 30 + 40
    return model, x, y, a


def state(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def changed(before, module):
    after = module.state_dict()
    return {k for k in before if not torch.equal(before[k], after[k])}


@pytest.mark.parametrize("head", [False, True])
def test_only_the_transform_and_grip_head_change(setup, head):
    model, x, y, a = setup
    before = state(model)
    calibrated = train_ringnet.calibrate(model, x, y, a, head=head)
    assert changed(before, model) == set(), "calibration changed the general network it was given"
    moved = changed(before, calibrated)
    allowed = {"transform.mix", "transform.log_gain"} | ({"grip_head.weight", "grip_head.bias"} if head else set())
    assert moved <= allowed, f"calibration changed {sorted(moved - allowed)}"
    assert {"transform.mix", "transform.log_gain"} <= moved
    # BatchNorm running statistics are part of the shared network too
    assert not any("running" in k for k in moved)


def test_calibration_starts_from_identity(setup):
    """A transform left over from an earlier calibration is reset first."""
    model, x, y, a = setup
    with torch.no_grad():
        model.transform.mix.mul_(2)
    calibrated = train_ringnet.calibrate(model, x[:1], y[:1], a[:1])
    # 20 small steps from the identity stay near it, nowhere near the doubled matrix
    assert (calibrated.transform.mix - torch.eye(8, device=DEVICE)).abs().max() < 0.5
