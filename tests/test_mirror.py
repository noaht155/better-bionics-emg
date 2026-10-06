"""A band worn mirrored (turned round, or on the other arm) has its channel order reversed, in training and live."""
import numpy as np
import pytest

import ml  # noqa: F401
from channels import mirrored
from gestures import GESTURE_SET


@pytest.mark.parametrize("arm, port, expected", [
    ("left", "shoulder", False),    # the reference, every session up to 2026-10-06
    ("left", "hand", True),         # band turned round
    ("right", "shoulder", True),    # other arm, worn the same way
    ("right", "hand", False),       # both: the two mirrors cancel
])
def test_mirrored(arm, port, expected):
    assert mirrored({"band_arm": arm, "port_facing": port}) is expected


def test_old_sessions_without_port_count_as_reference():
    assert mirrored({"band_arm": "left"}) is False
    assert mirrored({}) is False


def test_live_network_reverses_mirrored_windows():
    torch = pytest.importorskip("torch")
    from ml import ringnet
    from ml.train_ringnet import predict
    torch.manual_seed(0)
    classes = sorted(GESTURE_SET)
    net = ringnet.RingNet(classes).eval()
    x = (torch.randn(1, 8, 100) * 40).numpy()
    model = {"net": net, "classes": classes, "window": 100, "vote": 1, "mirrored": True}
    _, _, p = ringnet.NetPredictor(model, 0.0, "unsure").predict(x[0])
    probs = predict(net, torch.tensor(np.ascontiguousarray(x[:, ::-1])))[0][0]
    np.testing.assert_allclose([p[c] for c in classes], probs, atol=1e-3)


def test_live_lda_reverses_mirrored_windows():
    from ml import gesture_model
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 30, (40, 8, 100))
    feats = gesture_model.window_features(windows, True, True)
    lda = LinearDiscriminantAnalysis().fit(feats, np.repeat(["a", "b"], 20))
    model = {"lda": lda, "options": {"log": True, "accel": False, "extended": True}, "classes": ["a", "b"],
             "window_ms": 200, "rate": 500, "vote": 1, "mirrored": True}
    live = gesture_model.LivePredictor(model)
    live.threshold = 0.0
    _, _, p = live.predict(windows[0])
    expected = lda.predict_proba(gesture_model.window_features(windows[0][::-1], True, True)[None])[0]
    np.testing.assert_allclose([p["a"], p["b"]], expected, atol=1e-3)
