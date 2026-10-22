"""The shared evaluation: a held-out session never reaches training, calibration and test windows never overlap,
and notsign windows only count as wrist_as_sign."""
import numpy as np
import pytest

import ml  # noqa: F401
from ml import evaluate


def session(seed, n=60):
    rng = np.random.default_rng(seed)
    label = np.array(["rest", "fist", "pinch"] * (n // 3))
    unit = np.repeat([1, 2, 3], n // 3)
    x = rng.normal(0, 1, (n, 8, 100)).astype(np.float32)
    # Each class gets its own loud channel so LDA can tell them apart
    for k, c in enumerate(["rest", "fist", "pinch"]):
        x[label == c, k] *= 20
    return {"x": x, "label": label, "unit": unit}


class Spy:
    """Records what it was trained and calibrated on, always answers rest."""

    def __init__(self):
        self.fits, self.cals = [], []

    def fit(self, train):
        self.fits.append(set(train))
        return None

    def calibrate(self, model, cal):
        self.cals.append(cal)
        return None

    def predict(self, model, x):
        probs = np.zeros((len(x), 3))
        probs[:, 2] = 1
        return probs, ["fist", "pinch", "rest"]


def test_held_out_never_trains_on_the_held_session():
    sessions = {"a": session(1), "b": session(2), "c": session(3)}
    spy = Spy()
    result = evaluate.held_out(sessions, spy, log=None)
    assert spy.fits == [{"b", "c"}, {"a", "c"}, {"a", "b"}]
    # Always rest: perfect on rest, nothing on the grips
    assert result["mean"]["none"]["balanced"] == pytest.approx(1 / 3)
    assert result["mean"]["none"]["rest_false"] == 0


def test_rotating_calibration_holds_out_one_unit_each():
    spy = Spy()
    evaluate.held_out({"a": session(1), "b": session(2)}, spy, log=None)
    units = [tuple(sorted(set(c["unit"]))) for c in spy.cals[:3]]
    assert units == [(2, 3), (1, 3), (1, 2)]


def test_fixed_calibration_units():
    spy = Spy()
    evaluate.held_out({"a": session(1), "b": session(2)}, spy, cal_units=(1,), log=None)
    assert all(set(c["unit"]) == {1} for c in spy.cals)


def test_benchmark_sessions_stay_out_of_training():
    spy = Spy()
    evaluate.benchmark({"a": session(1), "b": session(2), "c": session(3)}, ["c"], spy)
    assert spy.fits == [{"a", "b"}]
    with pytest.raises(ValueError):
        evaluate.benchmark({"a": session(1)}, ["z"], spy)


def test_notsign_only_counts_as_wrist_as_sign():
    classes = ["fist", "notsign", "rest"]
    true = np.array(["fist", "rest", "notsign", "notsign"])
    probs = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    s = evaluate.scores(true, probs, classes)
    assert s["balanced"] == 1
    assert s["wrist_as_sign"] == pytest.approx(1 / 2)


def test_lda_separates_easy_sessions():
    sessions = {"a": session(1), "b": session(2), "c": session(3)}
    result = evaluate.held_out(sessions, evaluate.LDA(), log=None)
    assert result["mean"]["none"]["balanced"] > 0.9
    assert result["mean"]["calibrated"]["balanced"] > 0.9
