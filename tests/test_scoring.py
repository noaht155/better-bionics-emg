"""The scores every reported number comes from, on small examples worked out by hand, and the evaluation log."""
import csv

import numpy as np
import pytest

import ml  # noqa: F401
from ml.gesture_model import majority, score, threshold_score

CLASSES = ["fist", "pinch", "rest"]


def labels(*groups):
    return np.array([label for label, n in groups for _ in range(n)], dtype=object)


def test_score_by_hand():
    # 6 rest (5 right, 1 taken for fist), 2 fist (both right), 2 pinch (1 right, 1 taken for rest)
    true = labels(("rest", 6), ("fist", 2), ("pinch", 2))
    pred = labels(("rest", 5), ("fist", 1), ("fist", 2), ("pinch", 1), ("rest", 1))
    s = score(true, pred, CLASSES)
    assert s["accuracy"] == pytest.approx(8 / 10)
    assert s["balanced"] == pytest.approx(np.mean([5 / 6, 2 / 2, 1 / 2]))
    assert s["grips"] == pytest.approx(3 / 4)
    assert s["rest_false"] == pytest.approx(1 / 6)
    # Rows true, columns predicted, in the order of CLASSES
    assert s["confusion"] == [[2, 0, 0], [0, 1, 1], [1, 0, 5]]


def test_always_rest_scores_high_plain_but_not_balanced():
    """Why balanced accuracy is the headline: rest is about half of all windows."""
    true = labels(("rest", 46), ("fist", 27), ("pinch", 27))
    s = score(true, np.full(100, "rest", dtype=object), CLASSES)
    assert s["accuracy"] == pytest.approx(0.46)
    assert s["balanced"] == pytest.approx(1 / 3)
    assert s["grips"] == 0
    assert s["rest_false"] == 0


def test_balanced_ignores_classes_not_in_the_test_data():
    true = labels(("rest", 2), ("fist", 2))
    s = score(true, true.copy(), CLASSES)
    assert s["balanced"] == 1


def test_threshold_by_hand():
    true = np.array(["fist", "fist", "rest", "rest"])
    probs = np.array([[0.9, 0.05, 0.05],    # sure and right
                      [0.5, 0.4, 0.1],      # unsure, no answer
                      [0.85, 0.1, 0.05],    # sure and wrong: rest taken for a fist
                      [0.1, 0.1, 0.8]])     # sure and right, exactly at the threshold
    t = threshold_score(true, probs, CLASSES, 0.8)
    assert t["answers"] == pytest.approx(3 / 4)
    assert t["right"] == pytest.approx(2 / 3)
    assert t["wrong"] == pytest.approx(1 / 4)
    assert t["rest_false"] == pytest.approx(1 / 2)


def test_majority_vote():
    assert list(majority(["a", "a", "b", "a"], 3)) == ["a", "a", "a", "a"]
    # Ties go to the newest
    assert list(majority(["a", "b"], 2)) == ["a", "b"]
    assert list(majority(["a", "b", "b", "a"], 4)) == ["a", "b", "b", "a"]
    # Causal: an output only depends on the predictions up to it
    assert list(majority(["a", "a", "b", "b", "b"], 3)) == ["a", "a", "a", "b", "b"]


def test_evaluation_log_keeps_old_rows_when_columns_are_added(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from ml import train_ringnet
    monkeypatch.setattr(train_ringnet, "LOG", tmp_path / "evaluations.csv")
    train_ringnet.log_run({"epochs": 15, "net_none_balanced": 70.0})
    train_ringnet.log_run({"epochs": 20, "net_none_balanced": 71.0, "angle_effort_r": 0.3})
    with open(tmp_path / "evaluations.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["epochs", "net_none_balanced", "angle_effort_r"]
    assert rows[0] == {"epochs": "15", "net_none_balanced": "70.0", "angle_effort_r": ""}
    assert rows[1] == {"epochs": "20", "net_none_balanced": "71.0", "angle_effort_r": "0.3"}
