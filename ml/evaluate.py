"""One evaluation for every model: whole held-out sessions, calibrate-then-test, and the fixed benchmark.

Each session is a dict of windows: x (windows x 8 x samples of filtered EMG), label (str, rest included) and unit, the
repetition (grips) or round (signs) the window belongs to, which calibration splits on. grip_windows and sign_windows
build them.

A method is any object with:
- fit(train) -> model, train is {session name: windows}
- calibrate(model, cal) -> model fitted to one session's calibration windows (leave it out to skip calibration)
- predict(model, x) -> (probabilities, classes), one row per window

Every score comes from scores(), so models are compared on the same numbers. Run as a script for the LDA baseline:
emg-reading/.venv/Scripts/python -m ml.evaluate [--signs] [--cal 1]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ml import dataset, gesture_model, low_priority

METRICS = ("balanced", "grips", "rest_false", "wrong", "answers", "wrist_as_sign")


def grip_windows(folder):
    """The held-grip windows of a grip session (protocols A to D), unit is the repetition."""
    d = dataset.load(folder)
    keep = d["grip"] != ""
    return {"x": d["x"][keep], "label": d["grip"][keep], "unit": d["rep"][keep].astype(int)}


def sign_windows(folder):
    """The labelled windows of a sign session (protocols E and F), unit is the round."""
    from ml import networks
    x, labels, rounds = networks.sign_windows(folder)
    return {"x": x, "label": labels, "unit": rounds}


def scores(true, probs, classes):
    """Balanced accuracy, accuracy on the non-rest classes, rest taken for a class, and at the confidence threshold the
    wrong answers shown and how often there is an answer. Held wrist bends (notsign) count only as wrist_as_sign, the
    share of them read as a sign, never in the other scores."""
    classes = np.asarray(classes)
    true = np.asarray(true)
    pred = classes[probs.argmax(1)]
    positions = true != gesture_model.NOT_SIGN
    s = gesture_model.score(true[positions], pred[positions], list(classes))
    t = gesture_model.threshold_score(true[positions], probs[positions], classes, gesture_model.THRESHOLD)
    wrist = ~positions
    signs = [c for c in classes if c not in ("rest", gesture_model.NOT_SIGN)]
    return {"balanced": s["balanced"], "grips": s["grips"], "rest_false": s["rest_false"], "wrong": t["wrong"],
            "answers": t["answers"],
            "wrist_as_sign": float(np.isin(pred[wrist], signs).mean()) if wrist.any() else None}


def _score(method, model, d, mask):
    probs, classes = method.predict(model, d["x"][mask])
    known = np.isin(d["label"][mask], classes)
    return scores(d["label"][mask][known], probs[known], classes)


def _calibrated(method, model, d, cal_units):
    """Scores after calibration, averaged over the splits. cal_units None rotates: calibrate on every unit but one,
    test on that one, for each unit (as train_ringnet does). A tuple calibrates on those units and tests on the
    rest."""
    units = sorted(set(d["unit"][d["unit"] > 0]))
    splits = [[u for u in units if u != held] for held in units] if cal_units is None else [list(cal_units)]
    rows = []
    for cal_u in splits:
        cal = np.isin(d["unit"], cal_u)
        test = ~cal if cal_units is not None else (d["unit"] > 0) & ~cal
        if not cal.any() or not test.any():
            continue
        tuned = method.calibrate(model, {k: v[cal] for k, v in d.items()})
        rows.append(_score(method, tuned, d, test))
    return mean(rows) if rows else None


def mean(rows):
    """Each metric averaged over the rows that have it."""
    rows = [r for r in rows if r is not None]
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows if r.get(m) is not None]
        out[m] = float(np.mean(vals)) if vals else None
    return out


def held_out(sessions, method, cal_units=None, log=print):
    """Leave one session out: for each session, fit on all the others and score on it whole without calibration,
    then after calibrating on its own units. Returns {"sessions": {name: {"none", "calibrated"}}, "mean": ...}."""
    out = {}
    for name, d in sessions.items():
        model = method.fit({n: s for n, s in sessions.items() if n != name})
        row = {"none": _score(method, model, d, np.ones(len(d["label"]), bool))}
        if hasattr(method, "calibrate"):
            row["calibrated"] = _calibrated(method, model, d, cal_units)
        out[name] = row
        if log:
            log(f"held out {name}: {line(row)}")
    return {"sessions": out, "mean": {k: mean([r.get(k) for r in out.values()]) for k in ("none", "calibrated")}}


def benchmark(sessions, names, method, cal_units=None):
    """Fit once on every session not in names and score each benchmark session, the same way as held_out."""
    missing = [n for n in names if n not in sessions]
    if missing:
        raise ValueError(f"benchmark sessions not loaded: {', '.join(missing)}")
    model = method.fit({n: s for n, s in sessions.items() if n not in names})
    out = {}
    for name in names:
        d = sessions[name]
        out[name] = {"none": _score(method, model, d, np.ones(len(d["label"]), bool))}
        if hasattr(method, "calibrate"):
            out[name]["calibrated"] = _calibrated(method, model, d, cal_units)
    return {"sessions": out, "mean": {k: mean([r.get(k) for r in out.values()]) for k in ("none", "calibrated")}}


def line(row):
    def pct(s, m):
        return f"{100 * s[m]:.1f} %" if s and s.get(m) is not None else "-"
    return "  ".join(f"{k}: balanced {pct(s, 'balanced')} rest false {pct(s, 'rest_false')} wrong {pct(s, 'wrong')}"
                     for k, s in row.items())


class LDA:
    """The baseline: window features and LDA. Calibrating refits on the session's calibration windows alone, the
    same as the calibrated LDA in train_ringnet."""

    def _features(self, x):
        return gesture_model.window_features(x.astype(np.float64), True, True)

    def _fit(self, x, labels):
        return gesture_model.make_lda().fit(self._features(x), labels)

    def fit(self, train):
        return self._fit(np.concatenate([d["x"] for d in train.values()]),
                         np.concatenate([d["label"] for d in train.values()]))

    def calibrate(self, model, cal):
        return self._fit(cal["x"], cal["label"])

    def predict(self, model, x):
        return model.predict_proba(self._features(x)), list(model.classes_)


def is_signs(folder):
    meta = json.loads((Path(folder) / "session.json").read_text())
    return meta.get("settings", {}).get("protocol", "A") in gesture_model.SIGN_PROTOCOLS


def main():
    parser = argparse.ArgumentParser(description="LDA baseline through the shared evaluation, leave one session out")
    parser.add_argument("--signs", action="store_true", help="sign sessions (protocols E and F) instead of grips")
    parser.add_argument("--cal", type=int, nargs="*", help="calibration units (default: rotate)")
    args = parser.parse_args()
    low_priority()
    if args.signs:
        folders = [p.parent for p in sorted(dataset.DATA_DIR.glob("*/session.json"))
                   if json.loads(p.read_text()).get("completed") and is_signs(p.parent)]
    else:
        folders = [f for f in dataset.good_sessions() if not is_signs(f)]
    # Quick sessions are too short to hold out, see dataset.quick
    folders = [f for f in folders if not dataset.quick(f)]
    load = sign_windows if args.signs else grip_windows
    sessions = {Path(f).name: load(f) for f in folders}
    result = held_out(sessions, LDA(), tuple(args.cal) if args.cal else None)
    print(f"mean over {len(sessions)} sessions: {line(result['mean'])}")


if __name__ == "__main__":
    main()
