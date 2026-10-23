"""One evaluation for every model: whole held-out sessions, calibrate-then-test, and the fixed benchmark.

Each session is a dict of windows: x (windows x 8 x samples of filtered EMG), label (str, rest included) and unit, the
repetition (grips) or round (signs) the window belongs to, which calibration splits on. grip_windows and sign_windows
build them.

A method is any object with:
- fit(train) -> model, train is {session name: windows}
- calibrate(model, cal) -> model fitted to one session's calibration windows (leave it out to skip calibration)
- predict(model, x) -> (probabilities, classes), one row per window

Every score comes from scores(), so models are compared on the same numbers. Run as a script, LDA by default:
emg-reading/.venv/Scripts/python -m ml.evaluate [--signs] [--method lda|signnet] [--protocol F] [--cal 1]
    [--exclude open] [--benchmark]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ml import dataset, gesture_model, low_priority

METRICS = ("balanced", "grips", "rest_false", "wrong", "answers", "wrist_as_sign")
# Never trained on in an evaluation: one F session per day from 10-19 to 10-21, the last one recorded on the MacBook
BENCHMARK = {"signs": ["2026-10-19_092314_noah", "2026-10-20_105110_noah", "2026-10-21_141651_noah"]}
SIGN_BASE = "ringnet_2026-10-16_112322.pt"


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


class SignNet:
    """The sign network as the Train tab builds it: a general network with a new sign layer, fine-tuned on the sign
    sessions, calibrated through the transform and the sign layer."""

    def __init__(self, base=SIGN_BASE, epochs=20):
        self.base, self.epochs = base, epochs

    def fit(self, train):
        from ml import networks
        return networks.fine_tune_signs(self.base, [(d["x"], d["label"]) for d in train.values()], self.epochs)[0]

    def calibrate(self, model, cal):
        from ml import networks
        return networks.calibrate_sign_windows(model, cal["x"], cal["label"])

    def predict(self, model, x):
        import torch
        from ml import networks
        return networks.predict(model, torch.tensor(x, device=networks.DEVICE))[0], list(model.classes)


def protocol(folder):
    return json.loads((Path(folder) / "session.json").read_text()).get("settings", {}).get("protocol", "A")


def main():
    parser = argparse.ArgumentParser(description="A model through the shared evaluation, leave one session out")
    parser.add_argument("--signs", action="store_true", help="sign sessions instead of grips")
    parser.add_argument("--method", choices=("lda", "signnet"), default="lda")
    # E and F sign sets differ, so mixing them scores an E session on signs it never had (61.5 % calibrated)
    parser.add_argument("--protocol", nargs="*", help="protocols to include (default: F with --signs, A to D without)")
    parser.add_argument("--cal", type=int, nargs="*", help="calibration units (default: rotate)")
    parser.add_argument("--exclude", nargs="*", default=[], help="labels left out of training and scoring")
    parser.add_argument("--benchmark", action="store_true",
                        help="train once without the benchmark sessions and score those, instead of leave one out")
    args = parser.parse_args()
    if args.method == "signnet" and not args.signs:
        parser.error("signnet runs on sign sessions, add --signs")
    if args.benchmark and not args.signs:
        parser.error("no grip benchmark chosen yet, only --signs")
    low_priority()
    wanted = set(args.protocol or (["F"] if args.signs else ["A", "B", "C", "D"]))
    if args.signs:
        folders = [p.parent for p in sorted(dataset.DATA_DIR.glob("*/session.json"))
                   if json.loads(p.read_text()).get("completed")]
    else:
        folders = dataset.good_sessions()
    # Quick sessions are too short to hold out, see dataset.quick
    folders = [f for f in folders if protocol(f) in wanted and not dataset.quick(f)]
    load = sign_windows if args.signs else grip_windows
    sessions = {}
    for f in folders:
        d = load(f)
        keep = ~np.isin(d["label"], args.exclude)
        sessions[Path(f).name] = {k: v[keep] for k, v in d.items()}
    method = SignNet() if args.method == "signnet" else LDA()
    cal = tuple(args.cal) if args.cal else None
    if args.benchmark:
        result = benchmark(sessions, BENCHMARK["signs"], method, cal)
        for name, row in result["sessions"].items():
            print(f"benchmark {name}: {line(row)}")
    else:
        result = held_out(sessions, method, cal)
    print(f"mean over {len(result['sessions'])} sessions ({', '.join(sorted(wanted))}, {args.method}): "
          f"{line(result['mean'])}")


if __name__ == "__main__":
    main()
