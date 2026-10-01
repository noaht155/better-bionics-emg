"""Protocol A: discrete gestures from EMG with LDA.

Features per channel on sliding windows of the shared filtered EMG (never CAR): RMS, mean absolute value,
waveform length, zero crossings and slope sign changes, and by default also autoregressive coefficients per
channel and the correlation of every pair of channels (100 features in all). Training windows come from the held-gesture and rest
cues of recorded sessions, minus the start of each cue where the hand is still changing. Live prediction
computes the same features on the newest window and smooths the output with a majority vote.

    python -m ml.gesture_model emg-reading/data/<session> emg-reading/data/<session> ... [--save]

from the repo root, prints the leave-one-session-out and leave-one-posture-out evaluation, --save also trains on
all the given sessions and saves the model to emg-reading/models/gestures/.
"""
import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from gestures import GESTURE_SET
from ml import EMG_READING
from processing import FILTER_SETTLE_S, FILTER_VERSION, filter_block
from session import load_session, segments

# Next to the recordings, where the app has always kept them
MODEL_DIR = EMG_READING / "models" / "gestures"

WINDOW_MS = 200
STEP_MS = 50
# The EMG changed 0.2 to 1.4 s after a cue on 2026-09-30 (mostly about 0.5 s), for gestures and for letting go
TRIM_START_S = 1.0
VOTE = 5
# Below this probability the live output says unsure instead of guessing. On 2026-10-01 at 0.8, within one band
# position, wrong outputs fell from 27 % of windows to 5 % while 92 % of the answers given were right
THRESHOLD = 0.8
UNSURE = "unsure"
# Zero crossings and slope sign changes ignore steps smaller than this, so noise at rest isn't counted.
# Filtered EMG at rest is about 5 uV RMS
COUNT_THRESHOLD_UV = 1.0

FEATURES = ["rms", "mav", "wl", "zc", "ssc"]
# Extended set: per channel the coefficients of an order 4 autoregressive model (the shape of the frequency
# content), and the correlation of every pair of channels (the pattern across the band). On three sessions on
# 2026-10-01 this raised balanced accuracy by 9 to 10 points, with or without recalibration
AR_ORDER = 4


def autoregressive(x, order=AR_ORDER):
    """Yule-Walker coefficients per channel. x: (..., channels, samples)."""
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    r = np.stack([np.mean(x[..., :n - k] * x[..., k:], axis=-1) for k in range(order + 1)], axis=-1)
    toeplitz = np.stack([np.stack([r[..., abs(i - j)] for j in range(order)], axis=-1) for i in range(order)], axis=-2)
    # A tiny ridge keeps a flat (silent) channel from making the system singular
    toeplitz = toeplitz + 1e-6 * np.eye(order)
    return np.linalg.solve(toeplitz, r[..., 1:, None])[..., 0]


def channel_correlations(x):
    """Correlation of every pair of channels, upper triangle. x: (..., channels, samples)."""
    x = x - x.mean(axis=-1, keepdims=True)
    cov = np.einsum("...ci,...di->...cd", x, x)
    std = np.sqrt(np.einsum("...cc->...c", cov))
    corr = cov / (std[..., :, None] * std[..., None, :] + 1e-9)
    upper = np.triu_indices(x.shape[-2], 1)
    return corr[..., upper[0], upper[1]]


def window_features(x, log=True, extended=False):
    """x: channels x samples of filtered EMG, or a batch (..., channels, samples). Returns the 5 features per
    channel, flattened channel by channel, then with extended the autoregressive coefficients and the channel
    correlations. log puts the amplitude features on a log scale, which suits LDA's assumption of normal
    distributions better, EMG amplitudes are skewed."""
    d = np.diff(x, axis=-1)
    rms = np.sqrt(np.mean(x ** 2, axis=-1))
    mav = np.mean(np.abs(x), axis=-1)
    wl = np.sum(np.abs(d), axis=-1)
    zc = np.sum((x[..., :-1] * x[..., 1:] < 0) & (np.abs(d) >= COUNT_THRESHOLD_UV), axis=-1)
    ssc = np.sum((d[..., :-1] * d[..., 1:] < 0)
                 & ((np.abs(d[..., :-1]) >= COUNT_THRESHOLD_UV) | (np.abs(d[..., 1:]) >= COUNT_THRESHOLD_UV)), axis=-1)
    if log:
        rms, mav, wl = (np.log(np.maximum(v, 1e-3)) for v in (rms, mav, wl))
    feats = np.stack([rms, mav, wl, zc, ssc], axis=-1)
    feats = feats.reshape(*feats.shape[:-2], -1)
    if extended:
        ar = autoregressive(x)
        feats = np.concatenate([feats, ar.reshape(*ar.shape[:-2], -1), channel_correlations(x)], axis=-1)
    return feats


def session_windows(data, rate, options):
    """Features of every window in a session, in time order. Returns (features, window end sample, label or
    None, posture or None). Labels only for windows fully inside the trimmed part of a hold cue."""
    channels = data["meta"]["channels"]
    y = filter_block(data["emg"][channels], rate)
    win = int(WINDOW_MS / 1000 * rate)
    step = int(STEP_MS / 1000 * rate)
    ends = np.arange(max(win, int(FILTER_SETTLE_S * rate)), y.shape[1] + 1, step)
    starts = ends - win
    feats = np.concatenate([window_features(sliding_window_view(y, win, axis=1)[:, s].transpose(1, 0, 2),
                                            options["log"], options.get("extended", False))
                            for s in np.array_split(starts, max(1, len(starts) // 2000))])
    if options["accel"]:
        acc = data["accel"]
        feats = np.hstack([feats, np.stack([acc[:, s:e].mean(axis=1) for s, e in zip(starts, ends)])])

    labels = np.full(len(ends), None, dtype=object)
    postures = np.full(len(ends), None, dtype=object)
    trim = int(TRIM_START_S * rate)
    for seg in segments(data["events"]):
        # Older sessions also cued gestures that were dropped from the set, those windows are left out
        if seg["kind"] != "hold" or seg["bad"] or seg["sample1"] is None or seg["label"] not in GESTURE_SET:
            continue
        inside = (starts >= seg["sample0"] + trim) & (ends <= seg["sample1"])
        labels[inside] = seg["label"]
        postures[inside] = seg["posture"]
    return feats, ends, labels, postures


def make_lda():
    # Shrinkage keeps the covariance estimate stable with 40 correlated features and a few thousand windows
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def majority(predictions, vote):
    """Causal majority vote over the last vote predictions, ties go to the newest."""
    out = np.empty(len(predictions), dtype=object)
    recent = deque(maxlen=vote)
    for i, p in enumerate(predictions):
        recent.append(p)
        counts = Counter(recent)
        best = max(counts.values())
        out[i] = next(q for q in reversed(recent) if counts[q] == best)
    return out


def score(true, pred, classes):
    """Accuracy, balanced accuracy, accuracy on the grips alone, rest falsely taken for a gesture, and the
    confusion matrix (rows true, columns predicted).

    Rest is about half of all windows (a rest between every two grips), so plain accuracy is mostly a rest score:
    always answering rest already gets 46 %. Balanced accuracy counts every class equally and is the headline."""
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(true, pred):
        cm[index[t], index[p]] += 1
    rest = true == "rest"
    present = [c for c in classes if np.any(true == c)]
    return {"accuracy": float(np.mean(true == pred)) if len(true) else None,
            "balanced": float(np.mean([np.mean(pred[true == c] == c) for c in present])) if present else None,
            "grips": float(np.mean(pred[~rest] == true[~rest])) if (~rest).any() else None,
            "rest_false": float(np.mean(pred[rest] != "rest")) if rest.any() else None,
            "confusion": cm.tolist()}


def threshold_score(true, probs, classes, threshold):
    """What a confidence threshold does: how often an answer is given, how often it's right, and how often the
    output is a wrong gesture or rest is taken for a gesture (both as a share of all windows)."""
    classes = np.asarray(classes)
    best = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    answer = best >= threshold
    rest = true == "rest"
    return {"threshold": threshold, "answers": float(answer.mean()),
            "right": float(np.mean(pred[answer] == true[answer])) if answer.any() else None,
            "wrong": float(np.mean(answer & (pred != true))),
            "rest_false": float(np.mean(answer[rest] & (pred[rest] != "rest"))) if rest.any() else None}


def evaluate(sessions, options, votes=(1, 3, 5, 7)):
    """sessions: {name: session_windows output}. Leave one session out and leave one posture out.
    Predictions are smoothed over each held-out session's whole window stream, transitions included, then
    scored on the labelled windows only."""
    classes = sorted({str(l) for s in sessions.values() for l in s[2] if l is not None})
    result = {"classes": classes, "sessions": {}, "postures": {}}
    by_vote = {v: ([], []) for v in votes}
    sure = ([], [])
    if len(sessions) > 1:
        for name, (feats, _, labels, _) in sessions.items():
            train = [(f[l != None], l[l != None]) for n, (f, _, l, _) in sessions.items() if n != name]  # noqa: E711
            model = make_lda().fit(np.vstack([f for f, _ in train]), np.concatenate([l for _, l in train]))
            probs = model.predict_proba(feats)
            raw = model.classes_[probs.argmax(axis=1)]
            known = labels != None  # noqa: E711
            # Columns in the order of all classes, in case a class is missing from the training sessions
            full = np.zeros((len(feats), len(classes)))
            full[:, [classes.index(str(c)) for c in model.classes_]] = probs
            sure[0].append(labels[known].astype(str))
            sure[1].append(full[known])
            result["sessions"][name] = {}
            for v in votes:
                pred = majority(raw, v)
                by_vote[v][0].append(labels[known])
                by_vote[v][1].append(pred[known])
                result["sessions"][name][v] = score(labels[known], pred[known], classes)["balanced"]
        result["loso"] = {v: score(np.concatenate(t), np.concatenate(p), classes) for v, (t, p) in by_vote.items()}
        result["threshold"] = threshold_score(np.concatenate(sure[0]), np.vstack(sure[1]), classes, THRESHOLD)

    # Leave one posture out, over all sessions together, without smoothing (postures aren't one stream)
    feats = np.vstack([s[0] for s in sessions.values()])
    labels = np.concatenate([s[2] for s in sessions.values()])
    postures = np.concatenate([s[3] for s in sessions.values()])
    known = labels != None  # noqa: E711
    for p in sorted({p for p in postures[known]}):
        test = known & (postures == p)
        train = known & (postures != p)
        if not train.any():
            continue
        model = make_lda().fit(feats[train], labels[train])
        result["postures"][p] = score(labels[test], model.predict(feats[test]), classes)["balanced"]
    return result


def decision_delay_ms(vote):
    """Rough delay from a change in the muscle to a change in the smoothed output: the window is centred half a
    window back, and a majority of the vote has to see the new gesture."""
    return WINDOW_MS / 2 + (vote // 2) * STEP_MS


def load_sessions(folders, options):
    out = {}
    for folder in folders:
        data = load_session(folder)
        meta = data["meta"]
        if meta.get("filter_version") != FILTER_VERSION:
            raise ValueError(f"{Path(folder).name} was recorded with another filter version")
        windows = session_windows(data, meta["rate"], options)
        if not any(l is not None for l in windows[2]):
            continue
        out[Path(folder).name] = windows
    return out


def train(folders, options, evaluate_it=True):
    """Trains on all the given sessions. Returns (model dict ready to save, evaluation or None)."""
    sessions = load_sessions(folders, options)
    if not sessions:
        raise ValueError("none of these sessions has gesture cues")
    metas = [json.loads((Path(f) / "session.json").read_text()) for f in folders]
    channels = {tuple(m["channels"]) for m in metas}
    if len(channels) > 1:
        raise ValueError("these sessions use different EMG channels")
    feats = np.vstack([s[0] for s in sessions.values()])
    labels = np.concatenate([s[2] for s in sessions.values()])
    known = labels != None  # noqa: E711
    lda = make_lda().fit(feats[known], labels[known])
    model = {"lda": lda, "options": options, "classes": [str(c) for c in lda.classes_],
             "channels": list(channels.pop()), "rate": metas[0]["rate"], "filter_version": FILTER_VERSION,
             "window_ms": WINDOW_MS, "step_ms": STEP_MS,
             "vote": options["vote"], "sessions": list(sessions), "subject": metas[0].get("subject"),
             "trained": time.strftime("%Y-%m-%d %H:%M"), "windows": int(known.sum())}
    result = evaluate(sessions, options) if evaluate_it else None
    model["evaluation"] = result
    return model, result


def save(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model['subject'] or 'model'}_{time.strftime('%Y-%m-%d_%H%M%S')}.joblib"
    joblib.dump(model, path)
    return path


def load(path):
    model = joblib.load(path)
    if model["filter_version"] != FILTER_VERSION:
        raise ValueError("this model was trained with an older filter, train it again")
    return model


class LivePredictor:
    """Feeds the newest window of filtered EMG to a trained model and smooths the output."""

    def __init__(self, model):
        self.model = model
        self.win = int(model["window_ms"] / 1000 * model["rate"])
        self.recent = deque(maxlen=model["vote"])
        self.threshold = THRESHOLD

    def predict(self, filtered, accel=None):
        """filtered: channels x at least one window of filtered EMG, accel: 3 x the same samples (only if the
        model uses it). Returns (smoothed label, raw label, {class: probability}). A label is UNSURE when no
        class reaches the threshold."""
        x = filtered[:, -self.win:]
        # Models saved before the extended set existed have no "extended" option and keep the base features
        feats = window_features(x, self.model["options"]["log"], self.model["options"].get("extended", False))
        if self.model["options"]["accel"]:
            feats = np.concatenate([feats, accel[:, -self.win:].mean(axis=1)])
        probs = self.model["lda"].predict_proba(feats[None])[0]
        raw = self.model["classes"][int(np.argmax(probs))] if probs.max() >= self.threshold else UNSURE
        self.recent.append(raw)
        counts = Counter(self.recent)
        best = max(counts.values())
        label = next(q for q in reversed(self.recent) if counts[q] == best)
        return label, raw, dict(zip(self.model["classes"], probs.round(3).tolist()))


def report(result):
    lines = []
    classes = result["classes"]
    if "loso" in result:
        lines.append("leave one session out (balanced accuracy / grips only / plain accuracy / rest taken for a "
                     "gesture / decision delay):")
        for v, s in result["loso"].items():
            lines.append(f"  vote {v}: {100 * s['balanced']:5.1f} %  {100 * s['grips']:5.1f} %  "
                         f"{100 * s['accuracy']:5.1f} %  {100 * (s['rest_false'] or 0):5.1f} %  "
                         f"~{decision_delay_ms(v):.0f} ms")
        for name, per_vote in result["sessions"].items():
            lines.append(f"  held out {name}, balanced: "
                         + "  ".join(f"vote {v} {100 * a:.1f} %" for v, a in per_vote.items()))
        cm = np.array(result["loso"][VOTE]["confusion"])
        lines.append(f"confusion, vote {VOTE} (rows true, columns predicted, % of the row):")
        lines.append(" " * 11 + "".join(f"{c[:6]:>7s}" for c in classes))
        for c, row in zip(classes, cm):
            pct = 100 * row / max(row.sum(), 1)
            lines.append(f"{c:>10s} " + "".join(f"{p:7.0f}" for p in pct))
        t = result.get("threshold")
        if t:
            right = f"{100 * t['right']:.0f} %" if t["right"] is not None else "-"
            lines.append(f"with a {t['threshold']:.2f} confidence threshold, no vote: answers "
                         f"{100 * t['answers']:.0f} % of the time, right {right} when it answers, "
                         f"wrong gesture {100 * t['wrong']:.1f} %, rest taken for a gesture "
                         f"{100 * (t['rest_false'] or 0):.1f} %")
    else:
        lines.append("only one session, leave one session out needs at least two")
    if result["postures"]:
        lines.append("leave one posture out, no vote, balanced: "
                     + "  ".join(f"{p} {100 * a:.1f} %" for p, a in result["postures"].items()))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--no-log", action="store_true", help="amplitude features without the log")
    parser.add_argument("--base-features", action="store_true", help="only the 5 features per channel")
    parser.add_argument("--accel", action="store_true", help="add the mean accelerometer reading as features")
    parser.add_argument("--vote", type=int, default=VOTE, help="majority vote length for live use")
    parser.add_argument("--save", action="store_true", help="train on all sessions and save the model")
    args = parser.parse_args()
    options = {"log": not args.no_log, "accel": args.accel, "vote": args.vote, "extended": not args.base_features}
    model, result = train(args.folders, options)
    print(f"{model['windows']} labelled windows from {len(model['sessions'])} sessions, classes {model['classes']}")
    print("\n".join(report(result)))
    if args.save:
        print(f"saved {save(model)}")


if __name__ == "__main__":
    main()
