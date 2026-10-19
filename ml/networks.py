"""Train the general network on every good session, calibrate a saved one for a band position, or fine-tune one on
sign sessions.

    emg-reading/.venv/Scripts/python -m ml.networks train [--epochs 15] [--sessions name ...]
    emg-reading/.venv/Scripts/python -m ml.networks calibrate <network file> <session folder> [--reps 1 2]
    emg-reading/.venv/Scripts/python -m ml.networks signs <network file> --sessions name ... [--epochs 30]
        [--exclude sign ...]

calibrate fits the per-person transform and the grip head on the given repetitions of the session (normally a
short recording at the current band position) and scores the result on its other repetitions, if it has any.
signs starts from a general network, puts a new output layer on it for the signs of protocol E and F sessions (rest
and the wrist bends as notsign, labelled as the Train tab's LDA does) and fine-tunes the whole network on them. It
runs live without a calibration, but calibrating it (calibrate) was worth 10 points on the third F day (devlog
2026-10-19).
All save into emg-reading/models/networks/. --json writes the result for the web app.
"""
import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

from channels import mirrored
from ml import NETWORK_DIR, dataset, gesture_model, low_priority
from ml.ringnet import load, save
from ml.train_ringnet import (BATCH, CALIBRATION_STEPS, DEVICE, augment, calibrate, epoch_printer, grip_scores,
                              predict, to_tensors, train)
from processing import filter_block
from session import DATA_DIR, load_session, segments

STEP_MS = dataset.STEP * 1000 // dataset.RATE


def train_all(epochs, sessions=None):
    """sessions: folder names to train on, default every good session."""
    folders = dataset.good_sessions()
    if sessions:
        folders = [f for f in folders if f.name in sessions]
    if not folders:
        raise SystemExit("none of the given sessions is a completed session with grips")
    data = dataset.load_all(folders, verbose=True)
    print(f"training on {len(data)} sessions", flush=True)
    t0 = time.time()
    model = train(list(data.values()), epochs, seed=0, progress=epoch_printer())
    first = next(iter(data.values()))
    info = {"kind": "general", "sessions": list(data), "epochs": epochs, "created": time.strftime("%Y-%m-%d %H:%M"),
            "channels": [int(c) for c in first["channels"]], "rate": dataset.RATE, "window": dataset.WINDOW,
            "step_ms": STEP_MS, "vote": 3}
    path = NETWORK_DIR / f"ringnet_{time.strftime('%Y-%m-%d_%H%M%S')}.pt"
    save(model, info, path)
    print(f"trained in {time.time() - t0:.0f} s, saved {path.name}", flush=True)
    return {"saved": path.name, "sessions": len(data)}


def calibrate_on(network, session, reps):
    """For a sign network reps are rounds of a sign session."""
    model, info = load(NETWORK_DIR / network, DEVICE)
    model.train(False)
    result = {"calibrated_on": Path(session).name, "reps": reps}
    # The first sign networks were saved with kind signs only
    signs = bool(info.get("signs") or info.get("kind") == "signs")
    if signs:
        print(f"calibrating {network} on {Path(session).name}, rounds {reps}", flush=True)
        calibrated, scores = calibrate_signs(model, session, reps)
        if scores is not None:
            result["scores"] = scores
    else:
        d = dataset.load(session)
        labelled = d["grip"] != ""
        cal = labelled & np.isin(d["rep"], reps)
        test = labelled & (d["rep"] > 0) & ~np.isin(d["rep"], reps)
        if not cal.any():
            raise SystemExit(f"{Path(session).name} has no labelled windows in repetitions {reps}")
        print(f"calibrating {network} on {Path(session).name}, repetitions {reps} ({cal.sum()} windows)",
              flush=True)
        calibrated = calibrate(model, *to_tensors(d, cal), head=True)
        if test.any():
            x, y, _ = to_tensors(d, test)
            result["scores"] = grip_scores(y.cpu().numpy(), predict(calibrated, x)[0])
    # A session the network already trained on scores far too well, the test repetitions aren't new to it
    result["seen_in_training"] = Path(session).name in info.get("sessions", [])
    if result["seen_in_training"]:
        print("note: the network was trained on this session, so the score below is optimistic, not a real test",
              flush=True)
    if "scores" in result:
        scores = result["scores"]
        print(f"on the other {'rounds' if signs else 'repetitions'}: balanced {100 * scores['balanced']:.1f} %, "
              f"rest taken for a position {100 * scores['rest_false']:.1f} %, "
              f"wrong position at {gesture_model.THRESHOLD} "
              f"{100 * scores['wrong']:.1f} %", flush=True)
        if scores.get("wrist_as_sign") is not None:
            print(f"wrist bends read as a sign {100 * scores['wrist_as_sign']:.1f} %", flush=True)
    # The band sits as in this session while the calibrated network runs, so live windows are turned the same way
    settings = json.loads((Path(session) / "session.json").read_text()).get("settings", {})
    if signs:
        info = dict(info, signs=True)
    info = dict(info, kind="calibrated", base=network, calibrated_on=result["calibrated_on"], reps=reps,
                mirrored=mirrored(settings),
                created=time.strftime("%Y-%m-%d %H:%M"),
                balanced=result.get("scores", {}).get("balanced"), seen_in_training=result["seen_in_training"])
    stem = Path(network).stem
    path = NETWORK_DIR / f"{stem}_cal_{Path(session).name[:17]}.pt"
    save(calibrated.cpu(), info, path)
    print(f"saved {path.name}", flush=True)
    result["saved"] = path.name
    return result


def sign_windows(folder):
    """Filtered windows, labels and round of a sign session, the labelled windows only. Round k of a sign is its k-th
    cue in that posture; rest windows count to the round of the sign cue before them."""
    data = load_session(folder)
    y = filter_block(data["emg"][data["meta"]["channels"]], dataset.RATE).astype(np.float32)
    if mirrored(data["meta"].get("settings", {})):
        y = y[::-1]
    options = {"log": True, "accel": False, "signs": True}
    _, ends, labels, postures = gesture_model.session_windows(data, dataset.RATE, options)
    rounds = np.ones(len(ends), int)
    count = Counter()
    for seg in segments(data["events"]):
        if seg["bad"] or seg["sample1"] is None or seg["kind"] not in ("custom", "nogrip"):
            continue
        count[seg["posture"], seg["label"]] += 1
        rounds[(ends > seg["sample0"]) & (postures == seg["posture"])] = count[seg["posture"], seg["label"]]
    keep = labels != None  # noqa: E711
    x = sliding_window_view(y, dataset.WINDOW, axis=1)[:, ends[keep] - dataset.WINDOW].transpose(1, 0, 2)
    return np.ascontiguousarray(x), labels[keep].astype(str), rounds[keep]


def fit_signs(model, x, y, params, lr, steps, on_epoch=None):
    """Cross-entropy with class weights on augmented batches, as in diagnostics/signs/sign_net.py where the devlog's
    numbers come from. on_epoch(epoch, loss) after every pass's worth of steps."""
    counts = np.bincount(y.cpu().numpy(), minlength=len(model.classes)).astype(float)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(model.classes), dtype=torch.float32,
                           device=DEVICE)
    per_epoch = int(np.ceil(len(x) / BATCH))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(x), (min(BATCH, len(x)),), device=DEVICE)
        scaled, _ = augment(x[idx])
        loss = F.cross_entropy(model(scaled)[0], y[idx], weight=weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if on_epoch is not None and (step + 1) % per_epoch == 0:
            on_epoch((step + 1) // per_epoch, float(loss.detach()))
    model.eval()
    return model


def train_signs(network, folders, epochs, exclude=()):
    """exclude: signs (or notsign) left out, their windows aren't trained on."""
    folders = [Path(f) for f in folders]
    if "rest" in exclude:
        raise SystemExit("rest can't be left out")
    if not gesture_model.sign_sessions(folders):
        raise SystemExit("pick sign sessions (protocol E or F) only")
    parts = []
    for f in folders:
        print(f"loading {f.name}", flush=True)
        parts.append(sign_windows(f))
    # Only the signs every chosen session has, as the Train tab's LDA
    classes = sorted(str(c) for c in set.intersection(*(set(p[1]) for p in parts)) - set(exclude))
    keep = [np.isin(p[1], classes) for p in parts]
    x = torch.tensor(np.concatenate([p[0][k] for p, k in zip(parts, keep)]), device=DEVICE)
    y = torch.tensor(np.concatenate([[classes.index(l) for l in p[1][k]] for p, k in zip(parts, keep)]),
                     dtype=torch.long, device=DEVICE)
    print(f"{len(x)} windows, classes {', '.join(classes)}", flush=True)

    torch.manual_seed(0)
    model, base = load(NETWORK_DIR / network, DEVICE)
    model.grip_head = nn.Linear(model.grip_head.in_features, len(classes)).to(DEVICE)
    model.classes = classes
    per_epoch = int(np.ceil(len(x) / BATCH))
    total = epochs + epochs // 2
    progress = epoch_printer()

    t0 = time.time()
    # The new layer first, so its random start doesn't pull the backbone around, then everything but the transform,
    # which stays the identity like in the general network
    for p in model.parameters():
        p.requires_grad_(False)
    model.grip_head.requires_grad_(True)
    fit_signs(model, x, y, list(model.grip_head.parameters()), 3e-3, epochs // 2 * per_epoch,
              lambda e, loss: progress(e, total, loss))
    for p in model.parameters():
        p.requires_grad_(True)
    model.transform.requires_grad_(False)
    fit_signs(model, x, y, [p for p in model.parameters() if p.requires_grad], 3e-4, epochs * per_epoch,
              lambda e, loss: progress(epochs // 2 + e, total, loss))

    last = json.loads((folders[-1] / "session.json").read_text()).get("settings", {})
    info = dict(base, kind="signs", signs=True, base=network, sessions=[f.name for f in folders], epochs=epochs,
                mirrored=mirrored(last), created=time.strftime("%Y-%m-%d %H:%M"))
    for k in ("calibrated_on", "reps", "balanced", "seen_in_training"):
        info.pop(k, None)
    path = NETWORK_DIR / f"{Path(network).stem}_signs_{time.strftime('%Y-%m-%d_%H%M%S')}.pt"
    save(model.cpu(), info, path)
    print(f"trained in {time.time() - t0:.0f} s, saved {path.name}", flush=True)
    return {"saved": path.name, "sessions": len(folders), "classes": classes}


def sign_scores(classes, true, probs):
    """Balanced over the signs and rest (the devlog's numbers), rest taken for a sign, wrong sign shown at the
    threshold, and held wrist bends read as a sign."""
    true = np.asarray(classes)[true]
    pred = np.asarray(classes)[probs.argmax(1)]
    positions = true != gesture_model.NOT_SIGN
    s = gesture_model.score(true[positions], pred[positions], classes)
    t = gesture_model.threshold_score(true[positions], probs[positions], classes, gesture_model.THRESHOLD)
    wrist = ~positions
    signs = [c for c in classes if c not in ("rest", gesture_model.NOT_SIGN)]
    return {"balanced": s["balanced"], "rest_false": s["rest_false"], "wrong": t["wrong"], "answers": t["answers"],
            "wrist_as_sign": float(np.isin(pred[wrist], signs).mean()) if wrist.any() else None}


def calibrate_signs(model, session, reps):
    """The transform and the sign layer fitted on the given rounds of a sign session, the rest frozen. Returns the
    calibrated network and its scores on the other rounds (None without any)."""
    x, labels, rounds = sign_windows(session)
    known = np.isin(labels, model.classes)
    cal, test = known & np.isin(rounds, reps), known & ~np.isin(rounds, reps)
    if not cal.any():
        raise SystemExit(f"{Path(session).name} has no labelled windows of this network's signs in rounds {reps}")
    print(f"{cal.sum()} calibration windows", flush=True)
    y = np.array([model.classes.index(l) if k else 0 for l, k in zip(labels, known)])
    xt, yt = torch.tensor(x, device=DEVICE), torch.tensor(y, dtype=torch.long, device=DEVICE)
    torch.manual_seed(0)
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.transform.reset()
    tuned = list(model.transform.parameters()) + list(model.grip_head.parameters())
    for p in tuned:
        p.requires_grad_(True)
    idx = torch.tensor(np.flatnonzero(cal), device=DEVICE)
    fit_signs(model, xt[idx], yt[idx], tuned, 1e-2, CALIBRATION_STEPS)
    scores = None
    if test.any():
        idx = torch.tensor(np.flatnonzero(test), device=DEVICE)
        scores = sign_scores(model.classes, y[test], predict(model, xt[idx])[0])
    return model, scores


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--sessions", nargs="*", help="session folder names, default every good session")
    c = sub.add_parser("calibrate")
    c.add_argument("network", help="file name in emg-reading/models/networks/")
    c.add_argument("session", help="session folder")
    c.add_argument("--reps", type=int, nargs="+", default=[1, 2])
    s = sub.add_parser("signs")
    s.add_argument("network", help="general network file in emg-reading/models/networks/")
    s.add_argument("--sessions", nargs="+", required=True, help="protocol E and F session folder names")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--exclude", nargs="*", default=[], help="signs to leave out")
    for p in (t, c, s):
        p.add_argument("--json", help="write the result here")
    args = parser.parse_args()
    low_priority()
    if args.command == "train":
        result = train_all(args.epochs, args.sessions)
    elif args.command == "calibrate":
        result = calibrate_on(args.network, args.session, args.reps)
    else:
        result = train_signs(args.network, [DATA_DIR / n for n in args.sessions], args.epochs, args.exclude)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
