"""Train the ring network and compare it with LDA on the same splits and windows.

For every good session in turn (leave one session out):
- the network is trained on all other sessions
- no calibration: tested on the whole held-out session
- calibrated: the per-person transform (and in a second variant also the grip head) is fitted on two repetitions of
  the held-out session with everything else frozen, then tested on the third, for each of the three repetitions
LDA with the extended features gets the same two tests: trained on the other sessions, and trained on the two
calibration repetitions alone. Grips are scored with balanced accuracy (rest is about half of all windows).
Finger angles are scored separately on held grips and on moving fingers (finger and free cues). The held grips are
7 fixed shapes, so their score mostly says whether the grip was recognised; the moving score shows whether the
predicted hand follows the real one.

    emg-reading/.venv/Scripts/python -m ml.train_ringnet [--epochs 15] [--sessions name ...] [--held name ...]

--held only tests the named sessions (each still trained on all the others), for scoring a new session first.
Every run adds a row to emg-reading/models/evaluations.csv, so runs can be compared side by side.
"""
import argparse
import copy
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml import EMG_READING, dataset, gesture_model, low_priority
from ml.ringnet import ANGLE_INDEX, ANGLE_JOINTS, ANGLE_SCALE_DEG, RingNet
from gestures import GESTURE_SET

CLASSES = sorted(GESTURE_SET)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 512
CALIBRATION_STEPS = 300
THRESHOLD = gesture_model.THRESHOLD
MOVING = ["finger", "free"]
ANGLE_ROWS = {"none moving": "moving fingers, no calibration",
              "transform moving": "moving fingers, transform from 2 repetitions",
              "none held": "held grips, no calibration",
              "transform held": "held grips, transform from 2 repetitions",
              "train moving": "moving fingers, its training sessions (not a test)"}
# The effort head is scored on the same rows, the network's angle head stays under the old names
ANGLE_ROWS.update({f"effort {k}": f"{v}, effort head" for k, v in list(ANGLE_ROWS.items())})

# One row per evaluation run, next to the saved models (not in git, it names the sessions)
LOG = EMG_READING / "models" / "evaluations.csv"
# Column prefix per result row in the log
LOG_NAMES = {"lda others": "lda_none", "net none": "net_none", "lda cal": "lda_cal", "net transform": "net_tr",
             "net transform+head": "net_trhead", "net train": "net_train"}


def grip_index(grip):
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return np.array([lookup.get(g, -1) for g in grip])


def augment(x):
    """Per window: a gain per channel and a little noise, so contact differences don't look new to the network.
    Returns the window with and without an extra overall gain, the same window otherwise. The overall gain stands for
    effort: the grips learn to ignore it, the effort head learns from the copy without it."""
    b = x.shape[0]
    overall = torch.exp(torch.randn(b, 1, 1, device=x.device) * 0.3)
    per_channel = torch.exp(torch.randn(b, x.shape[1], 1, device=x.device) * 0.2)
    noise = torch.randn_like(x) * 2.0
    return x * overall * per_channel + noise, x * per_channel + noise


def angle_loss(pred, a):
    known = ~torch.isnan(a).any(dim=1)
    if pred is None or not known.any():
        return 0
    return F.smooth_l1_loss(pred[known] / ANGLE_SCALE_DEG, a[known] / ANGLE_SCALE_DEG)


def losses(model, x, y, a, weights, x_effort=None):
    """Grip and angle loss on x. The effort head learns from x_effort (the same windows without the overall gain) if
    given, else from x as well (calibration, where nothing is scaled). Both copies go through in one pass."""
    n = len(x)
    logits, angles, effort = model(x if x_effort is None else torch.cat([x, x_effort]))
    if x_effort is not None:
        logits, angles = logits[:n], (angles[:n] if angles is not None else None)
        effort = effort[n:] if effort is not None else None
    labelled = y >= 0
    loss = F.cross_entropy(logits[labelled], y[labelled], weight=weights) if labelled.any() else logits.sum() * 0
    return loss + angle_loss(angles, a) + angle_loss(effort, a)


def class_weights(y):
    counts = np.bincount(y[y >= 0], minlength=len(CLASSES)).astype(float)
    w = counts.sum() / np.maximum(counts, 1) / len(CLASSES)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def to_tensors(d, keep):
    x = torch.tensor(d["x"][keep], dtype=torch.float32, device=DEVICE)
    y = torch.tensor(grip_index(d["grip"][keep]), dtype=torch.long, device=DEVICE)
    a = torch.tensor(d["angles"][keep][:, ANGLE_INDEX], dtype=torch.float32, device=DEVICE)
    return x, y, a


def train(sessions, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = [to_tensors(d, np.ones(len(d["grip"]), bool)) for d in sessions]
    x = torch.cat([p[0] for p in parts]); y = torch.cat([p[1] for p in parts]); a = torch.cat([p[2] for p in parts])
    weights = class_weights(y.cpu().numpy())
    model = RingNet(CLASSES).to(DEVICE)
    # The transform stays the identity while the shared network learns, it is only fitted per person
    model.transform.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(x) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), device=DEVICE)
        for i in range(0, len(x), BATCH):
            idx = order[i:i + BATCH]
            scaled, unscaled = augment(x[idx])
            loss = losses(model, scaled, y[idx], a[idx], weights, unscaled)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


def calibrate(model, x, y, a, head=False):
    """Fit the transform (and with head the grip head) on calibration windows, the rest frozen."""
    model = copy.deepcopy(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.transform.reset()
    tuned = list(model.transform.parameters()) + (list(model.grip_head.parameters()) if head else [])
    for p in tuned:
        p.requires_grad_(True)
    opt = torch.optim.Adam(tuned, lr=1e-2)
    weights = class_weights(y.cpu().numpy())
    for _ in range(CALIBRATION_STEPS):
        idx = torch.randint(0, len(x), (min(BATCH, len(x)),), device=DEVICE)
        loss = losses(model, x[idx], y[idx], a[idx], weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def predict(model, x):
    """Grip probabilities, angles and effort-head angles (None without that head) for every window."""
    model.eval()
    out_p, out_a, out_e = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            logits, angles, effort = model(x[i:i + 4096])
            out_p.append(F.softmax(logits, dim=1).cpu())
            out_a.append(angles.cpu())
            if effort is not None:
                out_e.append(effort.cpu())
    return torch.cat(out_p).numpy(), torch.cat(out_a).numpy(), torch.cat(out_e).numpy() if out_e else None


def grip_scores(true, probs):
    true = np.asarray(CLASSES)[true]
    pred = np.asarray(CLASSES)[probs.argmax(1)]
    s = gesture_model.score(true, pred, CLASSES)
    t = gesture_model.threshold_score(true, probs, CLASSES, THRESHOLD)
    return {"balanced": s["balanced"], "grips": s["grips"], "rest_false": s["rest_false"], "wrong": t["wrong"],
            "answers": t["answers"]}


def angle_scores(true, pred):
    """Mean absolute error in degrees and correlation, averaged over the joints, on windows with tracked angles."""
    known = ~np.isnan(true).any(axis=1)
    if known.sum() < 20:
        return None
    t, p = true[known], pred[known]
    corr = [np.corrcoef(t[:, j], p[:, j])[0, 1] for j in range(t.shape[1]) if t[:, j].std() > 1]
    return {"mae": float(np.mean(np.abs(t - p))), "r": float(np.mean(corr))}


def lda(features, labels):
    return gesture_model.make_lda().fit(features, labels)


def lda_scores(model, features, true_idx):
    probs = np.zeros((len(features), len(CLASSES)))
    probs[:, [CLASSES.index(c) for c in model.classes_]] = model.predict_proba(features)
    return grip_scores(true_idx, probs)


def mean_of(rows, key):
    vals = [r[key] for r in rows if r is not None and r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def code_version():
    """Git commit of the code, with + if ml/ or emg-reading/ had uncommitted changes. Scores from different code
    aren't directly comparable."""
    try:
        repo = Path(__file__).resolve().parent.parent
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "ml", "emg-reading"], cwd=repo,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_run(row):
    """Appends a row to the log. When the columns changed since the file was started, the file is rewritten with
    all columns so old and new rows stay in one table."""
    rows = []
    if LOG.exists():
        with LOG.open(newline="") as f:
            rows = list(csv.DictReader(f))
    columns = list(rows[0].keys()) if rows else []
    columns += [c for c in row if c not in columns]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns, restval="")
        writer.writeheader()
        writer.writerows(rows + [row])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sessions", nargs="*", help="session folder names, default every good session")
    parser.add_argument("--held", nargs="*", help="only hold out these sessions, default each in turn")
    parser.add_argument("--json", help="write the summary here, for the web app")
    args = parser.parse_args()
    low_priority()
    torch.set_num_threads(2)

    folders = dataset.good_sessions()
    if args.sessions:
        folders = [f for f in folders if f.name in args.sessions]
    data = dataset.load_all(folders)
    names = list(data)
    print(f"{len(names)} sessions on {DEVICE}: {', '.join(n[:17] for n in names)}")
    feats = {n: gesture_model.window_features(d["x"].astype(np.float64), True, True) for n, d in data.items()}

    results = {k: [] for k in ("net none", "net transform", "net transform+head", "lda others", "lda cal",
                               "net train")}
    angles = {k: [] for k in ANGLE_ROWS}
    t0 = time.time()
    for held in [n for n in names if not args.held or n in args.held]:
        before = {k: len(v) for k, v in results.items()}
        others = [data[n] for n in names if n != held]
        model = train(others, args.epochs, args.seed)
        # The same network scored on the data it learned from. Far above the held-out score means it memorised
        # the training sessions (too many epochs), both low means it hasn't learned enough (too few)
        def add_angles(key, true, pred):
            angles[key].append(angle_scores(true, pred[1]))
            if pred[2] is not None:
                angles[f"effort {key}"].append(angle_scores(true, pred[2]))

        seen_p, seen_y = [], []
        for o in others:
            xo, yo, _ = to_tensors(o, o["grip"] != "")
            seen_p.append(predict(model, xo)[0])
            seen_y.append(yo.cpu().numpy())
            om = np.isin(o["kind"], MOVING)
            if om.any():
                xo, _, ao = to_tensors(o, om)
                add_angles("train moving", ao.cpu().numpy(), predict(model, xo))
        results["net train"].append(grip_scores(np.concatenate(seen_y), np.vstack(seen_p)))
        d = data[held]
        labelled = d["grip"] != ""
        x, y, a = to_tensors(d, np.ones(len(d["grip"]), bool))
        moving = np.isin(d["kind"], MOVING)
        out = predict(model, x)
        a_np = a.cpu().numpy()
        results["net none"].append(grip_scores(y.cpu().numpy()[labelled], out[0][labelled]))
        add_angles("none moving", a_np[moving], [o[moving] if o is not None else None for o in out])
        add_angles("none held", a_np[labelled], [o[labelled] if o is not None else None for o in out])

        other_f = np.vstack([feats[n][data[n]["grip"] != ""] for n in names if n != held])
        other_l = np.concatenate([data[n]["grip"][data[n]["grip"] != ""] for n in names if n != held])
        results["lda others"].append(lda_scores(lda(other_f, other_l), feats[held][labelled],
                                                y.cpu().numpy()[labelled]))

        for r in (1, 2, 3):
            cal = labelled & (d["rep"] != r) & (d["rep"] > 0)
            test = labelled & (d["rep"] == r)
            if not cal.any() or not test.any():
                continue
            xc, yc, ac = to_tensors(d, cal)
            xt, yt, at = to_tensors(d, test)
            for name, head in (("net transform", False), ("net transform+head", True)):
                calibrated = calibrate(model, xc, yc, ac, head)
                out_t = predict(calibrated, xt)
                results[name].append(grip_scores(yt.cpu().numpy(), out_t[0]))
                if name == "net transform":
                    add_angles("transform held", at.cpu().numpy(), out_t)
                    # Calibration only sees held grips, so every moving window is new to it
                    if moving.any():
                        add_angles("transform moving", a_np[moving], predict(calibrated, x[moving]))
            results["lda cal"].append(lda_scores(lda(feats[held][cal], d["grip"][cal]), feats[held][test],
                                                 yt.cpu().numpy()))
        fold = {k: v[before[k]:] for k, v in results.items()}
        print(f"  held out {held[:17]} ({time.time() - t0:.0f} s): balanced, no calibration LDA "
              f"{100 * mean_of(fold['lda others'], 'balanced'):.1f} % network "
              f"{100 * mean_of(fold['net none'], 'balanced'):.1f} % | calibrated LDA "
              f"{100 * mean_of(fold['lda cal'], 'balanced'):.1f} % network+transform+head "
              f"{100 * mean_of(fold['net transform+head'], 'balanced'):.1f} %", flush=True)

    print(f"\ngrips, {len(names)} sessions, mean over held-out sessions / repetitions "
          f"(balanced, grips only, rest taken for a grip, wrong grip shown at {THRESHOLD}, answers at {THRESHOLD}):")
    labels = {"lda others": "LDA, no calibration", "net none": "network, no calibration",
              "lda cal": "LDA, trained on the 2 calibration repetitions",
              "net transform": "network + transform from 2 repetitions",
              "net transform+head": "network + transform + grip head from 2 reps",
              "net train": "network on its training sessions (not a test)"}
    for key, label in labels.items():
        rows = results[key]
        print(f"  {label:46s} {100 * mean_of(rows, 'balanced'):5.1f} %  {100 * mean_of(rows, 'grips'):5.1f} %  "
              f"{100 * mean_of(rows, 'rest_false'):5.1f} %  {100 * mean_of(rows, 'wrong'):5.1f} %  "
              f"{100 * mean_of(rows, 'answers'):3.0f} %")
    print(f"\nfinger angles ({', '.join(ANGLE_JOINTS)}), mean absolute error and correlation:")
    for key, label in ANGLE_ROWS.items():
        print(f"  {label:46s} {mean_of(angles[key], 'mae'):5.1f} deg   r {mean_of(angles[key], 'r'):.2f}")

    held = [n for n in names if not args.held or n in args.held]
    row = {"date": time.strftime("%Y-%m-%d %H:%M"), "code": code_version(),
           "run": "all" if not args.held else "held", "held": " ".join(n[:17] for n in held),
           "sessions": len(names), "epochs": args.epochs, "seed": args.seed, "minutes": round((time.time() - t0) / 60, 1)}
    for key, prefix in LOG_NAMES.items():
        for m in ("balanced", "grips", "rest_false", "wrong", "answers"):
            v = mean_of(results[key], m)
            row[f"{prefix}_{m}"] = "" if np.isnan(v) else round(100 * v, 1)
    for key in ANGLE_ROWS:
        for m in ("mae", "r"):
            v = mean_of(angles[key], m)
            row[f"angle_{key.replace(' ', '_')}_{m}"] = "" if np.isnan(v) else round(v, 2)
    log_run(row)
    print(f"\nadded to {LOG.name}")
    if args.json:
        summary = {"sessions": names, "held": [n for n in names if not args.held or n in args.held],
                   "grips": {labels[k]: {m: mean_of(results[k], m)
                                         for m in ("balanced", "grips", "rest_false", "wrong", "answers")}
                             for k in labels},
                   "angles": {label: {m: mean_of(angles[k], m) for m in ("mae", "r")}
                              for k, label in ANGLE_ROWS.items()}}
        Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
