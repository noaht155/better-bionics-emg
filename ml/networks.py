"""Train the general network on every good session, and calibrate a saved one for a band position.

    emg-reading/.venv/Scripts/python -m ml.networks train [--epochs 15] [--sessions name ...]
    emg-reading/.venv/Scripts/python -m ml.networks calibrate <network file> <session folder> [--reps 1 2]

calibrate fits the per-person transform and the grip head on the given repetitions of the session (normally a
short recording at the current band position) and scores the result on its other repetitions, if it has any.
Both save into emg-reading/models/networks/. --json writes the result for the web app.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml import NETWORK_DIR, dataset, gesture_model, low_priority
from ml.ringnet import load, save
from ml.train_ringnet import calibrate, grip_scores, predict, to_tensors, train

STEP_MS = dataset.STEP * 1000 // dataset.RATE


def train_all(epochs, sessions=None):
    """sessions: folder names to train on, default every good session."""
    folders = dataset.good_sessions()
    if sessions:
        folders = [f for f in folders if f.name in sessions]
    if not folders:
        raise SystemExit("none of the given sessions is a completed session with grips")
    data = dataset.load_all(folders)
    print(f"training on {len(data)} sessions", flush=True)
    t0 = time.time()
    model = train(list(data.values()), epochs, seed=0)
    first = next(iter(data.values()))
    info = {"kind": "general", "sessions": list(data), "epochs": epochs, "created": time.strftime("%Y-%m-%d %H:%M"),
            "channels": [int(c) for c in first["channels"]], "rate": dataset.RATE, "window": dataset.WINDOW,
            "step_ms": STEP_MS, "vote": 3}
    path = NETWORK_DIR / f"ringnet_{time.strftime('%Y-%m-%d_%H%M%S')}.pt"
    save(model, info, path)
    print(f"trained in {time.time() - t0:.0f} s, saved {path.name}", flush=True)
    return {"saved": path.name, "sessions": len(data)}


def calibrate_on(network, session, reps):
    from ml.train_ringnet import DEVICE
    model, info = load(NETWORK_DIR / network, DEVICE)
    model.train(False)
    d = dataset.load(session)
    labelled = d["grip"] != ""
    cal = labelled & np.isin(d["rep"], reps)
    test = labelled & (d["rep"] > 0) & ~np.isin(d["rep"], reps)
    if not cal.any():
        raise SystemExit(f"{Path(session).name} has no labelled windows in repetitions {reps}")
    print(f"calibrating {network} on {Path(session).name}, repetitions {reps} ({cal.sum()} windows)", flush=True)
    calibrated = calibrate(model, *to_tensors(d, cal), head=True)
    result = {"calibrated_on": Path(session).name, "reps": reps}
    # A session the network already trained on scores far too well, the test repetitions aren't new to it
    result["seen_in_training"] = Path(session).name in info.get("sessions", [])
    if result["seen_in_training"]:
        print("note: the network was trained on this session, so the score below is optimistic, not a real test",
              flush=True)
    if test.any():
        x, y, _ = to_tensors(d, test)
        scores = grip_scores(y.cpu().numpy(), predict(calibrated, x)[0])
        result["scores"] = scores
        print(f"on the other repetitions: balanced {100 * scores['balanced']:.1f} %, rest taken for a grip "
              f"{100 * scores['rest_false']:.1f} %, wrong grip at {gesture_model.THRESHOLD} "
              f"{100 * scores['wrong']:.1f} %", flush=True)
    info = dict(info, kind="calibrated", base=network, calibrated_on=result["calibrated_on"], reps=reps,
                created=time.strftime("%Y-%m-%d %H:%M"),
                balanced=result.get("scores", {}).get("balanced"), seen_in_training=result["seen_in_training"])
    stem = Path(network).stem
    path = NETWORK_DIR / f"{stem}_cal_{Path(session).name[:17]}.pt"
    save(calibrated.cpu(), info, path)
    print(f"saved {path.name}", flush=True)
    result["saved"] = path.name
    return result


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
    for p in (t, c):
        p.add_argument("--json", help="write the result here")
    args = parser.parse_args()
    low_priority()
    result = train_all(args.epochs, args.sessions) if args.command == "train" else calibrate_on(args.network, args.session, args.reps)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
