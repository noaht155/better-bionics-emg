"""Compare mains hum and resting noise between recording setups (desktop vs MacBook, charger in or out).

record: stream the band for some seconds with the arm relaxed and save the raw board data under a label.
compare: one table per measure over every saved recording, or the ones named.
session: the same measures on the rest cues of recorded sessions.

Close the app first, only one program can hold the band. Run from emg-reading/:
    python hum_test.py record A1-desktop --seconds 60
    python hum_test.py compare
    python hum_test.py session 2026-10-21_141651_noah
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import signal

from armband import Armband
from gestures import settle_s
from processing import MAINS_HZ, contact_lost, filter_block
from quality import line_rms
from session import load_session, segments

DATA_DIR = Path(__file__).parent / "data"
# No session.json in it, so the sync never sends or deletes these
OUT_DIR = DATA_DIR / "hum-test"
SECOND = 1.0
# Outside the 40 to 200 Hz band the filter keeps: noise there isn't muscle, and it isn't a mains line either
OUT_OF_BAND = ((20.0, 35.0), (205.0, 245.0))


def band_rms(x, rate, lo, hi):
    """RMS in uV between lo and hi Hz, mains lines left out. x: channels x samples of raw data."""
    x = signal.detrend(x, axis=1)
    w = np.hanning(x.shape[1])
    psd = 2 * np.abs(np.fft.rfft(x * w, axis=1)) ** 2 / (rate * np.sum(w ** 2))
    f = np.fft.rfftfreq(x.shape[1], 1 / rate)
    keep = (f >= lo) & (f <= hi)
    for hz in MAINS_HZ:
        keep &= np.abs(f - hz) > 3
    return np.sqrt(psd[:, keep].sum(axis=1) * (f[1] - f[0]))


def measures(pieces, rate):
    """Median over 1 s windows of each measure, per channel. pieces: raw channels x samples blocks (one per rest
    cue, or the whole recording). Windows where a pad lost contact are counted, not measured."""
    n = int(SECOND * rate)
    # The synthetic board runs at 250 Hz, so only what lies under its Nyquist frequency
    lines = [hz for hz in MAINS_HZ if hz + 10 < rate / 2]
    bands = [(lo, hi) for lo, hi in OUT_OF_BAND if hi < rate / 2]
    rows = {"filtered rest": [], **{f"{hz:.0f} Hz line": [] for hz in lines},
            **{f"{lo:.0f}-{hi:.0f} Hz": [] for lo, hi in bands}}
    lost = total = 0
    for raw in pieces:
        y = filter_block(raw, rate)
        for i in range(n // 2, raw.shape[1] - n + 1, n):
            w = raw[:, i:i + n]
            total += 1
            if contact_lost(w).any():
                lost += 1
                continue
            rows["filtered rest"].append(np.sqrt(np.mean(y[:, i:i + n] ** 2, axis=1)))
            for hz in lines:
                rows[f"{hz:.0f} Hz line"].append(line_rms(w, rate, hz))
            for lo, hi in bands:
                rows[f"{lo:.0f}-{hi:.0f} Hz"].append(band_rms(w, rate, lo, hi))
    out = {k: np.median(v, axis=0) if v else None for k, v in rows.items()}
    return out, lost, total


def record(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.npy"
    if path.exists():
        raise SystemExit(f"{path.name} exists, pick another label")
    chunks = []
    with Armband(args.synthetic) as band:
        print(f"recording {args.seconds:g} s, arm relaxed and still")
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(0.1)
            chunks.append(band.read_all())
        chunks.append(band.read_all())
        rows, rate = band.rows, band.rate
    board = np.hstack(chunks)
    steps = np.diff(board[rows["package"]])
    dropped = int(np.where((steps > 1) & (steps < 10 * rate), steps - 1, 0).sum())
    np.save(path, board)
    path.with_suffix(".json").write_text(json.dumps({
        "label": args.label, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "rate": rate, "rows": rows,
        "source": "synthetic" if args.synthetic else "armband", "dropped": dropped, "note": args.note}, indent=2))
    print(f"saved {path.name}: {board.shape[1]} samples, {dropped} dropped")
    show({args.label: measures([board[rows["emg"]]], rate)})


def compare(args):
    names = args.labels or [p.stem for p in sorted(OUT_DIR.glob("*.npy"), key=lambda p: p.stat().st_mtime)]
    if not names:
        raise SystemExit(f"no recordings in {OUT_DIR}")
    results = {}
    for name in names:
        meta = json.loads((OUT_DIR / f"{name}.json").read_text())
        board = np.load(OUT_DIR / f"{name}.npy")
        results[name] = measures([board[meta["rows"]["emg"]]], meta["rate"])
    show(results)


def from_sessions(args):
    results = {}
    for name in args.names:
        data = load_session(DATA_DIR / name)
        rate = data["meta"].get("rate", 500)
        rest = [s for s in segments(data["events"]) if s["kind"] == "hold" and s["label"] == "rest"
                and s["sample1"] and not s["bad"]]
        cut = int(settle_s("hold", "rest") * rate)
        pieces = [data["emg"][:, s["sample0"] + cut:s["sample1"]] for s in rest if s["sample1"] - s["sample0"] > cut]
        results[name] = measures(pieces, rate)
    show(results)


def show(results):
    """One table per measure: a row per recording, uV per channel and the median over channels."""
    width = max(len(n) for n in results)
    first = next(iter(results.values()))[0]
    for measure in first:
        print(f"\n{measure} (uV)")
        print(f"{'':{width}}" + "".join(f"{f'ch {ch}':>8}" for ch in range(8)) + f"{'median':>8}")
        for name, (values, lost, total) in results.items():
            v = values[measure]
            if v is None:
                print(f"{name:{width}}  no windows with contact")
                continue
            print(f"{name:{width}}" + "".join(f"{x:8.1f}" for x in v) + f"{np.median(v):8.1f}")
    print()
    for name, (values, lost, total) in results.items():
        print(f"{name}: {total} one-second windows, {lost} with a pad losing contact")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("record", help="stream the band and save the raw data under a label")
    p.add_argument("label", help="for example A1-desktop or B2-mac-charger")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--note", default="", help="where the computer sits, charger, what the free hand touches")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    p.set_defaults(run=record)
    p = sub.add_parser("compare", help="tables over saved recordings, all of them by default")
    p.add_argument("labels", nargs="*")
    p.set_defaults(run=compare)
    p = sub.add_parser("session", help="the same tables on the rest cues of recorded sessions")
    p.add_argument("names", nargs="+")
    p.set_defaults(run=from_sessions)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
