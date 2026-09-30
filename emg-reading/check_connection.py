"""Connect to the MindRove armband, stream for a few seconds and report what arrived.

Join the armband's WiFi network first. Use --synthetic to test without the armband.
"""
import argparse
import time

import numpy as np
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams

from armband import board_rows


def report(data, rows, rate, seconds, battery=True):
    """Lines describing a few seconds of board data (all rows)."""
    samples = data.shape[1]
    expected = int(rate * seconds)
    lines = [f"samples: {samples} (expected about {expected} at {rate} Hz)"]
    if samples == 0:
        return lines + ["no data received"]
    steps = np.diff(data[rows["package"]])
    # Counter resets and wraps show up as negative or huge steps, those aren't drops
    missing = np.where((steps > 1) & (steps < 10 * rate), steps - 1, 0)
    lines.append(f"dropped: {int(missing.sum())}")
    if battery:
        lines.append(f"battery: {data[rows['battery'], -1]:.0f}%")

    # Remove the DC offset so the RMS shows muscle activity rather than electrode offset
    emg = data[rows["emg"]]
    rms = np.sqrt(np.mean((emg - emg.mean(axis=1, keepdims=True)) ** 2, axis=1))
    return lines + [f"ch {ch}: rms {value:10.1f} uV" for ch, value in enumerate(rms)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    board_id = BoardIds.SYNTHETIC_BOARD if args.synthetic else BoardIds.MINDROVE_WIFI_BOARD
    rate = BoardShim.get_sampling_rate(board_id)

    BoardShim.disable_board_logger()
    board = BoardShim(board_id, MindRoveInputParams())
    board.prepare_session()
    board.start_stream()
    print(f"streaming from {board_id.name} for {args.seconds:g} s")
    try:
        time.sleep(args.seconds)
        data = board.get_board_data()
    finally:
        board.stop_stream()
        board.release_session()

    print("\n".join(report(data, board_rows(board_id), rate, args.seconds, battery=not args.synthetic)))


if __name__ == "__main__":
    main()
