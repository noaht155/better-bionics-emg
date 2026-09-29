"""Connect to the MindRove armband, stream for a few seconds and report what arrived.

Join the armband's WiFi network first. Use --synthetic to test without the armband.
"""
import argparse
import time

import numpy as np
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    board_id = BoardIds.SYNTHETIC_BOARD if args.synthetic else BoardIds.MINDROVE_WIFI_BOARD
    rate = BoardShim.get_sampling_rate(board_id)
    emg_rows = BoardShim.get_emg_channels(board_id)

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

    samples = data.shape[1]
    expected = int(rate * args.seconds)
    print(f"samples: {samples} (expected about {expected} at {rate} Hz)")
    if samples == 0:
        print("no data received")
        return

    if not args.synthetic:
        battery = data[BoardShim.get_battery_channel(board_id), -1]
        print(f"battery: {battery:.0f}%")

    # Remove the DC offset so the RMS shows muscle activity rather than electrode offset
    emg = data[emg_rows]
    rms = np.sqrt(np.mean((emg - emg.mean(axis=1, keepdims=True)) ** 2, axis=1))
    for ch, value in enumerate(rms):
        print(f"ch {ch}: rms {value:10.1f} uV")


if __name__ == "__main__":
    main()
