"""Live plot of the armband's EMG channels.

Keys: f toggles filtering, s toggles the spectrum view.
Dead channels are hidden unless --all is given.
Use --synthetic to test without the armband.
"""
import argparse

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams

from channels import live_channels
from processing import filter_channel

WINDOW_S = 4
UPDATE_MS = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic board instead of the armband")
    parser.add_argument("--all", action="store_true", help="also show the dead channels")
    args = parser.parse_args()

    board_id = BoardIds.SYNTHETIC_BOARD if args.synthetic else BoardIds.MINDROVE_WIFI_BOARD
    rate = BoardShim.get_sampling_rate(board_id)
    emg_rows = BoardShim.get_emg_channels(board_id)[:8]
    channels = list(range(len(emg_rows))) if args.all else live_channels(len(emg_rows))
    rows = [emg_rows[ch] for ch in channels]
    n = WINDOW_S * rate

    BoardShim.disable_board_logger()
    board = BoardShim(board_id, MindRoveInputParams())
    board.prepare_session()
    board.start_stream()

    app = pg.mkQApp("EMG")
    win = pg.GraphicsLayoutWidget(title="EMG")
    win.resize(1000, 900)
    plots, curves = [], []
    for i, ch in enumerate(channels):
        p = win.addPlot(row=i, col=0)
        p.showGrid(x=True, y=True, alpha=0.3)
        plots.append(p)
        curves.append(p.plot(pen=pg.intColor(ch, hues=len(emg_rows))))
    win.show()

    state = {"filtered": False, "spectrum": False}
    history = np.zeros((len(rows), 0))
    sweep = np.full((len(rows), n), np.nan)
    sweep_t = np.arange(n) / rate
    sweep_pos = 0
    gap = rate // 20

    def toggle(key):
        state[key] = not state[key]
        sweep[:] = np.nan
        for p in plots:
            p.setLogMode(y=state["spectrum"])
            p.enableAutoRange()

    QtGui.QShortcut(QtGui.QKeySequence("F"), win, lambda: toggle("filtered"))
    QtGui.QShortcut(QtGui.QKeySequence("S"), win, lambda: toggle("spectrum"))

    def update():
        nonlocal history, sweep_pos
        new = board.get_board_data()[rows]
        history = np.hstack([history, new])[:, -n:]
        # Filtered mode drops the first half second, so wait for a full second of history
        if history.shape[1] < rate:
            return
        k = min(new.shape[1], n)
        write = (sweep_pos + np.arange(k)) % n
        blank = (sweep_pos + k + np.arange(gap)) % n

        mode = ("filtered" if state["filtered"] else "raw") + (" spectrum" if state["spectrum"] else "")
        win.setWindowTitle(f"EMG ({mode})  f: filter  s: spectrum")
        for i, ch in enumerate(channels):
            x = history[i]
            if state["filtered"]:
                # The history is filtered from scratch each frame, so drop the start-up transient
                x = filter_channel(x, rate)[rate // 2:]
            else:
                x = x - x.mean()
            rms = np.sqrt(np.mean(x ** 2))
            plots[i].setTitle(f"ch {ch}   rms {rms:.1f} uV", size="9pt")
            if state["spectrum"]:
                mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)
                curves[i].setData(np.fft.rfftfreq(len(x), 1 / rate), mag + 1e-9)
            else:
                # Sweep like a heart monitor: overwrite the oldest samples and leave a gap after the cursor
                if k:
                    sweep[i, write] = x[-k:]
                sweep[i, blank] = np.nan
                curves[i].setData(sweep_t, sweep[i], connect="finite")
        sweep_pos = (sweep_pos + k) % n

        # Shared y scale so a spike on one channel can be compared against the others
        if not state["spectrum"] and np.isfinite(sweep).any():
            lim = np.nanmax(np.abs(sweep))
            for p in plots:
                p.setYRange(-lim, lim)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UPDATE_MS)

    try:
        app.exec()
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
