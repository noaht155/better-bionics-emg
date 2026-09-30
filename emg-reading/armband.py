"""Open the armband (or the synthetic board) and read its live EMG channels."""
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams

from channels import live_channels


class Armband:
    def __init__(self, synthetic=False):
        board_id = BoardIds.SYNTHETIC_BOARD if synthetic else BoardIds.MINDROVE_WIFI_BOARD
        self.rate = BoardShim.get_sampling_rate(board_id)
        emg_rows = BoardShim.get_emg_channels(board_id)[:8]
        self.channels = live_channels(len(emg_rows))
        self._rows = [emg_rows[ch] for ch in self.channels]
        BoardShim.disable_board_logger()
        self._board = BoardShim(board_id, MindRoveInputParams())

    def read(self):
        """Samples that arrived since the last call, one row per live channel."""
        return self._board.get_board_data()[self._rows]

    def __enter__(self):
        self._board.prepare_session()
        self._board.start_stream()
        return self

    def __exit__(self, *exc):
        self._board.stop_stream()
        self._board.release_session()
