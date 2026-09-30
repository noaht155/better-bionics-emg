# Pads with no electrical connection (see hardware-status.md). Drop these before any processing.
DEAD_CHANNELS = ()


def live_channels(num_channels):
    return [ch for ch in range(num_channels) if ch not in DEAD_CHANNELS]
