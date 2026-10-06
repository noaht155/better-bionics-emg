# Pads with no electrical connection (see hardware-status.md). Drop these before any processing.
DEAD_CHANNELS = ()


def live_channels(num_channels):
    return [ch for ch in range(num_channels) if ch not in DEAD_CHANNELS]


def mirrored(settings):
    """True when the band sits mirrored against the reference orientation (left arm, charging port towards the
    shoulder, every session up to 2026-10-06). Turning the band round or moving it to the other arm reverses the
    electrode order round the arm relative to the muscles; both together cancel. The ring network ignores rotation
    but not this (70 % to 38 % without calibration, devlog 2026-10-05), so mirrored recordings and live windows get
    their channel order reversed. settings: a session's settings, older ones without port_facing count as shoulder."""
    return (settings.get("band_arm", "left") == "right") != (settings.get("port_facing", "shoulder") == "hand")
