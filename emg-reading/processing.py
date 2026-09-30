import numpy as np
from scipy import signal

# Chosen on recorded sessions from 2026-09-30, see filtering.md
# Bump FILTER_VERSION whenever the filter changes, saved calibrations are then refused
FILTER_VERSION = 2
BAND_LOW = 40.0
BAND_HIGH = 200.0
MAINS_HZ = (60.0, 120.0, 180.0)
NOTCH_Q = 30.0

ENVELOPE_MS = 150
# Kept as a safety margin after the start of a block, the filter starts in steady state so there is little to settle
FILTER_SETTLE_S = 0.5

# A pad losing skin contact swings its raw signal by hundreds of mV. Muscle and band movement stay well under this
CONTACT_LOSS_UV = 100_000


def design_filter(rate):
    """Band-pass plus mains notches as second-order sections."""
    nyquist = rate / 2
    # Filters above half the sample rate blow up (the synthetic board runs at 250 Hz)
    sections = [signal.butter(4, [BAND_LOW, min(BAND_HIGH, 0.8 * nyquist)], "bandpass", fs=rate, output="sos")]
    for hz in MAINS_HZ:
        if hz + 2 < nyquist:
            b, a = signal.iirnotch(hz, NOTCH_Q, fs=rate)
            sections.append(signal.tf2sos(b, a))
    return np.vstack(sections)


class StreamFilter:
    """Filters multi-channel data chunk by chunk and keeps the filter state in between,
    so the output is the same as filtering the whole recording at once."""

    def __init__(self, num_channels, rate):
        self.sos = design_filter(rate)
        self.num_channels = num_channels
        self.zi = None

    def process(self, x):
        """x: channels x samples of raw data. Returns the filtered samples."""
        if x.shape[1] == 0:
            return np.zeros_like(x, dtype=np.float64)
        if self.zi is None:
            # Start as if the first value had always been there, so the large DC offset causes no start-up spike
            base = signal.sosfilt_zi(self.sos)
            self.zi = base[:, None, :] * x[:, 0][None, :, None]
        y, self.zi = signal.sosfilt(self.sos, x, axis=1, zi=self.zi)
        return y


def filter_block(x, rate):
    """Filter a whole recording, channels x samples."""
    return StreamFilter(x.shape[0], rate).process(np.asarray(x, dtype=np.float64))


def filter_channel(x, rate):
    return filter_block(np.asarray(x, dtype=np.float64)[None, :], rate)[0]


def common_average(y, good=None):
    """Subtract the mean of the good channels from every channel. Removes activity shared by all
    channels, which is mostly noise at rest but also some real muscle signal, so only the haptics use it."""
    good = np.ones(y.shape[0], bool) if good is None else good
    if not good.any():
        return y
    return y - y[good].mean(axis=0, keepdims=True)


def contact_lost(raw):
    """raw: channels x samples of unfiltered data, one envelope window long. True where a pad lost contact."""
    return np.ptp(raw, axis=1) > CONTACT_LOSS_UV


def spectrum(x, rate):
    """Magnitude spectrum of one channel. Without removing the trend and windowing, the raw DC drift
    leaks into every bin. Returns (frequencies, magnitudes)."""
    x = signal.detrend(np.asarray(x, dtype=np.float64))
    return np.fft.rfftfreq(len(x), 1 / rate), np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)


def envelope(x):
    return float(np.sqrt(np.mean(np.square(x))))


def envelope_series(y, rate, step_ms=50):
    """Envelope of an already filtered recording, one value every step_ms."""
    y = y[int(FILTER_SETTLE_S * rate):]
    window = int(ENVELOPE_MS / 1000 * rate)
    step = int(step_ms / 1000 * rate)
    return np.array([envelope(y[i:i + window]) for i in range(0, len(y) - window + 1, step)])
