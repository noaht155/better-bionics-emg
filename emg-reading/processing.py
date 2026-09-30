import numpy as np
from mindrove.data_filter import DataFilter, DetrendOperations, FilterTypes

# At 500 Hz the highest frequency the armband can represent is 250 Hz
BAND_LOW = 20.0
BAND_HIGH = 200.0
MAINS_HZ = (60.0, 120.0, 180.0)

ENVELOPE_MS = 150
# The filters start from zero, so their first output samples are not usable
FILTER_SETTLE_S = 0.5


def filter_channel(x, rate):
    # DataFilter works in place, so filter a copy and leave the caller's data untouched
    x = np.array(x, dtype=np.float64)
    DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
    # Filters above half the sample rate blow up (the synthetic board runs at 250 Hz)
    nyquist = rate / 2
    DataFilter.perform_bandpass(x, rate, BAND_LOW, min(BAND_HIGH, 0.8 * nyquist), 4, FilterTypes.BUTTERWORTH.value, 0)
    # remove_environmental_noise only cuts 60 Hz to about 10% and leaves the harmonics
    for hz in MAINS_HZ:
        if hz + 2 < nyquist:
            DataFilter.perform_bandstop(x, rate, hz - 2, hz + 2, 4, FilterTypes.BUTTERWORTH.value, 0)
    return x


def envelope(x):
    return float(np.sqrt(np.mean(np.square(x))))


def envelope_series(x, rate, step_ms=50):
    """Envelope of a whole recording, one value every step_ms."""
    y = filter_channel(x, rate)[int(FILTER_SETTLE_S * rate):]
    window = int(ENVELOPE_MS / 1000 * rate)
    step = int(step_ms / 1000 * rate)
    return np.array([envelope(y[i:i + window]) for i in range(0, len(y) - window + 1, step)])
