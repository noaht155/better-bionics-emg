# EMG filtering

How the armband's EMG is cleaned up before anything uses it, and why. Code: `emg-reading/processing.py`.
Chosen on 2026-09-30 from a guided recording session (about 6 minutes, 12 segments, one person, band on the right forearm).

## Chain

```
raw EMG (8 channels, 500 Hz, µV)
  → band-pass 40 to 200 Hz      4th order Butterworth
  → notch 60, 120, 180 Hz       iirnotch, Q 30
  → envelope                    RMS over the last 150 ms
```

- Runs as a **streaming filter**: the filter state is kept between updates, so each new chunk of samples is filtered once. The output is identical to filtering a whole recording in one go (checked, 0.0 µV difference).
- The filter starts in its steady state for the first sample's value, so the large DC offset of the raw channels doesn't cause a start-up spike.
- All filters are second-order sections (biquads), the same structure a C++ version on the ESP32 would use.
- Total delay about 8 ms at 50 to 110 Hz, well inside the 200 ms budget.
- `FILTER_VERSION` in `processing.py` is saved in `calibration.json`. Change the filter, bump the version, and old calibrations are refused.

## Haptics only

These two are applied in `haptics.py` and `calibrate.py`, not in the shared filter, so recordings and models get the full information.

- **Common average reference (CAR):** subtract the mean of the good channels from every channel. At rest most of what is left after filtering is noise shared by all 8 channels, and CAR removes it, so resting outputs stay closer to 0 %. It also removes the part of real muscle activity that all channels share, which is why the models don't use it. On by default, saved in `calibration.json`, `--no-car` turns it off.
- **Pad contact loss:** if a channel's raw signal moves more than 100 mV within one 150 ms window, its pad is losing skin contact. That channel is left out of the common average and its output is set to 0 % until it recovers.

## Why these settings

Measured on the recorded segments. "Tug" is the band being pulled and twisted while the arm is relaxed.

Frequency content, RMS µV averaged over channels:

| Band | Strong clench | Band tug | Tug ÷ clench |
|---|---|---|---|
| 10 to 20 Hz | 85 | 514 | 6.0 |
| 20 to 30 Hz | 45 | 236 | 5.2 |
| 30 to 50 Hz | 66 | 185 | 2.8 |
| 50 to 80 Hz | 73 | 110 | 1.5 |
| 80 to 120 Hz | 56 | 59 | 1.05 |
| 120 to 200 Hz | 36 | 33 | 0.9 |

Muscle sits mostly at 30 to 120 Hz, the tug mostly below 50 Hz. Raising the low cut from 20 to 40 Hz:

| Low cut | Tug ÷ clench | Clench ÷ rest | Light grip ÷ rest |
|---|---|---|---|
| 20 Hz (old) | 9.7 | 10.2 | 2.04 |
| 30 Hz | 5.8 | 10.3 | 2.04 |
| **40 Hz** | **4.7** | **10.2** | **1.99** |
| 50 Hz | 3.9 | 10.1 | 1.93 |
| 60 Hz | 3.3 | 10.0 | 1.89 |

What 40 Hz costs, checked so it doesn't throw away useful signal:

| | 20 Hz (old) | 40 Hz | 40 Hz + CAR |
|---|---|---|---|
| Clench amplitude kept | 100 % | 86 to 90 % | 73 to 109 % |
| Gesture classification (rest, point, peace, pinch) | 97 % | 96 % | 97 % |
| Open vs close hand | 75 % | 72 % | 69 % |
| Effort levels (rest, light, medium, strong) | 100 % | 100 % | 100 % |
| Effort order right on all 8 channels | yes | yes | yes |

Classification: LDA on 150 ms envelopes, trained on one repetition and tested on the other. Small test (a few repetitions), so differences of a few percent are within noise. 40 Hz loses some amplitude but no measurable information. CAR loses a little (the shared part of the signal), which is why it stays out of the shared filter.

Other results:
- The notches: `iirnotch` at Q 30 removes the hum as well as ±2 Hz Butterworth band-stops, with about half the delay and far less computation.
- Streaming vs re-filtering the last second every update: identical output, less work.
- Contact loss: caught only on the one pad that actually lost contact, zero false flags on every other segment (tugs and hardest clenches included). Holding the flag for 0.5 to 2 s after a burst covered only a few more bad updates and switched off many good ones, so there is no hold.

## Known limits

- **Band movement is reduced, not removed.** A hard tug is still about 2.7 times a clench after filtering and CAR, because it overlaps the muscle's own frequencies. Two detectors were tried and neither is clean enough to switch outputs off:
  - EMG power at 5 to 25 Hz caught 62 to 84 % of tugs but also flagged about 20 % of strong clenches.
  - Accelerometer movement caught 96 % of arm movement but only 62 % of tugs, and flagged up to 11 % of clenches.
  The fix is mechanical: a snug band now, fixed pads in the socket later. The accelerometer is worth giving to the models as an extra input.
- **Light grips are faint:** about 2 times the resting level. Gentle grips will be hard for any threshold or model.
- **Arm posture raises the resting level:** arm held forward or raised gives 3 to 5 times the resting level on the desk, from the muscles holding the arm up. This is for calibration and the models to handle, not the filter.
- **Hum depends on where the arm is:** strongest near the desk and the PC, almost gone with the arm lifted. The notches deal with it.
- One session, one person. The direction of each result is clear, the exact low cut (40 vs 50 Hz) is worth rechecking on later sessions.

## Re-evaluating

Recordings from the session are in `emg-reading/data/2026-09-30-filter/` (not in git): one raw `.npy` per segment (all 39 board rows) and a `.json` with the cue labels and the sample index of each cue change. Any new filter can be scored on them the same way.
