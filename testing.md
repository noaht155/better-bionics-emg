# Testing and validation

What is checked automatically, what each check protects against, and what isn't checked yet. Model results
(accuracy, angle scores) are not tests; those are in `devlog.md` and `emg-reading/models/evaluations.csv`.

## Running the tests

From the repo root, with the emg-reading venv:

```
emg-reading/.venv/Scripts/python -m pytest tests
```

41 tests, about 7 s. Run them before committing changes to `ml/`, the filter or the quality checks. Tests that need
recorded sessions (`emg-reading/data/`) or torch skip themselves where those are missing, so on the Mac or a fresh
clone only the synthetic tests run.

## What is checked

### Filter (`tests/test_filter.py`)

The models are trained on sessions filtered all at once (`filter_block`), but live data is filtered as it arrives
(`StreamFilter`). If the two differed, live predictions would not match the evaluation.

| Test | Checks |
|---|---|
| `test_chunks_match_whole_recording` | Filtering 20 s in 10-sample packets (how the recorder gets them from the armband) gives the same output as filtering it in one go, to 1e-6 uV |
| `test_uneven_chunks` | The same with random packet sizes of 1 to 40 samples |
| `test_offset_removed_without_startup_spike` | The tens of mV of DC offset are removed and the filter doesn't ring at the start (it starts in steady state) |
| `test_mains_removed` | 100 uV at 60, 120 and 180 Hz comes out under 1 uV RMS |

### Network (`tests/test_network.py`)

| Test | Checks |
|---|---|
| `test_whole_electrode_rotation_changes_nothing` | Rotating the input by 1 to 7 electrodes changes no output (grips, angles, effort head) by more than 1e-4. The rotation independence the design relies on |
| `test_mirrored_band_does_change_the_output` | A mirrored band does change the output. Documents that mirroring is not covered (see the mirrored band entry in `devlog.md`); if it ever fails, the planned channel flip isn't needed |
| `test_transform_starts_as_identity` | The per-person transform changes nothing until it is calibrated |
| `test_live_matches_batch` | `NetPredictor` (live, one window) gives the same grip probabilities, predicted grip and angles of both heads as `train_ringnet.predict` (evaluation, batches) |
| `test_save_and_load` | A saved network loads back with identical outputs and its info |
| `test_networks_without_effort_head_still_load` | Networks saved before the effort head existed load and run, with no effort output |

### Calibration (`tests/test_calibration.py`)

| Test | Checks |
|---|---|
| `test_only_the_transform_and_grip_head_change` | After calibrating, only the transform changed (and the grip head when asked). The convolution layers, both angle heads and the BatchNorm statistics are untouched, and so is the general network that was passed in. If this broke, calibration would quietly retrain the whole network on 2 repetitions |
| `test_calibration_starts_from_identity` | A transform left over from an earlier calibration is reset first |

### Scores (`tests/test_scoring.py`)

Every reported accuracy comes from these functions, so they are checked on examples worked out by hand.

| Test | Checks |
|---|---|
| `test_score_by_hand` | Plain accuracy, balanced accuracy, grips-only accuracy, rest taken for a grip and the confusion matrix on a 10-window example |
| `test_always_rest_scores_high_plain_but_not_balanced` | Always answering rest gets 46 % plain accuracy but 33 % balanced and 0 % on the grips. Why balanced accuracy is the headline |
| `test_balanced_ignores_classes_not_in_the_test_data` | A class missing from the test windows doesn't count as 0 % |
| `test_threshold_by_hand` | With the 0.8 threshold: share of windows answered, right when answered, wrong grip shown, rest taken for a grip |
| `test_majority_vote` | The live vote: majority of the last N, ties go to the newest, and it only looks backwards |
| `test_evaluation_log_keeps_old_rows_when_columns_are_added` | `evaluations.csv` keeps every earlier row when a new run adds columns |

### Training data (`tests/test_dataset.py`)

| Test | Checks |
|---|---|
| `test_shapes` | Windows are 8 x 100 samples, one label row per window, window ends in order (newest session) |
| `test_grip_windows_lie_inside_the_trimmed_hold` | Every window with a grip label, checked one by one against the session's event log, lies inside a hold cue of that grip, the cue isn't marked bad (hum), and the window starts at least 1 s after the cue (the hand is still moving before that) |
| `test_labels_are_in_range` | Grip labels are in the 7-grip set, held grips have repetition 1 to 3 |
| `test_few_impossible_angles` | Per recorded session: under 5 % of the angle windows have a trained joint bent backwards past -60 degrees (see below) |
| `test_moving_windows_count_as_much_as_the_rest` | The angle loss weighting: moving-finger windows add up to the same total weight as all other angle windows |
| `test_batch_features_match_one_window` | The LDA features of one window (live) equal the same window's features in a batch (training) |

### Camera timing (`tests/test_camera.py`)

Frames arrive in pairs with up to a frame of jitter; `dataset.camera_times` fits a straight line of time against
frame number and takes off the measured camera delay.

| Test | Checks |
|---|---|
| `test_fit_removes_jitter_and_delay` | On 60 s of simulated paired, jittered, delayed frames, every fitted time is within 10 ms of when the frame was taken, with no frame-to-frame jitter left |
| `test_unknown_delay_uses_the_default` | A session without a measured delay uses the 90 ms default |
| `test_wrong_delay_shifts_everything` | The fit can't fix a wrong delay: it shifts every label. The delay has to be measured |

### Signal quality (`tests/test_quality.py`)

These decide when the recorder marks a cue bad and redoes it.

| Test | Checks |
|---|---|
| `test_line_rms_measures_the_hum` | The 60 Hz estimate reads 10, 50, 100 and 150 uV of hum within 10 % (plus a few uV of noise floor) |
| `test_emg_noise_alone_is_not_hum` | Broadband EMG-like noise doesn't read as hum (under 10 uV, the warning is at 80) |
| `test_hum_on_one_channel_is_flagged_after_two_seconds` | 150 uV of hum on one channel is flagged on that channel only, once 2 s of estimates exist |
| `test_clean_signal_flags_nothing` | Clean data gives no warnings |
| `test_dropped_samples_counted_from_package_numbers` | Gaps in the package numbers count as dropped samples, a counter reset doesn't |
| `test_floating_pad_is_lost_contact` | A pad swinging over the input range is flagged as lost contact, on that channel only |

## Found by the tests

- **Camera glitches in the angle labels** (2026-10-06): some frames have a trained joint bent backwards past -60
  degrees (index PIP down to -175), which a finger can't do. MediaPipe lost or flipped the hand. 0.02 to 0.8 % of
  angle windows per session, 2.2 % in `2026-10-01_164337`. Planned fix: treat those frames as not tracked in
  `ml/dataset.py`. The recordings themselves stay as they are.
- Abduction angles wrap round +-180 degrees on bent fingers. Not a problem, abduction isn't trained.

## Validated by experiment, not by tests

Checks that need training runs, too slow for the test suite. Numbers in `devlog.md`.

- Leave one session out: every score is on a session the model never trained on, LDA and network on the same windows.
- Practice against exam score (training sessions against held-out) in every evaluation, for under- and overfitting.
- Epoch sweep (5 to 50): held-out scores stop improving at 10 to 15.
- Mirrored band simulated by reversing the channel order: 70 % to 38 % without calibration.
- Run-to-run noise (2026-10-06, 5 seeds): a single run is good to about +-1.5 points on grip accuracy and +-0.03 in angle r. LDA has no randomness and repeats exactly.

## Not checked yet

- The recorder and the web app (cue plan, session writer, websocket). Exercised in every session, problems show up
  there quickly.
- `sync.py` (camera delay from the taps).
- The ESP32 firmware and `esp32_link.py`.
- No automatic run on GitHub yet: the tests only run when started by hand.
