# Testing and validation

What is checked automatically, what each check protects against, and what isn't checked yet. Model results
(accuracy, angle scores) are not tests; those are in `devlog.md` and `emg-reading/models/evaluations.csv`.

## Running the tests

From the repo root, with the emg-reading venv:

```
emg-reading/.venv/Scripts/python -m pytest tests
```

52 tests, about 10 s. Run them before committing changes to `ml/`, the filter or the quality checks. Tests that need
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
frame number and takes off the fixed camera delay (28 ms, `CAMERA_DELAY_S`).

| Test | Checks |
|---|---|
| `test_fit_removes_jitter_and_delay` | On 60 s of simulated paired, jittered, delayed frames, every fitted time is within 10 ms of when the frame was taken, with no frame-to-frame jitter left |
| `test_tap_delay_in_session_is_ignored` | A tap delay stored in `session.json` is not used, the taps were biased |
| `test_wrong_delay_shifts_everything` | The fit can't fix a wrong delay: it shifts every label. The delay has to be right |

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

### Haptics (`tests/test_haptics.py`)

| Test | Checks |
|---|---|
| `test_session_levels_keep_rest_quiet` | Haptic levels taken from the newest session (`calibrate.from_session`) are sane (rest above 0, max above rest) and, played back over that session's rest cues, leave every output off in at least 95 % of rest windows |

### Automatic evaluation (`tests/test_auto_evaluate.py`)

| Test | Checks |
|---|---|
| `test_completed_session_is_evaluated_before_training` | When a completed session is saved, the app starts "evaluate the newest session" on it (15 epochs), so every session gets an honest score before anything trains on it |
| `test_session_without_grips_is_skipped` | A session without grips (a sync check, for example) starts nothing |

### Mirrored band (`tests/test_mirror.py`)

| Test | Checks |
|---|---|
| `test_mirrored` | Which band orientations count as mirrored: left arm with the port towards the hand, right arm with the port towards the shoulder. Right arm with the port towards the hand is mirrored twice and counts as normal |
| `test_old_sessions_without_port_count_as_reference` | Sessions from before the port field count as the reference orientation |
| `test_live_network_reverses_mirrored_windows` | A network calibrated on a mirrored session reverses the channel order of live windows |
| `test_live_lda_reverses_mirrored_windows` | The same for an LDA model whose newest training session was mirrored |

### Armband reader (`tests/test_armband.py`)

| Test | Checks |
|---|---|
| `test_reader_streams_all_rows` | The reader process opens the (synthetic) board and passes every board row through; `read` returns the live channels |
| `test_close_and_reopen` | Closing stops the data and marks the band not connected; opening again streams again |
| `test_reader_dying_leaves_the_app_running` | A reader killed from outside (a library crash) leaves the app running, reads as not connected with no data, and a new reader can be opened |
| `test_context_manager_closes_the_reader` | Leaving `with Armband()` ends the reader process |

### Control layer (`tests/test_control.py`)

| Test | Checks |
|---|---|
| `test_start_needs_vote_n_steps_during_onset` | A grip starts only after `vote_n` steps in a row of the same grip above the threshold and the start level; a different winner restarts the count |
| `test_no_start_after_onset_until_effort_drops` | Once `onset_s` has passed without a decision, nothing starts until effort drops below the start level again |
| `test_open_class_start_stays_idle` | Picking the open hand keeps the hand idle and open, then blocks until effort drops |
| `test_lock_ignores_other_classes` | A locked grip doesn't switch when another class wins |
| `test_position_closure_holds_on_relax` | Position closure follows effort and doesn't go back down while relaxing holds the grip |
| `test_speed_closure` | Speed closure closes above the hold level and holds below it |
| `test_release_modes` | Release on the open hand, on rest, or on either |
| `test_angles_follow_closure` | Output angles sit at the right share of the way from relaxed to the grip's pose, and the spring only moves part of the way in one step |

### Protocols (`tests/test_protocols.py`)

| Test | Checks |
|---|---|
| `test_c_plan_holds_the_selection_once_per_round` | Each picked movement once per round in every posture, cue kind `custom`, rest after each, and the calibration block of every grip twice as plain holds |
| `test_c_plan_rounds_are_shuffled_and_seeded` | Same seed gives the same plan, and rounds differ in order |
| `test_c_plan_refuses_an_empty_or_unknown_selection` | No movements, or an unknown one, raises an error instead of an empty session |
| `test_plan_for_c_uses_its_own_fields` | `plan_for` uses `moves`, `c_rounds` and `c_move_s`, not B's rounds and movement time |
| `test_a_and_b_plans_unchanged` | Protocol A and B plans (seed 5) match a hash taken before protocol C existed, and the extra C fields don't change them |
| `test_library_entries_are_complete` | Every entry has a category, two-line cue text, sources and 20 angles; reused keys keep their old angles; keyframes start and end at the relaxed hand |
| `test_c_session_trains_without_leaking_into_protocol_a` | A protocol C session counts as good, loads in `ml.dataset`, and its `custom` cues get no grip label in the network or LDA windows, even for fist and pinch; only the calibration holds do |
| `test_d_plan_holds_every_activity_once_per_round` | Protocol D: every activity once per round, all in the first posture with the set length, rest after each, the grip calibration twice as plain holds, and `plan_for` uses `d_rounds` and `d_activity_s` |
| `test_d_session_gets_no_grip_or_rest_labels` | `nogrip` cues get no label at all in the network or LDA windows, rest included, since the wrist and arm move there; only the calibration holds do |
| `test_hand_warning_only_where_the_camera_labels` | The hand-out-of-view warning only shows during cues the camera labels and the setup break before them; not between sessions, during protocol D's activities or the rest after one |

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
