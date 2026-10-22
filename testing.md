# Testing and validation

What is checked automatically, what each check protects against, and what isn't checked yet. Model results
(accuracy, angle scores) are not tests; those are in `devlog.md` and `emg-reading/models/evaluations.csv`.

## Running the tests

From the repo root, with the emg-reading venv:

```
emg-reading/.venv/Scripts/python -m pytest tests
```

149 tests, about 16 s. Run them before committing changes to `ml/`, the filter or the quality checks. Tests that need
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
| `test_mirrored_band_does_change_the_output` | A mirrored band does change the output. Documents that the ring design doesn't cover mirroring, which is why mirrored sessions get their channel order reversed (`channels.mirrored`, tests in `test_mirror.py`); if it ever fails, the flip isn't needed |
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
| `test_completed_session_is_evaluated_before_training` | When a completed session is saved, the app starts "evaluate the newest session" on it (15 epochs), so every session gets an honest score before anything trains on it, and asks for a backup (once at startup, once for the session) |
| `test_session_without_grips_is_skipped` | A session without grips (a sync check, for example) starts nothing |

### Network job progress (`tests/test_job_progress.py`)

| Test | Checks |
|---|---|
| `test_epochs_of_a_single_training` | The Train tab's bar follows the `epoch k/n` lines of a training run, with the time left they print |
| `test_folds_count_towards_the_whole_job` | In an evaluation the bar covers all folds, not just the current one, and a new fold restarts the epoch count |
| `test_nothing_to_count_before_the_first_epoch` | Before the first epoch there is no share, so the page shows a moving bar instead of a wrong number |

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
| `test_mode_switch_keeps_streaming` | Mode commands for the delay test reach the reader; a board that refuses them (the synthetic one) keeps streaming |

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
| `test_far_windows_count_as_rest` | A window farther than `max_distance` from every trained grip can't vote for one, whatever its probability; 0 turns the check off |
| `test_rest_or_class_release` | `rest_or_class` lets go when rest wins `release_n` steps even with effort up, or on low effort; a far window counts as rest for it |
| `test_distance_space_inside_and_outside_a_class` | `gesture_model.distance_space`: training windows sit a few spreads from their class centre, a window away from every class many spreads |
| `test_blend_closes_with_effort_straight_away` | Blend mode: no votes to wait for, the hand closes as soon as rest fades from the smoothed probabilities |
| `test_blend_shape_follows_the_probabilities` | Blend mode: half fist, half pinch closes halfway between their poses; a one-window flicker only nudges the shape |
| `test_blend_locks_and_lets_go` | Blend mode: a grip leading long enough locks the shape, another grip winning doesn't change it, low effort for release_s lets go |
| `test_blend_lock_releases_when_the_open_hand_wins` | Blend mode: a locked shape lets go when the open hand wins release_n steps under rest_or_class, even with effort up |
| `test_blend_rest_keeps_the_hand_open` | Blend mode: effort while rest wins (or windows too far from every grip) doesn't close the hand |
| `test_trigger_clench_patterns_pick_the_grip` | Trigger mode: a clench held to `long_s` picks fist, 1, 2 and 3 short clenches then a pause pick pinch, tripod and key, 4 pick nothing |
| `test_trigger_point_comes_from_the_classifier` | Trigger mode: point winning `vote_n` steps with effort up picks point |
| `test_trigger_squeeze_closes_relax_holds_open_opens` | Trigger mode: effort above the hold level closes, relaxing holds the closure, the open hand opens |
| `test_trigger_double_open_flick_goes_back` | Trigger mode, release with open: two open flicks within `double_s` go back to picking a grip; one flick, or a one window drop inside a flick, doesn't; the flicks don't count as clenches afterwards |
| `test_trigger_flicks_too_far_apart_do_nothing` | Trigger mode: two flicks further apart than `double_s` leave the grip on |
| `test_trigger_still_gate_ignores_clenches_while_moving` | Trigger mode, `still_gate`: while the arm moves neither clenches nor point pick a grip; with the gate off moving changes nothing; once still the same clench counts |
| `test_trigger_pinch_clenches_moves_the_count` | Trigger mode, `pinch_clenches` 2: one clench picks nothing, 2, 3 and 4 pick pinch, tripod and key, 5 nothing |
| `test_still_limit_and_movement` | The still limit sits above a still accelerometer and below a moving one; no held cues means no limit (the gate never fires) |
| `test_rules_default_mapping_picks_like_trigger_mode` | Rules mode with the default mapping: a 0.7 s clench is fist, 1 and 2 short clenches pick pinch and tripod after the pause, 3 (the highest count) picks key straight away, held point picks point |
| `test_rules_hold_squeeze_open_and_release` | While holding: squeezing closes, relaxing holds, open opens; one point flick or a one window drop inside one does nothing, two flicks release and open the hand |
| `test_rules_never_switch_grips_while_holding` | A clench count or held point while a grip is held changes nothing, so a stray pattern can't switch grips with an object in the hand |
| `test_rules_on_sign_classes_and_library_poses` | Rules on a sign model's classes pick a protocol C library pose (cylinder) and release it |
| `test_rules_close_action_closes_while_its_trigger_stays_on` | A close rule held 0 s acts on the first step and closes the hand while its trigger stays on, even below the hold level; it stops when the trigger goes off |
| `test_rules_clench_with_a_rule_class_on_is_that_class` | Effort that comes with a rule's class (opening) is not counted as a clench |
| `test_rules_still_gate` | With the still gate on, nothing is picked while the arm moves |
| `test_rules_are_checked` | Mappings with an unknown action or how, a value out of range, a fractional count, two rules on the same pattern, two held rules on one trigger in one list or no trigger are refused, either list takes poses and actions; grip models get the clench default, sign models the sign default without clench, and classes the model doesn't have are left out |
| `test_rules_saved_as_one_list_split_by_action` | Mappings saved as one list before the split load with poses in the relaxed list and open, close, release in the holding list |
| `test_rules_one_trigger_held_in_both_lists` | Open held 0 s while holding doesn't stop open held 0.7 s from picking a pose while relaxed, and still opens the hand once a grip is held |
| `test_rules_pose_while_holding_switches_at_the_same_closure` | A pose rule in the holding list switches to that grip and keeps how far the hand is closed |
| `test_rules_higher_in_the_list_wins` | Two rules firing on the same step: the one higher in the list picks, whatever the trigger names |
| `test_rules_lock_freezes_the_hand_until_unlocked` | A lock rule while holding freezes the hand: squeezing, open, a pose and the release pattern change nothing until the lock trigger fires again or an unlock rule does; squeezing closes again after |
| `test_trigger_double_point_goes_back_and_open_only_opens` | Trigger mode, release with point (default): opening twice only opens, a point flick neither opens nor closes, two point flicks go back, and the point still held after that can't pick point until it drops out |

### Protocols (`tests/test_protocols.py`)

| Test | Checks |
|---|---|
| `test_c_plan_holds_the_selection_once_per_round` | Each picked movement once per round in every posture, cue kind `custom`, rest after each, and the calibration block of every grip twice as plain holds |
| `test_c_plan_rounds_are_shuffled_and_seeded` | Same seed gives the same plan, and rounds differ in order |
| `test_c_plan_refuses_an_empty_or_unknown_selection` | No movements, or an unknown one, raises an error instead of an empty session |
| `test_plan_for_c_uses_its_own_fields` | `plan_for` uses `moves`, `c_rounds` and `c_move_s`, not B's rounds and movement time |
| `test_a_and_b_plans_unchanged` | Protocol A and B plans (seed 5) match a hash taken before protocol C existed, and the extra C fields don't change them |
| `test_library_entries_are_complete` | Every entry has a category, two-line cue text, sources and 20 angles; reused keys keep their old angles; keyframes start at the relaxed hand (the last pose holds to the end of the cue) |
| `test_c_session_trains_without_leaking_into_protocol_a` | A protocol C session counts as good, loads in `ml.dataset`, and its `custom` cues get no grip label in the network or LDA windows, even for fist and pinch; only the calibration holds do |
| `test_d_plan_holds_every_activity_once_per_round` | Protocol D: every activity once per round, all in the first posture with the set length, rest after each, the grip calibration twice as plain holds, and `plan_for` uses `d_rounds` and `d_activity_s` |
| `test_d_session_gets_no_grip_or_rest_labels` | `nogrip` cues get no label at all in the network or LDA windows, rest included, since the wrist and arm move there; only the calibration holds do |
| `test_e_plan_holds_every_sign_once_per_round` | Protocol E has every screening sign and held wrist bend once per round, 3 rounds in the first posture and one fewer in the next, signs as `custom` and wrist bends as `nogrip` cues, rest after each, and `plan_for` passes its own fields |
| `test_e_session_stays_out_of_grip_training` | An E session has no grip holds, so it is not a good session for grip and network training, and its signs (fist, open, point, pinch among them) never get a grip label |
| `test_f_plan_holds_the_sign_set_once_per_round_in_every_posture` | Protocol F has each sign of the chosen set and both held wrist bends once per round in every ticked posture, as `custom` and `nogrip` cues with rest after each, and `plan_for` passes its own fields |
| `test_sign_sessions_train_their_signs_and_refuse_a_mix` | The Train tab's LDA on E and F sessions learns the signs they share, rest and the wrist bends (`notsign`); grip sessions train the 7 grips as before; a mix of the two is refused. Positions left out (`exclude`) get no label in sign and grip sessions, and rest can't be left out |
| `test_haptic_levels_from_a_sign_session` | A sign session has no grip holds, so `calibrate.from_session` takes the max level from its held signs, which lets the Control tab and the haptics run with a sign model |
| `test_sign_network_learns_the_signs_and_loads_safely` | `ml.networks.train_signs` on E and F sessions gives a network whose classes are the shared signs, rest and `notsign`, saved as kind `signs` with its sessions and base (without the left-out signs when some are excluded), and it loads with `weights_only` (numpy strings in the classes once broke that). Calibrating it on round 1 of the F session scores round 2 and saves a calibrated network that is still marked as signs, so the Control tab keeps it out |
| `test_f_session_stays_out_of_grip_training` | An F session has no grip holds and its signs, open included, never get a protocol A grip label |
| `test_hand_warning_only_where_the_camera_labels` | The hand-out-of-view warning only shows during cues the camera labels and the setup break before them; not between sessions, during protocol D's activities or the rest after one |
| `test_cue_cut_short_by_a_pause_is_bad` | A cue interrupted by a pause is marked bad and left out of training; the redo after the resume counts |

### Cursor (`tests/test_cursor.py`)

The Cursor tab's fixed map from channel levels to a dot (`emg-reading/cursor.py`) and its target task.

| Test | Checks |
|---|---|
| `test_map_points_each_channel_its_way` | Each channel alone moves the dot in its own direction round the ring (45 deg apart), as far as its level times the gain; relaxed is the middle |
| `test_even_activity_stays_in_the_middle` | The same level on every channel cancels out, so squeezing everything evenly doesn't move the dot |
| `test_rotation_mirror_and_limit` | Rotation turns the map, mirror reverses the channel order, and the dot never goes past `MAX_R` |
| `test_hit_after_home_and_dwell` | A target only shows after `home_s` in the middle, counts as hit after `dwell_s` inside, and the trial keeps its path with the 8 levels per step |
| `test_miss_on_timeout_and_every_target_once_per_block` | Doing nothing misses at `timeout_s`, and a block shows every target once |
| `test_leaving_the_target_resets_the_dwell` | Leaving the target restarts the dwell time and counts a second entry |

### Hub (`tests/test_hub.py`)

Sessions recorded on other computers and sent to the hub (`emg-reading/hub.py`), and who gets into the app. The
transfer tests run the hub app on a free local port and send over real HTTP.

| Test | Checks |
|---|---|
| `test_tokens` | A device token is recognised, a wrong one isn't, only its hash is stored, and a removed device stops working |
| `test_only_devices_and_this_computer_get_in` | Another computer without a token gets nothing; with one it gets the hub's and the Train tab's requests but not the armband, haptics or live websocket. The hub's requests want a token even from this computer |
| `test_tablet_mode_still_open` | Started with `--host`, the page still opens for a tablet as before, the hub's requests still want a token |
| `test_send_carries_on_after_a_dropped_connection` | A send that drops part way leaves nothing in `data/`, the next send carries on and the session arrives byte for byte; sending it again is refused, not overwritten |
| `test_unknown_token_refused` | The client reports the hub's refusal of a wrong token |
| `test_damaged_file_is_dropped` | A file that doesn't match the sender's hash keeps the session out of `data/` and is dropped so it's sent again |
| `test_wrong_offset_and_names_refused` | A chunk sent twice can't go in twice, and names that would leave the data folder are refused |
| `test_session_still_recording_is_refused` | A session without an end can't be moved in |
| `test_download_session` | A session downloaded from the hub (for a model's levels) arrives byte for byte |
| `test_back_up_copies_and_checks` | A session lands in the backup folder byte for byte through `.incoming` and is listed with its hashes; backing up again is fine, a different copy of the same name is never overwritten, and a session still recording is refused |
| `test_backup_worker_and_missing_folder` | The hub's backup worker does nothing without a folder, reports a folder that isn't there (an unplugged USB stick), and copies only ended sessions |
| `test_device_deletes_on_close_only_what_the_hub_and_backup_hold` | Sending never deletes anything. When its app closes, a device deletes a session only when the hub and its backup both hold it byte for byte. Kept: everything while the hub has no backup folder, a session not backed up yet, one a model here takes its levels from, one with files that were never sent, and one whose backup copy differs |
| `test_merge_keeps_both_sides` | Mappings and profiles are added by name, a name both sides have with different content stays as it is |

### Mac WiFi (`tests/test_mac_wifi.py`)

| Test | Checks |
|---|---|
| `test_mac_scan_line` | Lines of `airport -s` give the network name and signal, with spaces in the name and with or without a BSSID |

### Delay test (`tests/test_delay.py`)

The delay test's analysis (System > Dev, `emg-reading/delay.py`) on made-up streams with known delays, so a wrong number from
the tab is the setup's fault, not the analysis'.

| Test | Checks |
|---|---|
| `test_round_trip_found_for_each_switch` | Mode switches that show up 17 ms after they were sent are found once each and timed to one sample |
| `test_round_trip_ignores_emg_and_lost_switches` | Noise of a few hundred uV on a large offset is never taken for a switch |
| `test_screen_delay_from_frames` | Flashes filmed 60 ms late, with frames that sometimes arrive a frame later, measure 60 ms to one frame |
| `test_screen_delay_refuses_unseen_flashes` | A camera that doesn't see the flashes gives an error, not a number |
| `test_camera_delay_between_two_directions` | Changes to light and to dark far apart (88 and 15 ms on the MacBook) give their mean, not a number in the gap a median over all of them lands in |

### Recording setups (`tests/test_devices.py`)

Each session's angle labels are shifted by the camera delay of the setup it was recorded on (`emg-reading/devices.json`).

| Test | Checks |
|---|---|
| `test_committed_setups_are_complete` | The committed setups have unique ids, every field the start page and training read, and the default exists |
| `test_shift_follows_the_setup` | A session from before setups gets the default setup's shift, a setup not measured yet the 28 ms stand-in |
| `test_measuring_later_fixes_older_sessions` | Saving a measured shift for a setup changes the shift of sessions already recorded on it |
| `test_setup_belongs_to_its_computer` | A setup is offered on its own computer only, the name compared without case |

## Found by the tests

- **Camera glitches in the angle labels** (2026-10-06): some frames have a trained joint bent backwards past -60
  degrees (index PIP down to -175), which a finger can't do. MediaPipe lost or flipped the hand. 0.02 to 0.8 % of
  angle windows per session, 2.2 % in `2026-10-01_164337`. Fixed: `ml/dataset.py` treats those frames as not
  tracked (`IMPOSSIBLE_DEG`). The recordings themselves stay as they are.
- Abduction angles wrap round +-180 degrees on bent fingers. Not a problem, abduction isn't trained.

## Validated by experiment, not by tests

Checks that need training runs, too slow for the test suite. Numbers in `devlog.md`.

- Leave one session out: every score is on a session the model never trained on, LDA and network on the same windows.
- Practice against exam score (training sessions against held-out) in every evaluation, for under- and overfitting.
- Epoch sweep (5 to 50): held-out scores stop improving at 10 to 15.
- Mirrored band simulated by reversing the channel order: 70 % to 38 % without calibration.
- Run-to-run noise (2026-10-06, 5 seeds): a single run is good to about +-1.5 points on grip accuracy and +-0.03 in angle r. LDA has no randomness and repeats exactly.

## Not checked yet

- The session writer, the websocket and the page. Cue plans, the hand warning and cues cut short by a pause are
  covered in `test_protocols.py`. The page gets a headless load of every tab after web changes, by hand, not as a
  test.
- `sync.py` (camera delay from the taps).
- The ESP32 firmware and `esp32_link.py`.
- No automatic run on GitHub yet: the tests only run when started by hand.
