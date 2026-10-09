# Dev log

What changed, what was found, and why. Newest at the bottom. Details live in the linked docs and in the code;
this file is the short version so the reasoning isn't lost.

## 2026-09-28 to 09-29: hardware and first signal

- Armband battery (Akyga 980 mAh) recovered from deep discharge, case opened.
- Pads on ch 0, 2 and 5 had no connection. Masked in `channels.py`, then reflowed and unmasked on 09-30.
- ESP32 firmware: Arduino and ESP-IDF in one PlatformIO project, serial protocol v0.1, 7 PWM outputs.
- Host: connection check, live EMG plot with filter and spectrum view.
- The desktop stopped streaming from the armband (0 samples) while the MacBook worked. The armband's DHCP never
  gives the desktop an address, so it needs the static IP 192.168.4.2. See `hardware-status.md`.

## 2026-09-30: haptics, filter, recorder

**Haptics working.** Calibration measures each channel's rest and squeeze level, `haptics.py` scales the envelope
between them onto the 7 outputs. Protocol v0.2 adds a WiFi transport, USB keeps priority.

**Streaming after an armband restart.** The static IP alone gave 0 samples after the armband power-cycled. Switching
the desktop's WiFi adapter to DHCP for about 20 s and back to static fixes it, likely because the armband only streams
to a device that asked it for an address since boot. Confirmed once, the same symptom came back on 10-01 after the
armband restarted mid-session. Steps in `emg-reading/SETUP.md`.

**Filter retuned** on a guided 6 minute recording (`filtering.md`):
- Low cut 20 Hz to 40 Hz. Pulling on the band put most of its energy below 50 Hz. Band tug went from 9.7 to
  4.7 times a clench, gesture classification unchanged (97 vs 96 %).
- Mains notches at 60, 120, 180 Hz as `iirnotch`, streaming filter that keeps its state. Identical to filtering a
  whole recording at once.
- Common average reference and a pad contact check for the haptics only. CAR removes real shared muscle signal, so
  the models never get it.

**Recorder web app** (step 1 of the ML plan). FastAPI on the desktop, plain HTML/JS. Records EMG, all armband rows
and webcam hand tracking (MediaPipe, 20 joint angles) on one clock, with cued gestures in three arm postures, sync
taps, signal quality warnings and one folder per session (`session.py`). The existing scripts became tabs in the app
(signals, calibration, haptics, connection), with their logic split out so both use the same code.

Findings while building it:
- The armband runs at 499.2 to 499.5 Hz, not 500. Over 15 minutes, assuming 500 Hz drifts 1.3 s against the camera.
  Sample times come from a line fitted against the package number (`session.sample_times`).
- The armband's timestamp row is the PC's clock at arrival, so EMG and camera already share a clock.
- MediaPipe's left/right label is the person's real hand on an unmirrored webcam image. An early version swapped it
  and tracked the wrong hand.
- Hum check: good pads stayed under 46 uV of 60 Hz line, a pad losing contact read 120 to 180 uV. Warning at 80 uV,
  on a 5 s median because muscle also has power at 60 Hz.

## 2026-10-01: camera, sync, gesture model

**Camera ran at 15 fps, not 30.** DirectShow dropped every other frame once hand tracking ran between reads. Media
Foundation keeps 30 fps but took 31 s to open the Brio 100. Turning off its hardware transforms brought that to
1.2 s (`camera.py`). The slow-camera warning only fired below 15 fps, which is why it went unnoticed.

**Camera delay measured from sync taps** (`sync.py`). The accelerometer sees each slap as a jump between readings,
the camera sees the palm stop. The accelerometer only updates once per 10-sample packet (50 Hz), not at 500 Hz.
EMG spikes from the taps were too inconsistent to use. First check: 68 ms, 6 of 6 taps.

**Camera delay wandered between sessions:** 70, 96, 126 and once 867 ms with all taps agreeing. Frames queued up
whenever tracking fell behind, and at 30 fps in and 30 fps out the queue never drained. A separate thread now grabs
frames and tracking takes the newest, so a slow frame is skipped instead of delayed.

**Protocol A built** (`gesture_model.py`, Train tab): RMS, mean absolute value, waveform length, zero crossings and
slope sign changes per channel on 200 ms windows every 50 ms, shrinkage LDA, majority vote. The first second of each
cue is dropped, the EMG changed 0.2 to 1.4 s after a cue. Live prediction uses the same feature code and matched the
batch evaluation on every window of a recorded session. Training on two sessions takes about 1 s, one prediction
0.2 ms.

**First session, 11 gestures: 74 % within the session** (train on 2 repetitions, test the third).
- Accelerometer features broke on an unseen posture (23 %), the model learned the posture instead of the gesture.
  Left off.
- Other classifiers (SVM, logistic regression) scored the same as LDA, and extra features added about 3 points.
  The limit is the signal and the data, not the classifier or the filter.
- More data still helped: about +5 points per extra repetition.

**Weak gestures were gentle ones.** With the arm forward or raised, key pinch, pinch and tripod rose only 1.4 to 1.7
times above rest. Cues now ask to hold each gesture firmly at medium effort. On the next session key pinch went from
12 to 57 %, tripod 57 to 75 %, rest taken for a gesture 3 % to 0.6 %.

**Moving the band is the big problem.** Training on one session and testing on the other: 39 % (11 gestures). The
band had rotated by about one electrode between sessions (shifting the channels by one recovered part of it), and
the first session had no firm-hold cue. The fix is a consistent band position plus a short top-up recording each
time the band goes on. Rotation augmentation and per-session normalisation still to test with 3+ sessions.

**Live test looked like garbage** because every saved model had been trained on the first session only. The Train
tab kept an old tick selection, so the newer session was left out. Fixed: new sessions start ticked, the model list
shows which sessions each model came from.

**Confidence threshold.** Below 0.8 probability the output says "unsure" instead of guessing. Within one band
position, wrong outputs fell from 13 to 4.4 % of windows (7 grips). After a band move the model is confidently wrong,
so the threshold doesn't fix that.

**Gesture set cut from 11 to 7** (`GESTURE_SET` in `gestures.py`): rest, open, fist, pinch, tripod, key, point, the
common prosthetic grips. Thumbs up, peace and OK are signs rather than grips, hook scored 51 % and was often taken for peace. Merging
pinch, tripod and OK did worse. Sessions got shorter (about 9 instead of 13 minutes).

Where it stands (two sessions, one person). Corrected later the same day: plain accuracy is mostly a rest score, rest
is 46 % of all windows, so balanced accuracy (every class counted equally) is the headline:

| | 11 gestures | 7 grips |
|---|---|---|
| Same session, unseen repetition, balanced | 61 to 62 % | 66 to 80 % |
| Same, grips only (rest left out) | 58 to 59 % | 61 to 77 % |
| Same, plain accuracy (first reported) | 73 % | 81 to 87 % |
| Other session (band moved), balanced | 17 to 22 % | 30 to 34 % |
| Other session, grips only | 11 to 16 % | 23 to 26 % |
| Other session, plain accuracy (first reported) | 34 to 42 % | 51 to 52 % |
| Wrong grip shown, same position, 0.8 threshold | 4.7 % | 4.4 % |

Across a band move the grips are barely above chance (14 % for 7 grips), the plain number was mostly rest being
right. Most of the jump from 11 to 7 gestures comes from dropping the four weakest, not from a better model.
Accuracy on the training data is only 4 to 6 points above held-out data, so the model isn't overfitting.

Latency from muscle change to a steady correct output on recorded gestures: median 290 ms (450 ms with the
threshold), over the 200 ms target. During free finger movement the output still changes about 3 times a second, so
in everyday use it would trigger grips that weren't meant. Not yet measured: false grips per minute during normal
activity.

**Recalibrating after the band moves.** Train on one session, test on repetitions 2 and 3 of the other (7 grips):

| | old to new | new to old |
|---|---|---|
| Old session only | 52 % | 52 % |
| Adapt without labels (move averages towards its own confident guesses) | 60 % | 51 % |
| Shift everything by the new rest level (about 10 s of rest) | 57 % | 58 % |
| 1 repetition of the new session alone | 85 % | 75 % |
| Old session pooled with that repetition | 74 % | 69 % |
| Adaptive LDA: old spread, class averages from that repetition | 83 % | 73 % |

These are plain accuracy. Redone later with balanced accuracy on three sessions (each in turn the new one, tested on
its repetitions 2 and 3): other sessions only 25 / 52 / 28 %, one repetition of the new session alone 58 / 76 / 69 %,
other sessions pooled with that repetition 35 / 66 / 39 %. Same conclusion, lower level.

One repetition at the new position (about 3 minutes) gets close to a full session. With only one other placement,
old data adds nothing and pooling it hurts. Adapting without labels follows its own mistakes. Old sessions should
only start to help once a model learns features that survive the band moving, which needs many placements (a
network trained across 5+ sessions with an LDA head refitted per fitting is the idea to test then).

**Aligning the input instead of retraining.** Old LDA frozen, the new session's 8 channels transformed to look like
the old ones, fitted on one calibration repetition:

| | old 1 to new 2 | old 2 to new 1 |
|---|---|---|
| No transform | 53 % | 54 % |
| Best channel rotation | 53 % (no shift) | 63 % (+2) |
| Rotation and a gain per channel | 60 % | 38 % |
| Full 8x8 channel mix, matching each grip's channel covariance | 54 % | 70 % |

Inconsistent and below plain retraining on the same repetition (75 to 85 %). The gains likely mixed up band
movement with the effort difference (session 1 had no firm-hold cue). The covariance matching is only a stand-in;
with a network the alignment layer could be trained on the classification itself. Retest with consistent sessions.

**Is the data usable for individual finger flexion (protocol B)?** Partly. The camera angles are believable (in a
held point the index reads -4 degrees, the other fingers 112 to 142; in a fist every finger 144 to 160) and tracking
covered 93 % of the session. But the fingers hardly move on their own: held grips are 7 fixed shapes, and in the 90 s
of free movement per session the fingers moved together (middle and ring 0.87, ring and pinky 0.93). A model would
learn 7 hand shapes, not single fingers. Sessions now add a single-finger block in every posture: each finger bent
and straightened 3 times on its own (8 s), then all fingers one after another. About 3.5 minutes more per session
(12 instead of 8.6). Sessions recorded before this (up to `150735`) don't have it.

**Smaller fixes the same day:**
- An error in the armband loop ended its thread for good, which looked like a disconnect. The loop now keeps going
  and shows the error.
- Hand tracking dropped when the wrist reached the bottom 5 % of the picture, the tracker finds the palm first. The
  warning bar now says which edge the hand is near.
- Firefox kept an old `hand.js` after an update and the camera overlay froze on gesture cues. Page files are sent
  with `Cache-Control: no-cache`, and one failing drawing step can no longer stop the others.
- Practice runs: the same session flow with nothing saved, for checking the camera and setup.
- Battery in the app: level in the top bar (yellow below 30 %, red below 15 %), and on the Connection tab the drain
  per minute, time left and full sessions left, from a straight line over the last 15 minutes. Streaming drained
  about 0.5 % a minute.
- The dropped-samples figure in the summary and `session.json` counted every drop since the app started, not per
  session (a sync check with 0 real drops showed 62). Fixed; earlier session files still have the inflated number,
  count drops from the package numbers instead.
- With the camera read in its own thread, frames arrive in pairs (two within 10 ms, then about 55 ms), still
  30 fps on average. Arrival times jitter by up to a frame, which widens the tap spread (69 ms on the first check,
  delay 77 ms). Fixable offline with a straight-line fit of time against the grab number, like the EMG clock.
- Calibration max levels vary 20 to 100 % between single clenches (effort isn't controlled), rest levels within
  about 1 uV. Only affects the haptics, left as is.

**Correction: accuracy was mostly a rest score.** Rest is 46 % of the windows (one between every two grips), so
always answering rest already scores 46 %, not the 14 % chance level quoted earlier. On session 2, unseen
repetition: plain 87.2 %, balanced 80.1 %, grips only 76.9 %. With three sessions, leave one session out: balanced
35.7 %, grips only 27.3 %, plain 54.3 %. The evaluation now reports balanced and grips-only accuracy first, and the
model list in the app shows balanced. The tables above are corrected.

Other checks on the same data:
- Test splits: random windows 90.6 %, unseen repetition 87.0 %, unseen session 52.4 % (plain). Neighbouring
  windows leak only a little within a session, the big drop is the band placement.
- Shrinkage hardly matters here: Ledoit-Wolf picked alpha 0.001, accuracy 86.4 vs 87.0 % same placement, 40.0 vs
  40.5 % across. The features are very redundant (RMS and mean absolute value correlate 0.999, 18 of 40 directions
  hold 95 % of the variance), but 7600 windows are enough to estimate the covariance. It may matter for short
  calibrations, not checked yet.
- The three repetitions of one session scored 83, 85 and 93 %, so single figures carry about 5 points of noise.

**Extended features are the new default** (`ml/gesture_model.py`, `--base-features` or the Train tab checkbox for
the old set). On top of the 5 features per channel: 4 autoregressive coefficients per channel (the shape of the
frequency content) and the 28 channel-to-channel correlations (the pattern across the band), 100 features in all.
Live prediction 0.3 ms, live and batch agree on every window, saved models keep the feature set they were trained
with. Three sessions, balanced accuracy:

| | base, 40 features | extended, 100 features |
|---|---|---|
| Calibrated on 1 repetition, tested on the other 2 | 67.9 % | 76.7 % |
| Calibrated on 2 repetitions, tested on the third | 72.1 % | 80.4 % |
| Rest taken for a grip, 1 / 2 repetitions | 4.4 / 2.8 % | 4.4 / 3.6 % |
| Wrong grip shown at 0.8, 1 / 2 repetitions (answers given) | 7.9 / 5.5 % (73 %) | 9.7 / 7.1 % (88 %) |
| No calibration (leave one session out) | 35.7 % | 45.5 % |
| No calibration, rest taken for a grip | 13.7 % | 26.2 % |

With calibration the extended set is clearly better: more grips right, about the same share of wrong answers, and
it answers far more often. Without calibration it finds more grips but fires twice as often at rest, one more
reason calibration is required. A 300 ms window added another 1.5 to 2 points but costs 50 ms of delay, not adopted.

Target for protocol A, defined: 85 % balanced accuracy on a session the model hasn't seen, after a 2-repetition
calibration at the start of that session. Best so far 80 % (one session reached 86 to 88 %).

**Model code moved to `ml/`** at the repo root, so `emg-reading/` stays about recording. `gesture_model.py` is the
first file there; the general network, dataset building and evaluation go there too. Saved models and data stay in
`emg-reading/`. Run with `python -m ml.gesture_model` from the repo root. Checked after the move: same evaluation
numbers, all saved models load, training and live prediction work in the app.

Data policy for the general network: all good data from all users and calibrations, where good means a completed
session (not stopped early).

**Architecture decided for the general model:**
- One network trained on all good data (all users, sessions and calibrations), raw filtered EMG in, a grip head and
  a finger-angle head out.
- The 8 electrodes treated as a ring: convolutions wrap around the channels, so rotating the band by whole electrodes
  only shifts the features. Aimed at the one-electrode rotation measured between sessions.
- A per-person 8x8 transform in front of the frozen network, trained on that person's calibration, for the rest of
  the placement and personal differences.
- Continual learning as the end goal: it learns from each new completed session without getting worse. Each session
  is tested before it is learned from, updates replay earlier data so nothing is forgotten, and an update only goes
  live if a fixed set of never-trained sessions doesn't get worse. Versions are kept for rollback. No learning from
  unlabelled live use.
- Porting to the ESP32 and the prosthetic's input format are out of scope for now.

**First ring network** (`ml/ringnet.py`, `ml/dataset.py`, `ml/train_ringnet.py`). 44,376 weights: 4 ring
convolution blocks over (electrode, time) with wrap-around, mean and max over the ring, a grip head (7) and a
finger-angle head (9 reliable joints), and the per-person 8x8 transform with gains in front. Rotating the input by
1 to 7 electrodes changes the output by under 1e-8, rotation independence holds by construction. Trained class
balanced with random gain and noise, on the GPU about 30 s per fold. Windows and labels are cached per session in
`ml/cache/`; camera angles are aligned with a straight-line fit of frame time against frame number minus the
measured delay (fist windows read 76 degrees mean finger flexion, open 10 to 16, in every session).

Leave one session out, 3 sessions, so each network learned from only 2. Balanced accuracy, grips only, rest taken
for a grip, wrong grip shown at 0.8, answers at 0.8:

| | balanced | grips | rest false | wrong | answers |
|---|---|---|---|---|---|
| LDA (extended), no calibration | 44.1 % | 39.4 % | 27.3 % | 25.9 % | 70 % |
| Network, no calibration | 53.0 % | 46.0 % | 5.1 % | 10.7 % | 60 % |
| LDA on the 2 calibration repetitions | 80.4 % | 77.7 % | 3.6 % | 7.1 % | 87 % |
| Network + transform from 2 repetitions | 76.8 % | 74.7 % | 10.7 % | 2.2 % | 54 % |
| Network + transform + grip head | 81.5 % | 79.8 % | 8.4 % | 4.7 % | 78 % |

Without calibration the network is already better than LDA and fires at rest far less (5 vs 27 %). Calibrated it
matches LDA and shows the fewest wrong grips so far, but takes rest for a grip more often (8 to 11 %), likely from
the equal class weighting during calibration. Finger angles, 9 joints: 16.5 degrees error and r 0.38 without
calibration, 13.0 degrees and r 0.60 with the transform (target under 15 degrees and r 0.8). Training ran at low
priority during a recording, the camera stayed at 29 fps.

**Fourth session (`164337`, first with the single-finger block).** Scored first with models that never saw it:
no calibration LDA 45 %, network 61 %; calibrated LDA 79 %, network with transform and head 83 % (balanced). Then
leave one session out over all four, so each network learned from three:

| | 3 sessions | 4 sessions |
|---|---|---|
| LDA, no calibration | 44.1 % | 48.0 % |
| Network, no calibration | 53.0 % | 61.4 % |
| LDA on the 2 calibration repetitions | 80.4 % | 80.1 % |
| Network + transform + grip head | 81.5 % | 80.6 % |
| Network, no calibration: rest taken for a grip / wrong grip at 0.8 | 5.1 / 10.7 % | 4.3 / 4.8 % |

One more training session moved the uncalibrated network 8 points, LDA 4. Calibrated, both sit near 80 %. Finger
angles didn't improve (13.9 degrees, r 0.56) despite the finger block. That session's camera delay is shaky: taps
read 174, 163, 242 and 107 ms (168 ms used), which blurs the angle labels by up to about 70 ms.

## 2026-10-02: first next-day session

**Session `2026-10-02_142500`, scored first with models trained only on 2026-10-01** (balanced accuracy):
no calibration LDA 75.1 %, network 79.4 % (rest taken for a grip 0.2 %, wrong grip at 0.8 2.6 %); calibrated on 2
repetitions LDA 86.1 %, network with transform and head 85.8 %. Both pass the 85 % target on this session. The band
position (7.75 cm from the elbow crease) was close to the last session of the day before (7.5 cm), but the pads are
pushed under the band by hand without the casing, so the electrode positions vary between sessions more than the
placement note shows. The note only describes the band.

Leave one session out as sessions were added:

| balanced | 3 sessions | 4 sessions | 5 sessions |
|---|---|---|---|
| LDA, no calibration | 44.1 % | 48.0 % | 56.7 % |
| Network, no calibration | 53.0 % | 61.4 % | 68.9 % |
| LDA on the 2 calibration repetitions | 80.4 % | 80.1 % | 81.3 % |
| Network + transform + grip head | 81.5 % | 80.6 % | 82.7 % |

Without calibration the network beats LDA on all five held-out sessions (by 4 to 20 points) and takes rest for a
grip less often (3.3 vs 9.2 %), and it gains about 8 points per added session so far. Calibrated, it is ahead on four
of five, two sessions reach 85 %. Finger angles stay at about 13.6 degrees and r 0.58; the label timing (shaky camera
delay) is the likely limit. Camera delays since the frame reader moved to its own thread read 77 to 168 ms, higher
than the 68 to 96 ms before it, worth checking with the whole-session measurement.

**Deep learning tab in the app.** Evaluate (all sessions, or only the newest before it is trained on), train one
network on all completed sessions, calibrate it on chosen repetitions of a session, and run a calibrated network
live with the same threshold, vote and Record tab view as LDA. Jobs run `ml/train_ringnet.py` and `ml/networks.py`
as a separate low-priority process, so training can't stall the armband or camera loops during a recording. Saved
networks go to `emg-reading/models/networks/`; a calibration on a session the network trained on is flagged, its
score is optimistic (95 % on `142500` against 86 % when it was held out). Live inference runs on the CPU, one
window per 50 ms step.

**Hum stops the cue.** Channels 5 and 6 hummed on and off while setting up a session (pads loose under the band).
During any cue whose data is trained on (grip, rest, finger, free), the recorder now checks the 60 Hz line every
second; over 80 uV on any channel in both of the last 2 seconds marks the cue bad (event reason `hum`, with the
channels), pauses and names the pads. Continue redoes the cue. If the hum starts in the first 2 s of a cue, the cue
before is marked and redone too, since the check looks back 2 s. The warning bar keeps its slower 5 s median.

## 2026-10-05: live finger angles

Hum on ch 5 to 7 fixed before session `112731` (14.7 min, completed, no hum redos). The two sessions from the evening
of 10-02 were stopped early, so 6 good sessions now. Live, the calibrated network felt good apart from key pinch,
pinch and point getting mixed up: key and point share the three curled fingers and differ only in the index, the thumb
position comes mostly from hand muscles the band can't see. Key pinch stays in the set (common in daily use).

**Live finger angles in the Deep learning tab.** The network's angle head (9 base and middle joints) now runs live
with the grips, smoothed (half weight on the newest 50 ms window), drawn as a wireframe hand you can turn, over the
camera-tracked hand, with the same match score as the Record tab's pose match. Unpredicted joints are drawn relaxed.
Expect it to follow the hand loosely: offline r 0.58, target 0.8. The confidence threshold can now also be set on
that tab.

**Hand pictures drawn as a mirror.** The hand pictures (cue, pose view, model view, angle view) are drawn like a
mirror, the same as the camera preview: palm towards you, and a left hand's reflection has the shape of a right hand,
so the left hand is drawn unmirrored and the right hand mirrored (`mirrored()` in `web/app.js`). Drawing the true left
hand was correct but less intuitive to copy. Also fixed: the server starts with the right hand after every restart and
the page took that over the saved choice, so after a restart the camera only looked for a right hand. The page now
sends the saved hand on load and saves it as soon as it changes; the Deep learning tab has its own hand selector for
the same setting.

**Camera tracking quality** (6 completed sessions, held grips from 1 s after the cue). The hand is found in about 100 %
of frames during cues, so detection isn't the problem. During a hold the angles jitter under 1 degree frame to frame
(about 1.5 degrees std). Problems:
- Straight fingers read bent: open hand gives middle, ring and pinky PIP 23 to 41 degrees and index DIP 42. The offset
  stays the same across sessions and postures, so it shifts the labels but doesn't hurt r.
- Base knuckle flexion of the open hand depends on posture: -17 to +17 degrees, lowest with the arm held forward.
- Thumb MCP barely moves: 7 to 31 degrees across all grips, 25 degree range during the thumb cue. It is in the
  9 trained joints but carries almost no signal.
- Abduction is meaningless when the finger is bent (fist ring abduction -85).
- Single fingers come through: in each `flex_<finger>` cue the cued finger's PIP covers 40 to 57 degrees, its
  neighbours 14 to 27 (part of that is real coupling, ring and pinky the most).

**Mirrored band.** The ring network ignores whole-electrode rotations but not a mirrored electrode order, which is
what a band turned upside down gives (and the same band orientation on the other arm, relative to the muscles; both
together cancel out). All 8 completed sessions are left arm, charging port facing the shoulder. Simulated by reversing the
channel order of the held-out session, leave one session out over 6 sessions, balanced accuracy, normal / mirrored:

| | normal | mirrored |
|---|---|---|
| Network, no calibration | 70.5 % | 38.1 % |
| Network + transform | 81.2 % | 71.6 % |
| Network + transform + grip head | 83.2 % | 80.1 % |
| LDA, no calibration | 59.5 % | 38.0 % (rest taken for a grip 38.5 %) |
| LDA on the 2 calibration repetitions | 82.0 % | 82.0 % |

Without calibration a mirrored band costs the network 32 points on every session. Transform and grip head recover
most of it, the transform alone doesn't get to a full reversal in 300 steps. A real other arm will do somewhat worse
than this (different arm, not just mirrored). The setup form now records which way the charging port faces
(shoulder or hand, `port_facing` in the settings), filled in as shoulder for every earlier session. With the arm,
that is enough to reverse the channels before the network for a mirrored band, which would cost nothing.

**Why the predicted hand is poor.** The offline angle score (r 0.58) was measured on held grips of the third
repetition only, so it mostly scores grip shape. Scored on moving windows (finger and free cues), leave one session
out over 6 sessions, the network reaches r 0.28 without and 0.34 with the usual calibration (transform + grip head on
2 repetitions). Thumb MCP r 0.04, nothing to learn there. Variants, one change each (moving r, no cal / cal):

| | no cal | cal | held r cal | grips balanced cal |
|---|---|---|---|---|
| current network | 0.28 | 0.34 | 0.58 | 78 % |
| labels with a 30 ms delay instead of the taps | 0.27 | 0.32 | 0.58 | 79 % |
| calibrated on the first posture block, tested on the others | 0.27 | 0.26 | 0.45 | 64 % |
| moving windows weighted 3x in the angle loss | 0.33 | 0.35 | 0.57 | 78 % |
| 1 s of EMG, ring blocks then a GRU over time, moving 3x | 0.26 | 0.31 | 0.69 | 82 % |

- Within one session a plain ridge regression on the LDA features reaches r 0.45 to 0.60 (train on two posture
  blocks, test on the third), against about 0.30 across sessions. The shift between sessions is the main loss, and
  the calibration only sees held grips, never moving fingers. Corrected 2026-10-06: the 0.45 to 0.60 is the score on
  held grips (it mostly measures which grip is held). On moving fingers the same split gives 0.05 to 0.32 (6
  sessions with the finger block), so the session shift is not the main loss for moving fingers.
- The longer window helps held grips and the grip classifier (+4 points) but not moving fingers.
- Calibration fitted in one posture doesn't carry to the others.
- Thumb MCP stays in: the thumb cue was done with too little movement, not a tracking limit.

**Scope of protocol B (Noah).** Recordings and models keep the full joint set. Amputees likely won't control all
joints independently (the intrinsic hand muscles are gone, so thumb opposition and abduction are hardest), but fewer
targets (hand closing, index, thumb) can be derived from the same recordings later without recording again.

**Protocol changes for the angle data** (from the next session on). Finger cues 8 to 12 s and now ask to stop
halfway for a second before bending fully, since the grips are all end positions and half-bent fingers were barely
recorded. The thumb cue asks for the full range (across the palm and out as far as it goes). Free movement default 30
to 60 s per posture. A 3-posture session is now about 15.4 min instead of 12.9. Sessions up to `2026-10-05_112731`
have the old cues. The setup form has a Preview cues button that steps through every distinct cue of the planned
session with its hand picture, to go through with the person before recording.

`train_ringnet.py` (and the Deep learning tab's evaluation) now reports angles on moving fingers and on held grips
separately; the dataset keeps each window's cue kind (cache version 2).

**Camera delay.** The taps are unreliable: stored delays include -147 and -132 ms (impossible) and up to 552 ms.
Gyro rotation against camera hand rotation over the whole session gives 15 to 50 ms, against the camera's 96 to 168
ms from the taps. But for the labels, the alignment that best predicts the angles from EMG is about 100 to 150 ms
(ridge sweep, r flat within 0.03 from 50 to 200 ms), since it includes the delay from muscle activity to finger
movement. A 30 ms label delay scored worse than the taps. So delay is not what limits the angles, and the
replacement should fit the label delay to the EMG over the whole session instead of measuring the camera alone.

**Evaluation log.** Every `train_ringnet.py` run (both evaluate buttons on the Deep learning tab) adds one row to
`emg-reading/models/evaluations.csv`: date, code commit (+ for uncommitted changes), which sessions were held out,
session count, epochs, and every score in the summary. It also scores the network on its own training sessions, so
the gap to the held-out score shows whether the epoch count is too high or too low.

**Epoch sweep** (7 sessions, leave one session out, 5 to 50 epochs in steps of 5, all rows in `evaluations.csv`).
Balanced accuracy on the training sessions rises the whole way (83 % at 5 to 96 % at 50), the held-out session
doesn't: no calibration 67.9 % at 5, 70.0 % at 10, then 70.0 to 71.5 % from 15 on; calibrated (transform + head) 83.5
to 84.6 % at every count. Moving-finger r 0.22 at 5, 0.27 to 0.29 from 15 on. Past 15 to 20 epochs it only memorises
the training sessions, without getting worse on new ones. Default stays at 15 (20 is within noise and takes 9 instead
of 7 minutes per evaluation). Calibrated LDA is 82.2 % in every row, as it should be, the splits were the same.

**Effort head.** The random overall gain in training (about +-35 %) teaches the grips to ignore effort, but it also
taught the angle head to ignore it, while a finger bent halfway and fully may differ mostly in how hard the same
muscles fire. The network now has a second angle head on the same features (45,537 weights instead of 44,376):
each training batch holds every window twice, with and without the overall gain (same per-channel gain and noise),
in one pass. Grips and the old angle head learn from the scaled copy, the effort head from the unscaled one. Both
run live; the Deep learning tab draws them side by side over the camera's hand (orange angle head, green effort
head), each with its own match score, and the evaluation scores both (`effort` rows and columns). Networks saved
before this load without the effort head. The evaluation also scores the angles on the training sessions' moving
windows, as the practice score for angles. Network training can now use any ticked subset of sessions, also one.

First result, 8 sessions, 15 epochs: the effort head scores the same as the angle head on every angle row (moving
fingers r 0.27 / 0.30 without / with calibration, held grips 0.57 / 0.61, within 0.01 everywhere). Effort within the
+-35 % the augmentation covers isn't what the angles are missing. Only 3 sessions have the halfway holds yet, so
recheck with more. Grips with 8 sessions: no calibration network 74.2 % (LDA 62.2 %), calibrated 84.1 % (LDA 82.1 %).
More telling: moving-finger r on the network's own training sessions is only 0.33, against 90 % balanced for grips
there. The network can't follow moving fingers even on data it learned from, so for angles it underfits, the session
gap isn't the only problem. Likely cause: held grips are most of the angle windows, so the angle loss is mostly a
grip-shape loss. Next for angles: weight the moving windows, a time-aware angle head.

**Angle head experiment** (8 sessions, leave one session out, 15 epochs, scratch script, not in the code). Moving
windows are 10 to 15 % of the angle windows in the sessions before the finger block and 20 to 43 % after. Moving-finger
r, mean over the 9 joints, all exam scores on the second half of the held-out session's moving windows:

| | own training sessions | held out, no cal | cal on grips | cal on grips + first half moving | grips balanced |
|---|---|---|---|---|---|
| current network | 0.30 | 0.23 | 0.29 | 0.30 | 73.7 % |
| moving windows weighted to equal the rest | 0.37 | 0.29 | 0.34 | 0.35 | 73.7 % |
| hidden layer in the angle head (MLP) | 0.34 | 0.25 | 0.28 | 0.30 | 73.7 % |
| angle head on the 64 x 8 table before ring pooling | 0.39 | 0.25 | 0.30 | 0.31 | 73.8 % |
| MLP + weighted | 0.41 | 0.31 | 0.33 | 0.34 | 73.5 % |
| before pooling + weighted | 0.46 | 0.30 | 0.34 | 0.35 | 73.5 % |
| ridge on the extended LDA features | 0.31 | 0.21 | | | |

- Weighting the moving windows is the one clear gain: +0.06 to +0.07 on every column, grips unchanged.
- Reading the table before ring pooling fits the training sessions best (0.46 with weighting) but barely moves the
  held-out score, so the extra is mostly memorised placement. Not worth giving up rotation independence for now.
- Calibrating on moving windows as well adds only 0.01 over calibrating on grips.
- Ridge across sessions fits no better than the network (0.31). The 0.45 to 0.60 quoted earlier was within one
  session.
- Per joint (weighted): PIPs and middle/ring knuckles 0.36 to 0.45 calibrated, index knuckle 0.24, thumb 0.14.
- Even the best variant reaches only 0.46 on its own training data, still far from 0.8. The gap to the held-out score
  (0.46 against 0.30) is now the session shift; only 4 of 8 sessions have the finger block.

**Moving windows weighted in training** (`angle_weights` in `train_ringnet.py`): in each session the moving-finger
windows count as much in the angle loss as all its other angle windows. Full evaluation, 8 sessions, 15 epochs,
before / after: moving fingers r 0.27 / 0.33 without calibration, 0.30 / 0.35 calibrated, 0.33 / 0.39 on the training
sessions; held grips 0.61 / 0.59 calibrated (less weight on them, small cost); grips 84.1 / 83.9 % calibrated, within
noise.

## 2026-10-06: tests

**Run-to-run noise.** The same evaluation (8 sessions, 15 epochs, moving windows weighted) with 5 random seeds, all
rows in `evaluations.csv`. Mean, standard deviation, full range:

| | mean | sd | range |
|---|---|---|---|
| Network, no calibration, balanced | 73.8 % | 0.55 | 73.0 to 74.3 |
| Network + transform + grip head, balanced | 84.8 % | 0.54 | 83.9 to 85.2 |
| Network + transform, balanced | 82.6 % | 0.23 | 82.4 to 83.0 |
| Rest taken for a grip, calibrated | 3.5 % | 0.30 | 3.1 to 3.7 |
| Moving fingers r, no calibration / calibrated | 0.32 / 0.35 | 0.01 | 0.30 to 0.33 / 0.33 to 0.36 |
| Held grips r, calibrated | 0.59 | 0.01 | 0.58 to 0.60 |
| LDA, both | 62.2 / 82.1 % | 0 | (no randomness, as it should be) |

So within one set of sessions, a single run is good to about +-1.5 points on grips and +-0.03 in r: a change has to
beat that to count. The moving-window weighting (+0.06 r) is real. The calibrated network sits at 84.8 % on average,
the 85 % target is inside the noise. This is only the training randomness; which sessions there are moves the numbers
far more (single held-out sessions range from 77 to 88 %).

**Unit tests** in `tests/`, run with `emg-reading/.venv/Scripts/python -m pytest tests` (pytest added to the
requirements), every test and what it protects against in `testing.md`. 41 tests, 7 s: live filter against whole
sessions, live network prediction against the evaluation, rotation independence (and that a mirrored band is still
not covered), calibration only touching the transform and grip head, the accuracy scores on examples worked out by
hand, the evaluation log, the grip labels checked against each session's event log, camera frame timing, the hum,
dropped-sample and contact checks. Tests on recorded sessions and torch skip where those are missing.

**Camera glitches in the angle labels**, found by the tests: the trained joints sometimes read bent backwards past
-60 degrees (index PIP down to -175), which a finger can't do; MediaPipe lost or flipped the hand for those frames.
0.02 to 0.8 % of the angle windows per session, 2.2 % in `2026-10-01_164337`. They are still in the training labels.
Abduction angles also wrap round +-180 degrees on bent fingers, but abduction isn't trained.

**Haptics buzzed at rest.** The rest level was the 95th percentile of a 5 s relaxed recording on the table, so 5 % of
the calibration itself already sat above it. Played back over the recorded sessions (levels from each session's first
rest cues and first fist, like the Calibrate tab), an output was on in 69 % of later rest windows: 47 % on the table,
73 % arm forward, 85 % raised (median rest envelope 4.6, 5.0, 6.1 uV, so tiny rises cross a line sitting in the
noise). Higher margins on the 10 s recording still left 4 to 15 %, mostly from the other postures.

Now the levels come from a recorded session (`calibrate.from_session`): rest from every rest cue in every posture,
max from every held grip, first second of each cue left out. Rest level = 99th percentile x 1.5. Tested on 9 sessions
with levels from repetitions 1 and 2 and playback over repetition 3: output on in 1.7 % of rest windows (worst session
4.2 %), grips switch an output on in 89 % of their windows, strongest output in a grip 56 % on average. Every other
rule tried (95th x 2, 90th x 2.5, median x 4, ...) sits on the same trade-off between quiet rest and weaker grips.
Rest levels from a session are 15 to 50 uV, far above the 5 uV of a quiet table rest, because rest cues include
relaxing after a grip and holding the arm up. Rest cues per session, filtered RMS, median over channels and cues:
table 5.9 to 10.4 uV, arm forward 7.6 to 18.1, raised 9.7 to 18.6, about double off the table. The 5 uV quiet rest
from the filter session is a fully relaxed arm on the table, not what a rest cue between grips measures.

The app now starts the haptics with every prediction model (Use this model / Use this network), with levels from the
session the model was calibrated on (calibrated network) or its newest training session (LDA), on the ESP32 link set
on the Haptics tab; stopping the model stops them. If the ESP32 can't be reached the model keeps running and the
Haptics tab says why. The Calibrate tab is kept for troubleshooting, with the same 99th x 1.5 rule.

**Camera glitch frames dropped** (`IMPOSSIBLE_DEG` in `ml/dataset.py`, cache version 3): a camera frame with a
trained joint bent backwards past -60 degrees counts as not tracked, so windows only get angle labels from good
frames. The raw recordings are unchanged. The test now checks that no such angle is left in any session. Same 8
sessions, 15 epochs: moving fingers r 0.32 / 0.35, held grips 0.59 calibrated, grips 84.1 % calibrated, all within
the run-to-run noise of before. Too rare to matter for the scores, kept for clean labels.

**Camera delay from the whole session, tried and not adopted.** The taps fail when too few agree (`084252`: not
measured) and rely on a few seconds at each end of a 15 minute session. Tried instead: armband gyro rotation speed
against the camera's palm rotation speed (orientation from wrist, index and pinky knuckle world landmarks), cross
correlated over the whole session. Per session 0 to 70 ms, but the two halves of one session disagree by up to 85 ms
and the correlation is weak (r 0.12 to 0.50), so it isn't reliable yet. It also sits about 90 ms below the taps in
every session (taps 96 to 168 ms), so one of the two methods has a timing bias. Not resolved; the angle score barely
depends on the delay (r within 0.03 from 50 to 200 ms), so the taps stay for now and a missing value uses the
default. Script kept out of the repo.

**First score of `2026-10-06_084252`** (9th session, nothing trained on it; network on the other 8): grips balanced
70.5 % without calibration (LDA 66.5 %), 85.4 % calibrated (LDA 82.7 %), rest taken for a grip 2.1 % calibrated.
Moving fingers r only 0.15 with or without calibration.

**Learning curve.** Networks trained on the first 1 to 6 sessions in recording order, each tested on the same three
newest sessions (`152106`, `164417`, `084252`), means over the three. Calibrated LDA doesn't depend on the other
sessions (82.4 % in every row).

| sessions trained on | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| network, no calibration | 42.6 % | 43.2 % | 48.1 % | 64.7 % | 64.9 % | 70.6 % |
| LDA, no calibration | 22.6 % | 41.5 % | 49.7 % | 60.5 % | 61.0 % | 64.8 % |
| network, calibrated | 80.3 % | 82.5 % | 82.4 % | 83.3 % | 84.6 % | 85.6 % |
| moving fingers r, calibrated | 0.00 | 0.07 | 0.05 | 0.09 | 0.13 | 0.12 |

Without calibration both still climb steeply at 6 sessions, no plateau yet, so more sessions are still the biggest
lever for grips. Calibrated, the network passes calibrated LDA from 4 sessions and reaches the 85 % target at 6.
Angles stay very low on these three sessions (0.12 at 6), lower than the 0.32 of the 8-session leave one session out;
the three test sessions hold most of the finger-cue data and the newest two have the new halfway and thumb cues, so
the training sets here have little finger movement in them. Not conclusive for angles yet.

**Mirrored band handled** (`channels.mirrored`): left arm with the charging port towards the hand, or right arm
with it towards the shoulder, counts as mirrored (right arm with the port towards the hand mirrors twice and counts
as normal). Mirrored sessions get their channel order reversed when the training windows are built (`ml/dataset.py`
and the LDA's `session_windows`), so all training data is in the reference orientation; the ring network takes care
of any rotation left over. Live, a calibrated network reverses the window when its calibration session was mirrored,
an LDA model when its newest training session was. Haptics are unaffected, their levels belong to the physical
channels. Not tried on a real mirrored session yet, all sessions so far are the reference orientation.

**Angle ceiling at one placement.** Each of the 6 sessions with the finger block cut into 5 blocks in time, each block
tested with a model trained on the other 4 (no session shift, the band didn't move). Moving-finger r, mean over the
blocks:

| | 164337 | 142500 | 112731 | 152106 | 164417 | 084252 | mean |
|---|---|---|---|---|---|---|---|
| network on that session alone | 0.17 | 0.30 | 0.24 | 0.13 | 0.09 | 0.10 | 0.17 |
| network on all other sessions + that session | 0.24 | 0.32 | 0.28 | 0.08 | 0.15 | 0.14 | 0.20 |
| ridge on the LDA features | 0.18 | 0.29 | 0.20 | 0.16 | 0.12 | 0.08 | 0.17 |

Per joint (network with all data): ring knuckle 0.31, middle PIP 0.28, ring PIP 0.27, pinky PIP 0.24, middle knuckle
0.24, pinky knuckle 0.18, index PIP 0.14, index knuckle 0.11, thumb 0.05. Even without any session shift and with all
the data, three different models land at the same 0.2 to 0.3, far from the 0.8 target. So for moving single fingers
the limit is the signal (8 surface electrodes over deep, overlapping finger flexors, the thumb mostly moved by hand
muscles the band can't see) or the camera labels, not the model or the session shift. Which of the two would take a
second label source (a flex sensor glove) to separate. The three sessions with the new cues (halfway holds, full thumb
range) score lower (0.08 to 0.16) than the three before them (0.24 to 0.32), not yet explained. The same holds for
the within-session ridge split by posture (moving 0.05 to 0.32, see the correction on 2026-10-05).

**Hand closure instead of single joints.** Mean flexion of the 8 finger joints as one value, from the same network's
angle outputs, each of the 6 finger-block sessions tested with a network trained on all other sessions, no
calibration: r 0.61 over all windows (0.56 to 0.67) against 0.46 for single joints, 0.34 on moving windows (0.16 to
0.53) against 0.23. Index alone 0.60 / 0.19. Better than single fingers but not near 0.8; the moving windows are
mostly single-finger cues where the overall closure hardly changes. A model trained on closure directly, with
calibration, is the next thing to try for protocol B: grip from the classifier plus a continuous closure value.

**Every session is scored by itself.** When a completed session is saved (after its camera delay), the app starts
the Deep learning tab's "evaluate the newest session" on it in the background, at low priority: a network trained on
all the other sessions is tested on it, and the row goes to `evaluations.csv`. So every session gets its honest score
before anything trains on it, and the CSV builds the learning curve while recording. Skipped without torch (the Mac),
for sessions without grips, or while another network job runs.

**Angle diagnostics.** Full write-up in `diagnostics/REPORT.md` (local, git-ignored with the plots and the 25 GB of
Ninapro DB8). 9 sessions, leave one session out, scored per joint with r and R2:
- The pipeline is fine. The same lagged ridge (MAV, RMS, WL, 200 ms, 6 lags of 50 ms) on DB8 (12 subjects, A1 + A2
  train, A3 test) gives multivariate R2 0.58 at 2 kHz with 16 channels against the published 0.63, 0.55 after
  resampling to 500 Hz through our filter (so 500 Hz costs nothing), 0.46 with one row of 8.
- Same decoder, same 8 channels, rate and filter, one placement: DB8 joint r 0.66 (moving 0.51), ours 0.35 (moving
  0.16). Channels correlate much more on our band (mean |r| 0.60 between channels in finger cues, DB8 0.14), and
  single-finger cues only lift the EMG 3 to 6 dB over rest (grips 8 to 11 dB), with almost the same pattern on every
  channel. Common average or neighbour differences don't help (0.16 vs 0.17). Part of the gap is features, see
  covariance features below.
- Camera labels: about 1 degree frame-to-frame jitter and 2 degrees inside a held grip, against 17 to 22 degrees of
  spread, so random label noise allows r 0.97+. Thumb MCP only moves with SD 6 degrees over all moving cues and
  scores at chance in every model (corrected 2026-10-07: the thumb moves a lot, thumb MCP is the wrong label). Joints that aren't cued still move 10 to 23 degrees in single-finger cues (anatomy or MediaPipe, can't tell).
- R2 on moving windows is about 0 for every model even where r is 0.3 to 0.4: r overstates the angle output.
  Chance (labels shifted by 30 s or more) reaches r 0.14, so ridge on moving windows (0.16) is barely above it.
- Network vs ridge across sessions: moving 0.29 vs 0.16, held grips 0.58 vs 0.29. Within one session ridge gets 0.50
  on held grips, so most of the network's gain is handling band movement. Moving fingers have no session effect.
- Inside its own cue the network follows the index at r -0.11 and the thumb at 0.00 (ring 0.29). Free movement 0.30,
  single-finger cues 0.21.
- Targets: hand closure (mean of the 8 finger joints, or the first PCA synergy) and middle+ring+pinky reach r 0.42 on
  moving windows and 0.61 overall, the only ones with clearly positive R2 while moving (about 0.2). Velocity, synergies
  2 and 3 and the thumb are not decodable. Smoothing labels adds 0.03.
- Phases: static holds carry their share of the error (51 % of it, 50 % of windows), but a relaxed finger held still
  inside a finger cue scores like movement (0.30) against 0.59 for firm grips; mean EMG level barely follows hand
  closure (r 0.12). A relaxed bent finger looks like an open hand to the EMG.
- Raw data: no clipping or dead channels; all 8 channels repeat their last value together in 0.01 to 0.15 % of samples
  (WiFi stalls). 60 Hz on ch 5 15 uV median.
- Amount of data ruled out: DB8 trained on only the first 1 to 10 repetitions of each movement (8 channels, 500 Hz,
  our filter, same ridge) gives moving r 0.40 to 0.53 (median over 12 subjects) already at 1 repetition (2.5 min);
  at our amount (about 7 bends per finger, 13 min) 0.44 to 0.52 against our 0.16. More data mostly raises R2.
  DB8 is one posture, ours three, but posture turned out not to matter (next point).
- Arm posture doesn't matter for the angles. Ridge (amplitude + covariance features), scored inside each posture:
  across sessions trained on the same posture r 0.26, other postures 0.24, all postures 0.26; within a session, each
  moving cue predicted from the same posture 0.33, other postures 0.37 (`diagnostics/posture_fair.py`). An earlier
  version pooled the three postures' predictions before computing r and showed a large posture effect (0.36 to 0.46
  against 0.11 to 0.23). That was the scoring: a per-posture model learns that posture's average hand shape, and
  pooling turns the posture differences into a fake gain. Score r inside one posture (or one model) only. The
  camera-free finger test that seemed to back it (LDA on the 4 finger cues, 52 % within a posture against 44 % across)
  compared halves of the same cue, so it measures time closeness, not posture.
- Learning curve in one posture (forearm on the table, `diagnostics/table_curve.py`): ridge trained on the table
  windows of 1 to 8 other sessions, tested on each held-out session's table moving cues: r 0.10, 0.14, 0.17, 0.20,
  0.21, 0.21, 0.22, 0.23 (5.5 to 44 min of table data). Still rising but flattening: +0.10 over the first 4 sessions,
  +0.03 over the next 4. R2 negative throughout (-1.7 to -0.2), so the scale and offset are off between sessions.
- DB8 recipe copied onto our sessions (`diagnostics/krasoulis_recipe.py`): their 8 features per channel on 128 ms
  windows, 6 lags, ridge, tuned smoothing, 5 targets like their DOAs, the three posture blocks as A1/A2/A3 (train,
  tune, test). With covariance features and the other sessions added to training, medians over 9 sessions on the
  test block: index r 0.61 (R2 0.33), middle 0.66 (0.39), ring+pinky 0.68 (0.35); DB8 with 8 channels at 500 Hz
  on the same targets 0.69 (0.43), 0.72 (0.48), 0.78 (0.59). On moving windows only 0.15, 0.32, 0.38. Thumb fails
  (r 0.23, R2 negative; DB8 0.61 to 0.64), which pulls multivariate R2 down to 0.19 (DB8 0.46). Their features alone
  0.07, + covariance 0.15. So on the fingers, scored their way (every window of the test block), we are within about
  0.1 r of DB8; the thumb and moving fingers are where we fall short.
- Channel covariance features (log of the 8x8 covariance per window, 36 values) help: same-session ridge moving r
  0.21 with log amplitude features, 0.30 covariance, 0.34 both (scored per block, one model per block). Not yet
  tried on DB8 or in the network.
- Decided with Noah: the MindRove dry pads stay (kit hardware), and the band stays just below the elbow. Moving it
  further down the forearm might separate the finger muscles better, but a short residual limb doesn't reach there,
  so it wouldn't carry over to most transradial amputees. Better electrodes are for the socket front end (stage 6).

## 2026-10-07: thumb label

**The thumb moves, thumb MCP just doesn't show it** (`diagnostics/thumb_check.py`, `thumb_targets.py`). In the thumb
cues the thumb tip travels about 104 mm across the palm in MediaPipe's world landmarks, spread over all 4 thumb
joints (5th to 95th percentile range: CMC flexion 34, CMC abduction 19, MCP 31, IP 49 degrees). Only thumb MCP is in
`RELIABLE`, so the networks learn the thumb from the joint that carries the least of the movement. The "SD 6 degrees"
of 2026-10-06 was over all moving cues, mostly other fingers' cues with the thumb still.

Same-day DB8 recipe (features + covariance, block 1 train, 2 tune, 3 test), median r over 9 sessions on all test
windows (R2, r moving): thumb MCP 0.32 (0.01, 0.07), CMC flexion 0.39, CMC abduction 0.40, IP 0.41, mean of the 4 0.47
(0.12, 0.11), thumb tip position across the palm from the raw landmarks 0.49 (0.14, 0.13), tip to pinky knuckle 0.46
(0.14, 0.16). Index for reference 0.66 (0.31, 0.27). DB8's thumb DOAs 0.61 to 0.64. A landmark-based thumb target
closes about half of the gap; moving-window r stays low.

**Fingertip positions against joint angles** (`diagnostics/fingertip_targets.py`, same-day recipe, one ridge per
target, median over 9 sessions). All test windows r: about equal (index angle 0.66 vs tip-wrist 0.64, middle 0.67 vs
0.61 to 0.62, ring 0.62 vs 0.54 to 0.62, pinky 0.63 vs 0.60 to 0.62). Moving windows r, angle against the best tip
measure: index 0.27 vs 0.36 (tip-wrist / palm length), middle 0.42 vs 0.48 (tip along the palm), ring 0.39 vs 0.54
(tip along), pinky 0.40 vs 0.46 (tip-wrist), thumb 0.11 vs 0.16 (tip-wrist / palm; tip along 0.32 but R2 negative).
Tip out of the palm plane (the camera's depth direction) is the worst target for every finger. Tip measures are
better on moving fingers by 0.05 to 0.15, the ring the clearest; within noise for some. Tip-wrist / palm length is
the most even and cancels hand size, so it's the candidate target.

**Full validation** (`diagnostics/VALIDATION.md`, plan written before running; `validate.py`, `validate_net.py`). One
harness over every factor tried, on 9 sessions and DB8 as the control, 95 % bootstrap intervals over sessions, r
always inside one test block and one model. Best on our data: the ring network trained on the other sessions and
fine-tuned (all weights) on the day's first two posture blocks: r moving 0.42 fingers, 0.43 fingertips, 0.52 closure,
0.37 single joints (R2 moving 0.05, 0.09, 0.19, -0.17; all windows r 0.58 to 0.73, R2 0.04 to 0.45). Holds with
intervals: network beats ridge everywhere (+0.05 to +0.12 r moving); other sessions + same-day fine-tune beats either
alone; covariance features +0.05; closure easiest; fingertips equal to per-finger angles (the fingertip gain above was
against single joints); paper features no better than amplitude. DB8 under the identical ridge harness and capped to
our amount of data: r moving 0.58 single joints, 0.60 fingers, 0.66 closure, against our 0.29, 0.34, 0.42. The cap
costs DB8 only 0.03, so amount of data is ruled out (same as the repetition test). The gap left is sensors/contact,
camera labels or the movement protocol; the recordings so far can't separate them. Two harness bugs were caught and
fixed before reading results (NaN spreading through smoothing, a cap that kept only DB8's first movements).

**Grip-conditioned continuous output** (`diagnostics/validate_heads.py`, validation harness, pretrained on the other
sessions + same-day fine-tune, one seed). Parallel heads (as now) against a cascade (continuous head reads the
features plus the grip probabilities, one hidden layer of 64) and a mixture (one continuous head per grip, blended by
the grip probabilities). Against parallel, paired over 9 sessions: R2 all windows +0.04 to +0.06 for both (fingers,
fingertips, closure; intervals exclude zero), r moving +0.00 to +0.01 (no change), grip balanced accuracy unchanged
(80.7, 80.9, 81.1 %). Knowing the grip puts the continuous output at the right level for the held grip, it doesn't
help follow movement. Cascade and mixture are equal; the cascade also has an extra layer, so part of its gain may be
capacity. Single seed: run-to-run noise was about 0.01 to 0.03 r (2026-10-06).

**Two rings of 4 instead of one ring of 8** (separate test, `diagnostics/ring_layout/layout.py`, `out/results.md`
there). Idea: move 4 pads (2 loose + the hub's 2 on the bottom, 4 on top) into a second ring. Checked on DB8, which
has two aligned rows of 8 (3 and 5.5 cm below the elbow; row 2 sensor i matches row 1 sensor i by correlation),
same-day ridge and scoring of `validate.py`, S1 to S10 (S11 and S12 have dead row 2 sensors, so the old "row 2"
number in `diagnostics/REPORT.md` includes dead channels for them). Fingers r moving, amplitude + covariance: one ring
of 8 0.63, 4+4 aligned 0.62, 4+4 staggered 0.64, one ring of 4 0.48, all 16 0.73. Paired against one ring of 8:
4+4 aligned -0.01, staggered +0.01 (intervals include zero), ring of 4 -0.16, 16 channels +0.09. Same pattern for
closure and R2. With 8 channels, moving pads into a second ring changes nothing; channel count helps (4 to 8 +0.16,
8 to 16 +0.09). Not worth rearranging the band. A second ring is worth it in the socket only as extra channels.

**Camera delay from an LED** (separate test, `diagnostics/camera_led/led_delay.py`). The ESP32 switches the ch 0 LED
(`M` command over USB, round trip 3 ms) at a logged time, the webcam films it with the recorder's capture settings,
first frame past half brightness counts. 40 on and 40 off switches, dark room, CPU busy with the screen. Camera delay
about 64 ms with frame times fitted against the grab number, 80 ms with raw arrival times (median minus half a frame,
30.0 fps). On switches show up about 47 ms earlier than off switches (the lit LED saturates, so a partly lit frame
already passes the threshold); the median over both cancels that. Raw arrival times scatter -32 to +38 ms around the
fit. Repeated with the room light on: 63 ms fitted, 70 ms raw, on and off 37 ms apart, so room light doesn't change
it. With frame times built like `dataset.camera_times` (fitted line moved down to the earliest 5 % of arrivals,
about 26 ms lower) the delay is 37 ms in both runs, and that is the number for the labels. This is the camera
alone. A session's delay is the camera's minus the armband's own WiFi delay on the EMG stamps, so it should be under
37 ms: the taps (96 to 168 ms) can't be right unless the EMG stamps run early, the gyro (15 to 50 ms) is close. Confirm by measuring
the armband's delay once the band works again (done 2026-10-08, round trip 17 ms).

## 2026-10-08: armband round trip

**Round trip from mode switches** (`diagnostics/armband_rtt/armband_rtt.py`, band powered, not worn, pads not
needed). What the modes do to the EMG rows: `EEG_MODE` is the normal recording mode (offsets of tens of mV per
channel). `TEST_MODE` replaces the inputs with an internal 2 Hz square wave, 314 uV peak to peak, the same on all 8
channels, near 0 DC. `IMP_MODE` streams exact zeros, and back in EEG the inputs swung by tens of mV for seconds
after, so it isn't used. Each test: switch to TEST at a logged `time.time()`, back to EEG, 40 times each at random
gaps, first sample where most channels step by over 10 mV. Two runs, 80 switches each:
- Round trip 16.8 ms median in both runs (14.6 to 26.9 ms, 10 to 90 % about 15.5 to 18 ms), same in both
  directions. Raw arrival times 17.6 ms. ICMP ping to the band 1 ms median, so the WiFi link is a small part.
- The band stops sampling for 11.5 ms (7 to 13) at every switch without skipping package numbers, so a run with
  switches fits at 492 Hz. Each switch got its own clock fit for that reason. Recordings have no switches.
- The EMG stamps can't lag the true sample time by more than the round trip, so the label shift is between
  37 - 17 = 20 ms and 37 ms. Equal halves (28.6 ms) don't apply here: most of the 17 ms is the switch pause and
  packet filling on the band, not two equal WiFi legs. Either way the taps (96 to 168 ms) are wrong.
- Once (right after a survey that used `IMP_MODE`) the band opened a new stream still in TEST, though EEG had been
  sent last. Not repeated in two later opens. The recorder never sends a mode, and TEST looks clean to the hum and
  contact checks, so all 8 channels equal near 0 would be the sign.
- Method: much better than the taps. It needs no arm, no camera and no timing guess, gives a clean step on all 8
  channels, and the two runs agreed to 0.1 ms. Use it for the armband side from now on, the LED test for the
  camera side.

**Finger decoding screen** (`diagnostics/VALIDATION.md`, second protocol, written before running; full results
there). 30 candidates for continuous finger decoding on the existing recordings, each the baseline (ring network,
grip-conditioned continuous head, pretrained on the other sessions + same-day fine-tune) plus one change, same run,
3 seeds, scored on moving windows (fingers and closure, r and R2). Dev set 6 sessions; the 3 newest finger-block
sessions locked away and scored once at the end. A clear win needed +0.05 with a Bonferroni-corrected bootstrap
interval over 120 tests, all 3 seeds and no gain on shifted labels.
- No clear win. Dev baseline: fingers r 0.46 / R2 0.14, closure r 0.54 / R2 0.27.
- Clear losses: calibration of 0 or 1 min (-0.11, -0.13 r; 1 min is worse than none, fine-tuning all weights on it
  overfits), double augmentation, no holds in the continuous loss, smoothing 0.1, temporal layers. 5 min of
  calibration is as good as the 9 min of two blocks.
- No meaningful effect: covariance or filter bank inputs to the network, 20 Hz low cut, open/fist label scaling,
  session input normalisation, label smoothing, dropping ch 6, seed ensembles. Small consistent losses under 0.05:
  Huber, rest gating, dropping ch 5, dropping free movement, finger cues or the older sessions.
- Can't tell, under the bar anyway: IMU input +0.03 r, moving-window weighting +0.02 r / +0.03 R2 (5 of 6 sessions).
- Temporal context: the planned end-to-end TCN and GRU runs were invalid (training in chunks of consecutive windows
  broke the network by itself: -0.13 r, grips -23 points with no temporal layer). Rerun with the temporal layer on
  the baseline's frozen features: moving r unchanged, moving R2 -0.05 to -0.12. The GRU fits held hands better
  (+0.04 r all windows) but lags in movement.
- Image landmarks are less decodable than world landmarks; the thumb tip target ties the thumb angle.
- DB8: the ring network beats ridge there too (moving fingers r 0.71 vs 0.61, R2 0.47 vs 0.33, 12 subjects, intervals
  clear of zero), so its lead isn't specific to our band moving between sessions.
- Locked sessions, baseline scored once: fingers r 0.39 / R2 0.08, closure r 0.48 / R2 0.14 on moving windows,
  grips 84 % balanced.
- Bug found after the runs (rerun on 2026-10-09, below): the session-end sync segment carries the table posture, so the 5 s guard
  around the table block's whole time span left no calibration data and the table block was never tested. All arms
  and the locked run were tested on the forward and raised blocks only; comparisons stay paired, a third of the test
  data went unused.
- So nothing available on the existing recordings moves moving-finger decoding by 0.05. The gap to DB8 (0.71 vs
  0.39 to 0.46) is in what the recordings contain, not in the model or training.

**Fixed camera delay in training** (`ml/dataset.py`). Labels now use one fixed delay of 28 ms for every session
(`CAMERA_DELAY_S`), the middle of the 20 to 37 ms the LED test and the armband round trip allow, off by at most
9 ms. The tap delays in `session.json` (96 to 168 ms) are no longer read; the recorder still measures and shows
them. Camera frames now sit 70 to 140 ms later on the EMG clock than before. Cache version 4, so `ml/cache/` rebuilds on
the next load. The screen above and every earlier angle number used the tap delays. Tests updated
(`test_camera.py`).

**Grip + closure against per-finger output** (`diagnostics/validate.py blend`, the screen's dev baseline
predictions, old camera delay, forward and raised blocks). Grip + closure: the grip head picks a shape (calibration
medians of that grip and of the open hand), predicted closure sets how far along it is. Moving windows r / R2:
per-finger head 0.467 / 0.145, grip + closure 0.394 / -0.054, a per-finger blend with weights fitted on the other
block 0.459 / 0.105 (weights 0.73 to 0.88 towards per-finger). Per-finger also wins on all windows (R2 0.468 vs
0.371). The continuous head already reads the grip probabilities, so the templates add nothing it doesn't have.
No separate grip + closure head; a grip mode that snaps to the template could still feel steadier live, which only
a live test can show.

**Protocol B session type** (`gestures.protocol_b_plan`). Ninapro DB8's 9 movements (thumb
flex, thumb out and in, index, middle, ring and little, point, cylinder grip, key pinch, tripod), forearm on
the table by default (other postures can be ticked), each a slow close
and open over 5 s following an animated target hand, 3 s rest. Every cue's target hand animates now (`cueFrame` in
`web/app.js`): finger cues do their halfway, pause, full bend three times, the wave bends the fingers in turn, grips
close in half a second and hold; the cue preview plays them on a loop. In rounds (each
movement once per round, shuffled, 6 rounds) instead of DB8's repetitions in a row, so any few minutes hold every
movement. Each posture starts with a calibration block (every grip twice, each finger once), so every session can
calibrate a model and still counts as a good session. One posture: 9.9 min. New cue kind `move`, added to
`MOVING` in `ml/train_ringnet.py` and to the hum redo. Protocol A drops free movement by default (the screen showed
it helps only about 0.03, protocol B replaces it): 3.6 min per posture. Sync taps removed from both protocols
and the delay measurement at session end with them, since the camera delay is now fixed; `sync.py` stays for
older sessions. The Train tab's session table shows the protocol instead of the tap delay. The Record form has a protocol switch
that shows only that protocol's fields.

## 2026-10-09: screen rerun

**Finger decoding screen, rerun** (`diagnostics/VALIDATION.md`, change 3 and "Rerun results"). Same protocol with the
guard fixed (the table block is tested now) and the labels on the fixed 28 ms camera delay; the invalid end-to-end
temporal arms dropped, the temporal follow-up included, 120 tests. Overnight, 23:00 to 04:00.
- No clear win again. Dev baseline: fingers r 0.48 / R2 0.12, closure r 0.59 / R2 0.29 on moving windows.
- Held in both runs: losses from 1 min of calibration, double augmentation, smoothing 0.1 and the 1 s TCN head;
  5 min of calibration as good as 9; small consistent gains under the 0.05 bar from moving-window weighting
  (+0.03 r, all 6 sessions) and IMU input (+0.03 r, 5 of 6, R2 not better).
- Changed between runs, so not settled: no calibration (-0.11 then -0.06 r), dropping holds, the GRU and 3 s TCN
  heads (+0.02 r now, R2 worse), covariance input. Dropping free movement or the older sessions is now a clear
  loss (-0.04 r).
- Locked sessions, baseline: fingers r 0.40 / R2 0.07, closure r 0.51 / R2 0.17, grips 83 % (first run 0.39 /
  0.08, 0.48 / 0.14, 84 %).
- The baseline stays the pipeline. If anything is added later, moving-window weighting and IMU input are the
  first candidates, both small.

## Open

- Camera delay: re-measure with `diagnostics/camera_led/led_delay.py` only if the camera, its capture settings, the
  grab code or the computer change, and with `diagnostics/armband_rtt/armband_rtt.py rtt` if the band, its
  firmware or the WiFi setup change; then update `CAMERA_DELAY_S`.
- Angles (screen 2026-10-08): no change to the network beats the current baseline on the existing recordings, keep
  it (at least 5 min of same-day fine-tune). If the guard bug is ever fixed in a rerun, guard around each test window.
  Next: record protocol B sessions (DB8 movements, forearm on the table) and a protractor check of the camera, to
  split the remaining gap into protocol, sensors and labels; then a live control test, since offline r doesn't
  predict live control.

- Next priority (Noah, 2026-10-05): finger angles. Moving-finger r is 0.34 (see 2026-10-05). Record about 10
  sessions with the new cues and rerun the evaluation every 2 or 3 sessions: if the moving-finger r keeps rising it
  is a data problem, if it flattens the model or the sensors are the limit. Then: weight moving windows, an angle
  head with its own temporal part on the shared backbone, calibration that includes finger movement in every
  posture, per-session open/fist normalisation of the labels, causal convolutions so the network can stream on an
  ESP32-S3.
- More sessions with the 7 grips, firm holds and a marked band position.
- Quick recalibration in the app: a short calibration-only recording (1 to 2 repetitions of the grips) after
  putting the band on. The Deep learning tab can calibrate on any session, but there is no short recording type yet.
- Test per-session normalisation on unseen sessions. Rotation augmentation only for LDA, the ring network ignores
  whole-electrode rotations by construction.
- `ml/cache/` is about 50 MB per session (every window stored, each sample 4 times). Store the filtered signal once
  and cut windows on load, and drop cache files of sessions that no longer exist.
- Measure false grips per minute in everyday activity, and a "hold to switch" rule.
- Bring latency under 200 ms: shorter windows and vote, at some cost in accuracy.
- `ml/` foundations: session cache, dataset builder, shared evaluation harness, model registry, continual update
  flow tested with LDA first.
- Measure the camera delay over the whole session (accelerometer against camera wrist movement) instead of 6 taps,
  which disagreed by up to 135 ms in session `164337`.
- Tap test round the band to confirm channels 0 to 7 run in order round the arm, the ring assumes it. Then record
  one mirrored session to check the channel flip on real data.
- Ring network: rerun as sessions come in; fix the rest false alarms after calibration (rest weighting); then
  continual updates with replay and the benchmark.
