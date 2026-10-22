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

## 2026-10-13: motor drivers

**Motor drivers arrived as HG7881 dual H-bridge modules (2x L9110S), not the planned ULN2003.** Pins B1A, B1B,
GND, VCC, A1A, A1B; 2 motors per module, so 4 modules for 7 motors, one GPIO per motor as before. VCC 3.3 V from
its own regulator (motors rated about 3 V, the L9110S drops a few tenths), shared ground with the ESP32.
Wiring in `pinmap.md`.
- The module pulls its inputs up to VCC: with the ESP32 off and xIB grounded the motors ran at full speed. A
  1 kOhm pull-down on A1A only brought it to 1.6 V, so the pull-up is about 1 kOhm too, and beating it would load
  the GPIO near its 20 mA limit.
- Fix: inverted logic. xIB left open (held high by the pull-up), motor runs while xIA is low, both high is off, so
  an unpowered or booting ESP32 leaves the motors off. Firmware 0.2.1 inverts the duty (`OUTPUTS_INVERTED`); flip
  it back to false for LEDs or a low-side driver, or those run full on at 0 %. Protocol unchanged.
- The coin motors feel weak. Noah is considering LRAs instead.

**Pad wires repaired.** Two pad wires had snapped at the solder joint again, most likely ch 0 and 2 (Noah's count
was 1 and 3 from one; not confirmed with a press test). Spliced, and the joint covered with low-temperature hot glue
pressed flat with parchment paper, tapering towards the pad edge. After the repair, at rest: filtered RMS 3.2 to
4.5 uV on all 8 channels, hum 2.3 to 4.7 uV except ch 5 at 13.6 uV, no lost samples. Clench and channel order not
checked yet.

**First protocol B session** (`2026-10-13_170753`, 10 min, table, right after the pad repair). Scored with the screen (`diagnostics/protocol_b.py`)
baseline (`diagnostics/validate_net.py run_held`, unchanged), pretrained on all 9 older sessions. The session has one
posture block, so the split is by time instead: calibration block always in training, the 6 movement rounds in
thirds, fine-tune on the calibration + 4 rounds, test the other 2 (3 folds, 3 seeds). Moving windows:
- Fine-tuned: fingers r 0.56 (seeds 0.553 to 0.561), R2 0.14; closure r 0.66. Older sessions, same baseline:
  0.39 to 0.52 per session (dev mean 0.48, locked 0.40), R2 about 0.10.
- Without fine-tuning: fingers r 0.50, R2 0.20 (older sessions 0.22 to 0.51). Fine-tuning raises r but lowers R2 here.
- Part of the gain is the content: chance r (labels shifted 30 s or more) is 0.16 here vs 0.08 to 0.22, mean 0.12, on the
  older sessions. Above chance: 0.39 vs 0.34 on average.
- Not like for like: test windows are DB8 movements in the same posture, older sessions were tested on another
  posture block with finger cues and free movement; and the pads were just repaired.
- Passes the bar set before the run (above 0.53), so more protocol B sessions are worth recording. One session, so
  not settled: rerun after every 2 or 3.
- Second session (`2026-10-13_173556`, band off and on, 7 cm): fine-tuned fingers r 0.51 (0.50 to 0.53), R2 -0.16;
  without fine-tuning r 0.51, R2 0.10. Chance 0.13. Mean of the two fine-tuned 0.54, just over the bar; the second
  alone is at the top of the old range. Above chance both are 0.38 to 0.39 vs 0.34 on the older sessions.
- Without fine-tuning both sessions score 0.50 to 0.51, against 0.22 to 0.51 (mean about 0.41) on the older ones.
- Fine-tuning makes R2 on moving windows worse in both (0.20 to 0.14, 0.10 to -0.16): it follows the movement a
  little better but at the wrong size. Look into this before putting a same-day fine-tune into the app.

**Placement fields on the Record form.** The second session got the first one's placement and notes copied over by
the remembered form (fixed by hand in its `session.json`: 7 cm, no notes). Now `crease_distance` + `crease_unit`
(distance of the band from the anterior elbow crease) and `module_side` (main module on the muscle belly, anterior,
or the bony side, posterior) are required in `settings`, start empty, are cleared after every start and are never remembered (the form names them if one is missing);
`placement` (now optional free notes) and `notes` aren't remembered either.

## 2026-10-14: one decision per finger

**Idea (Noah):** instead of joint angles, one decision per finger, labelled from the cues rather than the camera;
the hand's tactile sensors decide when to stop. Checked on the existing 11 sessions (`diagnostics/finger_states.py`,
results in `out/finger_states/`). Four decisions: thumb, index, middle, ring + little (they share a flexor). LDA,
protocol A features, per held session without calibration and with its first half as calibration.
- Binary per grip (closed if knuckle + middle joint average 40 deg, thumb in for every grip but rest and open) makes
  fist, tripod and key pinch the same pattern. They differ in thumb position and how far the fingers close.
- Binary, calibrated: per finger 73 to 83 % balanced, all 4 right 71 %. The 7-grip LDA with its grip turned into
  finger states does better on every finger: 82 to 88 %, all 4 right 80 %. On the protocol B movements and animated
  finger cues, where fingers move on their own: all 4 right 55 % vs 60 % from the grip LDA.
- Single fingers on their own (older own-pace finger cues, scored against the camera): 51 to 59 % balanced, about
  chance. Those cues have no cue timing, so they weren't in training; only the 2 protocol B sessions had labelled
  single-finger movement.
- 3-way (closing / opening / no change, from the cue timeline slope, current + 200 ms delta features): 44 to 49 %
  balanced (chance 33 %), about half of the movements caught, 2 to 4 false commands per minute.
- Reaction delay to a grip cue (camera crossing halfway): median 0.15 to 0.45 s per session, spread (IQR) 0.1 to
  0.45 s, so cue-only labels are blurred by about 0.2 s. Transitions last 0.3 to 0.5 s, which hurts 3-way most.
- So per-finger decisions don't beat picking a grip, and single fingers aren't separable with this data and LDA. A
  grip classifier plus tactile stop and EMG-driven speed gives the same control idea with the better decoder.

**Ideas for continuous-feeling control (not decided, for brainstorming with Noah).** Split the job: the grip
classifier picks the shape (the most reliable output, 80 to 86 % calibrated, about 4 % wrong at the 0.8 threshold)
and one continuous signal sets how far the hand closes (effort: a clench is 9 to 14x rest; network closure r 0.59 to
0.66 on moving windows, better than any single finger).
1. Grip chosen only while effort rises from rest, then locked until back at rest, so it never switches mid-grasp.
2. Closure follows EMG intensity scaled between calibrated rest and max (as the haptics), along the path from open to
   the chosen grip's closed pose; position or speed from effort is still open.
3. Tactile sensors stop each finger on contact, so the hand wraps round the object.
4. Relaxing holds the grip; the open-hand class releases it.
5. The same critically damped spring as the cue hand on the motor targets, against jitter.
- Costs: no free single-finger movement (7 shapes, each closed continuously, like most commercial hands); the
  2026-10-08 offline test scored grip + closure below the per-finger head (r 0.39 vs 0.47), but as angle tracking,
  which is not this design's job; rest false alarms after calibration still open (the lock limits them).
- Possible next step: a live mode in the app (virtual hand with lock, effort closure and spring, haptics following
  closure), tried on grab, hold and release, counting wrong grips and twitches. Alternative still open: a protocol C
  of animated single-finger close/open cues to test independent fingers with real labels.

**Cue hand on the Record screen:** every cue now starts and ends at the relaxed pose (fingers bent 20 to 30 deg,
`rest` in `gestures.py`) instead of the flat open hand; during a rest the hand stays relaxed and only the text names
the next gesture (it used to jump to the next gesture's pose). Movement and finger targets move only the joints they
change, the rest of the hand stays relaxed. The drawn hand follows on a critically damped spring (`SPRING_W` 12 in
`web/app.js`) instead of an exponential ease, so speed never jumps; it trails an animated target by under 0.2 s.
Recorded labels are unaffected (angles come from the camera).

**Simpler pipeline first (Noah and his boss).** Leaning on the network made problems hard to trace. The main path is
now the grip LDA plus a control layer whose settings can be changed by hand and seen straight away, with settings
tuned on recorded sessions. The network stays as a benchmark next to it.

**Control layer replay** (`diagnostics/control_sim/`, results in `out/results.md`). LDA calibrated on repetitions 1
and 2, effort levels from the same repetitions, then the control layer (`emg-reading/control.py`) run over the
repetition 3 block of 9 protocol A sessions, scored per cue like use: grip locked right, wrong decisions per minute,
false starts at rest, twitches. 28,680 settings tuned on 6 sessions, the newest 3 held out.
- First guess (threshold 0.8, 3 votes, start level 0.15, onset 0.6 s): held out 59 % of grip cues locked right, 3.7
  wrong decisions per minute.
- Best: threshold 0.7, 8 votes, start level 0.02, onset 3 s, release on rest (below 0.03 for 1.5 s). Held out 87 %,
  0.28 wrong per minute, 0.37 false starts per minute of rest; about 0.9 s from cue to lock, reaction included.
  Pinch is the weak grip (56 %). Held-out cues are few (9 per grip), and the held-out set was looked at twice, so
  87 % is a little optimistic: the next new session is the clean test.
- What matters: votes (the LDA is often wrong on the rise into a grip, 1 vote 45 %, 7 to 8 votes 86 to 88 %), the
  start level (pinch and key holds only reach about 0.15 effort since fist sets the max, at 0.15 pinch locks in 22 %
  of cues) and the release (open class 44 %: protocol A relaxes without opening, so the grip stays locked). A short
  onset hurts (0.6 s 67 %). Threshold 0.5 to 0.8 barely matters. Older sessions added to the LDA do worse (59 %).
- So the effort gate and the "only while effort rises" rule are close to off in the best setting; the long vote
  and releasing on rest do the work. The vote is also what costs latency.

**Control tab** (live mode). The grip LDA's probabilities and the effort (mean over contacted channels of the
envelope scaled between the model session's rest and max levels, the haptics' levels) go through `control.py`
inside the existing prediction loop. The tab shows the hand from the controller, its state, locked grip and
closure, counts starts and releases, plots effort and closure over 10 s, and has one control per setting that
applies while it runs. Presets are saved in `models/control_presets.json`; "default" is the replay's best. Tried on
the 2026-09-30 replay only, not on the arm yet. Haptics still follow each channel's envelope, not closure.

**Protocol C, custom movements** (`gestures.LIBRARY`, `protocol_c_plan`), the recording side of "record and play":
the person picks movements, the EMG classifier recognises which one starts and the hand plays it back (not built
yet, C only records). A library of 57 hand movements on the Record form, grouped as power, precision, lateral and
flat, point and press, signs, single fingers and moving gestures. Sources: GRASP taxonomy (Feix 2016), Cutkosky,
Ninapro A to C, and the bebionic, i-Limb, TASKA, Ability Hand and Michelangelo grip lists. The most used grips are
marked (Bullock 2013: medium wrap 23 % of grasping time, precision disk 17 %, the top 10 cover 80 %; Vergara 2014:
pinch, cylinder, lateral pinch). Wrist movements are left out, the drawing has no wrist. Each posture: the grips twice
as calibration, then the picked movements in rounds (each once per round, shuffled), each a 6 s cue that follows the
target hand in, holds and relaxes; moving gestures (trigger, tap, count, snap) follow their own keyframes. 17 common
ones in one posture: about 14 min. New cue kind `custom`: never a protocol A grip label, counts as moving in
`train_ringnet`, redone on hum. The calibration holds feed protocol A like protocol B's (Noah). Poses were checked
against the drawing only (pinch 7 mm thumb to index tip), not a real hand.

**Protocol D, false start check** (`gestures.protocol_d_plan`). Every recording so far is cued grips and rest held
still, so nothing measures how often the control layer starts a grip during forearm activity that isn't one, and the
tuned start level (0.02) is close to off. A free recording of normal activity wouldn't work, real grabs in it are
meant. D is a cued list instead: 14 activities that use the forearm without gripping (wrist bending, side to side and
turning, arm raised, held out and overhead, leaning and pushing on the table, shaking the hand out, typing, drumming,
waving, talking with the hands, walking), 40 s each, after a grip calibration block like B and C. About 11.5 min.
New cue kind `nogrip`: never a grip or rest label, redone on hum. Any start the controller makes in one counts as a
false start. Real grabs are tested live on the Control tab instead.

**Armband WiFi from the app** (`emg-reading/connect_armband.ps1`, Connection tab). The power-up routine (forget and
rejoin `Mindrove_ARB_66bb04`, DHCP (5 s since the trials below), back to the static 192.168.4.2) is one button now; it asks for admin
rights through the UAC prompt (Noah's choice over a scheduled task), checks the band answers on 192.168.4.1, then
reopens the stream, since the band needs a new start request. The app used to refuse to start when the band wasn't
reachable (`prepare_session` fails at once, checked with the WiFi off), which made the button useless right after a
power-up: it now starts without the armband and warns until it's connected. Windows' saved profile allows WPA2 with
AES, which the script's profile uses. First version called itself (a helper named `Netsh`, PowerShell ignores
case) and failed with call depth overflow; fixed, and the log now shows every step with timestamps. Worked on the
real band 2026-10-14; after reopening the stream the app checks samples arrive (2 s).

**Warnings only where they matter (design rule, Noah).** A warning shows only when the thing it warns about affects
what is being recorded or done right now. The hand-out-of-view and hand-at-the-edge warnings now only show during
cues the camera labels (grips, rest, finger, free, B and C movements) and the setup break before them; not between
sessions, not during protocol D's activities (the arm leaves the picture on purpose) or the rest right after one
(`recorder._camera_needed`). Warnings about something broken (no camera, slow camera, armband not connected, hum)
always show. New warnings follow the same rule.

**The MindRove library crashed the app; it runs in its own process now** (`armband.Armband`, `armband.py --reader`).
Windows logged three crashes of the app's Python process on 2026-10-14, all in native code Python can't catch: twice
a fast-fail (`0xc0000409`) in `MindRoveWrapper.dll` when the WiFi script changed the network under an open stream,
once heap corruption (`0xc0000374`) when the app stopped the stream of a band that had been switched off. Closing,
releasing and reopening the stream on a stable network was fine (two cycles, about 500 samples/s, no crash), so the
library fails on dead connections, not on reopening. Now a small reader process holds the library and pipes the
samples to the app; closing kills the reader instead of stopping the stream, and a crash only ends the reader (the
app shows the band as not connected and Connect starts a new one). The app starts without the band; the command line
scripts still stop with an error. Opening succeeded with the band off and WiFi down once and failed at once another
time, so "connected" means the reader runs, and the no-data warning covers a reader that gets nothing.

**DHCP wait shortened from 20 to 5 s.** The 20 s came from one manual toggle on 2026-09-30 and was never tested. The
Connection tab now takes the DHCP time (0 skips the step). With the armband power-cycled before each connect, 3 s
streamed (986 of about 1000 samples in 2 s) and Noah found 5 s the reliable choice; default 5 s. 0 s (step skipped)
and 1 s fail, so the DHCP step is needed: confirms the 2026-09-30 explanation that the band only streams to a device
that asked it for an address since it booted.

**First protocol D session** (`2026-10-14_181042`, table, 10.4 min). Stopped early on a low armband battery; Noah
marked it good (`completed` set, reason in `accepted` in its `session.json`). Calibration complete (every grip
twice, one hum redo), 10 of 14 activities (typing, drum, push and arm raise missing).

**Several D activities bring out known hand shapes (Noah).** They aren't all "no grip": a classifier firing there can
be right. Two groups for scoring:
- Arm and wrist only, any start is a false start: wrist bend, wrist side to side, forearm rotation, arm raise, arm
  out, reach overhead, lean, walk, shake the hand out.
- Hand shapes expected: wave hello and push the table (open hand), talk with your hands (open, point and others),
  typing and drumming (single fingers moving, can look like point, key or tripod). Score these apart, with the open
  hand not counted as wrong in wave and push.
"Turn your palm" was unclear, now "rotate your forearm: palm up, then palm down, like turning a doorknob".

**Weakest channel across sessions.** Median grip RMS over median rest-cue RMS per channel (filtered, first 0.5 s of
each cue dropped), sessions `2026-10-02_175745` to `2026-10-14_181042`: ch 6 is the lowest in 6 of 7 sessions (2.0 to
3.0, others mostly 3 to 5.7), with a high rest level (up to 22.6 uV). Fist cues only: ch 6 rises the least in 7 of 8 sessions (1.6 to 3.9x,
others mostly 3 to 10x). Ch 0 has the lowest grip level in most
sessions but rests low too. `2026-10-14_181042` is weak on every channel (ratio 2.5 to 3.9, rest 16 to 24 uV against
10 to 14 before), possibly the low battery it stopped on (not checked); ch 7 rest high there and in `2026-10-06_084252`.

## 2026-10-15: settle time, protocol C groups

**How much of a cue training leaves out is measured now** (`gestures.settle_s`). It was a flat 1 s in three places
(LDA, network dataset, haptic levels). Now: the time the drawn hand needs to reach the pose plus a reaction margin,
and the rest of the cue (as long as the hold time set for the session) is used. Grips 0.5 s close + 0.17 s spring
lag + 0.5 s = 1.17 s, rest 0.9 s, held protocol C movements 1.67 s, moving cues nothing. The close times are sent to
the page, so the drawn hand and the trim can't drift apart. Reaction margin checked on 241 grip cues in 13 sessions
(`diagnostics/settle/settle.py`, hand within 10 deg of its held shape on camera): settled 0.49 s after the cue
(median), 0.87 s p90; since the spring animation (2026-10-13 on) median 0.6 to 0.78, p90 about 0.9 to 1.4 s, so 0.5 s
holds. Pinch is slow: median 1.0 s, p90 2.7 s (placing the thumb). Network cache version 6.
Held protocol C movements now close in 1 s and hold to the end of the cue (they used to relax over its last 30 %);
the rest cue after does the relaxing, like grips (Noah).

**A cue cut short by a pause counts as bad** (`session.segments`, Noah). Continue redoes the cue, so the attempt the
pause interrupted (often letting go to type or read) was going into training as the cued grip. Now it is left out
and the redo after the resume counts, in every session, old ones included. Hum pauses already did this.

**First protocol C session** (`2026-10-15_100023`, table, 17 common movements x 5 rounds, 19.5 min). Bad: both
knife cues Noah flagged and the redone pinch (cue 103), marked after the session. LDA with protocol A's features,
trained on rounds 1 to 3, tested on 4 and 5 (`diagnostics/c_groups/confusion.py`, `drift.py`):
- 17 movements 35 % balanced (chance 6 %). Rounds held out in between (1, 3, 5 to 2, 4) 47 %, own training rounds
  78 %, so part of the drop is drift over 20 minutes. The session's own grip calibration scored 74 % (7 grips, rep 1
  to rep 2), normal: the band was fine, 17 movements are far more than this signal separates.
- Noah's predictions held: knife taken for point 63 % (right 4 %), thumb and 2 fingers for tripod 42 % (right 0 %),
  the power grips mix (hook 4 %, precision disk 4 %). Fist 99 %, point 92 %: the firm, clear shapes.
- Groups from the confusions: 8 groups 60 % balanced (cylinder, hook, pinch, pinch others closed / point, knife,
  lateral tripod / thumb 2, thumb 3, tripod / trigger, light tool / power sphere, precision disk / key / open / fist),
  12 groups 50 %.
- Likely part of it: effort. The gentle shapes collapse into each other (as key pinch did before the firm-hold cue
  on 2026-10-01). Props are out (Noah: it has to work for people without a hand, whose data is an attempted grip
  with nothing in it); an effort target on the Record screen is the idea to discuss.
Drawings fixed: thumb and 2, 3 and 4 fingers (the thumb stopped 44 to 59 mm short of the fingers, now 12 to 18 mm,
a thin box between them), knife (index along the blade, thumb on the side of the handle, no longer the point
shape). Moving gestures (trigger, two finger trigger, tap, beckon, scissors) hold their grip to the end of the cue
instead of relaxing over the last 12 %.

**Protocol D replay: false fists during arm activity** (`diagnostics/control_sim/d_replay.py`, the app's control
layer, LDA and effort levels from the session's own grip block; `2026-10-14_181042`, 6.4 min of activities, one
session). With the tuned defaults the hand sits closed 32 % of the time in the arm and wrist activities, 27 % in the
hand-shape ones, 36 % in the rests between. Nearly every start is a fist: reach overhead, wrist bend, lean and arm out
tense the same forearm flexors, and the classifier has no class for "arm busy, not a grip". The lock keeps it closed
because release on rest needs 1.5 s of relaxed arm. Release on rest or open hand: 30 / 20 / 25 %; that plus start
level 0.15 and 12 votes: 18 / 13 / 17 %, still far off and the higher start level costs pinch and key. The control
layer can't fix it alone: next is giving the classifier the D activities as a "not a grip" class.
Effort target on the Record screen dropped (Noah): no time to read during a cue, and effort differs too much between
people (atrophy after an amputation). Smaller protocol C set to choose per person next, from that person's own
groups; train each group on all its members and let the user pick the active one, so swapping inside a group needs
no retraining (repetitions spread over the members, so not longer).

**False grips during arm activity: distance check and class release** (`control.py`, `gesture_model.distance_space`).
Second protocol D session (`2026-10-15_105900`, complete, all 14 activities, module posterior) was worse than the
first: hand closed 63 / 75 / 52 % of the time (arm / hand shapes / rests). Two findings:
- Probability can't flag them: during the false grips the winning grip has the same confidence as real grips
  (median 1.00 both), a 0.99 threshold removes 29 % of false and 22 % of real grip windows. The LDA only says which
  grip is nearest. The distance of the window from that grip's calibration windows does separate them (real
  median 3.6, arm activity 10.3; cutoff 6 drops 82 % of false windows and 10 % of real ones, `d_confidence.py`).
- The lock keeps a false grip closed: release on low effort needs 1.5 s below 0.03, which arm activity rarely gives.
Fix, no extra training data: windows farther than `max_distance` (6) count as rest, and a new release mode
`rest_or_class` also lets go when rest or the open hand wins `release_n` (6) steps. On the replays
(`tune_release.py`): D sessions 0 / 0 / 0 % and 4 / 7 / 1 % closed; protocol A unchanged (86.4 % correct locks,
0.44 wrong per minute, 0.09 twitches per cue), time to lock 0.95 to 1.00 s. A "not a grip" class trained on D's
activities (equal priors, each activity left out) worked too but adds nothing on top and needs the extra block,
Noah's point against it. The remaining wrong locks in protocol A are grips mixed with each other (key and point,
tripod and pinch: the thumb grips), which distance can't see. New defaults on the Control tab; models trained
before today have no distance (the check is off for them), retrain to use it. Two D sessions, one person, table.

**Live test of the lock design felt bad, back to the goal** (Noah). The distance check and release work above came
from replays of cued recordings and didn't carry over: live the hand waited most of a second for 8 votes in a row,
any flicker reset the count, and forearm effort fell into fist. The live test was also calibrated on a D session's
2 holds per grip (my suggestion), against the about 6 per grip across postures every replay number assumed, so it
wasn't a fair test. The real goal is a hand that feels continuous without decoding fingers, and voting then locking
is discrete by design. Calibrate live tests like the replays from now on (short protocol A, 2 repetitions, 2 postures).

**Blend mode** (`control.py` mode "blend", next to the lock mode, which stays the default). Closure follows effort
from the first window; the hand closes towards a mix of the grip poses weighted by the grip probabilities, smoothed
over `smooth_s` (0.25 s), so the shape settles into the grip as the classifier firms up and one flicker only nudges
it. One grip at the threshold for `lock_s` (0.5 s) locks the shape until the release mode lets go.

First live try (protocol A calibration `2026-10-15_140633`, model `noah_2026-10-15_141613`): the shape locked to fist
and stayed at 100 % while the open hand won at full probability. The locked shape only let go on effort under 0.03
for 1.5 s, but opening the hand takes effort too, and hold on relax kept the closure at its peak. Fixed: a locked
blend now uses the same releases as lock mode (`release_mode`, so rest_or_class lets go when open or rest wins
`release_n` steps). Live, `lock_s` set to 0 (never locks) to feel the plain blend first.

Second live run (logged from the app's live state, Noah ran a fixed sequence: each grip twice, fist to pinch, arm
waving, relax; start level 0.15, max distance 4, no lock):
- Waving the arm closed the hand to 100 % although its windows were 9.7 from every grip (over max distance, so
  counted as rest). Cause: blend left rest out of the shape mix, so a rest window only dropped out of the average and
  the last grip stayed. Fixed: closure is scaled by 1 minus the smoothed rest probability. Replayed on the same run:
  waving max closure 1.00 to 0.76, mean 0.14 to 0.06, fist unchanged (0.98). It adds a little lag on the rise, rest
  fades over `smooth_s`.
- Fist windows sat at distance 4.1 to 4.4, so max distance 4 cuts fist too (fist max closure 0.65 with the fix).
  6 keeps fist.
- Light grips only reach effort 0.4 (pinch, tripod), closure 0.38 at gain 1.5 with start level 0.15. Gain 2.5 gives
  0.64 but twitches at rest up to 0.48.
- Thumb grips flicker: key read as point (2 of 2 times), tripod mixed with pinch and point. Same as the replays.

**All sessions plus the first calibration, today's session held out** (`2026-10-15_140633`, 2 repetitions, table and
forward; calibrated on one repetition, tested on the other; `ml.train_ringnet --held`, `diagnostics/lda_all_plus_cal.py`):

| Method | Balanced | Grips only |
|---|---|---|
| LDA on the other 14 sessions | 57.7 % | 50.7 % |
| LDA on the 1 calibration repetition | 70.1 % | 65.4 % |
| LDA on the other 14 sessions + the calibration repetition | 58.8 % | 51.9 % |
| Network, no calibration | 73.4 % | 69.2 % |
| Network + transform + grip head from 1 repetition | 80.2 % | 77.4 % |

LDA can't use the old sessions (the old placements drown the one calibration repetition), the network gains 10 points
over LDA from them. Today's session is weak for LDA: 1-repetition calibration scores 73 to 92 % on the other sessions
(`diagnostics/lda_one_rep.py`), 70 % here. The earlier 80 % LDA numbers were 2-repetition calibrations on 3-repetition
sessions. The Control tab now also runs calibrated networks (no distance check for them yet).

**Three cheap grip changes, none worth adopting for accuracy** (`diagnostics/grip_checks.py`, log in
`diagnostics/out/`, 15 sessions leave one session out, 15 epochs, one seed). Calibrated on the other repetitions
(15 sessions) and on one repetition (9 sessions with 3):

| Method | Balanced (all other / 1 rep) | Wrong at 0.8 | Answers at 0.8 |
|---|---|---|---|
| LDA | 80.2 / 78.2 % | 9.4 / 9.3 % | 91 / 89 % |
| Network + transform + grip head | 84.7 / 81.4 % | 4.8 / 6.2 % | 87 / 86 % |
| Same, batch norm stats from the calibration windows | 84.2 / 81.4 % | 5.2 / 6.2 % | 87 / 86 % |
| LDA on the calibrated network's features | 84.2 / 80.9 % | 9.3 / 10.9 % | 97 / 96 % |
| LDA and network probabilities averaged | 84.7 / 82.7 % | 2.4 / 3.2 % | 80 / 79 % |

Batch norm adaptation and LDA on network features gain nothing. The average of LDA and network has the same balanced
accuracy but halves the wrong grips shown at the threshold, at the cost of answering about 7 points less often: an
option for the Control tab if wrong grips matter more than speed, not a gain in accuracy.

**Do combined hand shapes add up in the EMG?** (`diagnostics/additivity.py`, `2026-10-15_100023`). Idea (Noah):
detect several known patterns in a window at once and blend them, like object detection, so a half-closed fist or a key
pinch with the other fingers open come out as mixes. That only works if the EMG of a combination is close to the sum
of its parts. Templates: channel covariance of each grip's calibration holds minus rest; each protocol C cue fitted as
a non-negative mix of the 6 grip templates. 6 templates explain 89 % on average (one average template 50 %), so the
patterns carry information, but:
- Fist, point and open land on themselves (1.08, 1.03, 0.67). Key comes out as pinch 0.66 + point 0.45 (key 0.17),
  pinch matches nothing above 0.13: the thumb grips aren't separate patterns even within the session.
- Combinations don't split into their parts: pinch with the others closed is pinch 0.21 and no fist, hook is tripod
  + point. "Fingers closed" in a light grip doesn't look like a weak fist.
So whole grips aren't the building blocks. If blending is tried, the parts have to be what the band separates (effort,
index in or out, open, thumb engaged), not the thumb grips apart.

**Channel order round the arm, from the data.** Per-window log RMS minus its mean over channels (the pattern, not the
effort), correlated per session over all grip windows, averaged over 15 sessions: channels 0 to 5 correlate most with
their neighbours (0.36, 0.53, 0.36, 0.25, 0.32), falling with distance round the ring (1 apart 0.16, 4 apart -0.31).
Channels 6 and 7 don't fit (-0.37 with 5, -0.11 with each other, -0.05 with 0). So 0 to 5 run in order; 6 and 7 on
the base module are either out of order or see little of the grips (ch 6 is also the weakest channel). Not a tap test.

**Cursor tab: abstract control** (`cursor.py`, `tools.CursorTask`). Different approach from the classifiers: nothing
is trained, the person learns a fixed map (abstract myoelectric control, Dyson, Barnes and Nazarpour, who found
people learn it in minutes and keep it across days and arm positions). Each channel's level (haptics' levels from a
chosen session) pulls a dot towards that electrode's direction round the ring, so the side of the forearm that works
sets the direction and effort sets the distance; relaxed and even co-contraction stay in the middle. Center-out
targets with a home zone, dwell and timeout; every target run logged to `data/cursor/` with the path and levels per
step, so learning can be compared day to day. Rotation and mirror settings to line the map up.
Offline on `2026-10-15_140633` (no practice, just the map on recorded grips): natural grips all land between 130 and
185 deg (pinch near 0), so the other directions need new contractions; that is what gets learnt. The dot wobbles 0.11
within a held grip at 0.3 s smoothing (0.16 at 0.15 s), 0.01 to 0.02 at rest; defaults set easy from that (4 targets,
size 0.25, smoothing 0.3).
First live try: ch 0, 1 and 7 hardly moved the dot (Noah: likely the pads near the bone). Partly true (ch 0 and 1 rise
6.1x and 4.9x from rest median in a grip, ch 2 to 5 8 to 10x; ch 7 6.8x but rests at 10.6 uV against 4.4), mostly the
haptics' scale: rest 99th x 1.5 hit the max floor (2x rest) on every channel, leaving ch 1 and 6 at 0.45 and 0.30 in a
grip while 2, 3 and 5 sat at 1.0. The cursor now has its own scale (`cursor.LEVEL_SCALE`, rest 95th x 1.2, max 95th
of the grips, `calibrate.from_session(scale=...)`): every channel 0.69 to 0.85 in a grip, dot at rest still inside
the home zone (95th percentile 0.09). Haptics unchanged.
Practice logs (`data/cursor/2026-10-15_162815`, `162926`, 25 trials, 4 targets): 3 hits, none at 0 or 270 deg. Ch 6
never passed 0.10 in any trial while the others reached 0.8 to 1.0 (not a dead pad: ch 6 is the extensor pad, see
the open and close check below, and the practice was all squeezing), so nothing pulls towards 270, and pushing hard
fires every channel, which cancels out near the middle (the dot averaged 0.14 to 0.28 from it). The ring map assumes
working pads all round and one side firing alone; neither holds. A map fitted to natural grips instead (least squares
from the 8 smoothed levels to one target per grip, no offset, fitted on repetition 1, tested on 2, 3 sessions): rest
99 to 100 % in the home zone, open 40 to 72 %, point 0 to 64 %, fist 14 to 19 %, pinch 1 to 4 %; with 6 grips key and
tripod 0 to 26 %. A linear map of the envelopes separates about the same few things the classifiers do (open, point,
fist and effort). Cursor parked: a dead pad and co-contraction stop the ring map, and a fitted map doesn't separate more
than the classifiers. Kept on its tab for later (a fixed pad or socket electrodes may change that).

**Trigger mode** (`control.py` mode "trigger", Control tab, Noah's design). Since the band reliably gives effort, open
hand and point and not the thumb grips, the classifier is only asked for open and point; the other grips are picked
by clench pattern on effort alone, like the impulse triggers of commercial hands (i-Limb, bebionic). Hand open: one
clench held past 0.7 s is fist and keeps closing, 1, 2 or 3 short clenches then a 0.6 s pause are pinch, tripod, key;
point from the classifier (vote_n steps). Then effort above the hold level closes at a speed set by effort, the open
hand opens, relaxing holds (any partial closure can be held). Two open flicks within 1 s go back to picking; a pulse
during which the open hand wins never counts as a clench. Clench level 0.2, all timings settings on the tab. Not tried
on the arm yet. The distance check still turns far windows into rest, which can hide an open flick: set max distance
to 0 if opening doesn't register.
First try on the arm (Noah): feels good, but opening part way and the double open flick shared the open hand, so
opening a grip could drop out of it. Release is now two point flicks by default (`release_with`, open still
selectable): point isn't used while a grip is held, and the open hand only opens. After a release the point still
held can't pick point until it drops out. Watch for key read as point (seen live on 2026-10-15) firing a release.
End of the day: hum rose on ch 4 and 5 (median line RMS 55 and 103 uV, a minute later 385 and 176, the rest 25 to 30
from 7); ch 4 rested at 51 uV envelope against about 6. Floating pad level, so pad contact or the reference; signal
code unchanged. Not checked yet.

**Trigger mode against everyday arm use** (`diagnostics/trigger_replay.py`, `trigger_sweep.py`, `trigger_gates.py`;
the 2 protocol D sessions, every 50 ms step through the real controller; effort levels from each session's
calibration block, since its later rest cues follow arm activity and put rest far too high, fist then read 0.05
effort instead of 0.6). Hand idle at the start of each activity, back to idle after every pick:
- As built, clench level 0.2: 17.6 false grip picks per minute during the activities (arm raise, shake, talk, wrist
  bends worst), 12.2 in the rest cues after them. Mostly pinch, a single clench. LDA and network gave about the same
  counts: it is the clench detector, not the classifier. Deliberate fist holds reach effort 0.6 (10th percentile 0.36
  to 0.40), pinch holds about 0.
- Clench level alone: 14.9 / 11.8 per minute at 0.3 / 0.4.
- New options (`still_gate`: clenches and point only count while the accelerometer is as still as in the model
  session's holds, 99th percentile x 1.5; `pinch_clenches` 2: pinch, tripod, key at 2, 3, 4 clenches). At level 0.3:
  gate 5.6, 2 clenches 6.1, both 1.0 per minute (0.9 in rests); at 0.4 both 0.4. The calibration holds (deliberate
  clench, arm still) still pick a grip as often with the gate (16 of 24, the rest are light grips under the level).
  Both options off by default (Noah's design), on the Control tab.
- Holding a fist through the activities: double point releases 0 to 1.5 per minute with the network, up to 8.3 with
  the LDA (shake). Closure drift while holding up to 0.4 (arm raise): squeezing while the arm works keeps closing.

**Telling open from close while a grip is held** (8 newest sessions, levels on the cursor scale, grip holds only,
trained on repetition 1, tested on repetition 2, windows with mean level 0.1 or more). Open minus closing grips per
channel, mean: ch 0 +0.02, 1 +0.05, 2 +0.16, 3 +0.19, 4 +0.34, 5 +0.19, 6 +0.47, 7 +0.17. Ch 6 fires most on opening in
every session: it is the extensor pad. Open vs close, balanced: fixed pads (4 and 6 against 0 and 1) 68 % (worst
51 %), a linear split of the 8 levels 71 % (worst 52 %), the grip LDA (open against the summed closing grips) 84 %:
87 to 97 % in 6 sessions, 96 % on `2026-10-15_140633`, but 50 % and 69 % on `2026-10-13_173556` and
`2026-10-14_181042` (open almost never found). So trigger mode keeps the LDA's open hand for opening; a fresh
calibration is the fix when it fails.

**Which classifier for trigger mode** (`diagnostics/trigger_signals.py`, 15 sessions leave one out, network 3 seeds,
grips grouped into open, point, rest, close, threshold 0.7). Calibrated = transform + head on the other repetitions,
LDA = what the app runs live:

| | open found | open during a close | point found | point during key |
|---|---|---|---|---|
| network, calibrated | 87.2 % | 1.1 % | 83.6 % | 4.2 % |
| network, no calibration | 80.8 % | 0.9 % | 61.6 % | 5.2 % |
| network on the 4 groups, calibrated | 87.8 % | 1.2 % | 86.5 % | 6.8 % |
| LDA on the session's reps (live) | 78.1 % | 2.1 % | 84.4 % | 7.3 % |
| LDA on other sessions, no calibration | 59.4 % | 1.8 % | 51.0 % | 8.8 % |

The calibrated network finds the open hand 9 points more often than the live LDA, with half the false opens and fewer
point during key (accidental releases). Without calibration it still finds open 81 % of the time, but point drops to
62 %, so a general network with no calibration would open fine and release badly. Training on the 4 groups instead of
7 grips gains nothing and fires point during key more often. Use the calibrated 7-grip network in trigger mode.

## 2026-10-16: channel layout

**Channel order confirmed from the MindRove manual** (armband section, 8 + bias and ref, electrodes equidistant).
Numbered 1 to 8 there: 1 to 6 are the pods, 7 and 8 sit on the main module either side of ref and bias, 8 next to
pod 1 and 7 next to pod 6. Charging port towards the shoulder and looking from the shoulder, 1 to 6 count clockwise
(Noah). So in code channels 0 to 7 run in order round the ring with 7 next to 0, as the ring network assumes. The
2026-10-15 data check had 6 and 7 not fitting the ring: not an order problem, more likely the flat rigid module
sitting differently on the arm than the pods (and ref and bias between them). Ch 6 is still the extensor pad.

**Cleanup, no behaviour change** (branch `cleanup`). Duplicated code merged: session level ranges
(`calibrate.session_ranges`, used by haptics, control and cursor), the live majority vote (`gesture_model.vote_winner`,
LDA and network), the setup, grip and finger blocks of the four session plans (plans byte-identical before and
after for every protocol over 5 seeds), request checks in `app.py`, the settings controls and small helpers in
`web/`. Dead code removed (`gesture_model.FEATURES`, a duplicate `Replay.read`). Filter, model inputs and outputs,
file formats and routes are unchanged; 104 tests pass. Two visible differences:
- Session and model names in requests are checked the same way everywhere (`app.session_folder`, `app.model_file`):
  a name with a path in it or the wrong file type now gets "unknown session" or "unknown model" instead of a
  server error on `/api/train` and `/api/model/use`.
- `session.json` of new sessions lists the plan settings before subject and band position (`SessionSettings` now
  extends `PlanSettings`). Same keys and values, everything reads them by key.

"Protocol B" meant two things: recording protocol B (the Ninapro movements, `gestures.protocol_b_plan`) and the
continuous angle model. Code comments now say "finger angle model" for the second; older entries here use protocol B
for both. `sync.py` stays for old sessions' taps, in case a tap check is needed again.

**Calibrated network scores: only unseen sessions count.** `ringnet_2026-10-16_112322` (15 sessions) calibrated on
reps 1 and 2 of `2026-10-16_110400` scored 79.1 % balanced on rep 3 (rest taken for a grip 1.7 %, wrong grip at 0.8
6.1 %). It looked low next to the older calibrated networks (89.9 %, 92.9 %), but those were calibrated on sessions
their network had trained on (`seen_in_training`), so their test reps weren't new. 79.1 % is the first honest one in
the list and sits inside the held-out range of the evaluation log (network + transform + head 73.5 to 86.2 % over
the 8 newest runs). One test rep only, so it is noisier than an evaluation run. No evaluation row exists for
`110400`: the automatic evaluation after the session didn't run.

**Network jobs show progress.** Training prints each session as it loads and each epoch with its loss and the time
left; the evaluation also prints each fold. The Deep learning tab shows how long the job has run and how long since
its last line, so a stalled job shows. A stopped job now says "stopped" instead of its last line as the error, and an
old job's error no longer comes back after a click. Same weights with and without the progress output (checked).

**Sign research: which hand signs the band tells apart every time it goes on** (Noah's idea: find distinct signs,
however odd, and map them to useful outputs). Details, literature and the protocols to record in `sign-research.md`,
scripts in `diagnostics/signs/`. LDA on each session's own calibration only, tested on its later repetitions:
- Bar (90 % balanced on the worst session, under 1 false pick per minute on both D sessions): no set of today's signs
  clears all of it. With 2 holds per sign in each of 3 postures (10 sessions): tripod + point 97 / 95 % (mean /
  worst), open + pinch + point 95 / 90 % (89 % when chosen without that session), open + fist + pinch + point
  94 / 88 %, all six grips 83 / 75 %.
- False picks on the D sessions (`181042` / `105900`), lock mode defaults: 0.5 / 6.3 per minute for open, fist, point.
  With 12 votes and the still gate in front of lock mode: 0.0 / 1.1; nearly all left are wrist flexion read as point.
  On protocol A those two changes cost a few missed light signs (open, fist, point 96 % right first picks, 0 wrong,
  4 % none, 1.05 s).
- Effort and side of the forearm separate: point, open, fist are 3 to 4x rest; single finger bends (labelled from
  the camera) are 1.4 to 1.8x, 35 to 43 % right and mostly called rest. The thumb grips only collide with each other
  (one of them in a set is fine).
- Calibration must cover the postures: 3 holds per sign at the table only score 71 to 77 % in the other postures,
  one hold in each posture 88 to 92 %.
- Blending the calibration's class means (80 %) with earlier sessions and the covariance half and half: +1 to 2
  points mean, +4 to 9 on the worst session (13 sessions).
- DB8 reduced to one ring of 8 at 500 Hz: 9 movements 35 to 63 %, the 2 amputees 49 and 43 %; the amputees' best
  signs differ from the able-bodied group's (thumb abduction 96 %), so the set has to be chosen per person.
Literature review on top (about 110 sources, `diagnostics/out/signs/literature.md`, local): held finger postures
(thumb up, index and middle out, ring and little flexed, fist, point, fingers spread) beat grasps and wrist movements
in Ninapro DB2 and DB3 (40 intact, 10 amputees, one session); the thumb grips are one family everywhere; the best set
differs per amputee (2 to 11 movements above 90 %, Atzori 2016). No study tested horns, claw, held co-contraction or
little finger alone. Protocol E list extended with ring and little flexed and wrist cues as "not a sign".
Recommended for now: open = open / release, fist and point mapped to the two favourite grips, the rest cycled by
clench count as in trigger mode. One person, 2 D sessions.

**Protocol E, sign screening, in the app** (`gestures.protocol_e_plan`, Record form). 15 candidate signs (open, fist,
point, pinch, thumbs up, thumb tucked with fingers straight, claw, hook, two finger point, little finger out, horns,
fingers pushed back, rigid hand, thumb out, ring and little bent) and 2 held wrist bends as "not a sign", each 6 s
following the drawn hand, 3 rounds at the table and 2 with the arm forward, about 13 min. Signs are `custom` cues,
wrist bends `nogrip`, and an E session has no grip holds, so it stays out of grip and network training. 5 new
library drawings (two finger point, little finger out, claw, fingers back, rigid hand), checked in the preview on a
test copy. Analysis: `diagnostics/signs/screen_e.py <session>` (dry run on a fake session only). No eyes-closed
round: the cue has to be read. 107 tests pass.

**First protocol E session** (`2026-10-16_140854_noah`, table 5 rounds + forward 4, signs 3 s each, so about 1.3 s
held per cue after the 1.67 s close-in is left out; `diagnostics/signs/screen_e.py`, output in
`diagnostics/out/signs/screen_e_140854*.txt`). Calibrated on the early rounds, tested on the last table round and the
later forward rounds, one session, so an upper bound:
- Best sets: open + two finger point + thumb out 96.7 % balanced (table 100 %, forward 95.7 %); + little finger out
  94.1 %; + pinch 90.4 %. Thumb out, little finger out and two finger point are in most of the top 20 sets for
  N = 3 to 5. Thumb out is 95.6 % right even among all 15 signs, at only 2.0x rest; thumb abduction was also the
  DB8 amputees' best.
- Open + fist + point only 72.5 % here (protocol A sessions 94.5 %): fist was held at 2.2 to 8.9x rest from cue to
  cue and read as point 68 % of the time. Point alone 100 %.
- Held wrist bends: 92 % turned into rest by the distance check, 7 % read as claw. Weak candidates: thumb tucked
  (mostly rest), horns and ring + little bent (read as point), rigid and fingers back (read as claw).
Next: confirm across days (protocol F) with open, two finger point, thumb out, little finger out, with fist and point
alongside for comparison.
Live LDAs from that session for a feel test (`diagnostics/signs/train_sign_lda.py`, all rounds, extended features,
vote 5): `noah_signs4_open-twopoint-thumbout-little_2026-10-16` and `noah_signs5_with-pinch_2026-10-16` in
`models/gestures/`. Live prediction only: the Control tab's controller still knows only the 7 grips.
Live on the arm (Noah, same band position): the 4-sign model feels really good despite the little data. How the signs
were made, both as drawn: thumb out is the hand loose with only the thumb straight out (no fist); little finger out
is a fist with the thumb over the index and only the little finger out.
Live, thumb out and little finger out never get mixed up; the confusions are with two finger point and open (both
mostly the finger extensor; offline open read as two finger point 3 %, the other way 2 of 25 replay windows).
**Every combination vetted** (`diagnostics/signs/vet_sets.py`, all 16,263 sets of 3 to 7 of the 15 signs, each round
of each posture held out in turn, 9 folds, best 25 per size refitted; logs `diagnostics/out/signs/vet_sets*.log`).
With the wrist bends as a "not a sign" class, Noah's set (open, two finger point, little finger out, thumb out) is
the best 4-sign set: 95.2 % balanced mean, 90.4 % worst fold, 95.2 % at the table and forward, held wrist bends read
as a sign 25 % of the time (48 % without the class). Without the class it has the best mean (94.8 %) but a worst fold
of 85.2 %. Best 3: two finger point, little finger out, thumb out 96.3 / 90.1 %. Best 5: 91.3 / 85.5 %, so 5 signs
fall under the bar even within one session; pinch sets top out at 92.5 / 84.0 %. Little finger out and thumb out are
in nearly every top set of every size. One session, each fold is one cue per sign, so close sets are ties.

**Protocol F, the sign set, in the app** (`gestures.protocol_f_plan`, Record form; Noah: record more of these signs
and see how training goes). The 4 chosen signs (`F_SIGNS`: open, two finger point, little finger out, thumb out) and
the 2 held wrist bends of protocol E, once per round in every ticked posture, 6 s each with 3 s rest; 4 rounds at the
table and forward is about 7 min. Same cue kinds as E, so F sessions pool with the E session and stay out of grip
training. `diagnostics/signs/train_sign_lda.py` now takes several sessions (comma separated) and `--notsign` for the
wrist bend class. Checked on a test copy (plan, form, page). 109 tests pass. Networks only train on grip labels so
far; training one on the signs needs `ml/dataset.py` and `train_ringnet` to take sign labels.

**First protocol F session** (`2026-10-16_153635_noah`, band moved since the E session: 6 cm from the crease against
8, table and forward, 4 rounds, 6 s holds, 7.4 min). Calibrated on F's first round in each posture, tested on its
other rounds (`diagnostics/signs/sign_net.py across`):

| Model | Balanced | Wrist bends read as a sign |
|---|---|---|
| LDA on F's first round | 86.2 % | 7 % |
| LDA on E + F's first round | 86.2 % | 15 % |
| network (general grip backbone + new sign layer, fine-tuned on E only), no calibration | 86.3 % | 4 % |
| same, transform + sign layer fitted on F's first round | 87.6 % | 0 % |

The network trained only on the E session matches an LDA calibrated on the new band position, with fewer wrist false
picks. LDA within F alone (early rounds to the last): 90.5 %. Through the app's LDA code, E and F leave one session
out: 76 to 77 % balanced. Two sessions, one move, one day.

**Train tab trains sign sessions.** `gesture_model.train` on protocol E or F sessions labels the held signs, rest and
the wrist bends (`notsign`), keeping only the signs every chosen session has; grip sessions train as before, a mix is
refused. The Train tab lists E and F sessions (unticked by default, "good" only ticks grip sessions). Live
prediction works with these models; the Control tab still only knows the grips. 110 tests pass.
Live (Noah): the `notsign` class, trained only on the held wrist bends, takes up most of the noise between and around
the signs. To watch: whether it swallows the lightest sign (thumb out, about 2x rest).

## 2026-10-19: second protocol F session

**Second F session** (`2026-10-19_092314_noah`, band at 7 cm from the crease, table and forward, 4 rounds, 7.4 min,
no dropped samples). Cross-session test with `diagnostics/signs/sign_net.py across`, each F session tested in turn
with the E session and the other F session as training, calibration on the tested session's first round per posture,
test on its other rounds. Balanced over the 4 signs and rest / wrist bends read as a sign:

| Model | Test `092314` | Test `153635` |
|---|---|---|
| LDA on the calibration round | 75.6 % / 13 % | 86.2 % / 7 % |
| LDA on the other sessions + calibration round | 81.7 % / 22 % | 91.9 % / 23 % |
| network, frozen backbone + sign layer, calibrated | 81.7 % / 13 % | 81.2 % / 2 % |
| network fine-tuned, no calibration (3 seeds) | 84.3 to 87.2 % / 10 to 11 % | 94.7 to 95.9 % / 0 to 1 % |
| network fine-tuned, transform + sign layer calibrated (3 seeds) | 87.4 to 88.2 % / 11 % | 93.3 to 93.8 % / 0 % |

- The fine-tuned network beats the best LDA by 3 to 6 points on both sessions with about half the wrist false reads,
  and on these two sessions needed no calibration round to do it (not true on the third F day, below). Calibration
  adds 0 to 3 points on the new session and loses 1 to 2 on the other. The frozen backbone alone is worse than LDA, the gain comes from fine-tuning on sign data.
- `092314` is harder for every model. Its weak sign is open (59 to 67 % right with the network, 64 % LDA); the other
  signs are 79 to 97 %. Neither model reaches the 90 % bar on it.
- Two test sessions on two days, one person. Decision: the sign network moves into `ml/`.

**Sign network in `ml/` and the Deep learning tab.** `ml.networks signs <general network> --sessions ...` (Signs
section of the tab) loads the E and F sessions with the Train tab's sign labels (`gesture_model.session_windows`,
only the signs every ticked session has, rest, `notsign`), puts a new output layer on the general network, trains it
alone for 15 epochs and then fine-tunes everything but the transform for 30, the `fine` variant above. Saved as kind
`signs`, listed for the live test without a calibration; haptic levels come from its newest session, as for a sign
LDA. The Control tab doesn't list it (grips only). Checked against the diagnostic: trained on E + `153635`, tested on
all of `092314`, 86.7 % balanced, wrist bends read as a sign 8 %. 12 s on the 4070. First one trained on all three
sessions: `ringnet_2026-10-16_112322_signs_2026-10-19_094303.pt`. 111 tests pass.

**Third F session** (`2026-10-19_095340_noah`, 7.5 cm, 11 min with 3 pauses, 4 bad cues redone, 78 dropped
samples). Each F session held out in turn, trained on E and the other two F sessions, one seed:

| Model | Test `095340` | Test `092314` | Test `153635` |
|---|---|---|---|
| LDA on the calibration round | 82.5 % / 9 % | 75.6 % / 13 % | 86.2 % / 7 % |
| LDA on the other sessions + calibration round | 76.6 % / 22 % | 81.2 % / 9 % | 89.8 % / 23 % |
| network fine-tuned, no calibration | 76.7 % / 11 % | 87.7 % / 5 % | 92.4 % / 0 % |
| network fine-tuned, transform + sign layer calibrated | 86.3 % / 4 % | 88.7 % / 8 % | 92.2 % / 0 % |

- Calibrated, the fine-tuned network is the best model on all three sessions (mean 89.1 %, worst 86.3 %; best LDA
  per session 82.5 to 89.8 %) with the fewest wrist false reads. Without calibration it falls under LDA on `095340`,
  so the calibration round is needed after all: it adds 10 points there, 0 to 1 on the other two.
- The saved network (trained on E, `153635`, `092314`) on all of `095340`: 78.5 %. Its weak sign is two finger point
  (55 %), read as open (both mostly the finger extensor, as seen live on 10-16) and, arm forward, as thumb out.
- Still under the 90 % bar on the worst session. One seed; seeds varied by up to 3 points before.
- Kept, not re-recorded (Noah asked): the bad cues were redone and training skips them, and the worst session is
  the one the bar is about.

**Sign networks calibrate on the Deep learning tab.** Calibrate takes sign networks and sign sessions; repetitions
are rounds there (round k of a sign is its k-th cue in that posture, rest counts to the sign before it). Fits the
transform and the sign layer, as `sign_net.py`, and scores balanced over signs and rest, rest taken for a sign,
wrong answers at 0.8 and wrist bends read as a sign. Calibrated sign networks stay marked `signs`, so the Control
tab doesn't list them and refuses them. Grip calibration unchanged: same session and seed before and after 79.8 /
79.9 % balanced, the old code alone 79.7 to 79.8 % from run to run (GPU). Through the app, the 3-session network
calibrated on round 1 of `095340`: 86.7 % (sign_net.py 86.3 %). Grip and sign labels stay two paths in the code for
now (Noah: both are just position labels; unifying them is open). 111 tests pass.
Networks: `ringnet_2026-10-16_112322_signs_2026-10-19_101915.pt` (all 4 sign sessions) and
`..._signs_2026-10-19_094303_cal_2026-10-19_095340.pt` (3 sessions, calibrated on `095340` round 1).

**Without open** (3 signs + rest, same held-out test, one seed; `sign_net.py` with open dropped). Balanced / wrist
bends read as a sign, test `095340` / `092314` / `153635`:
- network fine-tuned + calibrated: 93.2 % / 1 %, 96.6 % / 6 %, 96.0 % / 1 %. Worst session 93.2 % against 86.3 %
  with open, so the 3-sign set clears the 90 % bar on every F session.
- best LDA per session: 89.6 %, 89.9 %, 93.4 %.
Part of the rise is one class fewer (chance 25 % instead of 20 %), most is open and two finger point no longer
taking each other's windows. Without an open sign, trigger mode can't open the hand part way by sign; a release
still opens it fully. Network on all 4 sessions without open: `ringnet_2026-10-16_112322_signs_2026-10-19_103625.pt`.

**Train and Deep learning tabs merged into one Train tab** (Noah: two trains and two calibrations were confusing).
One session table with buttons per protocol (all grip sessions, all A to D, all sign sessions, all E, all F, deselect
all), one Train button for LDA or a network (a sign network when sign sessions are ticked, with the general network to
start from), one Calibrate, one Live test list with LDA models and networks, evaluation and finger angles below.
Ticking a sign session unticks the grip sessions and the other way round, and the server still refuses a mix.
"Leave positions out" lists the positions every ticked session has; the ticked ones get no label (LDA, sign networks;
grip networks refuse it, their output layer is the 7 grips). Rest can't be left out. A sign network always starts
from the newest general network, named in the note under the model choice (a "Start from" list was confusing).

**Start from the general network or from scratch: no difference.** Same held-out test, both calibrated on round 1,
2 seeds each; scratch is a fresh ring network trained 60 epochs on the sign sessions alone. Mean balanced over the 3
F sessions: with open 89.4 % (general start) vs 89.0 % (scratch), without open 94.1 % vs 94.3 %; per session within
0 to 2.4 points, which is the seed spread (`095340` with open 85.2 to 88.4 % between seeds). Wrist bends read as a
sign about the same. So the grip network neither helps nor hurts the signs yet, and the low `095340` is the session,
not the start. Kept the general start for now (no cost); scratch would make sign networks independent of grip
training, worth rechecking with more people.

**Control tab: rules mode, mapping page, profiles** (Noah's design). A new mode beside lock, blend and trigger (all
three unchanged): the mapping is a list of rules "if (clench or a model class) (held for s, or repeated n times)
then (release, open, or any grip or protocol C library pose)", edited on its own page and saved on its own. Poses are
only picked while the hand is open and empty; release and open only work while holding, so a stray pattern can't
switch grips with an object in the hand (my call, Noah left it open). Closing stays proportional to effort as in
trigger mode. Repeats are counted as each one starts, so the highest count of a trigger acts straight away (two
point flicks release on the second one, as before), lower counts after the pause. The default mapping is trigger
mode written as rules, two finger point in place of point for sign models, without classes the model lacks.
Lock, blend and trigger take a sign model's class-to-grip map from the held rules (replaces the map dropdowns of
earlier today). Profiles carry settings, the mapping and the model they were built on; the list shows "built on this
model" or all, and loading one loads its mapping, which can then be changed. Settings split in basic (mode, clench
level, mapping) and advanced (only what the mode uses; max distance hidden for networks, which have no distance).
Network files now list their classes for the trigger dropdowns. 119 tests pass; checked headless on a test copy.
Layout fixes after the first look: radio buttons were stretched by the global full-width input style (radios now
excluded), the mapping line and its button sit in their own row, the mode help only describes the chosen mode, the
Control tab starts in rules mode (`control.DEFAULTS` stays lock for the replays and tests), the model output plot
only shows while running, and an untouched mapping follows the chosen model's default.
Then (Noah): a "close grip" action beside open and release (closes while its trigger stays on, speed set by effort,
squeezing still closes too), the three actions first in the list, and the Then dropdown replaced by a list that draws
the hovered pose beside it (a dropdown can't report the hovered item). Held 0 s acts on the first 50 ms step the
class is on; there is no vote in rules mode, so 0 s picks on a single window. 120 tests pass.

**Control tab runs sign models.** Each sign's probability goes to a grip through a map shown on the Control tab
(default open = open, two finger point = point, little finger out = fist, thumb out = pinch, notsign = rest), so lock,
blend and trigger run unchanged; trigger picks point and releases with two finger point. Uncalibrated sign networks
take their levels from their newest training session. Levels from a sign session (also for haptics) take the max from
the held signs, since it has no grips; on `095340` the max sits at the 2x rest floor on 7 of 8 channels, so effort
saturates early and the clench level may need raising. Grip sessions' levels unchanged. 112 tests pass.

**App restyled like the Better Bionics site** (dark). Tokens in `emg-reading/web/style.css` `:root`: black page,
panels `#0a0a0b`, white text, muted `#a1a1aa` (about 8:1 on black), one accent `#ff3d8b` in an orange to pink to
magenta gradient (wordmark, active tab, progress bars, primary hover, pill switches, section numbers). Inter for text,
JetBrains Mono for labels, tabs and table headers. Warnings keep solid yellow `#ffc233` and red `#ff3b3b` fills, the
only filled colour on the page. Canvases read the same tokens (`COLORS` in `app.js`). The tracked hand and live
signals went white instead of blue so a lost channel (red) can't be mistaken for the pink accent; the colour notes on
Train and Control were updated to match. New UI uses only these tokens, no hard-coded colours.
Then motion where it shows a change of state, never on the data: a live dot in the tab bar (pulses while state
arrives, "no data" after 1 s without it or with the no_data warning), the cue text slides in on a new cue and the cue
bar pulses in its last second, new warnings slide in (the bar is now rebuilt only when a warning changes, not 20 times
a second), buttons ignore clicks while their post is waiting and show a sweep after 150 ms, idle probability, output
and cursor canvases say what to start, the predicted label gets a fading underline when the class changes, the tab
underline slides, and touch screens get 44 px targets. All off with reduced motion.
Layout: tabs split into workflow (Record, Train, Control) and tools (Signals, Connection, Haptics, Calibrate,
Cursor); Train puts the session list (scrolls under a fixed header) beside the model, calibrate and evaluate jobs on
screens 1300 px and wider, so Train stays in view; the Record form has section labels (subject and placement,
protocol, timing).

**Cue screen** (Noah's live check): the cue bar now moves every screen frame from the local cue clock like the target
hand, at 20 updates a second it stepped. Mark bad shows: the recorder reports `marked_bad` (the current run of the cue
was marked, a redo after a pause starts clean, matching `segments()`) and `bad_mark` for 3 s (`BAD_NOTE_S`), so a mark
in the first 1.5 s of a rest names the grip before it. The page shows a red "marked bad" box and the button goes red.
**Wrist cues drawn**: the target hand gets a forearm and bends at the wrist for D's wrist_flex, wrist_side,
wrist_turn and wave_hello and the held wrist bends of E and F (`wristFrame` in `app.js`, `turnAtWrist` in `hand.js`);
flexion cues turn the view 70 degrees to the side so the bend can be seen. The palm is drawn as a 10 mm slab and the
forearm as a tapered solid, 32 by 20 mm at the wrist (outlines of their 3D points), so neither goes flat seen from the
side. A first try at 24 mm and 54 mm looked far too thick at the wrist (Noah).
Timing check (Noah asked whether transitions got slower): the spring (SPRING_W 12) and close times (grip 0.5 s,
sign 1.0 s) are unchanged, so grips still match their 1.17 s training trim (`settle_s`). The new held wrist bends first
bent in the grip's 0.5 s, but training trims them like signs (1.67 s), so they now bend in `C_CLOSE_S` too. D's moving
wrist cues have no trim and no pose to reach.

**Responsive pass** (every tab at 1440, 820 and 390 px): the tabs are one row that scrolls sideways under the wordmark
below 900 px (they wrapped into 2 or 3 rows); the 44 px touch minimum no longer applies to checkboxes (it spread the
posture list out) and checkboxes are 20 px on touch screens; the pill switch no longer breaks in two on a phone; help
text is capped at 90 characters per line; number inputs on the single-column tabs stop at 14em; plots get the same
outlined box as the hands; the probability plot and the finger angle hands hide while nothing runs instead of
showing empty boxes; the session table scrolls sideways below the two-column width instead of squeezing to 5-line
rows; probability labels scale with the canvas (they were 7 px on the Control tab).

**Training progress on the Train tab** (Noah: there was no bar). Network jobs: the server reads the `fold k/n` and
`epoch k/n` lines the scripts already print (`tools.job_progress`, tests in `tests/test_job_progress.py`) and the page
shows a bar over the whole job (all folds of an evaluation), the fold, epoch and time left; before the first epoch,
and for calibration which prints no epochs, a block moves along instead. LDA: `gesture_model.train` takes an optional
`log(text, share)`, called per loaded session, for the final fit and per held-out session and posture, so the bar
and the step line follow it (loading is the first half). Both have a Log dropdown with the full output, 200 lines kept
for networks (was 40) and the network's last line shown in its title. On 3 sessions the LDA log reads "loaded ...: 1904
labelled windows", "left out ...: balanced 55.3 % (vote 5)", "left out posture table: balanced 64.1 %".

**System group** (Noah's idea): the top bar is Record, Train, Control, System. System has its own row: Signals,
Connection, Haptics, Calibrate, and a Show archived switch that adds Cursor (parked); it opens on the last page used
(Signals by default) and the switch is remembered per browser. Replaces the workflow and tools split from earlier
today. Page ids and code unchanged.
**Sharp canvases**: every plot and hand was drawn at its fixed width and height attributes and stretched to the panel
by the browser, so the EMG bars were blurry (Noah), worse on a high-dpi screen. `sharpen()` in `app.js` sizes each
bitmap to its size on screen times the pixel density; plots draw in screen pixels, labels now a fixed 12 to 13 px.

**Final sweep**: the browser's own parts (scrollbars, dropdown lists, number spinners) follow the dark theme
(`color-scheme`); a gradient-dot favicon (the page's last console error was its 404) and the title Better Bionics
recorder; progress bars on Calibrate and Connection only show while their job runs; Stop network job only shows while
a network job runs (next to Train it read as part of LDA training); Control's section numbers count per column, so
Model output appearing while running no longer renumbers Profile and Basic; the mapping page's back button is a small
"Grip control" link above the heading; the pose picker says when a hovered entry is an action with no pose; "Connection
check" is now "Connection" (the page also holds WiFi and battery), the Train type select is labelled Type.

**Next line before wrist bends and activities** (Noah, found while recording): the rest before a cue only named the
next one when it was in the gesture list, so the held wrist bends of E and F and 13 of D's 14 activities never
showed under Next. Now every "nogrip" cue is named too; the hand stays relaxed during that rest (Noah: the text
matters more than the picture matching). Display only, recordings and labels unchanged.
The pill switches on Control and Mapping slide their gradient piece to the chosen side (200 ms) instead of
snapping (`placePill` in `tools.js` measures the labels, which differ in width).
**Keyboard shortcuts ignored caps lock**: Mark bad checked `e.key === "b"`, and with caps lock on the key is "B", so
`b` did nothing while the button worked (Noah). Same for `f` and `s` on Signals. Now case-insensitive, and a shortcut
shows on its button like a click.

**Fourth F session** (`2026-10-19_141720_noah`, 6.5 cm, 8 min, no bad cues, 140 dropped samples). Each F session
held out in turn, trained on E and the other three F sessions, one seed, `sign_net.py across`. Network fine-tuned,
balanced / wrist bends read as a sign:

| Test | with open, calibrated | with open, no calibration | without open, calibrated | without open, no calibration |
|---|---|---|---|---|
| `141720` (new) | 88.8 % / 7 % | 93.9 % / 1 % | 93.0 % / 5 % | 92.2 % / 2 % |
| `095340` | 85.6 % / 2 % | 75.7 % / 8 % | 91.9 % / 1 % | 81.1 % / 6 % |
| `092314` | 88.5 % / 10 % | 84.3 % / 3 % | 96.9 % / 3 % | 95.3 % / 0 % |
| `153635` | 94.2 % / 0 % | 94.4 % / 0 % | 95.0 % / 1 % | 98.1 % / 0 % |

- A fourth session in training didn't move the other three: calibrated with open 85.6 / 88.5 / 94.2 against 86.3 /
  88.7 / 92.2 with three, without open 91.9 / 96.9 / 95.0 against 93.2 / 96.6 / 96.0. All inside the 3 point seed
  spread. Best LDA per session 82.5 to 90.4 % with open, 88.4 to 94.1 % without.
- The new session is easy for the network uncalibrated (93.9 % with open) and calibration costs it 5 points there,
  against +10 on `095340`. One seed, so not yet a reason to make calibration optional.
- `095340` stays the worst session. Without open every F session is over 90 % calibrated (worst 91.9 %).

**Mapping page: protocol choice for the If list** (Noah: the list offered fist and pinch while building an F
mapping). A Protocol select above the rules (grips A to D, or signs F) sets the If options: only the classes that
protocol trains (`sign_sets` in `/api/config`), by their pose names; clench only for grips (Noah: F has no clench
cue); notsign is never offered. It follows the model when the model changes, and saved mappings keep their
protocol. A loaded rule on a class outside the protocol keeps that class in its own row only.

**Clench with a sign model** is effort only, so it fires with any model, but it isn't reliable there: F has no
clench, its effort max comes from the held signs (on `095340` at the 2x rest floor on 7 of 8 channels), and the sign
model has never seen a clench, so it may read one as a sign at the same time. Not measured.

**Mapping in two lists** (Noah: a held-0 s open while holding mustn't get in the way of a 0.7 s rule from relaxed,
and a visible split helps). A mapping is now `{"relaxed": [...], "holding": [...]}`, shown as two tables on the
mapping page, "From relaxed" and "From holding a grip". Only the current state's list runs, and either list takes any
rule (Noah: define them freely): a pose while holding switches to it at the same closure (until now nothing could
switch grips with an object in the hand), close, open and release do nothing while relaxed. A trigger can have a held
rule in each list, and when two rules fire on one step the higher one in the list wins (it was the trigger name's
alphabetical order). Mappings and profiles saved as one list load split by action. Rules drag by a handle on the
left, within a list or into the other one: the other rows slide out of the way while dragging (160 ms, off with
reduced motion), a cancelled drag slides back. No new move while rows are still sliding: without it downward drags
jittered on Noah's desktop, fixed with it (headless drags never showed it, every move happened once). Mouse only,
HTML drag and drop doesn't work on a touch screen, so each rule also has up and down arrows (up from the top of
holding goes to the end of relaxed and back).
One change in behaviour: a trigger with no held rule in the current list counts every episode as a repeat, however
long (with the old single list, point's held rule while relaxed also kept long points from counting while holding).

**Sign default mapping** (Noah's table): relaxed, two finger point held 0.4 s point, little finger out held 0.5 s
fist, thumb out held 0.5 s pinch, thumb out twice tripod, little finger out twice key; holding, open hand held 0 s
opens, two finger point twice releases. No clench. Grip models keep the clench default. The repeat counts for tripod
and key are guesses, not tested live. 127 tests pass; checked headless on a test copy.

**Readable model lists** (Noah: the file names told him nothing). Every model dropdown (Control, live test,
calibrate) groups models into sign networks, grip networks, sign LDA and grip LDA, newest first, and labels them from
their own info: when made, sessions, signs (and whether open is in), epochs, the session it was calibrated on, score,
and the subject where it differs from the newest model's. File names stay as they are (profiles, mappings and
calibrated networks refer to them) and show on hover. The early 11 grip set models count as grips ("older set").
Deleted at Noah's request (to the Recycle Bin): the three sign LDA models of 2026-10-16 (`noah_signs5_with-pinch`,
`noah_signs4_open-twopoint-thumbout-little`, `noah_signs4_F_..._153635`), a stale test. The saved mappings
`protocol-f-v1` and `protocol-f-default` were built on the first, so they now only list under "All mappings".

**Sign network epoch sweep.** Grip networks use 15 from the 2026-10-02 sweep; sign networks used 30 untested, because
`sign_net.py` did (output layer alone for half the epochs, then fine-tuning for all of them). Swept 10 to 60 with
`diagnostics/signs/epoch_sweep.py` (rows in `diagnostics/out/signs/epoch_sweep.csv`): each of the 4 F sessions held
out, trained on E and the other 3, 2 seeds, scored on its rounds 2 on, balanced over signs and rest. Means over the
4 sessions and seeds:

| Epochs | with open, no cal | with open, calibrated (worst session) | no open, no cal | no open, calibrated (worst session) | s per network |
|---|---|---|---|---|---|
| 10 | 85.1 % | 88.7 % (86.9) | 90.6 % | 94.2 % (91.9) | 10 |
| 20 | 85.3 % | 89.6 % (86.8) | 91.6 % | 94.6 % (91.1) | 18 |
| 30 | 86.2 % | 88.9 % (85.4) | 91.7 % | 94.0 % (90.3) | 26 |
| 40 | 84.1 % | 89.7 % (86.2) | 91.9 % | 94.7 % (92.9) | 34 |
| 50 | 85.1 % | 89.7 % (88.0) | 91.4 % | 94.2 % (90.6) | 42 |
| 60 | 84.9 % | 89.4 % (86.0) | 92.6 % | 94.4 % (91.4) | 50 |

- Flat from 10 on: calibrated means within 1 point of each other in both sets, about the size of the seed spread
  (0.6 to 1.4 points per session). Wrist bends read as a sign 2 to 5 % throughout. More epochs don't help on 5
  sessions; 30 costs 2.5 times the time of 10 for nothing.
- `095340` is the worst session at every count (85 to 88 % with open), so the gap there isn't undertraining.
- Sign default changed to 20 (Noah), the smallest count with no loss, in `ml/networks.py` and the Train tab.
  Recheck when there are more sessions, steps per epoch grow with the data.

## 2026-10-20: app startup

Noah: after the app has been off a while, the page shows but the warnings and the hand skeleton take a while to
appear. Timed each step on a test copy (warm start, replay, no webcam since the running app holds it):

- Imports 2.3 s (scipy.signal 0.8, mediapipe 0.5, fastapi 0.3, sklearn 0.24). Hand model load 0.04 s, first
  tracking pass 0.02 s. Page served 2.7 s after launch, first live message 0.03 s after that.
- The running app answers the page in 28 ms and sends state every 60 ms, so nothing is slow once it's up.
- Not measured: a cold start (files not in the disk cache), the armband connect (blocks before the page is served,
  up to 15 s when the band can't be reached) and opening the real webcam (1.2 s to 31 s measured earlier, see
  `camera.py`). These are the likely causes. The app now prints `armband connected in X s` and
  `camera ready X s after start (opening it took Y s)` at every launch, to settle it.
- The page now shows what it's waiting for: "Starting up, waiting for the recorder" in the warning bar until the
  first state message, and the camera warning names the step (loading the hand model, opening the camera, waiting
  for the first frame) and stays until the first frame is tracked (it used to clear when the camera opened). The
  step also shows next to the view dropdown and on the pose canvas.

## 2026-10-20: end of session screen

Noah wanted a clear visual cue when a session ends instead of the jump back to setup. The cue panel now shows an end
card for 6 s (held while the mouse is over it, Enter or Escape closes it): a gradient check for a completed session, a
white check for a practice run, a grey cross for stopped early (only a session that trains gets the
gradient), then fades back to setup. It shows duration, cues marked bad, hum redos, dropped samples and hand
detected; marked bad cues are called out in yellow because they are left out of training. It only plays on the page that watched the session end, not on a reload. The
summary now also carries `cues`, `cues_done`, `marked_bad` and `hum_pauses` (`recorder._finish`).

New System > Dev tab for testing page changes without recording: buttons that show each end card with made-up
numbers, and a short run (a real practice run cut to N cues of at most 1.5 s, breaks left out, `dev_cues` in the
session settings). A dev run is never saved, even without the practice box ticked. Tests of approved changes get
archived (only shown with Show archived); the end card test was approved and archived the same day.

## 2026-10-20: seven F sessions

Three more F sessions (`2026-10-19_161702`, `2026-10-20_093404`, `2026-10-20_105110`; 7.5 to 14.7 min, 0 to 34
dropped samples, placement fields empty). Each of the 7 held out, trained on E and the other 6 (the
`epoch_sweep.py` folds, 20 epochs, 2 seeds; the F list there now has all 7), calibrated on round 1, scored on the
rest. Network fine-tuned, balanced / wrist bends read as a sign:

| Test | with open, calibrated | with open, no cal | no open, calibrated | no open, no cal | best LDA with open |
|---|---|---|---|---|---|
| `153635` | 92.8 % / 0 % | 93.5 % | 95.9 % / 1 % | 97.1 % | 88.0 % |
| `092314` | 89.0 % / 4 % | 83.9 % | 96.7 % / 1 % | 93.8 % | 81.0 % |
| `095340` | 85.6 % / 2 % | 69.9 % | 91.9 % / 2 % | 84.2 % | 82.5 % |
| `141720` | 89.9 % / 6 % | 93.1 % | 94.5 % / 5 % | 98.5 % | 85.9 % |
| `161702` (new) | 94.6 % / 3 % | 87.0 % | 95.4 % / 3 % | 91.5 % | 85.9 % |
| `093404` (new) | 96.4 % / 0 % | 96.2 % | 98.6 % / 3 % | 98.4 % | 94.1 % |
| `105110` (new) | 92.1 % / 0 % | 75.5 % | 93.9 % / 0 % | 78.0 % | 92.4 % |

- Mean calibrated 91.5 % with open, 95.3 % without (worst 85.6 / 91.9 %, still `095340`).
- More sessions didn't move the 4 old ones: calibrated with open 89.3 % mean against 89.6 % trained with 4 F
  sessions (the 20 epoch sweep row), without open 94.8 % against 94.6 %.
- The new sessions are just better recordings: LDA on their own calibration round alone gets 85.9 to 92.4 %, against
  75.6 to 86.2 % on the old four. All three clear 90 % calibrated even with open.
- Trained on E and the old 4 only, the honest first test of the new sessions: calibrated with open 93.4 / 95.7 /
  90.1 %, so the extra sessions add 1 to 2 points calibrated. Uncalibrated they add more on today's two (89.0 to 96.2 %
  and 60.4 to 75.5 %), partly because they share a day with each other.
- `105110` needs the calibration round: 75.5 % without, 92.1 % with. Calibration still matters on some sessions
  and costs up to 3 points on others (`141720`, `153635`).
- Wrist bends read as a sign stay at 0 to 6 % with the network, 6 to 36 % with LDA on all sessions.
- Plateau for one person and one band (Noah: expected). More F sessions of Noah won't add much; one every few
  days to catch drift. Next gains from new conditions (other people, mirrored, placements) and the live test in
  rules mode (wrong picks per minute, stray grips).

## 2026-10-20: lock position in rules mode

Rules mode gets two holding actions, `lock` and `unlock` (`control.py`). Lock freezes the hand while holding a grip:
squeezing, the open action, pose switches and release are all ignored, and only the holding list's lock rules (fire
again to undo) and unlock rules run. Meant for carrying something while the arm does other work. Not in the default
mappings, added on the mapping page. Changing the mapping unlocks the hand so it can't stay frozen with no way out.
The Control tab shows "position locked" while it's on. Not tested live yet.

## 2026-10-20: recording on other computers, training on the desktop

Recording can happen on any computer with the app and an armband, training stays on the desktop's 4070. The
desktop is the hub (`app.py --hub`, `emg-reading/hub.py`), the other computers are devices with their own token
(`hub.py add-device NAME`), over Tailscale. Steps in `emg-reading/SETUP.md`.
- System > Sync on a device: sends ended sessions in 4 MB chunks that carry on after a dropped connection; a
  session only appears in the hub's `data/` once every file matches its hash, and is then evaluated like a local
  one. Downloads models with the session their levels come from, and swaps Control mappings and profiles both
  ways (same name with different content: each side keeps its own). Nothing is overwritten on either side.
- Train tab on a device: "The hub" sends the session list and every job to the hub through the app; the live test
  stays local.
- With `--hub` the app listens on every network, but other computers only get the hub's and Train tab's requests
  with a valid token, never the armband, haptics or camera. `--host` (tablet) works as before.
- New sessions store `device` (computer name) in `session.json`.
- Measured on the desktop with two copies of the app: a 115 MB session took 75 s with a new connection per chunk,
  because a name that also resolves to IPv6 costs 2 s per connection on Windows (the hub listens on IPv4 only).
  With IPv4 first and one kept connection: 6 s, byte for byte identical. Model download with its 169 MB levels
  session and training an LDA on the hub from the device's Train tab worked the same way.
- Not tried yet across two real computers: the Mac over Tailscale, and the Windows firewall prompt for the hub.
- The Mac is a 2017 Intel one. mediapipe has no Intel Mac build after 0.10.21 (Python 3.12 at most, needs
  numpy 1), so `requirements.txt` pins mediapipe 0.10.21, numpy 1.26.4, opencv 4.11 and scipy 1.17.1 there by
  platform marker; everything else stays the same. The code parses as 3.12. Not run on the Mac yet.

## 2026-10-21: app on the Intel Mac

- venv with Python 3.12 through uv (the system python3 is 3.15 and stays). The pinned Intel Mac versions install
  cleanly, scikit-learn 1.9.1 runs on numpy 1.26.4. Tests: 104 passed, 12 skipped (no sessions or torch there).
- The first start had no camera: macOS only shows the camera permission prompt from the main thread, and
  `camera.py` opens the camera in its own thread, so OpenCV reports "not authorized" and gives up. Opening the
  camera once from a main-thread one-liner brought up the prompt; after that the app's camera works. Only needed
  once per terminal app (in `emg-reading/SETUP.md`).
- Practice run on the Record tab with mediapipe 0.10.21: hand tracked, 30 fps.
- Hub: the Mac reaches `noah-desktop:8000` over Tailscale and the token is accepted. Upload checked with a fake
  session of zeros (`2026-10-21_130000_SYNCTEST-not-a-recording`, `completed` false so nothing trains on or
  evaluates it): arrived, hashes matched, same sizes on both sides; deleted on both computers after.
- Hum with the band on, first time on the Mac: the last hum the app kept before the band dropped off was 32 to
  98 uV on seven channels (good pads stay under 46) and 453 uV on ch 5 (a pad losing contact level). Not
  diagnosed; needs a recording to look at. Ch 0 hum came up later too.

**Armband WiFi on the Mac.** The Connection tab's Connect armband WiFi now works on macOS too (`mac_wifi.py`):
finds the strongest `Mindrove_ARB_` network (`airport -s`, gone from macOS 14.4, then join from the WiFi menu
first), joins it with `networksetup`, waits for a 192.168.4.x address and restarts the stream. The Mac has one WiFi
adapter, so on the band's network it is off the home network and the hub; going back is done from the WiFi menu
(a Back button that remembered the last network was removed, Noah: only if it's set simply per device). First
run: joined `Mindrove_ARB_66bb04`, address 192.168.4.2, 990 of about 1000 samples in 2 s.

**Delay test without the ESP32** (System > Dev, `delay.py`; the ESP32 now drives the motors). Run once per
recording setup, not per session; "Save for this device" writes the label shift into `devices.json`.
- Armband round trip: same method as 2026-10-08, now in the app. The reader process takes TEST/EEG mode commands on
  stdin and sends them on at once (it waits on the command queue instead of sleeping 10 ms between reads, which
  would have added up to 10 ms). 80 switches at 0.3 to 0.6 s gaps, always back to EEG at the end.
- Camera: the page flashes the screen black and white 60 times (each held 0.4 to 0.8 s, under 3 flashes a second),
  the camera films it in a mirror, first frame past half brightness on the fitted frame times
  (`ml.dataset.fitted_frame_times`, split out of `camera_times`, same numbers). Includes the screen's own delay, so
  it is an upper bound on the camera; to light and to dark are reported apart like the LED test.
- Label shift = camera - round trip / 2, the same rule as the 28 ms. Analysis tested on made-up streams
  (`tests/test_delay.py`).
- First run, MacBook + FaceTime camera: round trip 16.2 ms median over 80 of 80 switches (desktop 16.8), but
  10 to 90 % spread 0.7 to 67 ms (desktop 15.5 to 18). Under 11 ms is impossible (the band pauses that long at a
  switch), so the stamps on the Mac aren't plain arrival times or the WiFi jitters a lot; not looked into, the
  median agrees with the desktop. Flashes: 60 of 60 seen at 30 fps, to light 88 ms, to dark 15 ms, the opposite way
  round from the LED (on earlier). The first version took the median of all 60, which with two clusters lands in
  the gap (46 ms, saved as a 38.3 ms shift); corrected to the mean of the two directions' medians
  (`delay.camera_delay`): camera + screen 51.5 ms, label shift 43.4 ms in `devices.json`, 15 ms more than the
  desktop. Cause of the 73 ms light/dark gap unknown (camera exposure in the dark is a guess).

**Recording setups and the start page** (Noah: the delay belongs to the device, measured once). `devices.json`
(committed) lists setups, a computer plus a camera, each with its camera index, label shift and when it was
measured: Desktop + Brio 100 at 28 ms, MacBook + FaceTime camera 43.4 ms (above). The app opens on a start page
(Better Bionics wordmark, "Select device", one card per setup); only this computer's setups can be picked, matched
on a name that doesn't change with the network (`scutil --get LocalHostName` on a Mac, its hostname is
`Noahs-MBP.cogeco.local` at home). Picking one switches the camera live and the recorder stores the setup in
`session.json`; real sessions refuse to start without one, practice runs don't. `ml/dataset.py` looks each
session's shift up in `devices.json` by setup id (sessions from before: the desktop's 28 ms; not measured: 28 ms),
so measuring a setup later fixes its older sessions. The shift is part of the cache version, so every cached
session rebuilds once. The start page's entrance and exit run about 1 s, the one exception to the 200 ms motion rule
(Noah). The camera is still chosen by index; on a Mac the page checks the setup's camera is plugged in by name.

**Hand tracking switch** (Record tab, "Hand tracking" on/off, remembered per browser). Off stops the camera thread
and the hand model; turning it on starts them again, refused during a session because frame numbers restart at 0.
Sessions recorded with it off have no camera frames and `"camera": null`, like `--no-camera`; the angle labels are
NaN there and nothing else uses the camera. Measured on the MacBook with the synthetic board: the app at 77 to 90 %
of one core with tracking, 4 % without (the i5-7360U has 2 cores, 4 threads). Whether tracking hurts the EMG
recording on the Mac (dropped samples) is not checked yet. With tracking off, "Hand tracking is off" warns during
cues the camera labels, except in sign sessions (E, F), which train on the cue labels only.

**Hub starts minimal** (Noah: the hub only trains). `--hub` now starts without the camera and without connecting the
armband, the Connection tab can still connect it; their warnings are hidden until the band is connected. Recording on the desktop is done without `--hub`. Test copy on
the MacBook: 0.5 % of a core.

**First MacBook session** (`2026-10-21_141651_noah`, protocol F, tracking off, 8.3 min, completed):
- Lost samples: 99 of 249,036 (0.04 %), 71 gaps of 1 to 3 samples spread through the session. Lost samples aren't NaN, they are just missing; the filter runs across the gap. Arrival stamps
  jitter up to 0.38 s behind the package-number fit (WiFi bursts), the fit takes that out.
- Hum fine this time: median 7 to 15 uV per channel, 95th percentile 22 to 30.
- Sign LDA on this session alone, leave one posture out: forward 89.1 %, table 73.1 % balanced.
- Network windows build without camera frames (angle labels all NaN, as expected).
- Sent to the hub over Tailscale (relayed, no direct connection) in about 40 s, hashes matched.
- Sign network `ringnet_2026-10-16_112322_signs_2026-10-20_122442.pt` calibrated on its rounds 1 and 2 on the hub:
  97.9 % balanced on the other rounds, rest taken for a sign 0 %, wrong sign at 0.8 1.0 %, wrist bends read as a
  sign 1.2 %. The seven F sessions of 2026-10-20 had a calibrated mean of 91.5 % with open in the set.
- Not tested: live prediction with the network on the Mac (no torch there yet).

**Sign network learning curve** (`diagnostics/signs/learning_curve.py` on the desktop, rows in
`diagnostics/out/signs/learning_curve.csv`). Each of the last 4 sign sessions held out, trained on the k sign
sessions recorded just before it (20 epochs, 2 seeds), calibrated on its rounds 1 and 2, scored on the rest,
balanced over the signs and rest. Means of the 2 seeds:

| Held out | k = 1 | 2 | 4 | all earlier | no calibration, k = 1 / all |
|---|---|---|---|---|---|
| 2026-10-19_161702 | 93.2 % | 94.3 % | 94.3 % | 94.2 % (5) | 87.2 / 84.6 % |
| 2026-10-20_093404 | 95.5 % | 96.2 % | 96.3 % | 96.5 % (6) | 85.2 / 88.3 % |
| 2026-10-20_105110 | 96.5 % | 95.5 % | 95.8 % | 94.7 % (7) | 84.5 / 75.9 % |
| 2026-10-21_141651 (MacBook) | 96.2 % | 97.2 % | 97.3 % | 97.8 % (8) | 62.2 / 91.8 % |

- Calibrated, more sessions barely matter: at most 1.8 points between any two k for a session, about the seed spread
  (up to 1.8). Even trained on one session the MacBook session reaches 96.2 %. The rise from session to session
  (94 to 98 %) comes with the held-out session, not with the amount of training data: the later sessions are
  easier to classify (signs held more consistently, or the placement), not the model learning more.
- Without calibration there is no steady trend: the MacBook session went from 62 to 92 % with more sessions, the
  2026-10-20_105110 session fell from 85 to 76 %. More sessions help a new placement only sometimes.
- False alarms (rerun with them added, same splits), means over the 4 held-out sessions for k = 1 / 2 / 4 / all:
  wrist bend read as a sign 1.2 / 0.9 / 0.8 / 0.4 % calibrated, 2.3 / 1.9 / 0.7 / 0.7 % without; rest read as a sign
  (plain argmax, the app's threshold would turn some into unsure) 0.9 / 0.9 / 0.9 / 1.0 % calibrated, almost all of it
  2026-10-19_161702 at about 3 %, the others under 0.6 %; a sign read as another sign at probability 0.8 or more
  1.7 / 1.7 / 1.6 / 1.4 % calibrated, 3.0 / 3.4 / 3.6 / 3.9 % without (2026-10-20_105110 up to 6 to 7 %).
- So more sessions clearly cut the wrist bend false alarm (to a third calibrated, Noah's point that false alarms
  matter as much as accuracy); rest and confident wrong signs don't move beyond the session to session spread. A
  short calibration still does almost all the work for accuracy.

## 2026-10-22: backups and freeing space on devices

- `data/` on the desktop is 4.2 GB over about 45 sessions (100 to 350 MB each), with no backup until now.
- The hub now copies every ended session into a backup folder (System > Sync > Backup, saved in `hub.json`):
  Proton Drive's synced folder for now, a USB stick works the same. Each copy is hashed against the original before
  it is listed in the folder's `backups.json`; nothing there is overwritten or deleted. Runs at startup, after every
  session saved here and after every session a device sends.
- A device deletes sessions only when its app closes (Ctrl+C), never during use or on the hub (Noah), and only those
  whose files, the hub's and the backup's all have the same hashes, so there are always two copies off the device.
  It keeps the sessions its models take their levels from (haptics need them). Hub not reachable: nothing deleted.
- The backup is in Noah's own Proton account until the handoff. Move it to the lab's storage before then, and check
  where other people's recordings may be kept before anyone else records.
- Merged into `main` after the test below.
- Tried on the real setup the same day: the first backup copied all 44 hub sessions (4.1 GB) into Proton Drive in
  about 25 s, a rehash of 5 matched. The Mac sent a fake session and its real 2026-10-21 session; on Ctrl+C it deleted
  both (Noah OKed the real one), data/ on the Mac is empty. The fake session was removed from the hub and the backup.
  Not checked yet: that Proton finished uploading to the cloud.
- A network downloaded to the Mac didn't show in Control: the app hides networks where torch is missing. torch 2.2.2
  (last Intel Mac build) runs the downloaded sign network at about 5 ms per prediction on the 2017 MacBook. Torch is
  only how the PC runs it; the ESP32 port will run the exported weights in C++.
- Closing a test copy of the app (`--port 8010`) ran the cleanup and deleted `2026-10-22_100023_noah` from the Mac
  while the main app was still open (hub and backup both had it, nothing lost). Only the app on the default port
  8000 cleans up now.

## 2026-10-22: Control crash on the Mac session

- Control (and haptics) with the network calibrated on `2026-10-21_141651_noah` crashed with a division by zero in
  `calibrate.levels`. Cause: `lost_channels` dropped a channel from the common average if its pad lost contact anywhere
  in the recording. Three band bumps (3 of 3319 windows) tripped it on 7 of 8 channels, so the average was ch 1 alone,
  ch 1 minus itself was 0 and its rest level 0. Fine for the 10 s calibration it was written for, not since levels
  come from whole sessions.
- `from_session` now counts a channel as lost only when contact was off in more than 1 % of windows
  (`SESSION_LOST_SHARE`); no completed session has more than 0.1 %. It refuses with a message when fewer than 2 pads
  kept contact, and `levels` refuses a zero rest. Levels change only for the 4 sessions with a bump.
- The session itself is usable but hums: rest 18 to 26 uV filtered (Mac session's numbers) against 5 to 6 on the
  desktop, and 7 channels got "grips barely above rest" (ratio 2.0). Mac hum stays an Open item. Correction
  (2026-10-22): the 5 to 6 was the 2026-09-30 number. Recent desktop sessions rest at the same level as this one,
  see "Mac hum test" below.
- Control tab Simulate (Noah): runs the control layer without the band or a model, the page plays the model's output
  (class buttons or number keys, certainty, effort slider, space for full effort, arm moving). Every mode can be tried
  and tuned without the band. A session replay was left out on purpose: it can't react, so trigger mode's clenches
  wouldn't be yours. `Predictor.simulate`, `Control.prepare_sim` and `advance`.

## 2026-10-22: sync status and auto send

- Sessions only went to the hub when Send was pressed on the device. Now a chip in the header shows the state live
  (`tools.SyncWatch`, checked every 30 s, every 2 s while sending): on a device "synced", "N to send", "sending", "hub
  out of reach" or "hub refused"; on the hub "backed up" or "N to back up". Click opens System > Sync, the full line
  is there too.
- System > Sync has "Send new sessions by themselves" (hub.json `auto_send`, off by default). It sends whenever the hub
  answers and no session or practice run is going. The Mac has one WiFi, so while it records on the armband's network
  the hub is out of reach and sending waits until it is back on the home network.
- Merged into `main` together with the Control tab's Simulate (Noah, without a Mac test first).
- A device now also deletes its models when its app closes, those the hub has byte for byte (`Sync.free_models`,
  Noah), before the sessions, so a session kept only for a downloaded model's levels goes too. Models only the device
  has stay. Next start, download the models again in System > Sync. Tried on the Mac the same day: on close it
  deleted the downloaded sign network and then `2026-10-21_141651_noah`, both still on the hub and in the backup.

**Mac hum test.** `emg-reading/hum_test.py` records the relaxed arm for a set time under a label (into
`data/hum-test/`, no session.json so sync leaves it alone) and prints per channel, median over 1 s windows: filtered
rest RMS (shared filter), the raw 60, 120 and 180 Hz lines (`quality.line_rms`, as the app's warning), and raw noise
at 20 to 35 and 205 to 245 Hz (outside the filter band, so neither muscle nor mains). `session` mode does the same on
the rest cues of recorded sessions.
- Baseline on rest cues: the Mac session `2026-10-21_141651_noah` rests at 13.7 uV filtered (median over channels),
  the last 7 desktop sessions (10-16 to 10-20) at 14.0 to 18.3. Out of band the same (20 to 35 Hz 6.4 vs 6.5 to 10.0,
  205 to 245 Hz 1.9 vs 1.8 to 1.9). The raw 60 Hz line is higher on the Mac: 10.8 vs 1.6 to 4.7, 6 to 14 on every
  channel. So the Mac session had 2 to 6 times more mains in the raw data, but the notch takes it out and the filtered
  signal the models and haptics see was no worse.
- Test agreed with the Mac session, to separate the computer from the band placement and the room: band on once, 5 min
  for the pads to settle, same seat, arm relaxed on the table, 60 s each, one receiver at a time (desktop off the
  band's WiFi during the Mac blocks, both get 192.168.4.2). Desktop hands off, desktop with the free hand on the mouse,
  Mac on battery hands off, Mac on charger hands off, charger with the free hand on the palm rest, battery with the
  hand on the palm rest, desktop hands off again. Mac on the desk pad in one spot for all its blocks, where it was on
  10-21 (charger in, same seat as the desktop recordings, so the charger hands-off block matches that session). The Mac is worse only if its
  battery hands-off block is over both desktop blocks by more than they differ from each other. The desktop is earthed
  (3-prong), the Mac's charger is a 2-prong one, so touch is the likely path.
- The same test runs from the app on System > Dev (Noah): pick the block, Record block, no need to close the app.
  Each computer keeps its own blocks in `data/hum-test/`, so the Mac's blocks have to be copied over to compare
  in one table.

## Open

- Backup: check that Proton Drive finished uploading `bb-sessions` to the cloud. Before the handoff, move the backup
  to the lab's storage (Sync > Backup folder).
- Hub: per-person consent before other people record.
- Tracking switch: an F session with tracking off lost 0.04 % of samples; compare with tracking on, then merge
  `tracking-toggle`.
- Mac hum: run the A/B test agreed with the Mac session (see "Mac hum test", `emg-reading/hum_test.py`).
- Delay test: run it on the desktop too. Its flash result minus the LED test's 37 ms is that screen's own delay,
  which shows how far the flash method overestimates (the MacBook's 43 ms shift includes its screen). Rerun the
  MacBook once to check the 88 / 15 ms light and dark split repeats.

- Trigger mode on the arm: tune clench level and timings live, count wrong picks (short clench read as two, a grab
  read as a clench), check opening still registers with the distance check on, haptic confirmation of the picked grip.
- Cursor tab parked; if a pad fix or socket electrodes change the picture, rerun the practice (logs in `data/cursor/`).
- Control tab: try it on the arm (grab, hold, release; count wrong grips and twitches), check the 0.9 s to lock
  against fewer votes, and whether pinch closes far enough at gain 1.5 (per-grip max levels or a higher gain).
  Try the distance check and rest_or_class release live (retrain the LDA first).
- Blend mode live: calibrate with a short protocol A session first, tune with the live loop (set settings over
  the API, Noah tests). Then log Control runs to a file and a cued live test with scores.
- Arm animation for cues: the forearm and wrist are drawn for the wrist cues since 2026-10-19. Still missing: the
  upper arm (D's arm raise, reach, lean), the forearm lying along the table in the table posture, and the wrist
  movements left out of protocol C.
- Sign network (in `ml/` with calibration since 2026-10-19; calibrated score flat from 1 training session on, see the
  2026-10-21 learning curve): try it live against the sign LDA, and trigger mode with
  it on the Control tab (sign map, clench level against the sign session's levels); decide whether open stays in the
  set (seven F sessions 2026-10-20: calibrated mean 95.3 % without it, 91.5 % with, worst 91.9 / 85.6 %). Epochs: 20 since the 2026-10-19 sweep, recheck with
  more sessions. Later: one label path for grips and signs (both are held positions), without changing grip
  training results.
- Sign research (`sign-research.md`, tests and times in its section 7): "not a sign" class for wrist flexion on the
  D sessions; calibrated ring network on the 2 to 4 sign sets (1 to 2 h GPU, ask first); still gate and 12 votes
  live in lock mode with open, fist, point; record a protocol E session and run `screen_e.py`; then build protocol F
  (confirmation on 3 days) from its best set; Ninapro DB3 check if downloaded.
- Record and play: train the classifier on protocol C's `custom` labels and play back each movement's target
  (or its recorded camera trajectory) under the control layer.

- Camera delay: re-measure with `diagnostics/camera_led/led_delay.py` only if the camera, its capture settings, the
  grab code or the computer change, and with `diagnostics/armband_rtt/armband_rtt.py rtt` if the band, its
  firmware or the WiFi setup change; then update `CAMERA_DELAY_S`.
- Angles (screen 2026-10-08): no change to the network beats the current baseline on the existing recordings, keep
  it (at least 5 min of same-day fine-tune). If the guard bug is ever fixed in a rerun, guard around each test window.
  Next: record protocol B sessions (DB8 movements, forearm on the table; first one 2026-10-13 passed the pilot bar,
  rerun the time-split test every 2 or 3 sessions) and a protractor check of the camera, to
  split the remaining gap into protocol, sensors and labels; then a live control test, since offline r doesn't
  predict live control.

- Finger angles (priority since 2026-10-05, the recording plan is in the angles item above). Moving windows are
  weighted since 2026-10-05. Still to try: an angle head with its own temporal part on the shared backbone,
  calibration that includes finger movement in every posture, per-session open/fist normalisation of the labels,
  causal convolutions so the network can stream on an ESP32-S3.
- More sessions with the 7 grips, firm holds and a marked band position.
- Quick recalibration in the app: a short calibration-only recording (1 to 2 repetitions of the grips) after
  putting the band on. The Deep learning tab can calibrate on any session, but there is no short recording type yet.
- Test per-session normalisation on unseen sessions. Rotation augmentation only for LDA, the ring network ignores
  whole-electrode rotations by construction.
- `ml/cache/` is about 50 MB per session (every window stored, each sample 4 times). Store the filtered signal once
  and cut windows on load, and drop cache files of sessions that no longer exist. System > Dev has a Clear cache button since
  2026-10-22 (895 MB, 16 sessions, that day).
- Measure false grips per minute in everyday activity, and a "hold to switch" rule.
- Bring latency under 200 ms: shorter windows and vote, at some cost in accuracy.
- `ml/` foundations: session cache, dataset builder, shared evaluation harness, model registry, continual update
  flow tested with LDA first.
- Record one mirrored session to check the channel flip on real data. (Channel order round the band confirmed from
  the MindRove manual, 2026-10-16.)
- Motor pickup: run `diagnostics/motor_pickup.py` with the motor band below the pads, relaxed and with a steady
  clench.
- Motors: firmware 0.2.1 is flashed and channels 0 and 1 follow the duty. When the ULN2003, MOSFETs or LEDs go
  back on, set `OUTPUTS_INVERTED` to false. Still to measure: lowest duty that starts a motor (the haptics floor),
  how many levels can be told apart, and EMG spectra with the motors off and on to see if they show up.
- Haptics: decide on LRAs vs coin motors. LRAs need an AC drive at their resonance (about 170 to 235 Hz), which is
  inside the 40 to 200 Hz EMG band, so check for motor pickup in the EMG either way.
- Ring network: rerun as sessions come in; fix the rest false alarms after calibration (rest weighting); then
  continual updates with replay and the benchmark.
