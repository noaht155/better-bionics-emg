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

Where it stands (two sessions, one person):

| | 11 gestures | 7 grips |
|---|---|---|
| Same session, unseen repetition | 73 % | 81 to 87 % |
| Other session (band moved) | 34 to 42 % | 51 to 52 % |
| Wrong grip shown, same position, 0.8 threshold | 4.7 % | 4.4 % |

Most of the jump from 74 to 87 % comes from dropping the four weakest gestures, not from a better model. Accuracy on
the training data is only 4 to 6 points above held-out data, so the model isn't overfitting.

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
- The dropped-samples figure in the summary and `session.json` counted every drop since the app started, not per
  session (a sync check with 0 real drops showed 62). Fixed; earlier session files still have the inflated number,
  count drops from the package numbers instead.
- With the camera read in its own thread, frames arrive in pairs (two within 10 ms, then about 55 ms), still
  30 fps on average. Arrival times jitter by up to a frame, which widens the tap spread (69 ms on the first check,
  delay 77 ms). Fixable offline with a straight-line fit of time against the grab number, like the EMG clock.
- Calibration max levels vary 20 to 100 % between single clenches (effort isn't controlled), rest levels within
  about 1 uV. Only affects the haptics, left as is.

## Open

- More sessions with the 7 grips, firm holds and a marked band position.
- Quick recalibration in the app: 1 repetition after putting the band on, retrain with one click.
- Test rotation augmentation and per-session normalisation on unseen sessions.
- With 5+ sessions: a small network as a placement-independent feature extractor with an LDA head per fitting.
- Measure false grips per minute in everyday activity, and a "hold to switch" rule.
- Bring latency under 200 ms: shorter windows and vote, at some cost in accuracy.
- Protocol B (continuous joint angles), which may also replace LDA for gestures.
