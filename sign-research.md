# Sign research: hand signs the band tells apart across days

Question: which attempted hand signs give forearm EMG patterns distinct enough to tell apart every time the band goes
on, so each can be mapped to a useful prosthetic hand output, whatever the sign looks like as a hand shape. Recorded
2026-10-16 on the existing sessions (one person, Noah, able-bodied, 7 days) and Ninapro DB8. Scripts in
`diagnostics/signs/`, results in `diagnostics/out/signs/`.

Short answer:
- From the signs already recorded, only **2 signs + rest** clear the accuracy bar on every session, and **no set**
  clears the false pick bar on the worst protocol D session. The nearest sets are **open, fist, point** and
  **open, pinch, point** (3 signs + rest): 94 to 95 % balanced, worst session 87 to 89 %, and 0.0 / 1.1 false
  picks per minute on the two D sessions with a longer vote and the still gate.
- What separates is effort and which side of the forearm works (open and point: extensors; fist: everything, hard).
  Low-effort signs (single fingers, the thumb grips against each other) don't, and more calibration doesn't fix them.
- Calibration has to cover the arm postures: the same number of holds from one posture scores 14 to 20 points
  worse in the others.
- New signs are the only way to a bigger set. Protocol E below screens them, protocol F confirms a chosen set across
  days.

## Success bar used

From the task: on a session never trained on, after a short calibration at the start of that session, at least
90 % balanced accuracy with rest as a class, per session and on the worst session; and under 1 false pick per minute
during the protocol D activities (`2026-10-14_181042`, `2026-10-15_105900`), counting any pick in any activity or
the rest cues after them (Noah, 2026-10-16). Sets with finger signs get accuracy only, since D's calibration block
has no finger cues (Noah, 2026-10-16).

The LDA here is trained only on the session's own calibration, so no other session is ever in training; "across
days" means the calibrate-then-use procedure has to work on every one of the 7 days, and the worst session is what
counts.

## Data

| Signs | Sessions | Days | What it allows |
|---|---|---|---|
| 6 held grips + rest | 16 | 7 | every analysis |
| 5 single finger bends | 11 (10 with every finger in both splits) | 5 | accuracy only |
| 14 DB8 movements (protocol B), 17 library movements (protocol C) | 2 and 1 | 1 each | within a session only, not used again here (C: devlog 2026-10-15) |
| Ninapro DB8, 9 movements | 10 able-bodied, 2 transradial amputees | 1 day each | cross-check with other people |

Finger bends are found on camera (`diagnostics/signs/data.py`): the cued finger's flexion above 60 % of its range in
the cue for 0.3 s or more; the k-th bend in a posture is repetition k, so a finger cue gives 3 repetitions like the
grips. Of 125 finger cues, 83 show 3 bends, 28 show 4 or 5 (the "stop halfway" splits a bend) and 14 show 2 or
fewer; the first 3 are used.

Repetition k of a sign means its k-th hold in each posture, so "1 repetition" is 1 hold per sign in single-posture
sessions (B, C, D) and 3 in the 3-posture protocol A sessions. Features and model as the app: protocol A's
extended features (100), shrinkage LDA, 200 ms windows every 50 ms, shared causal filter, mirrored sessions flipped.

## 1. What separates and why

### On this data (`diagnostics/signs/why.py`, 16 sessions)

Per sign, RMS during the hold over rest (median over channels), and the channel pattern with effort removed (log
ratio minus its median over channels, mean over sessions):

| Sign | RMS over rest | Pattern, ch 0 to 7 |
|---|---|---|
| point | 4.2x | -0.15 -0.04 +0.05 +0.10 +0.15 +0.07 -0.41 -0.21 |
| open | 3.7x | -0.10 -0.07 -0.02 +0.08 +0.18 +0.09 -0.19 +0.01 |
| fist | 3.2x | +0.02 -0.07 -0.07 +0.17 +0.14 +0.31 -0.44 +0.16 |
| key | 2.9x | -0.14 -0.04 -0.00 +0.07 +0.20 +0.09 -0.25 +0.07 |
| tripod | 2.6x | -0.01 +0.01 -0.00 +0.10 +0.02 +0.00 -0.21 +0.00 |
| pinch | 2.1x | +0.02 +0.02 -0.03 +0.06 +0.00 +0.00 -0.27 -0.03 |
| finger bends (5) | 1.4 to 1.8x | all within about 0.1 of each other |

Separation within a session (d' on the 8 log RMS values, in units of the within-sign spread; mean over sessions,
lowest session in brackets):
- The single finger bends against each other: 1.3 to 2.0 (0.4 to 0.7). Tripod and key 2.2 (0.8), pinch and
  tripod 3.1 (1.0), open and point 2.9 (1.0), open and fist 3.1 (2.1).
- Pinch against open and against point 7.3 (3.4, 2.3); open and point against any finger bend 7 to 12.

So:
- **Effort** is the biggest factor. The three signs at 3 to 4x rest are the ones the classifiers have always found;
  finger bends at under 2x rest look like rest (20 to 35 % of their windows are called rest) and like each other.
- **Side of the forearm.** Open and point load ch 3 to 5 and spare ch 0; fist adds ch 5 and 7. The thumb grips
  share one pattern at lower effort. Ch 6 (the extensor pad on the module) is low for every sign relative to the
  others, as found before (devlog 2026-10-14).
- **Within a confusable group, one member is fine.** Pinch is mixed up with tripod, not with open or point, so a set
  with pinch and without tripod and key keeps pinch. The same holds for the protocol C groups (devlog 2026-10-15):
  pick one member per group.
- **Held vs moving.** The moving finger bends are the worst signs here, but they are also the weakest, so this data
  can't separate "moving" from "weak". DB8's moving movements through 8 channels at 500 Hz score 35 to 63 % for
  9 classes (section 3), far below our held grips at similar N. Timing patterns on effort (trigger mode's clench
  counts) are a different kind of moving sign and need no classifier (devlog 2026-10-15).

### From the literature

- Effort level changes the pattern: force variation degraded classification in 9 transradial amputees doing 6
  classes at 3 force levels, and training with all force levels reduced the loss (Al-Timemy et al. 2016, IEEE TNSRE
  24(6)). For signs it means each sign should
  be cued at one firm level and calibrated at it.
- Intrinsic hand muscles carry information the forearm doesn't: on the intact arm of 5 transradial amputees, EMG
  from intrinsic hand muscles was added to show its value (Li, Schultz, Kuiken 2010, IEEE TNSRE 18(2), 10 movements,
  real time); with abstract decoders, intrinsic hand muscles gave higher performance and faster learning than
  forearm muscles (Dyson, Barnes, Nazarpour 2018, J Neural Eng 15(5), able-bodied). After a transradial amputation
  the intrinsic muscles are gone, so signs that rely on them (finger spread, fine thumb opposition) should be
  expected to lose most of their pattern on an amputee even if they work on Noah.
- Amputees reach fewer separable classes than intact subjects: 40 movements, 5 amputees vs 40 intact, maximum
  average accuracy 61.1 % for amputees, 15.9 points below intact (Atzori et al. 2014, EMBC, "Classification of hand
  movements in amputated subjects by sEMG and accelerometers"); 50 movements, 3 amputees and 40 intact, balanced up
  to 58.6 %, but per subject a subset of up to 9 movements above 90 % (Atzori et al. 2014, EMBC, "Natural control
  capabilities of robotic hands by hand amputated subjects"). Electrode setup not given in the abstracts; how the
  test data was split isn't either, so treat both as upper bounds for our 8 dry pads.

## 2. How many signs, and which set

### Accuracy (`diagnostics/signs/subsets.py`)

Every subset of the pool scored per session from one LDA fit restricted to the subset's columns; the best set per
size was refitted on its own classes (refit within 2 points of the shortcut everywhere). "Chosen without the session"
picks the set on the other sessions only and scores it on the held-out one, the honest number for a set chosen on
past data.

Calibration 2 repetitions per posture (the 10 protocol A sessions with 3 postures and 3 repetitions, so 6 holds per
sign), balanced
accuracy with rest as a class, mean / worst session:

| N | Best set (worst session) | Balanced | Signs only | Chosen without the session |
|---|---|---|---|---|
| 1 | tripod | 98 / 96 % | 97 / 94 % | 95 / 83 % |
| 2 | tripod, point | 97 / 95 % | 97 / 94 % | 95 / 82 % |
| 3 | open, pinch, point | 95 / 90 % | 94 / 89 % | 94 / 89 % |
| 4 | open, fist, pinch, point | 94 / 88 % | 93 / 85 % | 94 / 88 % |
| 5 | open, fist, pinch, tripod, point | 87 / 80 % | 85 / 76 % | 86 / 77 % |
| 6 | all six grips | 83 / 75 % | 81 / 71 % | 83 / 75 % |

Refitted on their own classes, the named sets at 2 repetitions (`diagnostics/out/signs/`): open, point 95.1 / 88.7 %;
fist, point 96.6 / 85.5 %; open, fist, point 94.5 / 87.4 %; open, pinch, point 94.4 / 89.3 %; open, fist, pinch,
point 93.5 / 88.1 %. With the 5-window live vote: same within 2 points.

Calibration 1 repetition per posture (16 sessions, 1 to 3 holds per sign): open, fist, point 90.2 / 67.7 %; open,
pinch, point 90.3 / 73.6 %; open, fist, pinch, point 88.9 / 70.2 %. The worst sessions are the single-posture
sessions (1 hold per sign). One hold isn't enough.

With the 5 finger bends in the pool (10 sessions): no finger bend enters a best set below N = 4 (flex_index at 1
repetition, N = 5 at 2). All 11 signs: 59 / 51 % at 1 repetition.

So: N = 2 clears 90 % on the worst session; N = 3 sits on the line (90 % worst for the best set chosen on all
sessions, 89 % when chosen without the session); N = 4 is 2 points under. Rests on 10 sessions from 7 days.

### Calibration amount and posture (`calibration_curve.py`, `posture_cal.py`)

Same holds per sign, different postures (10 sessions with 3 postures, tested outside the table posture):

| Set | 3 holds, all at the table | 1 hold in each of 3 postures |
|---|---|---|
| open, fist, point | 76.2 / 65.4 % | 92.1 / 75.9 % |
| open, pinch, point | 76.6 / 53.4 % | 90.2 / 74.4 % |
| open, fist, pinch, point | 70.6 / 56.3 % | 88.3 / 77.6 % |
| all six | 54.8 / 48.5 % | 75.5 / 62.1 % |

Calibrating in one posture costs 14 to 20 points in the others. Part of that may be time (the one-posture holds are
all in the first minutes), not checked apart. Fougner et al. 2011 (IEEE TNSRE 19(6), 10 able-bodied, LDA) found
the same: classification error 3.8 % in the training position, 18 % across positions, 5.7 % when
trained in several positions and 5.0 % with accelerometers added.

Calibration length for a 3-sign set at 2 holds per sign in each of 3 postures: 3 x 6 x (4 s hold + 3 s rest), about
2 min plus the posture changes.

### False picks (`false_picks.py`, `replay_a.py`)

LDA on the set + rest from each D session's own grip block (2 holds per sign, table), the app's control layer in
lock mode (`control.py` DEFAULTS: threshold 0.7, distance check 6, rest_or_class release), every 50 ms step of the
activities and the rest cues after them, fresh controller per cue and after each pick. False picks per minute,
`181042` / `105900`:

| Set | Lock, defaults (8 votes) | 12 votes + still gate |
|---|---|---|
| point | 4.7 / 16.9 | 0.5 / 3.9 |
| open, point | 1.6 / 6.7 | 0.0 / 1.2 |
| tripod, point | 1.9 / 13.4 | 0.0 / 1.7 |
| open, fist, point | 0.5 / 6.3 | 0.0 / 1.1 |
| open, pinch, point | 0.8 / 6.4 | 0.0 / 1.5 |
| open, fist, pinch, point | 0.6 / 6.8 | 0.0 / 1.1 |
| all six grips | 0.0 / 4.4 | 0.0 / 0.6 |

(The still gate is trigger mode's: windows count as rest while the accelerometer moves more than in the calibration
holds; here applied in front of lock mode as a test, lock mode doesn't have it yet.) In the rest cues after the
activities, with both changes: 0 to 1.4 per minute for the sets in the table (pinch, point alone: 8.6).

Nearly all remaining false picks are **wrist flexion read as point** in `105900`: for open, fist, point every one of
the 10 activity picks came in the 0.7 min of wrist bending, plus 1 in a rest cue. The table posture keeps the forearm still, so the still gate doesn't see it. More signs
lower the false pick rate (each sign has its own cloud, so the distance check rejects more), which is why the 6-grip
set does best here.

What the two changes cost real signs, protocol A repetition 3 through the same control layer (10 sessions,
calibrated on repetitions 1 and 2), first pick in the cue:

| Set | Defaults: right / wrong / none, time | 12 votes + gate |
|---|---|---|
| open, fist, point | 94 / 6 / 0 %, 0.83 s | 96 / 0 / 4 %, 1.05 s |
| open, pinch, point | 93 / 1 / 6 %, 0.84 s | 88 / 0 / 12 %, 1.07 s |
| open, fist, pinch, point | 91 / 4 / 5 %, 0.87 s | 88 / 1 / 11 %, 1.09 s |
| all six grips | 82 / 4 / 13 %, 1.01 s | 76 / 2 / 23 %, 1.31 s |

The 12 votes trade wrong picks for missed ones (pinch is held at low effort and flickers) and add about 0.2 s. The
still gate costs nothing here (the arm is still in protocol A). Worst session for open, fist, point: 78 % right.

### Trade-off

| N | Balanced, worst session (2 repetitions) | False picks, worst D session (12 votes + gate) | Clears both |
|---|---|---|---|
| 1 | 96 % | 3.9 (point) | no |
| 2 | 95 % (tripod, point) | 1.2 to 1.7 | no, false picks |
| 3 | 87 to 90 % | 1.1 to 1.5 | no, both just short |
| 4 | 88 % | 1.1 | no |
| 6 | 75 % | 0.6 | no, accuracy |

**No N clears the whole bar on the worst session with today's signs.** N = 2 and 3 are within 0 to 3 points and
0.1 to 0.7 picks per minute. The false pick gap is one activity in one session (wrist flexion) and the D sessions
were only calibrated at the table; a "not a sign" class for wrist movement is the obvious next test (section 5).

## 3. Other people and amputees (DB8, `db8_signs.py`)

DB8 (Krasoulis, Vijayakumar, Nazarpour 2019, Frontiers in Neuroscience, "Effect of user adaptation on prosthetic
finger control with an intuitive myoelectric decoder"): 10 able-bodied and 2 right-hand transradial amputees (S11,
S12, per the Newcastle data record), 16 Delsys Trigno sensors in two rows, 9 movements repeated with both hands at
once (mirrored), glove on the other hand. Reduced to our setup: the ring nearer the elbow (8 sensors), 500 Hz, the
shared filter, our features and LDA. Calibrated on repetitions 1 and 2 of acquisition 1, tested on acquisitions 2 and
3 (same day). Movements are flex and extend cycles, not holds, and the sensors are commercial Trigno sensors,
not our band's pads.

- All 9 movements: able-bodied 35 to 63 % balanced, amputees 49 % and 43 %. 10 repetitions of calibration instead
  of 2 change it by -1 to +20 points.
- Best single movement for the able-bodied group: lateral grip (88 % mean, 82 % worst); for the amputees it scores
  93 % and 74 %. The amputees' own best is thumb abduction (96 % both), which isn't in the able-bodied group's best
  set at any N.
- Best 3 for the able-bodied group (thumb flexion, index point, lateral): 74 / 63 %; amputees 73 and 76 %. The
  amputees' own best 3 (thumb abduction, index flexion, ring + little flexion): 85 and 84 %, picked on the test
  data, so optimistic.

What it shows: on other people the same reduced setup also gets only a few signs above 90 %, the two amputees are
not worse than the able-bodied subjects on average, and **the best signs differ between people**, most clearly
between the amputees and the able-bodied group. A sign set has to be chosen per person. Two amputees, one day each,
moving movements: a weak test of "holds up for amputees". Ninapro DB3 (11 transradial amputees, about 50 movements)
would be the stronger check (follow-up, needs Noah to download it).

## 4. Co-adaptation

### Literature

- **User training works for amputees.** 4 unilateral amputees (2 transradial, 2 wrist disarticulation), 8-channel
  cuff, LDA, 9 classes, 10 one-hour sessions over 3 weeks with visual feedback of how consistent and distinct their
  movements were: movement completion 70.8 % to 99.0 %, post-selection accuracy 77.5 % to 94.4 %; 8 hours later
  without retraining 95.8 % and 91.0 %. Each subject found their own strategy (bigger distances between classes,
  tighter clusters, or both) (Powell, Kaliki, Thakor 2014, IEEE TNSRE 22(3)).
- **Learning together beats calibrating once.** Closed-loop co-adaptation (user and regressor learning at the same
  time towards one target) beat open-loop training in 10 able-bodied subjects; 2 people with congenital limb
  deficiency reached similar 2-D proportional control (Hahne et al. 2015, IEEE TNSRE 23(4)).
- **The machine must not adapt too fast.** In a model and experiments of co-adaptation, a low machine gain gave slow,
  incomplete adaptation, a high gain adapted faster but amplified noise and raised errors; a variable gain worked
  best (Couraud et al. 2018, J Neural Eng 15(2), simplified myoelectric wrist force task with a rotation
  perturbation).
- **Abstract mappings are learnable by people with a limb difference.** 25 limb-intact and 8 limb-different
  participants learned abstract control, skills transferred from easy to harder tasks, and it scaled to more
  channels (Dyson, Dupan, Jones, Nazarpour 2020, IEEE TNSRE 28(7)); 4 intrinsic hand muscles learnt within about
  12 minutes for a cursor and a robotic hand (Pistohl et al. 2013, Ann Biomed Eng 41(12), 21 able-bodied).
- **Across days, short supervised recalibration is the established fix.** Over 7 days, 10 able-bodied and 6
  transradial amputees, surface EMG classification error 9.1 % within a day and 21.9 % across days (Waris et al.
  2019, IEEE JBHI 23(4)). Covariate shift adaptation, blending a trained LDA's parameters with a calibration set of
  under a minute, improved accuracy over 5 days in 7 able-bodied and 4 amputees (Vidovic et al. 2016, IEEE TNSRE
  24(9)).
- **Offline accuracy doesn't predict live control.** A pilot in Li, Schultz, Kuiken 2010 found low correlation
  between classification accuracy and real-time performance. Every set picked offline needs a live test.

### Tested here (`adapt.py`)

LDA written out so its parts can be mixed: class means 80 % from this session's calibration and 20 % from earlier
sessions (shifted to this session's rest level), covariance half and half. On 13 sessions with 3 or more earlier
ones, 1 repetition of calibration: open, fist, point 89.8 / 69.8 % to 89.9 / 73.9 %; open, fist, pinch, point
87.5 / 67.4 % to 88.4 / 74.2 %; all six 76.3 / 63.1 % to 78.0 / 71.8 %. Small on the mean, 4 to 9 points on the worst
session. 9 settings were tried on the same sessions, so the best is a little optimistic. Using earlier sessions as
they are (devlog 2026-10-15: 58.8 % with all 14 sessions) is far worse; they have to be pulled to the new session.

### Plan (within the repo's rules: labelled, per session, benchmark-guarded, never from unlabelled live use)

What the machine updates, and when:
1. **Every time the band goes on:** a short labelled calibration (the chosen signs, 2 holds each in each posture used,
   about 2 min). The LDA takes class means mostly from it and the covariance partly from the person's history
   (the `adapt.py` blend). Seconds to compute.
2. **After each completed session:** the session joins the person's history (the LDA's running totals, the continual
   update plan), and the network's per-person transform updates. Benchmark first, roll back if worse.
3. **Every few sessions:** re-choose the sign set from the person's confirmation sessions (section 6). Changing the
   set or the mapping is a big change for the user, so do it rarely (Couraud: a fast machine fights the user's
   learning) and only with a clear gain on the worst session.

What the user gets, so the user adapts too:
- **Separability feedback during calibration** (Powell 2014): after each hold, show its distance to its own sign's
  cloud and to the nearest other sign (the distance check already computes this). Holds that land between two signs
  are redone. A score per sign in the Train tab: how tight its cloud is and how far the nearest other sign is.
- **A practice screen** with the chosen signs and the live classifier output, to practise making each sign the same
  every time, like Powell's training sessions. Its data is never trained on unless it is recorded as a labelled
  calibration.
- **Haptic confirmation** of each pick (already on the Open list for trigger mode), so the user learns the mapping
  without looking.

The user learns over days, the machine updates once per session and no faster, so the user always practises against
a fixed decoder within a session.

## 5. Mapping signs to outputs

Recommended:
- **Open hand: open and release.** It is one of the most reliable signs (and in trigger mode the open hand opens and
  point releases, devlog 2026-10-15). Keep it as the open command, not a grip.
- **Fist: favourite power grip** (cylinder or fist, the person's choice).
- **Point: favourite precision grip** (pinch or tripod) or point itself; the sign doesn't have to look like the
  output.
- **4th sign only if it passes protocol F** for that person: pinch on this data (88 % worst session as part of a
  4-sign set).
- **Everything else cyclable** with effort timing, as trigger mode does now: 2, 3, 4 clenches pick the other grips
  (`pinch_clenches` 2, still gate on: 1.0 false picks per minute at clench level 0.3, 0.4 at 0.4 on the D sessions,
  devlog 2026-10-15).
- Release with two point flicks as now.

False activations, in order of what they cost:
1. Distance check (already on, cutoff 6).
2. Still gate in front of lock mode too, not only trigger mode (no cost on protocol A here).
3. A longer vote for the direct signs (12 instead of 8): fewer wrong picks, about 0.2 s slower, more missed light
   signs.
4. A "not a sign" class from wrist movements recorded in the calibration (section 6, protocol F), since wrist
   flexion read as point is the remaining false pick. The earlier "not a grip" class from all D activities worked
   but added nothing over the distance check (devlog 2026-10-15); this would target the one activity that is left.

Choosing the set per person: from protocol E, keep the signs with high effort and low spread, drop all but one of
any pair with d' under about 3, then take the largest set whose worst session in protocol F clears the bar. The DB8
result says the set will differ between people.

Not recommended for now: single finger bends as signs (35 to 43 % each, mostly called rest); the three thumb grips
together in one set.

## 6. What to record next

Both are new plans beside A to D (`gestures.plan_for`), nothing replaced. Cue kind for the new signs: a held cue
like `hold`, with a label outside `GESTURE_SET` so protocol A training ignores it (like `custom`).

### Protocol E: screening (one session, about 13 min plus 2 posture breaks; in the app since 2026-10-16)

Goal: try many candidate signs on one person quickly and rank them. Built as `gestures.protocol_e_plan` (Record
form, protocol E), analysed with `diagnostics/signs/screen_e.py <session>`.
- Candidates, all finger and hand configurations a transradial amputee can attempt, no wrist, no props, each cued
  "firmly" and following the drawn hand in. Anchors: open, fist, point, pinch. New, chosen to use different extrinsic
  forearm muscles and high effort: thumb out and up with fingers closed (thumbs-up shape), thumb tucked across the
  palm with fingers straight, claw (knuckles straight, fingertips curled), hook (fingers curled, thumb out), index
  and middle straight and together, little finger straight with the others closed, index and little straight
  (horns), all fingers pushed back hard, hand made rigid (squeeze open and closed at once, co-contraction), thumb
  abduction away from the palm (the DB8 amputees' best), ring and little flexed with the others straight (Ninapro
  posture 3, the only ulnar-finger sign with literature support), plus wrist flexion and extension recorded as "not a
  sign" so every candidate gets a distance to the main false pick. 15 signs + 2 wrist cues. Pairs that differ only in
  effort or the thumb (fingers pushed back vs open, hook and thumbs-up vs fist) are scored against each other first.
  Literature behind the list: `diagnostics/out/signs/literature.md` (held finger postures beat grasps and wrist
  movements in Ninapro DB2 and DB3, Ghaderi 2022; thumb grips collapse into one family in every dataset that reports
  confusions; thumb opposition is done by the thenar muscles, which are lost after amputation).
- Layout: table posture 3 rounds, arm forward 2 rounds (the first ticked posture gets the Rounds setting, later
  ones one fewer), each round every sign and wrist bend once in a new order, 6 s per cue (the first 1.67 s is the
  hand closing in and is left out, like protocol C), 3 s rest between. 17 x 5 x 9 s = about 13 min. Signs are
  `custom` cues and the wrist bends `nogrip` cues, so nothing becomes a protocol A grip label, and an E session has
  no grip holds, so `ml.dataset.good_sessions` leaves it out of grip and network training.
- Analysis (minutes, LDA): per sign effort, nearest other sign (d'), recall and what it is taken for; how often the
  held wrist bends are read as each sign after the distance check; the best set for N = 2 to 6. Calibration on the
  table rounds 1 to 2 and the forward round 1, test on the rest.
  Within one session it is an upper bound (devlog 2026-10-15: protocol C lost 31 points between its own rounds and
  later rounds).
- Phantom attempt: Noah can't remove the visual and proprioceptive feedback an amputee lacks. An eyes-closed round
  was left out of the built version: the cue is read off the screen, so it would need spoken cues.

### Protocol F: confirmation (about 15 min, at least 3 days, band off and on between)

Built in a simpler form on 2026-10-16 (`gestures.protocol_f_plan`): the chosen 4 signs and the 2 held wrist bends in
rounds per posture, for training data. The calibration, test and activity blocks below are not in it yet.

Goal: test a chosen set (up to 5 signs) the way it will be used.
- Calibration: every sign 2 times in each of 3 postures, plus rest, plus 2 holds each of wrist flexion and extension
  as a "not a sign" class (about 3 min).
- Test: every sign 3 more times per posture (about 4 min).
- False pick block: 6 protocol D activities, 40 s each (wrist bend, wrist side to side, forearm rotation, arm raise,
  typing, talking with the hands), about 5 min.
- Scored exactly as here: balanced accuracy on the test block per session and worst session, false picks per minute
  through the control layer, picks right / wrong / missed with time to pick. 3 days is the minimum to have a worst
  session; the bar is met when every day passes.

## 7. Tests for Noah to schedule

| Test | What it answers | Estimated time |
|---|---|---|
| "Not a sign" class from wrist movement on the D sessions (`d_nullclass.py` adapted to the sign sets, wrist flexion held out within `105900`) | whether the last false picks go away | under 10 min CPU |
| Ring network, calibrated, on the 2 to 4 sign sets, leave one session out (16 folds, 3 seeds) | whether the network clears 90 % worst session where LDA doesn't | about 1 to 2 h GPU, check one fold first |
| Lock mode with the still gate and 12 votes, live on the Control tab with open, fist, point | live picks, misses, feel | 20 min of Noah's time |
| Protocol E session, then its analysis | which new signs separate | 15 min recording, 5 min analysis |
| Protocol F on 3 days | the bar on a chosen set | 15 min per day, 5 min analysis |
| Ninapro DB3 subset analysis (11 amputees), if downloaded | do the chosen kinds of signs hold for amputees | about 1 h CPU |
| Rerun `subsets.py 2`, `false_picks.py`, `replay_a.py` after each new session | numbers stay current | under 2 min |

## Scripts

All in `diagnostics/signs/`, run from the repo root with `emg-reading/.venv/Scripts/python -m diagnostics.signs.<name>`:
`data.py` (sign windows and finger bends, cached), `subsets.py [K] [vote]` (subset search), `calibration_curve.py`,
`posture_cal.py`, `why.py`, `false_picks.py`, `replay_a.py`, `adapt.py`, `db8_signs.py`.

## Limits

- One person, 16 sessions, one placement routine, pads loose under the band. Every "worst session" is the worst of
  10 or 16.
- Two D sessions, both table, calibrated at the table only. The false pick numbers rest on these two.
- Calibration and test are minutes apart in the same session; nothing here measures a model kept across days
  without recalibrating (the network does that, devlog 2026-10-15).
- The finger bend labels come from the camera, which the end user won't have; for screening on Noah that is fine.
- Subset choice uses the same sessions it reports on, except the "chosen without the session" column.
