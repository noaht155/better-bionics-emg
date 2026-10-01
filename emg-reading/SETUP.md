# Setup

## Python

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Connecting the armband

1. Turn the armband on and join its WiFi network (`Mindrove_ARB_...`, password `#mindrove`).
2. Check the WiFi address with `ipconfig`. It should be `192.168.4.x`. On the desktop PC (Realtek 8852CE) the armband's DHCP reply never arrives and the address is `169.254.x.x`. The MacBook gets an address normally. If it's `169.254.x.x`, set a static IP on the WiFi adapter. Run in an admin PowerShell:

   ```powershell
   netsh interface ip set address name="WiFi" static 192.168.4.2 255.255.255.0
   ```

   The armband is at `192.168.4.1`. No gateway is set, so internet stays on ethernet.

   **After every armband power-up, toggle DHCP once before streaming.** With only the static IP the armband streams nothing (0 samples) after it restarts. Switching the adapter to DHCP for about 20 s and back fixed it on 2026-09-30, likely because the armband only streams to a device that has asked it for an address since it booted. Admin PowerShell:

   ```powershell
   netsh interface ip set address name="WiFi" source=dhcp
   # wait about 20 s, ipconfig shows 169.254.x.x, that's expected
   netsh interface ip set address name="WiFi" static 192.168.4.2 255.255.255.0
   ```

3. Check the connection:

   ```powershell
   .venv\Scripts\python check_connection.py
   ```

## Running the haptics

Close the PlatformIO monitor first, only one program can use the ESP32's serial port.

1. Put the armband on and calibrate. Follow the prompts (relax, then squeeze a fist):

   ```powershell
   .venv\Scripts\python calibrate.py
   ```

   This writes `calibration.json`. Redo it whenever the armband is taken off or moved.

2. Start the haptics. Ctrl+C stops it and turns the outputs off:

   ```powershell
   .venv\Scripts\python haptics.py
   ```

The ESP32 is reached over USB if it's plugged into this computer, otherwise over WiFi. It joins the armband's network by itself at `192.168.4.10`, so it only needs power. Add `--wifi` to use WiFi while the USB cable is plugged in. USB has priority: the ESP32 refuses WiFi commands for 3 s after any USB command.

`live_plot.py` shows the signals (f toggles filtering, s the spectrum). `esp32_link.py` runs a self-test of the ESP32 outputs.
Add `--synthetic` to any of these to run without the armband.

## Recording sessions

Connect the armband, plug in the webcam, then:

```powershell
.venv\Scripts\python app.py
```

Open http://localhost:8000, fill in the form and start. The first run downloads MediaPipe's hand model (about 8 MB) into `models/`.

- Put the camera where it sees the tracked hand in every posture. The warning bar shows when the hand is out of view.
- Space pauses and continues. b marks the current cue as bad, or the gesture before it during the first 1.5 s of a rest. Stop needs two clicks.
- Each session is saved to `data/<date>_<time>_<subject>/` as it records. `session.load_session()` reads it back.
- Tick **Practice run** to go through the same cues, camera tracking and warnings without saving anything, for checking the camera position or the setup. The tick is not remembered between page loads, so a real session is never skipped by accident.
- The camera delay is measured from the sync taps when a session ends and saved as `camera_delay_s` in `session.json`. Subtract it from the camera times. For a quick check, start a session with no postures ticked: it only does the taps. `python sync.py data/<session>` shows each tap, `--save` writes the result again.
- For the taps, keep the hand in view, lift it and slap the table hard once. The tracker must not lose the hand during the slap, a tap with missed frames is skipped.
- `--camera 1` picks another webcam, `--no-camera` records EMG only, `--host 0.0.0.0` allows a tablet on the same network.
- `--replay data/2026-09-30-filter` plays recorded EMG instead of the armband, `--camera clip.mp4` uses a video file, for testing the app.

The other tabs run the existing programs on the app's armband stream: Signals (`live_plot.py`, f and s work there too), Calibrate (`calibrate.py`), Haptics (`haptics.py`, plus the `esp32_link.py` output test) and Connection (`check_connection.py`). The scripts still work on their own. Stop the app first, they would both try to open the armband. Haptics running during a recording is logged in the session's events.

## Undo the static IP

The WiFi adapter won't work on normal networks until this is reverted. Run in an admin PowerShell:

```powershell
netsh interface ip set address name="WiFi" source=dhcp
```

## If the desktop PC gets no address or no data

The armband streams fine to the MacBook, so this is the PC's WiFi card. DHCP still fails on the PC, so the static IP is always needed there. On 2026-09-30 streaming came back after steps 1 and 2 plus the static IP. It's not clear which step did it, so try them in order.

1. Restart the PC, reconnect to the armband and check `ipconfig` again.
2. Turn off two adapter features. Run in an admin PowerShell:

   ```powershell
   Set-NetAdapterAdvancedProperty -Name WiFi -DisplayName "MAC Randomization" -DisplayValue "Disabled"
   Set-NetAdapterAdvancedProperty -Name WiFi -DisplayName "Multi-Channel Concurrent" -DisplayValue "Disabled"
   Restart-NetAdapter -Name WiFi
   ```

   To undo, run the same commands with `"Enabled"`.

3. Use a USB WiFi dongle for the armband (MindRove recommends one).

The adapter is called `WiFi` on this PC. Check the name with `Get-NetAdapter` on another machine.

## Gesture model (protocol A)

On the Train tab, tick the sessions to use and press Train and evaluate. Each session is tested on a model trained on the others (leave one session out), so it needs at least two sessions to say anything about new sessions. The result shows accuracy, how often rest is taken for a gesture, the confusion matrix and accuracy per left-out posture. The model is saved to `models/gestures/` and starts predicting live: the Train tab shows the probabilities, the Record tab's Model prediction view shows the predicted gesture as a hand.

Without the app, from the repo root: `emg-reading/.venv/Scripts/python -m ml.gesture_model emg-reading/data/<session> emg-reading/data/<session> ... [--save]`. The model code lives in `ml/` (see `ml/__init__.py`).
