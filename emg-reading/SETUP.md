# Setup

## Python

Python 3.14 (Apple Silicon Mac: 3.14 or newer). On a Mac `python3` is Apple's 3.9, too old: install another from python.org or Homebrew and make the venv with it, for example `python3.14 -m venv .venv`. Intel Macs need Python 3.12, because mediapipe's last Intel Mac build (0.10.21) stops there; `requirements.txt` picks the older mediapipe, numpy, opencv and scipy for them by itself. Homebrew no longer installs on Intel Macs, uv gives the venv its own 3.12 without touching `python3`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Connecting the armband

On the Windows desktop all of step 1 and 2 is automated: switch the armband on, then press **Connect armband WiFi** on
the app's Connection tab (or run `powershell -ExecutionPolicy Bypass -File connect_armband.ps1`). It forgets and
rejoins the armband's WiFi, switches to DHCP for 5 s, sets the static IP back, checks the band answers and restarts
the stream. Windows asks for admin rights on the PC's screen each time. The app starts without the armband and
shows a warning until it is connected.

On a Mac the same button finds the `Mindrove_ARB_` network, joins it and restarts the stream. The Mac has one WiFi
adapter, so it is off the home network (and the hub) while it is on the band's; rejoin the home network from the
WiFi menu before syncing. macOS 14.4 and newer can't scan from the app: join the band from the WiFi menu, then press
the button.

The manual steps:

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

   This writes `calibration.json`. Redo it whenever the armband is taken off or moved. In the app this short
   calibration is only for troubleshooting: the haptics start with every prediction model, with levels from the
   session the model was trained or calibrated on.

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
- The desktop's camera delay is 28 ms (`devices.json`), from an LED test of the camera (37 ms) and the armband round trip (17 ms). Sessions no longer have sync taps. Older sessions still have `camera_delay_s` from their taps in `session.json`; it is not used (the taps read 96 to 168 ms). `python sync.py data/<session>` still shows an old session's taps.
- The start page asks which device (computer + camera setup, `devices.json`) you record on. It picks the camera, and each session stores the device, so training uses that device's camera delay. Only this computer's devices can be picked; add a new setup to `devices.json` and commit it. A new setup is measured once with the delay test on System > Dev (screen flashes filmed in a mirror and the armband round trip, no ESP32), which saves the result into `devices.json`. Until then its sessions use 28 ms.
- On a Mac the first start has no camera: macOS only asks for camera permission from the main thread and the app opens the camera in its own. Run `.venv/bin/python -c "import cv2, time; c = cv2.VideoCapture(0); time.sleep(5)"` once, allow the camera for the terminal app, then start the app.
- `--camera 1` picks another webcam, `--no-camera` records EMG only, `--host 0.0.0.0` allows a tablet on the same network.
- `--replay data/2026-09-30-filter` plays recorded EMG instead of the armband, `--camera clip.mp4` uses a video file, for testing the app.

The other tabs run the existing programs on the app's armband stream: Signals (`live_plot.py`, f and s work there too), Calibrate (`calibrate.py`), Haptics (`haptics.py`, plus the `esp32_link.py` output test) and Connection (`check_connection.py`). The scripts still work on their own. Stop the app first, they would both try to open the armband. Haptics running during a recording is logged in the session's events.

## Recording on another computer (hub)

Any computer with the app and an armband can record. The desktop with the GPU is the hub: the other computers (devices) send it their sessions, train on it from their Train tab and download its models. Both need to be on the tailnet.

On the desktop, once per device, then start the app as the hub:

```powershell
.venv\Scripts\python hub.py add-device mac
.venv\Scripts\python app.py --hub
```

`add-device` prints a token once, only its hash is kept in `hub.json`. `hub.py remove-device mac` stops it working straight away, `hub.py devices` lists them. With `--hub` the app listens on every network, but other computers only get the hub's requests and the Train tab's, with a valid token; the armband, haptics, camera and live page stay this computer's. Windows asks once whether Python may accept connections; allow it for the network the Tailscale adapter is on.

On the device, System > Sync: the hub's address (`http://<desktop tailnet name>:8000`) and the token, then Connect.

- **Send new sessions** uploads every ended session the hub doesn't have, in 4 MB chunks. A broken connection carries on from where it stopped on the next send. The hub keeps the files in `data/.incoming/` until every file matches its hash, then moves the session into `data/` and evaluates it like one recorded there. A session name the hub already has is never overwritten.
- **Train tab**: pick The hub at the top. The session list, training, calibration and evaluation are then the hub's, the live test stays on this computer. An LDA trained this way isn't made live on the hub.
- **Download ticked** fetches models from the hub, and the session each one takes its effort and haptic levels from if the device doesn't have it. It also swaps Control mappings and profiles both ways; a name both computers have with different content stays as it is on each.
- Networks need torch on the device to run live, the CPU build is enough.
- New sessions note the computer they were recorded on (`device` in `session.json`).

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

For the networks in `ml/` (training desktop only), also install PyTorch with CUDA:
`emg-reading/.venv/Scripts/python -m pip install -r ml/requirements.txt`.

Without the app, from the repo root: `emg-reading/.venv/Scripts/python -m ml.gesture_model emg-reading/data/<session> emg-reading/data/<session> ... [--save]`. The model code lives in `ml/` (see `ml/__init__.py`).
