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
