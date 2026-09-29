# Setup

## Python

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Connecting the armband

1. Turn the armband on and join its WiFi network (`Mindrove_ARB_...`, password `#mindrove`).
2. The armband's DHCP doesn't hand the PC an address, so set a static IP on the WiFi adapter. Run in an admin PowerShell:

   ```powershell
   netsh interface ip set address name="WiFi" static 192.168.4.2 255.255.255.0
   ```

   The armband is at `192.168.4.1`. No gateway is set, so internet stays on ethernet.

3. Check the connection:

   ```powershell
   .venv\Scripts\python check_connection.py
   ```

## Undo the static IP

The WiFi adapter won't work on normal networks until this is reverted. Run in an admin PowerShell:

```powershell
netsh interface ip set address name="WiFi" source=dhcp
```
