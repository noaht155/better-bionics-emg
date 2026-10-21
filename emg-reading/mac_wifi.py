"""Joining the armband's WiFi on a Mac, the counterpart of connect_armband.ps1. The Mac gets its address from the
band by DHCP, so it only has to find the band's network, join it and wait for a 192.168.4.x address.

The Mac has one WiFi adapter: while it is on the band's network it is off the home network, and Tailscale (the
hub) with it. Going back is done from the WiFi menu.
"""
import re
import subprocess
import time
from pathlib import Path

ARMBAND_PREFIX = "Mindrove_ARB_"
ARMBAND_PASSWORD = "#mindrove"
ARMBAND_SUBNET = "192.168.4."
# Removed in macOS 14.4, there the band has to be joined from the WiFi menu once (joining by name still works)
AIRPORT = Path("/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport")
ADDRESS_TIMEOUT_S = 20
# One line of `airport -s`: right-aligned SSID, the BSSID (blank without location access), signal in dBm
SCAN_LINE = re.compile(r"^\s*(.+?)\s+(?:[0-9a-f]{2}(?::[0-9a-f]{2}){5}\s+)?(-\d+)\s")


def _run(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()


def device():
    """The WiFi adapter's name, en0 on most Macs."""
    ports = _run("networksetup", "-listallhardwareports")
    found = re.search(r"Hardware Port: Wi-Fi\nDevice: (\S+)", ports)
    if not found:
        raise RuntimeError("this Mac has no WiFi adapter")
    return found.group(1)


def current(dev):
    """Name of the network the adapter is on, or None."""
    found = re.match(r"Current Wi-Fi Network: (.+)", _run("networksetup", "-getairportnetwork", dev))
    return found.group(1) if found else None


def armbands():
    """Armband networks in range, strongest first. None when this macOS can't scan from the command line."""
    if not AIRPORT.exists():
        return None
    found = {}
    for line in _run(str(AIRPORT), "-s").splitlines()[1:]:
        m = SCAN_LINE.match(line)
        if m and m.group(1).startswith(ARMBAND_PREFIX):
            found[m.group(1)] = max(found.get(m.group(1), -999), int(m.group(2)))
    return sorted(found, key=found.get, reverse=True)


def join(dev, name, password):
    """networksetup exits with 0 even when joining fails, it only prints why."""
    out = _run("networksetup", "-setairportnetwork", dev, name, password)
    if out:
        raise RuntimeError(f"joining {name} failed: {out}")


def wait_for_address(dev, timeout=ADDRESS_TIMEOUT_S):
    until = time.time() + timeout
    while time.time() < until:
        address = _run("ipconfig", "getifaddr", dev)
        if address.startswith(ARMBAND_SUBNET):
            return address
        time.sleep(0.5)
    raise RuntimeError(f"no {ARMBAND_SUBNET}x address after {timeout} s (now {address or 'none'}), switch the armband "
                       "off and on and try again")
