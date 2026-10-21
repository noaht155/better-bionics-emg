"""Joining the armband's WiFi on a Mac (mac_wifi.py): reading the network scan."""
import ml  # noqa: F401
import mac_wifi


def test_mac_scan_line():
    lines = ["                     Acanac28726                   -58  6,+1    Y  -- RSN(PSK/AES/AES) ",
             "                    Clark family                   -54  1       Y  -- RSN(PSK/AES/AES) ",
             "      Mindrove_ARB_1234 a4:cf:12:00:11:22 -40  6  Y  -- RSN(PSK/AES/AES) "]
    parsed = [mac_wifi.SCAN_LINE.match(line).groups() for line in lines]
    assert parsed == [("Acanac28726", "-58"), ("Clark family", "-54"), ("Mindrove_ARB_1234", "-40")]
