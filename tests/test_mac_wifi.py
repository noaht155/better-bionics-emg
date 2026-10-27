"""Joining and forgetting the armband's WiFi (mac_wifi.py, tools.forget_armband_wifi): reading the command output."""
import ml  # noqa: F401
import mac_wifi
import tools


def test_mac_scan_line():
    lines = ["                     Acanac28726                   -58  6,+1    Y  -- RSN(PSK/AES/AES) ",
             "                    Clark family                   -54  1       Y  -- RSN(PSK/AES/AES) ",
             "      Mindrove_ARB_1234 a4:cf:12:00:11:22 -40  6  Y  -- RSN(PSK/AES/AES) "]
    parsed = [mac_wifi.SCAN_LINE.match(line).groups() for line in lines]
    assert parsed == [("Acanac28726", "-58"), ("Clark family", "-54"), ("Mindrove_ARB_1234", "-40")]


def test_mac_saved_armbands(monkeypatch):
    out = "Preferred networks on en0:\n\tClark family\n\tMindrove_ARB_66bb04\n\tMindrove_ARB_1234"
    monkeypatch.setattr(mac_wifi, "_run", lambda *args: out)
    assert mac_wifi.saved_armbands("en0") == ["Mindrove_ARB_66bb04", "Mindrove_ARB_1234"]


def test_windows_profile_line():
    out = ("Profiles on interface WiFi:\n\nGroup policy profiles (read only)\n---------------------------------\n"
           "    <None>\n\nUser profiles\n-------------\n    All User Profile     : Clark family\n"
           "    All User Profile     : Mindrove_ARB_66bb04\n    Current User Profile : Mindrove_ARB_1234 \n")
    assert tools.WIN_PROFILE_LINE.findall(out) == ["Mindrove_ARB_66bb04", "Mindrove_ARB_1234"]
