# Hardware status

## Armband
- Ref and bias OK
- Broken pads: none. Ch 0, 2, 5 had no connection, reflowed 2026-09-29, verified 2026-09-30 (all 8 channels rise 9 to 14x on a clench)
- 2026-10-13: two pad wires snapped at the solder joint, most likely ch 0 and 2 (not confirmed by a press test). Spliced, joint covered with low-temperature hot glue. Rest 3.2 to 4.5 uV, hum under 14 uV on all channels afterwards
- Battery: Akyga AKY0081 980 mAh, soldered. Recovered from deep discharge 2026-09-28
- Case opened 2026-09-28
- Since then the electrode pads are pushed under the band by hand, without the casing. Their positions vary from session to session beyond what the placement note (band position) records
- 2026-10-02: hum comes and goes on ch 5, 6 and 7 (over the 80 uV warning), 6 and 7 on the main module. Not diagnosed. Several channels at once points to the reference or bias pad contact, grounding (charger plugged in) or the band loose on that side, before a single joint. Ch 5 is a reflowed pad, check it last. Fixed before the 2026-10-05 session, cause not recorded
- 2026-10-29: same pattern, hum moving between ch 0, 3 and 5 (up to 420 uV on ch 5) with the band tight and the pads pressed hard. Pressing the main module (ref and bias) in by hand cleared it, so the ref and bias contact is the cause. The 2026-10-02 hum was most likely the same
- WiFi access point `Mindrove_ARB_66bb04` at 192.168.4.1. Accepts a second device (the ESP32) and keeps streaming

## ESP32
- HW-395B, CP210x USB to serial (USB id 10C4:EA60)
- Joins the armband's access point at 192.168.4.10, TCP port 4211

## Driver
- Now: HG7881 modules (L9110S H-bridges) for the coin motors, inverted logic (`pinmap.md`)
- End goal is still the ULN2003APG

## Computers
- Windows desktop (Realtek 8852CE WiFi card, no antennas attached): never gets a DHCP address from the armband, needs the static IP 192.168.4.2. Streaming stopped on 2026-09-29 and came back after a reboot and two adapter settings, see `emg-reading/SETUP.md`
- Desktop, 2026-09-30: after an armband power cycle the static IP alone gives 0 samples. Toggling the adapter to DHCP and back to static fixed it: the armband only streams to a device that asked it for an address since it booted. Confirmed 2026-10-14 with power-cycled trials: DHCP 0 s and 1 s fail, 3 s and 5 s stream (5 s is the default of `connect_armband.ps1`)
- MacBook: gets a DHCP address from the armband and streams without any changes
