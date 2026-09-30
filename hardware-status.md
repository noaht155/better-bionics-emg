# Hardware status

## Armband
- Ref and bias OK
- Broken pads: none. Ch 0, 2, 5 had no connection, reflowed 2026-09-29, verified 2026-09-30 (all 8 channels rise 9 to 14x on a clench)
- Battery: Akyga AKY0081 980 mAh, soldered. Recovered from deep discharge 2026-09-28
- Case opened 2026-09-28
- WiFi access point `Mindrove_ARB_66bb04` at 192.168.4.1. Accepts a second device (the ESP32) and keeps streaming

## ESP32
- HW-395B, CP210x USB to serial (USB id 10C4:EA60)
- Joins the armband's access point at 192.168.4.10, TCP port 4211

## Driver
- ULN2003APG on order, LEDs for now

## Computers
- Windows desktop (Realtek 8852CE WiFi card, no antennas attached): never gets a DHCP address from the armband, needs the static IP 192.168.4.2. Streaming stopped on 2026-09-29 and came back after a reboot and two adapter settings, see `emg-reading/SETUP.md`
- MacBook: gets a DHCP address from the armband and streams without any changes
