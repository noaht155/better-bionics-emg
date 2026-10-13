# Pin map

ESP32 HW-395B (ESP32-WROOM-32). Outputs are 3.3 V, about 20 mA max per pin.

## Outputs

| Channel | GPIO | LEDC channel | Driven by EMG ch (haptics.py) | Now (L9110S modules) | Later (ULN2003APG) |
|---|---|---|---|---|---|
| 0 | 18 | 0 | 0 | module 1 A1A → motor 0 | IN1 → motor 0 |
| 1 | 19 | 1 | 1 | module 1 B1A → motor 1 | IN2 → motor 1 |
| 2 | 23 | 2 | 2 | module 2 A1A → motor 2 | IN3 → motor 2 |
| 3 | 25 | 3 | 3 | module 2 B1A → motor 3 | IN4 → motor 3 |
| 4 | 26 | 4 | 4 | module 3 A1A → motor 4 | IN5 → motor 4 |
| 5 | 27 | 5 | 5 | module 3 B1A → motor 5 | IN6 → motor 5 |
| 6 | 32 | 6 | 6 and 7 averaged (base module) | module 4 A1A → motor 6 | IN7 → motor 6 |

PWM is 5 kHz, 8 bit. Coin motors will need about 20 kHz so they don't whine.

## LED wiring (before the motors, needs OUTPUTS_INVERTED false)

GPIO → resistor → LED anode, LED cathode → GND. Use the same resistor value on every channel, a different value makes that LED visibly dimmer (happened on channel 0). 220 to 330 Ω for red LEDs.

## Motor wiring (now): HG7881 modules, 2x L9110S each

Stopgap, a ULN2003 was ordered for but these arrived. Each module drives 2 motors, so 4 modules for 7.

- VCC: motor supply, about 3.3 V from its own regulator (coin motors are rated about 3 V, the L9110S drops a few tenths). Not the ESP32's 3V3 pin. GND to the supply and to ESP32 GND.
- A1A, B1A: from the GPIOs above. A1B, B1B: left unconnected.
- Motor A and motor B on the two screw terminals. Flyback diodes are in the chip.
- The module pulls every input up to VCC through about 1 kΩ (measured: A1A at 1.6 V against a 1 kΩ pull-down). A pull-down can't beat that without loading the GPIO near its limit, so the logic is inverted instead: xIB sits high, the motor runs while xIA is low, both high is off. An unpowered or booting ESP32 leaves the motors off.
- The firmware inverts the duty for this (`OUTPUTS_INVERTED` in `main_arduino.cpp`). Set it to false and reflash for LEDs or a low-side driver (ULN2003, MOSFET), otherwise those run full on at 0 %.
- If the module has power while the ESP32 has none, the pull-ups feed into the ESP32's pins and the motors may run. Power both from the same supply.

## Motor wiring (planned, ULN2003)

- ULN2003APG IN1 to IN7 from the GPIOs above, ULN2003 GND to ESP32 GND (common ground).
- Each motor between the motor supply (+) and its ULN2003 output (OUT1 to OUT7).
- ULN2003 COM (pin 9) to the motor supply, so the built-in flyback diodes clamp the motor kickback.
- Motors never on a GPIO directly.

## Reserved

- GPIO 21, 22: kept free for I2C.
- GPIO 33: spare.
- Never use for outputs: 0, 2, 12, 15 (boot strapping), 1, 3 (USB serial), 6 to 11 (flash), 14, 34 to 39 (input only).
