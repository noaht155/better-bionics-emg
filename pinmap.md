# Pin map

ESP32 HW-395B (ESP32-WROOM-32). Outputs are 3.3 V, about 20 mA max per pin.

## Outputs

| Channel | GPIO | LEDC channel | Driven by EMG ch (haptics.py) | Now | Later (ULN2003APG) |
|---|---|---|---|---|---|
| 0 | 18 | 0 | 0 | LED | IN1 → motor 0 |
| 1 | 19 | 1 | 1 | LED | IN2 → motor 1 |
| 2 | 23 | 2 | 2 | LED | IN3 → motor 2 |
| 3 | 25 | 3 | 3 | LED | IN4 → motor 3 |
| 4 | 26 | 4 | 4 | LED | IN5 → motor 4 |
| 5 | 27 | 5 | 5 | LED | IN6 → motor 5 |
| 6 | 32 | 6 | 6 and 7 averaged (base module) | LED | IN7 → motor 6 |

PWM is 5 kHz, 8 bit. Coin motors will need about 20 kHz so they don't whine.

## LED wiring (now)

GPIO → resistor → LED anode, LED cathode → GND. Use the same resistor value on every channel, a different value makes that LED visibly dimmer (happened on channel 0). 220 to 330 Ω for red LEDs.

## Motor wiring (planned)

- ULN2003APG IN1 to IN7 from the GPIOs above, ULN2003 GND to ESP32 GND (common ground).
- Each motor between the motor supply (+) and its ULN2003 output (OUT1 to OUT7).
- ULN2003 COM (pin 9) to the motor supply, so the built-in flyback diodes clamp the motor kickback.
- Motors never on a GPIO directly.

## Reserved

- GPIO 21, 22: kept free for I2C.
- GPIO 33: spare.
- Never use for outputs: 0, 2, 12, 15 (boot strapping), 1, 3 (USB serial), 6 to 11 (flash), 14, 34 to 39 (input only).
