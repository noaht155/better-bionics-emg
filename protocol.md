# Serial protocol v0.1

USB serial, 115200 baud, one message per line (`\n`), comma separated.
Wait for `READY` after opening the port (the board resets).
A trailing `\r` is ignored, empty lines are ignored, lines over 64 characters are rejected.
Commands are case sensitive. Numbers are plain unsigned integers (no sign, no spaces).

## Host → ESP32

| Message | Does | Reply |
|---|---|---|
| `PING` | health check | `PONG,<proto>,<fw>` |
| `M,<ch>,<duty>` | set channel 0-6 to 0-100% | `ACK,M,<ch>,<duty>` |
| `A,<duty>` | set all channels | `ACK,A,<duty>` |
| `X` | all off | `ACK,X` |
| `P,<ch>,<duty>,<ms>` | pulse then off (1-5000 ms) | `ACK,P,<ch>,<duty>,<ms>` |
| `W,<ms>` | watchdog timeout 0-60000 ms, 0 = off | `ACK,W,<ms>` |
| `T,<seq>` | latency echo | `T,<seq>,<micros>` |

Duty is linear: `<duty>` percent of the PWM period (5 kHz, 8 bit), 100 = fully on.
`M`, `A` and `X` cancel a running pulse on the channels they set.

## ESP32 → host

`READY,<proto>,<fw>` on boot, `ERR,<code>,<detail>` on bad input, `WDT` if the watchdog fired.

| Code | Meaning | Detail |
|---|---|---|
| `UNKNOWN` | command not recognised | the command |
| `ARGS` | wrong number of fields | the command |
| `RANGE` | field not a number or out of range | the command |
| `LONG` | line over 64 characters | `line` |

## Watchdog

Default 2000 ms. Any complete line (valid or not) resets the timer.
If outputs are on and nothing arrives for the timeout, everything turns off and `WDT` is sent once.
Send `W,0` before typing commands by hand.
