# Serial protocol v0.1

USB serial, 115200 baud, one message per line (`\n`), comma separated.
Wait for `READY` after opening the port (the board resets).

## Host → ESP32

| Message | Does | Reply |
|---|---|---|
| `PING` | health check | `PONG,<proto>,<fw>` |
| `M,<ch>,<duty>` | set channel 0-6 to 0-100% | `ACK,...` |
| `A,<duty>` | set all channels | `ACK,...` |
| `X` | all off | `ACK,X` |
| `P,<ch>,<duty>,<ms>` | pulse then off (1-5000 ms) | `ACK,...` |
| `W,<ms>` | watchdog timeout, 0 = off | `ACK,...` |
| `T,<seq>` | latency echo | `T,<seq>,<micros>` |

## ESP32 → host

`READY,<proto>,<fw>` on boot, `ERR,<code>,<detail>` on bad input, `WDT` if the watchdog fired.

## Watchdog

If outputs are on and nothing arrives for 2 s, everything turns off. Send `W,0` before typing commands by hand.