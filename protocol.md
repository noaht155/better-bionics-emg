# Serial protocol v0.2

One message per line (`\n`), comma separated. The same messages work over two transports:

- **USB serial**, 115200 baud. Wait for `READY` after opening the port (the board resets).
- **WiFi**, TCP port 4211. The ESP32 joins the armband's access point with the fixed address `192.168.4.10` (set in `motor-control/include/wifi_config.h`). It sends `READY` when a client connects. One client at a time, a new connection replaces the old one.

Replies go back on the transport the command came from.

USB has priority: while a USB line arrived within the last 3 s, WiFi commands are not run and get `ERR,BUSY,serial`.

A trailing `\r` is ignored, empty lines are ignored, lines over 64 characters are rejected.
Commands are case sensitive. Numbers are plain unsigned integers (no sign, no spaces).

## Host → ESP32

| Message | Does | Reply |
|---|---|---|
| `PING` | health check | `PONG,<proto>,<fw>` |
| `M,<ch>,<duty>` | set channel 0-6 to 0-100% | `ACK,M,<ch>,<duty>` |
| `S,<d0>,<d1>,...,<d6>` | set all 7 channels, one duty each | `ACK,S` |
| `A,<duty>` | set all channels to the same duty | `ACK,A,<duty>` |
| `X` | all off | `ACK,X` |
| `P,<ch>,<duty>,<ms>` | pulse then off (1-5000 ms) | `ACK,P,<ch>,<duty>,<ms>` |
| `W,<ms>` | watchdog timeout 0-60000 ms, 0 = off | `ACK,W,<ms>` |
| `T,<seq>` | latency echo | `T,<seq>,<micros>` |

Duty is linear: `<duty>` percent of the PWM period (5 kHz, 8 bit), 100 = fully on.
`M`, `S`, `A` and `X` cancel a running pulse on the channels they set.
`S` changes nothing if any of its values is invalid.
Use `S` for streaming updates. Over WiFi a round trip takes about 10 to 40 ms, so one message per update matters.

## ESP32 → host

`READY,<proto>,<fw>` on boot (USB) or on connect (WiFi), `ERR,<code>,<detail>` on bad input, `WDT` if the watchdog fired (sent on both transports).

On USB only, `WIFI,<ip>` when the ESP32 joins the armband's network and `WIFI,DOWN` when it loses it. These can arrive at any time.

| Code | Meaning | Detail |
|---|---|---|
| `UNKNOWN` | command not recognised | the command |
| `ARGS` | wrong number of fields | the command |
| `RANGE` | field not a number or out of range | the command |
| `LONG` | line over 64 characters | `line` |
| `BUSY` | WiFi command refused because USB is in control | `serial` |

## Watchdog

Default 2000 ms. Any complete line that is run (valid or not) resets the timer. Refused WiFi lines don't.
If outputs are on and nothing arrives for the timeout, everything turns off and `WDT` is sent once.
Send `W,0` before typing commands by hand.
