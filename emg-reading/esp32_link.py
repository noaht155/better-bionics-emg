"""Talk to the ESP32 haptics firmware over USB serial (see protocol.md).

Run this file directly for a self-test that sweeps the outputs:
    python esp32_link.py [--port COM3] [--verbose]
"""
import argparse
import time

import serial
import serial.tools.list_ports

PROTO_VERSION = "0.1"
BAUD = 115200
NUM_CH = 7

# USB to serial chips used on ESP32 dev boards: CP210x and CH340
KNOWN_USB_IDS = {(0x10C4, 0xEA60), (0x1A86, 0x7523)}


class Esp32Error(Exception):
    pass


def find_port():
    matches = [p.device for p in serial.tools.list_ports.comports() if (p.vid, p.pid) in KNOWN_USB_IDS]
    if not matches:
        raise Esp32Error("no ESP32 found, pass the port explicitly")
    if len(matches) > 1:
        raise Esp32Error(f"several possible boards {matches}, pass the port explicitly")
    return matches[0]


class Esp32:
    def __init__(self, port=None, verbose=False):
        self.verbose = verbose
        self.watchdog_fired = False
        port = port or find_port()
        try:
            self.ser = serial.Serial(port, BAUD, timeout=1.0)
        except serial.SerialException as e:
            raise Esp32Error(f"can't open {port}, is the PlatformIO monitor or another script using it? ({e})")
        self.fw_version = self._reset_and_wait_ready()

    def _reset_and_wait_ready(self):
        # DTR high would hold GPIO0 low and boot into the flasher, so keep it low and pulse EN through RTS
        self.ser.dtr = False
        self.ser.rts = True
        time.sleep(0.1)
        self.ser.rts = False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            line = self._readline()
            if line.startswith("READY,"):
                _, proto, fw = line.split(",")
                if proto != PROTO_VERSION:
                    raise Esp32Error(f"firmware speaks protocol {proto}, host expects {PROTO_VERSION}")
                return fw
        raise Esp32Error("no READY from board")

    def _readline(self):
        # The ROM bootloader prints at a different baud rate, so boot output can be garbage
        return self.ser.readline().decode(errors="replace").strip()

    def _send(self, msg, reply_prefix):
        self.ser.write((msg + "\n").encode())
        if self.verbose:
            print(">", msg)
        while True:
            line = self._readline()
            if self.verbose and line:
                print("<", line)
            if not line:
                raise Esp32Error(f"no reply to {msg}")
            if line == "WDT":
                self.watchdog_fired = True
                continue
            if line.startswith("ERR,"):
                raise Esp32Error(f"{msg} -> {line}")
            if line.startswith(reply_prefix):
                return line
            raise Esp32Error(f"{msg} -> unexpected reply {line}")

    def ping(self):
        return self._send("PING", "PONG,")

    def set(self, ch, duty):
        self._send(f"M,{ch},{duty}", f"ACK,M,{ch},{duty}")

    def set_all(self, duty):
        self._send(f"A,{duty}", f"ACK,A,{duty}")

    def off(self):
        self._send("X", "ACK,X")

    def pulse(self, ch, duty, ms):
        self._send(f"P,{ch},{duty},{ms}", f"ACK,P,{ch},{duty},{ms}")

    def watchdog(self, ms):
        self._send(f"W,{ms}", f"ACK,W,{ms}")

    def round_trip_ms(self, seq=0):
        start = time.perf_counter()
        self._send(f"T,{seq}", f"T,{seq},")
        return (time.perf_counter() - start) * 1000

    def close(self):
        # Ctrl+C can land mid-command and leave a stale reply behind, so skip anything before ACK,X
        try:
            self.ser.reset_input_buffer()
            self.ser.write(b"X\n")
            deadline = time.time() + 1.0
            while time.time() < deadline and self._readline() != "ACK,X":
                pass
        finally:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def self_test(board):
    print(f"connected to {board.ser.port}, firmware {board.fw_version}")
    print(board.ping())

    print("channels one at a time")
    for ch in range(NUM_CH):
        board.set(ch, 100)
        time.sleep(0.3)
        board.set(ch, 0)

    print("all channels fading up and down")
    for duty in list(range(0, 101, 10)) + list(range(90, -1, -10)):
        board.set_all(duty)
        time.sleep(0.08)

    print("pulse on channel 3 for 500 ms")
    board.pulse(3, 100, 500)
    time.sleep(0.8)

    times = sorted(board.round_trip_ms(i) for i in range(50))
    print(f"round trip: median {times[len(times) // 2]:.1f} ms, worst {times[-1]:.1f} ms")
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="serial port, auto-detected if omitted")
    parser.add_argument("--verbose", action="store_true", help="print every message sent and received")
    args = parser.parse_args()
    with Esp32(args.port, verbose=args.verbose) as board:
        self_test(board)


if __name__ == "__main__":
    main()
