"""Talk to the ESP32 haptics firmware over USB serial or WiFi (see protocol.md).

USB is used when a board is plugged in, WiFi otherwise. --wifi forces WiFi.
Run this file directly for a self-test that sweeps the outputs:
    python esp32_link.py [--port COM3] [--wifi [IP]] [--verbose]
"""
import argparse
import socket
import time

import serial
import serial.tools.list_ports

PROTO_VERSION = "0.2"
BAUD = 115200
NUM_CH = 7

# Must match motor-control/include/wifi_config.h
WIFI_IP = "192.168.4.10"
TCP_PORT = 4211

# USB to serial chips used on ESP32 dev boards: CP210x and CH340
KNOWN_USB_IDS = {(0x10C4, 0xEA60), (0x1A86, 0x7523)}

READ_TIMEOUT_S = 1.0


class Esp32Error(Exception):
    pass


def find_port():
    """The serial port of the one plugged-in board, or None if there is none."""
    matches = [p.device for p in serial.tools.list_ports.comports() if (p.vid, p.pid) in KNOWN_USB_IDS]
    if len(matches) > 1:
        raise Esp32Error(f"several possible boards {matches}, pass the port explicitly")
    return matches[0] if matches else None


class SerialTransport:
    def __init__(self, port):
        self.name = port
        try:
            self.ser = serial.Serial(port, BAUD, timeout=READ_TIMEOUT_S)
        except serial.SerialException as e:
            raise Esp32Error(f"can't open {port}, is the PlatformIO monitor or another script using it? ({e})")
        # Reset the board so it sends READY. DTR high would hold GPIO0 low and boot into the
        # flasher, so keep it low and pulse EN through RTS
        self.ser.dtr = False
        self.ser.rts = True
        time.sleep(0.1)
        self.ser.rts = False

    def write(self, data):
        self.ser.write(data)

    def readline(self):
        # The ROM bootloader prints at a different baud rate, so boot output can be garbage
        return self.ser.readline().decode(errors="replace").strip()

    def discard_input(self):
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()


class TcpTransport:
    def __init__(self, host):
        self.name = f"{host} (wifi)"
        try:
            self.sock = socket.create_connection((host, TCP_PORT), timeout=5.0)
        except OSError as e:
            raise Esp32Error(f"can't reach the ESP32 at {host}:{TCP_PORT}, is this computer on the armband's WiFi? ({e})")
        # Commands are tiny, so send each one at once and don't wait to batch them
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(READ_TIMEOUT_S)
        self.buf = b""

    def write(self, data):
        self.sock.sendall(data)

    def readline(self):
        while b"\n" not in self.buf:
            try:
                chunk = self.sock.recv(256)
            except socket.timeout:
                return ""
            if not chunk:
                raise Esp32Error("the ESP32 closed the WiFi connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def discard_input(self):
        self.buf = b""
        self.sock.settimeout(0.05)
        try:
            while self.sock.recv(256):
                pass
        except OSError:
            pass
        self.sock.settimeout(READ_TIMEOUT_S)

    def close(self):
        self.sock.close()


class Esp32:
    def __init__(self, port=None, wifi=None, verbose=False):
        """port: serial port. wifi: IP address. With neither, USB if a board is plugged in, else WiFi."""
        self.verbose = verbose
        self.watchdog_fired = False
        if port is None and wifi is None:
            port = find_port()
        self.link = SerialTransport(port) if port else TcpTransport(wifi or WIFI_IP)
        self.fw_version = self._wait_ready()

    def _wait_ready(self):
        deadline = time.time() + 5.0
        while time.time() < deadline:
            line = self.link.readline()
            if line.startswith("READY,"):
                _, proto, fw = line.split(",")
                if proto != PROTO_VERSION:
                    raise Esp32Error(f"firmware speaks protocol {proto}, host expects {PROTO_VERSION}")
                return fw
        raise Esp32Error("no READY from board")

    def _send(self, msg, reply_prefix):
        self.link.write((msg + "\n").encode())
        if self.verbose:
            print(">", msg)
        while True:
            line = self.link.readline()
            if self.verbose and line:
                print("<", line)
            if not line:
                raise Esp32Error(f"no reply to {msg}")
            if line == "WDT":
                self.watchdog_fired = True
                continue
            if line.startswith("WIFI,"):
                continue
            if line == "ERR,BUSY,serial":
                raise Esp32Error("the ESP32 is being controlled over USB, WiFi commands are refused")
            if line.startswith("ERR,"):
                raise Esp32Error(f"{msg} -> {line}")
            if line.startswith(reply_prefix):
                return line
            raise Esp32Error(f"{msg} -> unexpected reply {line}")

    def ping(self):
        return self._send("PING", "PONG,")

    def set(self, ch, duty):
        self._send(f"M,{ch},{duty}", f"ACK,M,{ch},{duty}")

    def set_outputs(self, duties):
        """Set all outputs in one message. duties: one percentage per channel."""
        self._send("S," + ",".join(str(d) for d in duties), "ACK,S")

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
            self.link.discard_input()
            self.link.write(b"X\n")
            deadline = time.time() + 1.0
            while time.time() < deadline and self.link.readline() != "ACK,X":
                pass
        finally:
            self.link.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def add_link_args(parser):
    parser.add_argument("--port", help="ESP32 serial port, auto-detected if omitted")
    parser.add_argument("--wifi", nargs="?", const=WIFI_IP, metavar="IP",
                        help=f"use WiFi even if a board is plugged in (default address {WIFI_IP})")


def self_test(board):
    print(f"connected to {board.link.name}, firmware {board.fw_version}")
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

    print("staircase across the outputs")
    board.set_outputs([round(100 * ch / (NUM_CH - 1)) for ch in range(NUM_CH)])
    time.sleep(1.0)
    board.off()

    print("pulse on channel 3 for 500 ms")
    board.pulse(3, 100, 500)
    time.sleep(0.8)

    times = sorted(board.round_trip_ms(i) for i in range(50))
    print(f"round trip: median {times[len(times) // 2]:.1f} ms, worst {times[-1]:.1f} ms")
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser()
    add_link_args(parser)
    parser.add_argument("--verbose", action="store_true", help="print every message sent and received")
    args = parser.parse_args()
    with Esp32(args.port, args.wifi, verbose=args.verbose) as board:
        self_test(board)


if __name__ == "__main__":
    main()
