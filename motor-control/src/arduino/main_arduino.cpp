#include <Arduino.h>
#include <WiFi.h>

#include "wifi_config.h"

const char *PROTO_VERSION = "0.2";
const char *FW_VERSION = "0.2.0";

const uint8_t PINS[] = {18, 19, 23, 25, 26, 27, 32};
const uint8_t NUM_CH = sizeof(PINS) / sizeof(PINS[0]);

const uint32_t PWM_FREQ = 5000;
const uint8_t PWM_BITS = 8;
const uint32_t PWM_MAX = (1 << PWM_BITS) - 1;

const uint32_t PULSE_MAX_MS = 5000;
const uint32_t WATCHDOG_MAX_MS = 60000;
const uint8_t LINE_LEN_MAX = 64;
const uint8_t MAX_FIELDS = NUM_CH + 1;

// WiFi commands are refused while serial has been used within this time
const uint32_t SERIAL_HOLD_MS = 3000;

struct LineBuffer {
    char text[LINE_LEN_MAX + 1];
    uint8_t len = 0;
    bool tooLong = false;
};

uint8_t duty[NUM_CH];
uint32_t pulseStart[NUM_CH];
uint32_t pulseLen[NUM_CH];  // 0 = no pulse running

uint32_t watchdogMs = 2000;
uint32_t lastRx = 0;

LineBuffer serialLine;
LineBuffer wifiLine;
bool serialUsed = false;
uint32_t lastSerialRx = 0;

WiFiServer server(TCP_PORT);
WiFiClient client;
bool wifiUp = false;

// Where the reply to the command being handled goes
Print *reply = &Serial;

void setDuty(uint8_t ch, uint8_t percent) {
    duty[ch] = percent;
    pulseLen[ch] = 0;
    // 100% maps to PWM_MAX, which the core treats as fully on
    ledcWrite(ch, (percent * PWM_MAX + 50) / 100);
}

void allOff() {
    for (uint8_t ch = 0; ch < NUM_CH; ch++) {
        setDuty(ch, 0);
    }
}

bool anyOn() {
    for (uint8_t ch = 0; ch < NUM_CH; ch++) {
        if (duty[ch] > 0) return true;
    }
    return false;
}

void sendErr(const char *code, const char *detail) {
    reply->printf("ERR,%s,%s\n", code, detail);
}

// Strict unsigned parse: rejects empty fields, signs, trailing junk and values above max
bool parseUint(const char *s, uint32_t max, uint32_t &out) {
    if (*s < '0' || *s > '9') return false;
    char *end;
    unsigned long v = strtoul(s, &end, 10);
    if (*end != '\0' || v > max) return false;
    out = v;
    return true;
}

bool checkArgs(uint8_t count, uint8_t expected, const char *cmd) {
    if (count == expected) return true;
    sendErr("ARGS", cmd);
    return false;
}

void handleLine(char *msg) {
    char *fields[MAX_FIELDS + 1];
    uint8_t count = 0;
    for (char *tok = strtok(msg, ","); tok && count <= MAX_FIELDS; tok = strtok(nullptr, ",")) {
        fields[count++] = tok;
    }
    const char *cmd = fields[0];
    uint32_t ch, pct, ms, seq;

    if (strcmp(cmd, "PING") == 0) {
        if (!checkArgs(count, 1, cmd)) return;
        reply->printf("PONG,%s,%s\n", PROTO_VERSION, FW_VERSION);
    } else if (strcmp(cmd, "M") == 0) {
        if (!checkArgs(count, 3, cmd)) return;
        if (!parseUint(fields[1], NUM_CH - 1, ch) || !parseUint(fields[2], 100, pct)) {
            sendErr("RANGE", cmd);
            return;
        }
        setDuty(ch, pct);
        reply->printf("ACK,M,%u,%u\n", ch, pct);
    } else if (strcmp(cmd, "A") == 0) {
        if (!checkArgs(count, 2, cmd)) return;
        if (!parseUint(fields[1], 100, pct)) {
            sendErr("RANGE", cmd);
            return;
        }
        for (uint8_t i = 0; i < NUM_CH; i++) {
            setDuty(i, pct);
        }
        reply->printf("ACK,A,%u\n", pct);
    } else if (strcmp(cmd, "S") == 0) {
        if (!checkArgs(count, NUM_CH + 1, cmd)) return;
        uint32_t pcts[NUM_CH];
        for (uint8_t i = 0; i < NUM_CH; i++) {
            if (!parseUint(fields[i + 1], 100, pcts[i])) {
                sendErr("RANGE", cmd);
                return;
            }
        }
        for (uint8_t i = 0; i < NUM_CH; i++) {
            setDuty(i, pcts[i]);
        }
        reply->println("ACK,S");
    } else if (strcmp(cmd, "X") == 0) {
        if (!checkArgs(count, 1, cmd)) return;
        allOff();
        reply->println("ACK,X");
    } else if (strcmp(cmd, "P") == 0) {
        if (!checkArgs(count, 4, cmd)) return;
        if (!parseUint(fields[1], NUM_CH - 1, ch) || !parseUint(fields[2], 100, pct) ||
            !parseUint(fields[3], PULSE_MAX_MS, ms) || ms == 0) {
            sendErr("RANGE", cmd);
            return;
        }
        setDuty(ch, pct);
        pulseStart[ch] = millis();
        pulseLen[ch] = ms;
        reply->printf("ACK,P,%u,%u,%u\n", ch, pct, ms);
    } else if (strcmp(cmd, "W") == 0) {
        if (!checkArgs(count, 2, cmd)) return;
        if (!parseUint(fields[1], WATCHDOG_MAX_MS, ms)) {
            sendErr("RANGE", cmd);
            return;
        }
        watchdogMs = ms;
        reply->printf("ACK,W,%u\n", ms);
    } else if (strcmp(cmd, "T") == 0) {
        if (!checkArgs(count, 2, cmd)) return;
        if (!parseUint(fields[1], UINT32_MAX, seq)) {
            sendErr("RANGE", cmd);
            return;
        }
        reply->printf("T,%u,%lu\n", seq, micros());
    } else {
        sendErr("UNKNOWN", cmd);
    }
}

// Collects characters into buf. Returns true when a full line is ready in buf.text
bool readLine(Stream &in, LineBuffer &buf) {
    while (in.available()) {
        char c = in.read();
        if (c == '\r') continue;  // terminals on Windows send \r\n
        if (c != '\n') {
            if (buf.len < LINE_LEN_MAX) {
                buf.text[buf.len++] = c;
            } else {
                buf.tooLong = true;
            }
            continue;
        }
        buf.text[buf.len] = '\0';
        return true;
    }
    return false;
}

void dispatch(LineBuffer &buf) {
    lastRx = millis();
    if (buf.tooLong) {
        sendErr("LONG", "line");
    } else if (buf.len > 0) {
        handleLine(buf.text);
    }
    buf.len = 0;
    buf.tooLong = false;
}

bool serialActive() {
    return serialUsed && millis() - lastSerialRx < SERIAL_HOLD_MS;
}

void readSerial() {
    while (readLine(Serial, serialLine)) {
        serialUsed = true;
        lastSerialRx = millis();
        reply = &Serial;
        dispatch(serialLine);
    }
}

void updateWifi() {
    bool up = WiFi.status() == WL_CONNECTED;
    if (up != wifiUp) {
        wifiUp = up;
        if (up) {
            server.begin();
            Serial.printf("WIFI,%s\n", WiFi.localIP().toString().c_str());
        } else {
            Serial.println("WIFI,DOWN");
        }
    }
    if (!up) return;

    if (server.hasClient()) {
        // A new connection replaces the old one, which may be a host that went away without closing
        client.stop();
        client = server.available();
        client.setNoDelay(true);
        wifiLine.len = 0;
        wifiLine.tooLong = false;
        client.printf("READY,%s,%s\n", PROTO_VERSION, FW_VERSION);
    }
    if (!client.connected()) return;

    while (readLine(client, wifiLine)) {
        reply = &client;
        if (serialActive()) {
            sendErr("BUSY", "serial");
            wifiLine.len = 0;
            wifiLine.tooLong = false;
        } else {
            dispatch(wifiLine);
        }
    }
}

void updatePulses() {
    uint32_t now = millis();
    for (uint8_t ch = 0; ch < NUM_CH; ch++) {
        if (pulseLen[ch] && now - pulseStart[ch] >= pulseLen[ch]) {
            setDuty(ch, 0);
        }
    }
}

void updateWatchdog() {
    if (watchdogMs == 0 || !anyOn()) return;
    if (millis() - lastRx >= watchdogMs) {
        allOff();
        Serial.println("WDT");
        if (client.connected()) client.println("WDT");
    }
}

void setup() {
    Serial.begin(115200);
    for (uint8_t ch = 0; ch < NUM_CH; ch++) {
        ledcSetup(ch, PWM_FREQ, PWM_BITS);
        ledcAttachPin(PINS[ch], ch);
    }
    allOff();
    lastRx = millis();

    // begin() returns at once and the core keeps retrying, so serial works with the armband off
    WiFi.mode(WIFI_STA);
    WiFi.config(IPAddress(WIFI_IP), IPAddress(WIFI_GATEWAY), IPAddress(WIFI_SUBNET));
    WiFi.setAutoReconnect(true);
    // Modem sleep parks the radio between beacons and adds up to 100 ms to every command
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.printf("READY,%s,%s\n", PROTO_VERSION, FW_VERSION);
}

void loop() {
    readSerial();
    updateWifi();
    updatePulses();
    updateWatchdog();
}
