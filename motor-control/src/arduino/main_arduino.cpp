#include <Arduino.h>

const char *PROTO_VERSION = "0.1";
const char *FW_VERSION = "0.1.0";

const uint8_t PINS[] = {18, 19, 23, 25, 26, 27, 32};
const uint8_t NUM_CH = sizeof(PINS) / sizeof(PINS[0]);

const uint32_t PWM_FREQ = 5000;
const uint8_t PWM_BITS = 8;
const uint32_t PWM_MAX = (1 << PWM_BITS) - 1;

const uint32_t PULSE_MAX_MS = 5000;
const uint32_t WATCHDOG_MAX_MS = 60000;
const uint8_t LINE_LEN_MAX = 64;
const uint8_t MAX_FIELDS = 4;

uint8_t duty[NUM_CH];
uint32_t pulseStart[NUM_CH];
uint32_t pulseLen[NUM_CH];  // 0 = no pulse running

uint32_t watchdogMs = 2000;
uint32_t lastRx = 0;

char line[LINE_LEN_MAX + 1];
uint8_t lineLen = 0;
bool lineTooLong = false;

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
    Serial.printf("ERR,%s,%s\n", code, detail);
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
        Serial.printf("PONG,%s,%s\n", PROTO_VERSION, FW_VERSION);
    } else if (strcmp(cmd, "M") == 0) {
        if (!checkArgs(count, 3, cmd)) return;
        if (!parseUint(fields[1], NUM_CH - 1, ch) || !parseUint(fields[2], 100, pct)) {
            sendErr("RANGE", cmd);
            return;
        }
        setDuty(ch, pct);
        Serial.printf("ACK,M,%u,%u\n", ch, pct);
    } else if (strcmp(cmd, "A") == 0) {
        if (!checkArgs(count, 2, cmd)) return;
        if (!parseUint(fields[1], 100, pct)) {
            sendErr("RANGE", cmd);
            return;
        }
        for (uint8_t i = 0; i < NUM_CH; i++) {
            setDuty(i, pct);
        }
        Serial.printf("ACK,A,%u\n", pct);
    } else if (strcmp(cmd, "X") == 0) {
        if (!checkArgs(count, 1, cmd)) return;
        allOff();
        Serial.println("ACK,X");
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
        Serial.printf("ACK,P,%u,%u,%u\n", ch, pct, ms);
    } else if (strcmp(cmd, "W") == 0) {
        if (!checkArgs(count, 2, cmd)) return;
        if (!parseUint(fields[1], WATCHDOG_MAX_MS, ms)) {
            sendErr("RANGE", cmd);
            return;
        }
        watchdogMs = ms;
        Serial.printf("ACK,W,%u\n", ms);
    } else if (strcmp(cmd, "T") == 0) {
        if (!checkArgs(count, 2, cmd)) return;
        if (!parseUint(fields[1], UINT32_MAX, seq)) {
            sendErr("RANGE", cmd);
            return;
        }
        Serial.printf("T,%u,%lu\n", seq, micros());
    } else {
        sendErr("UNKNOWN", cmd);
    }
}

void readSerial() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r') continue;  // terminals on Windows send \r\n
        if (c != '\n') {
            if (lineLen < LINE_LEN_MAX) {
                line[lineLen++] = c;
            } else {
                lineTooLong = true;
            }
            continue;
        }
        lastRx = millis();
        line[lineLen] = '\0';
        if (lineTooLong) {
            sendErr("LONG", "line");
        } else if (lineLen > 0) {
            handleLine(line);
        }
        lineLen = 0;
        lineTooLong = false;
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
    Serial.printf("READY,%s,%s\n", PROTO_VERSION, FW_VERSION);
}

void loop() {
    readSerial();
    updatePulses();
    updateWatchdog();
}
