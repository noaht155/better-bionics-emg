#include <Arduino.h>

const uint8_t PINS[] = {18, 19, 23, 25, 26, 27, 32};
const uint8_t NUM_CH = sizeof(PINS) / sizeof(PINS[0]);

const uint32_t PWM_FREQ = 5000;
const uint8_t PWM_BITS = 8;

void setup() {
    Serial.begin(115200);
    for (uint8_t ch = 0; ch < NUM_CH; ch++) {
        ledcSetup(ch, PWM_FREQ, PWM_BITS);
        ledcAttachPin(PINS[ch], ch);
    }
    Serial.println("PWM test started");
}

void loop() {
    for (uint8_t led = 0; led < NUM_CH; led++) {
        ledcWrite(led, 255);
        delay(1000);
    }
    for (uint8_t led = 0; led < NUM_CH; led++) {
        ledcWrite(led, 0);
        delay(1000);
    }
}
