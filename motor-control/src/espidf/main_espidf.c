#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"

// Most ESP32 dev boards use GPIO 2 for the onboard blue LED
#define LED_PIN 2

void app_main(void) {
    printf("ESP-IDF Blink Test Started!\n");
    
    // Reset and configure the GPIO pin as output
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    while (1) {
        gpio_set_level(LED_PIN, 1); // Turn the LED on
        printf("LED ON\n");
        vTaskDelay(pdMS_TO_TICKS(10));
        
        gpio_set_level(LED_PIN, 0); // Turn the LED off
        printf("LED OFF\n");
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
