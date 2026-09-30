#pragma once

// The armband's own access point. The password is MindRove's factory default
#define WIFI_SSID "Mindrove_ARB_66bb04"
#define WIFI_PASS "#mindrove"

// Fixed address, outside the low addresses the armband's DHCP hands out
#define WIFI_IP      192, 168, 4, 10
#define WIFI_GATEWAY 192, 168, 4, 1
#define WIFI_SUBNET  255, 255, 255, 0

#define TCP_PORT 4211
