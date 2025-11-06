#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

LiquidCrystal_I2C lcd(0x27, 16, 2); // 0x27, 16x2

String ssid;
String password;
const char* serverUrl = "http://192.168.1.2:8080/node/node_2/insert_queue";

unsigned long lastPostMs = 0;
const unsigned long postIntervalMs = 60000; // 60s

unsigned long lastDhtReadMs = 0;
const unsigned long dhtIntervalMs = 2000;   // read DHT every 2s
int node_num = 1;

// int counter_retrain = 1; // fixed name

// --- Function to get user input from Serial Monitor ---
String getInput(const String& label) {
  Serial.print(label);
  while (Serial.available() == 0) {
    delay(10);
  }
  String input = Serial.readStringUntil('\n');
  input.trim();
  return input;
}

// --- Connect to WiFi (blocking with limited retries) ---
void connectToWiFi() {
  Serial.println("*Attempting to connect to WiFi*...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid.c_str(), password.c_str());

  int retryCount = 0;
  while (WiFi.status() != WL_CONNECTED && retryCount < 40) {
    retryCount++;
    Serial.printf("⏳ Connecting... attempt %d\n", retryCount);
    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("==== WiFi connected! ====");
    Serial.print("📡 IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("🔗 RSSI: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println("==== WiFi connection failed. ====");
    WiFi.disconnect(true);
  }
}

void setup() {
  Serial.begin(115200);
  // Wait for Serial (harmless on ESP32; prevents losing the first prompts)
  while (!Serial) { delay(10); }

  // I2C (explicit for ESP32)
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("DHT11 Sensor");

  dht.begin();

  Serial.println("\n--- ESP32 WiFi Setup ---");
  ssid = getInput("Enter WiFi SSID: ");
  password = getInput("Enter WiFi Password: ");
  Serial.println("\nConnecting to: " + ssid);

  connectToWiFi();
}

void loop() {
  unsigned long now = millis();

  // Reconnect WiFi if needed (non-blocking-ish)
  if (WiFi.status() != WL_CONNECTED && (now % 5000 < 50)) {
    Serial.println("WiFi not connected. Reconnecting...");
    connectToWiFi();
  }

  // --- Periodic DHT read ---
  static float humidity = NAN;
  static float temperature = NAN;

  if (now - lastDhtReadMs >= dhtIntervalMs) {
    lastDhtReadMs = now;
    humidity = dht.readHumidity();
    temperature = dht.readTemperature();
  }

  // --- LCD display ---
  if (!isnan(humidity) && !isnan(temperature)) {
    lcd.setCursor(0, 0);
    lcd.print("Temp:" + String(temperature, 1) + "C   ");
    lcd.setCursor(0, 1);
    lcd.print("Hum: " + String(humidity, 1) + "%   ");
  } else {
    lcd.setCursor(0, 0);
    lcd.print("Sensor error      ");
    lcd.setCursor(0, 1);
    lcd.print("Check DHT11       ");
  }

  // --- Post every 60 seconds ---
  if (now - lastPostMs >= postIntervalMs) {
    lastPostMs = now;

    if (isnan(humidity) || isnan(temperature)) {
      Serial.println("==== Failed to read from DHT sensor! ====");
      return;
    }

    // if (counter_retrain > 10) {   // reset condition
    //   Serial.println("🔁 counter_retrain reached 10 — resetting to 1");
    //   counter_retrain = 1;
    // }

    // Build JSON payload
    String payload = "{";
    payload += "\"temperature\":" + String(temperature, 1) + ",";
    payload += "\"humidity\":" + String(humidity, 1) + ",";
    // payload += "\"node_name\": \"node_1\",";  // no need for this kasi meron namna <node_name sa route>
    payload += "\"node_num\": "+ String(node_num);
    payload += "}";


    Serial.println("Payload: " + payload);

    if (WiFi.status() == WL_CONNECTED) {
      WiFiClient client;
      HTTPClient http;
      if (http.begin(client, serverUrl)) {
        http.addHeader("Content-Type", "application/json");
        int code = http.POST(payload);
        if (code > 0) {
          Serial.println("POST sent. Code: " + String(code));
          String resp = http.getString();
          Serial.println("Response: " + resp);
          // counter_retrain += 1; // increment counter
        } else {
          Serial.println("POST failed. Code: " + String(code));
        }
        http.end();
      } else {
        Serial.println("HTTP begin() failed.");
      }
    } else {
      Serial.println("Skipped POST: WiFi not connected.");
    }
  }
}
