#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// -------------------------------
// Sensor Configuration
// -------------------------------
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// -------------------------------
// LCD Configuration
// -------------------------------
LiquidCrystal_I2C lcd(0x27, 16, 2); // address 0x27, 16x2 LCD

// -------------------------------
// WiFi Configuration
// -------------------------------
String ssid;
String password;

// Update this with your server’s IP and Flask port
const char* serverBaseUrl = "http://192.168.1.2:8080";  
String node_name = "node_1";  // same as Flask route: /node/node_2/insert_queue

// -------------------------------
// Node Information
// -------------------------------
int node_num = 1;  // match with node_name
int site_num = 1;  // optional, defaults to 1 in Flask

// -------------------------------
// Timers
// -------------------------------
unsigned long lastPostMs = 0;
const unsigned long postIntervalMs = 60000; // 60 seconds

unsigned long lastDhtReadMs = 0;
const unsigned long dhtIntervalMs = 2000;   // 2 seconds

// -------------------------------
// Helper Functions
// -------------------------------

String getInput(const String& label) {
  Serial.print(label);
  while (Serial.available() == 0) {
    delay(10);
  }
  String input = Serial.readStringUntil('\n');
  input.trim();
  return input;
}

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

// -------------------------------
// Setup
// -------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  // Initialize LCD and DHT
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("DHT11 Sensor");
  dht.begin();

  // Ask WiFi credentials
  Serial.println("\n--- ESP32 WiFi Setup ---");
  ssid = getInput("Enter WiFi SSID: ");
  password = getInput("Enter WiFi Password: ");
  Serial.println("\nConnecting to: " + ssid);

  connectToWiFi();
}

// -------------------------------
// Loop
// -------------------------------
void loop() {
  unsigned long now = millis();

  // Auto reconnect every ~5s if disconnected
  if (WiFi.status() != WL_CONNECTED && (now % 5000 < 50)) {
    Serial.println("WiFi not connected. Reconnecting...");
    connectToWiFi();
  }

  // --- Periodic DHT Read ---
  static float humidity = NAN;
  static float temperature = NAN;

  if (now - lastDhtReadMs >= dhtIntervalMs) {
    lastDhtReadMs = now;
    humidity = dht.readHumidity();
    temperature = dht.readTemperature();
  }

  // --- LCD Display ---
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

  // --- Post to Flask every 60 seconds ---
  if (now - lastPostMs >= postIntervalMs) {
    lastPostMs = now;

    if (isnan(humidity) || isnan(temperature)) {
      Serial.println("==== Failed to read from DHT sensor! ====");
      return;
    }

    // Construct URL dynamically
    String url = String(serverBaseUrl) + "/node/" + node_name + "/insert_queue";

    // Construct JSON payload
    String payload = "{";
    payload += "\"temperature\":" + String(temperature, 1) + ",";
    payload += "\"humidity\":" + String(humidity, 1) + ",";
    payload += "\"node_num\":" + String(node_num) + ",";
    payload += "\"site_num\":" + String(site_num);
    payload += "}";

    Serial.println("\n--- Sending POST ---");
    Serial.println("URL: " + url);
    Serial.println("Payload: " + payload);

    if (WiFi.status() == WL_CONNECTED) {
      WiFiClient client;
      HTTPClient http;
      if (http.begin(client, url)) {
        http.addHeader("Content-Type", "application/json");
        int code = http.POST(payload);
        if (code > 0) {
          Serial.println("✅ POST sent. Code: " + String(code));
          String resp = http.getString();
          Serial.println("Response: " + resp);
        } else {
          Serial.println("❌ POST failed. Code: " + String(code));
        }
        http.end();
      } else {
        Serial.println("❌ HTTP begin() failed.");
      }
    } else {
      Serial.println("⚠️ Skipped POST: WiFi not connected.");
    }
  }
}
