#include <Wire.h>

#define ADXL345_ADDR 0x53  // ✅ ADXL345 default I2C address

void setup() {
    Wire.begin();
    Serial.begin(115200);
    
    Serial.println("🔍 Checking for ADXL345...");
    Wire.beginTransmission(ADXL345_ADDR);
    if (Wire.endTransmission() == 0) {
        Serial.println("✅ ADXL345 Found!");
    } else {
        Serial.println("❌ No ADXL345 detected!");
        while (1);
    }

    // Activate ADXL345
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x2D);  // Power register
    Wire.write(8);     // Enable measurement mode
    Wire.endTransmission();
}

void loop() {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x32);  // Start reading acceleration data
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL345_ADDR, 6, true);

    if (Wire.available() == 6) {
        int16_t AcX = Wire.read() | (Wire.read() << 8);
        int16_t AcY = Wire.read() | (Wire.read() << 8);
        int16_t AcZ = Wire.read() | (Wire.read() << 8);

        Serial.print("📡 ADXL345 X: "); Serial.print(AcX);
        Serial.print(" | Y: "); Serial.print(AcY);
        Serial.print(" | Z: "); Serial.println(AcZ);
    } else {
        Serial.println("❌ Error reading ADXL345 data.");
    }

    delay(100);
}
