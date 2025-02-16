#include <Wire.h>

#define ADXL345_ADDR 0x53  // ADXL345 default I2C address

// A4 -> SCL
// A5 -> SDA

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

    // Read and print the g-range
    readGRange();
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

}

void readGRange() {
    // Read DATA_FORMAT register (0x31)
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x31);  // DATA_FORMAT register
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL345_ADDR, 1, true);

    if (Wire.available()) {
        uint8_t dataFormat = Wire.read();
        uint8_t rangeBits = dataFormat & 0b00000011;  // Extract bits 0 and 1

        String gRange;
        switch (rangeBits) {
            case 0: gRange = "±2g"; break;
            case 1: gRange = "±4g"; break;
            case 2: gRange = "±8g"; break;
            case 3: gRange = "±16g"; break;
            default: gRange = "Unknown"; break;
        }

        Serial.print("📊 Current g-range: ");
        Serial.println(gRange);
    } else {
        Serial.println("❌ Error reading DATA_FORMAT register.");
    }
}
