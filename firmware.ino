#include <ArduinoJson.h>
#include <Servo.h>

int servopin1 = 5;
int servopin2 = 6;

String str = "";
Servo servo1;
Servo servo2;

StaticJsonDocument<200> doc;

void setup() {
  Serial.begin(9600);
  pinMode(7, OUTPUT);
  servo1.attach(servopin1);
  servo2.attach(servopin2);
  pinMode(9, OUTPUT);
  
  while (!Serial) continue;
}

void loop() {
  if(Serial.available())
  {
    str = Serial.readStringUntil('\n');
    DeserializationError error = deserializeJson(doc, str);

    if (error) {
      Serial.print(F("deserializeJson() failed: "));
      Serial.println(error.f_str());
      return;
    }
  }

  int conveyor_step = doc["conveyor_step"];
  int sorting_step = doc["sorting_step"];

  servo1.write(conveyor_step);
  servo2.write(sorting_step);
  digitalWrite(9, HIGH);
  delay(10);
}
