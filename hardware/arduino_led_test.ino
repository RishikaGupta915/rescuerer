const int RED_PIN = 3;
const int YELLOW_PIN = 4;
const int GREEN_PIN = 5;
const unsigned long BLINK_INTERVAL_MS = 300;

int activePin = -1;
bool blinkState = false;
unsigned long lastBlinkAt = 0;

void clearAll() {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
}

void setActivePin(int pin) {
  clearAll();
  activePin = pin;
  blinkState = false;
  lastBlinkAt = millis();
}

void handleCommand(char c) {
  if (c == '\n' || c == '\r') {
    return;
  }

  if (c == 'H') {
    setActivePin(RED_PIN);
  } else if (c == 'M') {
    setActivePin(YELLOW_PIN);
  } else if (c == 'L') {
    setActivePin(GREEN_PIN);
  } else if (c == 'N') {
    activePin = -1;
    clearAll();
  }
}

void updateBlink() {
  if (activePin < 0) {
    return;
  }

  unsigned long now = millis();
  if (now - lastBlinkAt < BLINK_INTERVAL_MS) {
    return;
  }

  blinkState = !blinkState;
  digitalWrite(activePin, blinkState ? HIGH : LOW);
  lastBlinkAt = now;
}

void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);

  clearAll();
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    handleCommand(Serial.read());
  }

  updateBlink();
}
