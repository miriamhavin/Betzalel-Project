// Wiring — each button: one leg to its pin, other leg to GND (internal pull-up, no resistor needed).
//
// BUTTON_MAIN -> D2   BUTTON_YES -> D4   BUTTON_NO -> D7
//
// Serial out: "ENTER" / "YES" / "NO" on each button press.

const int BUTTON_MAIN = 2;
const int BUTTON_YES  = 4;
const int BUTTON_NO   = 7;

const unsigned long DEBOUNCE_MS = 50;

struct Button {
  int pin;
  const char* label;
  int lastReading;
  int stableState;
  unsigned long lastChangeMs;
};

Button buttons[] = {
  { BUTTON_MAIN, "ENTER", HIGH, HIGH, 0 },
  { BUTTON_YES,  "YES",   HIGH, HIGH, 0 },
  { BUTTON_NO,   "NO",    HIGH, HIGH, 0 },
};
const int NUM_BUTTONS = 3;

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < NUM_BUTTONS; i++) {
    pinMode(buttons[i].pin, INPUT_PULLUP);
  }
}

void loop() {
  unsigned long now = millis();

  for (int i = 0; i < NUM_BUTTONS; i++) {
    Button &b = buttons[i];
    int reading = digitalRead(b.pin);

    if (reading != b.lastReading) {
      b.lastChangeMs = now;
    }

    if ((now - b.lastChangeMs) > DEBOUNCE_MS && reading != b.stableState) {
      b.stableState = reading;
      if (b.stableState == LOW) {   // pressed (pulled to GND)
        Serial.println(b.label);
      }
    }

    b.lastReading = reading;
  }
}
