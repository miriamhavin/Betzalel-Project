// Wiring — each button: one leg to its pin, other leg to GND (internal pull-up, no resistor needed).
// LEDs: through a current-limiting resistor to GND. The main button's LED is wired
// straight to power and isn't controlled from here.
//
// BUTTON_MAIN -> D2   BUTTON_YES -> D4   BUTTON_NO -> D7
// LED_YES     -> D5   LED_NO     -> D6
//
// Serial out: "ENTER" / "YES" / "NO" on each button press.
// Serial in:  "RESULT_ON" turns the yes/no LEDs on, "RESULT_OFF" turns them off.

const int BUTTON_MAIN = 2;
const int BUTTON_YES  = 4;
const int BUTTON_NO   = 7;
const int LED_YES     = 5;
const int LED_NO      = 6;

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
  pinMode(LED_YES, OUTPUT);
  pinMode(LED_NO, OUTPUT);
  digitalWrite(LED_YES, LOW);
  digitalWrite(LED_NO, LOW);
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

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line == "RESULT_ON") {
      digitalWrite(LED_YES, HIGH);
      digitalWrite(LED_NO, HIGH);
    } else if (line == "RESULT_OFF") {
      digitalWrite(LED_YES, LOW);
      digitalWrite(LED_NO, LOW);
    }
  }
}
