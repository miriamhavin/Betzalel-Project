# AI Pipeline

A 3-round guessing game for an exhibit kiosk. A camera points at objects arranged on a table; a physical white button submits the photo, and Gemini guesses what scene is hidden in the arrangement, drawing its guess on top. Physical green/red buttons mark the guess right or wrong. Runs as a Flask web page, driven by an Arduino wired to the three buttons.

---

## Setup (one time)

**1. Clone the repo**
```
git clone https://github.com/miriamhavin/Betzalel-Project.git
cd Betzalel-Project
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Create a `.env` file** in the project folder:
```
GEMINI_API_KEY=your_key_here
ARDUINO_PORT=COM3
ARDUINO_BAUD=9600
```
Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com). `ARDUINO_PORT` is the serial port your Arduino shows up on (Device Manager on Windows, or `ls /dev/tty.*` on Mac/Linux) — the app still runs fine without one plugged in, it just won't receive button presses.

**4. Font** — the `font/` folder (Masada family) must stay next to `server.py`; the page loads it from there via a `/fonts/...` route.

**5. Arduino sketch** — flash `arduino_button/arduino_button.ino` to the Arduino with the Arduino IDE. Wiring: main button → D2+GND, yes button → D4+GND, no button → D7+GND.

---

## Run

```
python server.py
```
Open **http://localhost:5000** in your browser.

To let others on the same network join, share your local IP:
```
ipconfig        # find your IPv4 address, e.g. 192.168.1.42
```
They open `http://192.168.1.42:5000`.

`app.py` is an older standalone desktop (Tkinter) prototype, kept for reference — `server.py` is the current version.

---

## How it works

1. A gallery of past rounds cycles as an attract screen; the white button starts a new game.
2. A tutorial screen explains the controls, then 3 rounds run in sequence.
3. Each round: arrange objects, press white to submit the photo — it's sent to Gemini.
4. Gemini interprets the objects and invents a scene unique to their arrangement, then a second Gemini call draws the scene on top of the photo.
5. Press green (correct) or red (wrong) to score the round — the files are written to `saves/` and pushed to this repo automatically.
6. After 3 rounds, a summary screen shows the score and how it compares to everyone else who's played.

---

## Files

| File | What it does |
|------|-------------|
| `server.py` | The app — Flask server, game logic, Arduino serial integration, and the embedded frontend |
| `arduino_button/arduino_button.ino` | Arduino sketch for the 3 physical buttons + LEDs |
| `font/` | Self-hosted Masada font files, served at `/fonts/...` |
| `wsgi.py` | WSGI entry point for production deployment |
| `app.py` | Older standalone desktop (Tkinter) prototype — not the current version |
| `saves/` | All saved rounds (original + prediction + scene description) |
| `leaderboard.json` | Running tally of past game scores, used for the summary screen's percentile stat |
| `.env` | Your API key and Arduino port — never commit this |
