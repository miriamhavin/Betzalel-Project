# AI Pipeline

Point a camera at any objects. Press **snap**. The AI finds a hidden scene inside the arrangement and draws it. Save or discard the result — saved snaps go to the shared gallery and are pushed to this repo automatically.

---

## Setup (one time)

**1. Clone the repo**
```
git clone <repo-url>
cd betzalelproject
```

**2. Install dependencies**
```
pip install flask google-genai pillow opencv-python python-dotenv
```

**3. Create a `.env` file** in the project folder:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

---

## Run

### Web version (recommended — works in any browser)
```
python server.py
```
Open **http://localhost:5000** in your browser.

To let others on the same network join, share your local IP:
```
ipconfig        # find your IPv4 address, e.g. 192.168.1.42
```
They open `http://192.168.1.42:5000`.

### Desktop version (requires a connected webcam)
```
python app.py
```

---

## How it works

1. Press **snap** — the camera frame is sent to Gemini
2. Gemini interprets the objects and invents a scene unique to their arrangement
3. A second Gemini model draws the scene on top of the photo
4. Press **save** to keep it — the files are written to `saves/` and pushed to this repo
5. Press **discard** to throw it away — nothing is saved
6. The gallery at the bottom of the web page shows every snap ever saved by anyone

---

## Files

| File | What it does |
|------|-------------|
| `server.py` | Web server — run this for the browser version |
| `app.py` | Desktop app — run this for the local tkinter version |
| `saves/` | All saved snaps (original + prediction + scene description) |
| `.env` | Your API key — never commit this |
