import os, sys, json, base64, datetime, subprocess, threading, time, builtins as _builtins

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

_orig_print = _builtins.print
def print(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _orig_print(f"[{ts}]", *args, **kwargs)

from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types
from dotenv import load_dotenv
import serial

load_dotenv()

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVES_DIR = os.path.join(APP_DIR, "saves")
FONT_DIR  = os.path.join(APP_DIR, "font", "פונט מסדה")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
app    = Flask(__name__)

# ── Arduino buttons ──────────────────────────────────────────────────────────
# Three physical buttons replace the keyboard: ENTER (advance), YES (mark
# correct), NO (mark wrong). The Arduino sketch prints one of those words over
# serial on each press; this thread reads them into a queue that /button-poll
# hands to the browser one at a time, the same way it already polls /status
# during a prediction. RESULT_ON/RESULT_OFF are sent back to the Arduino to
# light the yes/no buttons' LEDs while a result is on screen.

ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM3")
ARDUINO_BAUD = int(os.getenv("ARDUINO_BAUD", "9600"))
ARDUINO_EVENTS = {"ENTER", "YES", "NO"}

_button_queue: list[str] = []
_button_lock = threading.Lock()
_ser = None
_ser_lock = threading.Lock()


def _arduino_listener():
    global _ser
    while True:
        try:
            ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
            with _ser_lock:
                _ser = ser
            print(f"[Arduino] connected on {ARDUINO_PORT}")
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if line in ARDUINO_EVENTS:
                    with _button_lock:
                        _button_queue.append(line)
                    print(f"[Arduino] button: {line}")
        except Exception as exc:
            with _ser_lock:
                _ser = None
            print(f"[Arduino] {exc} — retrying in 3s (set ARDUINO_PORT in .env if this persists)")
            time.sleep(3)


def _send_arduino(cmd: str):
    with _ser_lock:
        if _ser is None:
            return
        try:
            _ser.write((cmd + "\n").encode())
        except Exception as exc:
            print(f"[Arduino] write failed: {exc}")


threading.Thread(target=_arduino_listener, daemon=True).start()

def _load_example(ts: str) -> tuple[bytes, bytes] | None:
    orig = os.path.join(APP_DIR, "saves", f"{ts}_original.jpg")
    pred = os.path.join(APP_DIR, "saves", f"{ts}_prediction.jpg")
    if os.path.exists(orig) and os.path.exists(pred):
        return open(orig, "rb").read(), open(pred, "rb").read()
    return None

_FEW_SHOT_EXAMPLES = [
    ex for ts in ("20260614_184328", "20260614_193139")
    if (ex := _load_example(ts)) is not None
]

# ── prompts ───────────────────────────────────────────────────────────────────

INTERPRET_PROMPT = (
    "A person has intentionally arranged these physical objects on a table to represent a specific hidden scene or entity they had in mind — this is not a random arrangement. "
    "You are playing a guessing game with them: you win only if your guess matches what they actually intended to build, based purely on the exact shapes, sizes, and spatial relationships they chose. "
    "This is one round of a multi-round game — if round history is provided below, it shows which of your past guesses were marked correct or wrong; learn from it and adjust your guessing style accordingly.\n\n"
    "1. DISCOVERY: Guess the one concrete plausable object, scene or character the person intended based on these specific shapes, sizes, and their relative proximity to one another. Never guess an abstract process, diagram, or industrial/technical procedure. Your guess must be explainable in a single, concrete term.\n"
    "2. HYPOTHESIS: You must provide a guess that is physically believable. Explain how the objects' current spatial orientation allows for this interpretation. Avoid abstract, metaphorical, or technical/industrial framing; focus on tangible, real-world forms.\n"
    "3. ANCHORING: Every object in the image must be assigned a functional, physical role AND a specific connecting mark that will be drawn from its exact visible edge. If an object is present, it must be essential to the scene's physical logic and must be drawn from.\n"
    "4. NO REPEATS: You are strictly forbidden from guessing any scene or character listed under BANNED SCENES below (if present) — not even a reworded, renamed, or rephrased version of one. Every round in this game must be a genuinely different guess.\n\n"
    "JSON REQUIREMENTS:\n"
    "- 'scene': A minimal description of the discovered character or scene.\n"
    "- 'reasoning': Explain how the physical arrangement supports this specific, grounded interpretation.\n"
    "- 'additions': Additional full features that would help visualize the scene and give it personality:\n"
    "    - 'feature': A concrete, fully-formed physical part that grows from that object.\n"
    "    - 'placement': The exact object and edge the mark starts from .\n\n"
    "Return ONLY valid JSON."
)

DRAW_PROMPT = (
"(Context only — do not draw this scene directly): {scene}\n\n"
    "SPECIFIC ADDITIONS TO DRAW — use these as your core guide:\n"
    "{anchors}\n\n"
    "Draw these additions as black overlay doodles onto the photo. Treat the list as a strong starting point, not a rigid checklist — "
    "you have creative freedom to adjust a mark's exact shape, add a small extra touch, or skip one that wouldn't "
    "read well, if it makes the overall result feel more alive and cohesive as a single drawing.\n\n"
    "STRICT RULES:\n"
    "1. PRESERVATION: The original photo must remain 100% unchanged underneath — no recoloring, no re-painting, no regenerating or reconstructing any part of the objects or background. You may not remove or add objects.\n"
    "2. ANCHORING: Each line must begin exactly at an object's edge, whether from the list or your own small addition. Stay attached to the objects — don't float marks in empty space.\n"
    "3. NO TRACING: Do not outline or trace the existing objects themselves — only draw what extends from them.\n"
    "4. STYLE: black doodle style drawing on top of original photo\n"
    "5. SUPPORTING ROLE: Every addition is a minor, secondary detail — never the main subject or visual focal point. The existing objects remain the primary subject of the image; if the original objects were removed, the additions alone should look like disconnected spare parts, not a recognizable creature, body, or full scene.\n\n"
    "Output the photo with these additions drawn on top of it, so the user can see if you 'see' what they built."
    )

VALIDATE_PROMPT = (
    "You proposed a scene guess and additions to draw onto a sketch of physical objects.\n\n"
    "SCENE: {scene}\n"
    "PROPOSED ADDITIONS:\n{anchors}\n\n"
    "Check: does any addition merely restate, rename, or redraw a shape that is ALREADY visibly present in the sketch, "
    "rather than being a genuinely new mark extending from it?\n\n"
    "Return ONLY valid JSON: {{\"ok\": true or false, \"violation\": \"<if false, which addition duplicates an existing object and why>\"}}"
)

def _to_silhouette(jpeg: bytes) -> bytes:
    import cv2, numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image for sketch preprocessing")
    h, w = img.shape[:2]

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Multi-scale Canny: fine details + broad structure, both on white canvas
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_fine   = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_coarse = cv2.GaussianBlur(gray, (7, 7), 0)
    edges_fine   = cv2.Canny(blur_fine,   20,  60)
    edges_coarse = cv2.Canny(blur_coarse,  8,  30)
    edges = cv2.bitwise_or(edges_fine, edges_coarse)
    thick = cv2.dilate(edges, np.ones((4, 4), np.uint8), iterations=1)
    for cnt in cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < 100:
            continue
        cv2.drawContours(canvas, [cnt], -1, (0, 0, 0), 2)

    # Soften jagged corners
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    _, canvas = cv2.threshold(canvas, 200, 255, cv2.THRESH_BINARY)

    _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return bytes(buf)


def _parse_json_plan(text: str) -> tuple[str, str]:
    import json, re
    try:
        s = text.strip()
        m = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', s)
        if m:
            s = m.group(1)
        data = json.loads(s)
        scene = data.get("scene", "")
        lines = [
            f"- Draw {a.get('feature', '')} {a.get('placement', '')}".strip()
            for a in data.get("additions", [])
            if a.get('feature', '').strip()
        ]
        return scene, "\n".join(lines)
    except Exception:
        m2 = re.search(r'(?:INTERPRETATION|SCENE):\s*(.+)', text, re.IGNORECASE)
        return (m2.group(1).strip() if m2 else ""), text


_stage = ""   # set during run_predict, read by /status

ROUNDS_PER_GAME = 3
LEADERBOARD_PATH = os.path.join(APP_DIR, "leaderboard.json")

_game = {"round": 0, "history": []}  # history: [{round, scene, success}]


def _history_text(history: list) -> str:
    if not history:
        return "This is your first round — no history yet."
    lines = ["ROUND HISTORY (most recent last) — learn from these to improve your guessing:"]
    for h in history:
        outcome = "CORRECT" if h["success"] is True else "WRONG" if h["success"] is False else "pending"
        lines.append(f"- Round {h['round']}: you guessed '{h['scene']}' — {outcome}")

    banned = [h["scene"] for h in history if h.get("scene")]
    if banned:
        lines.append("")
        lines.append("BANNED SCENES — do not guess any of these again, in any form:")
        for b in banned:
            lines.append(f"- {b}")

    return "\n".join(lines)


def _load_leaderboard() -> list:
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    try:
        with open(LEADERBOARD_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _append_leaderboard(score: int):
    board = _load_leaderboard()
    board.append({"score": score, "ts": datetime.datetime.now().isoformat()})
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False, indent=2)


def _score_distribution() -> list:
    board = _load_leaderboard()
    total = len(board)
    dist = []
    for s in range(ROUNDS_PER_GAME + 1):
        count = sum(1 for e in board if e["score"] == s)
        pct = round(100 * count / total) if total else 0
        dist.append({"score": s, "count": count, "pct": pct})
    return dist


def _gemini_with_retry(call, retries=3, delay=5):
    import time
    for attempt in range(retries):
        try:
            return call()
        except Exception as e:
            msg = str(e)
            if attempt < retries - 1 and ("502" in msg or "503" in msg or "Bad Gateway" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg):
                wait = delay * (attempt + 1)
                print(f"[Retry {attempt+1}] {msg[:80]} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise


def _translate_to_hebrew(text: str) -> str:
    if not text.strip():
        return text
    try:
        r = _gemini_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Part.from_text(text=(
                "Translate the following short phrase into natural, simple Hebrew. "
                "Output ONLY the Hebrew translation — no quotes, no explanation, no other text.\n\n"
                f"{text}"
            ))],
            config=types.GenerateContentConfig(temperature=0.2),
        ))
        translated = (r.text or "").strip() if r else ""
        return translated or text
    except Exception as exc:
        print(f"[Translate error] {exc}")
        return text


def _validate_additions(sketch: bytes, scene: str, anchors_text: str) -> tuple[bool, str]:
    if not anchors_text.strip():
        return True, ""
    import re
    try:
        r = _gemini_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=sketch, mime_type="image/jpeg"),
                types.Part.from_text(text=VALIDATE_PROMPT.format(scene=scene, anchors=anchors_text)),
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        ))
        s = (r.text or "").strip() if r else ""
        m = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', s)
        if m:
            s = m.group(1)
        data = json.loads(s)
        return bool(data.get("ok", True)), str(data.get("violation", ""))
    except Exception as exc:
        print(f"[Validate error] {exc}")
        return True, ""  # fail open — don't block the round on a validator hiccup


def run_predict(jpeg: bytes, history: list | None = None) -> tuple[bytes, str, str, bytes]:
    global _stage
    sketch = _to_silhouette(jpeg)

    # Shrink silhouette for step 1 — interpretation only needs shapes, not detail
    import cv2 as _cv2i, numpy as _npi
    _si = _cv2i.imdecode(_npi.frombuffer(sketch, _npi.uint8), _cv2i.IMREAD_COLOR)
    if _si is not None and _si.shape[1] > 512:
        _sw = 512; _sh = int(_si.shape[0] * _sw / _si.shape[1])
        _si = _cv2i.resize(_si, (_sw, _sh))
        _, _sbuf = _cv2i.imencode('.jpg', _si, [_cv2i.IMWRITE_JPEG_QUALITY, 60])
        sketch_small = bytes(_sbuf)
    else:
        sketch_small = sketch

    # Step 1: interpret from sketch only (text model) — validated below; if the
    # validator finds an addition that just duplicates an existing object, we
    # re-run interpretation instead of proceeding to draw it.
    _stage = "interpreting"
    MAX_INTERPRET_ATTEMPTS = 3
    scene, anchors_text = "", ""
    for attempt in range(MAX_INTERPRET_ATTEMPTS):
        print(f"\n[Step 1] interpret  silhouette={len(sketch_small)}b → gemini-2.5-flash-lite (attempt {attempt + 1})")
        r1 = _gemini_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=sketch_small, mime_type="image/jpeg"),
                types.Part.from_text(text=INTERPRET_PROMPT),
                types.Part.from_text(text=_history_text(history or [])),
            ],
            config=types.GenerateContentConfig(temperature=1.0),
        ))
        raw = (r1.text or "") if r1 else ""
        scene, anchors_text = _parse_json_plan(raw)
        print(f"[Scene] {scene}\n[Anchors]\n{anchors_text}")

        ok, violation = _validate_additions(sketch_small, scene, anchors_text)
        if ok:
            break
        print(f"[Validate] rejected (attempt {attempt + 1}): {violation} — re-running interpretation")
    else:
        print("[Validate] still rejected after max attempts — proceeding with last result anyway")

    # Step 2: draw on the real photo — the model occasionally returns a
    # text-only response with no image part even on a successful call, so
    # this is retried a couple of times before giving up.
    _stage = "drawing"
    _draw_contents: list[types.Content] = []
    for _orig, _pred in _FEW_SHOT_EXAMPLES:
        _draw_contents.append(types.Content(role="user", parts=[
            types.Part.from_text(text="Add black line drawings to reveal a hidden scene in this photo:"),
            types.Part.from_bytes(data=_orig, mime_type="image/jpeg"),
        ]))
        _draw_contents.append(types.Content(role="model", parts=[
            types.Part.from_bytes(data=_pred, mime_type="image/jpeg"),
        ]))
    _draw_contents.append(types.Content(role="user", parts=[
        types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
        types.Part.from_text(text=DRAW_PROMPT.format(scene=scene, anchors=anchors_text)),
    ]))

    MAX_DRAW_ATTEMPTS = 3
    result_img = None
    for draw_attempt in range(MAX_DRAW_ATTEMPTS):
        print(f"\n[Step 2] draw  photo={len(jpeg)}b → gemini-2.5-flash-image ({len(_FEW_SHOT_EXAMPLES)} examples, attempt {draw_attempt + 1})")
        r2 = _gemini_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=_draw_contents,  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],  # type: ignore[call-arg]
                temperature=1.0),
        ))
        assert r2 is not None
        cands  = r2.candidates or []
        cparts = cands[0].content.parts if cands and cands[0].content else []  # type: ignore[union-attr]

        for part in (cparts or []):
            idata = getattr(part, "inline_data", None)
            if idata and getattr(idata, "data", None):
                result_img = bytes(idata.data)  # type: ignore[arg-type]

        if result_img is not None:
            break
        print(f"[Step 2] no image in response (attempt {draw_attempt + 1}) — retrying")

    _stage = ""
    if result_img is None:
        raise RuntimeError("Model returned no image")

    # Compress prediction for faster download to browser
    import cv2 as _cv2, numpy as _np
    _arr = _cv2.imdecode(_np.frombuffer(result_img, _np.uint8), _cv2.IMREAD_COLOR)
    if _arr is not None:
        _h, _w = _arr.shape[:2]
        if _w > 1024:
            _s = 1024 / _w
            _arr = _cv2.resize(_arr, (1024, int(_h * _s)))
        _, _buf = _cv2.imencode('.jpg', _arr, [_cv2.IMWRITE_JPEG_QUALITY, 82])
        result_img = bytes(_buf)

    # Translation is a fully separate call, made only after interpretation/drawing
    # are done — the Hebrew text is for display only and never re-enters the
    # game's history/banned-scenes context, which stays in English.
    caption_he = _translate_to_hebrew(scene)
    print(f"[Translate] '{scene}' → '{caption_he}'")

    return result_img, scene, caption_he, sketch


# ── git ───────────────────────────────────────────────────────────────────────

def git_push(ts: str):
    try:
        subprocess.run(["git", "add", "saves/"],         cwd=APP_DIR, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"snap {ts}"], cwd=APP_DIR, check=True, capture_output=True)
        subprocess.run(["git", "push"],                  cwd=APP_DIR, check=True, capture_output=True)
        print(f"[Git] pushed snap {ts}")
    except subprocess.CalledProcessError as e:
        print(f"[Git push failed] {e.stderr.decode()[:300]}")

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return PAGE

@app.route("/status")
def status():
    return jsonify({"stage": _stage})

@app.route("/button-poll")
def button_poll():
    with _button_lock:
        event = _button_queue.pop(0) if _button_queue else None
    return jsonify({"event": event})

@app.route("/silhouette", methods=["POST"])
def silhouette_route():
    try:
        jpeg = base64.b64decode(request.get_json()["image"])
        sketch = _to_silhouette(jpeg)
        return jsonify({"silhouette": base64.b64encode(sketch).decode()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/game/start", methods=["POST"])
def game_start():
    global _game
    _game = {"round": 1, "history": []}
    print("[Game] started")
    return jsonify({"round": _game["round"], "rounds": ROUNDS_PER_GAME})


@app.route("/snap", methods=["POST"])
def snap():
    try:
        if not (1 <= _game["round"] <= ROUNDS_PER_GAME):
            return jsonify({"error": "No active game — start a game first"}), 400

        jpeg = base64.b64decode(request.get_json()["image"])
        pred_bytes, scene, caption_he, sketch = run_predict(jpeg, history=_game["history"])

        os.makedirs(SAVES_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(SAVES_DIR, f"{ts}_original.jpg"),    "wb") as f: f.write(jpeg)
        with open(os.path.join(SAVES_DIR, f"{ts}_prediction.jpg"),  "wb") as f: f.write(pred_bytes)
        with open(os.path.join(SAVES_DIR, f"{ts}_silhouette.jpg"),  "wb") as f: f.write(sketch)
        if caption_he:
            with open(os.path.join(SAVES_DIR, f"{ts}_scene.txt"), "w", encoding="utf-8") as f:
                f.write(caption_he)
        print(f"[Saved] round {_game['round']} — {ts}")
        threading.Thread(target=git_push, args=(ts,), daemon=True).start()

        # "scene" (English) feeds the model's own history/banned-scenes context;
        # "caption" (Hebrew) is display-only, for the summary screen — translation
        # never re-enters the game's reasoning loop.
        _game["history"].append({
            "round": _game["round"], "scene": scene, "caption": caption_he,
            "success": None, "ts": ts,
        })

        _send_arduino("RESULT_ON")
        return jsonify({
            "prediction": base64.b64encode(pred_bytes).decode(),
            "original":   base64.b64encode(jpeg).decode(),
            "silhouette": base64.b64encode(sketch).decode(),
            "caption":    caption_he,
            "ts":         ts,
            "round":      _game["round"],
            "rounds":     ROUNDS_PER_GAME,
        })
    except Exception as exc:
        print(f"[snap error] {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/round-result", methods=["POST"])
def round_result():
    try:
        success = bool((request.get_json() or {}).get("success"))
        for h in reversed(_game["history"]):
            if h["success"] is None:
                h["success"] = success
                break

        _send_arduino("RESULT_OFF")

        if _game["round"] >= ROUNDS_PER_GAME:
            score = sum(1 for h in _game["history"] if h["success"])
            _append_leaderboard(score)
            result = {
                "done":        True,
                "score":       score,
                "rounds":      ROUNDS_PER_GAME,
                "history":     _game["history"],
                "leaderboard": _score_distribution(),
            }
            _game["round"] = 0
            return jsonify(result)

        _game["round"] += 1
        return jsonify({"done": False, "round": _game["round"], "rounds": ROUNDS_PER_GAME})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/leaderboard")
def leaderboard_route():
    return jsonify(_score_distribution())


@app.route("/gallery")
def gallery_route():
    subprocess.run(["git", "pull"], cwd=APP_DIR, capture_output=True)
    items = []
    if os.path.exists(SAVES_DIR):
        for f in sorted(os.listdir(SAVES_DIR), reverse=True):
            if not f.endswith("_prediction.jpg"):
                continue
            ts  = f[:-len("_prediction.jpg")]
            cap = ""
            txt = os.path.join(SAVES_DIR, f"{ts}_scene.txt")
            if os.path.exists(txt):
                with open(txt, encoding="utf-8") as tf:
                    cap = tf.read().strip()
            items.append({"ts": ts, "caption": cap})
    return jsonify(items)


@app.route("/saves/<path:filename>")
def serve_save(filename):
    return send_from_directory(SAVES_DIR, filename)

@app.route("/fonts/<path:filename>")
def serve_font(filename):
    return send_from_directory(FONT_DIR, filename)

# ── HTML page ─────────────────────────────────────────────────────────────────

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hidden Scene</title>
<style>
@font-face{font-family:'Masada';src:url('/fonts/Masada-Light.otf')  format('opentype');font-weight:300}
@font-face{font-family:'Masada';src:url('/fonts/Masada-LightItalic.otf') format('opentype');font-weight:300;font-style:italic}
@font-face{font-family:'Masada';src:url('/fonts/Masada-Book.otf')   format('opentype');font-weight:400}
@font-face{font-family:'Masada';src:url('/fonts/Masada-BookItalic.otf') format('opentype');font-weight:400;font-style:italic}
@font-face{font-family:'Masada';src:url('/fonts/Masada-Medium.otf') format('opentype');font-weight:500}
@font-face{font-family:'Masada';src:url('/fonts/Masada-Demi.otf')   format('opentype');font-weight:600}
@font-face{font-family:'Masada';src:url('/fonts/Masada-Bold.otf')   format('opentype');font-weight:700}
@font-face{font-family:'Masada';src:url('/fonts/Masada-Black.otf')  format('opentype');font-weight:900}
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#1e2021;--surface:#1a1a19;--fg:#fff;
  --dim:#2c2c2a;--med:#c3c2b7;--muted:#898781;
  --border:rgba(255,255,255,.10);
  --card-bg:#ffffff;--arrow:#e2c9da;--arrow-line:#dddddd;--score-orange:#eaac0f;
  --font-head:'Masada',serif;--font-body:'Masada',Helvetica,Arial,sans-serif;
  --series-1:#3987e5;--ok:#64bc69;--bad:#d35454;
  --dot-green:#57965a;--dot-red:#d15524;
  --round1:#d35454;--round2:#eaac0f;--round3:#64bc69;
  --dur-fast:180ms;--dur:320ms;--dur-slow:650ms;
  --ease:cubic-bezier(.4,0,.2,1)
}
body{background:var(--bg);color:var(--fg);font-family:var(--font-body);height:100vh;overflow:hidden}
.phase{
  display:flex;align-items:center;justify-content:center;
  position:fixed;inset:0;background:var(--bg);
  opacity:0;pointer-events:none;transition:opacity 320ms ease
}
.phase.active{opacity:1;pointer-events:auto}

/* ── shared: corner hints, card, round dots, bottom caption ── */
.hint{
  position:fixed;top:20px;z-index:40;white-space:nowrap;
  font-size:20px;font-weight:600;color:#fff;direction:rtl;
  display:flex;align-items:center;gap:10px;line-height:1;
  text-shadow:0 1px 6px rgba(0,0,0,.6)
}
.hint-tl{left:24px}
.hint-tr{right:24px}
.hint-dot{width:24px;height:24px;border-radius:50%;flex-shrink:0;border:2px solid #dddddd}
.hint-dot.white{background:#e5e5e5;box-shadow:0 0 0 6px rgba(255,255,255,.25)}
.hint-dot.green{background:var(--dot-green);box-shadow:0 0 0 6px rgba(87,150,90,.25)}
.hint-dot.red{background:var(--dot-red);box-shadow:0 0 0 6px rgba(209,85,36,.25)}
.hint-q{opacity:.85}
.result-stack{display:flex;flex-direction:column;align-items:center;gap:18px}
.hint-line{
  display:inline-flex;align-items:center;gap:24px;
  direction:rtl;white-space:nowrap;max-width:calc(100vw - 48px);
  font-size:20px;font-weight:600;color:#fff;
  text-shadow:0 1px 6px rgba(0,0,0,.6)
}
.hint-chunk{display:inline-flex;align-items:center;gap:8px;flex-shrink:0}

.card{
  position:relative;width:min(80vw,1000px);aspect-ratio:3/2;
  background:var(--card-bg);border-radius:36px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 20px 60px rgba(0,0,0,.4)
}

.bottom-caption{
  position:fixed;left:0;right:0;bottom:36px;z-index:40;
  text-align:center;color:#fff;font-size:21px;font-weight:600;direction:rtl;
  text-shadow:0 1px 6px rgba(0,0,0,.6);padding:0 24px
}

.round-dots{position:fixed;top:20px;right:24px;z-index:40;display:flex;align-items:center;gap:12px;direction:rtl}
.round-dots .label{font-size:18px;font-weight:700;color:#fff;text-shadow:0 1px 6px rgba(0,0,0,.6)}
.round-dots .dot{width:16px;height:16px;border-radius:50%;border:2px solid var(--med)}
.round-dots .dot.dot-r1.filled{background:var(--round1);border-color:var(--round1)}
.round-dots .dot.dot-r2.filled{background:var(--round2);border-color:var(--round2)}
.round-dots .dot.dot-r3.filled{background:var(--round3);border-color:var(--round3)}

/* ── Gallery ── */
.instr-bg-layer{
  position:absolute;inset:0;width:100%;height:100%;object-fit:cover;background:var(--card-bg);
  opacity:0;z-index:1;
  transition:opacity 1.4s var(--ease);
  animation:kenBurns 6.5s linear forwards
}
.instr-bg-layer.active{opacity:1;z-index:2}
@keyframes kenBurns{from{transform:scale(1)}to{transform:scale(1.06)}}
#instr-none{position:absolute;inset:0;background:var(--card-bg);z-index:0}
#gal-counter{direction:rtl;font-variant-numeric:tabular-nums}

/* ── Tutorial ── */
.tut-content{padding:48px 64px;text-align:center;direction:rtl}
.tut-content h1{font-family:var(--font-head);font-size:60px;font-weight:300;font-style:italic;color:#111;margin-bottom:18px}
.tut-rule{border:none;border-top:2px solid #ddd;width:65%;margin:0 auto 26px;border-radius:2px}
.tut-content p{font-size:22px;font-weight:500;color:#333;line-height:1.85;white-space:pre-line}
.tut-note{position:fixed;z-index:41;font-size:22px;font-weight:600;color:var(--med);line-height:1.4;max-width:220px}
.tut-note-tl{top:122px;left:20px;text-align:left}
.tut-note-tr{top:154px;right:20px;text-align:right}
.tut-note-r{top:calc(50% - 40px);right:20px;text-align:right}
.tut-note-bl{bottom:154px;left:20px;text-align:left}
.tut-arrow{position:fixed;z-index:41}
.tut-arrow-tl{top:50px;left:46px;width:56px;height:64px}
.tut-arrow-tr{top:50px;right:46px;width:56px;height:64px;transform:scaleX(-1)}
.tut-arrow-r{top:calc(50% + 26px);right:110px;width:70px;height:55px}
.tut-arrow-bl{bottom:50px;left:64px;width:70px;height:96px}

/* ── Build ── */
#video-full{width:100%;height:100%;object-fit:cover}
#round-overlay{
  position:absolute;inset:0;z-index:5;
  background:rgba(0,0,0,.5);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;
  opacity:1;pointer-events:none;transition:opacity 700ms ease
}
#round-overlay.hide{opacity:0}
.round-overlay-hint{font-size:22px;font-weight:600;color:#fff;direction:rtl;text-shadow:0 1px 6px rgba(0,0,0,.6)}
.round-overlay-title{font-family:var(--font-head);font-size:72px;font-weight:300;font-style:italic;color:#fff;direction:rtl;text-shadow:0 2px 10px rgba(0,0,0,.6)}

/* ── Loading ── */
#load-original,#load-silhouette{
  position:absolute;inset:0;
  width:100%;height:100%;object-fit:contain;
  opacity:0
}
#load-stage-text{
  opacity:0;transform:translateY(8px);
  transition:opacity var(--dur) var(--ease), transform var(--dur) var(--ease)
}
#load-stage-text.show{opacity:1;transform:translateY(0)}

/* ── Result ── */
#result-img{width:100%;height:100%;object-fit:contain}

/* ── Idle overlay ── */
#ph-idle{
  position:fixed;inset:0;z-index:90;background:rgba(160,160,160,.82);
  display:none;align-items:center;justify-content:center
}
#ph-idle.show{display:flex}
.idle-card{
  background:#fff;border-radius:28px;padding:60px 90px;
  display:flex;flex-direction:column;align-items:center;
  box-shadow:0 20px 60px rgba(0,0,0,.3)
}
.idle-title{font-family:var(--font-head);font-size:64px;font-weight:300;font-style:italic;color:#111;direction:rtl;margin-bottom:26px}
.idle-dots{display:flex;gap:14px;margin-bottom:26px}
.idle-dots span{width:16px;height:16px;border-radius:50%;background:var(--arrow);animation:idlePulse 1.2s ease-in-out infinite}
.idle-dots span:nth-child(2){animation-delay:.15s}
.idle-dots span:nth-child(3){animation-delay:.3s}
@keyframes idlePulse{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
.idle-sub{font-size:22px;font-weight:600;color:#111;direction:rtl}

/* ── Summary ── */
.summary-title{font-family:var(--font-head);font-size:58px;font-weight:300;font-style:italic;color:#fff;text-align:center;margin-bottom:8px}
.summary-subtitle{font-size:19px;font-weight:600;color:var(--muted);text-align:center;margin-bottom:24px;letter-spacing:.04em}
#summary-score{
  font-size:88px;font-weight:700;text-align:center;margin-bottom:20px;
  font-variant-numeric:proportional-nums;animation:popIn var(--dur-slow) var(--ease) both
}
#summary-score.score-bad{color:var(--bad)}
#summary-score.score-mid{color:var(--score-orange)}
#summary-score.score-ok{color:var(--ok)}
#summary-percentile{font-size:20px;font-weight:600;color:#fff;text-align:center;margin-bottom:38px}
@keyframes popIn{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:scale(1)}}
@keyframes cardIn{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
#summary-list{display:flex;direction:rtl;gap:28px;justify-content:center;flex-wrap:wrap;padding:0 40px;max-width:1400px}
.summary-card{
  width:360px;display:flex;flex-direction:column;align-items:center;gap:10px;
  animation:cardIn var(--dur-slow) var(--ease) both
}
.summary-thumb{width:100%;aspect-ratio:3/2;object-fit:contain;background:var(--card-bg);border-radius:14px;cursor:pointer;display:block}
.summary-meta{display:flex;align-items:center;gap:8px}
.summary-label{color:#fff;font-size:17px;font-weight:600}
.summary-icon{
  width:24px;height:24px;border-radius:50%;border:2px solid #dddddd;
  display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#fff;flex-shrink:0
}
.summary-icon.ok{background:var(--dot-green)}
.summary-icon.bad{background:var(--dot-red)}
.summary-return-inline{color:#fff}

/* ── Modal ── */
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:100;align-items:center;justify-content:center;flex-direction:column;padding:24px}
#modal.open{display:flex}
#modal-img{max-width:min(700px,90vw);max-height:70vh;object-fit:contain}
#modal-cap{font-size:18px;font-weight:600;color:var(--fg);text-align:center;margin-top:20px;max-width:min(700px,90vw);line-height:1.5}
#modal-close{position:absolute;top:20px;right:28px;font-size:32px;color:var(--med);cursor:pointer;background:none;border:none;line-height:1;font-family:inherit}
#modal-close:hover{color:var(--fg)}
</style>
</head>
<body>

<svg width="0" height="0">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
      <path d="M0,1 C3,1 5,3 9,5 C5,7 3,9 0,9 C1.5,7 1.5,3 0,1 Z" fill="var(--arrow)"></path>
    </marker>
  </defs>
</svg>

<!-- Gallery -->
<div id="ph-gallery" class="phase active">
  <div class="hint hint-tl"><span class="hint-dot white"></span><span>להתחלה לחצו על הכפתור הלבן</span></div>
  <div class="hint hint-tr" id="gal-counter"></div>
  <div class="card">
    <img id="instr-bg-a" class="instr-bg-layer" alt="">
    <img id="instr-bg-b" class="instr-bg-layer" alt="">
    <div id="instr-none"></div>
  </div>
  <div class="bottom-caption" id="instr-caption"></div>
</div>

<!-- Welcome / pick objects -->
<div id="ph-welcome" class="phase">
  <div class="hint hint-tl"><span class="hint-dot white"></span><span>להמשך לחצו על הכפתור הלבן</span></div>
  <div class="card">
    <div class="tut-content">
      <h1>ברוכים הבאים!</h1>
      <hr class="tut-rule">
      <p>כדי להתחיל ליצור גשו לקיר החפצים מימנכם.
בחרו 7 חפצים שמעניינים אתכם והביאו אותם לשולחן.
לאחר שבחרתם לחצו על הכפתור הלבן.</p>
    </div>
  </div>
</div>

<!-- Tutorial -->
<div id="ph-tutorial" class="phase">
  <div class="hint hint-tl"><span class="hint-dot white"></span><span>להמשך לחצו על הכפתור הלבן</span></div>
  <div class="round-dots" id="tutorial-dots"></div>
  <div class="card">
    <div class="tut-content">
      <h1>ברוכים הבאים!</h1>
      <hr class="tut-rule">
      <p>למשחק יש שלושה סבבים,
בכל סבב עליכם לצייר מאותם חפצים משהו חדש.
לאחר שתבנו המכונה תנסה לנחש מה ציירתם
להתחלת הסבב הראשון לחצו על הכפתור הלבן.

אנא עיינו בהערות סביב המסך לפני תחילת הפעילות.</p>
    </div>
  </div>
  <div class="tut-note tut-note-tl" dir="rtl">להוראות המשך הסבב</div>
  <div class="tut-note tut-note-tr" dir="rtl">לצפייה במספר סבב שאתם נמצאים בו כעת</div>
  <div class="tut-note tut-note-r" dir="rtl">במסך זה יוצג הפעולות שלכם לאורך המשימה</div>
  <div class="tut-note tut-note-bl" dir="rtl">לצפייה בשלבים והצעת התשובות של AI</div>
  <svg class="tut-arrow tut-arrow-tl" viewBox="0 0 56 64"><path d="M8,58 C8,20 32,6 46,4" stroke="var(--arrow-line)" stroke-width="3" stroke-linecap="round" fill="none" marker-end="url(#arrowhead)"></path></svg>
  <svg class="tut-arrow tut-arrow-tr" viewBox="0 0 56 64"><path d="M8,58 C8,20 32,6 46,4" stroke="var(--arrow-line)" stroke-width="3" stroke-linecap="round" fill="none" marker-end="url(#arrowhead)"></path></svg>
  <svg class="tut-arrow tut-arrow-r" viewBox="0 0 70 55"><path d="M60,6 C50,20 25,32 8,42" stroke="var(--arrow-line)" stroke-width="3" stroke-linecap="round" fill="none" marker-end="url(#arrowhead)"></path></svg>
  <svg class="tut-arrow tut-arrow-bl" viewBox="0 0 70 96"><path d="M8,8 C8,60 30,80 62,90" stroke="var(--arrow-line)" stroke-width="3" stroke-linecap="round" fill="none" marker-end="url(#arrowhead)"></path></svg>
</div>

<!-- Build -->
<div id="ph-build" class="phase">
  <div class="hint hint-tl"><span class="hint-dot white"></span><span>להגשה תלחצו על הכפתור הלבן</span></div>
  <div class="round-dots" id="build-dots"></div>
  <div class="card">
    <video id="video-full" autoplay playsinline muted></video>
    <div id="round-overlay">
      <div class="round-overlay-hint" id="round-overlay-hint"></div>
      <div class="round-overlay-title" id="round-overlay-title"></div>
    </div>
  </div>
</div>

<!-- Loading -->
<div id="ph-loading" class="phase">
  <div class="hint hint-tl" dir="rtl">המתינו לתשובת המכונה</div>
  <div class="round-dots" id="loading-dots"></div>
  <div class="card">
    <img id="load-original"   alt="">
    <img id="load-silhouette" alt="">
  </div>
  <div class="bottom-caption" id="load-stage-text"></div>
</div>

<!-- Result -->
<div id="ph-result" class="phase">
  <div class="result-stack">
    <div class="hint-line">
      <span class="hint-q" dir="rtl">האם זה תואם למה שדמיינתם?</span>
      <span class="hint-chunk"><span class="hint-dot green"></span><span dir="rtl">לחצו על הכפתור הירוק אם כן</span></span>
      <span class="hint-chunk"><span class="hint-dot red"></span><span dir="rtl">לחצו על הכפתור האדום אם לא</span></span>
    </div>
    <div class="card"><img id="result-img" alt=""></div>
  </div>
  <div class="bottom-caption" id="result-caption"></div>
</div>

<!-- Idle check -->
<div id="ph-idle">
  <div class="idle-card">
    <div class="idle-title">עדיין איתנו?</div>
    <div class="idle-dots"><span></span><span></span><span></span></div>
    <div class="idle-sub">לחצו על הכפתור הלבן על מנת להמשיך</div>
  </div>
</div>

<!-- Summary -->
<div id="ph-summary" class="phase" style="flex-direction:column">
  <div class="hint hint-tl"><span class="hint-dot white"></span><span>לניסיון נוסף אנא לחצו על הכפתור הלבן</span></div>
  <div class="hint hint-tr" dir="rtl">כל הכבוד, השלמתם את כל הסבבים!</div>
  <div class="summary-title" dir="rtl">סיימנו! <span class="summary-return-inline">בבקשה תחזירו את כל החפצים למקום על הקיר</span></div>
  <div class="summary-subtitle" dir="rtl">התוצאות שלך</div>
  <div id="summary-score"></div>
  <div id="summary-percentile" dir="rtl"></div>
  <div id="summary-list"></div>

</div>

<!-- Modal -->
<div id="modal" onclick="closeModal(event)">
  <button id="modal-close" onclick="closeModal()">&times;</button>
  <img id="modal-img" alt="">
  <div id="modal-cap"></div>
</div>

<script>
let stream       = null;
let currentRound = 0;
let roundsTotal  = 3;

function show(id){
  document.querySelectorAll('.phase').forEach(p => p.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  armIdle();
}

// round-transition screen content per round
const ROUND_INTROS = {
  1: { hint: 'יאללה מתחילים!',        title: 'סבב ראשון!'  },
  2: { hint: 'יאללה ממשיכים!',        title: 'סבב שני!'    },
  3: { hint: 'יאללה עוד אחד ואחרון!', title: 'סבב שלישי!'  },
};
function renderRoundDots(elId, round){
  let html = '<span class="label">סבב</span>';
  for(let i = 1; i <= roundsTotal; i++){
    html += `<span class="dot dot-r${i}${i === round ? ' filled' : ''}"></span>`;
  }
  document.getElementById(elId).innerHTML = html;
}
function configureRoundScreen(n){
  renderRoundDots('build-dots', n);
  renderRoundDots('loading-dots', n);
}

// round title shows as a semi-transparent overlay on top of the live camera
// feed for a couple seconds, then fades out smoothly on its own
const ROUND_OVERLAY_MS = 2200;
let roundOverlayTimer = null;
function showRoundScreen(n){
  const info = ROUND_INTROS[n] || ROUND_INTROS[1];
  configureRoundScreen(n);
  document.getElementById('round-overlay-hint').textContent  = info.hint;
  document.getElementById('round-overlay-title').textContent = info.title;
  const overlay = document.getElementById('round-overlay');
  overlay.classList.remove('hide');
  show('ph-build');
  clearTimeout(roundOverlayTimer);
  roundOverlayTimer = setTimeout(() => overlay.classList.add('hide'), ROUND_OVERLAY_MS);
}

// idle "still with us?" overlay — shown after 60s with no button press
// while waiting on the camera, a prediction, or a result
let idleTimer = null;
let idleResetTimer = null;
function armIdle(){
  clearTimeout(idleTimer);
  const active  = document.querySelector('.phase.active');
  const watched = ['ph-build', 'ph-loading', 'ph-result'];
  if(active && watched.includes(active.id)){
    idleTimer = setTimeout(showIdle, 120000);
  }
}
function showIdle(){
  document.getElementById('ph-idle').classList.add('show');
  clearTimeout(idleResetTimer);
  idleResetTimer = setTimeout(() => { hideIdle(); playAgain(); }, 30000);
}
function hideIdle(){
  document.getElementById('ph-idle').classList.remove('show');
  clearTimeout(idleResetTimer);
}
function dismissIdleIfShown(){
  if(document.getElementById('ph-idle').classList.contains('show')){
    hideIdle();
    armIdle();
    return true;
  }
  return false;
}

function openModal(src, caption){
  document.getElementById('modal-img').src = src;
  document.getElementById('modal-cap').textContent = caption || '';
  document.getElementById('modal').classList.add('open');
}

// camera
async function initCamera(){
  if(stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
  document.getElementById('video-full').srcObject = stream;
}

// instructions gallery slideshow
let galItems   = [];
let galIdx     = 0;
let galSlideIv = null;
let galFront   = 'a'; // which layer ('a'/'b') is currently on top

async function initGallerySlideshow(){
  try{ galItems = await (await fetch('/gallery')).json(); }
  catch(e){ galItems = []; }
  document.getElementById('instr-none').style.display = galItems.length ? 'none' : 'block';
  document.getElementById('gal-counter').textContent = '';
  if(galItems.length){ galIdx = 0; showSlide(galIdx); startSlideshow(); }
  else { document.getElementById('instr-caption').textContent = ''; }
}

function showSlide(idx){
  const it = galItems[idx];
  if(!it) return;

  const back  = document.getElementById('instr-bg-' + (galFront === 'a' ? 'b' : 'a'));
  const front = document.getElementById('instr-bg-' + galFront);
  const cap   = document.getElementById('instr-caption');

  back.src = '/saves/' + it.ts + '_prediction.jpg';
  back.style.animation = 'none';
  void back.offsetWidth; // restart the Ken Burns zoom for the incoming image
  back.style.animation = '';
  back.classList.add('active');
  front.classList.remove('active');
  galFront = (galFront === 'a' ? 'b' : 'a');

  cap.classList.add('fading');
  setTimeout(() => {
    cap.textContent = it.caption || '';
    document.getElementById('gal-counter').textContent = `${idx + 1} / ${galItems.length}`;
    cap.classList.remove('fading');
  }, 320);
}

function startSlideshow(){
  if(galSlideIv) clearInterval(galSlideIv);
  if(galItems.length < 2) return;
  galSlideIv = setInterval(() => { galIdx = (galIdx + 1) % galItems.length; showSlide(galIdx); }, 6000);
}

function stopSlideshow(){
  if(galSlideIv){ clearInterval(galSlideIv); galSlideIv = null; }
  document.getElementById('instr-bg-a').classList.remove('active');
  document.getElementById('instr-bg-b').classList.remove('active');
}

// game
function goWelcome(){
  stopSlideshow();
  show('ph-welcome');
}

function goTutorial(){
  renderRoundDots('tutorial-dots', 0);
  show('ph-tutorial');
}

async function beginGame(){
  const res  = await fetch('/game/start', { method:'POST', headers:{'Content-Type':'application/json'} });
  const data = await res.json();
  currentRound = data.round;
  roundsTotal  = data.rounds;
  await initCamera();
  showRoundScreen(currentRound);
}

// snap
const stageLabels = {
  interpreting: 'בתהליך דמיון',
  drawing:      'בתהליך ציור',
  finishing:    'ממם.. מעניין מה יצרתם פה',
};
let statusIv   = null;
let sawDrawing = false;

function setStageText(t){
  const el = document.getElementById('load-stage-text');
  if(el.textContent === t){ el.classList.add('show'); return; }
  el.classList.remove('show');
  setTimeout(() => { el.textContent = t; el.classList.add('show'); }, 220);
}
function startStatusPoll(){
  sawDrawing = false;
  setStageText(stageLabels.interpreting);
  statusIv = setInterval(async () => {
    try{
      const d = await (await fetch('/status')).json();
      if(d.stage === 'drawing') sawDrawing = true;
      const txt = d.stage ? stageLabels[d.stage] : (sawDrawing ? stageLabels.finishing : null);
      if(txt) setStageText(txt);
    } catch(e){}
  }, 700);
}
function stopStatusPoll(){
  if(statusIv){ clearInterval(statusIv); statusIv = null; }
  document.getElementById('load-stage-text').classList.remove('show');
}

let snapBusy = false;
async function fireSnap(){
  if(snapBusy) return;
  snapBusy = true;

  const vid = document.getElementById('video-full');
  const srcW = vid.videoWidth  || 1280;
  const srcH = vid.videoHeight || 720;

  // full-size for display
  const cvsFull = document.createElement('canvas');
  cvsFull.width  = srcW;
  cvsFull.height = srcH;
  cvsFull.getContext('2d').drawImage(vid, 0, 0);
  const displayB64 = cvsFull.toDataURL('image/jpeg', 0.9).split(',')[1];

  // smaller version for API upload
  const MAX = 640;
  const scale = srcW > MAX ? MAX / srcW : 1;
  const cvsSmall = document.createElement('canvas');
  cvsSmall.width  = Math.round(srcW * scale);
  cvsSmall.height = Math.round(srcH * scale);
  cvsSmall.getContext('2d').drawImage(vid, 0, 0, cvsSmall.width, cvsSmall.height);
  const b64 = cvsSmall.toDataURL('image/jpeg', 0.75).split(',')[1];

  const origEl = document.getElementById('load-original');
  const silEl  = document.getElementById('load-silhouette');
  origEl.style.opacity = '0';
  silEl.style.opacity  = '0';
  show('ph-loading');
  startStatusPoll();

  origEl.style.transition = 'none';
  origEl.src              = 'data:image/jpeg;base64,' + displayB64;
  origEl.style.opacity    = '1';

  fetch('/silhouette', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image: displayB64})
  }).then(r => r.json()).then(d => {
    if(d.silhouette){
      silEl.src = 'data:image/jpeg;base64,' + d.silhouette;
      const ease = 'opacity 6s cubic-bezier(0.4,0,0.2,1)';
      origEl.style.transition = ease;
      silEl.style.transition  = ease;
      requestAnimationFrame(() => {
        origEl.style.opacity = '0';
        silEl.style.opacity  = '1';
      });
    }
  }).catch(() => {});

  try{
    const res  = await fetch('/snap', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image: b64})
    });
    const data = await res.json();
    stopStatusPoll();
    if(data.error) throw new Error(data.error);
    currentRound = data.round;
    showResult(data);
  } catch(e){
    stopStatusPoll();
    console.error('[snap error]', e.message);
    show('ph-build');
    flashBuildError('המכונה לא הצליחה הפעם, נסו שוב');
  }
  snapBusy = false;
}

// brief visible error banner on the camera screen — reuses the round-overlay
// styling so a failed prediction doesn't silently drop the user back to the
// camera with no explanation
let buildErrorTimer = null;
function flashBuildError(msg){
  const overlay = document.getElementById('round-overlay');
  document.getElementById('round-overlay-hint').textContent  = msg;
  document.getElementById('round-overlay-title').textContent = '';
  overlay.classList.remove('hide');
  clearTimeout(buildErrorTimer);
  clearTimeout(roundOverlayTimer);
  buildErrorTimer = setTimeout(() => overlay.classList.add('hide'), 2600);
}

// result
function showResult(data){
  const ph = document.getElementById('ph-result');
  ph.style.transform = ''; ph.style.opacity = '';
  document.getElementById('result-img').src = 'data:image/jpeg;base64,' + data.prediction;
  document.getElementById('result-caption').textContent = data.caption || '';
  show('ph-result');
}

let resultBusy = false;
async function markCorrect(){ await submitRoundResult(true); }
async function markWrong(){ await submitRoundResult(false); }

async function submitRoundResult(success){
  if(resultBusy) return;
  resultBusy = true;

  const ph = document.getElementById('ph-result');
  ph.style.transition = 'transform .65s cubic-bezier(.4,0,.2,1), opacity .6s';
  ph.style.transform  = 'scale(0.08) translateY(60vh)';
  ph.style.opacity    = '0';
  await new Promise(r => setTimeout(r, 620));

  try{
    const res  = await fetch('/round-result', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({success})
    });
    const data = await res.json();
    if(data.done){
      showSummary(data);
    } else {
      currentRound = data.round;
      showRoundScreen(currentRound);
    }
  } catch(e){
    show('ph-build');
  }
  resultBusy = false;
}

// summary
const ROUND_ORDINALS = ['ראשונה', 'שנייה', 'שלישית'];

function showSummary(data){
  const scoreEl = document.getElementById('summary-score');
  scoreEl.textContent = `${data.score} / ${data.rounds}`;
  scoreEl.classList.remove('score-bad', 'score-mid', 'score-ok');
  scoreEl.classList.add(data.score <= 0 ? 'score-bad' : data.score === 1 ? 'score-mid' : 'score-ok');

  const list = document.getElementById('summary-list');
  list.innerHTML = '';
  data.history.forEach((h, i) => {
    const card = document.createElement('div');
    card.className = 'summary-card';
    card.style.animationDelay = (i * 90) + 'ms';

    const thumb = document.createElement('img');
    thumb.className = 'summary-thumb';
    thumb.src = '/saves/' + h.ts + '_prediction.jpg';
    thumb.alt = '';
    thumb.onclick = () => openModal(thumb.src, h.caption);

    const meta = document.createElement('div');
    meta.className = 'summary-meta';
    const label = document.createElement('span');
    label.className = 'summary-label';
    label.dir = 'rtl';
    label.textContent = `תשובה ${ROUND_ORDINALS[i] || (i + 1)}`;
    const icon = document.createElement('span');
    icon.className = 'summary-icon ' + (h.success ? 'ok' : 'bad');
    icon.textContent = h.success ? '✓' : '✗';
    meta.append(label, icon);

    card.append(thumb, meta);
    list.appendChild(card);
  });

  // gamified percentile stat — "you beat X% of exhibition participants"
  const pctEl = document.getElementById('summary-percentile');
  const lower = data.leaderboard.filter(e => e.score < data.score).reduce((a, e) => a + e.count, 0);
  const total = data.leaderboard.reduce((a, e) => a + e.count, 0);
  if(total > 1){
    const beat = Math.round(100 * lower / (total - 1));
    pctEl.textContent = `הצלחת יותר מ-${beat}% מהמשתתפים בתערוכה`;
  } else {
    pctEl.textContent = 'אתם המבקרים הראשונים שמנסים!';
  }

  currentRound = 0;
  show('ph-summary');
}

function playAgain(){
  show('ph-gallery');
  initGallerySlideshow();
}

// modal
function closeModal(e){
  if(e && e.target !== document.getElementById('modal') && e.target !== document.getElementById('modal-close')) return;
  document.getElementById('modal').classList.remove('open');
  document.getElementById('modal-img').src = '';
}

// keyboard — Enter advances, Escape marks wrong on the result screen
// debug: jump straight to the summary screen with real saved images, no rounds needed
async function debugSummary(){
  let items = [];
  try{ items = await (await fetch('/gallery')).json(); } catch(e){}

  const sample = items.slice(0, roundsTotal);
  const history = sample.map((it, i) => ({
    ts: it.ts, caption: it.caption || `(debug ${i + 1})`, success: i % 2 === 0,
  }));
  const score = history.filter(h => h.success).length;

  // fabricate a realistic spread across every score bucket so the debug view
  // always shows the real percentile stat, instead of depending on however
  // much (or little) real leaderboard.json data happens to exist right now
  const board = [];
  for(let s = 0; s <= roundsTotal; s++){
    board.push({score: s, count: s === score ? 4 : 1 + Math.round(Math.random() * 6), pct: 0});
  }
  const total = board.reduce((a, e) => a + e.count, 0);
  board.forEach(e => { e.pct = Math.round(100 * e.count / total); });

  showSummary({ score, rounds: roundsTotal, history, leaderboard: board });
}

// shared "advance" action — fired by the Enter key or the white Arduino button
function handleAdvance(){
  if(dismissIdleIfShown()) return;
  armIdle();
  if(document.getElementById('modal').classList.contains('open')){ closeModal(); return; }
  const active = document.querySelector('.phase.active').id;
  if(active === 'ph-gallery'){  goWelcome();  return; }
  if(active === 'ph-welcome'){  goTutorial(); return; }
  if(active === 'ph-tutorial'){ beginGame();  return; }
  if(active === 'ph-build'){    fireSnap();    return; }
  if(active === 'ph-result'){   markCorrect(); return; }
  if(active === 'ph-summary'){  playAgain();   return; }
}

// physical Arduino buttons
function handleYes(){
  if(dismissIdleIfShown()) return;
  armIdle();
  if(document.getElementById('ph-result').classList.contains('active')) markCorrect();
}
function handleNo(){
  if(dismissIdleIfShown()) return;
  armIdle();
  if(document.getElementById('modal').classList.contains('open')){ closeModal(); return; }
  if(document.getElementById('ph-result').classList.contains('active')) markWrong();
}

document.addEventListener('keydown', e => {
  if(e.key === 'd' || e.key === 'D'){ debugSummary(); return; }
  if(dismissIdleIfShown()) return;
  if(e.key === 'Escape'){
    if(document.getElementById('modal').classList.contains('open')){ closeModal(); return; }
    if(document.getElementById('ph-result').classList.contains('active')){ markWrong(); return; }
    return;
  }
  if(e.key === 'Enter'){ handleAdvance(); }
});

// polled continuously — same events physical buttons send over serial
setInterval(async () => {
  try{
    const d = await (await fetch('/button-poll')).json();
    if(d.event === 'ENTER') handleAdvance();
    else if(d.event === 'YES') handleYes();
    else if(d.event === 'NO') handleNo();
  } catch(e){}
}, 200);

initGallerySlideshow();
</script>
</body>
</html>"""

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Open http://localhost:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
