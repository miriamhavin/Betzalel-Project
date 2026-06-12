import os, io, re, base64, datetime, subprocess, threading
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVES_DIR = os.path.join(APP_DIR, "saves")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
app    = Flask(__name__)

# ── prompts ───────────────────────────────────────────────────────────────────

INTERPRETATION_PROMPT = (
    "You are a mythmaker. Look at this image and find the hidden scene inside it.\n\n"
    "A hidden scene is not a description of objects. It is a MOMENT — something happening,\n"
    "just finished, or about to begin — that the arrangement of objects secretly depicts.\n"
    "The objects are evidence. You are the detective who sees what they are evidence OF.\n\n"
    "What makes a scene good:\n"
    "  • It is a single dramatic moment with a before and after\n"
    "  • It uses the objects' shapes, positions, sizes and colors as its raw material\n"
    "  • It would collapse if you removed or moved any object\n"
    "  • It makes you feel something — wonder, dread, tenderness, awe\n"
    "  • It sounds like a sentence from a myth or a fairy tale, not a room description\n\n"
    "Examples of the quality required:\n"
    "  BAD:  'Objects arranged like a miniature landscape'\n"
    "  GOOD: 'The crater lake where the last dragon drank, still warm'\n\n"
    "  BAD:  'A workspace that represents a journey'\n"
    "  GOOD: 'The cartographer\\'s camp the morning the mountain swallowed it'\n\n"
    "  BAD:  'Items forming a metaphorical battle scene'\n"
    "  GOOD: 'The exact moment the giant fell and everything scattered'\n\n"
    "Step 1 — Note each object and its position.\n\n"
    "Step 2 — Write 3 candidate scenes (one sentence each). Push yourself — the third\n"
    "          should be stranger and more specific than the first.\n\n"
    "Step 3 — Pick the strongest. It wins if: (a) it could only be read from THIS exact\n"
    "          arrangement, (b) it contains no real object names, (c) it makes you feel something.\n\n"
    "Step 4 — Assign each object its role in that scene — what it IS in the scene's world,\n"
    "          not what it 'represents'. Concrete nouns only: 'the crater', 'the sleeping god',\n"
    "          'the frozen river', not 'the focal point' or 'the symbol of X'.\n\n"
    "Step 5 — Write drawing instructions: for each object, what lines to draw around it\n"
    "          so the viewer sees the scene role, not the real object. Be specific to position.\n\n"
    "OUTPUT FORMAT (strict, no extra text):\n\n"
    "SCENE: <the winning scene — one full sentence, reads like a story, no real object names>\n\n"
    "OBJECT: <real name> | BOX: [y_min, x_min, y_max, x_max] | ROLE: <what it is in the scene>\n"
    "...\n\n"
    "DRAWING INSTRUCTIONS:\n"
    "<per object: exact lines to draw around it, tied to its scene role and position>\n"
)

SCENE_DRAW_PROMPT = (
    "You are an illustrator expanding a photo into a scene.\n\n"
    "SCENE: {scene}\n\n"
    "The photo shows physical objects. Each object has become something else in the scene above.\n"
    "Your job: draw directly onto the photo to expand the world around each object so the "
    "viewer immediately understands what that object has BECOME — not what it literally is.\n\n"
    "How to expand:\n"
    "• Draw the environment, context, and extensions that GROW OUT OF each object\n"
    "• If an object is a character's body, draw the head, limbs, expression around it\n"
    "• If an object is a landscape feature, draw the horizon, sky, or ground that belongs to it\n"
    "• If an object is a machine part, draw the rest of the machine connecting to it\n"
    "• Extend lines FROM the object outward — make the object the seed of a larger drawing\n"
    "• Connect objects visually where the scene says they interact\n"
    "• The drawn world must make the everyday objects unrecognisable — a viewer should see\n"
    "  the SCENE, not a cup or a laptop or a bag\n\n"
    "DRAWING INSTRUCTIONS (what each object has become and what to draw around it):\n"
    "{instructions}\n\n"
    "STYLE: bold black lines, expressive weight variation, no color fills — "
    "sketch-like but confident. The drawn extensions should feel like they belong "
    "to the same world as the objects, not like labels floating above them.\n"
    "Make the scene readable at a glance — someone should look at the result and "
    "immediately see the scene, not just the original objects.\n"
)

# ── AI pipeline ───────────────────────────────────────────────────────────────

def run_predict(jpeg: bytes) -> tuple[bytes, str]:
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            types.Part.from_text(text=INTERPRETATION_PROMPT),
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)),  # type: ignore[call-arg]
    )
    interp       = r.text or ""
    scene        = _parse(r"SCENE:\s*(.+)", interp)
    short        = _parse(r"SHORT:\s*(.+)", interp)
    objects      = _parse_objects(interp)
    instructions = _parse_instructions(interp)

    if not objects:
        return jpeg, scene

    prompt = SCENE_DRAW_PROMPT.format(scene=scene, instructions=instructions)
    result = _draw(jpeg, prompt)
    return (result or jpeg), scene


def _draw(jpeg: bytes, prompt: str) -> bytes | None:
    r = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]),  # type: ignore[call-arg]
    )
    cands  = r.candidates or []
    cparts = cands[0].content.parts if cands and cands[0].content else []  # type: ignore[union-attr]
    for part in (cparts or []):
        idata = getattr(part, "inline_data", None)
        if idata and getattr(idata, "data", None):
            return bytes(idata.data)  # type: ignore[arg-type]
    return None


def _parse(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_objects(text: str) -> list:
    pat = re.compile(
        r"OBJECT:\s*[^|]+?\s*\|\s*BOX:\s*\[[^\]]+\]\s*\|\s*ROLE:\s*.+",
        re.IGNORECASE)
    return pat.findall(text)


def _parse_instructions(text: str) -> str:
    m = re.search(r"DRAWING INSTRUCTIONS:\s*\n([\s\S]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""

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

@app.route("/snap", methods=["POST"])
def snap():
    try:
        jpeg = base64.b64decode(request.get_json()["image"])
        pred_bytes, caption = run_predict(jpeg)
        return jsonify({
            "prediction": base64.b64encode(pred_bytes).decode(),
            "original":   base64.b64encode(jpeg).decode(),
            "caption":    caption,
        })
    except Exception as exc:
        print(f"[snap error] {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/save", methods=["POST"])
def save():
    try:
        data       = request.get_json()
        original   = base64.b64decode(data["original"])
        prediction = base64.b64decode(data["prediction"])
        caption    = data.get("caption", "")

        os.makedirs(SAVES_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(os.path.join(SAVES_DIR, f"{ts}_original.jpg"),   "wb") as f: f.write(original)
        with open(os.path.join(SAVES_DIR, f"{ts}_prediction.jpg"), "wb") as f: f.write(prediction)
        if caption:
            with open(os.path.join(SAVES_DIR, f"{ts}_scene.txt"), "w", encoding="utf-8") as f:
                f.write(caption)

        threading.Thread(target=git_push, args=(ts,), daemon=True).start()
        return jsonify({"ok": True, "ts": ts})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/gallery")
def gallery():
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

# ── HTML page ─────────────────────────────────────────────────────────────────

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Pipeline</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0a14;--card:#0f0f1e;--dim:#2e3a4e;--med:#64748b;--orange:#f97316;--green:#22c55e;--white:#e2e8f0}
body{background:var(--bg);color:var(--white);font-family:Helvetica,Arial,sans-serif;min-height:100vh}

/* ── layout ── */
#main{display:flex;flex-direction:column;align-items:center;padding:40px 20px 20px}
#panels{display:flex;align-items:flex-start}
.plabel{font-size:11px;color:var(--dim);letter-spacing:1px;margin-bottom:8px;text-align:center}

/* ── live ── */
#video{width:280px;height:210px;background:#000;display:block;object-fit:cover}

/* ── divider ── */
#div{width:1px;background:var(--dim);margin:24px 40px 0;align-self:stretch}

/* ── prediction ── */
#pred-box{width:480px;height:360px;background:var(--card);display:flex;align-items:center;justify-content:center;overflow:hidden}
#pred-img{width:100%;height:100%;object-fit:cover;display:none}
#pred-ph{color:var(--dim);font-size:13px}
#caption{font-family:Georgia,serif;font-style:italic;font-size:14px;color:var(--white);text-align:center;margin-top:18px;min-height:20px;max-width:480px;line-height:1.5}

/* ── controls ── */
#controls{margin-top:32px;display:flex;flex-direction:column;align-items:center;gap:10px}
.btn{border:none;cursor:pointer;font-family:inherit;font-weight:bold;font-size:15px;padding:12px 44px;letter-spacing:.5px}
#snap-btn{background:var(--orange);color:#fff}
#snap-btn:disabled{opacity:.45;cursor:default}
#sd-bar{display:none;gap:12px}
#save-btn{background:var(--green);color:#0a0a14}
#disc-btn{background:#374151;color:var(--white);font-weight:normal}
#status{font-size:13px;min-height:18px;color:var(--orange)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.pulsing{animation:pulse 1.4s ease-in-out infinite}

/* ── gallery ── */
#gal{width:100%;max-width:1100px;margin:64px auto 48px;padding:0 20px}
#gal-hdr{font-size:11px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;text-align:center;border-top:1px solid var(--dim);padding-top:24px;margin-bottom:28px}
#gal-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:18px}
.gi{background:var(--card);cursor:pointer;transition:opacity .15s}
.gi:hover{opacity:.8}
.gi img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}

/* ── modal ── */
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:100;align-items:center;justify-content:center;flex-direction:column;padding:24px}
#modal.open{display:flex}
#modal-img{max-width:min(700px,90vw);max-height:70vh;object-fit:contain;display:block}
#modal-cap{font-family:Georgia,serif;font-style:italic;font-size:15px;color:var(--white);text-align:center;margin-top:20px;max-width:min(700px,90vw);line-height:1.6}
#modal-close{position:absolute;top:20px;right:28px;font-size:28px;color:var(--med);cursor:pointer;line-height:1;background:none;border:none}
#modal-close:hover{color:var(--white)}
</style>
</head>
<body>

<div id="main">
  <div id="panels">
    <div>
      <div class="plabel">live</div>
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas" style="display:none"></canvas>
    </div>
    <div id="div"></div>
    <div>
      <div class="plabel">prediction</div>
      <div id="pred-box">
        <img id="pred-img" alt="">
        <div id="pred-ph">press &nbsp;snap&nbsp; to begin</div>
      </div>
      <div id="caption"></div>
    </div>
  </div>

  <div id="controls">
    <div id="snap-row">
      <button class="btn" id="snap-btn" onclick="doSnap()">snap</button>
    </div>
    <div id="sd-bar">
      <button class="btn" id="save-btn" onclick="doSave()">save</button>
      <button class="btn" id="disc-btn" onclick="doDiscard()">discard</button>
    </div>
    <div id="status"></div>
  </div>
</div>

<div id="gal">
  <div id="gal-hdr">everyone&rsquo;s snaps</div>
  <div id="gal-grid"></div>
</div>

<div id="modal" onclick="closeModal(event)">
  <button id="modal-close" onclick="closeModal()">&times;</button>
  <img id="modal-img" alt="">
  <div id="modal-cap"></div>
</div>

<script>
const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
let pendOrig = null, pendPred = null, pendCap = '';
let stTimer  = null;

// ── camera ────────────────────────────────────────────────────────────────────
navigator.mediaDevices.getUserMedia({
  video:{ width:{ideal:640}, height:{ideal:480} }, audio:false
}).then(s => video.srcObject = s)
  .catch(e => status('camera: ' + e.message, '#ef4444'));

// ── snap ──────────────────────────────────────────────────────────────────────
async function doSnap(){
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];

  document.getElementById('snap-btn').disabled = true;
  document.getElementById('sd-bar').style.display = 'none';
  document.getElementById('snap-row').style.display = 'flex';
  document.getElementById('caption').textContent = '';
  startLoading();

  try {
    const res  = await fetch('/snap', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({image:b64})});
    const data = await res.json();
    stopLoading();

    if(data.error){ status('error: '+data.error,'#ef4444'); document.getElementById('snap-btn').disabled=false; return; }

    const img = document.getElementById('pred-img');
    img.src = 'data:image/jpeg;base64,' + data.prediction;
    img.style.display = 'block';
    document.getElementById('pred-ph').style.display = 'none';

    const cap = data.caption || '';
    document.getElementById('caption').textContent = cap ? '[ '+cap+' ]' : '';

    pendOrig = data.original;
    pendPred = data.prediction;
    pendCap  = cap;

    status('');
    document.getElementById('snap-row').style.display = 'none';
    document.getElementById('sd-bar').style.display   = 'flex';

  } catch(e){
    stopLoading();
    status('network error: '+e.message,'#ef4444');
    document.getElementById('snap-btn').disabled = false;
  }
}

// ── save / discard ────────────────────────────────────────────────────────────
async function doSave(){
  showSnap();
  status('saving…','#64748b');
  try{
    const r = await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({original:pendOrig,prediction:pendPred,caption:pendCap})});
    const d = await r.json();
    if(d.ok){ status('saved','#22c55e'); setTimeout(()=>status(''),3000); setTimeout(loadGallery,1200); }
    else     { status('save failed: '+(d.error||''),'#ef4444'); }
  }catch(e){ status('save error: '+e.message,'#ef4444'); }
  clearPend();
}

function doDiscard(){
  showSnap();
  status('discarded','#2e3a4e');
  setTimeout(()=>status(''),2000);
  clearPend();
}

function showSnap(){
  document.getElementById('sd-bar').style.display   = 'none';
  document.getElementById('snap-row').style.display = 'flex';
  document.getElementById('snap-btn').disabled = false;
}
function clearPend(){ pendOrig=null; pendPred=null; pendCap=''; }

// ── status helpers ────────────────────────────────────────────────────────────
function status(msg, col){
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.color = col || '#f97316';
}
function startLoading(){
  const el = document.getElementById('status');
  const phases = ['finding hidden scene…','finding hidden scene…','drawing the scene…'];
  let i = 0;
  el.className = 'pulsing';
  status(phases[0]);
  stTimer = setInterval(()=>{ i++; if(i < phases.length) status(phases[i]); }, 12000);
}
function stopLoading(){
  if(stTimer){ clearInterval(stTimer); stTimer=null; }
  document.getElementById('status').className='';
}

// ── gallery ───────────────────────────────────────────────────────────────────
async function loadGallery(){
  try{
    const items = await (await fetch('/gallery')).json();
    const grid  = document.getElementById('gal-grid');
    grid.innerHTML = '';
    for(const it of items){
      const d   = document.createElement('div'); d.className='gi';
      d.onclick = ()=> openModal(it.ts, it.caption);
      const img = document.createElement('img');
      img.src     = '/saves/'+it.ts+'_original.jpg';
      img.loading = 'lazy';
      img.alt     = '';
      d.appendChild(img);
      grid.appendChild(d);
    }
  }catch(e){ console.warn('gallery:', e); }
}

// ── modal ─────────────────────────────────────────────────────────────────────
function openModal(ts, caption){
  document.getElementById('modal-img').src = '/saves/'+ts+'_prediction.jpg';
  document.getElementById('modal-cap').textContent = caption ? '[ '+caption+' ]' : '';
  document.getElementById('modal').classList.add('open');
}
function closeModal(e){
  if(e && e.target !== document.getElementById('modal') && e.target !== document.getElementById('modal-close')) return;
  document.getElementById('modal').classList.remove('open');
  document.getElementById('modal-img').src = '';
}
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeModal(); });

loadGallery();
</script>
</body>
</html>"""

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Open http://localhost:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
