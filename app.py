import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import queue
import io
import re
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

VIDEO_W,  VIDEO_H  = 400, 300
DRAW_BACKEND       = os.getenv("DRAW_BACKEND", "gemini")

BG       = "#0f0f1a"
BG_CARD  = "#1a1a2e"
TEXT_DIM = "#64748b"
ORANGE   = "#f97316"

INTERPRETATION_PROMPT_TEMPLATE = (
    "Analyze this image.\n\n"
    "Step 1 — Identify 3–12 distinct physical objects and note where each one sits.\n\n"
    "Step 2 — Invent ONE coherent scene that reinterprets ALL objects together.\n"
    "• Objects do NOT move — each stays at its exact position in the image\n"
    "• The scene must be IRREDUCIBLY SPECIFIC — it could only be invoked by THIS exact\n"
    "  combination of objects in THIS exact spatial arrangement, nothing else\n"
    "• Use the actual shapes, colors, sizes, textures and positions as your raw material\n"
    "• If you could swap one object for another and the scene still works, it is too generic\n"
    "• If you could rearrange the objects and the scene still works, it is too generic\n"
    "• The best interpretation is one that feels locked — remove any single object\n"
    "  and the whole reading collapses\n"
    "• The scene must be SIMPLY EXPLAINABLE — a child should be able to understand it\n"
    "  in one sentence. Complexity of feeling is fine; complexity of concept is not\n\n"
    "Step 3 — Assign each object a concrete visual role in that scene.\n\n"
    "Step 4 — Write DRAWING INSTRUCTIONS for a background illustrator.\n"
    "These instructions must be ENTIRELY SPECIFIC TO THIS SCENE AND THESE OBJECTS.\n"
    "Generic instructions (e.g. 'draw a dark sky', 'add shadows') are forbidden.\n"
    "Every element you describe must:\n"
    "  a) directly express what a specific object has BECOME in the scene\n"
    "  b) show the relationship or tension BETWEEN specific objects\n"
    "  c) make the viewer read the dark interpretation without being told it\n\n"
    "For each object, describe what the illustrator should draw AROUND or CONNECTING it\n"
    "to make its role in the scene unmistakable. Reference exact image positions\n"
    "(top-left, between X and Y, below Z, etc.). No generic atmosphere — only elements\n"
    "that could ONLY belong to this specific interpretation of these specific objects.\n\n"
    "OUTPUT STRICT FORMAT (no extra text):\n\n"
    "SCENE: <one sentence — the specific interpretation that only these objects could produce>\n"
    "SHORT: <the same idea in 7 words or fewer — plain, simple, concrete>\n\n"
    "OBJECT: <name> | BOX: [y_min, x_min, y_max, x_max] | ROLE: <concrete visual role>\n"
    "...\n\n"
    "DRAWING INSTRUCTIONS:\n"
    "<object-by-object directives — what to draw around each one to reveal its role>\n"
)

SCENE_DRAW_PROMPT_TEMPLATE = (
    "You are an illustrator expanding a photo into a scene.\n\n"
    "SCENE: {scene}\n\n"
    "The photo shows physical objects. Each object has become something else in the scene above.\n"
    "Your job: draw directly onto the photo to expand the world around each object so the "
    "viewer immediately understands what that object has BECOME.\n\n"
    "How to expand:\n"
    "• Draw the environment, context, and extensions that GROW OUT OF each object\n"
    "• If an object is a character's body, draw the head, limbs, expression around it\n"
    "• If an object is a landscape feature, draw the horizon, sky, or ground that belongs to it\n"
    "• If an object is a machine part, draw the rest of the machine connecting to it\n"
    "• Extend lines FROM the object outward — make the object the seed of a larger drawing\n"
    "• Connect objects visually where the scene says they interact\n\n"
    "DRAWING INSTRUCTIONS (what each object has become and what to draw around it):\n"
    "{instructions}\n\n"
    "STYLE: bold black lines, expressive weight variation, no color fills — "
    "sketch-like but confident. The drawn extensions should feel like they belong "
    "to the same world as the objects, not like labels floating above them.\n"
    "Make the scene readable at a glance — someone should look at the result and "
    "immediately see the scene, not just the original objects.\n"
)


class PipelineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Pipeline")
        self.configure(bg=BG)
        self.resizable(False, False)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            tk.messagebox.showerror("Missing API Key", "Set GEMINI_API_KEY in .env")
            self.destroy()
            return

        self.client        = genai.Client(api_key=api_key)
        self.running       = True
        self.current_frame = None
        self._frame_q      = queue.Queue(maxsize=2)
        self._ai_q         = queue.Queue()
        self._ai_busy      = False
        self._ai_stage     = ""

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_camera()
        self._poll_status()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        panels = tk.Frame(self, bg=BG)
        panels.pack(padx=20, pady=(20, 12))

        lf = tk.Frame(panels, bg=BG)
        lf.pack(side="left")
        tk.Label(lf, text="Live Feed", font=("Helvetica", 11, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(pady=(0, 6))
        live_box = tk.Frame(lf, width=VIDEO_W, height=VIDEO_H, bg="#000000")
        live_box.pack()
        live_box.pack_propagate(False)
        self.live_lbl = tk.Label(live_box, bg="#000000")
        self.live_lbl.pack(fill="both", expand=True)

        tk.Frame(panels, bg="#2a2a3e", width=2).pack(side="left", fill="y", padx=18)

        rf = tk.Frame(panels, bg=BG)
        rf.pack(side="left")
        tk.Label(rf, text="AI's Take", font=("Helvetica", 11, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(pady=(0, 6))
        ai_box = tk.Frame(rf, width=VIDEO_W, height=VIDEO_H, bg=BG_CARD)
        ai_box.pack()
        ai_box.pack_propagate(False)
        self.ai_lbl = tk.Label(ai_box, bg=BG_CARD,
                               text="Press  Snap!  to analyse",
                               font=("Helvetica", 12), fg=TEXT_DIM,
                               wraplength=VIDEO_W - 20)
        self.ai_lbl.pack(fill="both", expand=True)

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=20, pady=(0, 8))

        self.snap_btn = tk.Button(bottom, text="  Snap!  ",
                                  font=("Helvetica", 13, "bold"),
                                  bg=ORANGE, fg="white", relief="flat",
                                  command=self._fire_ai)
        self.snap_btn.pack(side="left")

        self.status_lbl = tk.Label(bottom, text="", font=("Helvetica", 10),
                                   bg=BG, fg=TEXT_DIM, anchor="w")
        self.status_lbl.pack(side="left", padx=(14, 0))

        self.scene_lbl = tk.Label(self, text="", font=("Helvetica", 11, "bold"),
                                  bg=BG, fg=ORANGE, anchor="center",
                                  wraplength=VIDEO_W * 2 + 56, justify="center")
        self.scene_lbl.pack(fill="x", padx=20, pady=(0, 12))

    # ── camera ────────────────────────────────────────────────────────────────

    def _start_camera(self):
        threading.Thread(target=self._camera_worker, daemon=True).start()
        self._refresh_live()

    def _camera_worker(self):
        import traceback as _tb
        try:
            self.__camera_worker_inner()
        except Exception:
            _tb.print_exc()

    def __camera_worker_inner(self):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame
                if not self._frame_q.full():
                    self._frame_q.put(frame)
        cap.release()

    def _refresh_live(self):
        if not self.running:
            return
        try:
            frame = self._frame_q.get_nowait()
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H)))
            self.live_lbl.config(image=photo)
            self.live_lbl.image = photo  # type: ignore[attr-defined]
        except queue.Empty:
            pass
        try:
            payload = self._ai_q.get_nowait()
            self._handle_ai_result(payload)
        except queue.Empty:
            pass
        self.after(30, self._refresh_live)

    def _handle_ai_result(self, payload):
        kind = payload.get("kind")
        if kind == "error":
            self._ai_busy  = False
            self._ai_stage = ""
            self.status_lbl.config(text=f"Error: {payload['msg'][:80]}", fg="#ef4444")
            self.snap_btn.config(state="normal")
        elif kind == "image":
            img_bytes = payload["data"]
            caption   = payload.get("caption", "")
            img   = Image.open(io.BytesIO(img_bytes))
            img   = img.resize((VIDEO_W, VIDEO_H), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.ai_lbl.config(image=photo, text="")
            self.ai_lbl.image = photo  # type: ignore[attr-defined]
            self._ai_busy  = False
            self._ai_stage = ""
            self.scene_lbl.config(text=caption if caption else "")
            self.status_lbl.config(text="Done — press Snap! for a new read", fg=TEXT_DIM)
            self.snap_btn.config(state="normal")

    # ── status ticker ─────────────────────────────────────────────────────────

    def _poll_status(self):
        if not self.running:
            return
        if self._ai_busy:
            labels = {
                "interpreting": "Finding hidden scene…",
                "drawing":      "Drawing the scene…",
            }
            self.status_lbl.config(
                text=labels.get(self._ai_stage, "Working…"), fg=ORANGE)
        self.after(500, self._poll_status)

    # ── predict ───────────────────────────────────────────────────────────────

    def _fire_ai(self):
        if self.current_frame is None or self._ai_busy:
            return
        self._ai_busy  = True
        self._ai_stage = "interpreting"
        self.snap_btn.config(state="disabled")
        self.status_lbl.config(text="Finding hidden scene…", fg=ORANGE)
        self.scene_lbl.config(text="")
        frame = self.current_frame.copy()
        threading.Thread(target=self._ai_worker, args=(frame,), daemon=True).start()

    def _ai_worker(self, frame):
        try:
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
            jpeg = buf.getvalue()
            final_bytes, label = self._run_predict(jpeg)
            self._save_result(jpeg, final_bytes, label)
            self._ai_q.put({"kind": "image", "data": final_bytes, "caption": label})
        except Exception as exc:
            print(f"[AI error] {exc}")
            self._ai_q.put({"kind": "error", "msg": str(exc)})

    def _save_result(self, original: bytes, prediction: bytes, caption: str):
        import datetime
        saves_dir = os.path.join(os.path.dirname(__file__), "saves")
        os.makedirs(saves_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(saves_dir, f"{ts}_original.jpg"), "wb") as f:
            f.write(original)
        with open(os.path.join(saves_dir, f"{ts}_prediction.jpg"), "wb") as f:
            f.write(prediction)
        if caption:
            with open(os.path.join(saves_dir, f"{ts}_scene.txt"), "w", encoding="utf-8") as f:
                f.write(caption)
        print(f"[Saved] saves/{ts}_*")

    def _run_predict(self, jpeg: bytes) -> tuple[bytes, str]:
        self._ai_stage = "interpreting"
        r2 = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                types.Part.from_text(text=INTERPRETATION_PROMPT_TEMPLATE),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)),  # type: ignore[call-arg]
        )
        interp       = r2.text or ""
        scene        = self._parse_scene(interp)
        short        = self._parse_short(interp)
        objects      = self._parse_objects_with_roles(interp)
        instructions = self._parse_instructions(interp)
        print(f"\n── Scene: {scene}  |  Short: {short}")
        for o in objects:
            print(f"  {o['name']} → {o['role']}")

        if not objects:
            return jpeg, scene or short

        self._ai_stage = "drawing"
        prompt = SCENE_DRAW_PROMPT_TEMPLATE.format(scene=scene, instructions=instructions)
        if DRAW_BACKEND == "replicate":
            result = self._draw_replicate(jpeg, scene, instructions) or jpeg
        else:
            result = self._draw_gemini(jpeg, prompt) or jpeg

        return result, short or scene

    # ── drawing backends ─────────────────────────────────────────────────────

    def _draw_gemini(self, jpeg: bytes, prompt: str) -> bytes | None:
        r = self.client.models.generate_content(
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

    def _draw_replicate(self, jpeg: bytes, scene: str, instructions: str) -> bytes | None:
        try:
            import replicate, base64, urllib.request
            b64 = base64.b64encode(jpeg).decode()
            prompt = f"{scene}. {instructions[:400]}"
            output = replicate.run(
                "black-forest-labs/flux-dev",
                input={
                    "image":               f"data:image/jpeg;base64,{b64}",
                    "prompt":              prompt,
                    "prompt_strength":     0.5,
                    "num_inference_steps": 28,
                    "guidance":            3.5,
                    "output_format":       "jpg",
                },
            )
            items = list(output) if hasattr(output, "__iter__") else [output]
            for item in items:
                if hasattr(item, "read"):
                    return item.read()  # type: ignore[union-attr]
                url = str(item)
                with urllib.request.urlopen(url) as resp:
                    return resp.read()
        except Exception as exc:
            print(f"[Replicate error] {exc}")
            return None

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_objects_with_roles(self, text: str) -> list[dict]:
        objects = []
        pat = re.compile(
            r'OBJECT:\s*(?P<name>[^|]+?)\s*\|\s*BOX:\s*\[(?P<coords>[^\]]+)\]\s*\|\s*ROLE:\s*(?P<role>.+)',
            re.IGNORECASE)
        for m in pat.finditer(text):
            try:
                coords = [int(x.strip()) for x in m.group("coords").split(",")]
                if len(coords) == 4:
                    objects.append({"name": m.group("name").strip(),
                                    "box":  coords,
                                    "role": m.group("role").strip()})
            except ValueError:
                continue
        return objects

    def _parse_scene(self, text: str) -> str:
        m = re.search(r'SCENE:\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _parse_short(self, text: str) -> str:
        m = re.search(r'SHORT:\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _parse_instructions(self, text: str) -> str:
        m = re.search(r'DRAWING INSTRUCTIONS:\s*\n([\s\S]+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # ── close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self.running = False
        self.destroy()


if __name__ == "__main__":
    PipelineApp().mainloop()
