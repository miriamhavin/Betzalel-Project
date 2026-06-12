import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import queue
import io
import re
import os
import subprocess
import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

CAM_W, CAM_H = 640, 480
DRAW_BACKEND  = os.getenv("DRAW_BACKEND", "gemini")
APP_DIR       = os.path.dirname(os.path.abspath(__file__))

BG       = "#0a0a14"
BG_CARD  = "#0f0f1e"
TEXT_DIM = "#2e3a4e"
TEXT_MED = "#64748b"
ORANGE   = "#f97316"
GREEN    = "#22c55e"
RED_DIM  = "#374151"
WHITE    = "#e2e8f0"

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
    "Step 4 — Write TRANSFORMATION INSTRUCTIONS for an image editor.\n"
    "These are NOT instructions to draw on top of the photo — they describe how to\n"
    "VISUALLY TRANSFORM each object so it fully becomes its role in the scene.\n"
    "The final image must look like a scene illustration, NOT a photo with sketches added.\n\n"
    "For each object describe THREE things:\n"
    "  1. OBJECT TRANSFORM: how to change its appearance (texture, color, shape, material)\n"
    "     so it looks like what it has become — not like the original everyday object\n"
    "  2. ENVIRONMENT: what background/setting/atmosphere should replace or surround it\n"
    "     to place it inside the scene world\n"
    "  3. LIGHTING & MOOD: what light direction, color grade, or shadow belongs to its role\n\n"
    "Rules:\n"
    "• Every instruction must be impossible to apply to a different scene — lock it to THIS\n"
    "  specific interpretation of THESE specific objects\n"
    "• No object should still look like its original everyday self in the final image\n"
    "• Reference exact image positions (top-left, between X and Y, below Z, etc.)\n\n"
    "OUTPUT STRICT FORMAT (no extra text):\n\n"
    "SCENE: <one sentence — the specific interpretation that only these objects could produce>\n"
    "SHORT: <the same idea in 7 words or fewer — plain, simple, concrete>\n\n"
    "OBJECT: <name> | BOX: [y_min, x_min, y_max, x_max] | ROLE: <concrete visual role>\n"
    "...\n\n"
    "DRAWING INSTRUCTIONS:\n"
    "<object-by-object transformation directives>\n"
)

SCENE_DRAW_PROMPT_TEMPLATE = (
    "Transform this photo into a vivid illustration of the following scene:\n\n"
    "SCENE: {scene}\n\n"
    "This is NOT about adding drawings on top of the photo. You must TRANSFORM the photo "
    "so that what you see IS the scene — not a photo of objects with art overlaid on it.\n\n"
    "What transformation means:\n"
    "• Each object must visually BECOME what it represents — change its texture, color,\n"
    "  material, and shape so it no longer looks like an everyday object\n"
    "• Replace the background entirely with the environment of the scene\n"
    "• Apply dramatic lighting and color grading that belongs to the scene's world\n"
    "• Objects that interact in the scene must visually connect — shared light, shared texture,\n"
    "  visual lines that flow between them\n"
    "• The viewer must read the scene immediately without recognising the original objects\n\n"
    "TRANSFORMATION INSTRUCTIONS (how each object changes and what environment surrounds it):\n"
    "{instructions}\n\n"
    "STYLE: painterly illustration with strong atmosphere — rich color, dramatic contrast, "
    "a clear sense of place and mood. The result should look like concept art or an illustrated "
    "storybook page, not a photograph. No object should still look like its original everyday self.\n"
)


class PipelineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Pipeline")
        self.configure(bg=BG)

        self.state("zoomed")
        self.resizable(True, True)
        self.update()
        sw = self.winfo_width()

        self._lw = max(300, int(sw * 0.22))
        self._lh = int(self._lw * 0.75)
        self._pw = max(480, int(sw * 0.38))
        self._ph = int(self._pw * 0.75)

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

        # pending result waiting for save/discard decision
        self._pending_orig    = None
        self._pending_pred    = None
        self._pending_caption = ""

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_camera()
        self._poll_status()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        wrap = tk.Frame(self, bg=BG)
        wrap.place(relx=0.5, rely=0.48, anchor="center")

        # panels row
        row = tk.Frame(wrap, bg=BG)
        row.pack()

        # Live feed
        lf = tk.Frame(row, bg=BG)
        lf.pack(side="left", anchor="n")
        tk.Label(lf, text="live", font=("Helvetica", 9),
                 bg=BG, fg=TEXT_DIM).pack(pady=(0, 5))
        live_box = tk.Frame(lf, width=self._lw, height=self._lh, bg="#000000")
        live_box.pack()
        live_box.pack_propagate(False)
        self.live_lbl = tk.Label(live_box, bg="#000000")
        self.live_lbl.pack(fill="both", expand=True)

        # Divider
        tk.Frame(row, bg=TEXT_DIM, width=1).pack(side="left", fill="y", padx=32)

        # Prediction panel + caption
        rf = tk.Frame(row, bg=BG)
        rf.pack(side="left", anchor="n")
        tk.Label(rf, text="prediction", font=("Helvetica", 9),
                 bg=BG, fg=TEXT_DIM).pack(pady=(0, 5))
        ai_box = tk.Frame(rf, width=self._pw, height=self._ph, bg=BG_CARD)
        ai_box.pack()
        ai_box.pack_propagate(False)
        self.ai_lbl = tk.Label(ai_box, bg=BG_CARD,
                               text="press  snap  to begin",
                               font=("Helvetica", 12), fg=TEXT_DIM,
                               wraplength=self._pw - 30)
        self.ai_lbl.pack(fill="both", expand=True)

        self.caption_lbl = tk.Label(rf, text="",
                                    font=("Georgia", 13, "italic"),
                                    bg=BG, fg=WHITE,
                                    wraplength=self._pw,
                                    justify="center")
        self.caption_lbl.pack(pady=(18, 0))

        # ── bottom bar: snap button + status ─────────────────────────────────
        self._snap_bar = tk.Frame(wrap, bg=BG)
        self._snap_bar.pack(pady=(32, 0))

        self.snap_btn = tk.Button(self._snap_bar, text="snap",
                                  font=("Helvetica", 14, "bold"),
                                  bg=ORANGE, fg=WHITE, relief="flat",
                                  padx=36, pady=10,
                                  cursor="hand2",
                                  command=self._fire_ai)
        self.snap_btn.pack(side="left")

        self.status_lbl = tk.Label(self._snap_bar, text="",
                                   font=("Helvetica", 10),
                                   bg=BG, fg=TEXT_MED, anchor="w")
        self.status_lbl.pack(side="left", padx=(20, 0))

        # ── save / discard row (hidden until result arrives) ──────────────────
        self._sd_bar = tk.Frame(wrap, bg=BG)
        # not packed yet — shown only after a result

        self.save_btn = tk.Button(self._sd_bar, text="save",
                                  font=("Helvetica", 14, "bold"),
                                  bg=GREEN, fg="#0a0a14", relief="flat",
                                  padx=36, pady=10, cursor="hand2",
                                  command=self._on_save)
        self.save_btn.pack(side="left", padx=(0, 12))

        self.discard_btn = tk.Button(self._sd_bar, text="discard",
                                     font=("Helvetica", 14),
                                     bg=RED_DIM, fg=WHITE, relief="flat",
                                     padx=36, pady=10, cursor="hand2",
                                     command=self._on_discard)
        self.discard_btn.pack(side="left")

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
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
            photo = ImageTk.PhotoImage(
                Image.fromarray(rgb).resize((self._lw, self._lh)))
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
            self.status_lbl.config(text=f"error: {payload['msg'][:80]}", fg="#ef4444")
            self._show_snap_bar()
        elif kind == "image":
            # display result
            img   = Image.open(io.BytesIO(payload["data"]))
            img   = img.resize((self._pw, self._ph), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.ai_lbl.config(image=photo, text="")
            self.ai_lbl.image = photo  # type: ignore[attr-defined]
            caption = payload.get("caption", "")
            self.caption_lbl.config(text=f"[ {caption} ]" if caption else "")

            # stash pending data, swap bars
            self._pending_orig    = payload["orig"]
            self._pending_pred    = payload["data"]
            self._pending_caption = caption

            self._ai_busy  = False
            self._ai_stage = ""
            self.status_lbl.config(text="", fg=TEXT_MED)
            self._show_sd_bar()

    # ── bar helpers ───────────────────────────────────────────────────────────

    def _show_snap_bar(self):
        self._sd_bar.pack_forget()
        self._snap_bar.pack(pady=(32, 0))
        self.snap_btn.config(state="normal")

    def _show_sd_bar(self):
        self._snap_bar.pack_forget()
        self._sd_bar.pack(pady=(32, 0))

    # ── save / discard actions ────────────────────────────────────────────────

    def _on_save(self):
        orig, pred, cap = self._pending_orig, self._pending_pred, self._pending_caption
        self._clear_pending()
        self._show_snap_bar()
        self.status_lbl.config(text="saving…", fg=TEXT_MED)
        threading.Thread(target=self._save_and_push,
                         args=(orig, pred, cap), daemon=True).start()

    def _on_discard(self):
        self._clear_pending()
        self._show_snap_bar()
        self.status_lbl.config(text="discarded", fg=TEXT_DIM)

    def _clear_pending(self):
        self._pending_orig    = None
        self._pending_pred    = None
        self._pending_caption = ""

    # ── status ticker ─────────────────────────────────────────────────────────

    def _poll_status(self):
        if not self.running:
            return
        if self._ai_busy:
            labels = {
                "interpreting": "finding hidden scene…",
                "drawing":      "drawing the scene…",
            }
            self.status_lbl.config(
                text=labels.get(self._ai_stage, "working…"), fg=ORANGE)
        self.after(500, self._poll_status)

    # ── predict ───────────────────────────────────────────────────────────────

    def _fire_ai(self):
        if self.current_frame is None or self._ai_busy:
            return
        self._ai_busy  = True
        self._ai_stage = "interpreting"
        self.snap_btn.config(state="disabled")
        self.caption_lbl.config(text="")
        self.status_lbl.config(text="finding hidden scene…", fg=ORANGE)
        frame = self.current_frame.copy()
        threading.Thread(target=self._ai_worker, args=(frame,), daemon=True).start()

    def _ai_worker(self, frame):
        try:
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
            jpeg = buf.getvalue()
            final_bytes, label = self._run_predict(jpeg)
            # result goes to queue WITHOUT saving — user decides
            self._ai_q.put({
                "kind":    "image",
                "data":    final_bytes,
                "orig":    jpeg,
                "caption": label,
            })
        except Exception as exc:
            print(f"[AI error] {exc}")
            self._ai_q.put({"kind": "error", "msg": str(exc)})

    # ── save + git push ───────────────────────────────────────────────────────

    def _save_and_push(self, original: bytes, prediction: bytes, caption: str):
        saves_dir = os.path.join(APP_DIR, "saves")
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
        self._git_push(ts)

    def _git_push(self, ts: str):
        try:
            subprocess.run(["git", "add", "saves/"],
                           cwd=APP_DIR, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"snap {ts}"],
                           cwd=APP_DIR, check=True, capture_output=True)
            subprocess.run(["git", "push"],
                           cwd=APP_DIR, check=True, capture_output=True)
            print(f"[Git] pushed snap {ts}")
        except subprocess.CalledProcessError as e:
            print(f"[Git push failed] {e.stderr.decode()[:200]}")

    # ── predict pipeline ──────────────────────────────────────────────────────

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
            output = replicate.run(
                "black-forest-labs/flux-dev",
                input={
                    "image":               f"data:image/jpeg;base64,{b64}",
                    "prompt":              f"{scene}. {instructions[:400]}",
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
                with urllib.request.urlopen(str(item)) as resp:
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
