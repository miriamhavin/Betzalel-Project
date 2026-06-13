import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
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
"You are interpreting an image composed of physical objects.\n\n"

"Your task is to find an imaginative hidden world that is ALREADY VISIBLE in the arrangement —\n"
"not invented on top of it, but latent inside it.\n"
"The test: if someone removed the drawn overlay and only saw the original photo,\n"
"they could still recognise the interpretation once told what to look for.\n"
"The scene must live in the actual shapes, positions, and relationships of the objects.\n\n"

"Step 1 — Visual perception:\n"
"Describe the image as a whole visual composition.\n"
"Then list all visible objects briefly as simple visual forms (no functions, no naming interpretation).\n\n"

"Step 2 — Interpretation:\n"
"Generate three candidate interpretations of the arrangement — each a different world.\n"
"Discard the first one: it is the obvious reading and therefore the least interesting.\n"
"Discard the second if it is merely a variation of the first.\n"
"Choose the third — the one that surprises even you, but is still fully supported\n"
"by the actual visual forms and spatial positions.\n"
"The winning interpretation makes someone say: 'I never would have thought of that,\n"
"but now I can't unsee it.' It must be latent in the shapes, not imposed on them.\n\n"

"Step 3 — Object roles, grounded in visual features:\n"
"For each object, look at its actual visual form: its specific edges, contours, silhouette, proportions.\n"
"Identify ONE specific visible feature of this object — an exact edge, curve, protrusion, flat side —\n"
"that IS the scene element. Not ‘this object looks like’, but ‘this specific part of it IS’.\n"
"Name the feature precisely (e.g. ‘its curved left edge’, ‘the flat top surface’, ‘the pointed tip’).\n"
"The role is what that feature becomes in the interpretation.\n\n"

"Step 4 — Drawing from the feature:\n"
"For each object, the drawing starts exactly from the feature you named and extends it into the scene.\n"
"The line makes visible what that feature is becoming — it completes the object’s role.\n"
"State: which edge/contour of the object the line starts from, and what it extends into.\n"
"Do not draw the object itself — only what grows from the named feature outward.\n\n"

"OUTPUT FORMAT (strict — follow exactly):\n\n"
"INTERPRETATION: <single title, max 10 words>\n\n"
"OBJECT 1: <visual form> | POSITION: <where in frame> | BOX: [x1,y1,x2,y2] | FEATURE: <exact edge/contour/shape> | ROLE: <what that feature becomes>\n"
"OBJECT 2: <visual form> | POSITION: <where in frame> | BOX: [x1,y1,x2,y2] | FEATURE: <exact edge/contour/shape> | ROLE: <what that feature becomes>\n"
"... (one line per object)\n\n"
"BOX coordinates: normalized 0.0–1.0, origin top-left, format [x1,y1,x2,y2].\n\n"
"VISUAL EXPANSION:\n"
"[OBJECT 1] FROM its <feature> → <what scene element to draw, extending in which direction>\n"
"[OBJECT 2] FROM its <feature> → <what scene element to draw, extending in which direction>\n"
"... (one line per object, same order)\n"
)

SCENE_DRAW_PROMPT_TEMPLATE = (
"You are generating ONLY an overlay layer — not a scene, not a photograph.\n\n"

"TASK: place sparse black strokes on top of the locked photograph.\n\n"

"OUTPUT CONTRACT:\n"
"• Transparent layer + sparse black strokes only\n"
"• Visual proof test: if the photograph were removed, only thin black lines would remain\n"
"• If anything else appears (fills, shading, reconstructed scene), the output is invalid\n\n"

"SCENE CONTEXT: {scene}\n\n"

"NEGATIVE CONSTRAINTS — never violate:\n"
"• no shading\n"
"• no fill\n"
"• no textures\n"
"• no color regions\n"
"• no background reconstruction\n"
"• no object redrawing\n"
"• no scene completion\n"
"• do not copy or reconstruct any part of the photograph\n\n"

"DRAWING INSTRUCTIONS — one per object, format: FROM its <feature> → <what to draw>:\n"
"{instructions}\n\n"

"Each instruction specifies an exact physical feature of the object and what grows from it.\n"
"Start your line at that feature. Extend it outward into the named scene element.\n"
"The object itself is untouched — only what extends from the named feature is drawn.\n"
"LINE STYLE: thin solid black lines only. No fill. No shading. No color. Pure overlay.\n"
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
        interp   = r2.text or ""
        scene    = self._parse_scene(interp)
        objects  = self._parse_objects_with_roles(interp)
        per_obj  = self._parse_per_object_instructions(self._parse_instructions(interp))

        print(f"\n── Scene: {scene}")
        for o in objects:
            print(f"  [{o['pos']}] {o['name']} | box: {o['box']} | feat: {o['feature']} → {o['role']}")

        if not objects:
            return jpeg, scene

        self._ai_stage = "drawing"
        orig_img  = Image.open(io.BytesIO(jpeg)).convert("RGB")
        W, H      = orig_img.size
        result_img = orig_img.copy()

        for i, obj in enumerate(objects):
            box         = obj.get("box")
            instruction = per_obj[i] if i < len(per_obj) else ""
            if not box or not instruction:
                print(f"  [skip obj {i+1}] no box or instruction")
                continue

            # Expand box by 15% padding so the drawing has context to attach to
            x1, y1, x2, y2 = box
            pad = 0.12
            cx1, cy1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
            cx2, cy2 = min(1.0, x2 + pad), min(1.0, y2 + pad)
            px1, py1, px2, py2 = int(cx1*W), int(cy1*H), int(cx2*W), int(cy2*H)
            if px2 - px1 < 16 or py2 - py1 < 16:
                continue

            crop = orig_img.crop((px1, py1, px2, py2))
            buf  = io.BytesIO()
            crop.save(buf, format="JPEG", quality=95)

            prompt = SCENE_DRAW_PROMPT_TEMPLATE.format(scene=scene, instructions=instruction)
            drawn  = self._draw_gemini(buf.getvalue(), prompt)
            if not drawn:
                continue

            drawn_img = Image.open(io.BytesIO(drawn)).convert("RGB")
            drawn_img = drawn_img.resize((px2 - px1, py2 - py1), Image.Resampling.LANCZOS)

            # Darker pixel wins → preserves black lines without touching the photo elsewhere
            orig_arr  = np.array(result_img.crop((px1, py1, px2, py2)), dtype=np.uint8)
            drawn_arr = np.array(drawn_img, dtype=np.uint8)
            merged    = np.minimum(orig_arr, drawn_arr)
            result_img.paste(Image.fromarray(merged), (px1, py1))
            print(f"  [drew obj {i+1}] {obj['name']}")

        out = io.BytesIO()
        result_img.save(out, format="JPEG", quality=95)
        return out.getvalue(), scene

    # ── drawing backends ─────────────────────────────────────────────────────

    def _draw_gemini(self, jpeg: bytes, prompt: str) -> bytes | None:
        r = self.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                types.Part.from_text(text="LOCKED IMAGE — DO NOT MODIFY, COPY, OR RECONSTRUCT THIS PHOTOGRAPH. It is fixed input only."),
                types.Part.from_text(text=prompt),
                types.Part.from_text(text="REMINDER: output ONLY sparse black line strokes as overlay. Do not reconstruct or redraw the photograph."),
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
            r'OBJECT\s*\d*:\s*(?P<name>[^|]+?)\s*\|'
            r'(?:\s*POSITION:\s*(?P<pos>[^|]+?)\s*\|)?'
            r'(?:\s*BOX:\s*\[(?P<box>[^\]]+)\]\s*\|)?'
            r'(?:\s*FEATURE:\s*(?P<feat>[^|]+?)\s*\|)?'
            r'\s*ROLE:\s*(?P<role>.+)',
            re.IGNORECASE)
        for m in pat.finditer(text):
            box_str = (m.group("box") or "").strip()
            try:
                box = [float(v) for v in box_str.split(",") if v.strip()] if box_str else None
                box = box if box and len(box) == 4 else None
            except ValueError:
                box = None
            objects.append({
                "name":    m.group("name").strip(),
                "pos":     (m.group("pos")  or "").strip(),
                "box":     box,
                "feature": (m.group("feat") or "").strip(),
                "role":    m.group("role").strip(),
            })
        return objects

    def _parse_scene(self, text: str) -> str:
        m = re.search(r'(?:INTERPRETATION|SCENE):\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _parse_instructions(self, text: str) -> str:
        m = re.search(r'(?:VISUAL EXPANSION|DRAWING INSTRUCTIONS):\s*\n([\s\S]+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _parse_per_object_instructions(self, expansion: str) -> list[str]:
        results: dict[int, str] = {}
        pat = re.compile(r'\[OBJECT\s*(\d+)\]\s*(.+?)(?=\n\s*\[OBJECT|\Z)', re.IGNORECASE | re.DOTALL)
        for m in pat.finditer(expansion):
            idx = int(m.group(1)) - 1
            results[idx] = m.group(2).strip()
        if not results:
            return []
        return [results.get(i, "") for i in range(max(results) + 1)]

    # ── close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self.running = False
        self.destroy()


if __name__ == "__main__":
    PipelineApp().mainloop()
