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

INTERPRET_PROMPT = (
    "Analyze the physical shapes of the objects on the table to identify a hidden, concrete scene or entity.\n\n"
    "1. DISCOVERY: Treat the arrangement as a complete system where every object contributes to the identity of a hidden scene, character, or object. You must define this entity based on the existing spatial relationships.\n"
    "2. REVELATION: Suggest up to 5 black-line additions that connect, complete, or define the scene. These additions must be independent marks that do not trace or outline the objects themselves.\n"
    "3. GROUNDING: The revealed scene must be visually coherent and physically plausible based on the shapes provided.\n\n"
    "JSON REQUIREMENTS:\n"
    "- 'scene': A 5-word description of the discovered entity or scene.\n"
    "- 'additions': An array of objects, each containing:\n"
    "    - 'feature': A description of the visual mark to be drawn.\n"
    "    - 'placement': The precise location of the mark in relation to the objects.\n\n"
    "Return ONLY valid JSON."
)
DRAW_PROMPT = (
    "Task: Overlay black-line drawings onto the photo to reveal the scene or entity identified: {scene}\n\n"
    "ADDITIONS TO DRAW:\n"
    "{anchors}\n\n"
    "STRICT RULES:\n"
    "1. NO TRACING: Do not outline, trace, or re-draw the existing objects. The drawn marks must exist solely in the negative space or between objects to suggest the form.\n"
    "2. COMPLETION: The lines should act as the 'missing' structural parts that bridge the objects together to manifest the scene.\n"
    "3. STYLE: Use clean, medium-weight, black ink-style lines. No shading, no fills, no color, no decorative flourishes.\n"
    "4. INTEGRATION: The lines must respect the camera perspective and scale of the objects, appearing as if they occupy the same 3D space as the items on the table."
)
RED_DIM  = "#374151"
WHITE    = "#e2e8f0"



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
        self.status_lbl.config(text="ready for the next one", fg=TEXT_DIM)

    def _clear_pending(self):
        self._pending_orig    = None
        self._pending_pred    = None
        self._pending_caption = ""

    # ── status ticker ─────────────────────────────────────────────────────────

    def _poll_status(self):
        if not self.running:
            return
        if self._ai_busy:
            text = {"interpreting": "reading shapes…", "drawing": "drawing scene…"}.get(
                self._ai_stage, "finding hidden scene…")
            self.status_lbl.config(text=text, fg=ORANGE)
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
        self.after(0, lambda: self.status_lbl.config(
            text="saved — ready for the next one", fg=GREEN))

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

    def _to_silhouette(self, jpeg: bytes) -> bytes:
        from rfdetr import RFDETRSegNano
        from PIL import Image as PILImage
        if not hasattr(self, "_seg"):
            self._seg = RFDETRSegNano()

        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image for sketch preprocessing")
        h, w = img.shape[:2]

        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Layer 1: mean-shift + dilated Canny — smooth, thick coverage of all structure
        shifted = cv2.pyrMeanShiftFiltering(img, sp=6, sr=25)
        gray    = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 15, 45)
        thick   = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        for cnt in cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
            if cv2.contourArea(cnt) < 150:
                continue
            cv2.drawContours(canvas, [cnt], -1, (0, 0, 0), 2)

        # Layer 2: RF-DETR Seg — blobby silhouette outlines for recognised objects
        pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        _det = self._seg.predict(pil, threshold=0.2)
        det  = _det[0] if isinstance(_det, list) else _det
        if det.mask is not None and len(det) > 0:  # type: ignore[union-attr]
            for mask in det.mask:  # type: ignore[union-attr]
                m = mask.astype(np.uint8) * 255
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = cv2.dilate(m, np.ones((9, 9), np.uint8), iterations=1)
                for cnt in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                    if cv2.contourArea(cnt) < 300:
                        continue
                    cv2.drawContours(canvas, [cnt], -1, (0, 0, 0), 7)

        # Soften to round any remaining jagged corners
        canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        _, canvas = cv2.threshold(canvas, 200, 255, cv2.THRESH_BINARY)

        _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return bytes(buf)

    def _to_orange_sketch(self, sketch: bytes) -> bytes:
        """Recolor black-on-white sketch to orange-on-white for the AI."""
        img = cv2.imdecode(np.frombuffer(sketch, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return sketch
        h, w  = img.shape
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        canvas[mask > 0] = [22, 115, 249]  # orange #f97316 in BGR
        _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return bytes(buf)

    def _gemini_call(self, **kwargs):
        import time
        for attempt in range(4):
            try:
                return self.client.models.generate_content(**kwargs)
            except Exception as e:
                msg = str(e)
                if attempt < 3 and any(c in msg for c in ("503", "502", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    wait = 2 ** attempt
                    print(f"[Retry {attempt+1}] {msg[:60]} — waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise

    def _run_predict(self, jpeg: bytes) -> tuple[bytes, str]:
        import time
        self._ai_stage = "interpreting"
        t0 = time.time()
        sketch = self._to_silhouette(jpeg)
        print(f"\n── Sketch: {time.time()-t0:.1f}s")

        with open(os.path.join(APP_DIR, "_debug_sketch.jpg"), "wb") as f:
            f.write(sketch)

        # Step 1: interpret from sketch only
        t1 = time.time()
        r1 = self._gemini_call(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=sketch, mime_type="image/jpeg"),
                types.Part.from_text(text=INTERPRET_PROMPT),
            ],
            config=types.GenerateContentConfig(temperature=1.0),  # type: ignore[call-arg]
        )
        print(f"── Interpret: {time.time()-t1:.1f}s")
        assert r1 is not None
        raw   = r1.text or ""
        scene, anchors_text = self._parse_json_plan(raw)
        print(f"\n── Interpretation:\n{raw}")

        # Step 2: draw on the real photo
        self._ai_stage = "drawing"
        t2 = time.time()
        r2 = self._gemini_call(
            model="gemini-2.5-flash-image",
            contents=[
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                types.Part.from_text(text=DRAW_PROMPT.format(scene=scene, anchors=anchors_text)),
            ],
            config=types.GenerateContentConfig(  # type: ignore[call-arg]
                response_modalities=["TEXT", "IMAGE"],
                temperature=1.0),  # type: ignore[call-arg]
        )
        print(f"── Draw: {time.time()-t2:.1f}s  |  Total: {time.time()-t0:.1f}s")
        assert r2 is not None
        cands  = r2.candidates or []
        cparts = cands[0].content.parts if cands and cands[0].content else []  # type: ignore[union-attr]
        print(f"\n── Draw parts: {[type(p).__name__ for p in (cparts or [])]}")

        result_img = None
        for part in (cparts or []):
            if hasattr(part, "text") and part.text:
                print(f"── Draw text: {part.text}")
            idata = getattr(part, "inline_data", None)
            if idata and getattr(idata, "data", None):
                result_img = bytes(idata.data)  # type: ignore[arg-type]
                print(f"── Image: {len(result_img)} bytes")

        if result_img is None:
            raise RuntimeError("Model returned no image — check console for response details")

        with open(os.path.join(APP_DIR, "_debug_final.jpg"), "wb") as f:
            f.write(result_img)

        return result_img, scene

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
            config=types.GenerateContentConfig(  # type: ignore[call-arg]
                response_modalities=["TEXT", "IMAGE"],
                temperature=1.0),  # type: ignore[call-arg]
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

    def _parse_json_plan(self, text: str) -> tuple[str, str]:
        import json
        try:
            json_str = text.strip()
            m = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', json_str)
            if m:
                json_str = m.group(1)
            data = json.loads(json_str)
            scene = data.get("scene", "")
            additions = data.get("additions", [])
            lines = [
                f"- Draw {a.get('feature', '')} {a.get('placement', '')}".strip()
                for a in additions
                if a.get('feature', '').strip()
            ]
            return scene, "\n".join(lines)
        except Exception:
            return self._parse_scene(text), text

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
