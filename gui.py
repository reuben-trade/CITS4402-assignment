import shutil
import threading
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from feature_extraction import Detection, FaceProcessor

# BGR colours per spec: right_eye=red, left_eye=green, nose=blue
COLOUR_RIGHT_EYE = (0, 0, 255)
COLOUR_LEFT_EYE = (0, 255, 0)
COLOUR_NOSE = (255, 0, 0)
COLOUR_BBOX = (0, 255, 0)

# Canonical target landmark positions (must match _align_to_125 in feature_extraction.py)
CANONICAL_LEFT_EYE = (40, 40)
CANONICAL_RIGHT_EYE = (85, 40)
CANONICAL_NOSE = (63, 70)

CROP_SIZE = 125
DISPLAY_MAX_W = 640
DISPLAY_MAX_H = 480

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def fit_within(img_bgr: np.ndarray, max_w: int = DISPLAY_MAX_W, max_h: int = DISPLAY_MAX_H) -> np.ndarray:
    """Downscale to fit within (max_w, max_h) keeping aspect ratio. Never upscales.

    Upscaling small images broke detection on obama-face.jpeg (275x183 → 640x426).
    Tried letterboxing to a 640x480 canvas too — that drops the face below
    MediaPipe's effective minimum size on small inputs. So we keep native res.
    """
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if abs(scale - 1.0) < 1e-3:
        return img_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def composite_on_canvas(
    img_bgr: np.ndarray,
    detections: list[Detection],
    canvas_w: int = DISPLAY_MAX_W,
    canvas_h: int = DISPLAY_MAX_H,
) -> np.ndarray:
    """Paste the image onto a canvas_w x canvas_h canvas and
    draw bboxes/landmarks on it; place 125x125 corner crops at the CANVAS
    corners so they never obscure the face. Detections are in img_bgr coords;
    they get shifted by the centering offset.
    """
    h, w = img_bgr.shape[:2]
    off_x = max(0, (canvas_w - w) // 2)
    off_y = max(0, (canvas_h - h) // 2)
    out_w = max(canvas_w, w)
    out_h = max(canvas_h, h)
    canvas = np.full((out_h, out_w, 3), 40, dtype=np.uint8)
    canvas[off_y:off_y + h, off_x:off_x + w] = img_bgr

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(
            canvas,
            (x1 + off_x, y1 + off_y),
            (x2 + off_x, y2 + off_y),
            COLOUR_BBOX,
            2,
        )
        shifted = {k: (v[0] + off_x, v[1] + off_y) for k, v in det.landmarks.items()}
        _draw_landmarks(canvas, shifted)

    corner_slots = [
        (0, 0),
        (max(0, out_w - CROP_SIZE), 0),
        (0, max(0, out_h - CROP_SIZE)),
        (max(0, out_w - CROP_SIZE), max(0, out_h - CROP_SIZE)),
    ]
    for det, (cx, cy) in zip(detections[:4], corner_slots):
        crop = det.aligned_crop.copy()
        canonical = {
            "left_eye": CANONICAL_LEFT_EYE,
            "right_eye": CANONICAL_RIGHT_EYE,
            "nose": CANONICAL_NOSE,
        }
        _draw_landmarks(crop, canonical)
        slot_h = min(CROP_SIZE, out_h - cy)
        slot_w = min(CROP_SIZE, out_w - cx)
        canvas[cy:cy + slot_h, cx:cx + slot_w] = crop[:slot_h, :slot_w]

    return canvas


def _draw_landmarks(img: np.ndarray, lm: dict[str, tuple[int, int]], radius: int = 3) -> None:
    cv2.circle(img, lm["right_eye"], radius, COLOUR_RIGHT_EYE, -1)
    cv2.circle(img, lm["left_eye"], radius, COLOUR_LEFT_EYE, -1)
    cv2.circle(img, lm["nose"], radius, COLOUR_NOSE, -1)


def annotate_output(img_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bboxes + landmark circles on the original and overlay aligned crops in the corners."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOUR_BBOX, 2)
        _draw_landmarks(out, det.landmarks)

    corner_slots = [
        (0, 0),
        (max(0, w - CROP_SIZE), 0),
        (0, max(0, h - CROP_SIZE)),
        (max(0, w - CROP_SIZE), max(0, h - CROP_SIZE)),
    ]
    for det, (cx, cy) in zip(detections[:4], corner_slots):
        crop = det.aligned_crop.copy()
        canonical = {
            "left_eye": CANONICAL_LEFT_EYE,
            "right_eye": CANONICAL_RIGHT_EYE,
            "nose": CANONICAL_NOSE,
        }
        _draw_landmarks(crop, canonical, radius=3)

        # Clip in case the image is smaller than CROP_SIZE in either dim.
        slot_h = min(CROP_SIZE, h - cy)
        slot_w = min(CROP_SIZE, w - cx)
        out[cy:cy + slot_h, cx:cx + slot_w] = crop[:slot_h, :slot_w]

    return out


def bgr_to_photoimage(img_bgr: np.ndarray) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class FaceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Detection & Matching — CITS4402")
        self.resizable(False, False)

        self.fp = FaceProcessor()

        # Pre-create a blank placeholder so the Labels size in PIXELS (not
        # characters/lines). Without this, tk.Label(width=640, height=480)
        # without an image is read as 640 chars wide × 480 lines tall and
        # blows the layout off-screen.
        self._placeholder = self._make_placeholder(DISPLAY_MAX_W, DISPLAY_MAX_H)

        tk.Label(self, text="Input Image").grid(row=0, column=0, pady=(8, 0))
        tk.Label(self, text="Output Image").grid(row=0, column=1, pady=(8, 0))

        self.input_label = tk.Label(self, image=self._placeholder, bg="#222", bd=0)
        self.input_label.image = self._placeholder
        self.input_label.grid(row=1, column=0, padx=8, pady=8)
        self.output_label = tk.Label(self, image=self._placeholder, bg="#222", bd=0)
        self.output_label.image = self._placeholder
        self.output_label.grid(row=1, column=1, padx=8, pady=8)

        self.status = tk.Label(
            self,
            text="Ready. Press 'Single Image' to load one image or 'Bulk Processing' to process a folder.",
            anchor="w",
            justify="left",
            wraplength=1280,
        )
        self.status.grid(row=2, column=0, columnspan=2, padx=8, sticky="we")

        button_frame = tk.Frame(self)
        button_frame.grid(row=3, column=0, columnspan=2, pady=8)

        self.single_btn = tk.Button(button_frame, text="Single Image", width=18, command=self._on_single)
        self.single_btn.pack(side="left", padx=12)
        self.bulk_btn = tk.Button(button_frame, text="Bulk Processing", width=18, command=self._on_bulk)
        self.bulk_btn.pack(side="left", padx=12)

    @staticmethod
    def _make_placeholder(w: int, h: int) -> ImageTk.PhotoImage:
        blank = Image.new("RGB", (w, h), color=(40, 40, 40))
        return ImageTk.PhotoImage(blank)

    # Display helpers

    def _show_image(self, label: tk.Label, img_bgr: np.ndarray) -> None:
        photo = bgr_to_photoimage(img_bgr)
        label.configure(image=photo)
        label.image = photo  # prevent garbage collection

    def _clear_image(self, label: tk.Label) -> None:
        label.configure(image=self._placeholder)
        label.image = self._placeholder

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.single_btn.configure(state=state)
        self.bulk_btn.configure(state=state)

    # Single image

    def _on_single(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self._run_single(path)

    def _run_single(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read image: {path}")
            return

        img = fit_within(img)
        # Show the (un-annotated) input on the left, centred on the canvas.
        self._show_image(self.input_label, composite_on_canvas(img, []))
        self._set_status("Processing…")
        self.update_idletasks()

        t0 = time.perf_counter()
        dets = self.fp.detect_one(img, do_embed=False)
        dt = time.perf_counter() - t0

        annotated = composite_on_canvas(img, dets)
        self._show_image(self.output_label, annotated)
        self._set_status(
            f"Single image processed in {dt:.2f} seconds. Found {len(dets)} face(s)."
        )

    # Bulk processing

    def _on_bulk(self) -> None:
        folder = filedialog.askdirectory(title="Select a folder of images")
        if not folder:
            return
        self._set_buttons_enabled(False)
        self._clear_image(self.input_label)
        self._clear_image(self.output_label)
        self._set_status(f"Bulk processing started: {folder}")
        self.update_idletasks()
        threading.Thread(target=self._run_bulk, args=(folder,), daemon=True).start()

    def _run_bulk(self, folder: str) -> None:
        try:
            self._do_bulk(folder)
        except Exception as exc:  # pragma: no cover - surface unexpected errors to UI
            self.after(0, lambda: self._set_status(f"Bulk processing failed: {exc}"))
        finally:
            self.after(0, lambda: self._set_buttons_enabled(True))

    def _do_bulk(self, folder: str) -> None:
        folder_path = Path(folder)
        out_dir = folder_path.parent / "Processed_Images"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        image_paths = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        n_images = len(image_paths)

        self.after(0, lambda: self._set_status(f"Bulk processing started: {n_images} images…"))

        t0 = time.perf_counter()
        all_dets: list[Detection] = []
        first_annotated: np.ndarray | None = None
        first_input: np.ndarray | None = None

        for i, p in enumerate(image_paths):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = fit_within(img)
            dets = self.fp.detect_one(img, do_embed=True)
            all_dets.extend(dets)

            if first_annotated is None and dets:
                first_input = composite_on_canvas(img, [])
                first_annotated = composite_on_canvas(img, dets)

            current = i + 1
            self.after(
                0,
                lambda c=current, total=n_images, name=p.name, k=len(dets):
                    self._set_status(f"Processed {c}/{total}: {name} ({k} face(s))"),
            )

        embeddings = [d.embedding for d in all_dets if d.embedding is not None]
        labels = self.fp.similarity_from_embeddings(embeddings) if embeddings else []

        face_idx_per_identity: dict[int, int] = defaultdict(int)
        det_iter = (d for d in all_dets if d.embedding is not None)
        for label, det in zip(labels, det_iter):
            m = face_idx_per_identity[label]
            face_idx_per_identity[label] += 1
            out_path = out_dir / f"Identity_{label}_face_{m}.jpg"
            cv2.imwrite(str(out_path), det.aligned_crop)

        dt = time.perf_counter() - t0
        n_faces = len(all_dets)
        n_identities = len(set(labels))

        summary = (
            f"Total {n_images} images processed in {dt:.2f} seconds. "
            f"{n_faces} faces detected corresponding to {n_identities} unique identities. "
            f"Crops saved to {out_dir}"
        )
        self.after(0, lambda: self._set_status(summary))

        if first_input is not None and first_annotated is not None:
            self.after(0, lambda: self._show_image(self.input_label, first_input))
            self.after(0, lambda: self._show_image(self.output_label, first_annotated))

        # Pop up the cluster grid on the main thread (matplotlib needs to be
        # called from the same thread that drives Tk's event loop).
        crops = [d.aligned_crop for d in all_dets if d.embedding is not None]
        if crops:
            self.after(0, lambda: self.fp.plot_clusters_inmem(crops, labels, block=False))


def main() -> None:
    app = FaceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
