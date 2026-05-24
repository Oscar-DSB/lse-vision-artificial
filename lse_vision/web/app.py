from __future__ import annotations

"""FastAPI web application for LSE sign recognition.

Endpoints:
  GET  /              → main page
  POST /api/predict   → accept image or video, return letter + annotated image
"""

import base64
import io
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from lse_vision.config.settings import load_config
from lse_vision.models.landmark_classifier import load_landmark_model, normalize_landmarks
from lse_vision.utils.mediapipe_utils import create_landmarker, detect_hands, draw_hand_landmarks
from lse_vision.utils.runtime import resolve_device

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="LSE Sign Recognition")
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_INDEX_HTML = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

_config = load_config()
_device = torch.device(resolve_device())
_landmark_ckpt = _config.paths.checkpoints_dir / "landmark_classifier.pt"
_cnn_ckpt = _config.paths.checkpoints_dir / "best_static_sign_cnn.pt"

# Load model (prefer landmark MLP)
_lm_model = None
_lm_classes: list[str] = []
_cnn_model = None
_cnn_classes: list[str] = []
_cnn_transform = None
_classifier_name = ""

try:
    if _landmark_ckpt.exists():
        _lm_model, _lm_classes = load_landmark_model(_landmark_ckpt)
        _lm_model = _lm_model.to(_device).eval()
        _classifier_name = "Landmark MLP"
    if _cnn_ckpt.exists():
        from lse_vision.utils.checkpointing import load_checkpoint
        from lse_vision.training.common import build_model_from_checkpoint
        from lse_vision.preprocessing.transforms import (
            build_eval_transforms, build_eval_transforms_imagenet)
        ckpt = load_checkpoint(_cnn_ckpt, map_location="cpu")
        _cnn_classes = ckpt["classes"]
        _cnn_model = build_model_from_checkpoint(ckpt, _config).to(_device)
        _cnn_model.load_state_dict(ckpt["model_state_dict"])
        _cnn_model.eval()
        arch = ckpt.get("architecture", "cnn")
        _cnn_transform = (build_eval_transforms_imagenet if arch == "mobilenetv2"
                          else build_eval_transforms)(_config.data.image_size)
        if _lm_model is not None:
            _classifier_name = f"Hybrid MLP+CNN ({arch})"
        else:
            _classifier_name = f"CNN ({arch})"
except Exception as e:
    print(f"Aviso: no se pudo cargar modelo: {e}")

# MediaPipe landmarker
_landmarker = None
try:
    _landmarker = create_landmarker(mode="image", num_hands=1)
except Exception as e:
    print(f"Aviso: MediaPipe no disponible: {e}")

# Pre-compute one reference landmark set per letter at startup
_reference_landmarks: dict = {}

def _precompute_reference_landmarks() -> None:
    if _landmarker is None:
        return
    dataset_root = _config.paths.dataset_root
    if not dataset_root.exists():
        return
    for letter_dir in sorted(dataset_root.iterdir()):
        if not letter_dir.is_dir():
            continue
        label = letter_dir.name.upper()
        if len(label) != 1 or not label.isalpha():
            continue
        img_files = sorted(
            p for p in letter_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for img_path in img_files[:5]:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
            lms = detect_hands(_landmarker, bgr)
            if lms:
                _reference_landmarks[label] = [
                    {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                    for lm in lms[0]
                ]
                break

_precompute_reference_landmarks()


# ── Visualization helpers ─────────────────────────────────────────────────────

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _make_landmark_diagram(landmarks, size: int = 280) -> np.ndarray:
    """Create a clean landmark diagram from normalized feature vector."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 245  # off-white

    features = normalize_landmarks(landmarks)
    coords = [(features[i * 3], features[i * 3 + 1]) for i in range(21)]

    # Map [-1,1] → [pad, size-pad]
    pad = 30
    pts = [
        (int((x + 1) / 2 * (size - 2 * pad) + pad),
         int((y + 1) / 2 * (size - 2 * pad) + pad))
        for x, y in coords
    ]

    for a, b in _HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (100, 180, 100), 2)
    for i, (px, py) in enumerate(pts):
        color = (200, 80, 80) if i == 0 else (60, 130, 220)
        cv2.circle(img, (px, py), 6 if i == 0 else 4, color, -1)

    cv2.putText(img, "Forma de la mano", (8, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    return img


def _annotate_frame(frame: np.ndarray, letter: str, confidence: float,
                    landmarks_list: list) -> np.ndarray:
    """Draw landmarks and prediction on a copy of the frame."""
    out = frame.copy()
    if landmarks_list:
        draw_hand_landmarks(out, landmarks_list)

    # Semi-transparent result banner
    h, w = out.shape[:2]
    banner_h = 80
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    color = (80, 230, 80) if confidence >= 0.6 else (80, 165, 230)
    cv2.putText(out, letter, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
    cv2.putText(out, f"{confidence * 100:.1f}%", (90, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(out, _classifier_name, (w - 200, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return out


def _to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _crop_hand(bgr_frame: np.ndarray, landmarks: list, pad: float = 0.3) -> np.ndarray:
    """Crop the hand bounding box from the frame so CNN sees only the hand."""
    h, w = bgr_frame.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    dx = (max(xs) - min(xs)) * pad
    dy = (max(ys) - min(ys)) * pad
    x1 = max(0, int((min(xs) - dx) * w))
    x2 = min(w, int((max(xs) + dx) * w))
    y1 = max(0, int((min(ys) - dy) * h))
    y2 = min(h, int((max(ys) + dy) * h))
    if x2 <= x1 or y2 <= y1:
        return bgr_frame
    return bgr_frame[y1:y2, x1:x2]


def _classify_frame(bgr_frame: np.ndarray, use_hand_crop: bool = False) -> dict:
    """Run landmark detection + classification on a single BGR frame.

    use_hand_crop=True crops the hand bounding box before running the CNN so
    background pixels don't confuse it (used for video frames with real-world backgrounds).
    """
    landmarks_list = detect_hands(_landmarker, bgr_frame) if _landmarker else []

    if _lm_model is not None:
        if not landmarks_list:
            return {"letter": None, "confidence": 0.0,
                    "landmarks_list": [], "diagram": None, "top3": []}
        lms = landmarks_list[0]
        features = normalize_landmarks(lms)
        t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(_device)
        with torch.inference_mode():
            lm_probs = torch.softmax(_lm_model(t), dim=1)[0]

        if _cnn_model is not None and _cnn_transform is not None:
            from PIL import Image as PILImage
            cnn_src = _crop_hand(bgr_frame, lms) if use_hand_crop else bgr_frame
            pil = PILImage.fromarray(cv2.cvtColor(cnn_src, cv2.COLOR_BGR2RGB))
            tc = _cnn_transform(pil).unsqueeze(0).to(_device)
            with torch.inference_mode():
                cnn_probs_raw = torch.softmax(_cnn_model(tc), dim=1)[0]
            cnn_probs = torch.zeros_like(lm_probs)
            for ci, cls in enumerate(_cnn_classes):
                if cls in {c: idx for idx, c in enumerate(_lm_classes)}:
                    cnn_probs[_lm_classes.index(cls)] = cnn_probs_raw[ci]
            # MLP is primary (95%), CNN secondary (5%)
            final_probs = 0.95 * lm_probs + 0.05 * cnn_probs
        else:
            final_probs = lm_probs

        top3_vals, top3_idxs = torch.topk(final_probs, min(3, len(_lm_classes)))
        top3 = [{"letter": _lm_classes[int(i)], "confidence": float(v)}
                for v, i in zip(top3_vals, top3_idxs)]
        landmarks_3d = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in lms]
        return {
            "letter": top3[0]["letter"],
            "confidence": top3[0]["confidence"],
            "landmarks_list": landmarks_list,
            "landmarks_3d": landmarks_3d,
            "diagram": _make_landmark_diagram(lms),
            "top3": top3,
        }

    if _cnn_model is not None and _cnn_transform is not None:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        t = _cnn_transform(pil).unsqueeze(0).to(_device)
        with torch.inference_mode():
            probs = torch.softmax(_cnn_model(t), dim=1)[0]
        top3_vals, top3_idxs = torch.topk(probs, min(3, len(_cnn_classes)))
        top3 = [{"letter": _cnn_classes[int(i)], "confidence": float(v)}
                for v, i in zip(top3_vals, top3_idxs)]
        diag = _make_landmark_diagram(landmarks_list[0]) if landmarks_list else None
        return {
            "letter": top3[0]["letter"],
            "confidence": top3[0]["confidence"],
            "landmarks_list": landmarks_list,
            "diagram": diag,
            "top3": top3,
        }

    return {"letter": None, "confidence": 0.0, "landmarks_list": [], "diagram": None, "top3": []}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    model_ready = _lm_model is not None or _cnn_model is not None
    warning = (
        '<span style="background:#742a2a;color:#fc8181;">Sin modelo — ejecuta train primero</span>'
        if not model_ready else ""
    )
    html = (
        _INDEX_HTML
        .replace("__CLASSIFIER__", _classifier_name or "Sin modelo")
        .replace("__WARNING_BANNER__", warning)
    )
    return HTMLResponse(content=html)


@app.get("/api/metrics")
async def get_metrics():
    metrics_path = _config.paths.metrics_dir / "test_metrics.json"
    if not metrics_path.exists():
        return JSONResponse({"error": "No hay métricas. Ejecuta --mode evaluate primero."})
    return JSONResponse(json.loads(metrics_path.read_text(encoding="utf-8")))


@app.post("/api/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
):
    if url:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                content = r.read()
            fname = url.split("/")[-1].split("?")[0].lower()
        except Exception as e:
            return {"error": f"No se pudo descargar la imagen desde la URL: {e}"}
    elif file:
        content = await file.read()
        fname = (file.filename or "").lower()
    else:
        return {"error": "Se requiere un archivo o URL."}

    is_video = any(fname.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"))

    if is_video:
        # Save to temp, extract frames, pick best
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(fname).suffix or ".mp4") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        best_conf = -1.0
        best_result = None
        best_frame = None
        frame_idx = 0
        step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 60))  # sample ~60 frames

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            result = _classify_frame(frame, use_hand_crop=True)
            if result["letter"] and result["confidence"] > best_conf:
                best_conf = result["confidence"]
                best_result = result
                best_frame = frame.copy()
            frame_idx += step

        cap.release()
        Path(tmp_path).unlink(missing_ok=True)

        if best_result is None or best_result["letter"] is None:
            return {"error": "No se detectó ninguna mano en el vídeo."}

        result, frame = best_result, best_frame

    else:
        # Image
        arr = np.frombuffer(content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "No se pudo leer la imagen."}
        result = _classify_frame(frame, use_hand_crop=(source == "video"))
        if result["letter"] is None:
            return {"error": "No se detectó ninguna mano en la imagen."}

    annotated = _annotate_frame(frame, result["letter"], result["confidence"],
                                 result["landmarks_list"])

    response = {
        "letter": result["letter"],
        "confidence": round(result["confidence"] * 100, 1),
        "annotated_image": _to_base64(annotated),
        "classifier": _classifier_name,
        "top3": [{"letter": r["letter"], "confidence": round(r["confidence"] * 100, 1)}
                 for r in result.get("top3", [])],
    }
    if result["diagram"] is not None:
        response["landmark_diagram"] = _to_base64(result["diagram"])
    if result.get("landmarks_3d"):
        response["landmarks_3d"] = result["landmarks_3d"]

    return response


@app.get("/api/reference-landmarks")
async def get_reference_landmarks():
    return JSONResponse(_reference_landmarks)


@app.post("/api/extract-sequence")
async def extract_sequence(file: Optional[UploadFile] = File(None)):
    """Extract a sequence of distinct letters from a video at ~5fps."""
    if not file:
        return JSONResponse({"error": "Se requiere un archivo de vídeo."})
    content = await file.read()
    fname = (file.filename or "").lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(fname).suffix or ".mp4") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps / 5))
    sequence: list[str] = []
    stable_buf: list[str] = []
    locked_letter: str | None = None
    unlock_count = 0
    MIN_CONF = 0.80
    LOCK_NEEDED = 3
    UNLOCK_NEEDED = 3
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        result = _classify_frame(frame, use_hand_crop=True)
        letter = result["letter"]
        conf = result["confidence"]
        if letter and conf >= MIN_CONF:
            if locked_letter is None:
                stable_buf.append(letter)
                if len(stable_buf) >= LOCK_NEEDED and len(set(stable_buf[-LOCK_NEEDED:])) == 1:
                    locked_letter = letter
                    if not sequence or sequence[-1] != locked_letter:
                        sequence.append(locked_letter)
                    stable_buf = []
                    unlock_count = 0
            else:
                if letter != locked_letter:
                    unlock_count += 1
                    if unlock_count >= UNLOCK_NEEDED:
                        locked_letter = None
                        stable_buf = [letter]
                        unlock_count = 0
                else:
                    unlock_count = 0
        else:
            stable_buf = []
            if locked_letter:
                unlock_count += 1
                if unlock_count >= UNLOCK_NEEDED:
                    locked_letter = None
                    unlock_count = 0
        frame_idx += step

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return JSONResponse({"sequence": sequence})


@app.post("/api/predict-landmarks")
async def predict_from_landmarks(body: dict):
    """Predict letter directly from 21 landmark points {x,y,z}. Used by the hand editor."""
    if _lm_model is None:
        return JSONResponse({"error": "Landmark model not loaded."})
    lms_data = body.get("landmarks", [])
    if len(lms_data) != 21:
        return JSONResponse({"error": "Expected 21 landmarks."})

    class _FakeLM:
        def __init__(self, x, y, z):
            self.x = x; self.y = y; self.z = z

    lms = [_FakeLM(**p) for p in lms_data]
    features = normalize_landmarks(lms)
    t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(_device)
    with torch.inference_mode():
        probs = torch.softmax(_lm_model(t), dim=1)[0]
    top3_vals, top3_idxs = torch.topk(probs, min(3, len(_lm_classes)))
    top3 = [{"letter": _lm_classes[int(i)], "confidence": round(float(v) * 100, 1)}
            for v, i in zip(top3_vals, top3_idxs)]
    return JSONResponse({"letter": top3[0]["letter"], "confidence": top3[0]["confidence"], "top3": top3})


@app.post("/api/save-capture")
async def save_capture(letter: str = Form(...), image: UploadFile = File(...)):
    """Save a webcam capture for training data."""
    from datetime import datetime
    captures_dir = _config.paths.data_dir / "captures"
    captures_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_letter = letter.upper()[:1] if letter else "X"
    path = captures_dir / f"{safe_letter}_{ts}.jpg"
    path.write_bytes(await image.read())
    return JSONResponse({"saved": str(path)})
