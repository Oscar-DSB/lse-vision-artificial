from __future__ import annotations

"""Wrapper for MediaPipe Tasks API (mediapipe >= 0.10).

The legacy mp.solutions API was removed in 0.10.x.
This module provides a uniform interface for hand landmark detection.
"""

import urllib.request
from pathlib import Path

import cv2

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"

# Hand skeleton connections (MediaPipe topology)
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]


def ensure_model() -> Path:
    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo MediaPipe hand_landmarker (~8 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
        print(f"Modelo guardado en {_MODEL_CACHE}")
    return _MODEL_CACHE


def create_landmarker(mode: str = "image", num_hands: int = 1):
    """Create a HandLandmarker. mode = 'image' or 'video'."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = ensure_model()
    running_mode = (
        mp_vision.RunningMode.VIDEO if mode == "video" else mp_vision.RunningMode.IMAGE
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, bgr_frame, timestamp_ms: int = 0, mode: str = "image"):
    """Return list of hand landmark lists (each item = 21 NormalizedLandmark).

    NormalizedLandmark objects have .x, .y, .z attributes — same as the old API.
    Returns an empty list when no hands are detected.
    """
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if mode == "video":
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        result = landmarker.detect(mp_image)
    return result.hand_landmarks  # list[list[NormalizedLandmark]]


def draw_hand_landmarks(frame, landmarks_list: list) -> None:
    """Draw hand skeleton on frame using OpenCV."""
    h, w = frame.shape[:2]
    for landmarks in landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 120), 2)
        for i, (px, py) in enumerate(pts):
            color = (255, 80, 80) if i == 0 else (50, 210, 255)
            cv2.circle(frame, (px, py), 4, color, -1)
