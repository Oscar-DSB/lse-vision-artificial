from __future__ import annotations

"""Extract landmark samples from the existing image dataset (no webcam needed).

Iterates over every image in dataset_root/{A-Z}/*.png, runs MediaPipe on each,
and writes valid detections to a CSV ready for --mode train-landmarks.
"""

import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from lse_vision.config.settings import AppConfig
from lse_vision.models.landmark_classifier import FEATURE_DIM, normalize_landmarks
from lse_vision.utils.mediapipe_utils import create_landmarker, detect_hands


def run_extract_dataset(config: AppConfig, output_csv: Path) -> int:
    dataset_root = config.paths.dataset_root
    extensions = set(config.data.allowed_extensions)

    image_paths: list[tuple[str, Path]] = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.upper()
        if len(label) != 1 or not label.isalpha():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                image_paths.append((label, img_path))

    if not image_paths:
        print(f"ERROR: No se encontraron imágenes en {dataset_root}")
        return 1

    print(f"\nExtrayendo landmarks de {len(image_paths)} imágenes del dataset...")
    landmarker = create_landmarker(mode="image", num_hands=1)

    rows: list[list] = []
    skipped = 0
    label_counts: dict[str, int] = {}

    MAX_DIM = 640
    # IMREAD_REDUCED_COLOR_4 tells the JPEG decoder to decode at 1/4 resolution
    # natively (much faster than decoding full 4K then resizing).
    _JPEG_FLAG = getattr(cv2, "IMREAD_REDUCED_COLOR_4", 33)

    for label, img_path in tqdm(image_paths, desc="Extrayendo"):
        ext = img_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            bgr = cv2.imread(str(img_path), _JPEG_FLAG)
        else:
            bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue
        h, w = bgr.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        landmarks_list = detect_hands(landmarker, bgr)
        if not landmarks_list:
            skipped += 1
            continue
        features = normalize_landmarks(landmarks_list[0])
        rows.append([label] + features)
        label_counts[label] = label_counts.get(label, 0) + 1

    landmarker.close()

    if not rows:
        print("ERROR: MediaPipe no detectó ninguna mano en el dataset.")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])
        writer.writerows(rows)

    print(f"\nGuardadas {len(rows)} muestras ({skipped} sin mano) en {output_csv}")
    print("Muestras por clase:")
    for letter, count in sorted(label_counts.items()):
        print(f"  {letter}: {count}")
    print(f"\nAhora entrena el MLP con:")
    print(f"  python -m lse_vision.main --mode train-landmarks --csv \"{output_csv}\"")
    return 0
