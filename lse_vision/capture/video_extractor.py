from __future__ import annotations

"""Extract labeled landmark samples from a video file.

Controls:
  A-Z        — label current frame with that letter
  SPACE       — skip this frame
  LEFT/RIGHT  — step ±30 frames
  Q / ESC     — save and quit
"""

import csv
from pathlib import Path

import cv2

from lse_vision.config.settings import AppConfig
from lse_vision.models.landmark_classifier import normalize_landmarks
from lse_vision.utils.mediapipe_utils import create_landmarker, detect_hands, draw_hand_landmarks


def _draw_hud(frame, frame_idx: int, total: int, label_counts: dict[str, int]) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    pct = frame_idx / max(total - 1, 1)
    cv2.putText(frame, f"Frame {frame_idx}/{total}  ({pct * 100:.1f}%)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    total_labeled = sum(label_counts.values())
    cv2.putText(frame, f"Etiquetadas: {total_labeled}  |  {len(label_counts)} clases",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 120), 1)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 36), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "[A-Z] Etiquetar  |  [SPACE] Saltar  |  [<][>] ±30f  |  [Q] Guardar y salir",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    bar_y = h - 40
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + 4), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (int(w * pct), bar_y + 4), (0, 180, 80), -1)


def run_extract_video(config: AppConfig, video_path: Path, output_csv: Path) -> int:
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarker = create_landmarker(mode="image", num_hands=1)

    rows: list[list] = []
    label_counts: dict[str, int] = {}
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()

    print(f"\nVídeo: {video_path.name}  ({total_frames} frames)")
    print("Pulsa A-Z para etiquetar, SPACE para saltar, Q para guardar y salir.\n")

    while ok and frame is not None:
        display = frame.copy()
        landmarks_list = detect_hands(landmarker, frame)

        if landmarks_list:
            draw_hand_landmarks(display, landmarks_list)
            hand_color = (0, 200, 80)
            hand_txt = "Mano detectada"
        else:
            hand_color = (0, 80, 200)
            hand_txt = "Sin mano"

        cv2.circle(display, (display.shape[1] - 20, 20), 8, hand_color, -1)
        cv2.putText(display, hand_txt, (display.shape[1] - 140, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        _draw_hud(display, frame_idx, total_frames, label_counts)
        cv2.imshow("Extractor de Video - LSE", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            frame_idx += 1
        elif 65 <= key <= 90 or 97 <= key <= 122:
            letter = chr(key).upper()
            if landmarks_list:
                features = normalize_landmarks(landmarks_list[0])
                rows.append([letter] + features)
                label_counts[letter] = label_counts.get(letter, 0) + 1
                print(f"  [{letter}] muestra #{label_counts[letter]} en frame {frame_idx}")
            else:
                print(f"  [{letter}] sin mano en frame {frame_idx}, saltando")
            frame_idx += 1
        elif key in (81, 2):   # left arrow
            frame_idx = max(0, frame_idx - 30)
        elif key in (83, 3):   # right arrow
            frame_idx = min(total_frames - 1, frame_idx + 30)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if not rows:
        print("No se etiquetaron muestras.")
        return 1

    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + [f"f{i}" for i in range(63)])
        writer.writerows(rows)

    print(f"\nGuardadas {len(rows)} muestras en {output_csv}")
    for letter, count in sorted(label_counts.items()):
        print(f"  {letter}: {count}")
    return 0
