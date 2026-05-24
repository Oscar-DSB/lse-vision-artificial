from __future__ import annotations

"""Guided webcam capture of MediaPipe hand landmarks for each sign letter."""

import csv
import time
from pathlib import Path

import cv2

from lse_vision.config.settings import AppConfig
from lse_vision.models.landmark_classifier import normalize_landmarks
from lse_vision.utils.mediapipe_utils import create_landmarker, detect_hands, draw_hand_landmarks

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_SAMPLES_PER_LETTER = 1000
_PANEL_H = 140


def _draw_panel(frame, letter: str, letter_idx: int, total_letters: int,
                collected: int, phase: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - _PANEL_H), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, f"Letra: {letter}  ({letter_idx + 1}/{total_letters})",
                (12, h - _PANEL_H + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    if phase == "ready":
        msg = "[ESPACIO] Iniciar captura     [Q] Salir"
        color = (80, 220, 255)
    elif phase == "capture":
        msg = f"Capturando...  {collected}/{_SAMPLES_PER_LETTER}     [P] Pausar     [Q] Salir"
        color = (80, 230, 80)
    elif phase == "paused":
        msg = f"PAUSADO  {collected}/{_SAMPLES_PER_LETTER}     [P] Reanudar     [Q] Salir"
        color = (80, 165, 255)
    else:  # done
        msg = f"Completada! {collected} muestras     [ESPACIO] Siguiente     [Q] Salir"
        color = (255, 200, 60)

    cv2.putText(frame, msg, (12, h - _PANEL_H + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    # Barra de progreso
    bar_x, bar_y, bar_w, bar_h_px = 12, h - _PANEL_H + 80, w - 24, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (50, 50, 50), -1)
    filled = int(bar_w * collected / _SAMPLES_PER_LETTER)
    bar_color = (0, 200, 80) if phase == "capture" else (60, 130, 220)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h_px), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (100, 100, 100), 1)

    pct = int(collected / _SAMPLES_PER_LETTER * 100)
    cv2.putText(frame, f"{pct}%", (bar_x + bar_w + 8, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Instrucción de mano si no hay detección en capture
    if phase == "capture":
        cv2.putText(frame, "Mantén el signo visible", (12, h - _PANEL_H + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)


def run_capture(config: AppConfig, output_csv: Path, letters: list[str] | None = None) -> int:
    letters_to_capture = letters or _LETTERS
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    landmarker = create_landmarker(mode="image", num_hands=1)
    cap = cv2.VideoCapture(config.inference.webcam_camera_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    rows: list[list] = []
    letter_idx = 0

    try:
        while letter_idx < len(letters_to_capture):
            letter = letters_to_capture[letter_idx]
            phase = "ready"   # ready → capture ↔ paused → done
            collected = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if config.inference.mirror_webcam:
                    frame = cv2.flip(frame, 1)

                landmarks_list = detect_hands(landmarker, frame)
                hand_detected = len(landmarks_list) > 0

                if hand_detected:
                    draw_hand_landmarks(frame, landmarks_list)

                if phase == "capture" and hand_detected and collected < _SAMPLES_PER_LETTER:
                    features = normalize_landmarks(landmarks_list[0])
                    rows.append([letter] + features)
                    collected += 1
                    if collected >= _SAMPLES_PER_LETTER:
                        phase = "done"

                _draw_panel(frame, letter, letter_idx, len(letters_to_capture),
                            collected, phase)
                cv2.imshow("Captura de Gestos LSE", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    letter_idx = len(letters_to_capture)
                    break

                if key == ord(" ") or key == 13:
                    if phase == "ready":
                        phase = "capture"
                    elif phase == "done":
                        break

                if key == ord("p"):
                    if phase == "capture":
                        phase = "paused"
                    elif phase == "paused":
                        phase = "capture"

            letter_idx += 1

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    if not rows:
        print("No se capturaron datos.")
        return 1

    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + [f"f{i}" for i in range(81)])
        writer.writerows(rows)

    print(f"\nGuardadas {len(rows)} muestras en {output_csv}")
    return 0
