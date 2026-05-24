from __future__ import annotations

"""Migrate landmarks.csv from 81 features to 91 features.

The new 10 features (palm normal x3, fingertip heights x5, cross dists x2)
are computed from the 63 existing coordinate features already in the CSV.
"""

import csv
import sys
from pathlib import Path

import numpy as np


def _new_features_from_coords(lm: np.ndarray) -> list[float]:
    """lm: shape (21, 3) — coordinates relative to wrist, normalized by max_val."""
    hand_scale = float(np.linalg.norm(lm[12] - lm[0])) or 1e-6

    # Palm normal
    v1 = lm[5] - lm[0]
    v2 = lm[17] - lm[0]
    normal = np.cross(v1, v2)
    mag = float(np.linalg.norm(normal)) or 1e-6
    palm_normal = (normal / mag).tolist()

    # Fingertip heights above palm plane
    palm_center = lm[[0, 5, 9, 13, 17]].mean(axis=0)
    heights = []
    for tip in [4, 8, 12, 16, 20]:
        diff = lm[tip] - palm_center
        h = float(np.dot(diff, normal / mag)) / hand_scale
        heights.append(h)

    # Cross-finger distances
    cross = [
        float(np.linalg.norm(lm[8] - lm[16])) / hand_scale,
        float(np.linalg.norm(lm[12] - lm[20])) / hand_scale,
    ]

    return palm_normal + heights + cross  # 10 values


def migrate_csv(input_path: Path, output_path: Path) -> None:
    print(f"Migrando {input_path} -> {output_path}")
    rows_out = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_existing = len(header) - 1  # exclude label column

        new_cols = [f"f{i}" for i in range(n_existing, n_existing + 10)]
        new_header = header + new_cols

        for i, row in enumerate(reader):
            label = row[0]
            features = [float(v) for v in row[1:]]

            if len(features) == 91:
                rows_out.append(row)
                continue

            if len(features) < 63:
                print(f"  Fila {i} ignorada: solo {len(features)} features")
                continue

            coords_flat = features[:63]
            lm = np.array(coords_flat, dtype=np.float64).reshape(21, 3)
            new_feats = _new_features_from_coords(lm)
            rows_out.append([label] + features + new_feats)

            if i % 5000 == 0:
                print(f"  Procesadas {i} filas...")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(rows_out)

    print(f"Migracion completada: {len(rows_out)} filas -> {output_path}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2] / "data"
    src = base / "landmarks.csv"
    dst = base / "landmarks.csv"
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    if len(sys.argv) > 2:
        dst = Path(sys.argv[2])
    # backup first
    backup = src.with_name("landmarks_81feat_backup.csv")
    import shutil
    shutil.copy(src, backup)
    print(f"Backup guardado en {backup}")
    migrate_csv(src, dst)
