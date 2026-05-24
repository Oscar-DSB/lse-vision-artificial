from __future__ import annotations

"""Train the landmark-based MLP classifier from a CSV of captured hand landmarks."""

import argparse
import csv
import json
from pathlib import Path

import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lse_vision.config.settings import load_config
from lse_vision.models.landmark_classifier import FEATURE_DIM, LandmarkMLP, save_landmark_model
from lse_vision.utils.logging_utils import configure_logging, get_logger
from lse_vision.utils.plotting import plot_training_history
from lse_vision.utils.runtime import ensure_project_directories
from lse_vision.utils.seed import set_seed


def _compute_derived_features(lm: np.ndarray) -> np.ndarray:
    """Compute all 28 derived features from coords array (21x3).
    Returns: 5 ext_ratios + 5 curl_angles + 4 thumb_dists + 4 adj_dists
             + 3 palm_normal + 5 heights + 2 cross_dists = 28 values."""
    hand_scale = float(np.linalg.norm(lm[12] - lm[0])) or 1e-6

    _FINGERS = [(4, 3, 2, 1), (8, 7, 6, 5), (12, 11, 10, 9),
                (16, 15, 14, 13), (20, 19, 18, 17)]

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(lm[a] - lm[b]))

    def angle(a: int, v: int, b: int) -> float:
        va = lm[a] - lm[v]; vb = lm[b] - lm[v]
        mag = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-6
        return float(np.arccos(np.clip(np.dot(va, vb) / mag, -1, 1)))

    ext_ratios, curl_angles = [], []
    for tip, pip, mcp, _ in _FINGERS:
        mcd = dist(mcp, 0) or 1e-6
        ext_ratios.append(dist(tip, 0) / mcd)
        curl_angles.append(angle(tip, pip, mcp) / math.pi)

    thumb_dists = [dist(4, t) / hand_scale for t in [8, 12, 16, 20]]
    adj_dists = [dist(8, 12) / hand_scale, dist(12, 16) / hand_scale,
                 dist(16, 20) / hand_scale, dist(8, 20) / hand_scale]

    v1 = lm[5] - lm[0]; v2 = lm[17] - lm[0]
    normal = np.cross(v1, v2)
    mag = np.linalg.norm(normal) or 1e-6
    palm_normal = (normal / mag).tolist()

    palm_center = lm[[0, 5, 9, 13, 17]].mean(axis=0)
    heights = [float(np.dot(lm[t] - palm_center, normal / mag)) / hand_scale
               for t in [4, 8, 12, 16, 20]]

    cross = [dist(8, 16) / hand_scale, dist(12, 20) / hand_scale]

    return np.array(ext_ratios + curl_angles + thumb_dists + adj_dists
                    + palm_normal + heights + cross, dtype=np.float32)


def _augment(features: np.ndarray) -> np.ndarray:
    """Augment coords and recompute ALL derived features consistently."""
    coords = features[:63].reshape(21, 3).copy()

    # Random rotation around Z axis (in-plane, +-25 degrees)
    angle = random.uniform(-25, 25) * math.pi / 180
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    new_x = coords[:, 0] * cos_a - coords[:, 1] * sin_a
    new_y = coords[:, 0] * sin_a + coords[:, 1] * cos_a
    coords[:, 0], coords[:, 1] = new_x.copy(), new_y.copy()

    # Random tilt around X axis (depth tilt, +-15 degrees)
    tilt = random.uniform(-15, 15) * math.pi / 180
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    new_y2 = coords[:, 1] * cos_t - coords[:, 2] * sin_t
    new_z2 = coords[:, 1] * sin_t + coords[:, 2] * cos_t
    coords[:, 1], coords[:, 2] = new_y2.copy(), new_z2.copy()

    # Random scale +-15%
    coords *= random.uniform(0.85, 1.15)

    # Random translation +-10%
    coords[:, 0] += random.uniform(-0.1, 0.1)
    coords[:, 1] += random.uniform(-0.1, 0.1)

    # Gaussian noise
    coords += np.random.normal(0, 0.012, coords.shape)

    # Recompute ALL derived features from augmented coords
    derived = _compute_derived_features(coords)

    return np.concatenate([coords.reshape(63), derived]).astype(np.float32)


class LandmarkDataset(Dataset):
    def __init__(self, rows: list[list], classes: list[str], augment: bool = False) -> None:
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.augment = augment
        self.samples = [
            (np.array([float(v) for v in row[1:]], dtype=np.float32),
             self.class_to_idx[row[0]])
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        features, label = self.samples[idx]
        if self.augment:
            features = _augment(features)
        return torch.tensor(features, dtype=torch.float32), label


def load_csv(csv_path: Path) -> tuple[list[list], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV de landmarks no encontrado: {csv_path}\n"
                                "Ejecuta primero: python -m src.main --mode capture")
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append(row)
    classes = sorted(set(row[0] for row in rows))
    return rows, classes


def run_epoch(model, loader, criterion, optimizer, device, training: bool, desc: str = "") -> tuple[float, float]:
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for features, targets in tqdm(loader, desc=desc, leave=False, unit="batch"):
        features, targets = features.to(device), targets.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        correct += (logits.argmax(1) == targets).sum().item()
        total_loss += loss.item() * features.size(0)
        total += features.size(0)
    return total_loss / total, correct / total


def run_train_landmarks(config, csv_path: Path | None = None) -> int:
    ensure_project_directories(config)
    configure_logging(config.paths.logs_dir)
    logger = get_logger(__name__)
    set_seed(config.random_seed)

    landmark_csv = csv_path or (config.paths.data_dir / "landmarks.csv")
    rows, classes = load_csv(landmark_csv)
    logger.info("Cargadas %d muestras de %d clases desde %s", len(rows), len(classes), landmark_csv)

    # Stratified split: separate by class to ensure all classes in both sets
    from collections import defaultdict
    by_class: dict[str, list] = defaultdict(list)
    for row in rows:
        by_class[row[0]].append(row)

    train_rows, val_rows = [], []
    for cls_rows in by_class.values():
        random.shuffle(cls_rows)
        n_val_cls = max(1, int(len(cls_rows) * 0.15))
        val_rows.extend(cls_rows[:n_val_cls])
        train_rows.extend(cls_rows[n_val_cls:])

    train_set = LandmarkDataset(train_rows, classes, augment=True)
    val_set = LandmarkDataset(val_rows, classes, augment=False)
    logger.info("Train: %d muestras (con augmentación) | Val: %d muestras", len(train_set), len(val_set))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkMLP(num_classes=len(classes), input_dim=FEATURE_DIM, dropout=0.4).to(device)

    # Class-weighted loss to handle dataset imbalance
    from collections import Counter
    train_class_idx = {c: i for i, c in enumerate(classes)}
    counts = Counter(row[0] for row in train_rows)
    total = sum(counts.values())
    n = len(classes)
    class_weights = torch.tensor(
        [total / (n * counts.get(cls, 1)) for cls in classes], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_count = 0
    history = []
    checkpoint_path = config.paths.checkpoints_dir / "landmark_classifier.pt"

    for epoch in range(1, 201):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                          True, f"Epoch {epoch} [train]")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device,
                                      False, f"Epoch {epoch} [val]")
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                         "val_loss": val_loss, "val_accuracy": val_acc})
        logger.info("Epoch %d | train_acc=%.4f | val_acc=%.4f", epoch, train_acc, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            save_landmark_model(model, classes, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= 20:
                logger.info("Early stopping en epoch %d", epoch)
                break

    # Save history and plot
    hist_path = config.paths.metrics_dir / "landmark_training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_training_history(history, config.paths.metrics_dir / "landmark_training_history.png")
    logger.info("Mejor val_acc=%.4f | Modelo guardado en %s", best_val_acc, checkpoint_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el clasificador de landmarks")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    return run_train_landmarks(config, csv_path=args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
