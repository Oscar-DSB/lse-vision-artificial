from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

# 63 normalized coords + 5 extension ratios (3D) + 5 curl angles (3D)
# + 4 thumb-to-fingertip distances + 4 adjacent-fingertip distances
# + 3 palm normal vector + 5 fingertip heights + 2 cross-finger distances = 91
FEATURE_DIM = 91


class LandmarkMLP(nn.Module):
    """MLP classifier over MediaPipe hand landmarks (91 features)."""

    def __init__(self, num_classes: int, input_dim: int = FEATURE_DIM, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_landmarks(landmarks: list) -> list[float]:
    """Return 91 features: 63 normalized coords + 5 extension ratios (3D) +
    5 curl angles (3D) + 4 thumb-to-fingertip + 4 adjacent-fingertip distances
    + 3 palm normal vector + 5 fingertip heights above palm + 2 cross-finger dists."""
    wrist_x, wrist_y, wrist_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

    coords = []
    for lm in landmarks:
        coords.append(lm.x - wrist_x)
        coords.append(lm.y - wrist_y)
        coords.append(lm.z - wrist_z)

    max_val = max(abs(v) for v in coords) or 1.0
    coords = [v / max_val for v in coords]

    # ── 3D helpers ──────────────────────────────────────────────────────────
    def _dist3d(a: int, b: int) -> float:
        return math.sqrt(
            (landmarks[a].x - landmarks[b].x) ** 2 +
            (landmarks[a].y - landmarks[b].y) ** 2 +
            (landmarks[a].z - landmarks[b].z) ** 2
        )

    def _angle3d(a: int, vertex: int, b: int) -> float:
        ax = landmarks[a].x - landmarks[vertex].x
        ay = landmarks[a].y - landmarks[vertex].y
        az = landmarks[a].z - landmarks[vertex].z
        bx = landmarks[b].x - landmarks[vertex].x
        by = landmarks[b].y - landmarks[vertex].y
        bz = landmarks[b].z - landmarks[vertex].z
        dot = ax * bx + ay * by + az * bz
        mag = (math.sqrt(ax**2 + ay**2 + az**2) * math.sqrt(bx**2 + by**2 + bz**2)) or 1e-6
        return math.acos(max(-1.0, min(1.0, dot / mag)))

    _FINGERS = [(4, 3, 2, 1), (8, 7, 6, 5), (12, 11, 10, 9), (16, 15, 14, 13), (20, 19, 18, 17)]
    hand_scale = _dist3d(0, 12) or 1e-6

    extension_ratios = []
    curl_angles = []
    for tip, pip, mcp, _ in _FINGERS:
        mcp_dist = _dist3d(mcp, 0) or 1e-6
        extension_ratios.append(_dist3d(tip, 0) / mcp_dist)
        curl_angles.append(_angle3d(tip, pip, mcp) / math.pi)

    TIPS = [8, 12, 16, 20]
    thumb_dists = [_dist3d(4, t) / hand_scale for t in TIPS]

    adjacent_dists = [
        _dist3d(8, 12) / hand_scale,
        _dist3d(12, 16) / hand_scale,
        _dist3d(16, 20) / hand_scale,
        _dist3d(8, 20) / hand_scale,
    ]

    # ── Palm normal vector (3) ───────────────────────────────────────────────
    # Cross product of (wrist→index_base) × (wrist→pinky_base)
    v1x = landmarks[5].x - landmarks[0].x
    v1y = landmarks[5].y - landmarks[0].y
    v1z = landmarks[5].z - landmarks[0].z
    v2x = landmarks[17].x - landmarks[0].x
    v2y = landmarks[17].y - landmarks[0].y
    v2z = landmarks[17].z - landmarks[0].z
    nx = v1y * v2z - v1z * v2y
    ny = v1z * v2x - v1x * v2z
    nz = v1x * v2y - v1y * v2x
    n_mag = math.sqrt(nx**2 + ny**2 + nz**2) or 1e-6
    palm_normal = [nx / n_mag, ny / n_mag, nz / n_mag]

    # ── Fingertip heights above palm plane (5) ───────────────────────────────
    palm_ids = [0, 5, 9, 13, 17]
    pcx = sum(landmarks[i].x for i in palm_ids) / 5
    pcy = sum(landmarks[i].y for i in palm_ids) / 5
    pcz = sum(landmarks[i].z for i in palm_ids) / 5
    fingertip_heights = []
    for tip in [4, 8, 12, 16, 20]:
        dx = landmarks[tip].x - pcx
        dy = landmarks[tip].y - pcy
        dz = landmarks[tip].z - pcz
        h = (dx * palm_normal[0] + dy * palm_normal[1] + dz * palm_normal[2]) / hand_scale
        fingertip_heights.append(h)

    # ── Additional cross-finger distances (2) ────────────────────────────────
    cross_dists = [
        _dist3d(8, 16) / hand_scale,   # index tip ↔ ring tip
        _dist3d(12, 20) / hand_scale,  # middle tip ↔ pinky tip
    ]

    return (coords + extension_ratios + curl_angles + thumb_dists + adjacent_dists
            + palm_normal + fingertip_heights + cross_dists)


def save_landmark_model(
    model: LandmarkMLP,
    classes: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "num_classes": len(classes),
            "input_dim": FEATURE_DIM,
            "architecture": "landmark_mlp",
        },
        output_path,
    )


def load_landmark_model(checkpoint_path: Path) -> tuple[LandmarkMLP, list[str]]:
    payload: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = payload["classes"]
    input_dim = payload.get("input_dim", FEATURE_DIM)
    model = LandmarkMLP(num_classes=len(classes), input_dim=input_dim)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, classes
