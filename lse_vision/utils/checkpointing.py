from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_checkpoint(checkpoint_path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)
