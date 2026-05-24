from __future__ import annotations

from pathlib import Path

from lse_vision.config.settings import AppConfig

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def ensure_project_directories(config: AppConfig) -> None:
    managed_directories: list[Path] = [
        config.paths.data_dir,
        config.paths.raw_dir,
        config.paths.interim_dir,
        config.paths.processed_dir,
        config.paths.outputs_dir,
        config.paths.checkpoints_dir,
        config.paths.logs_dir,
        config.paths.metrics_dir,
        config.paths.inference_dir,
    ]

    for directory in managed_directories:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_runtime_info(config: AppConfig) -> dict[str, object]:
    return {
        "device": resolve_device(),
        "dataset_root": str(config.paths.dataset_root),
        "directories": {
            "processed_dir": str(config.paths.processed_dir),
            "checkpoints_dir": str(config.paths.checkpoints_dir),
            "metrics_dir": str(config.paths.metrics_dir),
            "inference_dir": str(config.paths.inference_dir),
            "logs_dir": str(config.paths.logs_dir),
        },
    }
