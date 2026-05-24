from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    dataset_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_dir: Path
    inference_dir: Path


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    train_split: float
    val_split: float
    test_split: float
    batch_size: int
    num_workers: int
    allowed_extensions: list[str]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    label_smoothing: float
    save_every: int


@dataclass(frozen=True)
class InferenceConfig:
    top_k: int
    confidence_threshold: float
    webcam_camera_index: int
    roi_size: int
    mirror_webcam: bool


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    conv_channels: list[int]
    classifier_hidden_dim: int
    dropout: float
    architecture: str


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    random_seed: int
    paths: PathConfig
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig
    model: ModelConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError("Top-level configuration must be a JSON object.")

    return payload


def _resolve_path(root: Path, raw_value: str) -> Path:
    return (root / raw_value).resolve()


def load_config(config_path: Path | None = None) -> AppConfig:
    root = _project_root()
    config_file = config_path or Path(__file__).parent / "default_config.json"
    raw_config = _load_json(config_file)

    paths_section = raw_config["paths"]

    paths = PathConfig(
        project_root=root,
        dataset_root=_resolve_path(root, paths_section["dataset_root"]),
        data_dir=_resolve_path(root, paths_section["data_dir"]),
        raw_dir=_resolve_path(root, paths_section["raw_dir"]),
        interim_dir=_resolve_path(root, paths_section["interim_dir"]),
        processed_dir=_resolve_path(root, paths_section["processed_dir"]),
        outputs_dir=_resolve_path(root, paths_section["outputs_dir"]),
        checkpoints_dir=_resolve_path(root, paths_section["checkpoints_dir"]),
        logs_dir=_resolve_path(root, paths_section["logs_dir"]),
        metrics_dir=_resolve_path(root, paths_section["metrics_dir"]),
        inference_dir=_resolve_path(root, paths_section["inference_dir"]),
    )

    return AppConfig(
        project_name=str(raw_config["project_name"]),
        random_seed=int(raw_config["random_seed"]),
        paths=paths,
        data=DataConfig(
            image_size=int(raw_config["data"]["image_size"]),
            train_split=float(raw_config["data"]["train_split"]),
            val_split=float(raw_config["data"]["val_split"]),
            test_split=float(raw_config["data"]["test_split"]),
            batch_size=int(raw_config["data"]["batch_size"]),
            num_workers=int(raw_config["data"]["num_workers"]),
            allowed_extensions=list(raw_config["data"]["allowed_extensions"]),
        ),
        training=TrainingConfig(
            epochs=int(raw_config["training"]["epochs"]),
            learning_rate=float(raw_config["training"]["learning_rate"]),
            weight_decay=float(raw_config["training"]["weight_decay"]),
            early_stopping_patience=int(raw_config["training"]["early_stopping_patience"]),
            label_smoothing=float(raw_config["training"]["label_smoothing"]),
            save_every=int(raw_config["training"]["save_every"]),
        ),
        inference=InferenceConfig(
            top_k=int(raw_config["inference"]["top_k"]),
            confidence_threshold=float(raw_config["inference"]["confidence_threshold"]),
            webcam_camera_index=int(raw_config["inference"]["webcam_camera_index"]),
            roi_size=int(raw_config["inference"]["roi_size"]),
            mirror_webcam=bool(raw_config["inference"]["mirror_webcam"]),
        ),
        model=ModelConfig(
            input_channels=int(raw_config["model"]["input_channels"]),
            conv_channels=[int(value) for value in raw_config["model"]["conv_channels"]],
            classifier_hidden_dim=int(raw_config["model"]["classifier_hidden_dim"]),
            dropout=float(raw_config["model"]["dropout"]),
            architecture=str(raw_config["model"].get("architecture", "cnn")),
        ),
    )
