from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lse_vision.config.settings import AppConfig
from lse_vision.dataset.sign_image_dataset import (
    StaticSignImageDataset,
    discover_samples,
    load_split_manifest,
    save_split_manifest,
    split_samples,
)
from lse_vision.models.cnn_classifier import StaticSignCNN, build_mobilenetv2
from lse_vision.preprocessing.transforms import (
    build_eval_transforms,
    build_eval_transforms_imagenet,
    build_train_transforms,
    build_train_transforms_imagenet,
)
from lse_vision.utils.runtime import resolve_device


def split_manifest_path(config: AppConfig) -> Path:
    return config.paths.processed_dir / "image_splits.json"


def prepare_splits(config: AppConfig) -> tuple[list[str], dict[str, list]]:
    manifest_path = split_manifest_path(config)
    if manifest_path.exists():
        return load_split_manifest(manifest_path)

    samples, classes = discover_samples(
        dataset_root=config.paths.dataset_root,
        allowed_extensions=config.data.allowed_extensions,
    )
    split_map = split_samples(
        samples=samples,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        random_seed=config.random_seed,
    )
    save_split_manifest(manifest_path, classes, split_map)
    return classes, split_map


def build_dataloaders(config: AppConfig) -> tuple[list[str], dict[str, DataLoader]]:
    classes, split_map = prepare_splits(config)

    if config.model.architecture == "mobilenetv2":
        train_transform = build_train_transforms_imagenet(config.data.image_size)
        eval_transform = build_eval_transforms_imagenet(config.data.image_size)
    else:
        train_transform = build_train_transforms(config.data.image_size)
        eval_transform = build_eval_transforms(config.data.image_size)

    dataloaders = {
        "train": DataLoader(
            StaticSignImageDataset(split_map["train"], transform=train_transform),
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        ),
        "val": DataLoader(
            StaticSignImageDataset(split_map["val"], transform=eval_transform),
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ),
        "test": DataLoader(
            StaticSignImageDataset(split_map["test"], transform=eval_transform),
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ),
    }
    return classes, dataloaders


def build_model(config: AppConfig, num_classes: int) -> torch.nn.Module:
    if config.model.architecture == "mobilenetv2":
        return build_mobilenetv2(
            num_classes=num_classes,
            dropout=config.model.dropout,
        )
    return StaticSignCNN(
        num_classes=num_classes,
        input_channels=config.model.input_channels,
        conv_channels=config.model.conv_channels,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
        dropout=config.model.dropout,
    )


def build_model_from_checkpoint(checkpoint: dict[str, Any], config: AppConfig) -> torch.nn.Module:
    arch = checkpoint.get("architecture", "cnn")
    num_classes = len(checkpoint["classes"])
    if arch == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, dropout=config.model.dropout)
    return StaticSignCNN(
        num_classes=num_classes,
        input_channels=config.model.input_channels,
        conv_channels=config.model.conv_channels,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
        dropout=config.model.dropout,
    )


def build_device() -> torch.device:
    return torch.device(resolve_device())
