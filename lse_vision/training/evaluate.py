from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lse_vision.config.settings import AppConfig, load_config
from lse_vision.training.common import build_dataloaders, build_device, build_model_from_checkpoint
from lse_vision.utils.checkpointing import load_checkpoint
from lse_vision.utils.logging_utils import configure_logging, get_logger
from lse_vision.utils.metrics import build_classification_metrics, save_confusion_matrix_csv, save_metrics_json
from lse_vision.utils.runtime import ensure_project_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the static sign classifier")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def run_evaluate(config: AppConfig, checkpoint_path: Path) -> int:
    ensure_project_directories(config)
    configure_logging(config.paths.logs_dir)
    logger = get_logger(__name__)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    classes, dataloaders = build_dataloaders(config)
    device = build_device()
    model = build_model_from_checkpoint(checkpoint, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for inputs, batch_targets in dataloaders["test"]:
            inputs = inputs.to(device)
            logits = model(inputs)
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            targets.extend(batch_targets.tolist())

    metrics = build_classification_metrics(targets=targets, predictions=predictions, class_names=classes)
    metrics_path = config.paths.metrics_dir / "test_metrics.json"
    confusion_csv_path = config.paths.metrics_dir / "confusion_matrix.csv"
    save_metrics_json(metrics, metrics_path)
    save_confusion_matrix_csv(metrics["confusion_matrix"], classes, confusion_csv_path)

    logger.info("Test accuracy: %.4f", metrics["accuracy"])
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved confusion matrix to %s", confusion_csv_path)
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or (config.paths.checkpoints_dir / "best_static_sign_cnn.pt")
    return run_evaluate(config, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    raise SystemExit(main())
