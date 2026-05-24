from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from lse_vision.config.settings import AppConfig, load_config
from lse_vision.training.common import build_dataloaders, build_device, build_model
from lse_vision.utils.checkpointing import save_checkpoint
from lse_vision.utils.logging_utils import configure_logging, get_logger
from lse_vision.utils.plotting import plot_training_history
from lse_vision.utils.runtime import ensure_project_directories
from lse_vision.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the static sign image classifier")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for the number of epochs.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
    desc: str = "",
) -> tuple[float, float]:
    model.train(training)
    running_loss = 0.0
    running_correct = 0
    total = 0

    progress = tqdm(dataloader, desc=desc, leave=False, unit="batch")
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (predictions == targets).sum().item()
        total += inputs.size(0)
        progress.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{running_correct / total:.4f}")

    return running_loss / total, running_correct / total


def run_training(config: AppConfig, epochs_override: int | None = None) -> int:
    ensure_project_directories(config)
    configure_logging(config.paths.logs_dir)
    logger = get_logger(__name__)
    set_seed(config.random_seed)
    total_epochs = epochs_override or config.training.epochs

    classes, dataloaders = build_dataloaders(config)
    device = build_device()
    model = build_model(config, num_classes=len(classes)).to(device)

    # Class-weighted loss to handle dataset imbalance
    from collections import Counter
    train_labels = [s.class_index for s in dataloaders["train"].dataset.samples]
    counts = Counter(train_labels)
    total = sum(counts.values())
    n = len(classes)
    class_weights = torch.tensor(
        [total / (n * counts[i]) for i in range(n)], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.training.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    history: list[dict] = []
    best_checkpoint_path = config.paths.checkpoints_dir / "best_static_sign_cnn.pt"
    last_checkpoint_path = config.paths.checkpoints_dir / "last_static_sign_cnn.pt"

    logger.info(
        "Training on %s classes with %s training images.",
        len(classes),
        len(dataloaders["train"].dataset),
    )

    for epoch in range(1, total_epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
            desc=f"Epoch {epoch}/{total_epochs} [train]",
        )
        val_loss, val_accuracy = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
            desc=f"Epoch {epoch}/{total_epochs} [val]",
        )

        scheduler.step(val_loss)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)
        logger.info(
            "Epoch %s/%s | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            total_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
            "image_size": config.data.image_size,
            "architecture": config.model.architecture,
        }
        save_checkpoint(checkpoint_payload, last_checkpoint_path)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_payload, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.training.early_stopping_patience:
            logger.info("Early stopping triggered at epoch %s.", epoch)
            break

    history_path = config.paths.metrics_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    plot_path = config.paths.metrics_dir / "training_history.png"
    plot_training_history(history, plot_path)

    logger.info("Saved best checkpoint to %s", best_checkpoint_path)
    logger.info("Saved training history to %s", history_path)
    logger.info("Saved training plot to %s", plot_path)
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    return run_training(config, epochs_override=args.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
