from __future__ import annotations

from pathlib import Path


def plot_training_history(history: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax_loss.plot(epochs, val_loss, label="Val Loss", marker="o")
    ax_loss.set_title("Loss per Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    ax_acc.plot(epochs, val_acc, label="Val Accuracy", marker="o")
    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
