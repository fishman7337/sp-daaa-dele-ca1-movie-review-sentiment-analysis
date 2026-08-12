"""Visualization helpers for TensorFlow training histories."""

from __future__ import annotations


def plot_training_history(history, title: str = "Training History"):
    """Plot available training and validation metrics over epochs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting training history.") from exc

    metrics = history.history
    epochs = range(1, len(metrics.get("loss", [])) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "accuracy" in metrics:
        axes[0].plot(epochs, metrics["accuracy"], label="Train Accuracy")
        axes[0].plot(epochs, metrics.get("val_accuracy", []), label="Validation Accuracy")
        axes[0].set_title(f"{title} - Accuracy")
    else:
        axes[0].plot(epochs, metrics.get("mae", []), label="Train MAE")
        axes[0].plot(epochs, metrics.get("val_mae", []), label="Validation MAE")
        axes[0].set_title(f"{title} - MAE")
    axes[0].legend()

    axes[1].plot(epochs, metrics.get("loss", []), label="Train Loss")
    axes[1].plot(epochs, metrics.get("val_loss", []), label="Validation Loss")
    axes[1].set_title(f"{title} - Loss")
    axes[1].legend()
    figure.tight_layout()
    return figure
