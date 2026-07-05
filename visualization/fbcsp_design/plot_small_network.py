"""Plot compact FBCSP-MLP training and embedding figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import COLORS, CLASS_NAMES, load_npz, pca2, plot_confusion, rolling_axis, save_figure, set_style


def plot(metrics_dir: Path, output_dir: Path, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_style()
    arrays = load_npz(metrics_dir / "small_network" / "arrays.npz")
    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.9), constrained_layout=True)
    epochs = np.arange(1, len(arrays["history_loss"]) + 1)
    loss_line = axes[0, 0].plot(
        epochs,
        arrays["history_loss"],
        color="#dc2626",
        lw=1.8,
        label="train loss",
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[0, 0].set_title("Training loss")
    ax_acc = axes[0, 0].twinx()
    acc_line = ax_acc.plot(
        epochs,
        arrays["history_accuracy"],
        color="#2563eb",
        lw=1.6,
        label="train accuracy",
    )
    ax_acc.set_ylabel("Train accuracy")
    ax_acc.set_ylim(0, 1.03)
    lines = loss_line + acc_line
    axes[0, 0].legend(lines, [line.get_label() for line in lines], loc="center right")

    embeddings = np.concatenate([arrays["train_embeddings"], arrays["replay_embeddings"]], axis=0)
    coords = pca2(embeddings)
    n_train = len(arrays["train_embeddings"])
    labels = np.concatenate([arrays["train_labels"], arrays["replay_labels"]], axis=0)
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        axes[0, 1].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=np.where(np.arange(len(labels))[mask] < n_train, 17, 28),
            alpha=0.72,
            color=COLORS[class_index],
            label=class_name,
            edgecolors="none",
        )
    axes[0, 1].set_title("PCA view of learned embeddings")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].legend(loc="best")

    rolling_axis(
        axes[1, 0],
        arrays["rolling_command_accuracy"],
        arrays["rolling_reject_rate"],
        "Small-network replay traces",
    )
    plot_confusion(axes[1, 1], arrays["confusion"], "Small-network confusion")
    fig.suptitle("Embedding Line: FBCSP + Compact MLP", fontsize=14, weight="bold")
    saved.extend(save_figure(fig, output_dir, "small_network_training_embedding", formats))
    return saved
