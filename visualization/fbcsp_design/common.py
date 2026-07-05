"""Shared plotting helpers for FBCSP design figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ("left", "right", "foot")
OUTPUT_NAMES = ("left", "right", "foot", "reject")
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#64748b")


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def plot_confusion(ax: plt.Axes, confusion: np.ndarray, title: str) -> None:
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted output")
    ax.set_ylabel("True MI class")
    ax.set_xticks(np.arange(len(OUTPUT_NAMES)), OUTPUT_NAMES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    vmax = max(1, int(confusion.max()))
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            value = int(confusion[row, col])
            color = "white" if value > vmax * 0.55 else "#0f172a"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def pca2(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def rolling_axis(ax: plt.Axes, rolling_accuracy: np.ndarray, rolling_reject: np.ndarray, title: str) -> None:
    x = np.arange(1, len(rolling_accuracy) + 1)
    ax.plot(x, rolling_accuracy, color="#2563eb", lw=1.8, label="rolling command accuracy")
    ax.plot(x, rolling_reject, color="#dc2626", lw=1.5, label="rolling reject rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Replay decision window")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")


def bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )
