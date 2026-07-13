"""Plot photonic-vs-digital linear-compute accounting for FBCSP design lines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, save_figure, set_style


def plot(metrics_dir: Path, output_dir: Path, formats: Iterable[str]) -> list[Path]:
    accounting = load_json(metrics_dir / "compute_accounting.json")
    lines = list(accounting["lines"])
    labels = [_short_label(str(line["line"])) for line in lines]
    photonic = np.asarray(
        [line["summary"]["linear_macs_photonic"] for line in lines],
        dtype=np.float64,
    )
    digital = np.asarray(
        [line["summary"]["linear_macs_digital"] for line in lines],
        dtype=np.float64,
    )
    total_share = np.asarray(
        [line["summary"]["photonic_linear_share"] for line in lines],
        dtype=np.float64,
    )
    inference_share = np.asarray(
        [line["summary"]["photonic_linear_share_inference"] for line in lines],
        dtype=np.float64,
    )
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    x = np.arange(len(labels))
    width = 0.64
    scale = 1e9
    axes[0].bar(
        x,
        photonic / scale,
        width=width,
        color="#2563eb",
        label="MatrixOps + SignalOps photonic",
    )
    axes[0].bar(
        x,
        digital / scale,
        bottom=photonic / scale,
        width=width,
        color="#f97316",
        label="Digital linear compute",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Linear compute (GMAC)")
    axes[0].set_title("Forward linear MAC split")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for index, value in enumerate((photonic + digital) / scale):
        axes[0].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    share_width = 0.34
    axes[1].bar(x - share_width / 2, total_share, width=share_width, color="#0f766e", label="forward")
    axes[1].bar(
        x + share_width / 2,
        inference_share,
        width=share_width,
        color="#7c3aed",
        label="online inference",
    )
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Photonic share")
    axes[1].set_title("Photonic share of linear MACs")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for xpos, values in ((x - share_width / 2, total_share), (x + share_width / 2, inference_share)):
        for xi, value in zip(xpos, values):
            axes[1].text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Forward linear compute: backend matrix and signal ops photonic")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return save_figure(fig, output_dir, "compute_accounting_summary", formats)


def _short_label(label: str) -> str:
    if "library" in label:
        return "Mainline"
    if "small MLP" in label:
        return "FBCSP+MLP"
    if "LDA" in label:
        return "FBCSP+LDA"
    return label
