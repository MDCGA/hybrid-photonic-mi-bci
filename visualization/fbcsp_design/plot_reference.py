"""Plot FBCSP + shrinkage LDA reference figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, load_npz, plot_confusion, rolling_axis, save_figure, set_style


def plot(metrics_dir: Path, output_dir: Path, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_style()
    arrays = load_npz(metrics_dir / "reference" / "arrays.npz")
    summary = load_json(metrics_dir / "reference" / "summary.json")
    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.2), constrained_layout=True)
    rolling_axis(
        axes[0, 0],
        arrays["rolling_command_accuracy"],
        arrays["rolling_reject_rate"],
        "FBCSP-LDA online replay traces",
    )
    axes[0, 1].hist(arrays["confidence"], bins=24, color="#2563eb", alpha=0.78)
    axes[0, 1].axvline(
        summary["summary"]["reject_threshold"],
        color="#dc2626",
        lw=1.6,
        label="reject threshold",
    )
    axes[0, 1].set_title("Replay confidence distribution")
    axes[0, 1].set_xlabel("Max softmax probability")
    axes[0, 1].set_ylabel("Windows")
    axes[0, 1].legend()
    plot_confusion(axes[1, 0], arrays["confusion"], "FBCSP-LDA confusion")
    selected_names = [_short_feature(name) for name in arrays["selected_feature_names"].astype(str)[:10]]
    y = np.arange(len(selected_names))
    axes[1, 1].barh(y, np.arange(len(selected_names), 0, -1), color="#16a34a")
    axes[1, 1].set_yticks(y, selected_names)
    axes[1, 1].tick_params(axis="y", labelsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Top selected FBCSP features")
    axes[1, 1].set_xlabel("Selection rank score")
    fig.suptitle("Reference Line: FBCSP + Shrinkage LDA", fontsize=14, weight="bold")
    saved.extend(save_figure(fig, output_dir, "reference_fbcsp_lda_diagnostics", formats))
    return saved


def _short_feature(name: str) -> str:
    return name.replace("Hz_", " Hz / ").replace("_", " / ")
