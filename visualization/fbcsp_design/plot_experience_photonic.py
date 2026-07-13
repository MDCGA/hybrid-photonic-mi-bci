"""Plot experience-library retrieval and photonic scan figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import bar_labels, load_json, load_npz, plot_confusion, rolling_axis, save_figure, set_style


def plot(metrics_dir: Path, output_dir: Path, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_style()
    arrays = load_npz(metrics_dir / "experience_photonic" / "arrays.npz")
    summary = load_json(metrics_dir / "experience_photonic" / "summary.json")
    selected = summary["selected_entries"]
    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.2), constrained_layout=True)
    labels = [_short_entry(entry["entry_id"]) for entry in selected]
    x = np.arange(len(labels))
    bars = axes[0, 0].bar(x, arrays["retrieval_weights"], color="#2563eb")
    axes[0, 0].set_title("Calibration-aware candidate fusion weights")
    axes[0, 0].set_ylabel("Fusion weight")
    axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")
    bar_labels(axes[0, 0], bars)

    axes[0, 1].hist(arrays["library_train_accuracy"], bins=15, color="#94a3b8", alpha=0.82)
    axes[0, 1].scatter(
        arrays["selected_train_accuracy"],
        np.full_like(arrays["selected_train_accuracy"], 1.0),
        color="#dc2626",
        s=34,
        label="selected",
    )
    axes[0, 1].set_title("Experience-library head quality")
    axes[0, 1].set_xlabel("Train accuracy of candidate head")
    axes[0, 1].set_ylabel("Entries")
    axes[0, 1].legend()

    rolling_axis(
        axes[1, 0],
        arrays["rolling_command_accuracy"],
        arrays["rolling_reject_rate"],
        "Photonic candidate-scan replay traces",
    )
    plot_confusion(axes[1, 1], arrays["confusion"], "Photonic-scan confusion")
    fig.suptitle("Mainline: Experience Library + Tiled Candidate Linear-Head Scan", fontsize=14, weight="bold")
    saved.extend(save_figure(fig, output_dir, "experience_photonic_scan_diagnostics", formats))

    fig2, ax = plt.subplots(figsize=(9.0, 4.8))
    row = summary["summary"]
    total_tiles = row["tile_evaluations_per_window"]
    tile_rows, tile_cols = row["tile_shape"]
    class_count = int(arrays["eval_candidate_scores"].shape[2])
    augmented_dim = int(row["embedding_dim"]) + 1
    row_tiles = int(np.ceil(class_count / tile_rows))
    col_tiles = int(np.ceil(augmented_dim / tile_cols))
    values = [row["top_k"], row_tiles, col_tiles, total_tiles]
    names = [
        "Candidates K",
        f"row tiles\nceil({class_count}/{tile_rows})",
        f"col tiles\nceil({augmented_dim}/{tile_cols})",
        "tile evals\nper window",
    ]
    bars = ax.bar(names, values, color=["#2563eb", "#16a34a", "#16a34a", "#dc2626"])
    ax.set_title("2 x 8 photonic tile schedule for scanned linear heads")
    ax.set_ylabel("Count")
    ax.text(
        0.02,
        0.92,
        f"Each candidate head computes A_i [h, 1] with A_i in R^({class_count} x {augmented_dim}).\n"
        f"A {tile_rows} x {tile_cols} primitive scans row and column blocks; bias uses the constant input.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ed", "edgecolor": "#fdba74"},
    )
    bar_labels(ax, bars, "{:.0f}")
    saved.extend(save_figure(fig2, output_dir, "photonic_tile_schedule", formats))
    return saved


def _short_entry(entry_id: str) -> str:
    if entry_id == "anchor_mlp_classifier":
        return "MLP\nanchor"
    if entry_id == "anchor_embedding_lda":
        return "Emb-LDA\nanchor"
    return entry_id.replace("bootstrap_", "b")
