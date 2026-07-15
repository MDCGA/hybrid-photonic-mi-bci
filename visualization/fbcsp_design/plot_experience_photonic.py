"""Plot experience-library retrieval and photonic scan figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import COLORS, CLASS_NAMES, bar_labels, load_json, load_npz, pca2, plot_confusion, rolling_axis, save_figure, set_style


def plot(metrics_dir: Path, output_dir: Path, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_style()
    arrays = load_npz(metrics_dir / "experience_photonic" / "arrays.npz")
    small_arrays = load_npz(metrics_dir / "small_network" / "arrays.npz")
    summary = load_json(metrics_dir / "experience_photonic" / "summary.json")
    selected = summary["selected_entries"]
    saved = []

    fig_training, ax_loss = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    epochs = np.arange(1, len(small_arrays["history_loss"]) + 1)
    loss_line = ax_loss.plot(
        epochs,
        small_arrays["history_loss"],
        color="#dc2626",
        lw=1.8,
        label="train loss",
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Mainline shared encoder training")
    ax_accuracy = ax_loss.twinx()
    accuracy_line = ax_accuracy.plot(
        epochs,
        small_arrays["history_accuracy"],
        color="#2563eb",
        lw=1.6,
        label="train accuracy",
    )
    ax_accuracy.set_ylabel("Train accuracy")
    ax_accuracy.set_ylim(0, 1.03)
    lines = loss_line + accuracy_line
    ax_loss.legend(lines, [line.get_label() for line in lines], loc="center right")
    saved.extend(save_figure(fig_training, output_dir, "mainline_encoder_training", formats))

    evaluation_indices = arrays["evaluation_replay_indices"].astype(int)
    train_embeddings = small_arrays["train_embeddings"]
    eval_embeddings = small_arrays["replay_embeddings"][evaluation_indices]
    train_labels = small_arrays["train_labels"].astype(int)
    eval_labels = arrays["eval_labels"].astype(int)
    embeddings = np.concatenate([train_embeddings, eval_embeddings], axis=0)
    labels_all = np.concatenate([train_labels, eval_labels], axis=0)
    split = np.concatenate(
        [np.zeros(len(train_embeddings), dtype=int), np.ones(len(eval_embeddings), dtype=int)]
    )
    coords = pca2(embeddings)
    fig_embedding, ax_embedding = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    for class_index, class_name in enumerate(CLASS_NAMES):
        train_mask = (labels_all == class_index) & (split == 0)
        eval_mask = (labels_all == class_index) & (split == 1)
        ax_embedding.scatter(
            coords[train_mask, 0],
            coords[train_mask, 1],
            s=14,
            alpha=0.32,
            color=COLORS[class_index],
            label=f"{class_name} train",
            edgecolors="none",
        )
        ax_embedding.scatter(
            coords[eval_mask, 0],
            coords[eval_mask, 1],
            s=25,
            alpha=0.72,
            color=COLORS[class_index],
            marker="x",
            label=f"{class_name} eval",
        )
    ax_embedding.set_title("Mainline embedding PCA: train vs held-out evaluation")
    ax_embedding.set_xlabel("PC1")
    ax_embedding.set_ylabel("PC2")
    ax_embedding.legend(loc="best", ncol=2, fontsize=8)
    saved.extend(save_figure(fig_embedding, output_dir, "mainline_embedding_pca", formats))

    correct_trace = np.asarray(arrays["correct_trace"], dtype=np.float64)
    rejected = np.asarray(arrays["rejected"], dtype=bool)
    windows = np.arange(1, len(correct_trace) + 1)
    cumulative_command = np.cumsum(correct_trace) / windows
    cumulative_reject = np.cumsum(rejected.astype(np.float64)) / windows
    accepted_windows = np.maximum(1, windows - np.cumsum(rejected.astype(int)))
    cumulative_accepted = np.cumsum(correct_trace) / accepted_windows
    fig_cumulative, ax_cumulative = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax_cumulative.plot(
        windows,
        cumulative_command,
        color="#2563eb",
        lw=1.8,
        label="cumulative command accuracy",
    )
    ax_cumulative.plot(
        windows,
        cumulative_accepted,
        color="#16a34a",
        lw=1.6,
        label="cumulative accepted accuracy",
    )
    ax_cumulative.plot(
        windows,
        cumulative_reject,
        color="#dc2626",
        lw=1.5,
        label="cumulative reject rate",
    )
    ax_cumulative.set_ylim(-0.03, 1.03)
    ax_cumulative.set_xlabel("Held-out evaluation window")
    ax_cumulative.set_ylabel("Cumulative rate")
    ax_cumulative.set_title("Mainline cumulative replay metrics")
    ax_cumulative.legend(loc="best")
    saved.extend(save_figure(fig_cumulative, output_dir, "mainline_cumulative_metrics", formats))

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

    fig_weights, ax_weights = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    bars = ax_weights.bar(x, arrays["retrieval_weights"], color="#2563eb")
    ax_weights.set_title("Calibration-aware candidate fusion weights")
    ax_weights.set_ylabel("Fusion weight")
    ax_weights.set_xticks(x, labels, rotation=20, ha="right")
    bar_labels(ax_weights, bars)
    saved.extend(save_figure(fig_weights, output_dir, "mainline_retrieval_weights", formats))

    fig_quality, ax_quality = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax_quality.hist(arrays["library_train_accuracy"], bins=15, color="#94a3b8", alpha=0.82)
    ax_quality.scatter(
        arrays["selected_train_accuracy"],
        np.full_like(arrays["selected_train_accuracy"], 1.0),
        color="#dc2626",
        s=34,
        label="selected",
    )
    ax_quality.set_title("Experience-library head quality")
    ax_quality.set_xlabel("Train accuracy of candidate head")
    ax_quality.set_ylabel("Entries")
    ax_quality.legend()
    saved.extend(save_figure(fig_quality, output_dir, "mainline_candidate_head_quality", formats))

    fig_trace, ax_trace = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    rolling_axis(
        ax_trace,
        arrays["rolling_command_accuracy"],
        arrays["rolling_reject_rate"],
        "Photonic candidate-scan replay traces",
    )
    saved.extend(save_figure(fig_trace, output_dir, "mainline_replay_traces", formats))

    fig_confusion, ax_confusion = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    plot_confusion(ax_confusion, arrays["confusion"], "Photonic-scan confusion")
    saved.extend(save_figure(fig_confusion, output_dir, "mainline_confusion", formats))

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
        f"4-bit tile: qin 0..15, qwt -8..7; bias uses the constant input.",
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
