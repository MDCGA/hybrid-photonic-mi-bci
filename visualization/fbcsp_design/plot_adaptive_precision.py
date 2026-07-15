"""Plot adaptive logical-precision monitoring and tile usage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, save_figure, set_style


BIT_COLORS = {4: "#16a34a", 6: "#2563eb", 8: "#dc2626"}
POLICY_LABELS = {
    "car": "CAR",
    "sos_filter": "SOS filter",
    "fbcsp_projection": "FBCSP projection",
    "feature_standardization": "Standardization",
    "default": "Other linear ops",
}


def plot(metrics_dir: Path, output_dir: Path, formats: Iterable[str]) -> list[Path]:
    reports = sorted(metrics_dir.glob("adaptive_precision_eval_*.json"))
    if not reports:
        return []
    report = load_json(reports[-1])
    summary = list(report["precision_summary"])
    operations = list(report["precision_operations"])
    stage_tiles = dict(report["online_tile_counts"])
    validation = report.get("precision_validation") or {}
    validation_windows = len(validation.get("windows", [])) if isinstance(validation, dict) else 0
    observed_windows = max(1, len(report.get("online_repeats", [])) + validation_windows)

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    _plot_operation_distribution(axes[0, 0], summary)
    _plot_tile_distribution(axes[0, 1], summary, observed_windows)
    _plot_shadow_errors(axes[1, 0], operations)
    _plot_stage_tiles(axes[1, 1], stage_tiles)
    fig.suptitle(
        "Adaptive photonic precision diagnostics | "
        f"evaluation window {report['evaluation_index']}",
        fontsize=14,
    )
    saved = save_figure(fig, output_dir, "adaptive_precision_diagnostics", formats)
    if isinstance(validation, dict) and validation.get("windows"):
        saved.extend(_plot_ab_validation(validation, output_dir, formats))
    return saved


def _plot_ab_validation(
    validation: dict[str, object],
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    summary = dict(validation["summary"])
    windows = list(validation["windows"])
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    metrics = ("Command accuracy", "Accepted accuracy", "Reject rate")
    adaptive = np.array(
        [
            summary["adaptive_command_accuracy"],
            summary["adaptive_accepted_accuracy"],
            summary["adaptive_reject_rate"],
        ],
        dtype=float,
    )
    fixed = np.array(
        [
            summary["fixed_8bit_command_accuracy"],
            summary["fixed_8bit_accepted_accuracy"],
            summary["fixed_8bit_reject_rate"],
        ],
        dtype=float,
    )
    x = np.arange(len(metrics))
    width = 0.36
    axes[0].bar(x - width / 2, adaptive, width, color="#2563eb", label="Adaptive")
    axes[0].bar(x + width / 2, fixed, width, color="#64748b", label="Fixed 8-bit")
    axes[0].set_xticks(x, metrics, rotation=18, ha="right")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Task metrics")
    axes[0].legend(loc="lower left")

    resource_labels = ("Median time", "Mean tiles")
    adaptive_resource = np.array(
        [
            float(summary["adaptive_median_ms"]) / float(summary["fixed_8bit_median_ms"]),
            float(summary["adaptive_mean_tiles"]) / float(summary["fixed_8bit_mean_tiles"]),
        ]
    )
    fixed_resource = np.ones(2, dtype=float)
    rx = np.arange(len(resource_labels))
    axes[1].bar(rx - width / 2, adaptive_resource, width, color="#16a34a", label="Adaptive")
    axes[1].bar(rx + width / 2, fixed_resource, width, color="#64748b", label="Fixed 8-bit")
    axes[1].axhline(1.0, color="#111827", lw=1.0, linestyle="--")
    axes[1].set_xticks(rx, resource_labels)
    axes[1].set_ylabel("Normalized to fixed 8-bit")
    axes[1].set_title("Runtime and tile cost")
    axes[1].set_ylim(0.0, 1.12)

    window_x = np.arange(len(windows))
    probability_error = np.array([float(row["probability_l2_error"]) for row in windows])
    agreement = np.array([bool(row["prediction_agreement"]) for row in windows])
    point_colors = np.where(agreement, "#16a34a", "#dc2626")
    axes[2].plot(window_x, probability_error, color="#2563eb", lw=1.5)
    axes[2].scatter(window_x, probability_error, c=point_colors, s=34, zorder=3)
    axes[2].set_xticks(window_x, [str(row["evaluation_index"]) for row in windows])
    axes[2].set_xlabel("Evaluation-window index")
    axes[2].set_ylabel("Probability L2 difference")
    axes[2].set_title(f"Prediction agreement = {float(summary['prediction_agreement']):.3f}")
    fig.suptitle("Adaptive precision vs fixed 8-bit forward", fontsize=14)
    return save_figure(fig, output_dir, "adaptive_vs_fixed8_validation", formats)


def _policy_order(summary: list[dict[str, object]]) -> list[str]:
    preferred = [
        "car",
        "sos_filter",
        "fbcsp_projection",
        "feature_standardization",
        "default",
    ]
    present = {str(row["policy"]) for row in summary}
    return [policy for policy in preferred if policy in present] + sorted(present - set(preferred))


def _plot_operation_distribution(ax: plt.Axes, summary: list[dict[str, object]]) -> None:
    policies = _policy_order(summary)
    x = np.arange(len(policies))
    bottoms = np.zeros(len(policies), dtype=float)
    for bits in (4, 6, 8):
        values = np.array(
            [
                sum(
                    int(row["operations"])
                    for row in summary
                    if row["policy"] == policy and int(row["current_bits"]) == bits
                )
                for policy in policies
            ],
            dtype=float,
        )
        ax.bar(x, values, bottom=bottoms, color=BIT_COLORS[bits], label=f"{bits}-bit")
        bottoms += values
    ax.set_xticks(x, [POLICY_LABELS.get(policy, policy) for policy in policies], rotation=20, ha="right")
    ax.set_ylabel("Independent operator states")
    ax.set_title("Selected logical precision")
    ax.legend(ncol=3, loc="upper right")


def _plot_tile_distribution(
    ax: plt.Axes,
    summary: list[dict[str, object]],
    observed_windows: int,
) -> None:
    policies = _policy_order(summary)
    x = np.arange(len(policies))
    bottoms = np.zeros(len(policies), dtype=float)
    for bits in (4, 6, 8):
        values = np.array(
            [
                sum(
                    int(row["tile_evaluations"])
                    for row in summary
                    if row["policy"] == policy and int(row["current_bits"]) == bits
                )
                for policy in policies
            ],
            dtype=float,
        )
        ax.bar(x, values / 1000.0, bottom=bottoms / 1000.0, color=BIT_COLORS[bits], label=f"{bits}-bit")
        bottoms += values
    ax.set_xticks(x, [POLICY_LABELS.get(policy, policy) for policy in policies], rotation=20, ha="right")
    ax.set_ylabel("Cumulative physical tile evaluations (thousands)")
    ax.set_title(f"Tile cost by precision and policy | {observed_windows} adaptive windows")


def _plot_shadow_errors(ax: plt.Axes, operations: list[dict[str, object]]) -> None:
    monitored = [row for row in operations if int(row["monitored_calls"]) > 0]
    monitored.sort(key=lambda row: (str(row["policy"]), str(row["operation"])))
    x = np.arange(len(monitored))
    errors = np.array([float(row["max_relative_error"]) for row in monitored])
    limits = np.array([float(row["relative_error_limit"]) for row in monitored])
    selected_bits = np.array([int(row["current_bits"]) for row in monitored])
    for bits in (4, 6, 8):
        mask = selected_bits == bits
        if np.any(mask):
            ax.scatter(
                x[mask],
                errors[mask],
                color=BIT_COLORS[bits],
                s=20,
                alpha=0.85,
                label=f"{bits}-bit error",
            )
    ax.plot(x, limits, color="#111827", lw=1.2, linestyle="--", label="operation limit")
    ax.set_xlabel("Monitored operator state")
    ax.set_ylabel("Normalized error vs 8-bit shadow")
    ax.set_title("Continuous precision guard")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92)


def _plot_stage_tiles(ax: plt.Axes, stage_tiles: dict[str, object]) -> None:
    labels = [
        "FBCSP",
        "Standardize",
        "Small MLP",
        "Candidate scan",
    ]
    keys = [
        "fbcsp_transform_one_window",
        "standardize_one_window",
        "small_mlp_forward_one_window",
        "pure_runtime_photonic_scan_one_window",
    ]
    values = np.array([int(stage_tiles.get(key, 0)) for key in keys], dtype=float)
    colors = ("#2563eb", "#16a34a", "#dc2626", "#64748b")
    bars = ax.bar(np.arange(len(labels)), values, color=colors)
    ax.set_yscale("symlog", linthresh=100.0)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylabel("Physical tile evaluations (symlog)")
    ax.set_title("Single-window tile distribution")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
