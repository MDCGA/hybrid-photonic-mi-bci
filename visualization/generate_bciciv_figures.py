"""Generate BCICIV_1_asc figures for the hybrid photonic MI-BCI baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_bciciv_replay import SUBJECTS, run_pooled, run_subject  # noqa: E402
from hybrid_photonic_mi_bci.datasets import (  # noqa: E402
    DEFAULT_BAND,
    DEFAULT_MOTOR_CHANNELS,
    DEFAULT_WINDOW,
    BCICIVFeatures,
)
from hybrid_photonic_mi_bci.experiment import (  # noqa: E402
    PipelineBuildConfig,
    class_targets,
)


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#7c3aed")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip())
    _set_style()

    data_dir = Path(args.data_dir)
    config = PipelineBuildConfig(
        n_train=args.n_train,
        n_candidates=args.n_candidates,
        candidate_noise=args.candidate_noise,
        library_kind=args.library_kind,
        selector_kind=args.selector,
        reject_threshold=args.reject_threshold,
        margin_threshold=args.margin_threshold,
        seed=args.seed,
    )
    subject_result = _run_requested_dataset(
        data_dir=data_dir,
        subject=args.subject,
        config=config,
        band=(args.band_low, args.band_high),
        window=(args.window_start, args.window_end),
        channels=tuple(args.channels.split(",")),
        n_train_per_subject=args.n_train,
        warmup_trials=args.warmup_trials,
    )
    reject_comparison = [
        (f"default {config.selector_kind}", subject_result),
        (
            "bandit baseline",
            _run_requested_dataset(
                data_dir=data_dir,
                subject=args.subject,
                config=replace(config, selector_kind="bandit"),
                band=(args.band_low, args.band_high),
                window=(args.window_start, args.window_end),
                channels=tuple(args.channels.split(",")),
                n_train_per_subject=args.n_train,
                warmup_trials=args.warmup_trials,
            ),
        ),
        (
            "high reject threshold",
            _run_requested_dataset(
                data_dir=data_dir,
                subject=args.subject,
                config=replace(config, reject_threshold=0.40, margin_threshold=0.0),
                band=(args.band_low, args.band_high),
                window=(args.window_start, args.window_end),
                channels=tuple(args.channels.split(",")),
                n_train_per_subject=args.n_train,
                warmup_trials=args.warmup_trials,
            ),
        ),
    ]
    summary_results = [
        run_subject(
            data_dir=data_dir,
            subject=subject,
            config=config,
            band=(args.band_low, args.band_high),
            window=(args.window_start, args.window_end),
            channels=tuple(args.channels.split(",")),
            warmup_trials=args.warmup_trials,
        )
        for subject in SUBJECTS
    ]

    generated: list[Path] = []
    generated.extend(plot_system_block_diagram(output_dir, formats))
    generated.extend(plot_feature_distributions(subject_result, output_dir, formats))
    generated.extend(plot_projection_fit(subject_result, output_dir, formats))
    generated.extend(plot_online_replay(subject_result, reject_comparison, output_dir, formats))
    generated.extend(plot_subject_summary(summary_results, output_dir, formats))
    generated.extend(plot_confusion_matrix(subject_result, output_dir, formats))

    print("Generated BCICIV figures:")
    for path in generated:
        print(f"- {path}")


def plot_system_block_diagram(output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(16.8, 9.2))
    ax.set_axis_off()
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 10.2)

    ax.text(
        9.1,
        9.8,
        "Hybrid Photonic MI-BCI Algorithm Implementation on BCICIV_1_asc",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    sections = [
        (0.25, 7.1, 17.65, 1.85, "Dataset and 8-D Feature Extraction", "#eff6ff"),
        (0.25, 4.75, 17.65, 1.75, "Offline Calibration and Candidate Library", "#fffbeb"),
        (0.25, 2.15, 17.65, 1.95, "Online Replay: One EEG Decision Window", "#f5f3ff"),
        (0.25, 0.35, 17.65, 1.25, "Outputs, Metrics, and Hardware Boundary", "#f8fafc"),
    ]
    for x, y, w, h, label, face in sections:
        _section_band(ax, x, y, w, h, label, face)

    feature_boxes = [
        (
            "ASCII files\ncnt/mrk/nfo a-g\n200 trials/file",
            0.55,
            7.55,
            2.25,
            1.0,
            "#dbeafe",
        ),
        (
            "Label pooling\nlocal +/-1 -> nfo\nleft/right/foot",
            3.15,
            7.55,
            2.25,
            1.0,
            "#e0f2fe",
        ),
        (
            "Trial window\nmarker + 1.0-4.0 s\n8 motor channels",
            5.75,
            7.55,
            2.35,
            1.0,
            "#dcfce7",
        ),
        (
            "Preprocess\nCAR + 4th-order SOS\n8-30 Hz sosfiltfilt",
            8.5,
            7.55,
            2.45,
            1.0,
            "#cffafe",
        ),
        (
            "Feature method\nlog-bandpower only\nno CSP/FBCSP",
            11.35,
            7.55,
            2.15,
            1.0,
            "#bbf7d0",
        ),
        (
            "Pooled arrays\nfeatures: (1400,8)\nlabels: (1400,)",
            13.95,
            7.55,
            2.45,
            1.0,
            "#dcfce7",
        ),
    ]
    _draw_box_chain(ax, feature_boxes)

    calibration_boxes = [
        ("Train split\n120/file = 840\npooled first", 0.55, 5.12, 2.2, 0.95, "#fef3c7"),
        ("Standardizer.fit\nmu, sigma from\ntrain only", 3.1, 5.12, 2.1, 0.95, "#fde68a"),
        ("Projection fit\nnp.linalg.lstsq\n8-D -> 2-D W0", 5.55, 5.12, 2.25, 0.95, "#fed7aa"),
        ("ProjectionLibrary\n32 x (2 x 8)\nperturb noise=0.02", 8.15, 5.12, 2.35, 0.95, "#fef08a"),
        ("Classifier state\n2-D class prototypes\nno SVM/CNN", 10.85, 5.12, 2.35, 0.95, "#e9d5ff"),
        ("Reject rule\nconfidence < 0.34\nor margin threshold", 13.55, 5.12, 1.95, 0.95, "#fecaca"),
    ]
    _draw_box_chain(ax, calibration_boxes)

    online_boxes = [
        ("Replay split\n80/file = 560\none window at a time", 0.55, 2.65, 2.25, 1.0, "#ede9fe"),
        ("Standardizer.transform\nx_std = (x - mu)/sigma", 3.15, 2.65, 2.25, 1.0, "#ddd6fe"),
        ("Candidate weights\nW[0:31]\nshape: (32,2,8)", 5.75, 2.65, 2.25, 1.0, "#fef3c7"),
        ("MVMBackend.scan\nNumPy: W @ x_std\nreturn: (32,2)", 8.35, 2.65, 2.4, 1.0, "#fdba74"),
        ("PrototypeDecisionHead\nnearest-prototype softmax\nclass prob + margin", 11.1, 2.65, 2.45, 1.0, "#fbcfe8"),
        ("ProbabilityFusionSelector\nfuse 32 candidates\nonline reward update", 13.9, 2.65, 2.55, 1.0, "#c4b5fd"),
    ]
    _draw_box_chain(ax, online_boxes)

    _rounded_box(
        ax,
        7.85,
        0.55,
        2.85,
        0.82,
        "Photonic backend slot\nsame scan() contract\n(32,2,8) x (8,) -> (32,2)",
        "#ffedd5",
        edge="#ea580c",
        fontsize=7.7,
    )
    _rounded_box(
        ax,
        10.98,
        0.55,
        2.35,
        0.82,
        "Hardware hides\nweight programming\nreadout/correction",
        "#fff7ed",
        edge="#ea580c",
        fontsize=7.6,
    )
    _arrow(ax, 9.28, 1.38, 9.58, 2.58, color="#ea580c")
    _arrow(ax, 10.76, 0.96, 10.94, 0.96, color="#ea580c")

    _rounded_box(
        ax,
        13.95,
        0.55,
        1.65,
        0.82,
        "Output\nleft/right/foot/reject",
        "#fee2e2",
        fontsize=8.3,
    )
    _rounded_box(
        ax,
        15.95,
        0.55,
        1.75,
        0.82,
        "Metrics/Figures\naccuracy, confusion,\nreject curves",
        "#e2e8f0",
        fontsize=7.8,
    )
    _arrow(ax, 16.45, 2.58, 14.78, 1.42, color="#7c3aed", connectionstyle="arc3,rad=-0.18")
    _arrow(ax, 15.62, 0.96, 15.9, 0.96)

    ax.text(
        15.15,
        7.24,
        "One pooled matrix is split into calibration and replay blocks below.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#475569",
    )
    _arrow(ax, 4.15, 5.1, 4.15, 3.68, color="#64748b")
    _arrow(ax, 9.35, 5.1, 6.85, 3.68, color="#b45309", connectionstyle="arc3,rad=0.1")
    _arrow(ax, 12.0, 5.1, 12.35, 3.68, color="#be185d")
    _arrow(ax, 14.5, 5.1, 12.95, 3.68, color="#be123c", connectionstyle="arc3,rad=-0.1")
    _arrow(ax, 16.2, 2.62, 16.2, 1.42, color="#7c3aed")
    _arrow(ax, 14.75, 2.58, 9.6, 5.05, color="#7c3aed", connectionstyle="arc3,rad=0.22")

    _rounded_box(
        ax,
        0.55,
        0.44,
        6.45,
        0.72,
        "Algorithm summary\nFeature: CAR + 8-30 Hz + log-bandpower, not CSP\nClassifier: least-squares 2-D projection + prototype softmax, not SVM",
        "#ffffff",
        edge="#64748b",
        fontsize=7.1,
    )
    file_notes = [
        ("datasets/bciciv_1_asc.py", 0.7, 0.24),
        ("experiment.py", 3.05, 0.24),
        ("backends.py", 8.85, 0.23),
        ("decision.py + calibration.py", 11.1, 0.23),
        ("run_bciciv_replay.py", 15.85, 0.23),
    ]
    for text, x, y in file_notes:
        ax.text(x, y, text, ha="left", va="center", fontsize=8.0, color="#475569")

    return _save_figure(fig, output_dir, "bciciv_system_block_diagram", formats)


def plot_feature_distributions(
    result: dict[str, object],
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    dataset = _dataset(result)
    labels = dataset.labels
    class_names = dataset.class_names
    x = np.arange(dataset.features.shape[1])
    means = np.stack([dataset.features[labels == idx].mean(axis=0) for idx in range(len(class_names))])
    stds = np.stack([dataset.features[labels == idx].std(axis=0) for idx in range(len(class_names))])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    width = min(0.8 / len(class_names), 0.32)
    for idx, class_name in enumerate(class_names):
        offset = (idx - (len(class_names) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            means[idx],
            yerr=stds[idx],
            width=width,
            color=COLORS[idx],
            alpha=0.78,
            capsize=3,
            label=class_name,
        )
    axes[0].set_title(f"{_format_dataset_label(dataset.subject)}: 8-D Log-Bandpower Features")
    axes[0].set_ylabel("log variance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(_short_feature_names(dataset.feature_names), rotation=35, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    centered = means - means.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(centered)) or 1.0
    im = axes[1].imshow(centered, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
    axes[1].set_title("Class Mean Offset by Feature")
    axes[1].set_xlabel("Feature")
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_yticklabels([name.capitalize() for name in class_names])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(_short_feature_names(dataset.feature_names), rotation=35, ha="right")
    for row in range(centered.shape[0]):
        for col in range(centered.shape[1]):
            axes[1].text(col, row, f"{centered[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle(
        f"BCICIV_1_asc Feature Diagnostics, {dataset.band[0]:.0f}-{dataset.band[1]:.0f} Hz, "
        f"{dataset.window[0]:.1f}-{dataset.window[1]:.1f} s",
        fontsize=14,
        weight="bold",
    )
    return _save_figure(fig, output_dir, "bciciv_feature_distributions", formats)


def plot_projection_fit(
    result: dict[str, object],
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    dataset = _dataset(result)
    pipeline = result["pipeline"]
    config = result["config"]
    if not isinstance(config, PipelineBuildConfig):
        raise TypeError("config result is not PipelineBuildConfig")
    if pipeline.standardizer is None:
        raise RuntimeError("pipeline should contain a fitted Standardizer")

    x_all = pipeline.standardizer.transform(dataset.features)
    weights = pipeline.projection_library[0].weights
    z_all = x_all @ weights.T
    labels = dataset.labels
    train_mask = np.zeros(len(labels), dtype=bool)
    train_mask[: config.n_train] = True
    prototypes = (
        pipeline.decision_head.prototypes[0]
        if pipeline.decision_head.prototypes.ndim == 3
        else pipeline.decision_head.prototypes
    )
    targets = class_targets(len(dataset.class_names))[labels]
    rmse_train = _rmse(z_all[train_mask], targets[train_mask])
    rmse_replay = _rmse(z_all[~train_mask], targets[~train_mask])

    if len(dataset.class_names) == 2:
        return _plot_binary_projection_fit(
            dataset=dataset,
            z_all=z_all,
            labels=labels,
            train_mask=train_mask,
            prototypes=prototypes,
            rmse_train=rmse_train,
            rmse_replay=rmse_replay,
            output_dir=output_dir,
            formats=formats,
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    for class_index, class_name in enumerate(dataset.class_names):
        mask = labels == class_index
        ax.scatter(
            z_all[mask & train_mask, 0],
            z_all[mask & train_mask, 1],
            color=COLORS[class_index],
            s=38,
            alpha=0.82,
            label=f"{class_name} train",
        )
        ax.scatter(
            z_all[mask & ~train_mask, 0],
            z_all[mask & ~train_mask, 1],
            facecolors="none",
            edgecolors=COLORS[class_index],
            s=42,
            linewidth=1.2,
            label=f"{class_name} replay",
        )
        ax.scatter(
            prototypes[class_index, 0],
            prototypes[class_index, 1],
            marker="*",
            s=250,
            color=COLORS[class_index],
            edgecolor="#0f172a",
            linewidth=0.8,
        )
    ax.axhline(0, color="#cbd5e1", linewidth=1)
    ax.axvline(0, color="#cbd5e1", linewidth=1)
    ax.set_title(f"{_format_dataset_label(dataset.subject)}: 2-D Projection Fit")
    ax.set_xlabel("Projection dimension 1")
    ax.set_ylabel("Projection dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    ax.text(
        0.02,
        0.02,
        f"Train RMSE: {rmse_train:.3f}\nReplay RMSE: {rmse_replay:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"),
    )
    return _save_figure(fig, output_dir, "bciciv_projection_fit", formats)


def _plot_binary_projection_fit(
    dataset: BCICIVFeatures,
    z_all: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    prototypes: np.ndarray,
    rmse_train: float,
    rmse_replay: float,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    projection = z_all[:, 0]
    boundary = float(prototypes[:, 0].mean())
    rng = np.random.default_rng(5)
    jitter = rng.normal(scale=0.035, size=len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    ax = axes[0]
    for class_index, class_name in enumerate(dataset.class_names):
        mask = labels == class_index
        y_base = np.full(np.sum(mask), class_index, dtype=float)
        train_points = mask & train_mask
        replay_points = mask & ~train_mask
        ax.scatter(
            projection[train_points],
            np.full(np.sum(train_points), class_index) + jitter[train_points],
            color=COLORS[class_index],
            s=38,
            alpha=0.82,
            label=f"{class_name} train",
        )
        ax.scatter(
            projection[replay_points],
            np.full(np.sum(replay_points), class_index) + jitter[replay_points],
            facecolors="none",
            edgecolors=COLORS[class_index],
            s=42,
            linewidth=1.2,
            label=f"{class_name} replay",
        )
        ax.axvline(
            prototypes[class_index, 0],
            color=COLORS[class_index],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
        )
    ax.axvline(boundary, color="#334155", linestyle="--", linewidth=1.5, label="prototype midpoint")
    ax.set_yticks(np.arange(len(dataset.class_names)))
    ax.set_yticklabels([name.capitalize() for name in dataset.class_names])
    ax.set_xlabel("Projection dimension 1")
    ax.set_title("Binary Discriminant Projection")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="best")
    ax.text(
        0.02,
        0.02,
        f"Train RMSE: {rmse_train:.3f}\nReplay RMSE: {rmse_replay:.3f}\nDim 2 is near zero by binary target design.",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"),
    )

    ax = axes[1]
    bins = np.linspace(np.percentile(projection, 2), np.percentile(projection, 98), 18)
    for class_index, class_name in enumerate(dataset.class_names):
        mask = labels == class_index
        ax.hist(
            projection[mask & train_mask],
            bins=bins,
            alpha=0.55,
            color=COLORS[class_index],
            label=f"{class_name} train",
        )
        ax.hist(
            projection[mask & ~train_mask],
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=COLORS[class_index],
            label=f"{class_name} replay",
        )
    ax.axvline(boundary, color="#334155", linestyle="--", linewidth=1.5)
    ax.set_title("Projection Distribution")
    ax.set_xlabel("Projection dimension 1")
    ax.set_ylabel("Trial count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{_format_dataset_label(dataset.subject)}: Binary Projection Fit",
        fontsize=14,
        weight="bold",
    )
    return _save_figure(fig, output_dir, "bciciv_projection_fit", formats)


def plot_online_replay(
    result: dict[str, object],
    reject_comparison: list[tuple[str, dict[str, object]]],
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    dataset = _dataset(result)
    metrics = result["metrics"]
    config = result["config"]
    if not isinstance(config, PipelineBuildConfig):
        raise TypeError("config result is not PipelineBuildConfig")
    outputs = metrics["outputs"]
    correct = np.asarray(
        [
            float((not output.rejected) and output.predicted_index == int(y))
            for output, y in zip(outputs, dataset.labels[config.n_train :])
        ],
        dtype=float,
    )
    rejected = np.asarray([float(output.rejected) for output in outputs], dtype=float)
    confidence = np.asarray([output.confidence for output in outputs], dtype=float)
    margin = np.asarray([output.margin for output in outputs], dtype=float)
    trials = np.arange(1, len(correct) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    axes[0, 0].plot(trials, np.cumsum(correct) / trials, color="#2563eb", linewidth=2, label="cumulative")
    axes[0, 0].plot(trials, _rolling_mean(correct, 15), color="#f97316", linewidth=2, label="rolling 15")
    axes[0, 0].set_title("Replay Accuracy")
    axes[0, 0].set_xlabel("Replay trial")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1.04)
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(trials, confidence, color="#16a34a", label="confidence")
    axes[0, 1].plot(trials, margin, color="#7c3aed", label="margin")
    axes[0, 1].axhline(config.reject_threshold, color="#64748b", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("Confidence and Margin")
    axes[0, 1].set_xlabel("Replay trial")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].set_ylim(0, 1.04)
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    reject_max = 0.0
    for idx, (label, comparison_result) in enumerate(reject_comparison):
        comparison_metrics = comparison_result["metrics"]
        comparison_outputs = comparison_metrics["outputs"]
        comparison_rejected = np.asarray(
            [float(output.rejected) for output in comparison_outputs],
            dtype=float,
        )
        rolling = _rolling_mean(comparison_rejected, 15)
        reject_max = max(reject_max, float(np.max(rolling)))
        axes[1, 0].plot(
            np.arange(1, len(rolling) + 1),
            rolling,
            color=COLORS[idx],
            linewidth=2,
            label=f"{label}, total={comparison_rejected.mean():.2f}",
        )
    axes[1, 0].set_title("Rolling Reject Rate by Policy")
    axes[1, 0].set_xlabel("Replay trial")
    axes[1, 0].set_ylabel("Reject rate, rolling 15")
    axes[1, 0].set_ylim(0, max(0.08, reject_max + 0.03))
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(fontsize=8, loc="upper left")

    pipeline = result["pipeline"]
    if hasattr(pipeline.selector, "state"):
        values = pipeline.selector.state.values
        top = np.argsort(values)[-8:][::-1]
        axes[1, 1].bar(np.arange(len(top)), values[top], color="#38bdf8", edgecolor="#0f172a")
        axes[1, 1].set_xticks(np.arange(len(top)))
        axes[1, 1].set_xticklabels([f"W{idx}" for idx in top])
        axes[1, 1].set_title("Final Learned Candidate Values")
        axes[1, 1].set_ylabel("Reward estimate")
        axes[1, 1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"BCICIV_1_asc {_format_dataset_label(dataset.subject)} Online Replay",
        fontsize=14,
        weight="bold",
    )
    return _save_figure(fig, output_dir, "bciciv_online_replay", formats)


def plot_subject_summary(
    results: list[dict[str, object]],
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    subjects = [f"ds1{result['subject']}" for result in results]
    command = np.asarray([result["metrics"]["command_accuracy"] for result in results], dtype=float)
    balanced = np.asarray(
        [result["metrics"]["balanced_command_accuracy"] for result in results],
        dtype=float,
    )
    reject = np.asarray([result["metrics"]["reject_rate"] for result in results], dtype=float)
    x = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, command, width=width, color="#38bdf8", edgecolor="#0f172a", label="command")
    ax.bar(x + width / 2, balanced, width=width, color="#a78bfa", edgecolor="#0f172a", label="balanced")
    ax2 = ax.twinx()
    ax2.plot(x, reject, color="#dc2626", marker="o", linewidth=1.8)
    ax2.set_ylabel("Reject rate", color="#dc2626")
    ax2.tick_params(axis="y", colors="#dc2626")
    ax2.set_ylim(0, max(0.05, reject.max() * 1.8 + 0.01))
    ax.set_title(
        f"BCICIV_1_asc Subject Summary, mean command={command.mean():.3f} +/- {command.std():.3f}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    return _save_figure(fig, output_dir, "bciciv_subject_summary", formats)


def plot_confusion_matrix(
    result: dict[str, object],
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    dataset = _dataset(result)
    metrics = result["metrics"]
    matrix = metrics["confusion"]
    labels = tuple(name.capitalize() for name in dataset.class_names) + ("Reject",)
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_title(
        f"{_format_dataset_label(dataset.subject)} Confusion\n"
        f"cmd acc={metrics['command_accuracy']:.3f}, reject={metrics['reject_rate']:.3f}"
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(dataset.class_names)))
    ax.set_yticklabels([name.capitalize() for name in dataset.class_names])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > matrix.max() * 0.55 else "#0f172a"
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save_figure(fig, output_dir, "bciciv_confusion_matrix", formats)


def _dataset(result: dict[str, object]) -> BCICIVFeatures:
    dataset = result["dataset"]
    if not isinstance(dataset, BCICIVFeatures):
        raise TypeError("dataset result is not BCICIVFeatures")
    return dataset


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str = "#334155",
    fontsize: float = 9.5,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _section_band(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    face: str,
) -> None:
    band = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=0.9,
        edgecolor="#cbd5e1",
        facecolor=face,
        alpha=0.85,
    )
    ax.add_patch(band)
    ax.text(
        x + 0.18,
        y + h - 0.23,
        label,
        ha="left",
        va="center",
        fontsize=9,
        weight="bold",
        color="#334155",
    )


def _draw_box_chain(
    ax: plt.Axes,
    boxes: list[tuple[str, float, float, float, float, str]],
) -> None:
    for text, x, y, w, h, color in boxes:
        _rounded_box(ax, x, y, w, h, text, color, fontsize=8.4)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + boxes[i][3]
        y0 = boxes[i][2] + boxes[i][4] / 2
        x1 = boxes[i + 1][1]
        y1 = boxes[i + 1][2] + boxes[i + 1][4] / 2
        _arrow(ax, x0 + 0.06, y0, x1 - 0.06, y1)


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#334155",
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def _run_requested_dataset(
    data_dir: Path,
    subject: str,
    config: PipelineBuildConfig,
    band: tuple[float, float],
    window: tuple[float, float],
    channels: tuple[str, ...],
    n_train_per_subject: int,
    warmup_trials: int,
) -> dict[str, object]:
    if subject == "pooled":
        return run_pooled(
            data_dir=data_dir,
            config=config,
            band=band,
            window=window,
            channels=channels,
            n_train_per_subject=n_train_per_subject,
            warmup_trials_per_subject=warmup_trials,
        )
    return run_subject(
        data_dir=data_dir,
        subject=subject,
        config=config,
        band=band,
        window=window,
        channels=channels,
        warmup_trials=warmup_trials,
    )


def _short_feature_names(names: tuple[str, ...]) -> list[str]:
    return [name.split("_")[0] for name in names]


def _format_dataset_label(subject: str) -> str:
    if subject.startswith("pooled_"):
        return f"Pooled subjects {subject.removeprefix('pooled_')}"
    return f"Subject ds1{subject}"


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out[i] = values[start : i + 1].mean()
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: tuple[str, ...],
) -> list[Path]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Dataset/BCICIV_1_asc")
    parser.add_argument("--output-dir", default="artifacts/figures/bciciv_1_asc")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--subject", choices=SUBJECTS + ("pooled",), default="pooled")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--warmup-trials", type=int, default=120)
    parser.add_argument("--channels", default=",".join(DEFAULT_MOTOR_CHANNELS))
    parser.add_argument("--band-low", type=float, default=DEFAULT_BAND[0])
    parser.add_argument("--band-high", type=float, default=DEFAULT_BAND[1])
    parser.add_argument("--window-start", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-end", type=float, default=DEFAULT_WINDOW[1])
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument("--candidate-noise", type=float, default=0.02)
    parser.add_argument(
        "--library-kind",
        choices=("perturb", "bootstrap", "mixed"),
        default="perturb",
    )
    parser.add_argument("--selector", choices=("bandit", "confidence", "fusion"), default="fusion")
    parser.add_argument("--reject-threshold", type=float, default=0.34)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
