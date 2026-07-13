"""Plot BNCI2014_004 single-subject personalization results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    formats = tuple(fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip())
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(metrics_dir / "arrays.npz")
    _set_style()
    generated = []
    generated.extend(_plot_mean_curves(arrays, output_dir, formats))
    generated.extend(_plot_subject_pairs(arrays, output_dir, formats, k=args.k))
    generated.extend(_plot_gain_heatmap(arrays, output_dir, formats))
    print("Generated BNCI2014_004 personalization figures:")
    for path in generated:
        print(f"- {path}")


def _plot_mean_curves(arrays, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    ks = np.unique(arrays["calibration_trials_per_class"])
    before = []
    cal = []
    after = []
    stderr = []
    for k in ks:
        mask = arrays["calibration_trials_per_class"] == k
        before.append(arrays["before_accuracy"][mask].mean())
        cal.append(arrays["calibration_only_accuracy"][mask].mean())
        values = arrays["experience_after_accuracy"][mask]
        after.append(values.mean())
        stderr.append(values.std(ddof=1) / np.sqrt(mask.sum()))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(ks, before, marker="o", lw=2, color="#64748b", label="before personalization")
    ax.plot(ks, cal, marker="o", lw=2, color="#f97316", label="calibration only")
    ax.errorbar(
        ks,
        after,
        yerr=stderr,
        marker="o",
        lw=2,
        capsize=4,
        color="#2563eb",
        label="experience + calibration",
    )
    ax.set_title("BNCI2014_004: mean target-session accuracy")
    ax.set_xlabel("Calibration trials per class from target session")
    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0.55, 0.84)
    ax.legend(loc="lower right")
    return _save(fig, output_dir, "personalization_mean_curves", formats)


def _plot_subject_pairs(arrays, output_dir: Path, formats: tuple[str, ...], k: int) -> list[Path]:
    mask = arrays["calibration_trials_per_class"] == k
    subjects = arrays["subjects"][mask]
    before = arrays["before_accuracy"][mask]
    after = arrays["experience_after_accuracy"][mask]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(subjects))
    for index in x:
        color = "#16a34a" if after[index] >= before[index] else "#dc2626"
        ax.plot([index, index], [before[index], after[index]], color=color, lw=2.4, alpha=0.85)
    ax.scatter(x, before, color="#64748b", s=52, label="before")
    ax.scatter(x, after, color="#2563eb", s=58, label="experience after")
    ax.set_xticks(x, [f"S{int(s)}" for s in subjects])
    ax.set_ylim(0.45, 0.95)
    ax.set_title(f"Per-subject before/after at {k} calibration trials/class")
    ax.set_xlabel("Target subject")
    ax.set_ylabel("Held-out accuracy")
    ax.legend(loc="lower right")
    return _save(fig, output_dir, f"subject_before_after_k{k}", formats)


def _plot_gain_heatmap(arrays, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    subjects = np.unique(arrays["subjects"])
    ks = np.unique(arrays["calibration_trials_per_class"])
    gain = np.zeros((len(subjects), len(ks)), dtype=np.float64)
    for row, subject in enumerate(subjects):
        for col, k in enumerate(ks):
            mask = (arrays["subjects"] == subject) & (arrays["calibration_trials_per_class"] == k)
            gain[row, col] = arrays["gain_vs_before"][mask][0]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    limit = max(0.05, float(np.abs(gain).max()))
    image = ax.imshow(gain, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(ks)), [str(int(k)) for k in ks])
    ax.set_yticks(np.arange(len(subjects)), [f"S{int(s)}" for s in subjects])
    ax.set_xlabel("Calibration trials per class")
    ax.set_ylabel("Target subject")
    ax.set_title("Experience-library gain vs before personalization")
    for row in range(gain.shape[0]):
        for col in range(gain.shape[1]):
            ax.text(col, row, f"{gain[row, col]:+.2f}", ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy gain")
    return _save(fig, output_dir, "personalization_gain_heatmap", formats)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    saved = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="artifacts/metrics/bnci2014_004_personalization")
    parser.add_argument("--output-dir", default="artifacts/figures/bnci2014_004_personalization")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--k", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
