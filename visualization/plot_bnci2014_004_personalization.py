"""Plot BNCI2014_004 three-line design comparison results."""

from __future__ import annotations

import argparse
import json
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
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    accounting = json.loads((metrics_dir / "compute_accounting.json").read_text(encoding="utf-8"))
    arrays = np.load(metrics_dir / "arrays.npz")
    _set_style()
    generated = []
    generated.extend(_plot_line_summary(summary["rows"], output_dir, formats))
    generated.extend(_plot_subject_lines(arrays, output_dir, formats))
    generated.extend(_plot_compute_accounting(accounting["lines"], output_dir, formats))
    print("Generated BNCI2014_004 design-comparison figures:")
    for path in generated:
        print(f"- {path}")


def _plot_line_summary(rows: list[dict[str, object]], output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    labels = [_short_label(str(row["line"])) for row in rows]
    x = np.arange(len(rows))
    command = np.asarray([row["command_accuracy"] for row in rows], dtype=float)
    balanced = np.asarray([row["balanced_command_accuracy"] for row in rows], dtype=float)
    reject = np.asarray([row["reject_rate"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    width = 0.34
    axes[0].bar(x - width / 2, command, width=width, color="#2563eb", label="command")
    axes[0].bar(x + width / 2, balanced, width=width, color="#0f766e", label="balanced")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.45, 0.90)
    axes[0].set_ylabel("Held-out accuracy")
    axes[0].set_title("BNCI2014_004 held-out target accuracy")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for index, value in enumerate(command):
        axes[0].text(index - width / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(x, reject, color="#dc2626", width=0.55)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, max(0.12, float(reject.max()) * 1.4))
    axes[1].set_ylabel("Reject rate")
    axes[1].set_title("Digital reject output")
    for index, value in enumerate(reject):
        axes[1].text(index, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, output_dir, "bnci_design_line_summary", formats)


def _plot_subject_lines(arrays: np.lib.npyio.NpzFile, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    subjects = np.unique(arrays["subjects"])
    lines = list(dict.fromkeys(str(item) for item in arrays["line"]))
    labels = [_short_label(line) for line in lines]
    values = np.zeros((len(subjects), len(lines)), dtype=np.float64)
    for subject_index, subject in enumerate(subjects):
        for line_index, line in enumerate(lines):
            mask = (arrays["subjects"] == subject) & (arrays["line"] == line)
            values[subject_index, line_index] = arrays["command_accuracy"][mask][0]
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    x = np.arange(len(subjects))
    width = 0.22
    colors = ("#64748b", "#2563eb", "#7c3aed")
    for line_index, label in enumerate(labels):
        offset = (line_index - (len(lines) - 1) / 2) * width
        ax.bar(x + offset, values[:, line_index], width=width, color=colors[line_index], label=label)
    ax.set_xticks(x, [f"S{int(subject)}" for subject in subjects])
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("Target subject")
    ax.set_ylabel("Command accuracy")
    ax.set_title("Per-subject comparison on the same held-out target windows")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, output_dir, "bnci_subject_line_comparison", formats)


def _plot_compute_accounting(lines: list[dict[str, object]], output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    labels = [_short_label(str(line["line"])) for line in lines]
    photonic = np.asarray([line["summary"]["linear_macs_photonic"] for line in lines], dtype=float)
    digital = np.asarray([line["summary"]["linear_macs_digital"] for line in lines], dtype=float)
    total_share = np.asarray([line["summary"]["photonic_linear_share"] for line in lines], dtype=float)
    inference_share = np.asarray([line["summary"]["photonic_linear_share_inference"] for line in lines], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    x = np.arange(len(lines))
    scale = 1e9
    axes[0].bar(x, photonic / scale, color="#2563eb", label="MatrixOps + SignalOps photonic")
    axes[0].bar(x, digital / scale, bottom=photonic / scale, color="#f97316", label="Digital linear compute")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Linear compute (GMAC)")
    axes[0].set_title("Forward linear compute")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for index, value in enumerate((photonic + digital) / scale):
        axes[0].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    width = 0.34
    axes[1].bar(x - width / 2, total_share, width=width, color="#0f766e", label="forward")
    axes[1].bar(x + width / 2, inference_share, width=width, color="#7c3aed", label="online inference")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Photonic share")
    axes[1].set_title("Photonic share")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for xpos, values in ((x - width / 2, total_share), (x + width / 2, inference_share)):
        for xi, value in zip(xpos, values):
            axes[1].text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, output_dir, "bnci_compute_accounting_summary", formats)


def _short_label(label: str) -> str:
    if "library" in label:
        return "MLP + Lib + scan"
    if "MLP" in label:
        return "MLP"
    if "LDA" in label:
        return "LDA"
    return label


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
