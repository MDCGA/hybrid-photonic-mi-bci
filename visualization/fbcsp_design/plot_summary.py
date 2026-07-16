"""Plot final comparison summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .common import bar_labels, load_json, save_figure, set_style
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from visualization.fbcsp_design.common import bar_labels, load_json, save_figure, set_style


def plot(metrics_dir: Path, output_dir: Path, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_style()
    summary = load_json(metrics_dir / "summary.json")
    rows = summary["rows"]
    labels = ["LDA", "MLP", "MLP + Lib + scan"]
    command = np.asarray([row["command_accuracy"] for row in rows], dtype=float)
    balanced = np.asarray([row["balanced_command_accuracy"] for row in rows], dtype=float)
    accepted = np.asarray([row["accepted_accuracy"] for row in rows], dtype=float)
    reject = np.asarray([row["reject_rate"] for row in rows], dtype=float)
    tiles = np.asarray([row.get("tile_evaluations_per_window", 0) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.24
    bars1 = axes[0].bar(x - width, command, width, label="command", color="#2563eb")
    bars2 = axes[0].bar(x, balanced, width, label="balanced", color="#16a34a")
    bars3 = axes[0].bar(x + width, accepted, width, label="accepted", color="#f97316")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0.55, 0.86)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy comparison")
    axes[0].legend()
    for bars in (bars1, bars2, bars3):
        bar_labels(axes[0], bars)

    ax2 = axes[1]
    bars_r = ax2.bar(x - width / 2, reject, width, color="#dc2626", label="reject rate")
    ax2.set_xticks(x, labels, rotation=15, ha="right")
    ax2.set_ylim(0, max(0.10, reject.max() * 1.4))
    ax2.set_ylabel("Reject rate")
    ax2.set_title("Reject and photonic work")
    ax_tiles = ax2.twinx()
    bars_t = ax_tiles.bar(x + width / 2, tiles, width, color="#64748b", label="tile evals/window")
    ax_tiles.set_ylabel("2 x 8 tile evaluations / window")
    ax_tiles.set_ylim(0, max(80, tiles.max() * 1.25))
    bar_labels(ax2, bars_r)
    bar_labels(ax_tiles, bars_t, "{:.0f}")
    handles = [bars_r, bars_t]
    ax2.legend(handles, [item.get_label() for item in handles], loc="upper left")
    fig.suptitle("Final Design-Line Comparison on BCICIV_1_asc", fontsize=14, weight="bold")
    return save_figure(fig, output_dir, "design_line_summary", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="artifacts/metrics/fbcsp_design")
    parser.add_argument("--output-dir", default="artifacts/figures/fbcsp_design/summary")
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    generated = plot(Path(args.metrics_dir), Path(args.output_dir), formats)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
