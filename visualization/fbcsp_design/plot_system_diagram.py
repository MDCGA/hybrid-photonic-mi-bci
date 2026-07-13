"""Draw the detailed implementation block diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import save_figure, set_style


def plot(output_dir: Path, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    set_style()
    fig, ax = plt.subplots(figsize=(18.2, 10.2))
    ax.set_axis_off()
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 10.2)
    ax.text(
        9.1,
        9.82,
        "Implemented Hybrid Photonic MI-BCI Pipeline on BCICIV_1_asc",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    _band(ax, 0.2, 8.55, 17.8, 0.95, "Dataset and Train/Test Protocol", "#eff6ff")
    _box(ax, 0.45, 8.75, 2.4, 0.5, "BCICIV_1_asc\na-g ASCII files")
    _box(ax, 3.15, 8.75, 2.55, 0.5, "Pool local labels\nleft/right/foot")
    _box(ax, 6.0, 8.75, 2.45, 0.5, "Train: 120/file\nReplay: 80/file")
    _box(ax, 8.75, 8.75, 3.0, 0.5, "Calibration query\n6 replay windows/file")
    _box(ax, 12.05, 8.75, 2.75, 0.5, "Online evaluation\nexcludes calibration")
    _box(ax, 15.1, 8.75, 2.45, 0.5, "Outputs\nleft/right/foot/reject")
    _chain(ax, [2.85, 5.7, 8.45, 11.75, 14.8], 9.0)

    _band(ax, 0.2, 6.95, 17.8, 1.05, "Shared FBCSP Feature Layer", "#f0fdf4")
    shared = [
        (0.45, "CAR via SignalOps\n8 motor channels"),
        (2.75, "Trial window\nmarker + 1.0-4.0 s"),
        (5.25, "3rd-order SOS filter bank\nvia SignalOps"),
        (8.0, "OVR CSP covariance +\nprojection via MatrixOps"),
        (10.8, "Log-variance\n72D raw FBCSP"),
        (13.35, "Fisher selection\n32D"),
        (15.65, "Standardizer affine\nvia MatrixOps"),
    ]
    for x, label in shared:
        _box(ax, x, 7.2, 2.05, 0.55, label, "#dcfce7")
    _chain(ax, [2.5, 5.0, 7.75, 10.55, 13.1, 15.4], 7.48)

    _band(ax, 0.2, 5.05, 8.55, 1.25, "Reference Baseline", "#fff7ed")
    _box(ax, 0.55, 5.38, 2.35, 0.58, "FBCSP 32D\ntrain/replay")
    _box(ax, 3.25, 5.38, 2.45, 0.58, "Shrinkage LDA\nA x + b via backend")
    _box(ax, 6.05, 5.38, 2.15, 0.58, "Softmax + reject\nmetrics")
    _chain(ax, [2.95, 5.75], 5.67)

    _band(ax, 9.05, 5.05, 8.95, 1.25, "Small-Network Embedding Line", "#f5f3ff")
    _box(ax, 9.35, 5.38, 2.35, 0.58, "FBCSP 32D\nLayerNorm input")
    _box(ax, 12.05, 5.38, 2.7, 0.58, "MLP encoder Linear\n32 -> 64 -> 32")
    _box(ax, 15.1, 5.38, 2.45, 0.58, "Embedding h\nclassifier via backend")
    _chain(ax, [11.75, 14.8], 5.67)

    _band(ax, 0.2, 2.6, 17.8, 1.7, "Mainline: Experience Library Retrieval + Photonic Candidate Scan", "#f8fafc")
    main = [
        (0.45, "Experience library\n64 bootstrap heads"),
        (2.9, "Anchor heads\nMLP classifier\n+ embedding LDA"),
        (5.35, "Calibration query\ncentroid dot via MatrixOps"),
        (7.8, "Select top-8\ncalibration-aware weights"),
        (10.25, "Candidate heads\nA_i [h,1] via MatrixOps"),
        (12.7, "TiledMVMBackend\n2 x 8 primitive"),
        (15.15, "Probability fusion via\nMatrixOps; reject digital"),
    ]
    for x, label in main:
        _box(ax, x, 2.98, 2.1, 0.74, label, "#e0f2fe")
    _chain(ax, [2.65, 5.1, 7.55, 10.0, 12.45, 14.9], 3.35)

    _band(ax, 0.2, 0.55, 17.8, 1.25, "Photonic Boundary and Measured Work", "#fffbeb")
    _box(ax, 0.55, 0.88, 3.0, 0.56, "Unified handoff\nMatrixOps + SignalOps")
    _box(ax, 3.95, 0.88, 3.2, 0.56, "MatrixOps routed\nCSP/LDA/MLP/std/bias/A_i/fusion")
    _box(ax, 7.55, 0.88, 3.2, 0.56, "SignalOps routed\nCAR + SOS filter bank")
    _box(ax, 11.15, 0.88, 3.15, 0.56, "Digital side\ntrain/backprop/variance/log/reject")
    _box(ax, 14.75, 0.88, 2.65, 0.56, "Saved artifacts\nmetrics + figures")
    _chain(ax, [3.65, 7.25, 10.85, 14.45], 1.16)

    return save_figure(fig, output_dir, "system_block_diagram_detailed", formats)


def _band(ax, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=1,
        edgecolor="#cbd5e1",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + 0.2, y + h - 0.18, label, ha="left", va="top", fontsize=10, weight="bold")


def _box(ax, x: float, y: float, w: float, h: float, label: str, color: str = "#ffffff") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1,
        edgecolor="#94a3b8",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.3)


def _arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color="#475569",
        )
    )


def _chain(ax, xs: list[float], y: float) -> None:
    for x in xs:
        _arrow(ax, x, y, x + 0.28, y)


if __name__ == "__main__":
    plot(Path("artifacts/figures/fbcsp_design/system"), ("png", "pdf"))
