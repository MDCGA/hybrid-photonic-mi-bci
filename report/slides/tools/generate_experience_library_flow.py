from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "assets" / "experience_library_selection_flow.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#3478b8"
GREEN = "#3c9064"
ORANGE = "#e58a2b"
PURPLE = "#7b58aa"
RED = "#ce5050"
INK = "#203047"
MUTED = "#667487"


def box(ax, x, y, w, h, title, detail, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor=color, edgecolor=color, linewidth=1.2,
    )
    ax.add_patch(patch)
    title_y = 0.70 if "\n" not in title else 0.73
    detail_y = 0.31 if "\n" not in title else 0.24
    ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", color="white", fontsize=11.5, weight="bold", linespacing=0.95)
    ax.text(x + w / 2, y + h * detail_y, detail, ha="center", va="center", color="white", fontsize=8.3, linespacing=1.18)


def arrow(ax, start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color=MUTED))
    if label:
        if abs(start[0] - end[0]) < 0.05:
            ax.text(start[0] + 0.48, (start[1] + end[1]) / 2, label, ha="left", va="center", fontsize=8.2, color=MUTED)
        else:
            ax.text((start[0] + end[0]) / 2, start[1] + 0.18, label, ha="center", fontsize=8.2, color=MUTED)


def main():
    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(0.25, 3.95, "经验库会话特化：从 66 个候选头到当前会话模型组合", fontsize=16, weight="bold", color=INK, va="top")

    # Candidate-library branch.
    box(ax, 0.35, 2.15, 2.15, 1.0, "2 个 Anchor Heads", "全局 MLP Head\n全局 Embedding-LDA", BLUE)
    box(ax, 0.35, 0.75, 2.15, 1.0, "64 个 Bootstrap\nHeads", "训练集重复采样\n多样化 LDA Heads", GREEN)

    # Merge and calibration.
    box(ax, 3.15, 1.45, 1.85, 1.0, "经验库", "共 66 个\n候选线性头", ORANGE)
    box(ax, 5.65, 1.45, 1.9, 1.0, "42 个校准窗口", "目标会话 EEG\n不计入最终评估", BLUE)
    box(ax, 8.2, 1.45, 2.05, 1.0, "候选评分与检索", "距离 · 训练质量\n校准准确率 · 置信度", PURPLE)
    box(ax, 10.9, 1.45, 1.75, 1.0, "Top-K = 8", "2 Anchors +\n6 Bootstrap Heads", RED)

    arrow(ax, (2.52, 2.65), (3.10, 2.05), "合并")
    arrow(ax, (2.52, 1.25), (3.10, 1.85))
    arrow(ax, (5.03, 1.95), (5.60, 1.95), "候选集合")
    arrow(ax, (7.58, 1.95), (8.15, 1.95), "会话查询")
    arrow(ax, (10.28, 1.95), (10.85, 1.95), "排序选择")

    # Output band.
    ax.add_patch(FancyBboxPatch((7.25, 0.35), 5.4, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor="#eef3f8", edgecolor="#cbd5e1"))
    ax.text(9.95, 0.66, "8 个候选概率按检索权重融合  →  当前会话命令 / 拒识", ha="center", va="center", fontsize=10.5, color=INK, weight="bold")
    arrow(ax, (11.78, 1.42), (11.78, 1.00), "加权融合")

    ax.text(0.35, 0.24, "核心含义：经验库保存候选模型；少量校准用于选择模型组合，不是重新训练整个系统。", fontsize=9.4, color=MUTED)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
