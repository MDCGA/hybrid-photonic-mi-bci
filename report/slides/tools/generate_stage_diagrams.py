from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "blue": "#2f6da8",
    "green": "#3c8a61",
    "orange": "#df862c",
    "red": "#c94c4c",
    "purple": "#7956a8",
    "ink": "#203047",
    "muted": "#5d6b7b",
}


def draw_stage(filename, title, subtitle, boxes, footer):
    fig, ax = plt.subplots(figsize=(13.0, 2.25))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.25)
    ax.axis("off")
    ax.text(0.22, 2.02, title, fontsize=16, weight="bold", color=COLORS["ink"], va="top")
    ax.text(12.78, 2.00, subtitle, fontsize=9.5, color=COLORS["muted"], ha="right", va="top")

    n = len(boxes)
    left, right, gap = 0.28, 12.72, 0.36
    width = (right - left - gap * (n - 1)) / n
    y, height = 0.68, 0.86
    for idx, (heading, detail, color) in enumerate(boxes):
        x = left + idx * (width + gap)
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            facecolor=color, edgecolor=color, linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.57, heading, ha="center", va="center", fontsize=11, color="white", weight="bold")
        ax.text(x + width / 2, y + 0.27, detail, ha="center", va="center", fontsize=8.4, color="white")
        if idx < n - 1:
            start = (x + width + 0.04, y + height / 2)
            end = (x + width + gap - 0.04, y + height / 2)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#64748b"))
    ax.text(0.28, 0.25, footer, fontsize=9.2, color=COLORS["muted"], va="center")
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def main():
    draw_stage(
        "stage_offline_training.png",
        "阶段一｜离线训练",
        "输入：训练集   输出：冻结模型与经验库",
        [
            ("训练数据", "通道 · 标签 · 事件窗", COLORS["blue"]),
            ("FBCSP 拟合", "滤波器组 · OVR CSP", COLORS["green"]),
            ("特征处理", "Fisher 72→32 · 标准化", COLORS["orange"]),
            ("小型 MLP", "32→64→32 Embedding", COLORS["purple"]),
            ("经验库构建", "2 Anchors + 64 Bootstrap", COLORS["red"]),
        ],
        "边界：仅使用训练集；Evaluation 窗口不参与拟合",
    )
    draw_stage(
        "stage_session_calibration.png",
        "阶段二｜会话校准",
        "输入：少量目标会话窗口   输出：会话部署状态",
        [
            ("校准窗口", "BCICIV：42 个", COLORS["blue"]),
            ("冻结主干前向", "FBCSP + MLP Embedding", COLORS["green"]),
            ("经验库查询", "距离 · 训练质量 · 校准表现", COLORS["orange"]),
            ("Top-K 候选", "K=8 · 融合权重", COLORS["purple"]),
            ("冻结会话状态", "Head Bank · 拒识阈值", COLORS["red"]),
        ],
        "边界：不重新训练 FBCSP/MLP；校准窗口不计入最终评估",
    )
    draw_stage(
        "stage_online_inference.png",
        "阶段三｜在线推理",
        "输入：单个 EEG 决策窗口   输出：命令或拒识",
        [
            ("EEG 窗口", "质量检查 · CAR · 滤波", COLORS["blue"]),
            ("FBCSP Transform", "固定 CSP · 32 维特征", COLORS["green"]),
            ("Embedding", "固定 MLP · 32 维 h", COLORS["orange"]),
            ("Photonic Scan", "8 Heads · 80 Tiles", COLORS["purple"]),
            ("安全输出", "概率融合 · 命令 / 拒识", COLORS["red"]),
        ],
        "边界：单窗口不训练；非线性、融合、控制与拒识保留在数字端",
    )


if __name__ == "__main__":
    main()
