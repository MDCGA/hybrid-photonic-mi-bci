"""Generate explanatory, non-experimental figures for the technical report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "figure" / "competition"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"blue": "#2f6db0", "green": "#3f8c65", "orange": "#d9822b", "red": "#c65353", "gray": "#667085"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def signal_concepts() -> None:
    rng = np.random.default_rng(7)
    time = np.linspace(-1.0, 4.0, 1250)
    envelope = np.where(time < 0.0, 1.0, 0.45 + 0.55 * np.exp(-np.maximum(time, 0.0) / 1.4))
    eeg = envelope * np.sin(2 * np.pi * 10 * time) + 0.22 * np.sin(2 * np.pi * 20 * time) + 0.18 * rng.normal(size=time.size)
    mu_power = 1.0 - 0.46 * np.exp(-((time - 1.7) / 0.85) ** 2)
    beta_power = 1.0 - 0.28 * np.exp(-((time - 2.0) / 0.95) ** 2)

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.3), sharex=True, gridspec_kw={"hspace": 0.18})
    axes[0].plot(time, eeg, color=COLORS["blue"], linewidth=0.85)
    axes[0].axvspan(0, 3, color="#f5e7d4", alpha=0.72, label="运动想象时段")
    axes[0].axvline(0, color=COLORS["orange"], linewidth=1.2, linestyle="--")
    axes[0].text(0.05, 1.33, "提示", color=COLORS["orange"], fontsize=10)
    axes[0].set_ylabel("归一化 EEG")
    axes[0].set_title("运动想象期间的脑电节律变化示意（非实测数据）", fontsize=11, weight="bold")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].grid(alpha=0.2)

    axes[1].plot(time, mu_power, label="mu 8-12 Hz", color=COLORS["blue"], linewidth=2.2)
    axes[1].plot(time, beta_power, label="beta 16-28 Hz", color=COLORS["green"], linewidth=2.2)
    axes[1].axhline(1.0, color=COLORS["gray"], linewidth=0.9, linestyle=":")
    axes[1].annotate("事件相关去同步化（ERD）", xy=(1.65, 0.55), xytext=(2.35, 0.47),
                     arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, color=COLORS["red"], fontsize=9)
    axes[1].set_xlabel("相对提示时间（秒）")
    axes[1].set_ylabel("相对频带功率")
    axes[1].set_ylim(0.35, 1.12)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="lower right", frameon=False, ncol=2, fontsize=9)
    save(fig, "eeg_mi_signal_concepts.png")


def box(ax, xy, width, height, text, color, textcolor="white", fontsize=10):
    rect = Rectangle(xy, width, height, linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.96)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", color=textcolor, fontsize=fontsize, weight="bold")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#475467"))


def mi_bci_task_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(6, 3.62, "运动想象 EEG-BCI：从决策窗口到控制命令", ha="center", fontsize=14, weight="bold")
    stages = [
        (0.35, 1.35, 2.15, 1.25, "EEG 决策窗口\n多通道 x 时间", COLORS["blue"]),
        (3.25, 1.35, 2.15, 1.25, "FBCSP + 小型嵌入\n提取判别表示", COLORS["green"]),
        (6.15, 1.35, 2.15, 1.25, "经验库候选\n光计算扫描", COLORS["orange"]),
        (9.05, 1.35, 2.55, 1.25, "安全输出\n左 / 右 / 脚 / 拒识", COLORS["red"]),
    ]
    for x, y, width, height, text, color in stages:
        box(ax, (x, y), width, height, text, color, fontsize=10)
    for start, end in [((2.55, 1.98), (3.2, 1.98)), ((5.45, 1.98), (6.1, 1.98)), ((8.35, 1.98), (9.0, 1.98))]:
        arrow(ax, start, end)
    ax.text(6, 0.62, "目标：识别运动想象意图；低质量或低置信度窗口不强制输出命令", ha="center", fontsize=9.5, color="#344054")
    save(fig, "mi_bci_task_overview.png")


def specialization_logic() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6, 4.58, "从 EEG 个体差异到会话特化与光计算候选扫描", ha="center", fontsize=14, weight="bold")
    box(ax, (0.35, 1.8), 2.25, 1.45, "跨被试 / 跨会话差异\n频段、空间激活、策略", COLORS["blue"], fontsize=9.5)
    box(ax, (3.25, 1.8), 2.25, 1.45, "少量校准窗口\n估计当前会话特征", COLORS["green"], fontsize=9.5)
    box(ax, (6.15, 1.8), 2.25, 1.45, "经验库特化\n选择并加权候选头", COLORS["orange"], fontsize=9.5)
    box(ax, (9.05, 1.8), 2.55, 1.45, "低位宽 photonic scan\n融合后命令 / 拒识", COLORS["red"], fontsize=9.5)
    for start, end in [((2.65, 2.52), (3.2, 2.52)), ((5.55, 2.52), (6.1, 2.52)), ((8.45, 2.52), (9.0, 2.52))]:
        arrow(ax, start, end)
    ax.text(7.25, 1.05, "经验库保存可复用的特化模型与统计，不是普通数据缓存", ha="center", fontsize=9.2, color="#344054")
    ax.text(7.25, 0.56, "FBCSP 保持可解释主干；小型 MLP 只改善嵌入空间", ha="center", fontsize=9.2, color="#344054")
    save(fig, "specialization_logic.png")


def fbcsp_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, (0.25, 2.0), 1.55, 1.05, "EEG 窗口\nX: 8 x T", COLORS["blue"])
    box(ax, (2.35, 2.0), 1.75, 1.05, "滤波器组\n6 个频带", COLORS["green"])
    box(ax, (4.7, 2.0), 1.75, 1.05, "OVR CSP\nW(4x8) X(8xT)", COLORS["orange"], fontsize=9)
    box(ax, (7.05, 2.0), 1.55, 1.05, "对数方差\n72 维特征", COLORS["red"])
    box(ax, (9.2, 2.0), 1.55, 1.05, "Fisher 选择\n32 维特征", COLORS["blue"])
    box(ax, (11.35, 2.0), 2.05, 1.05, "编码器 / 候选\n线性头", COLORS["green"])
    for start, end in [((1.82, 2.52), (2.32, 2.52)), ((4.13, 2.52), (4.67, 2.52)), ((6.48, 2.52), (7.02, 2.52)), ((8.63, 2.52), (9.17, 2.52)), ((10.78, 2.52), (11.32, 2.52))]:
        arrow(ax, start, end)
    ax.text(5.55, 1.42, "每个频带、每个类别：Z = W X", ha="center", fontsize=10, color="#344054")
    ax.text(5.55, 0.95, "线性矩阵乘：4 x 8 乘以 8 x T", ha="center", fontsize=9, color=COLORS["orange"])
    ax.text(12.37, 1.42, "仅线性步骤可作为 MVM 映射候选", ha="center", fontsize=9, color="#344054")
    ax.text(7.0, 4.36, "FBCSP 将多通道节律转为紧凑、可解释的特征", ha="center", fontsize=13, weight="bold")
    save(fig, "fbcsp_matrix_flow.png")


def photonic_mapping() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.3))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(7.5, 7.45, "候选线性头：3 x 33 增广矩阵映射为 2 x 8 光计算 tile", ha="center", fontsize=13, weight="bold")

    # Augmented input vector.
    ax.text(1.3, 6.55, "增广输入", ha="center", fontsize=10, weight="bold")
    for i in range(5):
        width = 0.43 if i == 4 else 1.15
        x = 0.2 + i * 0.57 if i == 4 else 0.2 + i * 1.15
        if i < 4:
            x = 0.2 + i * 1.15
        else:
            x = 4.8
        rect = Rectangle((x, 5.85), width, 0.5, facecolor="#dbeafe", edgecolor=COLORS["blue"])
        ax.add_patch(rect)
        label = f"x[{8*i}:{8*i+8}]" if i < 4 else "1"
        ax.text(x + width / 2, 6.10, label, ha="center", va="center", fontsize=8)
    ax.text(2.55, 5.42, "32 维特征 + 偏置 = 33 列 = 5 个列 tile", ha="center", fontsize=9, color="#344054")

    # Weight grid.
    origin_x, origin_y = 6.1, 1.0
    cell_w, cell_h = 1.05, 1.15
    grid_colors = ["#b7d4f3", "#d9ead3", "#f8d7a8", "#f1c0c0", "#d9d2e9"]
    for row_block in range(2):
        for col_block in range(5):
            x = origin_x + col_block * cell_w
            y = origin_y + (1 - row_block) * cell_h
            ax.add_patch(Rectangle((x, y), cell_w - 0.06, cell_h - 0.06, facecolor=grid_colors[col_block], edgecolor="#475467", linewidth=1.1))
            row_text = "第 0-1 行" if row_block == 0 else "第 2 行\n填充"
            ax.text(x + (cell_w - 0.06) / 2, y + (cell_h - 0.06) / 2, f"{row_text}\n列 {8*col_block}-{min(8*col_block+7,32)}", ha="center", va="center", fontsize=7.4)
    ax.text(8.67, 3.62, "一个候选权重矩阵 [A | b]：3 x 33", ha="center", fontsize=10, weight="bold")
    ax.text(8.67, 0.45, "2 个行块 x 5 个列块 = 每候选 10 次 tile 调用", ha="center", fontsize=9, color="#344054")
    arrow(ax, (5.25, 6.08), (6.0, 3.0))
    ax.text(5.58, 4.8, "写入\n权重", ha="center", fontsize=9, color=COLORS["orange"])

    # Candidate replication and outputs.
    ax.add_patch(Rectangle((12.15, 2.1), 1.7, 1.65, facecolor="#eef2f6", edgecolor="#475467", linewidth=1.1))
    ax.text(13.0, 3.30, "8 个检索到的\n候选模型", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(13.0, 2.55, "8 x 10 = 80\n次 tile 调用", ha="center", va="center", fontsize=9, color=COLORS["red"])
    arrow(ax, (11.4, 2.65), (12.1, 2.65))
    arrow(ax, (13.9, 2.9), (14.6, 2.9))
    box(ax, (14.65, 2.25), 0.30, 1.25, "s\n3", COLORS["green"], fontsize=9)
    ax.text(13.0, 1.35, "数字端：部分和累加\nsoftmax、融合、拒识", ha="center", fontsize=8.0, color="#344054")
    save(fig, "photonic_tile_mapping.png")


def end_to_end_inference_chain() -> None:
    """Draw the report's ten-step signal-to-command chain at reading scale."""

    fig, ax = plt.subplots(figsize=(14.2, 5.8))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(7.6, 6.75, "光电混合 MI-BCI：从 EEG 采集到最终指令", ha="center", fontsize=17, weight="bold")
    ax.text(7.6, 6.3, "由十步在线推理链归并而成的五个清晰阶段", ha="center", fontsize=10.5, color="#475467")

    stages = [
        (0.25, "1-2", "采集与对齐", "Cyton 计数 r(t)\n微伏换算与通道映射", "#e7f0fb", COLORS["blue"]),
        (3.25, "3-5", "数字预处理", "滑动窗口与质量门控\n滤波器组与 CAR", "#eaf6ee", COLORS["green"]),
        (6.25, "6-8", "特征与\n个性化", "FBCSP -> 32 维特征\n嵌入 h 与 top-K 经验库", "#fff4e4", COLORS["orange"]),
        (9.25, "9", "Gazelle 候选\n扫描", "[A | b] [h ; 1]\n2 x 8 tile 矩阵乘", "#f5ebfb", "#8250a5"),
        (12.25, "10", "融合与\n安全输出", "融合与置信度判定\n左 / 右 / 脚 / 拒识", "#fbecef", COLORS["red"]),
    ]
    y, w, h = 2.15, 2.55, 3.25
    for x, step, title, detail, fill, edge in stages:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.8, zorder=2))
        ax.text(x + 0.2, y + h - 0.32, step, ha="left", va="top", fontsize=11, weight="bold", color=edge)
        ax.text(x + w / 2, y + 2.25, title, ha="center", va="center", fontsize=11.2, weight="bold", color="#1d2939", linespacing=1.1)
        ax.text(x + w / 2, y + 1.35, detail, ha="center", va="center", fontsize=9.3, color="#344054", linespacing=1.55)

    for index in range(4):
        x0 = stages[index][0] + w
        x1 = stages[index + 1][0]
        ax.add_patch(FancyArrowPatch((x0 + 0.05, y + h / 2), (x1 - 0.05, y + h / 2), arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color="#475467", zorder=1))
    ax.text(2.95, y + h / 2 + 0.28, "u(t)", ha="center", fontsize=8.5, color="#475467")
    ax.text(5.95, y + h / 2 + 0.28, "有效窗口", ha="center", fontsize=8.5, color="#475467")
    ax.text(8.95, y + h / 2 + 0.28, "候选头与 h", ha="center", fontsize=8.5, color="#475467")
    ax.text(11.95, y + h / 2 + 0.28, "得分", ha="center", fontsize=8.5, color="#475467")
    ax.add_patch(FancyArrowPatch((4.55, y), (13.55, 1.36), arrowstyle="-|>", mutation_scale=14, linewidth=1.55, linestyle="--", color="#667085"))
    ax.text(8.9, 1.08, "信号质量不合格 -> 提前拒识（不分类，不写回经验库）", ha="center", fontsize=9.2, color="#475467")
    ax.text(7.6, 0.45, "实线：只分类有效窗口。紫色阶段：候选线性矩阵乘是 Gazelle 的首个硬件映射目标。", ha="center", fontsize=9, color="#475467")
    save(fig, "end_to_end_inference_chain.png")


if __name__ == "__main__":
    signal_concepts()
    mi_bci_task_overview()
    specialization_logic()
    fbcsp_flow()
    photonic_mapping()
    end_to_end_inference_chain()
