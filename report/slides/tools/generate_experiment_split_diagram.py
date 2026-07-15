from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "assets"
PNG = OUT / "bciciv_experiment_split.png"
DRAWIO = OUT / "bciciv_experiment_split.drawio"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#203047"
MUTED = "#667487"
BLUE = "#3478B8"
ORANGE = "#E58A2B"
GREEN = "#3C9064"
RED = "#C95757"
LINE = "#CAD5E1"


def rounded(ax, x, y, w, h, face, edge, lw=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.7, color=color))


def card(ax, x, color, label, number, use, note):
    rounded(ax, x, 1.62, 3.38, 2.45, color, color)
    ax.text(x + 1.69, 3.63, label, ha="center", va="center",
            fontsize=13.5, weight="bold", color="white")
    ax.text(x + 1.69, 3.05, number, ha="center", va="center",
            fontsize=25, weight="bold", color="white")
    ax.plot([x + 0.34, x + 3.04], [2.67, 2.67], color="white", alpha=0.45, lw=1)
    ax.text(x + 1.69, 2.30, use, ha="center", va="center",
            fontsize=10.2, weight="bold", color="white", linespacing=1.18)
    ax.text(x + 1.69, 1.88, note, ha="center", va="center",
            fontsize=8.5, color="white")


def generate_png():
    fig, ax = plt.subplots(figsize=(13.33, 5.6))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.42, 5.23, "BCICIV 实验数据划分", fontsize=19, weight="bold", color=INK)
    ax.text(0.43, 4.88, "训练、会话校准与最终评估严格隔离", fontsize=10.5, color=MUTED)

    rounded(ax, 0.43, 2.25, 1.30, 1.18, "#EEF3F8", BLUE)
    ax.text(1.08, 3.03, "MI 窗口", ha="center", va="center", fontsize=12,
            weight="bold", color=INK)
    ax.text(1.08, 2.65, "三类任务", ha="center", va="center", fontsize=9.2, color=MUTED)
    arrow(ax, (1.76, 2.84), (2.08, 2.84), BLUE)

    card(ax, 2.10, BLUE, "TRAIN", "840", "训练 FBCSP、Embedding\n与经验库候选 Heads", "拟合模型参数")
    card(ax, 5.82, ORANGE, "CALIBRATION", "42", "候选评分与 Top-K 选择\n形成当前会话组合", "不计入最终评估")
    card(ax, 9.54, GREEN, "EVALUATION", "518", "计算最终 Accuracy、Balanced\nAccepted 与 Reject 指标", "不参与训练或候选选择")

    arrow(ax, (5.50, 2.84), (5.79, 2.84), MUTED)
    arrow(ax, (9.22, 2.84), (9.51, 2.84), MUTED)

    # Isolation badges.
    rounded(ax, 2.10, 0.72, 7.10, 0.53, "#FFF3F3", "#E6A3A3", 1.1)
    ax.text(5.65, 0.985, "评估集不用于模型训练　｜　评估集不用于 Top-K 选择",
            ha="center", va="center", fontsize=10.8, weight="bold", color=RED)
    rounded(ax, 9.54, 0.72, 3.38, 0.53, "#EDF7F1", "#A5CFB6", 1.1)
    ax.text(11.23, 0.985, "相同评估窗口 · 公平对比", ha="center", va="center",
            fontsize=10.2, weight="bold", color=GREEN)

    ax.text(0.43, 0.25,
            "口径：Reject 是系统依据置信度产生的输出，不是数据集中的第四类标签。",
            fontsize=9.5, color=MUTED)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


BASE = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"


def vertex(root, ident, value, x, y, w, h, fill, stroke, font=INK, size=14, sw=2):
    style = BASE + f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={font};fontSize={size};"
    c = SubElement(root, "mxCell", id=ident, value=value, style=style, vertex="1", parent="1")
    SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})


def edge(root, ident, source, target, color=MUTED):
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;strokeColor={color};endArrow=block;endFill=1;"
    c = SubElement(root, "mxCell", id=ident, style=style, edge="1", parent="1", source=source, target=target)
    SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})


def generate_drawio():
    mx = Element("mxfile", host="app.diagrams.net", modified="2026-07-15T00:00:00.000Z", agent="Codex", version="24.7.17")
    diagram = SubElement(mx, "diagram", id="bciciv-split", name="Page-1")
    model = SubElement(diagram, "mxGraphModel", dx="1365", dy="767", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1333", pageHeight="560", math="0", shadow="0")
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")
    vertex(root, "title", "<b>BCICIV 实验数据划分</b><br><font color='#667487' style='font-size:12px'>训练、会话校准与最终评估严格隔离</font>", 40, 25, 1250, 65, "none", "none", INK, 24, 0)
    vertex(root, "input", "<b>MI 窗口</b><br><font color='#667487' style='font-size:11px'>三类任务</font>", 40, 225, 125, 90, "#EEF3F8", BLUE, INK, 14)
    vertex(root, "train", "<b>TRAIN</b><br><font style='font-size:28px'><b>840</b></font><br>训练 FBCSP、Embedding<br>与经验库候选 Heads<br><font style='font-size:11px'>拟合模型参数</font>", 205, 145, 315, 235, BLUE, BLUE, "#FFFFFF", 15)
    vertex(root, "cal", "<b>CALIBRATION</b><br><font style='font-size:28px'><b>42</b></font><br>候选评分与 Top-K 选择<br>形成当前会话组合<br><font style='font-size:11px'>不计入最终评估</font>", 570, 145, 315, 235, ORANGE, ORANGE, "#FFFFFF", 15)
    vertex(root, "eval", "<b>EVALUATION</b><br><font style='font-size:28px'><b>518</b></font><br>计算最终 Accuracy、Balanced<br>Accepted 与 Reject 指标<br><font style='font-size:11px'>不参与训练或候选选择</font>", 935, 145, 315, 235, GREEN, GREEN, "#FFFFFF", 15)
    vertex(root, "isolation", "<b>评估集不用于模型训练　｜　评估集不用于 Top-K 选择</b>", 205, 420, 680, 50, "#FFF3F3", "#E6A3A3", RED, 14, 1)
    vertex(root, "fair", "<b>相同评估窗口 · 公平对比</b>", 935, 420, 315, 50, "#EDF7F1", "#A5CFB6", GREEN, 13, 1)
    vertex(root, "note", "口径：Reject 是系统依据置信度产生的输出，不是数据集中的第四类标签。", 40, 500, 1210, 30, "none", "none", MUTED, 12, 0)
    edge(root, "e1", "input", "train", BLUE)
    edge(root, "e2", "train", "cal")
    edge(root, "e3", "cal", "eval")
    ElementTree(mx).write(DRAWIO, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    generate_png()
    generate_drawio()
