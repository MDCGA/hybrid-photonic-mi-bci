from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "assets"
PNG = OUT / "dual_path_quantization_flow.png"
DRAWIO = OUT / "dual_path_quantization_flow.drawio"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED = "#203047", "#667487"
BLUE, GREEN, ORANGE = "#3478B8", "#3C9064", "#E58A2B"
PURPLE, RED, LINE = "#7656A8", "#C95757", "#CAD5E1"


def rounded(ax, x, y, w, h, face, edge=LINE, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=face, edgecolor=edge, linewidth=lw,
    ))


def box(ax, x, y, w, h, title, detail, color, light=False):
    rounded(ax, x, y, w, h, "#F5F7FA" if light else color, color)
    ax.text(x + w / 2, y + h * .66, title, ha="center", va="center",
            fontsize=10.7, weight="bold", color=INK if light else "white")
    ax.text(x + w / 2, y + h * .28, detail, ha="center", va="center",
            fontsize=8.0, linespacing=1.12, color=MUTED if light else "white")


def arrow(ax, a, b, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
        linestyle="--" if dashed else "-", color=color,
    ))


def generate_png():
    fig, ax = plt.subplots(figsize=(13.33, 7.15))
    ax.set(xlim=(0, 13.33), ylim=(0, 7.15))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.38, 6.80, "量化策略：逻辑精度不等于单次物理位宽",
            fontsize=18.5, weight="bold", color=INK)
    ax.text(0.39, 6.45, "原生单次 4-bit 执行 · 更高逻辑精度由多个 4-bit slices 位权重构",
            fontsize=10.4, color=MUTED)

    # Candidate Scan: fixed single-call 4-bit path.
    rounded(ax, .38, 4.12, 12.55, 1.92, "#FFF8F0", "#F0C38D", 1.1)
    ax.text(.67, 5.70, "路径 A｜Candidate Scan：固定单次 4-bit",
            fontsize=12.2, weight="bold", color=ORANGE)
    ax.text(.68, 5.39, "候选 Head Bank 的重点低位宽路径", fontsize=8.8, color=MUTED)
    items = [
        (.68, 2.18, "输入 uint4", "qinmin=0 · qinmax=15", ORANGE, False),
        (3.30, 2.18, "权重 int4", "qwtmin=-8 · qwtmax=7", RED, False),
        (5.92, 2.08, "单次 4-bit 执行", "不进行多 slice 拆分", ORANGE, False),
        (8.44, 2.05, "TiledMVM", "候选 Bank 并行扫描", PURPLE, False),
        (10.93, 1.70, "数字端", "Top-K 融合 / 拒识", ORANGE, True),
    ]
    for x, w, title, detail, color, light in items:
        box(ax, x, 4.43, w, .78, title, detail, color, light)
    for a, b, c in [((2.88, 4.82), (3.27, 4.82), ORANGE),
                    ((5.50, 4.82), (5.89, 4.82), RED),
                    ((8.02, 4.82), (8.41, 4.82), ORANGE),
                    ((10.51, 4.82), (10.90, 4.82), PURPLE)]:
        arrow(ax, a, b, c)

    # Other operators: adaptive logical precision.
    rounded(ax, .38, 1.14, 12.55, 2.65, "#F5F8FB", "#B9C9D9", 1.1)
    ax.text(.67, 3.47, "路径 B｜其他前向线性算子：自适应逻辑精度",
            fontsize=12.2, weight="bold", color=BLUE)
    ax.text(.68, 3.18, "起始精度按敏感度分层；策略仍在完善，不代表已定版的最优配置",
            fontsize=8.8, color=MUTED)

    rounded(ax, .68, 1.53, 2.43, 1.38, "#EAF2FA", BLUE)
    ax.text(1.895, 2.66, "算子起始逻辑精度", ha="center", fontsize=10.3,
            weight="bold", color=INK)
    for y, bits, ops, color in [
        (2.28, "4-bit", "CAR", ORANGE),
        (1.96, "6-bit", "SOS · FBCSP · 标准化", GREEN),
        (1.64, "8-bit", "MLP 等敏感路径", PURPLE),
    ]:
        ax.text(.92, y, bits, fontsize=9.4, weight="bold", color=color)
        ax.text(1.55, y, ops, fontsize=8.5, color=INK)

    box(ax, 3.53, 1.72, 1.73, .96, "当前精度执行", "4-bit 原生或\n多 slice 路径", BLUE)
    box(ax, 5.67, 1.72, 1.78, .96, "8-bit Shadow", "参考结果监测\n低位宽误差", PURPLE)
    box(ax, 7.86, 1.72, 1.58, .96, "误差比较", "超过阈值？", RED, True)
    box(ax, 9.85, 2.34, 2.64, .76, "精度单调提升", "4 → 6 → 8（不回退）", RED)
    box(ax, 9.85, 1.36, 2.64, .76, "输出 / 位权重构", "4-bit 直接输出\n6/8-bit 按 slice 位权重构", GREEN)
    for a, b, c in [((3.13, 2.22), (3.50, 2.22), BLUE),
                    ((5.28, 2.22), (5.64, 2.22), BLUE),
                    ((7.47, 2.22), (7.83, 2.22), PURPLE),
                    ((9.46, 2.38), (9.82, 2.70), RED),
                    ((9.46, 2.05), (9.82, 1.74), GREEN)]:
        arrow(ax, a, b, c)
    ax.add_patch(FancyArrowPatch(
        (11.15, 3.12), (4.38, 2.72), connectionstyle="arc3,rad=0.13",
        arrowstyle="-|>", mutation_scale=12, linewidth=1.35,
        linestyle="--", color=RED,
    ))
    ax.text(7.75, 3.02, "超阈值：提升后重算", ha="center", fontsize=8.1,
            color=RED, weight="bold")
    ax.text(9.58, 2.54, "是", fontsize=8.1, color=RED, weight="bold")
    ax.text(9.50, 1.82, "否 / 满足", fontsize=8.1, color=GREEN, weight="bold")

    rounded(ax, .38, .34, 12.55, .52, "#EEF3F8", LINE, 1.0)
    ax.text(6.655, .60,
            "物理执行边界：每次光计算乘加均为 4-bit；6/8-bit 是由多个 4-bit slices 组合得到的逻辑精度。",
            ha="center", va="center", fontsize=9.7, weight="bold", color=INK)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=.08)
    plt.close(fig)


BASE = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"


def vertex(root, ident, value, x, y, w, h, fill, stroke, font=INK, size=14, sw=2):
    style = BASE + f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={font};fontSize={size};"
    cell = SubElement(root, "mxCell", id=ident, value=value, style=style, vertex="1", parent="1")
    SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})


def edge(root, ident, source, target, color=MUTED, label="", dashed=False):
    dash = "dashed=1;" if dashed else ""
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;strokeColor={color};endArrow=block;endFill=1;{dash}fontFamily=Microsoft YaHei;fontSize=11;fontColor={color};"
    cell = SubElement(root, "mxCell", id=ident, value=label, style=style,
                      edge="1", parent="1", source=source, target=target)
    SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})


def generate_drawio():
    mx = Element("mxfile", host="app.diagrams.net", modified="2026-07-16T00:00:00.000Z", agent="Codex", version="24.7.17")
    diagram = SubElement(mx, "diagram", id="dual-quant-v2", name="Page-1")
    model = SubElement(diagram, "mxGraphModel", dx="1365", dy="767", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1333", pageHeight="715", math="0", shadow="0")
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")
    vertex(root, "title", "<b>量化策略：逻辑精度不等于单次物理位宽</b><br><font color='#667487' style='font-size:12px'>原生单次 4-bit 执行 · 更高逻辑精度由多个 4-bit slices 位权重构</font>", 35, 20, 1260, 65, "none", "none", INK, 23, 0)
    vertex(root, "lane_a", "", 35, 105, 1260, 190, "#FFF8F0", "#F0C38D", INK, 14, 1)
    vertex(root, "label_a", "<b>路径 A｜Candidate Scan：固定单次 4-bit</b><br><font color='#667487' style='font-size:11px'>候选 Head Bank 的重点低位宽路径</font>", 60, 120, 500, 45, "none", "none", ORANGE, 15, 0)
    vertex(root, "qin", "<b>输入 uint4</b><br><font style='font-size:11px'>qinmin=0 · qinmax=15</font>", 65, 185, 205, 75, ORANGE, ORANGE, "#FFFFFF")
    vertex(root, "qwt", "<b>权重 int4</b><br><font style='font-size:11px'>qwtmin=-8 · qwtmax=7</font>", 315, 185, 205, 75, RED, RED, "#FFFFFF")
    vertex(root, "single", "<b>单次 4-bit 执行</b><br><font style='font-size:11px'>不进行多 slice 拆分</font>", 565, 185, 195, 75, ORANGE, ORANGE, "#FFFFFF")
    vertex(root, "tiled", "<b>TiledMVM</b><br><font style='font-size:11px'>候选 Bank 并行扫描</font>", 805, 185, 195, 75, PURPLE, PURPLE, "#FFFFFF")
    vertex(root, "topk", "<b>数字端</b><br><font color='#667487' style='font-size:11px'>Top-K 融合 / 拒识</font>", 1045, 185, 205, 75, "#F5F7FA", ORANGE)
    vertex(root, "lane_b", "", 35, 320, 1260, 300, "#F5F8FB", "#B9C9D9", INK, 14, 1)
    vertex(root, "label_b", "<b>路径 B｜其他前向线性算子：自适应逻辑精度</b><br><font color='#667487' style='font-size:11px'>起始精度按敏感度分层；策略仍在完善，不代表已定版的最优配置</font>", 60, 335, 760, 48, "none", "none", BLUE, 15, 0)
    vertex(root, "start", "<b>算子起始逻辑精度</b><br><font color='#E58A2B'><b>4-bit</b></font>　CAR<br><font color='#3C9064'><b>6-bit</b></font>　SOS · FBCSP · 标准化<br><font color='#7656A8'><b>8-bit</b></font>　MLP 等敏感路径", 65, 415, 245, 145, "#EAF2FA", BLUE, INK, 13)
    vertex(root, "execute", "<b>当前精度执行</b><br><font style='font-size:11px'>4-bit 原生或多 slice 路径</font>", 350, 440, 170, 90, BLUE, BLUE, "#FFFFFF")
    vertex(root, "shadow", "<b>8-bit Shadow</b><br><font style='font-size:11px'>参考结果监测低位宽误差</font>", 560, 440, 175, 90, PURPLE, PURPLE, "#FFFFFF")
    vertex(root, "compare", "<b>误差比较</b><br><font color='#667487' style='font-size:11px'>超过阈值？</font>", 775, 440, 155, 90, "#F5F7FA", RED)
    vertex(root, "upgrade", "<b>精度单调提升</b><br><font style='font-size:11px'>4 → 6 → 8（不回退）</font>", 970, 395, 270, 75, RED, RED, "#FFFFFF")
    vertex(root, "rebuild", "<b>输出 / 位权重构</b><br><font style='font-size:11px'>4-bit 直接输出<br>6/8-bit 按 slice 位权重构</font>", 970, 500, 270, 85, GREEN, GREEN, "#FFFFFF")
    vertex(root, "boundary", "<b>物理执行边界：每次光计算乘加均为 4-bit；6/8-bit 是由多个 4-bit slices 组合得到的逻辑精度。</b>", 35, 645, 1260, 45, "#EEF3F8", LINE, INK, 13, 1)
    for ident, source, target, color in [
        ("a1", "qin", "qwt", ORANGE), ("a2", "qwt", "single", RED),
        ("a3", "single", "tiled", ORANGE), ("a4", "tiled", "topk", PURPLE),
        ("b1", "start", "execute", BLUE), ("b2", "execute", "shadow", BLUE),
        ("b3", "shadow", "compare", PURPLE),
    ]:
        edge(root, ident, source, target, color)
    edge(root, "b4", "compare", "upgrade", RED, "超阈值")
    edge(root, "b5", "compare", "rebuild", GREEN, "满足阈值")
    edge(root, "loop", "upgrade", "execute", RED, "提升后重算", True)
    ElementTree(mx).write(DRAWIO, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    generate_png()
    generate_drawio()
