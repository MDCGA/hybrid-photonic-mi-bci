from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "slides" / "assets"
PNG = OUT_DIR / "dual_path_quantization_flow.png"
DRAWIO = OUT_DIR / "dual_path_quantization_flow.drawio"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#203047"
MUTED = "#667487"
BLUE = "#3478B8"
GREEN = "#3C9064"
ORANGE = "#E58A2B"
PURPLE = "#7656A8"
RED = "#C95757"
LINE = "#CAD5E1"


def box(ax, x, y, w, h, title, detail, color, light=False):
    face = "#F5F7FA" if light else color
    edge = color
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        facecolor=face, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(patch)
    tc = INK if light else "white"
    dc = MUTED if light else "white"
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
            fontsize=11.4, weight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
            fontsize=8.6, color=dc, linespacing=1.15)


def arrow(ax, a, b, color=MUTED, label=None, label_dy=0.16):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.55, color=color))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + label_dy, label,
                ha="center", va="center", fontsize=8.5, color=color, weight="bold")


def generate_png():
    fig, ax = plt.subplots(figsize=(13.33, 6.3))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.38, 5.93, "分路径混合量化", fontsize=19, weight="bold", color=INK)
    ax.text(0.39, 5.58, "主干保持数值稳定 · 候选扫描突出低位宽并行", fontsize=10.5, color=MUTED)

    box(ax, 0.42, 2.55, 1.48, 1.02, "前向线性算子", "统一 Backend\n接管入口", BLUE)
    box(ax, 2.30, 2.55, 1.52, 1.02, "路径识别", "按算子用途选择\n量化与执行策略", PURPLE)
    arrow(ax, (1.92, 3.06), (2.27, 3.06), BLUE)

    # Upper path.
    ax.text(4.15, 5.08, "路径 A｜常规线性路径", fontsize=11.2, weight="bold", color=BLUE)
    ax.text(4.15, 4.79, "FBCSP · 标准化 · small MLP 等", fontsize=8.9, color=MUTED)
    box(ax, 4.12, 3.78, 1.65, 0.92, "8-bit 逻辑精度", "兼顾动态范围与\n数值稳定性", BLUE)
    box(ax, 6.18, 3.78, 1.70, 0.92, "radix-16 拆分", "展开为多个\n4-bit slices", GREEN)
    box(ax, 8.29, 3.78, 1.76, 0.92, "分片线性执行", "软件模拟执行核\n逐 slice 计算", PURPLE)
    box(ax, 10.46, 3.78, 1.72, 0.92, "数字端累加", "重构较高逻辑精度\n线性输出", BLUE, light=True)
    arrow(ax, (3.84, 3.24), (4.08, 4.18), BLUE)
    arrow(ax, (5.79, 4.24), (6.15, 4.24), BLUE)
    arrow(ax, (7.90, 4.24), (8.26, 4.24), GREEN)
    arrow(ax, (10.07, 4.24), (10.43, 4.24), PURPLE)

    # Lower path.
    ax.text(4.15, 2.13, "路径 B｜候选 Head Bank 扫描", fontsize=11.2, weight="bold", color=ORANGE)
    ax.text(4.15, 1.84, "经验库候选线性头的重点低位宽路径", fontsize=8.9, color=MUTED)
    box(ax, 4.12, 0.82, 1.65, 0.92, "激活 uint4", "非负输入\n4-bit 量化", ORANGE)
    box(ax, 6.18, 0.82, 1.70, 0.92, "权重 int4", "有符号权重\n4-bit 量化", RED)
    box(ax, 8.29, 0.82, 1.76, 0.92, "TiledMVM", "候选 Bank\n单次低位宽扫描", ORANGE)
    box(ax, 10.46, 0.82, 1.72, 0.92, "Top-K 融合", "数字端加权融合\n并执行拒识", ORANGE, light=True)
    arrow(ax, (3.84, 2.88), (4.08, 1.33), ORANGE)
    arrow(ax, (5.79, 1.28), (6.15, 1.28), ORANGE)
    arrow(ax, (7.90, 1.28), (8.26, 1.28), RED)
    arrow(ax, (10.07, 1.28), (10.43, 1.28), ORANGE)

    # Shared output.
    box(ax, 12.52, 2.55, 0.55, 1.02, "输出", "进入数字端", PURPLE, light=True)
    arrow(ax, (12.20, 4.24), (12.48, 3.38), BLUE)
    arrow(ax, (12.20, 1.28), (12.48, 2.74), ORANGE)

    ax.text(0.42, 0.25,
            "边界：量化对象是前向线性计算；非线性、控制逻辑、经验库管理与拒识仍在数字端。当前执行核为软件模拟。",
            fontsize=9.5, color=MUTED)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


BASE_STYLE = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"


def vertex(root, ident, value, x, y, w, h, fill, stroke, font="#FFFFFF", size=14):
    style = BASE_STYLE + f"fillColor={fill};strokeColor={stroke};strokeWidth=2;fontColor={font};fontSize={size};"
    cell = SubElement(root, "mxCell", id=ident, value=value, style=style, vertex="1", parent="1")
    SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})


def edge(root, ident, source, target, color, label=""):
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;strokeColor={color};endArrow=block;endFill=1;fontFamily=Microsoft YaHei;fontSize=11;fontColor={color};"
    cell = SubElement(root, "mxCell", id=ident, value=label, style=style, edge="1", parent="1", source=source, target=target)
    SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})


def generate_drawio():
    mx = Element("mxfile", host="app.diagrams.net", modified="2026-07-15T00:00:00.000Z", agent="Codex", version="24.7.17")
    diagram = SubElement(mx, "diagram", id="dual-quant", name="Page-1")
    model = SubElement(diagram, "mxGraphModel", dx="1365", dy="767", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1333", pageHeight="630", math="0", shadow="0")
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")
    vertex(root, "title", "<b>分路径混合量化</b><br><font color='#667487' style='font-size:12px'>主干保持数值稳定 · 候选扫描突出低位宽并行</font>", 40, 25, 1250, 60, "none", "none", INK, 24)
    vertex(root, "input", "<b>前向线性算子</b><br><font style='font-size:11px'>统一 Backend 接管入口</font>", 40, 255, 145, 85, BLUE, BLUE)
    vertex(root, "route", "<b>路径识别</b><br><font style='font-size:11px'>按算子用途选择策略</font>", 225, 255, 150, 85, PURPLE, PURPLE)
    vertex(root, "a_label", "<b>路径 A｜常规线性路径</b><br><font color='#667487' style='font-size:11px'>FBCSP · 标准化 · small MLP 等</font>", 410, 105, 770, 45, "none", "none", BLUE, 15)
    vertex(root, "a1", "<b>8-bit 逻辑精度</b><br><font style='font-size:11px'>兼顾动态范围与数值稳定性</font>", 410, 165, 175, 85, BLUE, BLUE)
    vertex(root, "a2", "<b>radix-16 拆分</b><br><font style='font-size:11px'>展开为多个 4-bit slices</font>", 615, 165, 175, 85, GREEN, GREEN)
    vertex(root, "a3", "<b>分片线性执行</b><br><font style='font-size:11px'>软件模拟执行核</font>", 820, 165, 175, 85, PURPLE, PURPLE)
    vertex(root, "a4", "<b>数字端累加</b><br><font color='#667487' style='font-size:11px'>重构线性输出</font>", 1025, 165, 175, 85, "#F5F7FA", BLUE, INK)
    vertex(root, "b_label", "<b>路径 B｜候选 Head Bank 扫描</b><br><font color='#667487' style='font-size:11px'>经验库候选线性头的重点低位宽路径</font>", 410, 330, 770, 45, "none", "none", ORANGE, 15)
    vertex(root, "b1", "<b>激活 uint4</b><br><font style='font-size:11px'>非负输入 4-bit 量化</font>", 410, 390, 175, 85, ORANGE, ORANGE)
    vertex(root, "b2", "<b>权重 int4</b><br><font style='font-size:11px'>有符号权重 4-bit 量化</font>", 615, 390, 175, 85, RED, RED)
    vertex(root, "b3", "<b>TiledMVM</b><br><font style='font-size:11px'>候选 Bank 单次低位宽扫描</font>", 820, 390, 175, 85, ORANGE, ORANGE)
    vertex(root, "b4", "<b>Top-K 融合</b><br><font color='#667487' style='font-size:11px'>数字端加权融合并拒识</font>", 1025, 390, 175, 85, "#F5F7FA", ORANGE, INK)
    vertex(root, "output", "<b>输出</b><br><font color='#667487' style='font-size:11px'>进入数字端</font>", 1225, 255, 75, 85, "#F5F7FA", PURPLE, INK)
    vertex(root, "note", "边界：量化对象是前向线性计算；非线性、控制逻辑、经验库管理与拒识仍在数字端。当前执行核为软件模拟。", 40, 540, 1260, 35, "none", "none", MUTED, 12)
    edge(root, "e0", "input", "route", BLUE)
    edge(root, "ea0", "route", "a1", BLUE, "常规算子")
    edge(root, "ea1", "a1", "a2", BLUE)
    edge(root, "ea2", "a2", "a3", GREEN)
    edge(root, "ea3", "a3", "a4", PURPLE)
    edge(root, "eb0", "route", "b1", ORANGE, "候选扫描")
    edge(root, "eb1", "b1", "b2", ORANGE)
    edge(root, "eb2", "b2", "b3", RED)
    edge(root, "eb3", "b3", "b4", ORANGE)
    edge(root, "eo1", "a4", "output", BLUE)
    edge(root, "eo2", "b4", "output", ORANGE)
    ElementTree(mx).write(DRAWIO, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_png()
    generate_drawio()
