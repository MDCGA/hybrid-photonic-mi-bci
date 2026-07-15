from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "slides" / "assets"
PNG = ASSET_DIR / "backend_interface_architecture.png"
DRAWIO = ASSET_DIR / "backend_interface_architecture.drawio"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#203047"
MUTED = "#667487"
BLUE = "#3478B8"
GREEN = "#3C9064"
ORANGE = "#E58A2B"
PURPLE = "#7656A8"
LIGHT = "#F3F6F9"
LINE = "#CAD5E1"


def rounded(ax, x, y, w, h, face, edge=LINE, radius=0.06, lw=1.4):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                mutation_scale=13, linewidth=lw, color=color))


def draw_png():
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.45, 7.08, "统一后端接口与执行边界", fontsize=20, weight="bold", color=INK)
    ax.text(0.46, 6.69, "算法调用接口，不绑定具体执行核", fontsize=10.8, color=MUTED)

    rounded(ax, 0.65, 5.55, 12.03, 0.72, "#EAF2FA", BLUE, lw=1.5)
    ax.text(6.665, 5.98, "上层算法：FBCSP · small MLP embedding · 经验库候选扫描",
            ha="center", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(6.665, 5.69, "只描述算子与张量形状", ha="center", va="center", fontsize=9.3, color=MUTED)

    ax.text(0.65, 5.20, "统一后端接口层", fontsize=11.2, weight="bold", color=INK)
    cards = [
        (0.65, BLUE, "MatrixOps", "矩阵 / 张量", "通用矩阵乘、批量线性变换"),
        (4.75, GREEN, "SignalOps", "CAR / SOS", "信号预处理与滤波线性算子"),
        (8.85, ORANGE, "TiledMVM", "候选 Bank", "低位宽分块候选头扫描"),
    ]
    for x, color, title, subtitle, detail in cards:
        rounded(ax, x, 3.78, 3.83, 1.20, color, color, lw=1.2)
        ax.text(x + 1.915, 4.61, title, ha="center", va="center", fontsize=14,
                weight="bold", color="white")
        ax.text(x + 1.915, 4.25, subtitle, ha="center", va="center", fontsize=12,
                weight="bold", color="white")
        ax.text(x + 1.915, 3.96, detail, ha="center", va="center", fontsize=8.7, color="white")
        arrow(ax, x + 1.915, 5.53, x + 1.915, 5.02, color)

    rounded(ax, 0.65, 3.18, 12.03, 0.36, "#EEF3F8", LINE, lw=1.0)
    ax.text(6.665, 3.36, "上层算法与执行核解耦：接口稳定，执行实现可替换",
            ha="center", va="center", fontsize=10.7, weight="bold", color=INK)

    for x, color in [(2.565, BLUE), (6.665, GREEN), (10.765, ORANGE)]:
        arrow(ax, x, 3.76, x, 2.91, color)

    ax.text(0.65, 2.76, "当前执行边界", fontsize=11.2, weight="bold", color=INK)
    rounded(ax, 0.65, 1.07, 7.45, 1.48, "#F1ECF8", PURPLE, lw=1.5)
    ax.text(4.375, 2.20, "前向线性算子", ha="center", fontsize=13.5, weight="bold", color=PURPLE)
    ax.text(4.375, 1.83, "全部经统一 Backend 接管", ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(4.375, 1.43, "当前执行核：软件模拟", ha="center", fontsize=11, color=MUTED)

    rounded(ax, 8.38, 1.07, 4.30, 1.48, "#F6F7F9", "#98A4B3", lw=1.5)
    ax.text(10.53, 2.20, "数字端保留", ha="center", fontsize=13.5, weight="bold", color=INK)
    ax.text(10.53, 1.83, "非线性 · 控制逻辑 · 拒识", ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(10.53, 1.43, "经验库管理与流程调度", ha="center", fontsize=10.5, color=MUTED)

    ax.text(0.67, 0.52, "边界说明：接管的是全部前向线性计算，不等于替代完整算法；当前结果来自软件模拟执行核。",
            fontsize=10.2, color=MUTED)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def add_vertex(root, ident, value, x, y, w, h, style):
    cell = SubElement(root, "mxCell", id=ident, value=value, style=style, vertex="1", parent="1")
    SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})


def add_edge(root, ident, source, target, color):
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor={color};endArrow=block;endFill=1;"
    cell = SubElement(root, "mxCell", id=ident, style=style, edge="1", parent="1", source=source, target=target)
    SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})


def draw_drawio():
    mxfile = Element("mxfile", host="app.diagrams.net", modified="2026-07-15T00:00:00.000Z", agent="Codex", version="24.7.17")
    diagram = SubElement(mxfile, "diagram", id="backend-interface", name="Page-1")
    model = SubElement(diagram, "mxGraphModel", dx="1365", dy="767", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1333", pageHeight="750", math="0", shadow="0")
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")
    text = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"
    add_vertex(root, "title", "<b>统一后端接口与执行边界</b><br><font color='#667487' style='font-size:12px'>算法调用接口，不绑定具体执行核</font>", 45, 30, 1240, 60, text + "strokeColor=none;fillColor=none;fontSize=24;fontColor=#203047;align=left;")
    add_vertex(root, "algorithm", "<b>上层算法：FBCSP · small MLP embedding · 经验库候选扫描</b><br><font style='font-size:11px' color='#667487'>只描述算子与张量形状</font>", 65, 115, 1205, 75, text + "fillColor=#EAF2FA;strokeColor=#3478B8;strokeWidth=2;fontSize=16;fontColor=#203047;")
    add_vertex(root, "matrix", "<b>MatrixOps</b><br><b>矩阵 / 张量</b><br><font style='font-size:11px'>通用矩阵乘、批量线性变换</font>", 65, 245, 365, 115, text + "fillColor=#3478B8;strokeColor=#3478B8;fontColor=#FFFFFF;fontSize=16;")
    add_vertex(root, "signal", "<b>SignalOps</b><br><b>CAR / SOS</b><br><font style='font-size:11px'>信号预处理与滤波线性算子</font>", 484, 245, 365, 115, text + "fillColor=#3C9064;strokeColor=#3C9064;fontColor=#FFFFFF;fontSize=16;")
    add_vertex(root, "tiled", "<b>TiledMVM</b><br><b>候选 Bank</b><br><font style='font-size:11px'>低位宽分块候选头扫描</font>", 905, 245, 365, 115, text + "fillColor=#E58A2B;strokeColor=#E58A2B;fontColor=#FFFFFF;fontSize=16;")
    add_vertex(root, "decouple", "<b>上层算法与执行核解耦：接口稳定，执行实现可替换</b>", 65, 385, 1205, 38, text + "fillColor=#EEF3F8;strokeColor=#CAD5E1;fontColor=#203047;fontSize=13;")
    add_vertex(root, "linear", "<b>前向线性算子</b><br><b>全部经统一 Backend 接管</b><br><font color='#667487'>当前执行核：软件模拟</font>", 65, 485, 745, 145, text + "fillColor=#F1ECF8;strokeColor=#7656A8;strokeWidth=2;fontColor=#203047;fontSize=16;")
    add_vertex(root, "digital", "<b>数字端保留</b><br><b>非线性 · 控制逻辑 · 拒识</b><br><font color='#667487'>经验库管理与流程调度</font>", 840, 485, 430, 145, text + "fillColor=#F6F7F9;strokeColor=#98A4B3;strokeWidth=2;fontColor=#203047;fontSize=16;")
    add_vertex(root, "note", "边界说明：接管的是全部前向线性计算，不等于替代完整算法；当前结果来自软件模拟执行核。", 65, 660, 1205, 35, text + "strokeColor=none;fillColor=none;fontColor=#667487;fontSize=12;align=left;")
    for i, node in enumerate(("matrix", "signal", "tiled"), 1):
        add_edge(root, f"a{i}", "algorithm", node, (BLUE, GREEN, ORANGE)[i - 1])
        add_edge(root, f"b{i}", node, "linear", (BLUE, GREEN, ORANGE)[i - 1])
    ElementTree(mxfile).write(DRAWIO, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    draw_png()
    draw_drawio()
