from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "assets"
PNG = OUT / "project_key_questions.png"
DRAWIO = OUT / "project_key_questions.drawio"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#203047"
MUTED = "#667487"
BLUE = "#3478B8"
GREEN = "#3C9064"
ORANGE = "#E58A2B"
PURPLE = "#7656A8"
RED = "#C95757"
TEAL = "#2F8791"
LINE = "#CAD5E1"

ITEMS = [
    ("01", "为什么需要经验库？", "个体 / 会话差异明显\n固定全局模型难以覆盖所有使用者", BLUE),
    ("02", "为什么需要拒识？", "非目标状态；光 / 声 / 触觉 / 情绪刺激\n可能掩盖 MI，低置信度不强制输出", RED),
    ("03", "为什么适合光计算？", "滤波、空间投影、MLP 与扫描\n包含大量可统一表达的线性计算", PURPLE),
    ("04", "为什么扫描多个候选头？", "低功耗并行 / 快速扫描小型模型\n以候选组合支持当前会话特化", ORANGE),
    ("05", "实际使用流程是什么？", "Cyton 采集 → 上位机校准 → 经验库\n在线推理 → 命令输出 / 拒识", GREEN),
    ("06", "当前工程边界是什么？", "在线采集与上位机仍在完善\n尚未完成真实被试闭环验证", TEAL),
]


def rounded(ax, x, y, w, h, face, edge, lw=1.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        facecolor=face, edgecolor=edge, linewidth=lw,
    ))


def generate_png():
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set(xlim=(0, 13.33), ylim=(0, 7.5))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(.48, 7.04, "从真实使用约束出发：我们需要回答什么？",
            fontsize=20, weight="bold", color=INK)
    ax.text(.49, 6.62, "从 MI-EEG 个体差异，到光计算执行，再到可用的在线闭环",
            fontsize=10.5, color=MUTED)

    group_labels = [
        (.48, "需求动机", BLUE),
        (4.77, "计算方案", PURPLE),
        (9.06, "使用落地", GREEN),
    ]
    for x, label, color in group_labels:
        ax.text(x, 6.10, label, fontsize=10.5, weight="bold", color=color)
        ax.plot([x, x + 3.80], [5.98, 5.98], color=color, linewidth=2.2)

    positions = [(.48, 4.02), (.48, 1.78), (4.77, 4.02), (4.77, 1.78), (9.06, 4.02), (9.06, 1.78)]
    for (num, question, answer, color), (x, y) in zip(ITEMS, positions):
        rounded(ax, x, y, 3.80, 1.70, "#F7F9FB", LINE, 1.1)
        rounded(ax, x + .18, y + 1.05, .58, .45, color, color, 1.0)
        ax.text(x + .47, y + 1.275, num, ha="center", va="center",
                fontsize=10.5, weight="bold", color="white")
        ax.text(x + .88, y + 1.28, question, ha="left", va="center",
                fontsize=12.0, weight="bold", color=INK)
        ax.text(x + .22, y + .57, answer, ha="left", va="center",
                fontsize=9.4, color=MUTED, linespacing=1.35)
        ax.plot([x + .20, x + 3.60], [y + .91, y + .91], color=color,
                linewidth=1.2, alpha=.45)

    rounded(ax, .48, .55, 12.38, .67, "#EEF3F8", LINE, 1.0)
    ax.text(6.67, .885,
            "设计主线：会话特化经验库 × 置信度拒识 × 前向线性计算接管",
            ha="center", va="center", fontsize=11.8, weight="bold", color=INK)
    ax.text(.50, .20,
            "当前状态：公开数据离线回放与软件模拟已完成；Cyton 上位机、在线采集和真实被试闭环仍在完善。",
            fontsize=8.9, color=MUTED)

    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=.08)
    plt.close(fig)


BASE = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"


def vertex(root, ident, value, x, y, w, h, fill, stroke, font=INK, size=14, sw=1):
    style = BASE + f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={font};fontSize={size};"
    cell = SubElement(root, "mxCell", id=ident, value=value, style=style, vertex="1", parent="1")
    SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})


def generate_drawio():
    mx = Element("mxfile", host="app.diagrams.net", modified="2026-07-16T00:00:00.000Z", agent="Codex", version="24.7.17")
    diagram = SubElement(mx, "diagram", id="project-questions", name="Page-1")
    model = SubElement(diagram, "mxGraphModel", dx="1365", dy="767", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1333", pageHeight="750", math="0", shadow="0")
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")

    vertex(root, "title", "<b>从真实使用约束出发：我们需要回答什么？</b><br><font color='#667487' style='font-size:12px'>从 MI-EEG 个体差异，到光计算执行，再到可用的在线闭环</font>", 45, 25, 1240, 70, "none", "none", INK, 24, 0)
    for i, (x, label, color) in enumerate([(45, "需求动机", BLUE), (475, "计算方案", PURPLE), (905, "使用落地", GREEN)]):
        vertex(root, f"group{i}", f"<b>{label}</b>", x, 110, 380, 35, "none", color, color, 13, 0)

    positions = [(45, 165), (45, 390), (475, 165), (475, 390), (905, 165), (905, 390)]
    for i, ((num, question, answer, color), (x, y)) in enumerate(zip(ITEMS, positions)):
        value = (f"<table cellpadding='0' cellspacing='0' width='100%'><tr>"
                 f"<td width='52' bgcolor='{color}'><font color='#FFFFFF'><b>{num}</b></font></td>"
                 f"<td align='left' style='padding-left:10px'><b>{question}</b></td></tr></table>"
                 f"<br><font color='#667487' style='font-size:12px'>{answer.replace(chr(10), '<br>')}</font>")
        vertex(root, f"q{i}", value, x, y, 380, 170, "#F7F9FB", LINE, INK, 15, 1)

    vertex(root, "mainline", "<b>设计主线：会话特化经验库 × 置信度拒识 × 前向线性计算接管</b>", 45, 610, 1240, 60, "#EEF3F8", LINE, INK, 15, 1)
    vertex(root, "status", "当前状态：公开数据离线回放与软件模拟已完成；Cyton 上位机、在线采集和真实被试闭环仍在完善。", 45, 690, 1240, 30, "none", "none", MUTED, 11, 0)
    ElementTree(mx).write(DRAWIO, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    generate_png()
    generate_drawio()
