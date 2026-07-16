import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "artifacts" / "metrics" / "fbcsp_design" / "compute_accounting.json"
OUT = REPORT / "slides" / "assets"
PNG = OUT / "forward_linear_compute_share.png"
SVG = OUT / "forward_linear_compute_share.svg"
CSV = OUT / "forward_linear_compute_share.csv"

LINE_NAME = "FBCSP + MLP embedding + library + photonic scan"
ORDER = [
    ("bandpass_filter", "SOS 带通滤波"),
    ("fbcsp_projection", "FBCSP 空间投影"),
    ("mlp_linear_forward", "MLP 前向线性层"),
    ("candidate_head_scan", "Candidate Scan"),
    ("experience_fusion", "经验融合"),
]
COLORS = ["#3478B8", "#3C9064", "#7656A8", "#E58A2B", "#C95757"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_values():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    line = next(item for item in data["lines"] if item["line"] == LINE_NAME)
    totals = {}
    for event in line["events"]:
        if event["stage"] == "inference":
            totals[event["category"]] = totals.get(event["category"], 0) + int(event["macs"])
    values = [(label, totals.get(category, 0)) for category, label in ORDER]
    expected = int(line["summary"]["linear_macs_inference"])
    actual = sum(value for _, value in values)
    if actual != expected:
        raise ValueError(f"Inference MAC mismatch: categories={actual}, summary={expected}")
    return values, expected


def write_csv(values, total):
    with CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "linear_macs", "share_percent"])
        for label, value in values:
            writer.writerow([label, value, f"{value / total * 100:.6f}"])


def make_chart(values, total):
    labels = [item[0] for item in values]
    macs = [item[1] for item in values]
    shares = [value / total * 100 for value in macs]

    fig = plt.figure(figsize=(13.33, 7.0), facecolor="white")
    ax = fig.add_axes([0.055, 0.13, 0.56, 0.72])
    ax.axis("equal")

    def autopct(pct):
        return f"{pct:.2f}%" if pct >= 2 else ""

    wedges, _, autotexts = ax.pie(
        macs, startangle=90, counterclock=False, colors=COLORS,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        autopct=autopct, pctdistance=0.77,
        textprops={"color": "white", "fontsize": 11, "weight": "bold"},
    )
    for text in autotexts:
        text.set_fontweight("bold")

    ax.text(0, 0.08, "主线 Inference", ha="center", va="center",
            fontsize=13, weight="bold", color="#203047")
    ax.text(0, -0.13, "342.105 M MAC", ha="center", va="center",
            fontsize=17, weight="bold", color="#203047")

    fig.text(0.055, 0.93, "前向线性计算各模块占比", fontsize=21,
             weight="bold", color="#203047")
    fig.text(0.057, 0.885, "主线 inference 计算账本 · forward-only linear MAC 口径",
             fontsize=10.5, color="#667487")

    x0, y0 = 0.64, 0.78
    fig.text(x0, y0 + 0.08, "模块明细", fontsize=13.5, weight="bold", color="#203047")
    for i, (label, value) in enumerate(values):
        y = y0 - i * 0.112
        fig.patches.append(plt.Rectangle((x0, y - 0.012), 0.018, 0.028,
                                         transform=fig.transFigure, facecolor=COLORS[i],
                                         edgecolor="none"))
        fig.text(x0 + 0.027, y + 0.005, label, fontsize=11.2,
                 weight="bold", color="#203047", va="center")
        pct = shares[i]
        pct_text = f"{pct:.2f}%" if pct >= 0.01 else f"{pct:.4f}%"
        fig.text(0.94, y + 0.005, pct_text, fontsize=11.2,
                 weight="bold", color=COLORS[i], ha="right", va="center")
        fig.text(x0 + 0.027, y - 0.033, f"{value:,} MAC",
                 fontsize=8.9, color="#667487", va="center")

    fig.text(0.64, 0.16,
             "SOS + FBCSP 投影占 98.99%\n候选扫描占比小，但承担会话特化的并行扫描",
             fontsize=10.5, color="#203047", linespacing=1.6,
             bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F3F6F9",
                   "edgecolor": "#CAD5E1"})
    fig.text(0.057, 0.045,
             "说明：占比表示 MAC 计算量构成，不表示运行时间、功耗或真实光芯片资源占比；当前执行核为软件模拟。",
             fontsize=9.3, color="#667487")

    fig.savefig(PNG, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SVG, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows, total_macs = load_values()
    write_csv(rows, total_macs)
    make_chart(rows, total_macs)
