from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "report" / "video" / "pics"
STATUS = "公开 EEG replay ｜ tiled MVM 软件仿真 ｜ 非 Gazelle 实机结果"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, LINE = "#203047", "#667487", "#CAD5E1"
BLUE, GREEN, ORANGE = "#3478B8", "#3C9064", "#E58A2B"
PURPLE, RED, TEAL = "#7656A8", "#C95757", "#2F8791"


def canvas(title, subtitle=""):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set(xlim=(0, 16), ylim=(0, 9)); ax.axis("off"); fig.patch.set_facecolor("white")
    ax.text(.62, 8.38, title, fontsize=24, weight="bold", color=INK, va="center")
    if subtitle:
        ax.text(.64, 7.87, subtitle, fontsize=11.5, color=MUTED, va="center")
    return fig, ax


def rounded(ax, x, y, w, h, face="#F6F8FA", edge=LINE, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.08", facecolor=face,
        edgecolor=edge, linewidth=lw))


def card(ax, x, y, w, h, title, detail, color=BLUE, light=False, size=13):
    rounded(ax, x, y, w, h, "#F6F8FA" if light else color, color, 1.4)
    ax.text(x+w/2, y+h*.64, title, ha="center", va="center", fontsize=size,
            weight="bold", color=INK if light else "white")
    ax.text(x+w/2, y+h*.27, detail, ha="center", va="center", fontsize=9.2,
            color=MUTED if light else "white", linespacing=1.25)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                linewidth=1.7, color=color))


def status(ax):
    rounded(ax, .62, .28, 5.15, .42, "#EEF3F8", LINE, 1.0)
    ax.text(.83, .49, STATUS, fontsize=8.5, color=MUTED, va="center")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=120, facecolor="white")
    plt.close(fig)


def frame01():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set(xlim=(0,16), ylim=(0,9)); ax.axis("off"); fig.patch.set_facecolor("white")
    ax.add_patch(Rectangle((0,0),16,9,facecolor="#F5F8FB",edgecolor="none"))
    ax.add_patch(Rectangle((0,0),.22,9,facecolor=BLUE,edgecolor="none"))
    ax.text(1.05, 6.95, "光电混合运动想象脑机接口", fontsize=31, weight="bold", color=INK)
    ax.text(1.08, 6.15, "软件原型验证与复现演示", fontsize=17, color=BLUE, weight="bold")
    ax.text(1.08, 5.58, "Hybrid Photonic Motor-Imagery BCI", fontsize=11.5, color=MUTED)
    rounded(ax, 1.05, 3.23, 13.65, 1.48, "white", LINE, 1.2)
    items=[("公开 EEG", "离线 replay"), ("三分类 + 拒识", "left / right / foot / reject"),
           ("tiled MVM", "软件仿真"), ("工程复现", "指标 · 图表 · 测试")]
    for i,(a,b) in enumerate(items):
        x=1.42+i*3.35
        ax.text(x,4.18,a,fontsize=13,weight="bold",color=[BLUE,GREEN,ORANGE,PURPLE][i])
        ax.text(x,3.72,b,fontsize=9.2,color=MUTED)
    rounded(ax, 1.05, 1.55, 13.65, .86, "#FFF4F2", "#E8B4AD", 1.0)
    ax.text(7.88,1.98,"非 Gazelle 实机结果 ｜ 非 DeepBCI / 干电极实采结果",
            ha="center",va="center",fontsize=12.5,weight="bold",color=RED)
    status(ax); save(fig,"01_title_scope")


def frame02():
    fig,ax=canvas("数据来源与评估协议","BCICIV_1_asc · 所有拟合仅使用训练段")
    card(ax,.65,3.25,2.65,2.0,"7 个采集文件","局部标签映射为\n左手 · 右手 · 脚",BLUE)
    card(ax,3.75,3.25,2.65,2.0,"TRAIN · 840","每文件前 120 trials\n拟合 CSP / Fisher / 标准化",BLUE)
    card(ax,6.85,3.25,2.65,2.0,"REPLAY · 560","每文件后 80 trials\n模拟在线数据流",GREEN)
    card(ax,9.95,3.25,2.65,2.0,"CALIBRATION · 42","每文件 6 个窗口\n仅作经验库查询",ORANGE)
    card(ax,13.05,3.25,2.30,2.0,"EVALUATION · 518","最终在线评估\n不参与训练 / 选择",PURPLE)
    for x,c in [(3.32,BLUE),(6.42,BLUE),(9.52,GREEN),(12.62,ORANGE)]: arrow(ax,(x,4.25),(x+.40,4.25),c)
    rounded(ax,.65,1.62,14.70,.78,"#EEF3F8",LINE,1.0)
    ax.text(8,2.01,"训练集拟合参数　｜　42 个窗口只用于查询校准　｜　518 个窗口只用于最终指标",
            ha="center",va="center",fontsize=11.5,weight="bold",color=INK)
    status(ax); save(fig,"02_dataset_protocol")


def frame03():
    fig,ax=canvas("预处理与 FBCSP 特征层","保留运动想象的频带与空间先验，避免 replay 信息泄漏")
    steps=[("EEG 窗口","8 channels × 300 samples",BLUE),("CAR + 质量门控","公共平均参考 · 异常检查",TEAL),
           ("6 个子带","mu / beta · 8–32 Hz",GREEN),("OVR CSP","left / right / foot",PURPLE),
           ("72D 对数方差","6 bands × 3 tasks × 4",ORANGE),("Fisher 选择","仅训练段选择 32D",RED)]
    for i,(t,d,c) in enumerate(steps):
        x=.62+i*2.55; card(ax,x,4.05,2.12,1.48,t,d,c)
        if i<5: arrow(ax,(x+2.15,4.79),(x+2.50,4.79),c)
    rounded(ax,.62,1.55,14.70,1.35,"#F5F8FB",LINE,1.0)
    ax.text(1.05,2.46,"特征维数",fontsize=10.2,color=MUTED)
    ax.text(1.05,1.92,"72D",fontsize=24,weight="bold",color=ORANGE)
    arrow(ax,(2.35,2.18),(3.15,2.18),ORANGE)
    ax.text(3.48,1.92,"32D",fontsize=24,weight="bold",color=RED)
    ax.text(5.10,2.15,"空间模式 × 频带响应 × 小样本可解释性",fontsize=13,weight="bold",color=INK)
    ax.text(5.10,1.75,"CSP、Fisher 与标准化参数均不接触 replay / evaluation 标签",fontsize=9.5,color=MUTED)
    status(ax); save(fig,"03_feature_pipeline")


def frame04():
    fig,ax=canvas("小型 MLP Embedding 与经验库检索","轻量特征改善，不替代 FBCSP")
    card(ax,.70,4.05,2.25,1.55,"FBCSP 32D","稳定、可解释\n小样本主干",BLUE)
    card(ax,3.45,4.05,2.25,1.55,"Linear 32→64","LayerNorm + ReLU\n轻量非线性重加权",PURPLE)
    card(ax,6.20,4.05,2.25,1.55,"Embedding 32D","统一候选头\n输入特征空间",PURPLE)
    card(ax,8.95,4.05,2.25,1.55,"校准均值 Query","42 个窗口的\n平均 embedding",ORANGE)
    card(ax,11.70,4.05,3.00,1.55,"经验库 · 66 Heads","2 Anchor + 64 Bootstrap\n检索 Top-K = 8",GREEN)
    for x,c in [(2.98,BLUE),(5.73,PURPLE),(8.48,PURPLE),(11.23,ORANGE)]: arrow(ax,(x,4.82),(x+.43,4.82),c)
    rounded(ax,.70,1.55,14.00,1.25,"#F5F8FB",LINE,1.0)
    ax.text(1.05,2.35,"经验库不是数据缓存",fontsize=12,weight="bold",color=GREEN)
    ax.text(1.05,1.92,"它保存多样化候选模型；少量校准用于选择当前会话组合，而不是重训完整系统。",fontsize=10.2,color=MUTED)
    status(ax); save(fig,"04_embedding_library")


def frame05():
    fig,ax=canvas("候选 Head 到 Tiled MVM 的映射","低位宽候选扫描 · 当前为 NumPy tiled MVM 软件验证")
    rounded(ax,.65,4.12,4.05,2.05,"#F5F8FB",LINE,1.2)
    ax.text(2.68,5.55,"候选线性头",ha="center",fontsize=12,weight="bold",color=INK)
    ax.text(2.68,4.82,"s_i = A_i h + b_i",ha="center",fontsize=22,weight="bold",color=PURPLE)
    ax.text(2.68,4.38,"A_i ∈ R^(3×32)　·　h ∈ R^32",ha="center",fontsize=10,color=MUTED)
    rounded(ax,5.12,4.12,4.20,2.05,"#FFF8F0",ORANGE,1.2)
    ax.text(7.22,5.55,"2 × 8 基础 Tile",ha="center",fontsize=12,weight="bold",color=ORANGE)
    for r in range(2):
        for c in range(8):
            ax.add_patch(Rectangle((5.65+c*.38,4.56+r*.38),.32,.32,facecolor="#F6C98F",edgecolor="white"))
    ax.text(7.22,4.35,"分块 · 补零 · 部分和累加",ha="center",fontsize=9.4,color=MUTED)
    rounded(ax,9.75,4.12,5.60,2.05,"#F4F0FA",PURPLE,1.2)
    ax.text(12.55,5.53,"单窗口调度",ha="center",fontsize=12,weight="bold",color=PURPLE)
    ax.text(12.55,4.88,"8 candidates × 8 tiles",ha="center",fontsize=17,weight="bold",color=INK)
    ax.text(12.55,4.43,"= 64 logical tile evaluations / window",ha="center",fontsize=10.5,color=MUTED)
    arrow(ax,(4.73,5.14),(5.08,5.14),ORANGE); arrow(ax,(9.35,5.14),(9.71,5.14),PURPLE)
    rounded(ax,.65,1.52,14.70,1.33,"#EEF3F8",LINE,1.0)
    ax.text(8,2.28,"验证对象：tile 分块、零填充、乘加与位权累加逻辑",ha="center",fontsize=12,weight="bold",color=INK)
    ax.text(8,1.84,"不表述为 Gazelle 板卡时延、功耗或精度实测",ha="center",fontsize=10,color=RED)
    status(ax); save(fig,"05_candidate_tile_scan")


def frame06():
    fig,ax=canvas("Replay 在线决策与拒识","候选概率融合后，低置信窗口不强制转换为控制命令")
    steps=[("32D Embedding","当前 EEG 窗口",BLUE),("8 个候选得分","A_i h + b_i",PURPLE),("Softmax","候选类别概率",ORANGE),
           ("加权融合","检索权重 × 概率",GREEN),("三重门控","confidence · margin\nquality gate",RED)]
    for i,(t,d,c) in enumerate(steps):
        x=.62+i*2.72; card(ax,x,4.25,2.22,1.52,t,d,c)
        if i<4: arrow(ax,(x+2.25,5.01),(x+2.67,5.01),c)
    x=14.22
    rounded(ax,x,3.55,1.15,2.92,"#F6F8FA",LINE,1.1)
    for j,(label,color) in enumerate([("LEFT",BLUE),("RIGHT",GREEN),("FOOT",ORANGE),("REJECT",RED)]):
        rounded(ax,x+.14,5.88-j*.61,.87,.43,color,color,1.0)
        ax.text(x+.575,6.095-j*.61,label,ha="center",va="center",fontsize=8.1,weight="bold",color="white")
    arrow(ax,(13.77,5.01),(14.18,5.01),RED)
    rounded(ax,.62,1.55,13.25,1.25,"#FFF4F2","#E8B4AD",1.0)
    ax.text(1.02,2.34,"为什么保留 Reject？",fontsize=12,weight="bold",color=RED)
    ax.text(1.02,1.91,"非目标状态以及光、声、触觉、情绪等刺激可能掩盖 MI；拒识可阻止低置信窗口直接触发控制。",fontsize=10,color=MUTED)
    status(ax); save(fig,"06_online_decision")


def frame07():
    fig,ax=canvas("结果与工程复现性汇总","指标、图表和测试均由工程命令生成")
    metrics=[("518","evaluation windows",BLUE),("75.10%","command accuracy",GREEN),("72.88%","balanced accuracy",PURPLE),
             ("78.27%","accepted accuracy",ORANGE),("4.05%","reject rate",RED),("64","tiles / window",TEAL)]
    for i,(v,l,c) in enumerate(metrics):
        x=.65+(i%3)*3.35; y=4.75-(i//3)*1.58
        rounded(ax,x,y,3.00,1.22,"#F6F8FA",c,1.35)
        ax.text(x+.25,y+.76,v,fontsize=20,weight="bold",color=c)
        ax.text(x+.25,y+.32,l,fontsize=9.2,color=MUTED)
    rounded(ax,11.08,3.17,4.28,2.80,"#EEF3F8",BLUE,1.35)
    ax.text(13.22,5.38,"工程复现证据",ha="center",fontsize=13,weight="bold",color=INK)
    ax.text(11.50,4.80,"已保存　metrics JSON",fontsize=10.3,color=INK)
    ax.text(11.50,4.32,"已生成　命令输出结果图",fontsize=10.3,color=INK)
    ax.text(11.50,3.84,"已通过　16 passed 单元测试",fontsize=10.3,color=INK)
    ax.text(11.50,3.38,"可追溯　配置、候选与日志",fontsize=10.3,color=INK)
    rounded(ax,.65,1.46,14.71,.86,"#FFF4F2","#E8B4AD",1.0)
    ax.text(8,1.89,"公开数据结果用于算法与软件链路验证，不外推为真实用户、电极或光子硬件性能。",ha="center",fontsize=10.5,weight="bold",color=RED)
    status(ax); save(fig,"07_results_reproducibility")


def frame08():
    fig,ax=canvas("阶段结论与下一步","从软件原型验证走向硬件在环与真实采集闭环")
    rounded(ax,.65,2.10,7.05,4.55,"#F1F8F4",GREEN,1.3)
    ax.text(1.05,6.15,"已完成",fontsize=15,weight="bold",color=GREEN)
    done=["公开 EEG 数据 replay","FBCSP + small MLP embedding","经验库候选检索与 Top-K 扫描","tiled MVM 分块 / 补零 / 累加验证","保存指标、图表与回归测试"]
    for i,t in enumerate(done): ax.text(1.10,5.52-i*.66,f"已完成　{t}",fontsize=11,color=INK)
    rounded(ax,8.15,2.10,7.20,4.55,"#FFF8F0",ORANGE,1.3)
    ax.text(8.55,6.15,"尚未完成 / 下一步",fontsize=15,weight="bold",color=ORANGE)
    todo=["完善自适应 uint / int 量化策略","Gazelle 硬件在环调用与数值核验","Cyton / DeepBCI 上位机与在线采集","真实被试端到端闭环验证","实测通信、时延、误差与功耗"]
    for i,t in enumerate(todo): ax.text(8.60,5.52-i*.66,f"→  {t}",fontsize=11,color=INK)
    rounded(ax,.65,.92,14.70,.72,"#EEF3F8",LINE,1.0)
    ax.text(8,1.28,"本视频为软件原型验证与复现留档；现场答辩以实际可运行程序为准。",ha="center",fontsize=11.2,weight="bold",color=INK)
    ax.text(14.90,.48,"谢谢观看",ha="right",fontsize=12,weight="bold",color=BLUE)
    status(ax); save(fig,"08_closing_next_steps")


BASE="rounded=1;whiteSpace=wrap;html=1;arcSize=8;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"


def v(root, ident, value, x,y,w,h, fill="#F6F8FA", stroke=LINE, font=INK, size=16, sw=1):
    style=BASE+f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={font};fontSize={size};"
    c=SubElement(root,"mxCell",id=ident,value=value,style=style,vertex="1",parent="1")
    SubElement(c,"mxGeometry",x=str(x),y=str(y),width=str(w),height=str(h),**{"as":"geometry"})


DRAWIO_PAGES = {
    "01_title_scope": ("光电混合运动想象脑机接口", "软件原型验证与复现演示", [
        ("公开 EEG", "离线 replay", BLUE), ("三分类 + 拒识", "left / right / foot / reject", GREEN),
        ("tiled MVM", "软件仿真", ORANGE), ("工程复现", "指标 · 图表 · 测试", PURPLE)]),
    "02_dataset_protocol": ("数据来源与评估协议", "BCICIV_1_asc · 所有拟合仅使用训练段", [
        ("7 个采集文件", "三类标签", BLUE), ("TRAIN · 840", "拟合全部参数", BLUE),
        ("REPLAY · 560", "模拟在线数据流", GREEN), ("CALIBRATION · 42", "仅经验库查询", ORANGE),
        ("EVALUATION · 518", "只计算最终指标", PURPLE)]),
    "03_feature_pipeline": ("预处理与 FBCSP 特征层", "6 子带 × 3 OVR × 4 CSP = 72D；Fisher 选择 32D", [
        ("EEG 窗口", "8 × 300", BLUE), ("CAR + 质量门控", "参考与异常检查", TEAL),
        ("6 个子带", "8–32 Hz", GREEN), ("OVR CSP", "left / right / foot", PURPLE),
        ("72D", "对数方差", ORANGE), ("Fisher 32D", "仅训练段选择", RED)]),
    "04_embedding_library": ("小型 MLP Embedding 与经验库检索", "轻量特征改善，不替代 FBCSP", [
        ("FBCSP 32D", "小样本主干", BLUE), ("Linear 32→64", "LayerNorm + ReLU", PURPLE),
        ("Embedding 32D", "候选头输入空间", PURPLE), ("校准均值 Query", "42 个窗口", ORANGE),
        ("经验库 66 Heads", "Top-K = 8", GREEN)]),
    "05_candidate_tile_scan": ("候选 Head 到 Tiled MVM 的映射", "当前为 NumPy tiled MVM 软件验证", [
        ("s_i = A_i h + b_i", "A_i: 3×32", PURPLE), ("2×8 Tile", "分块 · 补零 · 累加", ORANGE),
        ("8 Candidates", "每候选 8 tiles", GREEN), ("64", "logical tiles / window", BLUE)]),
    "06_online_decision": ("Replay 在线决策与拒识", "低置信窗口不强制转换为控制命令", [
        ("Embedding", "32D", BLUE), ("候选得分", "A_i h + b_i", PURPLE), ("Softmax", "候选概率", ORANGE),
        ("加权融合", "检索权重", GREEN), ("三重门控", "confidence / margin / quality", RED),
        ("输出", "left / right / foot / reject", TEAL)]),
    "07_results_reproducibility": ("结果与工程复现性汇总", "指标、图表和测试均由工程命令生成", [
        ("518", "evaluation windows", BLUE), ("75.10%", "command accuracy", GREEN),
        ("72.88%", "balanced accuracy", PURPLE), ("78.27%", "accepted accuracy", ORANGE),
        ("4.05%", "reject rate", RED), ("16 passed", "回归测试", TEAL)]),
    "08_closing_next_steps": ("阶段结论与下一步", "软件原型 → 硬件在环 → 真实采集闭环", [
        ("已完成", "公开 replay · FBCSP · embedding · 经验库 · tiled MVM", GREEN),
        ("下一步", "自适应量化 · Gazelle HIL · Cyton / DeepBCI · 闭环实测", ORANGE)])
}


def drawio(name, title, subtitle, items):
    mx=Element("mxfile",host="app.diagrams.net",modified="2026-07-16T00:00:00.000Z",agent="Codex",version="24.7.17")
    d=SubElement(mx,"diagram",id=name,name="Page-1")
    m=SubElement(d,"mxGraphModel",dx="1920",dy="1080",grid="1",gridSize="10",guides="1",tooltips="1",connect="1",arrows="1",fold="1",page="1",pageScale="1",pageWidth="1920",pageHeight="1080",math="0",shadow="0")
    root=SubElement(m,"root"); SubElement(root,"mxCell",id="0"); SubElement(root,"mxCell",id="1",parent="0")
    v(root,"title",f"<b>{title}</b><br><font color='#667487' style='font-size:20px'>{subtitle}</font>",70,45,1780,110,"none","none",INK,34,0)
    n=len(items); cols=min(n,3); rows=(n+cols-1)//cols; gap=45; totalw=1780; w=(totalw-gap*(cols-1))/cols; h=230 if rows==1 else 205
    for i,(t,detail,color) in enumerate(items):
        col=i%cols; row=i//cols; x=70+col*(w+gap); y=220+row*(h+55)
        v(root,f"item{i}",f"<b>{t}</b><br><font style='font-size:18px'>{detail}</font>",x,y,w,h,color,color,"#FFFFFF",24,2)
    v(root,"status",STATUS,70,990,850,45,"#EEF3F8",LINE,MUTED,14,1)
    ElementTree(mx).write(OUT/f"{name}.drawio",encoding="utf-8",xml_declaration=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for fn in (frame01,frame02,frame03,frame04,frame05,frame06,frame07,frame08): fn()
    for name,(title,subtitle,items) in DRAWIO_PAGES.items(): drawio(name,title,subtitle,items)


if __name__ == "__main__": main()
