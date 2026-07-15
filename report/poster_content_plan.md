# 光电混合运动想象脑机接口项目海报方案

## 1. 海报目标

海报需要让评委在约 30 秒内看懂四件事：

1. 项目解决什么问题：从运动想象 EEG 识别用户意图并输出控制命令。
2. 核心方案是什么：FBCSP 主干、小型 MLP embedding、经验库会话特化和低位宽候选扫描。
3. 光计算在哪里：全部前向线性算子通过统一 Backend 接管，非线性和控制逻辑留在数字端。
4. 当前做到什么程度：完成公开数据离线回放和软件模拟验证，真实 Cyton 闭环及评估板端到端实测仍未完成。

建议采用 A0 竖版三栏布局。若比赛提供固定尺寸，可保持下述信息层级，仅调整栏宽和字号。

---

## 2. 推荐版面结构

### 顶部：标题与一句话定位

**项目标题**

光电混合运动想象脑机接口

**英文副标题（可选）**

Hybrid Photonic Motor-Imagery Brain–Computer Interface

**一句话定位**

面向个体差异的 MI-BCI：以 FBCSP 提取运动节律特征，通过经验库完成会话特化，并以统一后端接管全部前向线性计算。

标题区不要放学校、指导教师、队员真实姓名或 Logo，保持匿名要求。

### 中部：三栏主体

建议按照“问题与方案 → 核心方法 → 实验与结论”的阅读顺序组织。

| 左栏 | 中栏 | 右栏 |
|---|---|---|
| 问题背景、系统总流程、三阶段 | 关键模块、经验库、统一 Backend、量化 | 实验协议、核心结果、计算账本、边界与下一步 |

### 底部：创新点、项目边界与二维码

底部使用三个短卡片总结创新点，右下角可预留演示视频或项目说明二维码。二维码内容应在提交前确认可访问，避免放临时本地链接。

---

## 3. 建议放入的具体内容

## 3.1 问题背景

**小标题：为什么需要个体特化的 MI-BCI？**

建议只保留三条短语：

- EEG：低信噪比、小样本、非平稳、易受伪迹影响
- MI 节律：关注中央区 C3/Cz/C4 附近的 mu/beta ERD/ERS
- 个体差异：响应频段、空间激活和想象策略因人而异

**配图建议**

- 10-20 电极位置概念图，突出 C3/Cz/C4
- ERD/ERS 概念图，明确标注“概念示意，非项目实测”

这一块的目的不是讲完整脑电理论，而是引出“经验库会话特化”的必要性。

## 3.2 核心流程

**小标题：从 EEG 到命令/拒识**

流程框建议保持一行：

`EEG → FBCSP → small MLP embedding → experience library → photonic scan → reject / command`

中文短注释：

- FBCSP：多频带空间特征
- Embedding：轻量特征改善
- Experience Library：候选模型与会话特化
- Photonic Scan：低位宽候选 Head Bank 扫描
- Reject/Command：置信度拒识或输出指令

不要把经验库写成普通数据缓存；它保存的是用于应对 EEG 个体/会话差异的候选模型。

## 3.3 训练、校准与在线推理

建议使用三张独立中文流程图，横向或纵向并列：

- 离线训练：训练 FBCSP、encoder 与候选 Heads，构建经验库
- 会话校准：使用少量校准窗口评分候选模型并选择 Top-K
- 在线推理：当前 EEG 经相同特征链后扫描 Top-K，融合概率并输出命令或拒识

**可用图片**

- `slides/assets/stage_offline_training.png`
- `slides/assets/stage_session_calibration.png`
- `slides/assets/stage_online_inference.png`

如果版面有限，可只保留三个阶段的输入、关键步骤和输出，不放详细算子。

## 3.4 经验库与会话特化

**小标题：少量校准选择当前会话模型组合**

推荐使用流程：

`2 Anchor Heads + 64 Bootstrap Heads → 66 个候选头 → 42 个校准窗口 → 候选评分 → Top-K=8 → 加权融合`

旁边增加一句结论：

> 经验库保存多样化候选模型；校准阶段选择适合当前会话的组合，而不是重新训练完整系统。

**可用图片**

- `slides/assets/experience_library_selection_flow.png`

## 3.5 FBCSP 与小型 MLP 的分工

建议用左右对照卡片：

**FBCSP**

- 运动想象经典方法
- 稳定、可解释、小样本友好
- 滤波、空间投影、协方差/Gram 等线性计算密集

**Small MLP Embedding**

- 仅作轻量特征映射
- 改善候选头的输入空间
- 不替代 FBCSP，不构成纯深度学习 BCI

如有空间，可以放 embedding PCA 图；图注应说明这是特征诊断，不是泛化性能证明。

## 3.6 统一后端与光计算边界

**小标题：算法与执行核解耦**

展示三种接口：

- MatrixOps：矩阵/张量线性运算
- SignalOps：CAR/SOS 信号线性算子
- TiledMVM：候选 Bank 分块扫描

页面核心数字卡：

- Forward Linear MAC：342.105 M / window
- Backend Routed Linear MAC：342.105 M / window
- Forward Linear Takeover：100%

建议附公式：

`前向线性算子接管率 = Backend 路由的前向线性 MAC / 全部前向线性 MAC = 100%`

必须在同一视觉区域标注：

- 统计口径：forward-only linear MAC
- 非线性、拒识、经验库管理和控制逻辑仍在数字端
- 当前执行核：软件模拟
- 100% 不等于真实光芯片功耗占比

**可用图片**

- `slides/assets/backend_interface_architecture.png`

## 3.7 量化策略

**小标题：分路径混合量化**

建议放一张简化双路径图：

- 常规线性路径：8-bit 逻辑精度 → radix-16 → 多个 4-bit slices → 数字端累加
- Candidate Scan：激活 uint4 + 权重 int4 → TiledMVM → Top-K 融合

短结论：

- 主干优先保持数值稳定
- 候选扫描重点采用单次低位宽路径
- 逻辑精度不等于物理位宽
- 当前为动态量化/软件模拟，不是 QAT 或真实芯片量化实测

**可用图片**

- `slides/assets/dual_path_quantization_flow.png`

## 3.8 实验协议

**小标题：训练、校准、评估严格隔离**

BCICIV 主线：

- Train：840 窗口
- Calibration：42 窗口
- Evaluation：518 独立窗口

BNCI 补充验证：

- 9 名被试
- 1296 个窗口
- 保留 subject-wise 结果以观察个体差异

必须强调：

- Evaluation 不参与模型训练
- Evaluation 不参与 Top-K 选择
- Reject 是系统输出，不是数据集的第四类标签
- 各设计线使用相同评估窗口

**可用图片**

- `slides/assets/bciciv_experiment_split.png`
- `hybrid-photonic-mi-bci/artifacts/figures/bnci2014_004_personalization/bnci_subject_line_comparison.png`

## 3.9 核心实验结果

建议用四个大数字卡展示 BCICIV 主线：

- Accuracy：70.46%
- Balanced Accuracy：73.24%
- Accepted Accuracy：72.42%
- Reject Rate：2.70%

旁边放混淆矩阵和累计指标图，图注保持短句：

- 518 个独立 evaluation 窗口
- Accepted Accuracy 只在未拒识窗口上计算
- 累计曲线用于观察整体稳定趋势

**可用图片**

- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/experience_photonic/mainline_confusion.png`
- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/experience_photonic/mainline_cumulative_metrics.png`

如展示三线对比，应诚实标注：

- LDA：71.25%
- Small MLP：72.32%
- Mainline：70.46%
- 主线当前未稳定超过 LDA baseline，但已验证完整的经验库特化与候选扫描路径

不要只挑选有利指标隐藏 baseline。

## 3.10 创新点

建议用三个编号卡片，每张控制在两行以内：

**01｜FBCSP 光计算主干**

将适合小样本 MI-BCI 的经典方法与线性计算后端结合，而不是完全替换为大型深度网络。

**02｜经验库会话特化**

通过少量校准检索候选模型组合，应对 EEG 明显的个体与会话差异。

**03｜多候选低位宽扫描**

将候选 Head Bank 组织为 TiledMVM 扫描路径，实现 uint4/int4 的重点低位宽量化。

## 3.11 项目边界与下一步

建议使用“已完成 / 待完成”两列，避免夸大工程状态。

**已完成**

- 公开 EEG 数据离线回放
- FBCSP、embedding、经验库与候选扫描主线
- 三类统一 Backend 接口
- 软件量化与执行核模拟
- 计算量和 Backend 路由账本

**待完成**

- Cyton 上位机与模拟流
- 真实 EEG 在线采集闭环
- Gazelle/评估板端到端实测
- 板端误差、通信时延和功耗测量
- 逐算子最低可用精度搜索

建议在边界区放一句醒目标注：

> 公开数据回放 ≠ 真实在线；软件模拟 ≠ 真实光芯片实测。

---

## 4. 推荐的海报视觉主次

### 一级视觉：必须一眼看到

- 项目标题与一句话定位
- EEG 到命令的一行核心流程
- 经验库会话特化流程
- 统一 Backend 与 100% 前向线性接管口径
- 四个核心指标

### 二级视觉：靠近后可以读懂

- 三阶段流程
- 混合量化策略
- Train/Calibration/Evaluation 划分
- 三个创新点

### 三级视觉：用于回答追问

- 统计口径说明
- baseline 对比
- 当前工程边界
- 后续评估板与 Cyton 工作

---

## 5. 图片取舍建议

海报不宜塞入 PPT 的所有图片。建议主版最多使用 6–8 张图：

1. 一行核心流程图——建议重新绘制成海报主视觉。
2. 经验库会话特化流程——保留。
3. 统一 Backend 接口框图——保留或简化。
4. BCICIV 数据划分图——保留。
5. 主线混淆矩阵——保留。
6. 累计指标图——保留。
7. 三阶段流程——根据空间三选一或合并为简图。
8. BNCI subject-wise 图——空间足够时作为补充证据。

以下内容可以不放或缩小：

- 完整系统详细框图：信息密度过高。
- 大段网络结构参数：不适合海报阅读距离。
- 完整训练曲线：除非需要回答 encoder 收敛问题。
- 多张相似结果图：优先保留混淆矩阵和累计指标。

---

## 6. 可直接放在海报上的摘要

运动想象脑机接口需要从短时 EEG 窗口识别用户意图，但信号具有低信噪比、小样本、非平稳和显著个体差异。本项目以稳定、可解释的 FBCSP 为主干，使用小型 MLP 构建 embedding，并通过经验库与少量会话校准选择 Top-K 候选线性头。前向线性计算统一经 MatrixOps、SignalOps 和 TiledMVM Backend 路由，其中候选 Bank 采用 uint4/int4 低位宽扫描；非线性、控制和拒识仍在数字端。BCICIV 的 518 个独立评估窗口上，主线取得 70.46% Accuracy、73.24% Balanced Accuracy 和 2.70% Reject Rate。当前验证基于公开数据离线回放与软件模拟，真实 Cyton 在线闭环和评估板端到端性能仍待完成。

---

## 7. 海报讲解顺序（约 90 秒）

1. **问题（15 秒）**：MI-EEG 个体差异明显，因此需要少量校准后的会话特化。
2. **方案（25 秒）**：介绍 EEG → FBCSP → embedding → 经验库 →候选扫描 → 命令/拒识。
3. **光计算（20 秒）**：三类 Backend 接管全部前向线性 MAC；候选扫描采用 uint4/int4，非线性与控制仍在数字端。
4. **实验（20 秒）**：说明 840/42/518 严格划分并报告四个主线指标。
5. **边界（10 秒）**：当前为公开数据回放和软件模拟，真实 Cyton 与评估板端到端实测是下一步。

---

## 8. 排版注意事项

- 正文尽量使用短语，每个模块不超过 4 条要点。
- 主标题建议 80–110 pt，模块标题 34–44 pt，正文不低于 24–28 pt（A0 参考）。
- 颜色沿用现有素材：蓝色表示主干/通用矩阵，绿色表示信号或评估，橙色表示候选扫描，紫色表示 Backend/执行边界。
- 图表必须带简短结论式图注，不只写文件名或指标名。
- 不使用低清终端截图；优先使用已有 PNG/PDF 或重新绘制的矢量图。
- 所有数字在提交前再次对照当前 metrics，避免实验更新后海报数据滞后。
- 不声称真实在线、真实光芯片精度、功耗或端到端加速已经完成。

