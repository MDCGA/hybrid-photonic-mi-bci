# Hybrid Photonic MI-BCI 技术文档大纲

本文档是技术文档写作大纲，不是最终完整版正文。每一节给出建议展开内容、应放图表、以及需要特别注意的边界。

## 1. 项目概述

### 1.1 背景与问题定义

需要说明：

- 运动想象 EEG-BCI 的基本任务：从 EEG 窗口中识别用户意图。
- EEG 信号小样本、低信噪比、跨被试差异大的特点。
- 传统 FBCSP 方法稳定但个体适配能力有限。
- 深度学习方法表达能力强，但在小样本 EEG 场景中容易过拟合。
- 光计算适合高并行、低功耗线性计算，但需要明确接入位置。

建议图表：

- MI-BCI 任务示意图。
- EEG 决策窗口到控制命令的流程图。

注意点：

- 不要把项目写成纯深度学习 BCI。
- 不要把光计算写成替代全部算法，而是加速/接管可线性化的前向计算路径。

### 1.2 项目目标

需要说明：

- 构建一个 FBCSP 主导、经验库增强、可接光计算的 MI-BCI 原型。
- 在公开数据集上验证三条实验线。
- 通过少量校准样本选择经验库候选头，实现模型特化。
- 将矩阵乘、线性头扫描、部分信号处理路径统一暴露到 backend。
- 为后续 OpenBCI Cyton 上位机和真实光芯片接入预留结构。

建议用一句话概括主线：

```text
FBCSP + small MLP embedding + experience library retrieval + photonic candidate scan
```

注意点：

- 当前是公开数据集回放验证，不是真实在线闭环验证。
- 当前光计算路径是接口和量化模拟，不是真实光芯片实测。

### 1.3 系统输出定义

需要说明：

- 系统命令输出：

```text
left / right / foot / reject
```

- `left/right/foot` 来自 MI 命令类别。
- `reject` 来自置信度/边际阈值，是拒识输出。

注意点：

- 不要把 `reject` 写成数据集原生第四类。
- 如果文档中写“四分类”，要解释为“三个运动想象命令 + 一个系统拒识状态”。

## 2. 系统总体架构

### 2.1 总体流程

建议框图必须包含：

```text
Raw EEG window
  -> Common average reference
  -> Band-pass / filter bank
  -> FBCSP spatial projection
  -> log-variance feature
  -> Fisher feature selection
  -> feature standardization
  -> small MLP encoder
  -> embedding vector
  -> experience library retrieval
  -> candidate linear heads
  -> tiled photonic MVM scan
  -> probability fusion
  -> confidence / margin reject
  -> command output
```

建议图表：

- 系统总体框图。
- 在线推理路径图。
- 训练/校准/推理三阶段分层图。

注意点：

- 框图中必须写出 FBCSP，而不是只写 feature extraction。
- 分类路径必须写出 candidate linear heads 和 probability fusion。
- photonic scan 要放在候选线性头扫描位置，不要画成替代整个 EEG pipeline。

### 2.2 三个阶段划分

#### 离线训练阶段

包含：

- 数据集加载。
- FBCSP 拟合。
- Fisher 特征选择。
- 标准化参数拟合。
- 小型 MLP 训练。
- 初始经验库构建。

#### 校准阶段

包含：

- 新被试/新会话少量校准样本输入。
- 计算 calibration embeddings。
- 查询经验库。
- 选择 top-K 候选经验头。
- 设置或更新拒识阈值。

#### 在线推理阶段

包含：

- 单个 EEG 决策窗口输入。
- FBCSP transform。
- 标准化。
- MLP embedding。
- 候选线性头 photonic scan。
- 概率融合。
- 拒识判断。

注意点：

- setup/calibration 不算单次前向推理。
- 单次前向只对应一个 EEG decision window 的在线路径。

## 3. 算法路线设计

### 3.1 Baseline 1: FBCSP + shrinkage LDA

需要展开：

- Filter bank 的频段设置。
- CSP 的 one-vs-rest 多类处理。
- log-variance 特征。
- Fisher score 特征筛选。
- shrinkage LDA 的作用。
- 拒识阈值如何从训练/校准统计中得到。

建议图表：

- FBCSP 特征维度示意。
- LDA baseline 流程图。
- baseline 混淆矩阵。

重点解释：

- 为什么把它作为主 baseline：稳定、可解释、小样本友好。
- 为什么不用普通 LDA：shrinkage 对 EEG 小样本协方差估计更稳。

### 3.2 Baseline 2: FBCSP + small MLP embedding

需要展开：

- 输入是筛选后的 FBCSP 特征。
- MLP 层数、hidden dim、embedding dim。
- MLP 输出包括 embedding 和分类 logits。
- 训练曲线：loss / accuracy。
- 与 FBCSP + LDA 的差异。

建议图表：

- 小型 MLP 结构图。
- 训练 loss 曲线。
- 训练 accuracy 曲线。
- embedding 可视化，可选 PCA/t-SNE。

重点解释：

- 小型网络不是为了取代 FBCSP，而是把 FBCSP 特征映射到更适合经验库检索的空间。
- 避免将其描述为大型端到端 EEG 深度网络。

### 3.3 Mainline: FBCSP + MLP embedding + experience library + photonic scan

需要展开：

- 经验库条目结构：
  - entry id
  - centroid
  - linear head
  - source
  - train indices
  - train accuracy
- 经验库构建方式：
  - anchor heads
  - bootstrap heads
  - embedding LDA heads
- 校准样本如何查询经验库。
- top-K 候选头如何选出。
- 候选头如何通过 photonic scan 得到 candidate scores。
- 多候选头概率如何融合。
- reject 如何产生。

建议图表：

- 经验库结构图。
- 经验库检索流程图。
- top-K 候选头权重图。
- candidate scan tile schedule。
- 主线混淆矩阵。

重点解释：

- 经验库不是简单样本缓存，而是候选模型/候选线性头集合。
- 主线的核心创新是“校准样本检索经验库 + 多候选头扫描 + 融合决策”。
- photonic scan 的意义是一个 EEG 决策窗口内快速扫描多个候选头。

## 4. 数据集与实验协议

### 4.1 BCICIV_1_asc

需要展开：

- 数据集文件结构。
- a-g 文件含义。
- 每个文件原始类别可能不同。
- 工程如何合并为 pooled dataset。
- 全局类别映射：

```text
left / right / foot
```

- train / calibration / evaluation 的划分规则。
- 三条实验线如何在同一划分下对比。

建议表格：

| 项目 | 内容 |
| --- | --- |
| 数据集 | BCICIV_1_asc |
| 使用文件 | a-g |
| 命令类别 | left/right/foot |
| 系统额外输出 | reject |
| 训练用途 | FBCSP/MLP/经验库 |
| 校准用途 | 经验库检索 |
| 评估用途 | 在线推理回放 |

注意点：

- a-g 是采集文件，不是最终类别。
- `reject` 不能写进数据集标签。
- evaluation windows 不能参与训练或经验库选择。

### 4.2 BNCI2014_004

需要展开：

- 数据集基本情况。
- 被试/session 组织方式。
- 为什么适合验证个体特化。
- 如何划分 history、calibration、evaluation。
- 三条线在 BNCI 上如何比较。

建议图表：

- 每个被试 session 划分示意。
- 个体特化前后结果对比表。
- subject-wise accuracy / reject rate 图。

注意点：

- 验证经验库是否有用，本质是看特化前后对目标被试的效果。
- 不应只做跨数据集总平均，需要保留 subject-wise 结果。

## 5. 光计算接口设计

### 5.1 MatrixOpsBackend

需要展开：

- 为什么统一封装矩阵乘。
- 当前哪些操作通过该接口。
- 后续真实光芯片如何接管。

应说明典型操作：

- LDA scores
- MLP linear layers
- candidate linear head scores
- probability fusion 中的线性组合
- 部分 einsum 路径

### 5.2 SignalOpsBackend

需要展开：

- CAR 是否属于线性计算。
- SOS filtering 是否计入前向线性计算。
- 当前如何统计 signal ops。

注意点：

- 文档要解释“全部线性计算”的口径。
- 只统计前向传播，不统计训练反向传播。

### 5.3 TiledMVMBackend

需要展开：

- `2 x 8` tile 的含义。
- 大矩阵如何拆成 tile。
- row block / column block / partial sum。
- tile evaluations per window 如何计算。

注意点：

- `2 x 8` 是硬件 tile，不是系统只能处理 `2 x 8`。
- 大矩阵通过滑动窗口、拆分、拼接完成。

### 5.4 PurePhotonicScanRuntime

需要展开：

- 为什么需要 pure runtime。
- 它只保留部署前向。
- 它不包含训练、评估、绘图。
- 典型调用方式：

```python
runtime.calibrate(calibration_embeddings)
outputs = runtime.predict(online_embeddings)
```

注意点：

- `calibrate` 是校准阶段，不是每个窗口都跑。
- 单次前向是 `predict(one_window_embedding)`。

## 6. 量化策略

### 6.1 当前支持

需要写清楚：

```text
支持：4-bit / 8-bit
默认：4-bit
当前主线：4-bit
```

### 6.2 4-bit 配置

```text
input activation: uint4, qin=[0, 15]
weight:           int4,  qwt=[-8, 7]
```

需要解释：

- 4-bit 是当前工业甜点位宽。
- 位宽越低，潜在抗噪能力更强。
- 当前 photonic candidate scan 默认使用该配置。

### 6.3 8-bit 配置

```text
input activation: uint8, qin=[0, 255]
weight:           int8,  qwt=[-128, 127]
```

需要解释：

- 8-bit 是支持项。
- 不是当前默认实验结果。
- 后续可用于精度/抗噪/动态范围对比。

### 6.4 当前实现边界

需要说明：

- photonic scan 实际使用低位宽量化路径。
- 其他 MatrixOps 路径当前更多是接口接管和高精度模拟。
- 后续可以继续扩大量化覆盖范围。

注意点：

- 不要把所有路径都写成已经完整 4-bit QAT。
- 如果写 QAT，需要明确当前是否真的训练时量化。

## 7. 前向传播、进度与耗时

### 7.1 Setup timings

需要列出：

- load dataset
- fit FBCSP
- transform replay
- feature selection
- standardization
- train MLP
- build experience library
- calibrate runtime
- calibrate reject threshold

注意点：

- 这些不算单次前向推理。
- 它们属于准备、训练或校准。

### 7.2 Single online forward timings

需要列出：

- FBCSP transform one window
- standardize one window
- small MLP forward one window
- pure runtime photonic scan one window
- online forward total

建议表格：

| 阶段 | 是否属于单次前向 | 说明 |
| --- | --- | --- |
| FBCSP transform | 是 | 单窗口特征提取 |
| MLP forward | 是 | embedding 计算 |
| photonic scan | 是 | 候选头扫描 |
| train MLP | 否 | 离线训练 |
| calibrate threshold | 否 | 校准阶段 |

### 7.3 实时进度与准确率

需要说明：

- 进度条显示 evaluation scan 进度。
- `acc` 是累计 command accuracy。
- `accepted_acc` 是非拒识样本累计准确率。
- `reject` 是累计拒识率。

注意点：

- 实时准确率只适用于公开数据集回放，因为有标签。
- 真实在线部署时没有即时标签，不能实时显示 accuracy。

## 8. 计算量统计

### 8.1 统计口径

需要说明：

- 只统计 forward-only。
- 统计全部线性计算。
- 区分 photonic linear MACs 和 digital linear MACs。

### 8.2 当前统计项

建议展开：

- preprocessing linear ops
- FBCSP spatial projection
- feature standardization affine
- MLP forward linear layers
- LDA / linear head scores
- candidate photonic scan
- probability fusion

### 8.3 结果解释

注意点：

- photonic share 是计算量占比，不是真实功耗占比。
- 当前是工程估算，不是芯片功耗实测。
- 训练反向传播不计入前向光计算占比。

## 9. 工程结构

需要展开各目录职责：

```text
hybrid_photonic_mi_bci/
  backends.py
  fbcsp.py
  small_networks.py
  experience.py
  progress.py
  workflows/
  datasets/
  host/

pure_runtime/
examples/
artifacts/
visualization/
docs/
```

建议说明：

- `backends.py`: 光计算接口和矩阵计算替换边界。
- `fbcsp.py`: FBCSP 特征提取。
- `small_networks.py`: 小型 MLP。
- `experience.py`: 经验库和候选头扫描。
- `workflows/`: 三条实验线和 BNCI 验证。
- `pure_runtime/`: 无评估、无绘图的纯前向部署代码。
- `examples/`: 命令行入口。
- `artifacts/`: 运行结果。
- `visualization/`: 绘图脚本和图片。

注意点：

- 文档中强调工程目录保持清晰。
- 不要把测试输出、临时文件、绘图代码混在源码根目录。

## 10. 运行方式

### 10.1 完整三线对比

```bash
python examples/run_fbcsp_design_comparison.py
```

说明输出：

- 三条实验线结果。
- summary.json。
- compute_accounting.json。
- run_progress.json。

### 10.2 只跑主线

```bash
python examples/run_experience_photonic_line.py
```

说明输出：

- 主线 summary。
- arrays.npz。
- 在线进度条。

### 10.3 单个样本全流程推理

```bash
python examples/run_single_window_inference.py --evaluation-index 0
```

需要解释：

- `--evaluation-index` 是 evaluation split 内部索引。
- 不是原始数据文件里的第几个 trial。

### 10.4 BNCI 个体特化验证

```bash
python examples/run_bnci2014_004_personalization.py
```

说明输出：

- subject-wise 结果。
- 三条线对比。
- 个体特化效果。

## 11. 可视化设计

### 11.1 必须包含的图

- 系统框图。
- 训练 loss / accuracy。
- 混淆矩阵。
- rolling command accuracy。
- rolling reject rate。
- cumulative command accuracy。
- cumulative reject rate。
- photonic tile schedule。
- compute accounting summary。
- 经验库候选头质量和权重分布。

### 11.2 图表注意点

- rolling 曲线是滑动窗口统计。
- cumulative 曲线是从第一个 evaluation window 开始累计。
- reject rate 图必须确认分母是全部 evaluation windows。
- 图表标题要写明数据集和实验线。

## 12. 实验结果分析

### 12.1 三条实验线对比

需要比较：

- command accuracy
- balanced command accuracy
- accepted accuracy
- reject rate
- forward MACs
- photonic share
- inference share

### 12.2 经验库特化分析

需要分析：

- 校准样本数量对结果的影响。
- top-K 候选数量对结果的影响。
- 经验库条目数量对结果的影响。
- 不同被试上的收益差异。

### 12.3 错误与拒识分析

需要分析：

- 哪些类别最容易混淆。
- reject 是否集中在低置信度窗口。
- accepted accuracy 与 reject rate 的权衡。

注意点：

- 不要只报告总 accuracy。
- 主线和 baseline 必须使用相同 evaluation split。

## 13. OpenBCI Cyton 上位机扩展

需要规划：

- Cyton 数据采集。
- 实时 EEG buffer。
- 窗口切片。
- 在线推理调用。
- 经验库新增。
- 经验库删除。
- 经验库分组。
- 被试/session 元数据管理。
- 校准流程管理。
- 运行状态显示。

注意点：

- 当前还不能自行采集脑电完成完整闭环。
- 文档中写成后续扩展或上位机原型设计。

## 14. 局限性

建议逐条写：

- 当前主要是公开数据集回放。
- 当前 photonic backend 是软件模拟或接口占位。
- 当前默认 4-bit 主要用于 candidate scan。
- 其他路径仍需进一步真实硬件接管。
- FBCSP/滤波实时优化仍有空间。
- EEG 跨被试差异仍是主要挑战。
- 真实在线无法直接获得实时准确率。

## 15. 后续工作

建议分方向写：

- 真实光芯片接入。
- 更完整的 QAT 或 PTQ 量化实验。
- OpenBCI Cyton 上位机。
- 经验库管理系统。
- 真实被试采集。
- 在线校准协议。
- 延迟、功耗、抗噪测试。
- 图表和报告自动生成。

## 16. 文档检查清单

- [ ] 是否明确当前默认量化位宽是 4-bit。
- [ ] 是否说明 8-bit 是支持项，不是默认实验设置。
- [ ] 是否说明 `2 x 8` 是 tile 尺寸，不是系统矩阵大小限制。
- [ ] 是否说明 `reject` 是拒识输出，不是训练类别。
- [ ] 是否区分 setup、calibration、online forward。
- [ ] 是否区分 rolling accuracy 和 cumulative accuracy。
- [ ] 是否说明实时 accuracy 只适用于离线回放。
- [ ] 是否说明当前不是真实光芯片实测结果。
- [ ] 是否保留 FBCSP + shrinkage LDA baseline。
- [ ] 是否解释经验库的模型特化意义。
- [ ] 是否给出可复现命令和结果路径。

