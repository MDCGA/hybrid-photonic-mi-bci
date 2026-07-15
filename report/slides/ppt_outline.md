# 光电混合运动想象脑机接口 PPT 大纲

建议规模：14 页；讲解约 10 分钟；演示另计。  
匿名要求：不出现学校、Logo、指导教师、成员姓名、实验室名称。

## 第 1 页｜作品名称

**页面文字**

- 光电混合运动想象脑机接口
- FBCSP · 经验库特化 · Photonic Scan
- 参赛编号：现场填写

**图片**

- 主图：`template/figure/competition/mi_bci_task_overview.png`

**讲解重点**

- EEG 意图识别
- 三类命令 + 拒识
- 软件光计算原型

## 第 2 页｜赛题与作品定位

**页面文字**

- 复赛任务：创新 AI 应用
- 光计算占比：≥ 50%
- 应用方向：MI-BCI
- 当前验证：公开数据回放

**图片**

- 左侧：`info/赛题.png` 局部截图（任务与复赛要求）
- 右侧：`template/figure/competition/mi_bci_task_overview.png`

**讲解重点**

- 赛题要求对应关系
- 非 MNIST 初赛路线
- 真实在线闭环未完成

## 第 3 页｜问题与需求

**页面文字**

- EEG：低信噪比
- 数据：小样本
- 状态：非平稳
- 个体：跨被试差异
- 输出：避免误触发

**图片**

- `template/figure/competition/eeg_mi_signal_concepts.png`
- 可选小图：`template/figure/competition/eeg_1020_electrodes.jpg`

**讲解重点**

- 8–30 Hz mu/beta
- ERD/ERS 概念
- 拒识必要性

## 第 4 页｜核心方案

**页面文字**

- EEG → FBCSP
- → Small MLP Embedding
- → Experience Library
- → Photonic Scan
- → Reject / Command

**图片**

- `template/figure/competition/end_to_end_inference_chain.drawio.png`

**讲解重点**

- FBCSP 主干
- 小型 MLP 非端到端替代
- 候选扫描重点路径

## 第 5 页｜训练、校准、在线推理

**页面文字**

- 训练：FBCSP / MLP / 经验库
- 校准：42 窗口 / Top-K
- 推理：单窗口前向
- 边界：Train ≠ Calibration ≠ Evaluation

**图片**

- 离线训练：`slides/assets/stage_offline_training.png`
- 会话校准：`slides/assets/stage_session_calibration.png`
- 在线推理：`slides/assets/stage_online_inference.png`

**讲解重点**

- 数据隔离
- 校准不计入评估
- 单窗口不重新训练

## 第 6 页｜FBCSP 主干

**页面文字**

- 6 个运动相关频带
- OVR CSP 空间投影
- Log-variance
- 72 → 32 维
- 小样本 · 可解释

**图片**

- `template/figure/competition/fbcsp_matrix_flow.drawio.png`

**讲解重点**

- 不是普通 baseline
- 线性计算密集
- CSP：`W(4×8) · X(8×T)`

## 第 7 页｜个体差异与经验库

**页面文字**

- 2 个 Anchor Heads
- 64 个 Bootstrap Heads
- 42 个校准窗口
- Top-K = 8
- 会话特化

**图片**

- `template/figure/competition/specialization_logic.png`
- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/experience_photonic/mainline_retrieval_weights.png`

**讲解重点**

- 经验库 ≠ 数据缓存
- 候选模型集合
- 距离 + 训练质量 + 校准表现

## 第 8 页｜Photonic Candidate Scan

**页面文字**

- Head Bank：`(8, 3, 33)`
- Tile：`2 × 8`
- 每候选：10 Tiles
- 每窗口：80 Tiles
- 输出：Candidate Scores

**图片**

- `template/figure/competition/photonic_tile_mapping.drawio.png`
- 辅图：`hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/experience_photonic/photonic_tile_schedule.png`

**讲解重点**

- 多候选 MVM
- 非频率扫描
- 非固定单一分类头
- 数字端融合与拒识

## 第 9 页｜统一后端接口

**页面文字**

- MatrixOps：矩阵 / 张量
- SignalOps：CAR / SOS
- TiledMVM：候选 Bank
- 上层算法与执行核解耦

**图片**

- 建议新绘：三层 backend 接口框图
- 备用：`template/figure/competition/system_block_diagram.png`

**讲解重点**

- 全部前向线性算子接管
- 非线性与控制仍在数字端
- 当前执行核：软件模拟

## 第 10 页｜量化策略

**页面文字**

- Candidate Scan：单次 4-bit
- 输入：uint4 `[0,15]`
- 权重：int4 `[-8,7]`
- 其他路径：8-bit Logic
- Radix-16 / 4-bit Slices

**图片**

- 建议新绘：4-bit 单次调用 vs 8-bit 位权拆分对比图
- 备用：`template/figure/competition/tile_schedule.png`

**讲解重点**

- 逻辑精度 ≠ 物理位宽
- 动态量化 ≠ QAT
- 精度 / Slice / 延迟权衡

## 第 11 页｜实验协议

**页面文字**

- BCICIV：3 类 MI
- Train：840
- Calibration：42
- Evaluation：518
- BNCI：9 被试 / 1296 窗口

**图片**

- `hybrid-photonic-mi-bci/artifacts/figures/bnci2014_004_personalization/bnci_subject_line_comparison.png`
- 建议新绘：BCICIV Train / Calibration / Evaluation 划分图

**讲解重点**

- 相同评估边界
- Reject 非数据集标签
- Subject-wise 结果保留

## 第 12 页｜核心结果

**页面文字**

- BCICIV 主线
- Accuracy：70.46%
- Balanced：73.24%
- Accepted：72.42%
- Reject：2.70%

**图片**

- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/experience_photonic/mainline_confusion.png`
- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/experience_photonic/mainline_cumulative_metrics.png`

**讲解重点**

- 518 个独立评估窗口
- 类别混淆
- Accuracy / Reject 权衡

## 第 13 页｜三线对比与光计算账本

**页面文字**

- LDA：71.25%
- Small MLP：72.32%
- Mainline：70.46%
- Forward Linear Share：100%
- Inference MAC：342.105 M

**图片**

- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/summary/design_line_summary.png`
- `hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/compute_accounting/compute_accounting_summary.png`

**讲解重点**

- 主线未稳定超过 LDA
- 已验证完整特化路径
- 100% = Backend 接管口径
- 非功耗 / 非真实芯片占比

## 第 14 页｜创新、边界与下一步

**页面文字**

- 创新 1：FBCSP 光计算主干
- 创新 2：经验库会话特化
- 创新 3：多候选低位宽扫描
- 未完成：Cyton 在线闭环
- 下一步：真实芯片 / 混合精度

**图片**

- `template/figure/competition/specialization_logic.png`
- 可选角标：`template/figure/competition/photonic_tile_mapping.drawio.png`

**讲解重点**

- 软件接口与工程证据
- 不夸大硬件和在线能力
- 收束到赛题价值

## 备用页 A｜小型 MLP 与 Embedding

**页面文字**

- 32 → 64 → 32
- Training Loss
- Embedding PCA
- Small-network Replay

**图片**

- 使用最新图：`hybrid-photonic-mi-bci/artifacts/figures/fbcsp_design/small_network/small_network_training_embedding.png`
- 不使用旧图：`template/figure/competition/embedding_diagnostics.png`

## 备用页 B｜BNCI 个体结果

**页面文字**

- 9 Subjects
- Mainline：74.92%
- Accepted：76.22%
- Reject：1.62%
- 个体收益差异

**图片**

- `hybrid-photonic-mi-bci/artifacts/figures/bnci2014_004_personalization/bnci_subject_line_comparison.png`
- `hybrid-photonic-mi-bci/artifacts/figures/bnci2014_004_personalization/bnci_design_line_summary.png`

## 备用页 C｜工程与复现

**页面文字**

- 30 Tests Passed
- JSON / NPZ
- Single-window Inference
- Pure Runtime
- Reproducible Figures

**图片**

- 建议新绘：工程目录与产物关系图
- 可选终端截图：测试通过、单窗口推理输出

## 制作约束

- 每页 1 个结论
- 每页 ≤ 5 条要点
- 每条 ≤ 14 个汉字或一个短语
- 结果页：数字字号优先
- 图表：裁掉多余留白
- 英文图：现场口头中文解释
- 主展示：纯图片版 PPT
- 备用：可编辑源文件
- 视频：1080P、MP4、≤ 5 分钟
- 全材料：匿名检查

# 逐页答辩脚本

## 第 1 页｜作品名称（20 秒）

各位评委好，我们的作品是“光电混合运动想象脑机接口”。项目使用 FBCSP 提取可解释的运动想象脑电特征，通过小型 MLP 和经验库完成会话特化，再用低位宽 photonic candidate scan 扫描多个候选分类头。系统输出左手、右手、脚三类命令；结果不可靠时输出拒识。

**过渡**：下面先说明本作品与赛题要求的对应关系。

## 第 2 页｜赛题与作品定位（30 秒）

复赛要求基于光计算平台开发创新 AI 应用，并要求光计算占比不低于 50%。我们选择运动想象脑机接口作为应用方向，将适合矩阵和向量运算的前向线性计算统一交给 photonic backend。当前结果来自公开 EEG 数据回放和软件量化模拟，不是 Cyton 真实在线闭环，也不是真实光芯片功耗实测。

**过渡**：选择这个方向，首先是因为运动想象 EEG 本身存在几个典型难点。

## 第 3 页｜问题与需求（40 秒）

EEG 信号幅值弱、信噪比低，数据规模又远小于图像任务，并且会受到眨眼、肌电、电极接触和状态变化影响。运动想象主要关注 8 到 30 赫兹附近的 mu 和 beta 节律，提示出现后常发生 ERD 功率下降。不同被试的有效频段和空间激活模式差异明显，所以系统既要保持小样本稳定性，也要允许会话特化，还要通过拒识减少错误控制命令。

**过渡**：针对这些问题，我们形成了下面这条核心流程。

## 第 4 页｜核心方案（40 秒）

系统首先从 EEG 窗口提取 FBCSP 时频和空间特征，再由小型 MLP 映射到紧凑 embedding。少量校准窗口用于查询经验库，选择并加权多个候选线性头；随后 photonic scan 计算候选得分，数字端完成 softmax、概率融合和拒识判断。这里 FBCSP 是主干，小型网络只改善特征空间，并不是纯深度学习端到端 BCI。

**过渡**：整个系统需要严格区分训练、校准和在线推理三个阶段。

## 第 5 页｜训练、校准、在线推理（45 秒）

离线训练阶段拟合 FBCSP、特征选择、标准化和小型 MLP，并构建经验库。校准阶段只使用少量目标会话窗口生成 embedding，查询 top-K 候选并冻结融合权重和拒识阈值。在线推理阶段每次只处理一个新窗口，不再训练模型。BCICIV 主线使用 42 个校准窗口，剩余 518 个窗口独立评估，避免训练、校准和评估之间的数据泄漏。

**过渡**：三个阶段共享的特征主干是 FBCSP。

## 第 6 页｜FBCSP 主干（45 秒）

FBCSP 将 EEG 分到六个运动相关频带，对每个类别执行 one-versus-rest CSP 空间投影，再计算归一化对数方差。原始特征共 72 维，通过 Fisher 分数筛选为 32 维。它适合小样本，频带和空间模式也便于解释。更重要的是，CSP 投影、协方差和 Gram 矩阵包含大量规则线性计算，因此 FBCSP 不只是 baseline，也是光计算接管的重要主干。

**过渡**：FBCSP 解决特征稳定性，经验库进一步处理跨被试和跨会话差异。

## 第 7 页｜个体差异与经验库（45 秒）

不同被试的节律频段、空间激活和想象策略并不一致，所以我们没有只依赖一个固定全局模型。当前经验库包含两个全局 anchor heads 和 64 个 bootstrap heads。42 个校准窗口产生当前会话 embedding，并综合距离、训练质量、校准准确率和置信度选择 top-8。图中的两个 anchor 权重较高，其他候选提供会话特化补充。经验库存储的是候选模型和质量统计，不是普通样本缓存。

**过渡**：多个候选头会自然形成一组规则的矩阵向量乘任务。

## 第 8 页｜Photonic Candidate Scan（50 秒）

八个候选头堆叠后的 weights shape 是 8 乘 3 乘 33，其中 33 包括 32 维 embedding 和一个偏置常数。物理 tile 为 2 乘 8，因此每个候选需要 2 个行块和 5 个列块，也就是 10 次 tile；每个窗口共 80 次。Scan 输出的是八组候选分类分数，不是最终类别。数字端再执行 softmax、检索权重融合和拒识。这里的 scan 是候选分类头扫描，不是 Goertzel 或频率扫描。

**过渡**：为了让这些算法算子能够替换执行核，我们设计了三个统一后端接口。

## 第 9 页｜统一后端接口（45 秒）

MatrixOpsBackend 负责通用 matmul 和 einsum，覆盖 CSP、标准化、MLP、LDA、距离项和概率融合。SignalOpsBackend 负责带通道轴或时间轴语义的 CAR 和 SOS 滤波。TiledMVMBackend 负责候选 head bank 的分块扫描和 tile 计数。上层算法只依赖接口，不直接依赖硬件。当前执行核仍是软件模拟，未来可以在不改动决策逻辑的情况下替换为官方模拟器或真实芯片驱动。

**过渡**：接口统一以后，还需要明确逻辑精度和物理调用位宽的区别。

## 第 10 页｜量化策略（50 秒）

候选扫描直接使用单次 4-bit：输入是 0 到 15 的 uint4，权重是负 8 到 7 的 int4。其他前向线性路径默认保留 8-bit 逻辑精度，但会采用 radix-16 拆成多个 4-bit slice，再按位权重构。因此 8-bit 逻辑精度不是一次原生 8-bit 光计算调用。当前采用运行时动态量化，不是 QAT。后续需要逐算子搜索最低可用精度，联合比较准确率、拒识率、slice 数、噪声、延迟和能耗。

**过渡**：下面说明这些设计采用什么数据和评估协议验证。

## 第 11 页｜实验协议（40 秒）

BCICIV 用于三类运动想象主流程验证。训练集合用于 FBCSP、MLP 和经验库构建，42 个窗口只用于校准，518 个窗口只用于最终回放评估。BNCI 数据包含 9 位被试，用 session 1 和 2 作为历史数据，session 3 作为目标会话，共保留 1296 个评估窗口。Reject 不是数据集原生类别，而是系统根据置信度额外产生的安全状态。

**过渡**：在这个协议下，主线得到以下结果。

## 第 12 页｜核心结果（45 秒）

在 BCICIV 的 518 个独立评估窗口上，主线 command accuracy 为 70.46%，balanced accuracy 为 73.24%，accepted accuracy 为 72.42%，reject rate 为 2.70%。混淆矩阵显示 left 和 right 仍有明显混淆，foot 样本相对较少。累计曲线用于观察回放稳定性。拒识率不能单独追求更高，它需要与 accepted accuracy 和可输出命令数量共同分析。

**过渡**：接下来把主线与两个 baseline 以及计算账本放在一起看。

## 第 13 页｜三线对比与光计算账本（55 秒）

同一协议下，LDA baseline 的准确率是 71.25%，小型 MLP 是 72.32%，主线是 70.46%。因此我们不声称经验库已经稳定提升平均准确率；当前已验证的是校准、特化、多候选扫描和拒识的完整路径。按 forward-only 账本，342.105 M 前向线性 MAC 全部路由到 photonic backend，接管比例为 100%。这个数字是接口和 MAC-equivalent 口径，不等于真实芯片功耗占比或实际加速比。

**过渡**：最后总结创新点、当前边界和下一步工作。

## 第 14 页｜创新、边界与下一步（50 秒）

项目有三个主要设计点：第一，FBCSP 既提供可解释的小样本主干，也暴露大量可接管线性计算；第二，经验库针对 MI-EEG 个体差异完成少量校准后的会话特化；第三，利用多候选结构设计低位宽 photonic scan，而不是只替换一个矩阵乘。目前 Cyton 在线闭环和真实光芯片实测尚未完成。下一步将接入真实驱动，开展逐算子混合精度、噪声、延迟和功耗测试，并完成真实被试在线验证。谢谢各位评委。

# 备用页答疑脚本

## 备用页 A｜小型 MLP 与 Embedding

这页用于回答小型网络是否正常训练。左上显示 loss 下降和训练准确率上升；右上是 32 维 embedding 的 PCA 二维投影，仅用于可视化，不是实际分类空间；下方是独立回放的滚动指标和混淆矩阵。训练准确率较高，因此必须结合回放结果判断泛化。这里应使用当前生成的 small-network 图，而不是旧版 embedding diagnostics。

## 备用页 B｜BNCI 个体结果

BNCI 用来观察跨被试和目标会话差异。九位被试的主线平均准确率为 74.92%，accepted accuracy 为 76.22%，reject rate 为 1.62%。LDA 平均值仍略高，因此经验库价值需要看 subject-wise 收益，而不能只看总体平均。后续会对校准窗口数、top-K 和经验库规模做消融。

## 备用页 C｜工程与复现

工程提供完整三线、单主线、单窗口和 BNCI 四类运行入口。JSON 保存 summary 和计算账本，NPZ 保存概率、embedding、rolling、cumulative 和混淆矩阵原始数组，绘图脚本可以重新生成所有结果图。当前回归测试为 30 项通过。Pure runtime 将部署前向与训练、评估和绘图分离，便于后续替换真实执行后端。

## 时间控制

- 第 1–3 页：约 1 分 30 秒
- 第 4–7 页：约 2 分 55 秒
- 第 8–10 页：约 2 分 25 秒
- 第 11–13 页：约 2 分 20 秒
- 第 14 页：约 50 秒
- 合计：约 10 分钟
