# Hybrid Photonic MI-BCI 技术文档大纲

本文档是技术文档写作大纲，不是最终完整版正文。每个条目都给出简要说明，正式写作时可在这些说明基础上扩展为完整段落。

## 0. 技术文档快速预览

建议在正式技术文档目录前放一页快速预览，方便评审先抓住项目价值。

- 作品目标  
  用 3 到 5 句话说明项目面向 MI-BCI，解决 EEG 个体差异大、校准成本高、便携端计算资源有限的问题。

- 核心流程  
  用一行流程概括系统：`EEG -> FBCSP -> small MLP embedding -> experience library -> photonic scan -> reject/command`。

- 核心创新  
  分三点写清楚：FBCSP 的可解释特征、经验库的个体/会话特化、photonic scan 的候选头并行扫描。

- 当前证据  
  简要列出已经完成的代码、测试、指标、图表、单次推理脚本和光计算/量化接口；同时说明在线采集/上位机能力正在完善，尚未完成真实闭环。

- 交付材料对应关系  
  用表格说明技术文档、源码、测试、metrics、figures、PPT、演示视频分别支撑哪些要求。

注意点：

- 这里适合“先讲结论”，不要堆公式。
- 结果数字要以当前 `artifacts/metrics/` 中最新文件为准，不要手工沿用旧报告里的数值。
- 若写 “LTSimulator/光计算仿真已完成”，要明确当前实际运行后端和是否为真实光芯片。
- 若写 “Cyton 在线采集/上位机”，要明确该能力正在完善但未完成，当前核心验证仍以公开数据集回放为主。

## 1. 项目概述

### 1.1 背景与问题定义

- 运动想象 EEG-BCI 任务  
  说明系统需要从一段 EEG 决策窗口中识别用户运动想象意图，并将其转换为控制命令。

- EEG 信号特点  
  说明 EEG 具有低信噪比、小样本、非平稳和容易受伪迹影响等特点，因此模型不能只追求复杂度。

- 脑电基础与电极位置  
  简要说明国际 10-20 系统、中央区 C3/Cz/C4 及其邻近通道和运动想象任务的关系。

- MI 节律现象  
  简要说明 mu/beta 节律、ERD/ERS 等概念，用于解释为什么要关注 8-30 Hz 附近的运动相关节律。

- MI 任务中的个体差异  
  说明运动想象 EEG 的跨被试差异尤其明显，不同人的节律响应频段、空间激活模式和想象策略都可能不同。

- 特化而非单一泛化的动机  
  说明主线选择少量校准后的个体/会话特化，而不是只训练一个对所有被试固定的泛化模型。

- FBCSP 的背景作用  
  说明 FBCSP 是运动想象 EEG 中稳定、可解释、小样本友好的经典方法，适合作为系统主干和 baseline。

- FBCSP 与光计算的适配性  
  说明 FBCSP 包含滤波、空间投影、协方差/Gram 矩阵等大量线性计算，与光计算矩阵/向量运算接口适配度较高。

- 深度学习的定位  
  说明深度学习只作为小型 embedding 模块使用，用于改善特征空间，而不是完全替代 FBCSP。

- 光计算的定位  
  说明当前工程按 forward-only 统计口径，前向传播中的全部线性计算均通过 MatrixOpsBackend/SignalOpsBackend/TiledMVMBackend 归入光计算接管路径，其中候选线性头扫描是重点低位宽量化路径。

建议图表：

- MI-BCI 任务示意图：展示 EEG 窗口、模型推理和命令输出。
- 10-20 电极位置示意图：说明 C3/Cz/C4 与运动想象的关系。
- ERD/ERS 概念图：用概念图说明 cue 后 mu/beta 功率变化，不必声称是项目实测图。
- 背景逻辑图：展示 EEG 个体差异、经验库特化和 photonic scan 之间的关系。

注意点：

- 不要写成纯深度学习 BCI。
- 不要把经验库写成普通数据缓存；它的来源是 EEG/MI 个体差异导致的特化需求。
- 不要把 FBCSP 只写成传统 baseline；它本身也是线性计算密集、适合光计算接管的模块。
- 不要把光计算写成已经替代全部算法；应表述为“接管全部前向线性计算”，非线性、控制逻辑、拒识、经验库管理仍在数字端。

### 1.2 项目目标

- 构建混合光计算 MI-BCI 原型  
  说明工程目标是实现 FBCSP 主导、经验库增强、可接光计算的运动想象 BCI 原型。

- 验证三条实验线  
  说明工程同时保留传统 baseline、小型 MLP embedding 线和经验库 photonic scan 主线，便于对比各模块贡献。

- 验证经验库特化效果  
  说明少量校准样本用于查询经验库，从历史候选头中选择更适合当前被试/会话的模型组合。

- 暴露光计算接口  
  说明矩阵乘、线性头扫描和前向信号处理线性算子通过 backend 统一封装；按当前工程统计口径，前向传播全部线性计算已经由光计算 backend 接管。

- 保持工程可复现  
  说明数据集加载、实验脚本、结果保存、图表生成和进度记录都应有明确路径。

- 为 OpenBCI Cyton 上位机预留扩展  
  说明实时采集、经验库管理、校准流程和在线推理界面正在完善，但尚未完成真实在线闭环。

主线一句话概括：

```text
FBCSP + small MLP embedding + experience library retrieval + photonic candidate scan
```

注意点：

- 当前是公开数据集回放验证，不是真实在线闭环验证。
- 当前光计算路径是接口和软件量化模拟，不是真实光芯片实测。
- 在线采集/上位机不能写成已完成产品，应写为正在完善中的原型能力。

### 1.3 系统输出定义

- 命令输出  
  系统输出包括 `left/right/foot/reject`，其中前三个是运动想象命令，`reject` 是系统拒识状态。

- `reject` 的含义  
  `reject` 由置信度和边际阈值产生，用于避免模型在低置信度窗口上误触发控制命令。

- `reject` 的合理性  
  真实采集中，被试不一定始终按提示进行运动想象，环境光照、声音、空气流动、身体状态等刺激也可能掩盖 MI 信号，因此拒识是正常且必要的。

注意点：

- `reject` 不是数据集原生第四类。
- 如果写“四分类”，要解释为“三个 MI 命令 + 一个系统拒识状态”。
- 不要把 `reject` 简单描述为模型失败。

## 2. 系统总体架构

### 2.1 总体流程

建议框图包含：

```text
Raw EEG window
  -> packet continuity / channel mapping
  -> signal quality check
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

各模块简要说明：

- Raw EEG window  
  表示一个待决策的 EEG 时间窗口，是在线推理的最小输入单位。

- Packet continuity / channel mapping  
  在线采集时需要检查样本序号和时间戳连续性，并将采集板通道映射到 C3/C4/Cz 等电极位置。

- Signal quality check  
  在分类前先检查平线、饱和、突变、工频干扰、坏通道比例等质量指标，必要时提前拒识。

- Common average reference  
  用于降低公共噪声和参考电极影响，是 EEG 前处理的一部分。

- Band-pass / filter bank  
  将 EEG 分解到多个运动想象相关频段，为后续 FBCSP 提供频带特征。

- FBCSP spatial projection  
  对不同类别的空间模式进行投影，提取区分运动想象类别的空间特征。

- Fisher feature selection  
  从 FBCSP 特征中选择区分度较高的维度，降低冗余和过拟合风险。

- Small MLP encoder  
  将筛选后的 FBCSP 特征映射为 compact embedding，服务于经验库检索和候选头扫描。

- Experience library retrieval  
  根据校准 embedding 查询经验库，选择与当前被试/会话状态更接近的候选头。

- Tiled photonic MVM scan  
  使用 `2 x 8` tile 模拟光计算单元，对多个候选线性头进行矩阵向量扫描。

- Probability fusion  
  融合多个候选头的概率输出，得到最终命令概率。

- Reject decision  
  根据置信度和概率边际决定是否输出命令或进入拒识状态。

建议图表：

- 系统总体框图：标出每个模块和数据流。
- 在线推理路径图：只画单窗口推理路径，不画训练细节。
- 训练/校准/推理分层图：说明哪些模块属于离线训练，哪些属于在线前向。
- 质量检查分支图：展示信号质量不合格时直接进入 `reject`，不继续分类。

注意点：

- 框图必须写出 FBCSP，而不是只写 feature extraction。
- photonic scan 要放在候选线性头扫描位置。
- 不要把光计算画成替代整个 EEG 处理流程。

### 2.2 三个阶段划分

#### 离线训练阶段

- 数据集加载  
  读取公开 EEG 数据集，并按工程协议形成训练、校准和评估划分。

- FBCSP 拟合  
  在训练集上拟合滤波器组和 CSP 空间投影参数。

- Fisher 特征选择  
  根据训练集统计选择区分度较高的 FBCSP 特征。

- 标准化参数拟合  
  在训练特征上估计均值和尺度，用于后续特征归一化。

- 小型 MLP 训练  
  使用筛选后的 FBCSP 特征训练小型网络，导出 embedding 和分类头。

- 初始经验库构建  
  通过 anchor heads、bootstrap heads、embedding LDA heads 等方式建立候选经验条目。

#### 校准阶段

- 新被试/新会话少量校准样本输入  
  使用少量带标签或可控提示样本刻画当前被试状态。

- 计算 calibration embeddings  
  将校准样本通过 FBCSP 和 MLP 转换到 embedding 空间。

- 查询经验库  
  根据 calibration embeddings 与经验库 centroid 的关系选择候选条目。

- 选择 top-K 候选经验头  
  保留若干个最适合当前状态的线性头，而不是只依赖一个全局头。

- 设置或更新拒识阈值  
  根据训练或校准统计设置置信度阈值，控制误触发风险。

#### 在线推理阶段

- 单个 EEG 决策窗口输入  
  每次在线推理只处理一个新的 EEG 窗口。

- FBCSP transform  
  使用训练好的 FBCSP 参数提取该窗口的特征。

- 标准化  
  使用训练阶段保存的均值和尺度处理特征。

- MLP embedding  
  将 FBCSP 特征映射到经验库检索和候选头分类使用的 embedding 空间。

- 候选线性头 photonic scan  
  对 top-K 候选头进行 tiled MVM 扫描，得到多个候选分数。

- 概率融合  
  按检索权重融合候选头概率，得到最终类别概率。

- 拒识判断  
  若置信度或概率边际不足，则输出 `reject`。

注意点：

- setup/calibration 不算单次前向推理。
- 单次前向只对应一个 EEG decision window 的在线路径。

### 2.3 公开数据回放与实采在线模式

- 公开数据回放模式  
  说明当前实验主要来自已经切好的公开数据 trial，适合复现实验、计算准确率和生成图表。

- 实采在线模式  
  说明 OpenBCI Cyton 在线流程正在完善但尚未完成真实闭环；目标流程从连续数据包开始，需要时间戳检查、ADC 换算、环形缓冲区和窗口切片。

- 两种模式的共同部分  
  说明两种模式最终都进入 FBCSP、embedding、经验库检索、photonic scan 和拒识决策。

- 两种模式的差异  
  说明公开数据回放有真值标签，可计算实时准确率；真实在线没有即时标签，只能显示置信度、拒识和信号质量。

注意点：

- 报告中若写实采流程，要明确在线采集/上位机能力正在完善但未完成，不能写成已完成真实闭环。
- 不要把公开数据回放的准确率直接写成真实在线闭环准确率。

### 2.4 信号质量与两级拒识

- 质量提前拒识  
  在分类前检查坏通道、平线、饱和、突变和工频/肌电干扰，质量不合格时直接拒识。

- 分类置信度拒识  
  在模型输出概率后，根据最高概率和前两类概率边际判断是否拒识。

- 为什么需要两级拒识  
  质量拒识处理明显不可用窗口，置信度拒识处理模型不确定窗口，两者共同降低误触发。

- 与经验库写入的关系  
  低质量窗口不应写入经验库，避免污染历史经验。

注意点：

- 质量检查是在线系统非常重要的安全分支。
- 如果当前工程只部分实现质量检查，应在正式文档中标注为上位机扩展或在线联调内容。

## 3. 算法路线设计

### 3.1 Baseline 1: FBCSP + shrinkage LDA

- Filter bank 频段设置  
  说明使用多个运动想象相关频段提取节律信息，避免只依赖单一频带。

- One-vs-rest CSP  
  说明多类任务中每个类别都与其他类别形成 one-vs-rest 空间滤波问题。

- Log-variance 特征  
  说明 CSP 投影后的方差经过归一化和对数变换，形成 FBCSP 特征向量。

- Fisher score 特征筛选  
  说明用类间/类内差异选择更有判别力的特征，减少维度和过拟合。

- Shrinkage LDA 分类  
  说明 shrinkage 可缓解小样本协方差估计不稳定，比普通 LDA 更适合 EEG。

- 拒识阈值  
  说明 baseline 也使用置信度或边际阈值产生 `reject`，保证输出口径一致。

建议图表：

- FBCSP 特征维度示意图：展示 band、class、component 如何展开成特征。
- LDA baseline 流程图：展示从 FBCSP 到 LDA scores 的路径。
- baseline 混淆矩阵：展示各类别和 reject 的分布。

重点解释：

- 该 baseline 稳定、可解释、小样本友好。
- FBCSP 本身包含大量线性计算，与光计算接口具有较高适配度。

### 3.2 Baseline 2: FBCSP + small MLP embedding

- 输入特征  
  输入不是原始 EEG，而是经过 FBCSP 和 Fisher 筛选后的特征。

- 网络结构  
  说明小型 MLP 的 hidden dim、embedding dim、分类头维度，以及为什么保持网络较小。

- Embedding 输出  
  说明 embedding 是主线经验库检索和候选头扫描的中间表示。

- 分类 logits  
  说明 MLP 也能直接输出分类分数，用作第二条实验线对比。

- 训练曲线  
  说明需要展示 loss 和 accuracy，以判断小型网络是否正常收敛。

- 与 LDA baseline 的差异  
  说明该线增加了非线性特征映射，但还没有引入经验库和 photonic scan。

建议图表：

- 小型 MLP 结构图。
- 训练 loss 曲线。
- 训练 accuracy 曲线。
- embedding 可视化，可选 PCA 或 t-SNE。

重点解释：

- 小型 MLP 不是为了取代 FBCSP，而是为经验库提供更适合检索的表示空间。
- 不要描述为大型端到端深度网络。

### 3.3 Mainline: FBCSP + MLP embedding + experience library + photonic scan

- 经验库设计来源  
  EEG 个体差异大，MI 任务差异更强，因此系统需要少量校准后的特化机制。

- 特化而非单一泛化  
  单一泛化模型难以覆盖所有被试/会话状态，经验库允许从历史候选模型中选择更适合当前状态的组合。

- 经验库条目结构  
  每个条目应包含 entry id、centroid、linear head、source、train indices、train accuracy 等信息。

- Anchor heads  
  用于保存全局 MLP 分类头或全局 embedding LDA 头，作为经验库中的稳定参考候选。

- Bootstrap heads  
  通过训练集采样构建多个候选 LDA 头，增加经验库多样性。

- Calibration query  
  将校准样本映射为 embeddings，并用其均值或统计量查询经验库 centroid。

- Top-K 候选头选择  
  根据距离、训练质量、校准表现等因素选择若干候选头，而不是只选一个。

- Photonic scan  
  将多个候选线性头堆叠为 head bank，并通过 tiled MVM 一次扫描多个候选器。

- Probability fusion  
  将候选头输出的概率按检索权重融合，得到更稳健的最终概率。

- Reject decision  
  根据融合后的概率置信度和边际进行拒识判断，降低误触发风险。

- Photonic scan 的创新逻辑  
  经验库产生多个候选头，光计算单元擅长低功耗并行 MVM，因此两者天然匹配。

建议图表：

- 经验库结构图：展示 entry、centroid、linear head 和质量指标。
- 经验库检索流程图：展示校准 embedding 如何选出 top-K。
- top-K 候选头权重图：展示各候选头在融合中的贡献。
- candidate scan tile schedule：展示 `2 x 8` tile 如何扫描候选头。
- 主线混淆矩阵：展示最终命令和 reject 分布。

重点解释：

- 经验库不是普通样本缓存，而是候选模型/候选线性头集合。
- 主线选择特化是因为 MI-EEG 个体差异显著。
- photonic scan 不是 Goertzel 类频率扫描，也不是只算一个固定头。
- photonic scan 是根据光计算单元低功耗并行 MVM 特点设计的候选头扫描机制。

## 4. 数据集与实验协议

### 4.1 BCICIV_1_asc

- 数据集文件结构  
  说明使用 `BCICIV_1_asc` 中 a-g 文件，每个文件是采集文件，不是最终类别。

- a-g 文件合并  
  说明工程将 a-g 合并为 pooled dataset，用统一流程训练和评估。

- 原始类别差异  
  说明单个采集文件可能只包含部分类别，需要映射到全局类别体系。

- 全局类别映射  
  工程使用 `left/right/foot` 作为 MI 命令类别，`reject` 是系统额外输出。

- Train split  
  用于 FBCSP 拟合、特征选择、标准化、小型 MLP 训练和经验库构建。

- Calibration split  
  用于模拟新被试/新会话的少量校准，查询经验库候选头。

- Evaluation split  
  用于模拟在线推理回放，计算最终 accuracy、reject rate 和混淆矩阵。

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

- `reject` 不能写进数据集标签。
- Evaluation windows 不能参与训练或经验库选择。

### 4.2 BNCI2014_004

- 数据集基本情况  
  说明该数据集包含多个被试和多个 session，适合观察个体差异。

- 被试/session 划分  
  说明 history、calibration、evaluation 分别来自哪些 session 或 trial。

- 个体特化验证  
  说明该数据集用于回答经验库是否能提升目标被试/目标会话效果。

- 三条线比较  
  说明 FBCSP+LDA、FBCSP+MLP、主线需要在相同 evaluation 协议下比较。

- Subject-wise 结果  
  说明除了平均值，还应保留每个被试的结果，避免掩盖个体差异。

建议图表：

- 每个被试 session 划分示意。
- 个体特化前后结果对比表。
- subject-wise accuracy / reject rate 图。

注意点：

- 验证经验库是否有用，本质是看特化前后对目标被试的效果。

## 5. 光计算接口设计

### 5.1 MatrixOpsBackend

- 统一封装矩阵乘的原因  
  说明所有算法路径中的前向矩阵运算均通过统一接口，当前前向传播中的全部线性计算已纳入光计算 backend 接管口径，后续真实硬件驱动可在不改动上层算法的情况下替换该接口。

- FBCSP 中的可接管线性计算  
  说明空间投影、协方差/Gram 矩阵等操作可表达为矩阵运算，适合作为 backend 接管对象。

- 当前接管的典型操作  
  包括 FBCSP 空间投影、LDA scores、MLP linear layers、candidate linear head scores、probability fusion 和相关 einsum 路径。

- 后续真实光芯片接入方式  
  说明真实硬件驱动应实现相同接口，避免上层算法代码大规模修改。

注意点：

- 不要只说“矩阵乘可替换”，要说明哪些算法模块会调用该接口。
- 要明确当前工程口径是前向传播全部线性计算均已接管到光计算 backend；但真实光芯片尚未完成硬件实测。

### 5.2 SignalOpsBackend

- CAR 的线性计算属性  
  说明 common average reference 本质上是线性变换，可纳入前向线性计算统计。

- SOS filtering 的统计口径  
  说明带通滤波属于前向信号处理中的线性计算，应在“全部线性计算”中解释其口径。

- 当前如何统计 signal ops  
  说明 CAR、SOS filtering 等前向线性信号处理也纳入光计算接管和 MAC-equivalent 统计，不代表真实芯片功耗。

注意点：

- 只统计前向传播，不统计训练反向传播。
- 文档要解释“全部线性计算”的定义。
- 要明确前向 signal ops 属于“全部线性计算”的一部分，已经计入光计算接管口径。

### 5.3 TiledMVMBackend

- `2 x 8` tile 的含义  
  说明 `2 x 8` 是单个光计算单元尺寸，不是算法矩阵大小限制。

- 大矩阵拆分方式  
  说明大矩阵通过 row block 和 column block 拆成多个 tile 计算。

- Partial sum 累加  
  说明每个 tile 输出的是部分结果，需要累加得到完整 MVM 输出。

- Tile evaluations per window  
  说明该指标表示每个 EEG 决策窗口需要调用多少次硬件 tile。

- Tile 数量公式  
  对候选头矩阵 `A in R^{M x (D+1)}`，top-K 扫描的 tile 数可写为 `K * ceil(M / 2) * ceil((D + 1) / 8)`。

- 主线示例  
  当前主线若使用 `K=8, M=3, D=32`，加 bias 后是 `3 x 33` 候选头，每个候选需要 `2 * 5 = 10` 次 tile，8 个候选共 `80` 次 tile。

注意点：

- 大矩阵通过滑动、拆分、拼接完成。
- 不要写成系统只能算 `2 x 8`。
- Tile 次数是调度/硬件调用估算，不等同于完整系统耗时。

### 5.4 Photonic scan 设计动机

- 光计算单元优势  
  说明光计算单元适合低功耗、高并行矩阵/向量乘。

- 与经验库的匹配关系  
  说明经验库检索产生多个候选线性头，天然形成多候选 MVM 扫描任务。

- Candidate head bank  
  说明多个候选头可以堆叠为一个 head bank，统一交给 tiled MVM backend 扫描。

- Candidate scores  
  说明 photonic scan 输出多个候选头的分类分数，而不是直接输出最终类别。

- Probability fusion  
  说明最终结果来自多个候选头概率融合，体现经验库检索权重。

- 创新点  
  说明 photonic scan 不是简单替换 `np.matmul`，而是根据光计算适合并行扫描线性候选器的特点重新组织决策路径。

注意点：

- 不要把 photonic scan 解释为 Goertzel 类频率扫描。
- 不要写成只计算一个固定分类头。
- 应突出“一个 EEG 决策窗口内快速扫描多个候选校准器/线性头”。
- 注意区分 `photonic scan` 和频率扫描算法，它不是 Goertzel 类频域检测方法。

## 6. 量化策略

### 6.1 当前支持

- 两层位宽口径  
  需要区分算法逻辑精度与光计算单元单次调用位宽。当前逻辑精度按算子自适应选择 4/6/8-bit，但所有路径仍拆为 4-bit physical slices 交给光计算单元；candidate scan 使用单次 4-bit。

- 物理调用位宽  
  所有 bit-sliced 子计算均满足 `uint4 input + int4 weight`，即 `qin=[0,15]`、`qwt=[-8,7]`。

- 当前主线策略  
  CAR 从4-bit开始；SOS状态转移、FBCSP空间投影和标准化从6-bit开始；MLP、LDA及其他敏感路径保持8-bit。低精度算子周期性与8-bit bit-sliced数字影子比较，超差时对当前调用立即升档重算并保持该档位。Photonic candidate scan使用单次4-bit tiled MVM。

### 6.2 4-bit 配置

```text
input activation: uint4, qin=[0, 15]
weight:           int4,  qwt=[-8, 7]
```

- 输入量化  
  输入 activation 使用 unsigned 4-bit，范围为 0 到 15。

- 权重量化  
  权重使用 signed 4-bit，范围为 -8 到 7。

- 使用原因  
  4-bit 是当前关注的工业甜点位宽，兼顾动态范围、噪声鲁棒性和硬件实现。

- Candidate scan  
  候选头扫描直接使用单次 4-bit，是当前真正的一次调用低位宽路径。

### 6.3 位权拆分与自适应逻辑精度

```text
input activation: uint8, qin=[0, 255]
weight:           int8,  qwt=[-128, 127]
```

- 位权分解  
  对量化整数使用 radix-16 展开，例如 `q = q0 + 16*q1 + 16^2*q2 + ...`。输入 slice 保持在 `[0,15]`，有符号权重使用 balanced radix-16 slice 保持在 `[-8,7]`。

- 分块与位权拆分  
  矩阵尺寸先按输出维 2 行、输入维 8 列拆成 `2 x 8` tile；每个 tile 再按输入/权重 slice 组合多次调用，最后按 `16^(i+j)` 累加 partial sum。

- 当前状态  
  普通前向线性路径当前采用4/6/8-bit混合逻辑精度，各档位均由多个uint4/int4物理调用重构；这不是一次原生4/6/8-bit光计算调用。

### 6.4 当前实现边界

- Photonic scan 量化  
  候选头扫描路径使用单次 uint4/int4 量化 tiled MVM。

- 其他 MatrixOps 路径  
  其他前向线性计算由 AdaptivePrecisionPhotonicMatrixOpsBackend 接管。精度状态按频带、SOS section、滤波方向、CSP类别投影和其他算子分别维护，避免一个高敏感算子迫使全部前端路径升到8-bit。

- SignalOps 路径  
  CAR 展开为通道混合矩阵；SOS 每个二阶节展开为 `3 x 3` 状态转移矩阵，前向与反向滤波中的系数乘加均调用 bit-sliced MatrixOpsBackend。

- 持续精度监测
  每个低精度算子首次调用及之后按固定间隔生成8-bit bit-sliced数字影子，计算低位宽相对8-bit基准的新增归一化误差。超出阈值时当前调用按下一档位重算，并只升不降，避免运行期间精度来回抖动。

- 监测记录
  单窗口脚本在终端汇总各策略的当前位宽、监测次数、最大8-bit影子误差、升档次数和tile数，并将逐算子报告保存到`artifacts/metrics/fbcsp_design/adaptive_precision_eval_XXXX.json`。

- QAT 表述边界  
  当前采用运行时动态定点量化和位权拆分，不是 QAT。只有后续发现小型 MLP 的低位宽误差明显影响准确率时，才需要引入 QAT。

### 6.5 精度与资源权衡评估

- 算子实际精度需求  
  后续不能直接假设所有算子都需要相同逻辑位宽。应分别测量 CAR、SOS、FBCSP 投影、标准化、MLP 各层、距离交叉项、候选头和概率融合在不同逻辑位宽下的输出误差及任务指标变化。

- 最小可用精度  
  对每个算子寻找满足 command accuracy、accepted accuracy、reject rate 和数值稳定性要求的最低逻辑位宽，避免统一使用过高精度造成不必要的 slice 数、tile 调用、延迟和能耗。

- 混合精度策略  
  对误差敏感的滤波状态转移、FBCSP 或特定 MLP 层保留较高逻辑精度；对候选扫描、概率融合等容忍度较高的路径使用更低精度，形成逐算子 precision policy。

- 物理位宽拓宽研究  
  探索将光计算单元从 4-bit 拓宽到 5/6/8-bit。位宽增加会降低抗噪能力，但可减少位权 slice 组合数量和 partial-sum 累加次数，因此可能提高有效吞吐并降低调度开销。

- 工程可接受性判据  
  需要在噪声强度、有效位数、精度下降、拒识率、物理调用次数、延迟和能耗之间绘制 Pareto 曲线。若较宽物理位宽在真实噪声下仍满足任务指标，则可以用有限抗噪损失换取更高计算效率。

- 建议实验矩阵  
  逻辑位宽可比较 4/6/8/10/12-bit，物理位宽可比较 4/5/6/8-bit，并加入不同噪声等级。所有组合都应使用相同数据划分、模型参数和拒识协议。

注意点：

- 不要把4/6/8-bit逻辑精度写成对应位宽的一次原生光计算调用。
- 不要把 bit-sliced 动态量化写成 QAT。
- 要区分单次4-bit candidate scan、自适应逻辑精度的4-bit位权拆分、数字影子监测，以及尚未完成的真实光芯片实测。

## 7. 前向传播、进度与耗时

### 7.1 Setup timings

- load dataset  
  数据读取和预处理准备，不属于在线单次前向。

- fit FBCSP  
  在训练集上拟合 FBCSP 参数，是离线训练步骤。

- transform replay  
  批量生成 replay 特征，用于评估准备，不是单窗口推理。

- feature selection  
  Fisher 特征选择在训练阶段完成，在线阶段只使用已选特征索引。

- standardization  
  训练阶段拟合标准化参数，在线阶段只应用该参数。

- train MLP  
  小型 MLP 训练属于离线训练，不应计入单次推理延迟。

- build experience library  
  经验库构建属于离线或周期性维护步骤。

- calibrate runtime  
  校准阶段用于选择候选头，不是每个在线窗口都执行。

- calibrate reject threshold  
  阈值校准可能较耗时，但属于 setup/calibration，不属于单次前向。

注意点：

- Setup 总耗时不能当作在线 latency。

### 7.2 Single online forward timings

- Setup/backend 隔离  
  单窗口脚本在 setup 阶段使用 NumPy/SciPy 完成数据加载、模型拟合、批量 replay 缓存和阈值校准；退出 setup 后恢复 bit-sliced photonic backend，避免把离线准备耗时和资源计入在线前向。

- FBCSP transform one window  
  单个 EEG 窗口通过已拟合 FBCSP 得到特征。

- Standardize one window  
  使用训练阶段保存的均值和尺度处理当前特征。

- Small MLP forward one window  
  通过小型 MLP 得到 embedding 和可选 logits。

- Photonic scan one window  
  对当前 embedding 扫描 top-K 候选线性头。

- Online forward total  
  上述在线步骤的总耗时才是单次前向推理延迟。

- 物理 tile/slice 调用  
  除阶段耗时外，还应记录每个在线阶段的 bit-sliced physical tile evaluations。该值包含 `2 x 8` 空间分块和位权 slice 组合，不能只报告 candidate scan 的逻辑 tile 数。

- 重复计时
  使用`--online-repeats N`可在一次setup后重复同一在线窗口，报告首轮、中位数、P90、末轮和预测一致率。首轮包含影子监测与可能的升档重算，末轮更接近稳定精度配置下的软件模拟耗时。

- 自适应/固定8-bit A/B
  使用`--precision-validation-windows N`可在相同原始EEG窗口上比较自适应精度与固定8-bit完整前向，实时报告累计准确率和预测一致率，最终输出command accuracy、accepted accuracy、reject rate、概率L2差、耗时和tile下降比例。

- 当前通路验证示例  
  在3秒BCICIV_1_asc窗口、缩短训练配置和固定拒识阈值的工程验证中，同一窗口连续运行5次后，稳态末轮约77.7万次physical tile evaluation，其中FBCSP前端约77.6万次；全8-bit逻辑精度基线约91.6万次，因此tile调用下降约15.1%。5次预测一致率为1.000，均保持正确的`left`且未拒识；在线软件耗时中位数约1.545 s、P90约1.600 s。该结果用于验证路径和资源变化，不作为完整数据集精度或真实芯片延迟结论。

- 初步跨窗口验证
  3个不同evaluation窗口的自适应/固定8-bit A/B中，两者command accuracy均为1.000、reject rate均为0、预测一致率为1.000，平均概率L2差为0.00775；自适应平均tile下降15.7%，同进程中位耗时约1.540 s，对照约1.634 s。窗口数过少，该结果只能作为工程sanity check，正式结论必须扩大到完整被试与session。

建议表格：

| 阶段 | 是否属于单次前向 | 说明 |
| --- | --- | --- |
| FBCSP transform | 是 | 单窗口特征提取 |
| MLP forward | 是 | embedding 计算 |
| photonic scan | 是 | 候选头扫描 |
| train MLP | 否 | 离线训练 |
| calibrate threshold | 否 | 校准阶段 |

### 7.3 实时进度与准确率

- 进度条  
  显示 evaluation scan 的当前窗口数、总窗口数、耗时和 ETA。

- `acc`  
  表示累计 command accuracy，分母是全部已评估窗口。

- `accepted_acc`  
  表示非拒识窗口上的累计准确率，分母是已接受窗口。

- `reject`  
  表示累计拒识率，分母是全部已评估窗口。

- Reject rate 解释  
  需要结合采集条件、被试配合程度、环境刺激和任务难度解释。

注意点：

- 实时准确率只适用于公开数据集回放，因为有标签。
- 真实在线部署时没有即时标签，不能实时显示 accuracy。
- 较高 reject rate 不一定代表系统不可用，也可能是避免误触发的保护机制。

## 8. 计算量统计

### 8.1 统计口径

- Forward-only  
  只统计前向传播中的线性计算，不统计训练反向传播。

- 全部线性计算  
  包括前处理、FBCSP、标准化、MLP linear layer、线性头、候选扫描和概率融合中的线性计算；当前工程将这些前向线性计算全部接管到光计算 backend/统计口径。

- Photonic linear MACs  
  表示当前由光计算 backend 接管的前向线性计算量；在当前工程口径下，前向线性计算全部归入该项。

- Digital linear MACs  
  表示未被归入光计算接管路径的前向线性计算量；当前工程定义下该项应为 0，若出现非零值，应检查是否存在 backend 接口遗漏或统计口径错误。非线性与控制逻辑不属于该线性 MAC 口径。

### 8.2 当前统计项

- Preprocessing linear ops  
  包括 CAR、滤波等前向信号处理线性操作。

- FBCSP spatial projection  
  统计 CSP 空间投影相关矩阵运算。

- Feature standardization affine  
  统计特征标准化中的逐维仿射操作。

- MLP forward linear layers  
  统计小型 MLP 前向中的线性层。

- LDA / linear head scores  
  统计 LDA 和候选线性头的分数计算。

- Candidate photonic scan  
  统计经验库候选头 tiled MVM 扫描。

- Probability fusion  
  统计多个候选头概率融合中的线性组合。

- 逐算子矩阵拆分表  
  建议在正文或附录中列出每个前向线性算子的矩阵表达式、矩阵大小、单窗口 MAC、是否可 tile 映射和当前实现证据。

- 调用证据  
  每个计入光计算路径的算子最好能对应 backend 调用、量化配置、tile 计数、日志或 compute accounting 记录。

### 8.3 结果解释

- Photonic share  
  表示前向线性计算量中已由光计算 backend 接管的比例；按当前工程定义应为 100%，但不等于真实硬件功耗占比，也不代表所有算子都已完成 4-bit 或真实芯片实测。

- Inference share  
  表示在线推理阶段的光计算线性占比，比全流程统计更贴近部署路径。

- 工程估算边界  
  当前统计是 MAC-equivalent 估算，表示全部前向线性计算已被光计算 backend 接管；它不是芯片实测功耗或实测吞吐。

- 结果数字来源  
  正式文档中的 MAC 和占比应从 `compute_accounting.json` 或当前运行输出中同步，不要手工沿用旧报告中的数字。

注意点：

- 要明确“全部前向线性计算已接管到光计算 backend”与“低位宽量化/真实硬件实测覆盖范围”是两个不同层次。
- 不要把光计算占比直接解释为真实芯片功耗占比。

## 9. 工程结构

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

- `backends.py`  
  定义 MatrixOps、SignalOps、量化 photonic backend 和 tiled MVM，是光计算接入边界。

- `fbcsp.py`  
  实现滤波器组 CSP 特征提取，是 EEG 特征主干。

- `small_networks.py`  
  实现小型 MLP embedding 网络。

- `experience.py`  
  实现经验库构建、检索、候选头扫描和概率融合。

- `progress.py`  
  实现运行进度、耗时记录和终端进度条。

- `workflows/`  
  存放完整实验线，包括 FBCSP baseline、小型 MLP、主线和 BNCI 个体特化验证。

- `datasets/`  
  存放公开数据集加载与划分逻辑。

- `host/`  
  存放正在完善的 OpenBCI Cyton 上位机相关模块；当前尚未完成真实 EEG 在线采集、推理与反馈闭环。

- `pure_runtime/`  
  存放纯净部署前向代码，不包含训练、评估和绘图。

- `examples/`  
  存放可运行命令行入口。

- `artifacts/`  
  存放实验指标、数组和生成图片。

- `visualization/`  
  存放绘图脚本，避免绘图逻辑混入核心算法。

- `docs/`  
  存放技术文档、设计大纲和后续报告材料。

注意点：

- 不要把临时测试输出混入源码根目录。
- 绘图脚本和图片应分别整理。

## 10. 运行方式

### 10.1 完整三线对比

```bash
python examples/run_fbcsp_design_comparison.py
```

- 作用  
  运行 FBCSP+LDA、FBCSP+MLP、主线三条实验线。

- 输出  
  保存 summary、compute accounting 和 progress 记录。

- 复现实验说明  
  正式文档中应写明输出路径，例如 `artifacts/metrics/fbcsp_design/summary.json` 和 `compute_accounting.json`。

### 10.2 只跑主线

```bash
python examples/run_experience_photonic_line.py
```

- 作用  
  只运行 `FBCSP + MLP embedding + experience library + photonic scan`。

- 输出  
  保存主线 summary、arrays 和在线扫描进度。

### 10.3 单个样本全流程推理

```bash
python examples/run_single_window_inference.py --evaluation-index 0
```

- 作用  
  从 evaluation split 中取一个样本，运行完整单窗口在线前向。

- `--evaluation-index`  
  表示 evaluation split 内部索引，不是原始数据文件中的 trial 编号。

### 10.4 BNCI 个体特化验证

```bash
python examples/run_bnci2014_004_personalization.py
```

- 作用  
  在 BNCI2014_004 上验证经验库对单一被试/会话特化是否有效。

- 输出  
  应包含 subject-wise 结果、三条线对比和总体统计。

### 10.5 技术数据与提交材料

- 源码  
  说明核心源码、examples、tests 和 pure runtime 均随工程提交。

- 指标文件  
  说明 JSON/NPZ 指标保存在 `artifacts/metrics/`，用于复查结果和重画图。

- 绘图脚本  
  说明图表生成脚本保存在 `visualization/`，图像保存在 `artifacts/figures/`。

- 测试命令  
  写明 `python -m unittest discover -s tests`，并说明测试覆盖 backend、FBCSP、经验库、host 等关键模块。

- 敏感信息清理  
  提交前检查本机串口配置、个人数据库、缓存、绝对路径、PDF 元数据和视频片尾。

## 11. 可视化设计

### 11.1 必须包含的图

- 系统框图  
  展示 FBCSP、MLP、经验库、photonic scan 和 reject 的整体关系。

- 训练 loss / accuracy  
  展示小型 MLP 是否正常收敛。

- 混淆矩阵  
  展示各类别和 reject 的预测分布。

- Rolling command accuracy  
  展示滑动窗口内命令准确率变化。

- Rolling reject rate  
  展示滑动窗口内拒识率变化。

- Cumulative command accuracy  
  展示从第一个 evaluation window 开始累计的实时准确率。

- Cumulative reject rate  
  展示累计拒识率，适合和终端实时进度对应。

- Photonic tile schedule  
  展示 `2 x 8` tile 如何覆盖候选头矩阵。

- Compute accounting summary  
  展示 photonic/digital 线性计算量占比。

- Adaptive precision diagnostics
  展示CAR、SOS、FBCSP投影、标准化和其他线性算子的最终4/6/8-bit分布、各位宽tile消耗、逐算子8-bit影子误差与阈值、升档次数和单窗口各阶段tile构成。

- Adaptive vs fixed-8-bit validation
  展示两种精度策略的command/accepted accuracy、reject rate、归一化耗时、平均tile数、逐窗口概率L2差和预测一致性。

- 经验库候选头质量和权重分布  
  展示 top-K 候选头的来源、质量和融合权重。

### 11.2 图表注意点

- Rolling 与 cumulative 区分  
  Rolling 是滑动窗口统计，cumulative 是从开头累计，二者不能混用。

- Reject rate 分母  
  Reject rate 的分母应是全部 evaluation windows。

- 图表标题  
  标题应写明数据集、实验线和指标含义。

- 图表来源  
  每张图应能追溯到对应 metrics 或 arrays 文件。

## 12. 实验结果分析

### 12.1 三条实验线对比

- Command accuracy  
  衡量所有 evaluation windows 中正确输出命令的比例，拒识不算正确命令。

- Balanced command accuracy  
  衡量各类别召回的平均表现，避免类别不均衡影响判断。

- Accepted accuracy  
  衡量非拒识窗口中的准确率，反映系统在决定输出时的可靠性。

- Reject rate  
  衡量系统拒识比例，需要结合误触发风险解释。

- Forward MACs  
  衡量前向线性计算量。

- Photonic share  
  衡量光计算接管路径在线性计算中的占比。

- Inference share  
  衡量在线推理阶段的光计算占比。

### 12.2 经验库特化分析

- 为什么选择特化  
  EEG/MI 个体差异显著，只追求全局泛化容易牺牲目标被试效果。

- 校准样本数量影响  
  分析更多或更少校准窗口是否影响经验库检索质量。

- Top-K 候选数量影响  
  分析候选头数量增加时准确率、拒识率和计算量如何变化。

- 经验库条目数量影响  
  分析经验库规模是否提高匹配概率，是否带来冗余或噪声候选。

- Subject-wise 收益差异  
  分析哪些被试从经验库中受益明显，哪些被试仍然困难。

- Photonic scan 对特化的贡献  
  说明多候选特化会增加扫描计算量，而光计算单元适合低延迟、低功耗扫描这些候选头。

### 12.3 错误与拒识分析

- 类别混淆  
  分析哪些 MI 命令最容易相互混淆。

- 低置信度窗口  
  分析 reject 是否集中在置信度低或概率边际小的窗口。

- Accepted accuracy 与 reject rate 权衡  
  说明提高拒识率通常可能提升 accepted accuracy，但会减少可输出命令数量。

- 采集状态影响  
  分析被试注意力、疲劳、眨眼、肌电和环境刺激是否可能造成拒识。

- 避免误触发  
  说明 reject 的重要价值是降低错误控制命令输出。

注意点：

- 不要只报告总 accuracy。
- 主线和 baseline 必须使用相同 evaluation split。
- Reject 是实际部署中的安全状态，不应只作为负面指标处理。

## 13. OpenBCI Cyton 上位机扩展

- 当前状态  
  OpenBCI Cyton 在线采集与上位机能力正在完善，已有接口与模块规划，但尚未完成可用于真实被试实验的稳定在线采集、推理、反馈闭环。因此，当前实验结论来自公开数据集离线回放，不应表述为已完成在线实采验证。

- Cyton 数据采集  
  正在实现从 OpenBCI Cyton 读取实时 EEG 数据流，并补充断线恢复、采样率核验和通道映射等工程能力。

- 实时 EEG buffer  
  正在完善连续数据缓存与固定窗口切片；该模块仍需在真实采集条件下验证数据连续性和时序稳定性。

- 窗口切片  
  说明在线推理以固定长度 EEG 决策窗口为单位。

- 在线推理调用  
  上位机计划调用已经训练/校准好的 runtime 进行推理，但真实数据流到控制反馈的端到端调用尚未完成验证。

- 经验库新增  
  说明新被试或新会话校准后，可将有效数据或候选头录入经验库。

- 经验库删除  
  说明需要支持删除低质量、过期或错误经验条目。

- 经验库分组  
  说明可按被试、session、任务、采集条件对经验库分组管理。

- 被试/session 元数据管理  
  说明经验库需要保存与采集状态相关的元数据，便于后续检索和分析。

- 校准流程管理  
  说明上位机应引导用户完成少量校准样本采集。

- 运行状态显示  
  说明应显示连接状态、信号质量、推理结果、置信度和拒识状态。

注意点：

- 当前在线采集/上位机能力正在完善，但还不能自行采集脑电并完成稳定的实时闭环。
- 文档中应明确写为“正在开发和验证的上位机原型”，不能表述为已完成真实在线部署。
- 上位机界面或模块存在，不等于在线采集、在线推理和控制反馈已经形成完整闭环。

## 14. 局限性

- 公开数据集回放  
  当前主要基于公开数据集验证，不能完全代表真实在线采集环境。

- 光计算硬件未实测  
  当前 photonic backend 是软件模拟或接口占位，不是真实光芯片功耗/延迟结果。

- 混合精度仍需系统评估  
  全部前向线性计算均已由光计算 backend 接管。Candidate scan使用单次4-bit，其他路径使用受8-bit影子监测的4/6/8-bit自适应逻辑精度；仍需在更多窗口和被试上比较位宽、slice数、噪声和累加误差对准确率、拒识率与延迟的影响。

- 真实硬件执行尚未完成  
  FBCSP、滤波、标准化、MLP、线性头扫描和概率融合等前向线性路径已经统一暴露并交由光计算 backend 接管，但当前主要是软件模拟、接口路由与 MAC-equivalent 统计，尚未在真实光芯片上逐算子完成执行与测量。

- 在线采集闭环未完成  
  OpenBCI Cyton 在线采集与上位机正在完善，当前尚未完成真实 EEG 数据流、信号质量检查、在线推理和控制反馈的稳定闭环验证。

- 实时优化空间  
  FBCSP 和滤波在在线系统中仍可能是延迟瓶颈。

- 跨被试差异  
  EEG/MI 个体差异仍是系统泛化和特化的主要挑战。

- 实时准确率不可得  
  真实在线没有即时真值标签，因此不能实时计算 accuracy。

- Reject 必要性  
  真实采集中被试意图、注意力和环境刺激不可完全控制，因此拒识机制是必要设计。

## 15. 后续工作

- 真实光芯片接入  
  保持上层算法不变，实现真实硬件驱动并替换当前软件模拟 backend，逐算子验证全部前向线性计算的真实芯片执行、数值误差、延迟和功耗。

- 完整量化实验  
  对单次 4-bit、8-bit 逻辑精度的 4-bit 位权拆分、原生 8-bit 和可选 QAT 进行精度、抗噪、物理调用次数与延迟对比。

- 逐算子最低精度搜索  
  建立浮点参考输出，对每个前向线性算子独立扫描逻辑位宽，确定满足任务指标的最低精度并形成混合精度配置，以减少无效 slice 和资源浪费。

- 光计算单元位宽拓宽  
  评估 5/6/8-bit 物理单元。重点验证位宽增加导致的抗噪下降是否仍在工程容限内，以及减少 slice 次数后能否带来更低延迟、更少累加误差和更高有效吞吐。

- 精度、抗噪与效率联合优化  
  不单独追求最低位宽或最高精度，而是以 command accuracy、accepted accuracy、reject rate、噪声鲁棒性、tile/slice 调用次数、延迟和能耗构建 Pareto 前沿并选择部署点。

- OpenBCI Cyton 上位机  
  继续完善并验证实时采集、信号质量检查、校准、推理、经验库管理和控制反馈，完成可重复的端到端在线闭环。

- 经验库管理系统  
  增加经验条目版本、分组、质量评估和回滚机制。

- 真实被试采集  
  建立真实采集协议，用于验证公开数据集之外的在线表现。

- 在线校准协议  
  优化少量校准样本数量、顺序和阈值设置方法。

- 延迟、功耗、抗噪测试  
  在真实硬件上评估系统实际部署价值。

- 图表和报告自动生成  
  将实验结果、可视化和文档输出流程进一步自动化。

## 16. 附录建议

- 附录 A：线性算子与计算量清单  
  列出主线每个前向线性算子的矩阵大小、MAC、tile 数、backend 调用、当前执行方式和证据文件，并明确全部前向线性算子均已纳入光计算 backend 接管口径。

- 附录 B：技术数据目录  
  列出源码、测试、metrics、figures、绘图脚本和运行命令，方便评审复查。

- 附录 C：提交材料核查表  
  对照赛题要求列出技术文档、源码、PPT、视频、匿名化和敏感信息检查项。

- 附录 D：实采在线流程说明  
  如果展示 Cyton 上位机，可单列数据包读取、ADC 换算、环形缓冲区、质量检查和在线窗口推理流程，并显著标注该能力正在完善、尚未完成真实闭环验证。

注意点：

- 附录中的结果数字必须和当前工程输出保持一致。
- 如果附录引用旧报告中的公式或表格，需要同步当前 4-bit 默认配置和当前 artifacts 指标。

## 17. 文档检查清单

- [ ] 是否明确 candidate scan 默认是单次 4-bit。
- [ ] 是否明确CAR从4-bit开始、SOS/FBCSP/标准化从6-bit开始、敏感路径保持8-bit，并统一由4-bit physical slices按位权重构。
- [ ] 是否说明低精度路径持续与8-bit数字影子比较，超差后当前调用立即升档重算且只升不降。
- [ ] 是否提出逐算子确认最低可用逻辑精度，避免统一高精度造成资源浪费。
- [ ] 是否讨论拓宽光计算单元物理位宽后抗噪能力与计算效率的折中。
- [ ] 是否说明 `2 x 8` 是 tile 尺寸，不是系统矩阵大小限制。
- [ ] 是否说明 `reject` 是拒识输出，不是训练类别。
- [ ] 是否说明 reject 在真实 EEG 采集中是正常现象，并具有避免误触发的安全意义。
- [ ] 是否区分 setup、calibration、online forward。
- [ ] 是否区分 rolling accuracy 和 cumulative accuracy。
- [ ] 是否说明实时 accuracy 只适用于离线回放。
- [ ] 是否说明当前不是真实光芯片实测结果。
- [ ] 是否明确按当前工程的 forward-only 接口与统计口径，前向传播全部线性计算均已由光计算 backend 接管。
- [ ] 是否区分“全部前向线性计算已由 backend 接管”“单次 4-bit”“bit-sliced 逻辑精度”和“真实光芯片逐算子实测”四个层次。
- [ ] 是否保留 FBCSP + shrinkage LDA baseline。
- [ ] 是否解释经验库的模型特化意义。
- [ ] 是否解释选择特化而非单一泛化模型的原因是 EEG/MI 个体差异显著。
- [ ] 是否解释 photonic scan 是根据光计算单元低功耗并行 MVM 特点设计的候选头扫描机制。
- [ ] 是否给出可复现命令和结果路径。
- [ ] 是否区分公开数据回放结果和真实在线采集结果。
- [ ] 是否明确 OpenBCI Cyton 在线采集/上位机能力正在完善，但尚未完成真实 EEG 在线闭环。
- [ ] 是否说明质量检查提前拒识与分类置信度拒识是两级机制。
- [ ] 是否确认所有结果数字来自当前 artifacts，而不是旧报告。
- [ ] 是否避免把“可接管线性计算占比”写成真实芯片功耗占比。
