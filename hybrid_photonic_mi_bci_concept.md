# 光电混合 EEG-MI BCI 设计稿

日期：2026-07-05

## 1. 目标

构建一个面向运动想象 EEG 的光电混合在线校准系统，输出四类决策：

```text
左手 / 右手 / 脚 / 拒识
```

核心目标不是证明光计算可以替代数字 EEG 解码，而是证明：

```text
在标准 FBCSP 解码框架下，
利用可 tiled 的光矩阵乘单元快速扫描经验库中的候选校准器，
从而支持新被试快速特化、在线校准和低功耗嵌入式部署。
```

主线系统：

```text
FBCSP + 小型网络 embedding + 经验库检索 + 光计算多候选线性头扫描
```

传统参考基线：

```text
FBCSP + shrinkage LDA
```

低复杂度消融：

```text
log-bandpower + LDA
```

普通 log-bandpower 仅保留为调试或消融对照，不作为主线或正式参考基线。

## 2. 总体架构

系统分为五个核心层：

```text
EEG 数据
  -> FBCSP 特征层
  -> 小型网络 embedding / 特征重加权
  -> 经验库检索与候选校准器生成
  -> 光计算多候选线性头扫描
  -> 数字端融合、拒识、在线特化和经验库回写
```

新被试首次使用或需要重新校准时：

```text
少量校准 trial
  -> 提取 FBCSP 特征
  -> 小型网络生成 subject/session embedding
  -> 从经验库检索 top-K 相似模板
  -> 初始化候选校准器集合
  -> 光计算多候选线性头扫描
  -> 数字端融合 / reject / bandit 更新
  -> 稳定后将新 session 写回经验库
```

## 3. FBCSP 特征与传统参考基线

### 3.1 预处理

推荐第一版处理流程：

```text
EEG
  -> 工频陷波，按数据集情况启用
  -> CAR 或 Laplacian
  -> trial window，例如 marker 后 1.0-4.0 s
  -> filter bank
```

所有参数必须只在训练段拟合。replay / test 段不得用于：

- CSP 协方差估计
- 特征选择
- 归一化参数拟合
- LDA / 网络训练

### 3.2 Filter Bank

第一版建议覆盖 mu / beta 节律：

```text
8-12 Hz
12-16 Hz
16-20 Hz
20-24 Hz
24-28 Hz
28-32 Hz
```

后续可评估重叠频带，但第一版优先控制复杂度和过拟合风险。

### 3.3 多分类 CSP

三分类任务推荐使用 one-vs-rest CSP：

```text
left  vs rest
right vs rest
foot  vs rest
```

每个频带、每个 one-vs-rest 任务：

```text
训练段协方差估计
  -> shrinkage / 正则化
  -> 广义特征值分解
  -> 选择前后 m 对 CSP filters
  -> log-variance features
```

原始 FBCSP 特征维度：

```text
D_raw = n_bands * n_ovr_tasks * 2m
```

示例：

```text
n_bands = 6
n_ovr_tasks = 3
m = 2
D_raw = 6 * 3 * 4 = 72
```

### 3.4 特征选择

FBCSP 特征维度较高，需要训练段内完成特征选择。可选方法：

- mutual information
- Fisher score
- mRMR
- cross-validation feature selection
- LDA shrinkage 辅助稳定估计

建议实验点：

```text
D_sel = 8, 16, 24, 32
```

`D_sel = 8` 便于直观映射单个 `2 x 8` tile；`D_sel > 8` 更贴近 FBCSP 实际性能需求，通过 tiled MVM 支持。

### 3.5 分类器

主分类器采用 shrinkage LDA：

```text
score = A x + b
```

其中：

```text
x in R^D_sel
A in R^(3 x D_sel)
b in R^3
```

光计算负责：

```text
A x
```

数字端负责：

- bias `b`
- softmax / probability calibration
- reject threshold
- temporal smoothing
- candidate fusion
- online adaptation

SVM、logistic regression、Riemannian tangent-space classifier 可作为对照，不作为主线系统。

## 4. FBCSP 后的小型网络

本项目需要小型网络，而不是大型端到端深度模型。小型网络的作用：

- 对 FBCSP 特征进行非线性重加权
- 学习 subject/session embedding
- 辅助经验库 top-K 模板检索
- 生成适合候选线性头扫描的低维表示

推荐结构：

```text
FBCSP features
  -> normalization
  -> small encoder
  -> embedding h
  -> candidate linear heads A_i h + b_i
```

光计算负责候选线性头：

```text
A_i h
```

### 4.1 FBCSP-MLP

小型网络第一实现：

```text
FBCSP vector x
  -> LayerNorm / BatchNorm
  -> Linear(D, 64)
  -> GELU / ReLU
  -> Dropout(0.1-0.3)
  -> Linear(64, 32)
  -> embedding h
```

建议参数规模：

```text
1k - 20k parameters
```

### 4.2 Band-Attention FBCSP Network

保留 FBCSP 的频带结构：

```text
X_fbcsp in R^(n_bands x n_tasks x n_csp_features)
```

网络结构：

```text
每个频带 CSP 特征
  -> shared band encoder
  -> band attention weights
  -> weighted aggregation
  -> embedding h
```

用途：

- 学习新被试更依赖 mu 还是 beta 频段
- 形成可解释的频带偏好
- 将 attention pattern 作为经验库检索特征

### 4.3 Metric Embedding Network

用于经验库检索：

```text
FBCSP features
  -> small encoder
  -> normalized embedding h
  -> 与经验库 embedding 比较
  -> top-K 模板检索
```

可选训练损失：

- supervised contrastive loss
- triplet loss
- prototypical loss
- classification loss + metric loss

## 5. 经验库

经验库是系统特化能力的核心。它不只存原始 EEG，而是存可复用校准经验。

### 5.1 经验条目

一个经验条目 `E_i` 包含：

```text
E_i = {
  subject_id / session_id,
  dataset / device metadata,
  channel set,
  preprocessing config,
  filter bank config,
  CSP filters,
  selected feature indices,
  feature normalization parameters,
  small-network encoder version,
  subject/session embedding,
  LDA score matrix A_i,
  LDA bias b_i,
  optional neural linear head,
  reject calibration parameters,
  performance metrics
}
```

### 5.2 候选生成

新被试校准时，从经验库生成候选集合：

```text
新被试少量 trial
  -> FBCSP features
  -> embedding h_new
  -> 检索 top-K 相似经验条目
  -> 得到候选校准器 C_1...C_K
```

候选来源：

- 相似 subject/session 模板
- 不同 filter bank 配置
- 不同 CSP 正则化强度
- 不同 feature subset
- 不同 LDA shrinkage 参数
- bootstrap / cross-validation 子模型
- subject-independent 模板加 subject-specific 微调

候选不是随机扰动矩阵，而应来自真实训练过程或真实历史经验。

### 5.3 经验库回写

新 session 满足稳定性要求后写回经验库：

```text
稳定 accuracy
合理 reject rate
无明显数据质量异常
校准 trial 数达到最低要求
```

回写内容包括：

- 新 subject/session embedding
- 更新后的特征统计
- 校准后的线性头
- reject 参数
- 在线表现指标

## 6. 光计算多候选线性头扫描

本文中的“光计算多候选线性头扫描”指：

```text
对经验库检索出的 K 个候选校准器，
使用光计算 tile 快速计算每个候选线性头的矩阵乘部分。
```

若当前 EEG 窗口的小型网络 embedding 为 `h`，第 `i` 个候选线性头为：

```text
score_i = A_i h + b_i
```

则光计算负责批量计算：

```text
A_1 h, A_2 h, ..., A_K h
```

数字端负责：

```text
b_i
softmax / probability calibration
reject
candidate fusion
contextual bandit update
```

这个过程不是只运行一个固定分类器，而是在每个 EEG 决策窗口内评估多个来自经验库的候选校准器。

`2 x 8` 是光计算 tile，不是算法维度上限。

目标计算：

```text
y = W x
W in R^(M x D)
x in R^D
```

使用 `2 x 8` tile 时，单个候选需要：

```text
tile_count = ceil(M / 2) * ceil(D / 8)
```

N 个候选需要：

```text
total_tiles = N * ceil(M / 2) * ceil(D / 8)
```

示例：FBCSP-LDA score matrix

```text
N = 32
M = 3
D = 24
total_tiles = 32 * ceil(3/2) * ceil(24/8)
            = 192 tile evaluations / decision window
```

数字端保留：

- bias
- softmax
- reject
- fusion
- partial sum accumulation
- calibration compensation

硬件非理想性需要评估：

- weight quantization
- input quantization
- non-negative weight constraint
- differential encoding
- detector noise
- gain / bias drift
- tile accumulation error
- ADC / DAC 开销

## 7. 在线特化

在线更新不直接无约束修改完整分类器，而是在经验库候选集合内选择、加权和小幅校正。

推荐第一版：

```text
contextual bandit
```

Action：

- 选择候选 `C_i`
- 调整候选权重 `alpha_i`
- 调整 reject threshold
- 调整类别 bias
- 调整 temporal smoothing
- 调整 fusion temperature

State：

- FBCSP 特征统计
- 小型网络 embedding
- 候选 confidence
- 候选间分歧
- 当前 reject rate
- rolling accuracy
- 类别分布
- 信号质量指标

Reward：

```text
正确分类：+1
错误分类：-1
高置信错误：额外惩罚
合理拒识：小惩罚
过度拒识：惩罚
候选高分歧时谨慎输出：低惩罚或小奖励
```

## 8. 实验设计

### 8.1 主线系统

```text
FBCSP
  -> small MLP / band attention embedding
  -> experience library top-K retrieval
  -> candidate linear heads
  -> 光计算多候选线性头扫描
  -> fusion / reject / online specialization
```

指标：

- accuracy
- balanced accuracy
- per-class recall
- confusion matrix
- reject rate
- calibration trial 敏感性
- top-K 检索收益
- 候选权重演化
- 新 session 回写后的经验库增益

### 8.2 传统参考基线

```text
FBCSP + shrinkage LDA
```

作用：

- 提供标准 MI 算法参照
- 评估主线系统相对传统 FBCSP-LDA 的增益
- 检查小型网络、经验库和候选扫描是否真正带来价值

指标：

- accuracy
- balanced accuracy
- per-class recall
- confusion matrix
- reject-free performance

### 8.3 小型网络消融

```text
FBCSP + small MLP
FBCSP + band attention
FBCSP + metric embedding
```

指标：

- 分类性能
- embedding 可视化
- band attention 权重
- 参数量
- calibration trial 敏感性

### 8.4 经验库消融

对比：

- 无经验库
- 随机候选
- 全库候选
- top-K 相似模板候选
- top-K + online bandit

指标：

- 少样本校准性能
- 达到稳定准确率所需 trial 数
- top-K 命中率
- 候选权重演化
- 新 session 回写后的经验库增益

### 8.5 光计算仿真

对比：

- ideal digital MVM
- tiled MVM
- quantized tiled MVM
- non-negative / differential encoded MVM
- noisy / drifting photonic MVM

指标：

- 每窗口 tile 数
- 估计延迟
- 估计能耗
- accuracy drop
- reject rate change

## 9. 数据集策略

### 9.1 主 benchmark

优先使用 BCI Competition IV 2a：

- 原生四类 MI
- 可选择 `left/right/foot` 三类，去掉 tongue
- 更适合 FBCSP 主实验

### 9.2 原型 replay

BCICIV_1_asc 可继续用于：

- 快速原型验证
- 经验库流程演示
- 候选检索 / 融合 / reject 机制验证

注意：

- `a-g` 文件本身是局部二分类
- 合并三分类时必须严格按 `nfo` 标签映射
- FBCSP 参数只能使用训练段估计

### 9.3 经验库扩展

后续可加入：

- PhysioNet EEG Motor Movement/Imagery
- 多 session / 多天 MI 数据集
- High-Gamma Dataset

用途：

- 增加 subject/session 多样性
- 构建跨被试经验库
- 验证模型特化和跨 session 鲁棒性

## 10. 实施路线

阶段 1：传统参考基线

- 实现 one-vs-rest FBCSP
- 实现 feature selection
- 实现 shrinkage LDA
- 输出标准分类指标和可视化

阶段 2：主线小型网络

- 实现 FBCSP-MLP
- 实现 band-attention network
- 输出 embedding 和 band attention 可视化

阶段 3：经验库

- 设计经验库数据结构
- 存储 subject/session 模板
- 实现 embedding top-K 检索
- 实现新 session 回写策略

阶段 4：候选扫描与在线特化

- 从经验库生成候选校准器
- 实现光计算多候选线性头扫描的软件接口
- 实现 candidate fusion
- 实现 contextual bandit
- 输出 rolling accuracy、reject rate、候选权重曲线

阶段 5：光计算 tiled MVM 仿真

- 将候选线性头映射到 `2 x 8` tile
- 统计 tile 数、延迟、能耗
- 加入量化、噪声、漂移和差分编码

阶段 6：硬件在环

- 数字端完成 FBCSP 和小型网络
- 光芯片完成多候选线性头 MVM
- 数字端完成 bias、fusion、reject 和在线更新

## 11. 系统框图必须体现

系统框图应明确标出：

```text
Filter Bank
  -> OVR CSP
  -> log-variance
  -> feature selection
  -> small MLP / band attention embedding
  -> experience library top-K retrieval
  -> candidate linear heads
  -> 光计算多候选线性头扫描
  -> digital bias / softmax / reject
  -> fusion / contextual bandit
  -> experience library write-back
```

必须明确：

```text
2 x 8 是光计算 tile，不是算法维度上限。
```

## 12. 修正后的研究问题

在标准 FBCSP 解码框架下，能否利用小型网络和经验库实现新被试快速特化，并利用可 tiled 的 `2 x 8` 光计算单元高吞吐扫描候选线性头，从而在在线校准、跨 session 鲁棒性和嵌入式能效方面优于固定数字模型？
