# 光电混合 EEG-MI BCI 项目构想

日期：2026-06-10

## 1. 核心想法

构建一个面向左手、右手、脚三分类运动想象 EEG-BCI 的光电混合嵌入式系统。

现有光芯片核心能力是 `2 x 8` 矩阵乘。这个规模如果只用于一次普通分类推理，计算量太小，很难体现光计算价值。因此更合理的定位是：让光芯片作为在线校准环节中的低功耗、高速矩阵乘协处理器，用于快速评估多个候选校准投影矩阵。

核心结构：

```text
8维 EEG 特征 x
  -> 多个候选 2 x 8 投影矩阵 W_i
  -> 光芯片计算 z_i = W_i x
  -> 数字端分类 / 融合 / 拒识
  -> 强化学习或 contextual bandit 在线校准
```

光芯片的价值不是“算一次 16 个乘加”，而是“在每个 EEG 决策窗口内低功耗快速扫描多个候选校准器”。

## 2. 系统定位

这个项目不应表述为：

```text
没有光计算就无法完成 EEG-MI 解码
```

因为三分类 MI 的基础模型很小，MCU 也能计算一次 `2 x 8` 矩阵乘。

更合理的表述是：

> 面向小型嵌入式 EEG-MI BCI 的光电混合在线校准架构：利用 `2 x 8` 光矩阵乘核心，对多候选判别投影进行高速评估，并通过数字端安全在线校准实现受试者自适应。

也就是说：

- 数字端负责 EEG 预处理、特征提取、安全策略、在线策略更新。
- 光芯片负责重复密集线性投影计算。
- RL / contextual bandit 根据当前受试者反馈选择、加权或微调候选投影。

## 3. 目标任务

三分类运动想象 EEG：

- 左手
- 右手
- 脚

以及：

- 拒识 / 不确定

拒识状态很重要。MI 信号噪声大、个体差异明显，如果每个低置信窗口都强行输出三类之一，会导致在线校准更不稳定。

## 4. 基线信号流程

推荐第一版流程：

```text
EEG 采集
  -> 工频陷波
  -> 8-30 Hz 带通滤波
  -> CAR 或 Laplacian 参考
  -> 1.5-3.0 s 滑动窗口
  -> 特征提取
  -> 光芯片投影
  -> 数字端分类与在线校准
```

候选 8 维特征：

1. CSP log-variance 特征。
2. Filter-bank CSP 压缩后的 8 维特征。
3. Riemannian tangent-space 压缩后的 8 维特征。
4. C3、C4、Cz 及邻近通道的频带能量特征。

推荐第一版：

```text
8个运动区相关通道：
C3, C4, Cz, FC3, FC4, CP3, CP4, CPz

8维特征：
x = CSP / log-bandpower / Riemannian 压缩特征
```

## 5. 为什么 2 x 8 适合三分类 MI

三分类不一定需要 3 个输出维度。对于 LDA 这类判别投影，`K` 类最多只需要 `K - 1` 个判别维度。

因此：

```text
3类 -> 2维判别空间
```

这和光芯片的 `2 x 8` 矩阵乘天然匹配：

```text
x in R^8 -> z in R^2
```

三类 MI 可以在二维判别空间里形成三个簇或三个决策区域。

数字端后处理可以使用：

- 最近质心
- LDA 决策边界
- 基于距离的 softmax
- 置信度阈值
- 拒识区间
- 时间平滑

## 6. 如何体现光计算价值

单次矩阵乘：

```text
z = W x, W in R^(2 x 8)
```

计算量太小，不足以体现光芯片优势。

需要把它扩展为候选矩阵库：

```text
z_i = W_i x, i = 1...N
```

每个 `W_i` 代表一种候选校准状态，例如：

- 不同受试者模板
- 不同 session 模板
- 不同频带权重
- 不同空间滤波状态
- 不同正则化强度
- 不同判别边界

示例：

```text
N = 32 个候选矩阵
每个 W_i 都是 2 x 8
每个 EEG 窗口内进行 32 次光矩阵乘
数字端选择或融合结果
```

更强版本：

```text
6个频带 x 32个候选矩阵 = 每个决策窗口 192 次 2 x 8 投影
```

这样光芯片就不再是象征性模块，而是在线校准中的重复线性代数引擎。

## 7. 在线校准策略


```text
离线预训练候选矩阵库
在线阶段由 RL / contextual bandit 选择或加权候选矩阵
数字端小幅更新阈值、类别先验、质心和拒识区间
```

可选 action：

- 选择某个候选矩阵 `W_i`
- 调整候选矩阵权重 `alpha_i`
- 调整类别偏置
- 调整 reject 阈值
- 调整时间平滑窗口
- 调整置信度阈值
- 小幅更新二维空间中的类别质心

可选 state：

- 当前二维投影 `z`
- 候选模型置信度
- 信号质量指标
- 最近正确率
- 最近类别分布
- 当前拒识率
- 当前受试者 / session 状态

可选 reward：

```text
分类正确：+1
分类错误：-1
高置信错误：更大惩罚
低置信拒识：小惩罚
连续稳定正确：额外奖励
拒识过多：惩罚
```

这个方向更接近安全在线自适应，而不是无约束强化学习。

## 8. 候选矩阵库来源

候选 `W_i` 可以来自：

1. 不同受试者训练出的投影矩阵。
2. 同一受试者不同 session 的投影矩阵。
3. 不同频带的投影矩阵。
4. 不同特征归一化状态。
5. 不同正则化强度。
6. 围绕 subject-independent baseline 的小扰动。
7. 不同训练子集得到的 CSP / Riemannian / LDA 投影。

在线学习器不需要搜索任意连续矩阵空间，只需要在受限候选库里选择或加权：

```text
W_online = sum_i alpha_i W_i
```

或：

```text
W_online = W_argmax
```

光芯片负责快速评估多个 `W_i`，数字端决定哪些输出可信。

## 9. 光芯片映射方式

`2 x 8` 光矩阵乘核心计算：

```text
y0 = w00*x0 + w01*x1 + ... + w07*x7
y1 = w10*x0 + w11*x1 + ... + w17*x7
```

需要注意：

- 如果光芯片权重只能为非负，需要差分编码：`W = W+ - W-`。
- 如果输入特征存在负值，需要偏置编码或差分输入编码。
- bias 可以放在数字端补偿。
- 权重量化和漂移需要标定表。
- ADC / DAC 开销必须计入延迟和能耗。

数字端应负责补偿光芯片非理想性：

- 输出归一化
- 增益补偿
- 偏置校正
- 温漂校正
- 置信度校准

## 10. 最小演示路线

阶段 1：纯软件基线

- 用公开 MI 数据集构建三分类解码器。
- 训练 8 维特征和 2 维判别投影。
- 输出离线准确率、balanced accuracy 和混淆矩阵。

阶段 2：多候选校准仿真

- 生成 `W_1...W_N`。
- 在 held-out session 上模拟在线校准。
- 比较固定 `W`、候选选择、候选加权。

阶段 3：光芯片非理想仿真

- 用量化、噪声、漂移、非负权重约束模拟光芯片输出。
- 比较理想数字矩阵乘和光芯片近似矩阵乘的准确率差异。

阶段 4：硬件在环 demo

- PC 或嵌入式端回放公开 EEG 数据。
- 数字端提取 8 维特征。
- 将特征和候选矩阵送入光芯片。
- 光芯片输出二维投影。
- 数字端完成融合、拒识、在线校准和三分类输出。

## 11. 评估指标

BCI 性能：

- 三分类准确率
- balanced accuracy
- 混淆矩阵
- 信息传输率 ITR
- 拒识率
- 在线适应速度
- 跨 session 准确率
- 个体差异

校准性能：

- 达到稳定准确率所需 trial 数
- 相比固定模型的提升幅度
- 对噪声反馈的鲁棒性
- 类别不平衡下的稳定性

硬件性能：

- 单次 MVM 延迟
- 单次 MVM 能耗
- 单次决策总能耗
- ADC / DAC 开销
- 多候选矩阵吞吐率
- 光芯片非理想性导致的准确率下降

关键对比不应是：

```text
一次光学 2 x 8 MVM vs 一次 MCU 2 x 8 MVM
```

而应是：

```text
在线校准中的多候选投影扫描
光芯片 MVM 循环/阵列 vs MCU 顺序计算
```

## 12. 主要风险

风险 1：MI 信号质量主导系统表现。

缓解：

- 使用 reject 状态。
- 先用公开数据集验证架构。
- 先使用稳定可解释特征。
- 避免过激在线学习。

风险 2：`2 x 8` 计算量太小，难以支撑光计算优势。

缓解：

- 使用候选矩阵库。
- 使用 filter-bank 扩展。
- 使用在线校准中的重复候选评估。
- 按完整候选扫描报告能耗和延迟。

风险 3：在线 RL 不稳定。

缓解：

- 先用 contextual bandit。
- 限制 action 空间。
- 优先更新阈值和候选权重，而不是直接更新分类器主体。
- 保留固定 baseline 作为安全回退。

风险 4：光芯片非理想性降低准确率。

缓解：

- bias 和最终决策留在数字端。
- 标定光输出。
- 使用低维鲁棒判别规则。
- 用实测芯片噪声模型评估。

## 13. 核心研究问题

一个 `2 x 8` 光矩阵乘核心，能否通过加速多候选在线校准，而不是加速单个固定分类器，在小型嵌入式 EEG-MI BCI 中体现实际价值？

更具体地说：

> 在三分类 EEG-MI BCI 中，能否利用 `2 x 8` 光矩阵乘核心快速评估一组受试者自适应判别投影，并由数字端 contextual bandit 策略选择或加权这些投影，从而提升在线校准速度和跨 session 鲁棒性？

## 14. 当前推荐架构

```text
EEG
  -> 模拟前端与 ADC
  -> 数字预处理
  -> 8维特征提取
  -> 光芯片 2 x 8 MVM 候选投影库
  -> 数字概率融合
  -> reject / 平滑 / 安全逻辑
  -> contextual bandit 在线校准
  -> 三分类 MI 指令
```

## 15. 公开数据集是否足够

公开数据集足够支持：

- 算法 baseline 验证
- 三分类 MI 特征工程
- 候选投影矩阵库构建
- 跨受试者和跨 session 仿真
- 在线校准 replay 实验
- 光芯片在环仿真
- 基于记录 EEG 流的硬件计算价值评估

公开数据集不足以独立证明：

- 消费级干电极现场可靠性
- 真实用户闭环体验
- 人和 BCI feedback 的实时共适应效果
- 电极接触变化、运动伪迹、佩戴误差下的鲁棒性

推荐数据集组合：

1. BCI Competition IV 2a：主三分类 benchmark，使用左手、右手、脚，去掉舌头。
2. PhysioNet EEG Motor Movement/Imagery：受试者多，适合跨受试者候选库预训练。
3. High-Gamma Dataset：trial 多、通道多，适合模型压力测试。
4. 多天 / 多 session MI 数据集：适合验证在线校准和跨天漂移。

第一阶段可以采用：

```text
公开 EEG 数据
  -> replay 在线校准
  -> 光矩阵乘仿真 / 硬件在环
```

需要明确边界：

```text
公开数据阶段验证混合解码架构和在线校准机制。
真实干电极现场实验用于最终验证可穿戴/消费级应用。
```

## 16. 下一步讨论重点

1. 第一版 baseline 选择 CSP-LDA、Riemannian-LDA，还是 EEGNet 压缩特征。
2. 如何从公开数据集中生成 `W_i` 候选矩阵库。
3. 在线校准算法选择：epsilon-greedy、LinUCB、Thompson sampling，还是安全 RL。
4. 如何定义最能体现光芯片价值的硬件指标。
5. 先完成软件仿真，再进入光芯片硬件在环。

## 17. 相关论文线索

直接相关的 MI + RL / 在线自适应文献：

1. Liu et al., "Online Adaptive Decoding of Motor Imagery Based on Reinforcement Learning", ICIEA 2019.
   - 关键词：motor imagery, online adaptive decoding, reinforcement learning。
   - 适合参考如何把 MI 在线解码写成 RL 问题。

2. Luo et al., "Latent Belief Reinforcement Learning for Online Motor Imagery Classification", Pattern Recognition 2026.
   - 关键词：online MI classification, POMDP, latent belief, adaptive halting。
   - 适合参考如何把在线 MI 分类写成部分可观测决策过程。

3. Aung et al., "EEG_RL-Net: Enhancing EEG MI Classification through Reinforcement Learning-Optimised Graph Neural Networks", arXiv 2024 / ICMLA 2024.
   - 关键词：MI classification, Dueling DQN, GNN, PhysioNet。
   - 更偏深度学习分类增强，不完全是安全在线校准，但可作为 RL+MI 的近期案例。

4. Fidencio et al., "Error-related Potential driven Reinforcement Learning for adaptive Brain-Computer Interfaces", arXiv 2025.
   - 关键词：ErrP, reinforcement learning, adaptive BCI, motor imagery。
   - 适合参考如何用错误相关电位作为隐式反馈信号。

强相关的 co-adaptive / calibration 文献：

5. Acqualagna et al., "Large-Scale Assessment of a Fully Automatic Co-Adaptive Motor Imagery-Based Brain Computer Interface", PLOS ONE 2016.
   - 关键词：co-adaptive SMR-BCI, large-scale, online feedback。
   - 证明 co-adaptive MI-BCI 在大样本新手用户中的可行性，也提醒存在 BCI illiteracy / performance variability。

6. Abu-Rmileh et al., "Co-adaptive Training Improves Efficacy of a Multi-Day EEG-Based Motor Imagery BCI Training", Frontiers in Human Neuroscience 2019.
   - 关键词：multi-day MI training, co-adaptation, fixed classifier comparison。
   - 适合支撑本项目的跨天在线校准动机。

7. Škola et al., "Progressive Training for Motor Imagery Brain-Computer Interfaces Using Gamification and Virtual Reality Embodiment", Frontiers in Human Neuroscience 2019.
   - 关键词：MI-BCI training, co-adaptive event-driven training, gamification, VR embodiment。
   - 适合参考反馈设计和用户训练设计。

8. "A Transfer Learning Algorithm to Reduce Brain-Computer Interface Calibration Time for Long-Term Users", Frontiers in Neuroergonomics 2022.
   - 关键词：calibration time reduction, long-term users, transfer learning, inter-session non-stationarity。
   - 适合支撑为什么需要在线校准和候选模型库。

对本项目最有用的结论：

```text
直接套复杂 RL 训练完整 MI 分类器风险较高；
更稳的是把 RL / bandit 用于在线选择、加权、拒识和阈值调节；
候选模型库 + 光芯片快速扫描，是更适合当前 2 x 8 光矩阵乘核心的路线。
```
