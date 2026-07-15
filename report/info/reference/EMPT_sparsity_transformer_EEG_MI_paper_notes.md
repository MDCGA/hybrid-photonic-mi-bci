# EMPT: a sparsity Transformer for EEG-based motor imagery recognition 论文解析

> 论文：Ming Liu 等，**EMPT: a sparsity Transformer for EEG-based motor imagery recognition**，Frontiers in Neuroscience，2024-04-18。  
> DOI: <https://doi.org/10.3389/fnins.2024.1366294>  
> 原文：<https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1366294/full>

## 1. 一句话总结

这篇论文把 **MST/CSP 手工特征** 输入到一个改造过的 Transformer 中，用 **MoE** 扩大模型容量并做样本相关的专家选择，用 **ProbSparse self-attention** 稀疏化深层注意力计算，目标是在 **脊髓损伤患者的左右手运动想象 EEG 二分类** 上提升跨被试识别性能。

它的核心不是“纯端到端 EEG Transformer”，而是：

```text
原始 EEG
  -> 预处理与时频/空间特征提取：MST + CSP
  -> Transformer 编码器
  -> MoE 替换 FFN
  -> 深层 ProbSparse attention
  -> 全连接分类左右手 MI
```

## 2. 论文要解决的问题

运动想象脑机接口（MI-BCI）需要从 EEG 中判断受试者想象的是哪类动作，例如左手握拳或右手握拳。EEG 的难点很明显：

- 信噪比低，个体差异大。
- 医疗人群数据少，尤其是脊髓损伤（SCI）患者。
- Transformer 的自注意力适合建模多通道之间的全局关系，但参数量、计算量和数据需求都偏高。

作者的判断是：**MI-EEG 的多通道注意力存在稀疏性**，并且不同被试需要不同的映射方式。因此他们把稀疏机制引入 Transformer：

- 用 **MoE** 让不同样本激活不同专家，适配个体差异。
- 用 **ProbSparse attention** 只保留更“活跃”的 query，降低冗余注意力计算。

## 3. 数据集与任务设置

### 3.1 自采 SCI 患者数据集

论文使用 Qilu Hospital 采集的 SCI 患者 MI-EEG 数据：

- 被试：10 名脊髓损伤患者。
- 采集设备：64 电极 EEG。
- 任务：左右手运动想象二分类。
- 动作想象：左手握拳想象、右手握拳想象。
- 单次实验：完整 trial 约 7 秒，其中运动想象 4 秒，间隔 3 秒。
- 每名被试：4 组实验，每组 30 个 MI trial，共 120 个 trial。
- 类别平衡：每人 60 个左手、60 个右手。

也就是说，自采数据总量大约是：

```text
10 subjects * 120 trials = 1200 trials
```

### 3.2 外部公开数据集

作者还在 **BCI Competition IV-2A** 上做了对比实验，报告 9 名被试平均准确率为 **93.39%**。论文中主要把它作为泛化验证，但对具体训练/测试协议的说明相对简略。

## 4. 特征工程：MST + CSP

这篇论文很重要的一点是：EMPT 不是直接吃原始 EEG，而是先做特征提取。

### 4.1 MST：Modified S-transform

MST 用来提取时频特征。论文理由是 MI 相关信息主要集中在：

- alpha：8-13 Hz
- beta：13-30 Hz

MST 的作用可以理解为：对 EEG 做多分辨率时频分析，通过可调高斯窗更好地定位不同频率的能量变化。作者的预处理流程是：

- Butterworth 滤波：8-30 Hz。
- 降采样：1000 Hz -> 100 Hz。
- MST 提取后形成 `T x CH x Fmst` 特征。
- MST 高斯窗参数：`P = 0.98`，`Q = 0.49`。

### 4.2 CSP：Common Spatial Pattern

CSP 用来提取空间判别特征。它的基本目标是找到一组空间滤波器，使一类任务的方差最大化，同时让另一类任务的方差最小化。

论文里的 CSP 流程不是单一频段 CSP，而是多频带处理：

- 用 Butterworth 滤波器把原始 EEG 分解到 55 个频带。
- 频带窗口宽度为 2、4、8 Hz，步长 1 Hz。
- 把一个通道的多个频带看作新的多通道输入，再做 CSP。
- 降采样到 100 Hz。
- CSP 特征形状为 `T x CH x Fcsp`。

### 4.3 特征工程的意义

MST 和 CSP 分别给模型提供：

| 特征 | 关注点 | 作用 |
|---|---|---|
| MST | 时频能量与相位信息 | 捕捉 MI 相关频段变化 |
| CSP | 空间可分性 | 强化左右手 MI 的空间判别模式 |

这让 Transformer 的输入已经是相对“整理过”的时频-空间特征，而不是高噪声原始 EEG。

## 5. 模型结构：EMPT 怎么搭起来

EMPT 全称是 **EEG MoE-Prob-Transformer**。它是在 Transformer encoder 基础上做两个稀疏化改造。

### 5.1 Baseline Transformer

作者只使用 Transformer 的 encoder 结构。基础 TransformerBlock 包含：

- Multi-head self-attention
- Feed-forward network
- Residual connection
- Layer normalization

在 EEG 场景下，多头注意力的作用是让不同注意力头学习不同通道之间的全局关系。

### 5.2 MoE-TransformerBlock

作者把 TransformerBlock 中的普通 FFN 替换为 **MoE layer**。

MoE 的逻辑是：

- 准备多个 expert 子模型。
- gating network 根据输入给每个 expert 分配权重。
- 只保留 top-k 个专家参与计算。
- 其他专家权重置零，不参与当前样本的计算。

论文中的 expert 是全连接层，gating network 是线性层 + softmax。作者最终选择：

```text
K = 4
```

原因是实验中 `K >= 4` 后准确率提升很小，而 K=4 更省计算。

MoE 对 EEG 的解释是：不同被试、不同通道、不同 MI 表现可能需要不同映射，MoE 可以用专家选择来吸收个体差异。

### 5.3 ProbSparse self-attention

普通 self-attention 会计算所有 query-key 组合，复杂度高，而且很多注意力分数贡献很小。作者借鉴 Informer 中的 ProbSparse 思路：

- 如果某个 query 对 keys 的注意力分布接近均匀，说明它不太能挑出关键通道，称为“lazy query”。
- 如果某个 query 的注意力分布明显偏向少数 key，说明它更有信息量，称为“active query”。
- 用 KL divergence 风格的稀疏性度量选出 top-u query。
- 对主要 query 做注意力计算，减少冗余。

论文里一个关键观察是：EEG 多头注意力的点积分布呈长尾，少量通道或特征对注意力贡献更大。因此，深层注意力可以稀疏化。

### 5.4 MoE-Prob-TransformerBlock

MoE-Prob-TransformerBlock 同时包含：

- MoE 替换 FFN。
- ProbSparse self-attention 替换普通 multi-head self-attention。

论文最终最优结构是：

```text
M-P-FC
```

其中：

- `M` = MoE-TransformerBlock
- `P` = MoE-Prob-TransformerBlock
- `FC` = fully connected classifier

这意味着作者发现：**先用普通注意力聚合通道关系，再在深层使用 ProbSparse 更合适**。如果一开始就用 ProbSparse，可能会过早丢掉浅层尚未聚合充分的脑活动信息。

## 6. 训练设置

论文给出的主要训练参数如下：

| 参数 | 值 |
|---|---:|
| FC dropout | 0.5 |
| MoE-TransformerBlock dropout | 0.2 |
| MoE-Prob-TransformerBlock dropout | 0.2 |
| learning rate | 0.00005 |
| batch size | 256 |
| epochs | 300 |
| multi-head number | 8 |
| attention head hidden size | 128 |

实验评估使用十次 10-fold cross validation，并称为 cross-individual model training。

## 7. 关键实验结果

### 7.1 MoE 的 K 值选择

单层 MoE-Transformer 在不同 K 下的准确率：

| K | Accuracy |
|---:|---:|
| 1 | 86.74% |
| 2 | 88.43% |
| 4 | 89.73% |
| 6 | 89.75% |
| 8 | 89.88% |

结论：K=4 后收益很小，因此选 K=4。

### 7.2 消融实验

| Model | Block | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Transformer-Base | 1 | 88.52% | 89.34% | 87.68% |
| Transformer-Base | 2 | 93.56% | 94.19% | 92.38% |
| Transformer-Base | 3 | 90.07% | 89.46% | 90.67% |
| Transformer-Base | 4 | 86.67% | 87.72% | 85.63% |
| Transformer-Base | 5 | 85.34% | 85.12% | 84.88% |
| MoE-Transformer | 2 | 94.73% | 95.68% | 93.36% |
| Prob-Transformer | 2 | 93.85% | 92.61% | 93.96% |

解读：

- MoE 和 ProbSparse 都能提升 baseline。
- 两层结构表现最好。
- 更深反而下降，作者认为可能与数据规模小、注意力对噪声敏感有关。

### 7.3 最优堆叠结构

| Stacking | Blocks | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| M-FC | 1 | 89.73% | 90.52% | 88.98% |
| P-FC | 1 | 89.23% | 89.11% | 90.16% |
| M-P-FC | 2 | **95.24%** | **96.38%** | **94.88%** |
| M-M-FC | 2 | 94.73% | 95.68% | 93.36% |
| P-P-FC | 2 | 93.85% | 92.61% | 93.96% |
| P-M-FC | 2 | 93.66% | 92.82% | 94.08% |

作者认为 `M-P-FC` 最好，是因为浅层需要保留更完整的跨通道信息，深层特征已经聚合后更适合稀疏注意力。

### 7.4 与其他模型对比

在 SCI 患者自采数据集上：

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| CWT/PCA+SVM | 86.24% | 87.39% | 85.22% |
| EEGNet | 88.73% | 87.91% | 89.47% |
| HS-CNN | 89.36% | 90.27% | 89.34% |
| CNN+LSTM | 90.21% | 89.32% | 90.45% |
| ATC-Net | 92.44% | 91.62% | 93.33% |
| MSATNet | 93.59% | 94.45% | 93.18% |
| MSFT | 94.18% | 94.74% | 93.69% |
| EMPT | **95.24%** | **96.38%** | **94.88%** |

在 BCI Competition IV-2A 上：

| Model | Average Accuracy |
|---|---:|
| EEGNet | 74.61% |
| MI-DABAN | 76.16% |
| CNN-LSTM | 82.84% |
| EEG-Inception | 88.39% |
| CS-CNN | 90.37% |
| EMPT | **93.39%** |

## 8. 可解释性分析

论文做了两类可视化。

### 8.1 MoE gating 可视化

作者把 MoE gating network 的输出按被试平均，观察不同被试和不同通道对 8 个专家的选择情况。

结论是：

- MoE-TransformerBlock 中的 gating 对被试差异响应更明显。
- MoE-Prob-TransformerBlock 中的 gating 更相似，说明深层特征已经减少了一部分个体差异。
- 某些被试的部分通道出现特殊 gating 模式，作者认为这反映了 SCI 患者脑活动个体差异，也体现 MoE 的适配作用。

### 8.2 ProbSparse channel selection 可视化

作者把 ProbSparse 中的 query 稀疏度指标映射到 EEG 通道上，观察哪些通道更可能被保留。

他们强调：这个图不能直接等同于原始脑区重要性。因为深层输入已经经过浅层注意力加权，某个“通道特征”里可能混合了其他脑区信息。

这个提醒很关键：**深层 attention/channel selection 的可视化不是生理因果解释，只能作为模型内部特征选择倾向的参考。**

## 9. 这篇论文真正的贡献

我认为贡献可以归纳成三点：

1. **把 MoE 用到 MI-EEG Transformer 中**  
   通过专家选择机制处理多被试 EEG 的个体差异，同时扩大模型容量。

2. **把 ProbSparse attention 放到 EEG 通道特征重构中**  
   利用注意力分布长尾现象，减少冗余注意力计算，并提升深层特征稳定性。

3. **给出一个经验性结构原则**  
   对这个任务，`MoE-TransformerBlock -> MoE-Prob-TransformerBlock -> FC` 比一开始就稀疏化注意力更好。也就是说：浅层先充分融合，深层再稀疏筛选。

## 10. 论文的亮点

- 任务场景有实际康复意义：SCI 患者 MI-BCI。
- 没有只堆一个 Transformer，而是围绕 EEG 小样本、多被试差异和注意力冗余做结构改造。
- 消融实验比较清楚，能看到 MoE 和 ProbSparse 各自的增益。
- 对 MoE gating 和 ProbSparse channel selection 做了可视化，至少尝试解释模型行为。
- 同时报告自采数据集和 BCI IV-2A 结果。

## 11. 需要谨慎看的地方

### 11.1 数据集很小

自采 SCI 数据只有 10 名被试、约 1200 个 trial。对于 Transformer/MoE 这类模型来说，样本量偏小。论文也承认更深网络性能下降可能与数据规模不足有关。

### 11.2 自采数据不可公开

论文说明数据因伦理原因不能直接公开，需要向作者申请。这会影响独立复现。

### 11.3 cross-individual 的评估协议需要细看

论文写的是十次 10-fold cross validation 和 cross-individual model training，但没有把“训练集/测试集是否严格按被试划分”讲得非常细。

如果 k-fold 是在所有 trial 上随机划分，那么同一被试的 trial 可能同时出现在训练和测试中，结果会高于真正的 leave-one-subject-out 跨被试泛化。  
如果是按被试划分，则说服力更强。论文表述更像“多被试混合训练 + 交叉验证”，读结果时需要保留这个疑问。

### 11.4 特征工程占了很大贡献

EMPT 的输入已经经过 MST 和 CSP 处理，模型性能不能完全归因于 Transformer 稀疏结构。若要证明模型本身优势，最好补充：

- 仅 MST/CSP + 传统分类器的强基线。
- 原始 EEG 端到端输入的对照。
- 相同特征输入下更多轻量模型对比。

### 11.5 计算效率没有被充分量化

论文强调 MoE 和 ProbSparse 可以降低计算冗余，但主要报告分类指标，没有系统报告：

- 参数量。
- FLOPs。
- 推理延迟。
- 训练时间。
- 显存占用。

因此“稀疏带来效率提升”在这篇文章里更像结构动机，实证支撑不如准确率部分充分。

## 12. 如果要复现，应重点关注什么

复现时最容易踩坑的是特征和划分协议。

### 12.1 预处理

- 原始采样率：1000 Hz。
- MST：8-30 Hz Butterworth，降采样到 100 Hz。
- CSP：55 个频带，窗口宽 2/4/8 Hz，步长 1 Hz，降采样到 100 Hz。
- 注意 CSP 应该只在训练集上拟合空间滤波器，再用于测试集，避免数据泄漏。

### 12.2 模型

建议先复现最优结构：

```text
Input features
  -> MoE-TransformerBlock
  -> MoE-Prob-TransformerBlock
  -> FC classifier
```

关键超参：

```text
experts = 8
top_k = 4
heads = 8
head_hidden = 128
dropout_block = 0.2
dropout_fc = 0.5
lr = 5e-5
batch_size = 256
epochs = 300
```

### 12.3 评估

建议同时报告两套结果：

- trial-level random 10-fold，用来对齐论文。
- leave-one-subject-out，用来验证真正跨被试泛化。

如果二者差距很大，说明论文中的高准确率可能更多来自被试内模式被训练集覆盖。

## 13. 对你做 EEG/BCI 项目的启发

如果你要基于这篇论文继续做工程或研究，我建议把它看作一个“稀疏 Transformer + 手工 EEG 特征”的设计模板：

- 对小样本 EEG，不要急着端到端。MST、CSP、FBCSP 这类特征仍然很强。
- Transformer 可以用来做多通道特征重构，但要控制深度。
- MoE 适合多被试场景，尤其当你怀疑不同被试需要不同映射时。
- ProbSparse 更适合放在深层，而不是一上来就筛掉浅层通道信息。
- 论文结果很漂亮，但要用更严格的跨被试划分重新验证。

## 14. 总体评价

EMPT 的思路是合理的：先用传统 EEG 特征降低输入难度，再用 Transformer 建模通道间关系，最后用 MoE 和 ProbSparse 引入两种稀疏性，分别应对个体差异和注意力冗余。

这篇论文最有价值的地方不是“又一个 Transformer 分类器”，而是提出了一个适合小样本、多被试 EEG 的结构直觉：

```text
浅层：保留信息，充分融合；
中间：用 MoE 适配个体差异；
深层：用 ProbSparse 去掉冗余注意力。
```

但从严格科研角度看，它还需要更透明的数据划分、更完整的效率指标和更开放的复现实验来支撑“跨个体泛化”和“计算效率提升”这两个核心卖点。

