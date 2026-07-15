# EEG-MI 数字预处理详细方案

日期：2026-07-03

## 1. 目标

本方案规划从 OpenBCI Cyton 采集到的原始 EEG 数据，经过数字预处理后，生成可送入 Gazelle 光子计算平台的运动想象特征。

预处理目标不是直接完成最终分类，而是把低信噪比、易漂移、带伪迹的原始脑电波，变成稳定、可量化、可用于三分类判别的特征向量。

最终输出：

```text
原始 EEG 数据
  -> 数字预处理
  -> 8维运动想象特征 x_float
  -> uint8 量化特征 x_uint8
  -> Gazelle 光矩阵乘候选投影
  -> 左手 / 右手 / 脚
```

## 2. 预处理在系统中的位置

```text
OpenBCI Cyton
  -> 数据接收与解码
  -> 单位换算与通道整理
  -> 信号质量检测
  -> 去漂移 / 陷波 / 带通滤波
  -> 重参考
  -> 滑动窗口切片
  -> 伪迹检测与拒识标记
  -> 特征提取
  -> 标准化与量化
  -> Gazelle 光子计算
```

数字预处理承担四个职责：

1. 降噪：去掉工频干扰、低频漂移、无关高频噪声。
2. 对齐：把连续 EEG 流切成可判别的时间窗口。
3. 提纯：提取与运动想象相关的 mu / beta 节律特征。
4. 适配硬件：把浮点 EEG 特征转换成 Gazelle 接收的 `uint8` 输入。

## 3. 输入条件

当前目录中的 OpenBCI Cyton 规格给出以下关键条件：

- 采集板：OpenBCI Cyton Board。
- 模拟前端：ADS1299。
- 通道数：Cyton 本体 8 通道，Daisy 模块可扩展到 16 通道。
- 信噪比：121 dB。
- 电压分辨率：0.298 microvolt / bit。
- 输入范围：+2.5V 到 -2.5V。
- 数据链路：USB Dongle 或 microSD。
- 供电：3-6V DC 电池。

本项目第一版建议使用 8 通道，避免 Daisy 扩展增加佩戴和调试复杂度。

推荐 8 个运动区相关位置：

```text
C3, C4, Cz, FC3, FC4, CP3, CP4, CPz
```

如果实际电极帽或 OpenBCI 头戴方案无法完整覆盖这些位置，可以先采用：

```text
C3, C4, Cz, FC1, FC2, CP1, CP2, Pz
```

核心原则是覆盖左右运动皮层及其前后邻域。

## 4. 输出定义

每个决策窗口输出一组结构化结果：

```python
{
    "window_id": int,
    "timestamp": float,
    "x_float": float[8],
    "x_uint8": uint8[8],
    "signal_quality": {
        "bad_channels": list[int],
        "saturation": bool,
        "motion_artifact": bool,
        "line_noise_score": float,
        "reject_recommended": bool
    },
    "scale": float,
    "zero_point": int
}
```

其中：

- `x_float` 是模型训练和调试使用的浮点特征。
- `x_uint8` 是送入 Gazelle 的量化特征。
- `signal_quality` 供后续分类器决定是否拒识。
- `scale` 和 `zero_point` 用于光计算输出反量化和调试复现。

## 5. 具体预处理流程

### 5.1 数据接收与包解析

输入：

```text
Cyton 原始数据包
```

处理：

1. 从 USB Dongle 串口或 OpenBCI GUI 数据流接收样本。
2. 解析样本序号、8 通道 EEG 原始计数值、可选加速度计数据。
3. 检查丢包、乱序和时间戳间隔。
4. 写入环形缓冲区。

目的：

- 保证实时 EEG 流连续。
- 发现无线传输丢包。
- 为后续滑动窗口提供稳定数据源。

建议：

- 每个样本保留原始计数值和换算后的微伏值。
- 若发现连续丢包，应标记当前窗口为低质量，不用于在线校准。

### 5.2 单位换算

输入：

```text
raw_counts[channel, sample]
```

根据规格中的分辨率：

```text
eeg_uv = raw_counts * 0.298
```

输出：

```text
eeg_uv[channel, sample]
```

目的：

- 将硬件原始计数转为微伏单位。
- 让阈值、滤波和质量检测具有物理意义。

说明：

- 0.298 microvolt / bit 来自当前 OpenBCI Cyton 规格文件。
- 如果后续 SDK 或配置给出不同增益，应以实际配置为准。

### 5.3 通道映射与坏通道初筛

输入：

```text
8通道 eeg_uv
```

处理：

1. 建立通道编号到电极位置的映射。
2. 检测每个通道是否长时间接近常数。
3. 检测是否出现饱和、断连、异常大幅值。
4. 记录坏通道列表。

建议阈值：

```text
平线检测：连续 1 s 内标准差 < 0.5 μV
大幅值检测：窗口内绝对值 > 150 μV
跳变检测：相邻样本差值 > 80 μV
```

目的：

- 早期发现电极接触问题。
- 防止坏通道污染 CAR / CSP 特征。

输出：

```text
valid_channels
bad_channels
```

### 5.4 去直流与低频漂移

输入：

```text
eeg_uv
```

处理：

第一版推荐使用高通滤波：

```text
高通截止频率：0.5 Hz 或 1.0 Hz
滤波器：Butterworth IIR，2阶或4阶
```

在线实时模式：

```text
使用 causal IIR，保留滤波器状态
```

离线训练模式：

```text
可使用 zero-phase filtfilt，避免相位延迟
```

目的：

- 去除电极极化、电位慢漂和姿态变化带来的低频趋势。
- 避免低频大幅波动影响后续带通滤波和特征标准化。

注意：

- 线上和离线滤波方式不同会造成分布差异。
- 最终比赛 demo 应尽量使用与在线一致的 causal 滤波链路。

### 5.5 工频陷波

输入：

```text
去漂移后的 EEG
```

处理：

中国大陆环境通常使用：

```text
陷波频率：50 Hz
滤波器：IIR notch
Q 值：30 左右
```

如果实际环境使用 60 Hz 电网，则切换为 60 Hz。

目的：

- 去除电源线工频干扰。
- 降低窄带噪声对 mu / beta 频带估计的影响。

注意：

- 如果后续只使用 8-30 Hz 频段，50 Hz 本身不在目标频段内，但强工频可能通过滤波器边带、放大器非理想和伪迹影响整体质量。
- 保留 `line_noise_score`，用于判断当前窗口是否适合分类。

### 5.6 运动想象频段带通

输入：

```text
陷波后的 EEG
```

处理：

第一版主带通：

```text
8-30 Hz
```

可选 filter-bank 版本：

```text
8-12 Hz    mu
12-16 Hz   low beta
16-20 Hz   beta
20-24 Hz   high beta
24-30 Hz   high beta
```

目的：

- 运动想象主要表现为感觉运动节律变化，常集中在 mu 和 beta 频段。
- 带通后能提高左右手、脚运动想象的可分性。

建议：

- 第一版先用单一 8-30 Hz，便于调试和解释。
- 第二版再引入 filter-bank，用于提高准确率和扩大光计算候选矩阵规模。

### 5.7 重参考

输入：

```text
带通后的 8通道 EEG
```

推荐两种方案：

方案 A：Common Average Reference，CAR

```text
x_car[c] = x[c] - mean(valid_channels)
```

方案 B：局部 Laplacian

```text
x_lap[C3] = x[C3] - mean(FC3, CP3, Cz)
x_lap[C4] = x[C4] - mean(FC4, CP4, Cz)
```

第一版建议：

```text
优先使用 CAR
```

目的：

- 减弱共同噪声。
- 强化局部运动皮层活动。
- 提高 CSP / bandpower 特征稳定性。

注意：

- 如果坏通道存在，CAR 只应使用有效通道。
- 8通道情况下 Laplacian 对电极布局更敏感，适合作为第二版增强。

### 5.8 滑动窗口切片

输入：

```text
连续 EEG 流
```

处理：

在线连续识别：

```text
窗口长度：1.5-3.0 s
推荐第一版：2.0 s
步长：0.25 s 或 0.5 s
```

带提示范式的训练数据：

```text
使用 cue 后 0.5-3.5 s 的时间段
```

目的：

- 运动想象不是瞬时信号，需要一定时间积累频带能量和协方差信息。
- 滑窗可以在延迟和稳定性之间折中。

建议：

- 第一版用 2.0 s 窗口、0.5 s 步长。
- 如果演示强调实时交互，可改成 1.5 s 窗口、0.25 s 步长。
- 如果演示强调准确率，可用 3.0 s 窗口。

### 5.9 窗口级伪迹检测

输入：

```text
一个 EEG 窗口
```

检测指标：

1. 幅值异常：

```text
max(abs(x)) > 150 μV
```

2. 快速跳变：

```text
max(abs(diff(x))) > 80 μV
```

3. 高频肌电污染：

```text
power(30-45 Hz) / power(8-30 Hz) 过高
```

4. 工频污染：

```text
power(48-52 Hz) / power(8-30 Hz) 过高
```

5. 加速度计异常：

```text
运动幅度突增
```

输出：

```text
reject_recommended = true / false
artifact_score
```

目的：

- 不让明显伪迹窗口强行进入分类。
- 保护在线校准，不用脏数据更新模型。

建议：

- 第一版只做拒识标记，不直接删除数据。
- 后续分类阶段根据置信度和 `artifact_score` 共同决定是否输出“不确定”。

## 6. 特征提取方案

建议准备三个层级，按项目进度逐步推进。

### 6.1 第一版：log-bandpower 特征

输入：

```text
8通道，8-30 Hz 窗口 EEG
```

处理：

计算每个通道在目标频段内的功率：

```text
p_c = mean(x_c(t)^2)
f_c = log(p_c + epsilon)
```

得到：

```text
x_float = [f_C3, f_C4, f_Cz, f_FC3, f_FC4, f_CP3, f_CP4, f_CPz]
```

输出维度：

```text
8维
```

目的：

- 实现最快。
- 可解释性强。
- 适合打通 Cyton 到 Gazelle 的完整链路。

缺点：

- 分类精度通常低于 CSP / FBCSP。
- 个体差异较大。

适用阶段：

```text
硬件链路打通、实时 demo、量化验证
```

### 6.2 第二版：CSP log-variance 特征

输入：

```text
8通道 EEG 窗口
```

离线训练：

1. 使用训练集估计每类协方差矩阵。
2. 对三分类任务采用 one-vs-rest CSP 或联合近似对角化。
3. 选取最有判别力的空间滤波器。
4. 输出 8 个 CSP log-variance 特征。

在线推理：

```text
z_k(t) = w_k^T X(t)
feature_k = log(var(z_k) / sum_j var(z_j))
```

输出维度：

```text
8维
```

目的：

- 将多通道 EEG 投影到更具判别性的空间方向。
- 强化左右运动皮层差异。
- 与 LDA / 光矩阵投影自然衔接。

建议：

- 第一版三分类可采用 one-vs-rest 训练 3 组 CSP，再从中选择总共 8 个特征。
- CSP 滤波器固定在数字端，Gazelle 负责后续候选判别投影。

### 6.3 第三版：Filter-bank CSP 特征

输入：

```text
多个频带的 EEG 窗口
```

处理：

```text
8-12 Hz
12-16 Hz
16-20 Hz
20-24 Hz
24-30 Hz
```

每个频带计算 CSP log-variance，然后用特征选择或线性压缩得到 8维。

目的：

- 适应不同受试者的个体频段差异。
- 提高跨 session 鲁棒性。
- 扩大可由 Gazelle 批量计算的候选投影规模。

建议：

- 如果比赛重点是光计算占比，FBCSP 更适合，因为它天然产生更多候选频带和矩阵乘。
- 如果比赛重点是稳定演示，先不要一上来使用太多频带。

## 7. 标准化与自适应

输入：

```text
x_float_raw
```

处理：

训练阶段保存每个特征的统计量：

```text
mu_i = mean(x_i)
sigma_i = std(x_i)
```

在线阶段：

```text
x_norm_i = (x_i - mu_i) / (sigma_i + epsilon)
```

为了增强鲁棒性，也可使用：

```text
x_norm_i = (x_i - median_i) / (IQR_i + epsilon)
```

目的：

- 消除不同通道、不同频带、不同受试者的尺度差异。
- 让后续量化范围稳定。
- 避免某一维特征主导光矩阵乘。

在线自适应建议：

```text
只在高置信、低伪迹窗口上缓慢更新均值和方差
```

更新公式：

```text
mu_new = (1 - alpha) * mu_old + alpha * x
sigma_new = (1 - alpha) * sigma_old + alpha * abs(x - mu_new)
```

建议：

```text
alpha = 0.001 到 0.01
```

注意：

- 不要用低置信或伪迹窗口更新统计量。
- 不要在比赛演示中让标准化快速漂移，否则输出会不稳定。

## 8. Gazelle 输入量化方案

Gazelle 的输入向量为 `uint8`，权重矩阵为 `int8`。因此数字预处理必须输出非负整数特征。

推荐量化方式：

```text
x_norm_clip = clip(x_norm, -3.0, 3.0)
x_uint8 = round((x_norm_clip + 3.0) / 6.0 * 255)
```

等价参数：

```text
zero_point = 128
scale = 6.0 / 255
```

反量化近似：

```text
x_approx = (x_uint8 - 128) * scale
```

目的：

- 将带正负的标准化 EEG 特征映射到 `0-255`。
- 保持 0 附近对应 `uint8` 的中间值。
- 减少异常值对量化范围的破坏。

权重量化：

```text
W_int8 = round(W_float / weight_scale)
W_int8 = clip(W_int8, -128, 127)
```

bias 处理：

```text
分类 bias 留在数字端补偿
```

原因：

- 输入零点不为 0，会给矩阵乘带来常数偏置。
- 在数字端补偿 bias 更简单，也更方便标定。

## 9. 推荐第一版完整链路

第一版应以稳定跑通为目标。

```text
OpenBCI Cyton 8通道
  -> raw counts 转 μV
  -> 坏通道检测
  -> 0.5 Hz 高通
  -> 50 Hz 陷波
  -> 8-30 Hz 带通
  -> CAR 重参考
  -> 2.0 s 滑窗，0.5 s 步长
  -> 窗口伪迹检测
  -> 8通道 log-bandpower
  -> z-score 标准化
  -> clip 到 [-3, 3]
  -> uint8 量化
  -> Gazelle 候选矩阵投影
  -> 数字端融合
  -> 左手 / 右手 / 脚 / 不确定
```

第一版特点：

- 工程实现简单。
- 每一步都容易验证。
- 能快速与 Cyton 和 Gazelle 联调。
- 准确率不一定最高，但适合作为 demo 基线。

## 10. 推荐第二版完整链路

第二版以提升准确率为目标。

```text
OpenBCI Cyton 8通道
  -> raw counts 转 μV
  -> 信号质量检测
  -> 0.5 Hz 高通
  -> 50 Hz 陷波
  -> 8-30 Hz 带通
  -> CAR 或 Laplacian
  -> 2.0 s 滑窗
  -> CSP 空间滤波
  -> 8维 CSP log-variance
  -> 鲁棒标准化
  -> uint8 量化
  -> Gazelle 批量候选投影
  -> contextual bandit 选择 / 加权
  -> 三分类结果
```

第二版特点：

- 更符合运动想象 BCI 的经典方法。
- 与三分类二维判别投影配合更自然。
- 适合正式答辩和技术报告。

## 11. 推荐第三版完整链路

第三版以竞赛表现和光计算占比为目标。

```text
OpenBCI Cyton / 公开 EEG 数据
  -> 多频带滤波
  -> FBCSP 特征
  -> 多候选特征压缩
  -> 特征标准化与 uint8 量化
  -> Gazelle 大批量 matmul
  -> 多候选二维投影
  -> 数字端融合、拒识、在线校准
  -> 三分类结果
```

第三版重点：

- 将多个频带、多个候选投影矩阵打包为大矩阵乘。
- 提高 Gazelle 光计算参与度。
- 用硬件在环实验报告量化损失和延迟。

## 12. 训练阶段与在线阶段的区别

### 12.1 离线训练阶段

输入：

```text
带标签 EEG 数据
```

输出：

```text
滤波器参数
通道映射
CSP 矩阵
标准化统计量
候选投影矩阵 W_i
分类质心 / LDA 参数
量化 scale / zero_point
```

流程：

```text
读取训练数据
  -> 与在线一致的预处理
  -> 特征提取
  -> 训练 CSP / LDA / 候选矩阵
  -> 量化仿真
  -> 保存模型包
```

### 12.2 在线推理阶段

输入：

```text
实时 EEG 流
```

输出：

```text
每 0.25-0.5 s 更新一次分类结果
```

流程：

```text
环形缓冲区
  -> 状态滤波
  -> 滑窗特征
  -> 量化
  -> Gazelle matmul
  -> 融合与拒识
```

在线阶段禁止做的事情：

- 不要重新训练 CSP 主体。
- 不要用低质量窗口更新标准化参数。
- 不要让 RL / bandit 直接大幅修改分类器权重。

在线阶段允许做的事情：

- 微调类别先验。
- 微调 reject 阈值。
- 在候选矩阵库中选择或加权。
- 缓慢更新特征均值和方差。

## 13. 与三分类输出的关系

数字预处理输出的是特征，不是最终类别。

分类流程：

```text
x_uint8
  -> Gazelle: z_i = W_i x
  -> 得到多个二维投影 z_i
  -> 数字端融合
  -> 三个类别分数
  -> softmax / 阈值
  -> 左手 / 右手 / 脚 / 不确定
```

推荐分数计算：

```text
score_k = -distance(z, centroid_k)^2 + bias_k
```

或：

```text
score = A z + b
```

最终类别：

```text
class = argmax(score_left, score_right, score_foot)
```

拒识条件：

```text
max_probability < threshold
或 artifact_score > threshold
或 top1_score - top2_score < margin
```

## 14. 实时性评估

数字预处理的主要计算量来自滤波和特征提取。对于 8通道 EEG，计算量较小，PC 或嵌入式 Linux 均可实时运行。

建议延迟预算：

```text
数据接收与解码：< 10 ms
滤波与滑窗更新：< 20 ms
特征提取：< 20 ms
量化：< 1 ms
Gazelle 调用：实测记录
数字端融合：< 5 ms
```

总延迟主要由窗口长度决定：

```text
2.0 s 窗口意味着决策依据最近 2.0 s EEG
0.5 s 步长意味着每 0.5 s 刷新一次结果
```

比赛演示建议：

- 屏幕每 0.5 s 更新一次分类结果。
- 输出结果经过 2-3 个窗口时间平滑。
- UI 展示当前置信度和是否拒识。

## 15. 验证指标

### 15.1 信号质量指标

```text
坏通道比例
窗口拒识率
工频噪声比
高频肌电污染比
饱和窗口数量
```

### 15.2 预处理有效性指标

```text
滤波前后功率谱对比
8-30 Hz 能量保留情况
50 Hz 干扰抑制幅度
特征分布稳定性
训练集和在线流分布差异
```

### 15.3 分类相关指标

```text
三分类准确率
balanced accuracy
混淆矩阵
左手 vs 右手可分性
脚 vs 手部可分性
拒识率
拒识后的有效准确率
```

### 15.4 量化与硬件相关指标

```text
浮点特征 vs uint8 特征误差
数字 matmul vs Gazelle matmul 输出误差
量化前后分类准确率差异
单窗口光计算耗时
光计算占比
```

## 16. 风险与缓解

### 风险 1：Cyton 实采信号质量不稳定

表现：

```text
电极接触差，运动伪迹大，左右手特征不可分
```

缓解：

- 先用公开数据集训练和验证算法。
- 实采阶段先做闭眼静息、睁眼静息、简单左右手任务检查频谱。
- 现场 demo 支持公开 EEG replay 模式。
- 使用拒识机制保护输出。

### 风险 2：log-bandpower 特征准确率不够

缓解：

- 第一版用 log-bandpower 打通链路。
- 第二版切换到 CSP log-variance。
- 第三版使用 FBCSP。

### 风险 3：在线标准化漂移

缓解：

- 固定训练阶段统计量作为 baseline。
- 在线更新只在高置信窗口启用。
- 保留一键恢复固定参数。

### 风险 4：uint8 量化降低分类性能

缓解：

- 使用 `[-3, 3]` 裁剪，避免异常值撑大量化范围。
- 训练阶段加入量化仿真。
- 比较浮点、int8 仿真、Gazelle 实测三种结果。

### 风险 5：预处理数字计算占比过高

缓解：

- 报告中明确预处理属于必要信号调理，Gazelle 负责核心候选线性投影。
- 使用 FBCSP 和候选矩阵库扩大光计算 matmul 规模。
- 将多个候选 `W_i` 堆叠成一次大矩阵乘，提高平台使用深度。

## 17. 推荐实现模块

建议代码结构：

```text
src/
  acquisition/
    cyton_stream.py
    packet_parser.py
  preprocessing/
    filters.py
    rereference.py
    windowing.py
    quality.py
  features/
    bandpower.py
    csp.py
    fb_csp.py
    normalization.py
    quantization.py
  photonic/
    gazelle_client.py
    matrix_pack.py
  classifier/
    fusion.py
    reject.py
    bandit.py
  demo/
    realtime_ui.py
    replay_public_dataset.py
```

核心类：

```python
class EEGPreprocessor:
    def update(raw_sample) -> None
    def ready() -> bool
    def extract_window() -> EEGWindow
    def transform(window) -> PreprocessOutput
```

核心输出：

```python
class PreprocessOutput:
    x_float: np.ndarray
    x_uint8: np.ndarray
    reject_recommended: bool
    artifact_score: float
    bad_channels: list[int]
```

## 18. 最小可行版本

一周内可完成的 MVP：

1. 读取 Cyton 或 CSV replay 数据。
2. raw counts 转 μV。
3. 0.5 Hz 高通、50 Hz 陷波、8-30 Hz 带通。
4. CAR 重参考。
5. 2 s 滑窗。
6. 8通道 log-bandpower 特征。
7. z-score 标准化。
8. uint8 量化。
9. 输出 `x_uint8` 给 Gazelle matmul。
10. 在界面显示左手 / 右手 / 脚 / 不确定。

MVP 验收标准：

```text
能够稳定接收或回放 EEG
每 0.5 s 产生一个 8维特征
特征无 NaN / Inf
uint8 范围正确
伪迹窗口能被标记
Gazelle 接口可收到输入向量
分类输出可视化刷新
```

## 19. 推荐最终方案

正式参赛建议采用第二版或第三版：

```text
CSP / FBCSP 特征
  + 鲁棒标准化
  + uint8 量化
  + Gazelle 批量候选投影
  + 数字端安全融合
  + 拒识机制
```

对外表述：

> 数字预处理负责将 OpenBCI Cyton 采集到的低信噪比 EEG 转换为稳定的运动想象特征；Gazelle 光子计算负责对多组候选判别投影进行批量矩阵乘加；数字端再完成置信度融合、拒识和在线校准，最终输出左手、右手、脚三类运动想象结果。

这个分工清晰、可实现，也符合脑电信号处理和光子矩阵乘平台的各自优势。
