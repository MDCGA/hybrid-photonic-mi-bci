# 实验数据划分图日志

## 第 11 页素材：BCICIV 实验数据划分

- 仅生成独立 PNG 与可编辑 Draw.io 文件，未修改 PPT。
- 主图展示三段严格隔离的数据用途：Train 840 用于训练 FBCSP、embedding 与候选 Heads；Calibration 42 仅用于候选评分和 Top-K 会话组合；Evaluation 518 仅用于最终指标。
- 增加“评估集不参与模型训练”和“评估集不参与 Top-K 选择”的数据泄漏防护提示，并标明各方案使用相同评估窗口。
- 明确 Reject 是系统输出而不是数据集的第四类标签。
- 输出：`slides/assets/bciciv_experiment_split.png`、`slides/assets/bciciv_experiment_split.drawio`。
