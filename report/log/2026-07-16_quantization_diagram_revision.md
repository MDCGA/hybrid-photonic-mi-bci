# 量化策略双路径图修改日志

## 第 10 页：固定 4-bit 扫描与自适应逻辑精度

- 更新 `report/slides/assets/dual_path_quantization_flow.png` 与可编辑源文件 `report/slides/assets/dual_path_quantization_flow.drawio`，未修改 PPT。
- Candidate Scan 固定为单次 4-bit：输入采用 uint4，范围 `qinmin=0, qinmax=15`；权重采用 int4，范围 `qwtmin=-8, qwtmax=7`。
- 其他前向线性算子按敏感度设置起始逻辑精度：CAR 从 4-bit 起始；SOS、FBCSP 和标准化从 6-bit 起始；MLP 等敏感路径从 8-bit 起始。
- 使用 8-bit shadow 监测低位宽误差；误差超阈值时按 `4 -> 6 -> 8` 单调提升并重新执行。
- 高于原生 4-bit 的逻辑精度由 radix-16 多 slice 光计算乘加和数字端位权重构实现，突出“逻辑精度不等于单次物理位宽”。
- 满足误差阈值后，4-bit 路径直接输出；仅 6/8-bit 逻辑精度路径执行多 slice 位权重构。
- 自适应精度策略仍在完善，当前配置不表述为已经定版的最优方案。
