# 量化策略双路径流程图日志

## 第十页素材：分路径混合量化

- 仅生成独立 PNG 与可编辑 Draw.io 文件，未修改 PPT。
- 常规线性路径展开为“8-bit 逻辑精度 → radix-16 拆分 → 多个 4-bit slices 分片执行 → 数字端累加”。
- 候选 Head Bank 路径展开为“激活 uint4 + 权重 int4 → TiledMVM 单次低位宽扫描 → Top-K 数字端融合与拒识”。
- 图中保留证据边界：量化对象为前向线性计算，当前执行核为软件模拟；非线性、控制逻辑和经验库管理仍在数字端。
- 输出：`slides/assets/dual_path_quantization_flow.png`、`slides/assets/dual_path_quantization_flow.drawio`。
