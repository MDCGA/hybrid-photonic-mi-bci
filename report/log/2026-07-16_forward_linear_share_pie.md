# 前向线性计算模块占比图日志

## 主线 Inference MAC 饼图

- 从 `artifacts/metrics/fbcsp_design/compute_accounting.json` 读取主线 `FBCSP + MLP embedding + library + photonic scan` 的 inference 阶段账本，未手工填写计算量。
- 总前向线性计算量为 342,105,008 MAC；模块包括 SOS 带通滤波、FBCSP 空间投影、MLP 前向线性层、Candidate Scan 和经验融合。
- 饼图保留所有模块的真实比例；对占比较小、无法在扇区内标注的 Candidate Scan 和经验融合，在右侧明细区单独列出准确数值与比例。
- 图中明确该占比是 forward-only linear MAC 构成，不代表运行时间、功耗或真实光芯片资源占比；当前执行核为软件模拟。
- 输出 `report/slides/assets/forward_linear_compute_share.png`、矢量版本 `.svg` 和可编辑数据 `.csv`；未修改 PPT。
