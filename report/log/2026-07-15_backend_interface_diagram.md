# 统一后端接口图绘制日志

## PPT 素材：统一后端接口与执行边界框图

- 仅新增独立图片与可编辑 Draw.io 素材，未修改 PPT 文件。
- 以 MatrixOps（矩阵/张量）、SignalOps（CAR/SOS）和 TiledMVM（候选 Bank）三类接口为中心，展示上层算法通过稳定接口与具体执行核解耦。
- 执行边界明确区分：全部前向线性算子由统一 Backend 接管，当前执行核为软件模拟；非线性、控制逻辑、拒识、经验库管理与流程调度仍保留在数字端。
- 输出文件：`slides/assets/backend_interface_architecture.png`、`slides/assets/backend_interface_architecture.drawio`。
- 可重复生成脚本：`slides/tools/generate_backend_interface_diagram.py`。
