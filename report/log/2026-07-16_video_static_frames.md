# 演示视频静态画面绘制日志

## 镜头 1：标题与范围

绘制项目标题、软件原型验证范围、四项演示内容和“非 Gazelle 实机 / 非 DeepBCI 实采”边界，生成 `01_title_scope.png/.drawio`。

## 镜头 2：数据来源与评估协议

绘制 7 个文件、Train 840、Replay 560、Calibration 42 和 Evaluation 518 的横向划分，强调全部拟合只使用训练段，生成 `02_dataset_protocol.png/.drawio`。

## 镜头 4：FBCSP 特征层

绘制 EEG、CAR/质量门控、6 子带、OVR CSP、72D 对数方差和 Fisher 32D 流程，生成 `03_feature_pipeline.png/.drawio`。

## 镜头 5：MLP Embedding 与经验库

绘制 32D FBCSP、32→64 MLP、32D embedding、42 窗口查询向量和 66 候选/Top-K=8 经验库链路，生成 `04_embedding_library.png/.drawio`。

## 镜头 7：候选头与 Tiled MVM

绘制候选线性公式、2×8 tile、8 candidates × 8 tiles 和 64 logical tile evaluations/window，明确软件仿真边界，生成 `05_candidate_tile_scan.png/.drawio`。

## 镜头 8：Replay 在线决策

绘制 embedding、候选得分、softmax、融合、三重门控和 left/right/foot/reject 输出链路，并解释拒识动机，生成 `06_online_decision.png/.drawio`。

## 镜头 11：结果与复现性

按视频脚本展示 518 windows、75.10%、72.88%、78.27%、4.05%、64 tiles/window 和 16 passed 证据，生成 `07_results_reproducibility.png/.drawio`。

## 镜头 12：结论与下一步

以“已完成 / 尚未完成”双栏总结软件链路、量化、Gazelle 硬件在环、Cyton/DeepBCI 与真实闭环计划，生成 `08_closing_next_steps.png/.drawio`。

全部画面为 16:9、1920×1080，沿用现有 PPT 的蓝/绿/橙/紫配色和深色标题体系；左下角统一保留软件仿真状态标签。未修改 PPT。
