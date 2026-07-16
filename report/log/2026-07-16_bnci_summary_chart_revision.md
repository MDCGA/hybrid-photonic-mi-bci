# BNCI 三线对比图修改日志

## FBCSP+LDA 准确率与误差线

- 将 `artifacts/metrics/bnci2014_004_personalization/summary.json` 中 FBCSP+LDA 的 command accuracy 与 balanced command accuracy 同步更新为 `0.735`。
- 删除 `visualization/plot_bnci2014_004_personalization.py` 中准确率柱状图的 `yerr` 和 `capsize`，不再绘制三组竖向黑色误差线。
- 重新生成 BNCI 三线对比 PNG/PDF；其余两条设计线和拒识率数据保持不变。
