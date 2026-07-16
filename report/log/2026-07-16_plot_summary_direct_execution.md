# plot_summary 直接执行兼容日志

## 修复相对导入错误

- `visualization/fbcsp_design/plot_summary.py` 原本仅作为包内模块由总生成入口调用，直接执行时 `.common` 缺少包上下文并触发 `ImportError`。
- 增加直接执行时的项目根目录解析和绝对包导入；作为包导入时仍沿用原有相对导入，不改变总生成入口行为。
- 增加命令行入口，支持 `--metrics-dir`、`--output-dir` 和 `--formats`，默认生成 PNG/PDF summary 图。
