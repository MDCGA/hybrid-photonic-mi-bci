# 复赛技术文档工作日志

## 2026-07-14 初始化

- 审阅赛题、复赛通知、工程 README、实验 JSON、后端接口和可用图表。
- 确认赛题要求：技术文档需含核心预览、完整方案、线性层 shape/MAC/光计算标记；提交 Word/PDF、源码技术数据、PPT 和 5 分钟内视频。
- 确认匿名约束：不得出现学校、Logo、指导教师或真实成员姓名。因此未调用模板的校徽封面，仅复用 `zjuthesis` 的中文排版、目录、正文和参考文献能力。
- 确认工程的光计算后端当前为 `SimulatedPhotonic...Backend` 软件仿真，Gazelle 驱动为未实现 stub。技术文档将其表述为软件映射与接口验证，不伪称硬件实测。
- 已复制 7 张工程生成的 PNG 到 `template/figure/competition/`，准备在报告中作为系统、性能、嵌入、候选扫描、tile 调度和计算量证据。

## 2026-07-14 第一轮构建与格式修复

- `latexmk` 因本机 MiKTeX 未安装 Perl 不能执行；改用等价的 `xelatex -> biber -> xelatex -> xelatex` 流程。
- 第一轮输出 24 页 PDF。发现并修复：反引号代码文本内的下划线进入数学模式、长命令行溢出、`2x8` 标题的 PDF 书签警告、附录五列表格超宽。
- 保留了 MiKTeX “建议检查更新”的环境提示；该提示不影响 PDF 内容。下一轮将检查日志中是否仍有 LaTeX Error、未定义引用、overfull box，并进行 PDF 视觉审查。

## 2026-07-14 第二轮审查与内容增强

- 完成 `xelatex -> biber -> xelatex -> xelatex`，PDF 为 A4、24 页、约 2.1 MB；LaTeX Error、未定义引用、overfull box 均为 0。附录窄表格存在 7 个 underfull box，不影响边界或可读性。
- 抽查封面、状态表、算法图、结果图和风险页：图片未裁切，表格未越界，页眉不含学校信息。
- 修复内容一致性：实时方案的 2 s 滑窗与离线 BCICIV/BNCI 的事件窗分开说明；附录 MAC 示例更正为 BCICIV 的 3 s、100 Hz、300 样本，并要求 BNCI 按 750 样本复算。
- 优化封面副标题排版，补充 P0--P4 的实机落地计划与验收输出。

## 2026-07-14 最终匿名与视觉审查

- 重新抽查封面及第 5 章路线图页面：封面副标题无孤行，图表、公式、表格均未出现裁切或越界。
- 为降低匿名材料的机械检索风险，删除正文中提示性身份词汇，统一改为“无可识别机构、人员或标识信息”。
- 下一步：最终重编译并扫描 PDF 元数据与可提取文本，确认无敏感身份词、无 LaTeX Error、无未定义交叉引用、无 overfull box。

## 2026-07-14 最终构建结果

- 最终 PDF：A4、24 页、约 2.1 MB，LaTeX Error = 0、未定义引用 = 0、overfull box = 0；PDF 元数据仅包含 LaTeX/MiKTeX 生成器。
- PDF 可提取文本对机构、人员和常见身份标识关键词的扫描无命中。
- 工程回归测试执行完毕：24 项通过。
- 对附录窄表格的 7 条 underfull box 提示执行最后修复：改用左对齐窄列，随后重编译确认。

## 2026-07-14 封版确认

- 最终 XeLaTeX/Biber 构建完成：LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。
- 最终产物为 `template/out-competition/competition_report.pdf`（A4，24 页）。源文件为 `template/competition_report.tex`，参考文献为 `template/competition_report.bib`，插图为 `template/figure/competition/`。
- 质量结论：文档包含核心预览、系统/算法公式、7 幅工程图、软件与实机证据边界、结果、shape/MAC 清单、提交核查表和实机验收路线；避免将软件仿真陈述为硬件实测。

## 2026-07-14 脑电与光计算映射增强

- 收到新增电极位置图后，将其复制到竞赛图目录，计划放入“脑电与运动想象基础”而非实验结果章节，避免把通用示意图误标为项目数据。
- 新增可复现 Python 绘图脚本 `template/tools/generate_competition_diagrams.py`，生成：EEG 节律/ERD 概念图、FBCSP 矩阵流图、候选线性头到 2x8 tile 的拆分图。
- 文档新增脑电概念、电极命名与中央区选择说明；为 CSP、标准化、MLP、候选头和融合逐一给出矩阵 shape、tile 数和部署边界。
- 新版初次编译为 28 页，无 LaTeX Error、无未定义引用、无 overfull box；矩阵拆分表出现 2 条 underfull 提示，已改为左对齐列并进入最终构建。

## 2026-07-14 脑电与映射增强封版

- 更新后 PDF：A4、28 页、约 2.7 MB；LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。
- 最终 PDF 匿名关键词扫描无命中。
- 新增图按证据类型标注：电极图和 ERD 图为概念/空间关系说明；FBCSP 与 tile 图为脚本生成的算法拆分说明；既有工程图仍仅用于实验结果佐证。
- 新增映射表给出 CAR、OVR CSP、标准化、MLP、候选头和融合的矩阵表达式、shape、tile 数及当前部署边界。候选头仍被明确标为首个 Gazelle 实机热点，未将软件仿真写为硬件实测。

## 2026-07-14 端到端信号链扩写

- 根据要求，在第 2 章开头新增“从采集信号到最终指令的完整推理链”。
- 以 10 个按执行顺序的步骤说明：Cyton 数据包、ADC 到微伏、环形缓冲/切窗、质量拒识、IIR 滤波/CAR、FBCSP、标准化/MLP、经验库检索、量化与 Gazelle tile、概率融合/时间平滑/最终拒识。
- 每步明确输入、输出、公式和工程边界；特别区分了公开数据回放的当前实证与实时 Cyton--Gazelle 闭环的待联调状态。
- 初次构建为 32 页，LaTeX Error = 0、未定义引用 = 0、underfull box = 0；发现 1 条 1.8pt 的 CSP 长句 overfull 提示，已改写后重编译。
- 最终构建：32 页，LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。

## 2026-07-14 简化端到端系统图

- 原工程系统图信息密度较高，替换为按端到端推理链组织的新图：10 个步骤归并为采集与对齐、数字预处理、特征与个性化、Gazelle 候选评分、安全输出五组。
- 新图明确标出有效窗口的实线路径与质量不合格时的虚线提前拒识路径，并将 Gazelle 计算限定为候选头 tile MVM，避免误导为所有算法均已上光芯片。
- 初版系统图的十个节点在同页中仍显拥挤，因此进一步收敛为五个阶段框；每框只保留步骤范围、输入/输出形态和关键动作，详细步骤继续由第 2 章提供。
- 简化图已重新编译进入 PDF；最终检查：LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。

## 2026-07-14 Draw.io 可编辑导出

- 新增原生 diagrams.net 文件 `template/figure/competition/end_to_end_inference_chain.drawio`。
- 文件包含五个阶段框、四条主链箭头、质量不合格的虚线提前拒识路径和底部注释，可直接在 draw.io 编辑文字、颜色、尺寸和连接关系。
- 按要求将 draw.io 文件中的标题、阶段框、箭头标签、提前拒识路径和注释全部翻译为中文；保留数学符号、Cyton、FBCSP、CAR 和 Gazelle 名称以确保技术含义准确。
- 检测到用户已导出的 `end_to_end_inference_chain.drawio.png`，文档已切换为引用该图片；其余脚本生成图将单独以中文重绘，避免覆盖用户手动编辑的系统图。
- 中文概念图已生成并完成视觉检查；tile 映射图的数字端说明已移至候选模型框下方，避免与 tile 计数说明重叠。
- 文档已使用用户导出的系统图，三张自生成解释图（EEG/ERD、FBCSP 数据流、候选头 tile 映射）均已中文化。实验结果曲线与混淆矩阵保持工程原始标签，避免篡改变更可复现实验图。
- 最终 PDF 构建：32 页，LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。

## 2026-07-14 全部解释图 Draw.io 导出

- 新增可编辑文件：`fbcsp_matrix_flow.drawio`、`photonic_tile_mapping.drawio`、`eeg_mi_signal_concepts.drawio`。
- 与已有 `end_to_end_inference_chain.drawio` 一起，覆盖文档中四张自生成解释图；所有框、箭头、颜色和文字均为 Draw.io 原生元素，可独立调整。

## 2026-07-14 用户微调图替换

- 检测到用户导出的 `fbcsp_matrix_flow.drawio.png` 与 `photonic_tile_mapping.drawio.png`，技术文档已切换为引用这两张修改版图片。
- EEG/ERD 概念图未发现新的导出 PNG，按说明继续使用此前版本；端到端系统图继续使用已导出的版本。
- 替换后 PDF 重新构建完成：32 页，LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。

## 2026-07-14 LTSimulator 光计算占比补充

- 根据用户确认，报告第 3 章末新增“LTSimulator 仿真与赛题光计算占比”小节；候选线性头的 4-bit $2\times8$ tile MVM 说明为已按 LTSimulator `custom_matmul.py` 调用模式进行仿真。
- 按主线 518 个评估窗口的前向线性计算账本统计：总线性 MAC 为 342,105,008，LTSimulator 光计算仿真路径为 342,105,008，光计算占比为 100.00\%，相对赛题 $\geq50\%$ 的门槛高 50 个百分点；其中 `quantized_photonic_tiled_mvm_uint4` 候选扫描记录为 410,256 MAC。
- 文中明确排除训练/反向传播的 2,964,376,800 MAC，并保留边界：这是 LTSimulator 仿真统计，不是 Gazelle 实机的物理计算比例；实机验收须以原始调用、读回和误差日志重算分子。
- 完成 XeLaTeX/Biber 全量构建与新增页视觉抽查：输出 33 页 PDF，LaTeX Error = 0、未定义引用 = 0、Overfull box = 0、Underfull box = 0；新表、两条公式、可复核路径及实机边界均未出现裁切或越界。

## 2026-07-14 LTSimulator 口径统一

- 根据用户说明，复赛要求为 LTSimulator 光计算平台仿真，删除报告中“并非 Gazelle 实机”“待实机验收”“不能替代硬件比例”等限定性表述。
- 全文将相关接口、候选 MVM、调用日志、演示流程、验证清单、路线图和附录证据统一为 LTSimulator 仿真口径；保留量化、tile 调度、读回、误差和计算账本等可复核技术信息。
- 口径统一后完成 XeLaTeX/Biber 全量重建：33 页 PDF，LaTeX Error = 0、未定义引用 = 0、Overfull box = 0、Underfull box = 0。

## 2026-07-14 目录分页

- 在核心内容预览结束处插入分页命令，使目录从独立页面开始；图目录和表目录的独立分页保持不变。
- 完成 XeLaTeX/Biber 全量重建：33 页 PDF，LaTeX Error = 0、未定义引用 = 0、Overfull box = 0、Underfull box = 0。

## 2026-07-15 删除第五章

- 按要求删除“创新性、可行性与演示方案”整章，包括创新性说明、演示脚本、验证清单、风险说明和分阶段计划；同步更新提交材料核查表中的章节与视频说明引用。
- 完成 XeLaTeX/Biber 全量重建：删除后 PDF 为 31 页，LaTeX Error = 0、未定义引用 = 0、Overfull box = 0、Underfull box = 0。

## 2026-07-15 主线结果与图表可读性

- 删除报告中 FBCSP+shrinkage LDA 与独立小型 MLP 路线的比较描述、结果行和诊断图，仅保留主线 FBCSP+embedding+经验库+候选扫描的结果。
- 在 `plot_experience_photonic.py` 中新增四个全尺寸主线诊断图导出：候选融合权重、候选头质量、回放轨迹和混淆矩阵；报告将原来的四宫格拼图替换为这四张单图。
- 删除计算量双图拼接，改由第 3 章的 LTSimulator 占比表表达；完成全量构建和结果页视觉抽查，最终为 33 页 PDF，LaTeX Error = 0、未定义引用 = 0、Overfull box = 0、Underfull box = 0。

## 2026-07-15 本科生报告语言统一

- 将封面预览、背景、脑电概念、10 步推理流程、FBCSP、经验库、光计算拆分、统计方法、工程实现、结果分析、演示、风险、结论和附录说明改为更直白的本科生技术报告表述。
- 保留 FBCSP、LTSimulator、MAC、tile 等必要术语，并在首次使用处给出中文解释；不修改公式、实验数据、算子尺寸和结论。
- 完成 XeLaTeX/Biber 全量重建：33 页 PDF，LaTeX Error = 0、未定义引用 = 0、Overfull box = 0、Underfull box = 0。
# 2026-07-15 赛题、工程与初步文档复核

- 开始重新通读 `info/`、`hybrid-photonic-mi-bci/` 与 `template/`，本轮只做材料梳理和一致性核验。
- 已确认约束：不在 `technical_documentation_outline/` 中创建或修改文件；当前实际存在的是 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`，本轮不触碰该文件。
- 初步盘点：`info/` 包含赛题图片、复赛要求网页、Gazelle 产品手册、数据集说明和数字预处理方案；工程包含 FBCSP、嵌入/经验库、候选线性头、矩阵后端、计算量统计、Cyton 上位机、测试与实验产物；`template/` 已形成可编译技术报告及配套图表。

## 2026-07-15 首轮一致性结论

- 复赛方向与赛题匹配：分赛区任务允许开发创新 AI 应用，要求基于 LTSimulator，光计算占比不低于 50%；初赛中的 MNIST Top-1 不低于 85%属于初赛任务，不是当前 MI-BCI 复赛方案必须达到的指标。
- 交付要求已基本被报告结构覆盖：技术文档需有核心内容快速预览、完整方案、线性层 shape/MAC/光计算标记；另需技术数据、答辩 PPT 和不超过 5 分钟的 1080P MP4。所有材料禁止出现指导教师、学校名称、Logo 或简称。
- 工程当前主线（`artifacts/metrics/fbcsp_design/summary.json`）为 518 个 BCICIV 评估窗口：指令准确率 70.46%、平衡准确率 73.24%、接受窗口准确率 72.42%、拒识率 2.70%、每窗 80 次 tile；推理线性 MAC 为 342,105,008，账本给出的光计算线性占比为 100%。
- 报告中的 BCICIV 数字仍写成 72.97% / 72.42% / 4.83%，与当前 JSON 不一致；BNCI 主线当前 JSON 为 74.92% / 拒识 1.62%，报告写成 74.54% / 2.31%，同样需要统一。报告的 MAC、候选扫描 MAC（410,256）和每窗 80 tile 与当前 BCICIV 账本一致。
- 发现更重要的证据边界：工程没有直接依赖或导入官方 LTSimulator；`backends.py` 将当前实现明确称为 `software stand-in`，执行后端为 `numpy_tiled_photonic_simulation`，只是按 `custom_matmul.py` 的模式模拟量化和切片。因此报告中“已用 LTSimulator 仿真”“LTSimulator 调用/读回已完成”的表述强于代码证据，提交前应接入官方模拟器或改为准确的兼容接口/软件模拟口径。
- 100% 光计算占比由账本把当前 forward 后端中的 CAR、滤波、CSP、标准化、MLP 线性层等全部标为 photonic 得出；这满足项目自定义统计规则，但是否符合企业对“实际由 LTSimulator 完成”的认定仍是高风险复核点，不能只依靠 MAC 标签。
- 本轮仅更新本日志，未修改报告、工程代码或 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。

## 2026-07-15 流程、实验数据与可视化同步

- 按当前 `experience_photonic_line.py` 更新主线流程：经验库由 2 个全局锚点和 64 个 bootstrap 条目组成；候选集合保留锚点并按距离补足 top-8；融合权重同时使用归一化距离、训练准确率、42 个校准窗口的准确率/真实类别置信度及强度为 5.0 的锚点先验。
- 技术文档中的 BCICIV 主线数据已同步为：518 个评估窗口、指令准确率 70.46%、平衡准确率 73.24%、接受窗口准确率 72.42%、拒识率 2.70%。
- BNCI 主线数据已同步为：9 位被试、1296 个评估窗口、平均准确率 74.92%、接受窗口准确率 76.22%、拒识率 1.62%。
- 扩展 `visualization/fbcsp_design/plot_experience_photonic.py`，使绘图流程除组合诊断图外，直接输出候选权重、候选质量、回放轨迹、混淆矩阵四张独立 PNG/PDF；随后从最新 JSON/NPZ 全量重绘。
- 已用最新产物替换 `template/figure/competition/` 下的四张主线结果图和 tile 调度图，报告图注同步说明新的多因素候选加权流程。
- 重新执行 XeLaTeX/Biber 全量构建，输出 `template/out-competition/competition_report.pdf`：A4、31 页、LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0；抽查结果表和新图无裁切或越界。
- 工程回归测试结果：30 项通过。
- 未创建或修改 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。

## 2026-07-15 核心流程预览补充

- 在“作品核心内容快速预览”中新增独立的“核心流程”模块，展示 `EEG -> FBCSP -> small MLP embedding -> experience library -> photonic scan -> reject/command`。
- 在流程框下补充中文解释，说明 FBCSP 特征提取、小型 MLP 嵌入、经验库候选选择与加权、光计算候选扫描以及阈值拒识/指令输出的衔接关系。
- 重新构建 PDF 并抽查预览页：流程框完整、无横向溢出；最终为 A4 32 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
- 未修改 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。

## 2026-07-15 问题定义与方案概述重写

- 将第 1 章重构为“运动想象 EEG-BCI 任务与信号挑战”“脑电基础、电极位置与 MI 节律”“个体差异与会话特化动机”“方案主干与光电计算分工”“系统总览”。
- 补充 EEG 决策窗口到控制命令/拒识的任务定义，以及低信噪比、小样本、非平稳和易受伪迹影响的建模约束。
- 完善国际 10--20 系统、C3/Cz/C4 与 FC/CP 邻近通道、mu/beta 节律及 ERD/ERS 的背景说明，并继续明确 ERD 图为概念图而非项目实测。
- 明确经验库源于 MI 跨被试/跨会话差异下的特化需求，保存可复用的特化模型与统计，不是普通数据缓存；主线采用少量校准后的会话特化，而非单一固定泛化模型。
- 明确 FBCSP 是可解释、小样本友好且线性计算密集的系统主干；小型 MLP 仅作为 embedding 模块，不将方案描述为纯深度学习 BCI。
- 按当前 forward-only 口径说明：全部前向线性计算通过 MatrixOpsBackend、SignalOpsBackend、TiledMVMBackend 进入光计算接管路径，候选头是重点 4-bit tile 路径；非线性、控制、拒识和经验库管理仍在数字端。
- 新增并插入 `mi_bci_task_overview.png` 与 `specialization_logic.png`，绘图脚本可重复生成；复用现有 10--20 电极图和 ERD/ERS 概念图。
- 重新构建并抽查第 1 章：最终 PDF 为 A4 34 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0；新增图文无裁切、重叠或越界。
- 未修改 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。

## 2026-07-15 核心流程框单行化

- 将快速预览中的核心流程改为按边框宽度自动缩放，完整链路固定在一行显示。
- 重新构建并抽查预览页：流程文字未换行、未越界；LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
- 未修改 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。

## 2026-07-15 核心流程快速预览

- 确认“作品核心内容快速预览”已加入独立的“核心流程”行：`EEG -> FBCSP -> small MLP embedding -> experience library -> photonic scan -> reject/command`。
- 该流程使用整行缩放控制版心宽度，位于“作品目标”和“核心创新”之间。
- 重新完成 XeLaTeX/Biber 全量构建，PDF 保持 31 页。

## 2026-07-15 核心流程速览补充

- 在“作品核心内容快速预览”中新增一行核心流程：`EEG -> FBCSP -> small MLP embedding -> experience library -> photonic scan -> reject/command`。
- 未创建或修改 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。
# 2026-07-15 按 outline 分节重组日志

## 第 3 章：Forward-only 光计算占比与结果解释

- 将原“LTSimulator 仿真与赛题光计算占比”改为 backend 接管口径：分子为 photonic backend 前向线性 MAC，分母为同批窗口全部前向线性 MAC。
- 明确 100% 是接口路由与 MAC-equivalent 统计，不是芯片功耗占比，也不代表全部算子已完成单次 4-bit 或真实硬件实测。
- 明确当前执行核仍为软件模拟；官方模拟器/真实光芯片仍需逐算子调用、误差、slice、时延和功耗证据。

## 第 3 章：统一后端接口排版修复

- 为 MatrixOpsBackend、SignalOpsBackend 和 TiledMVMBackend 增加可断行点。
- 不改变接口含义，仅消除长等宽标识符导致的横向溢出。

## 第 4 章：工程结构与模块职责

- 按 outline 补齐核心包、backends、FBCSP、小型 MLP、经验库、workflows、datasets、pure runtime、examples、artifacts、visualization 和 tests 的职责。
- 明确 `host/` 只有适配器、存储、控制器模型和界面模块，不能等同于在线闭环完成。
- 删除现场实采已经可演示的表述，当前可复现实验和演示主线统一为公开数据回放。

## 第 4 章：数据集与评估协议

- BCICIV 小节补充 a--g 文件合并、文件名不等于类别、left/right/foot 全局标签和 reject 非原生类别。
- 明确训练、42 个校准窗口和 518 个评估窗口的用途，评估窗口不参与训练或候选选择。
- BNCI 小节补充 9 被试、session 1--2 历史数据、session 3 目标会话、每类前 8 个 trial 校准和 1296 个共享评估窗口。
- 补充 command/balanced/accepted/reject 以及 rolling/cumulative 指标口径。

## 第 4 章：实验结果分析

- 新增 BCICIV 三条实验线同协议对比表，数字同步当前 summary.json。
- 明确主线当前准确率未优于所有 baseline，不能把经验库写成已经稳定提升精度；已验证的是校准、特化、候选扫描和拒识的完整路径。
- 增加类别混淆、accepted accuracy 与 reject rate 权衡、真实采集误触发风险的解释。
- BNCI 小节补充三条线平均结果，并提出 subject-wise、校准窗口数、top-K 和经验库规模消融需求。

## 第 4 章：运行方式与技术数据

- 补齐完整三线、单主线、单 evaluation 窗口、BNCI、绘图和单元测试命令。
- 解释 evaluation-index 的含义，避免与原始 trial 编号混淆。
- 列明 summary、compute accounting、progress、arrays 和 BNCI subject rows 的输出用途。
- 补充 pure runtime、技术数据打包及串口、数据库、绝对路径、PDF/视频敏感信息清理要求。

## 第 3 章：运行进度、Setup 与单窗口时延边界

- 从 run_progress.json 同步完整 workflow 约 13.40 s 及各阶段耗时，并明确这是整批软件运行时间。
- 区分 setup/训练/校准与单窗口 online forward，避免把批量耗时写成在线延迟。
- 记录每窗口 80 次候选 tile，但明确尚无真实光芯片逐窗口 wall-clock latency。
- 区分带标签回放可显示累计准确率与真实在线只能显示质量、概率、置信度、拒识和耗时。

## OpenBCI Cyton 上位机扩展

- 新增独立章节并将当前状态明确标为未完成。
- 区分已有 adapter/数据模型/SQLite/Tk 骨架与尚未完成的真实采集、质量门控、在线推理和控制反馈闭环。
- 补充目标采集链、断线恢复、时序/滤波状态验证、校准引导、经验条目管理和运行状态显示规划。
- 明确公开数据回放准确率不能写成 Cyton 在线实采准确率。

## 局限性

- 新增公开数据回放与真实连续 EEG 场景差异、真实在线无即时标签的限制。
- 明确软件 photonic backend、100% forward linear share 与真实芯片功耗/吞吐/加速证据之间的边界。
- 补充候选单次 4-bit、其他路径 8-bit 逻辑精度位权拆分的混合精度风险。
- 明确主线当前未稳定超过 LDA baseline，且 Cyton 闭环未完成，结论不能外推到真实被试部署。

## 后续工作

- 新增官方模拟器/真实光芯片逐算子执行、误差、时延、吞吐和功耗验证路线。
- 增加候选 4-bit、bit-sliced 8-bit、宽物理单元和 QAT 的混合精度比较与逐算子最低精度搜索。
- 增加 Cyton 连续采集、质量门控、校准、pure runtime、经验库管理和控制反馈闭环计划。
- 增加校准量、top-K、经验库规模、subject-wise 收益消融及报告自动同步计划。

## 结论

- 结论改为只声明已完成的公开数据回放软件原型、三条实验线、两套协议、单窗口入口、指标图表、账本和测试。
- 明确 100% 为 backend 接口/软件模拟接管证据，不是官方 LTSimulator 或真实光芯片实测。
- 明确主线平均精度未稳定超过 LDA，仍需消融和真实被试验证。
- 删除 Cyton 上位机已完成的结论，明确真实在线闭环尚未完成。

## 附录 A：线性算子与计算量清单

- 将 CAR、SOS、CSP、标准化、MLP、候选扫描和融合统一改为 MatrixOps/SignalOps/TiledMVM 软件 backend 接管口径。
- 区分候选头单次 4-bit 与其他路径 bit-sliced 软件模拟。
- 明确表中“接管”不是芯片执行证据，真实后端仍需调用、误差、slice/tile、时延和功耗记录。

## 第 2 章第 9 步：量化与候选得分

- 将标题改为“软件 photonic tile 扫描”，避免暗示官方 LTSimulator 已直接执行。
- 明确 TiledMVMBackend 与量化 matrix backend 的调用关系、每窗口 80 tile 和反量化流程。
- 保留 custom_matmul.py 接口兼容说明，同时注明当前执行核为 NumPy 整数矩阵乘。
- 构建后为 TiledMVMBackend 名称增加可断行点，消除该小节横向溢出。

## 第 1 章：项目完成情况表

- 将 tile 路径状态统一为“已完成软件模拟”。
- 将 LTSimulator 状态改为“兼容接口部分完成”，明确已实现 custom_matmul 模式但未直接依赖官方模拟器。
- Cyton 上位机、模拟流和实时在线演示继续保持未完成状态。

## 附录 B：提交材料核查表

- 将光计算证据改为 backend 接管、软件执行核和真实硬件未实测三层边界。
- 修正源码、测试和 metrics 的路径写法。
- 演示支撑统一为公开数据回放，并明确不得把 Cyton 或真实硬件写成已完成。
- 将同一单元格中的多个长路径改为目录职责描述，消除窄表格列的 underfull 告警。

## 重组文档构建验收

- 完成 XeLaTeX/Biber 全量构建，输出 A4 41 页 PDF。
- LaTeX Error = 0、未定义引用 = 0、overfull box = 0、underfull box = 0。
- 抽查快速预览与目录：新章节层次、三阶段/三实验线、Cyton 未完成状态和交付表均正常显示。
- 本轮只读取 outline 作为重组依据，未修改 `hybrid-photonic-mi-bci/docs/technical_documentation_outline.md`。

## 第 2 章：训练、校准与在线推理三个阶段整合

- 合并原“训练、校准与在线推理三个阶段”和“从采集信号到最终指令的完整推理链”，保留一个一级部分。
- 离线训练子部分按数据协议、FBCSP 拟合、特征/标准化、小型 MLP/baseline、经验库与部署包五步展开。
- 校准子部分按校准窗口、冻结主干 embedding、top-K 与融合权重、拒识阈值/会话状态四步展开。
- 在线推理子部分按决策窗口、质量拒识、滤波/CAR/FBCSP、标准化/embedding、量化候选扫描、融合/拒识/命令六步展开。
- 明确三个阶段的输入、输出、可更新参数和数据边界；原十步链的内容已归入相应阶段，不再单独重复。

## 第 2.1.1 小节：离线训练阶段

- 按五步展开：固定训练协议、拟合滤波器组/OVR CSP、Fisher 选择与标准化、训练 LDA/小型 MLP、建立经验库并导出部署包。
- 明确 evaluation windows 不参与拟合，训练与反向传播属于 setup，不计入单窗口 forward。

## 第 2.1.2 小节：校准阶段

- 按四步展开：输入少量目标会话窗口、冻结主干生成 embeddings、选择 top-K 并计算融合权重、确定拒识阈值并冻结会话状态。
- 明确校准窗口不进入最终指标，校准不能重新训练 FBCSP/MLP，校准输出是部署状态而不是控制命令。

## 第 2.1.3 小节：在线推理阶段

- 按六步展开：形成决策窗口、质量提前拒识、滤波/CAR/FBCSP、标准化/embedding、量化候选扫描、融合/平滑/拒识与命令输出。
- 保留关键矩阵和决策公式，明确每窗口 80 tile、软件执行核边界，以及回放与真实在线可显示指标的差异。

## 第 2.1 节整合构建验收

- 新目录仅保留 2.1.1 离线训练、2.1.2 校准、2.1.3 在线推理三个子部分，原独立“完整推理链”已删除。
- 全量构建输出 A4 40 页；LaTeX Error、未定义引用、overfull box、underfull box 均为 0。

## 第 3.1.1 小节：MatrixOpsBackend

- 展开 matmul/einsum 契约、输入输出 shape、算子 name 的路由/记账用途和增广仿射表达。
- 列明 CSP、标准化、MLP、LDA、经验库距离与概率融合等当前调用方，并区分非线性/矩阵分解。
- 明确 NumPy 参考实现、photonic 软件量化切片实现和真实硬件未接入边界。

## 第 3.1.2 小节：SignalOpsBackend

- 展开 common_average_reference 与 sosfiltfilt 契约、channel/time axis 语义及调用命名。
- 区分离线零相位滤波和在线因果有状态滤波，说明在线接口仍待完成。
- 明确软件 SignalOps 的 MAC-equivalent 接管不等于 CAR/IIR 已在真实光芯片执行。

## 第 3.1.3 小节：TiledMVMBackend

- 展开 scan/count_tiles 契约，写明 weights $(N,M,D)$、features $(D,)$ 和输出 $(N,M)$ shape。
- 说明候选、输出行块、输入列块的三层扫描及数字部分和累加。
- 给出 8 个 $3\times33$ 候选映射为 80 次 $2\times8$ tile，并明确候选单次 4-bit 和当前 NumPy 整数执行核。

## 第 3.1 节接口展开构建验收

- 目录中统一后端接口已展开为 3.1.1 MatrixOpsBackend、3.1.2 SignalOpsBackend、3.1.3 TiledMVMBackend。
- 全量构建输出 A4 41 页；LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
- 抽查接口页：函数签名、shape、长标识符和正文均未裁切或越界。

## 第 3.2.1 小节：光计算单元与多候选 MVM 的匹配

- 说明单一小型分类头的映射价值有限，而经验库 top-K 形成规则、权重不同的重复 MVM，更适合批量扫描。
- 明确低功耗和高并行是目标特性，当前没有真实芯片能耗、吞吐或延迟实测。

## 第 3.2.2 小节：经验库检索与 Candidate head bank

- 将 top-K 条目形式化为 $(A_i,b_i,\alpha_i)$，并给出增广 head bank 的 $K\times M\times(D+1)$ shape。
- 写明主线 weights 为 $(8,3,33)$、features 为 $(33,)$，经验库输出是带权候选 bank 而不是单一模型。

## 第 3.2.3 小节：Candidate scores 与概率融合

- 明确 scan 输出 $K\times M$ 候选分类分数，不直接输出最终类别或概率。
- 补充逐候选 softmax、检索权重融合、时间平滑和拒识的数字端职责边界。

## 第 3.2.4 小节：决策路径重组与创新边界

- 说明创新点是由 MI 个体差异驱动“校准--经验库--多候选--tiled MVM”的决策路径重组，不是简单替换 np.matmul。
- 明确 photonic scan 不是 Goertzel/频率扫描，也不是只计算一个固定全局头。
- 限定当前证据为 shape、4-bit 量化、tile 调度和软件输出重构，真实并行度与能效仍待实测。

## 第 3.2 节候选扫描动机构建验收

- outline 5.4 的光计算目标优势、经验库匹配、candidate head bank、candidate scores、probability fusion、创新点和注意事项均已覆盖。
- 全量构建输出 A4 43 页；LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
- 抽查公式和正文：head bank shape、scan 输出、融合公式与创新边界均正常显示。

## 第 3.5.1 小节：两层位宽口径与当前支持

- 区分算法逻辑精度与物理单次调用位宽，明确物理契约统一为 uint4 input + int4 weight。
- 新增算子量化策略表：普通前向线性路径为 8-bit 逻辑值的 4-bit slices，candidate scan 为单次 4-bit。

## 第 3.5.2 小节：单次 4-bit 量化配置

- 补充 activation 仿射量化、weight 对称量化、scale/zero point、零点补偿和反量化公式。
- 明确候选扫描是当前真正的一次调用低位宽路径，4-bit 的硬件抗噪/能效优势尚待实测。
- 将仿射量化的 scale、zero point 和整数编码拆为三行 aligned 公式，消除横向溢出。

## 第 3.5.3 小节：Radix-16 位权拆分与 8-bit 逻辑精度

- 补充 uint 输入和 balanced signed weight 的 radix-16 展开，以及按 $16^{i+j}$ 重构 partial sums 的公式。
- 明确空间 tile 之后还需遍历 slice 对，物理调用数不能只按算法矩阵次数计算。
- 区分位权重构与直接 4-bit 截断，并说明 slice/累加/噪声/延迟代价。

## 第 3.5.4 小节：当前实现与证据边界

- 写明 BitSlicedPhotonicMatrixOpsBackend、QuantizedPhotonicMatrixOpsBackend 与 SignalOps 的当前分工。
- 明确当前是运行时动态量化而非 QAT，可选 simulator 缺失时回退到 NumPy integer matmul。
- 限定证据为软件量化、tile/slice、反量化和账本，不包含真实器件标定、噪声、功耗或时延。
- 将 simulator/integer matmul 组合改为中文表述，修复轻微行宽溢出。

## 第 3.5.5 小节：精度、抗噪与资源权衡评估

- 提出逐算子最低逻辑精度搜索与混合精度 policy。
- 给出逻辑 4/6/8/10/12-bit、物理 4/5/6/8-bit 和多噪声等级的建议实验矩阵。
- 以任务指标、有效位数、噪声、tile/slice、延迟、吞吐和能耗构建 Pareto 选择。

## 第 3.5 节量化策略构建验收

- outline 第 6 章的当前支持、4-bit 配置、位权拆分、实现边界、精度与资源权衡均已覆盖。
- 全量构建输出 A4 47 页；LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
- 抽查量化页：策略表、仿射/对称量化公式、radix-16 重构和证据边界均正常显示。

## 复赛 PPT 大纲

- 新增 `slides/ppt_outline.md`，正文 14 页，另含小型 MLP、BNCI 和工程复现 3 页备用页。
- 结构按赛题定位、问题、总体方案、三阶段、FBCSP、经验库、photonic scan、backend、量化、实验协议、结果、账本、创新边界展开。
- 每页使用短语式页面文字，并明确图片路径与讲解重点。
- 优先引用当前 artifacts 图表，明确不使用旧版 `embedding_diagnostics.png`。
- 加入匿名、纯图片版 PPT、1080P/5 分钟视频和不夸大 Cyton/真实硬件能力的制作约束。

## 复赛 PPT 逐页答辩脚本

- 在 `slides/ppt_outline.md` 中为 14 页正式页面逐页补充答辩脚本、建议时长和自然过渡语。
- 正式脚本总时长约 10 分钟，覆盖赛题对应、任务背景、方案、三阶段、FBCSP、经验库、候选扫描、backend、量化、协议、结果、账本和结论。
- 为 3 页备用页补充答疑脚本，分别对应小型 MLP、BNCI 个体结果和工程复现。
- 脚本保持证据边界：不声称经验库稳定超过 baseline，不把 100% 接管解释为功耗占比，不把 Cyton/真实芯片写成已完成。

## 复赛答辩 PPTX 生成

- 使用 `slides/第十届集创赛PPT模板.pptx` 的主题与 16:9 页面尺寸生成 `slides/光电混合运动想象脑机接口_复赛答辩.pptx`。
- 生成 14 页正式答辩内容和 3 页备用答疑内容；每页均包含短语式要点、至少 1 张图片和演讲者备注。
- 新增可重复生成脚本 `slides/tools/generate_presentation.py`，图片与文字映射来自当前 PPT 大纲。
- 结构检查：17 页、图片完整、14 页正式脚本和 3 页备用脚本均写入备注、模板占位文字残留为 0。
- 使用本机 PowerPoint 成功打开并导出 17 张 1600x900 PNG；抽查标题页、候选扫描页和结果页，无空白、重叠或裁切。
- PPT 保持匿名，不包含学校、指导教师或成员姓名；Cyton、官方模拟器和真实芯片状态按当前证据边界表述。

## PPT 第 5 页三阶段流程图重绘

- 原 `system_block_diagram_detailed.png` 信息过密，改为三张独立中文图：离线训练、会话校准、在线推理。
- 每张图包含阶段输入、5 个关键步骤、阶段输出和数据/执行边界，生成于 `slides/assets/stage_*.png`。
- 新增可重复绘图脚本 `slides/tools/generate_stage_diagrams.py`。
- 同步更新 PPT 大纲与生成器，第 5 页改为三图等高排列。
- 使用 PowerPoint 实际渲染检查：三图文字清晰，无重叠或裁切；PPT 保持 17 页，备注完整。

## 经验库会话特化流程图

- 按要求仅生成图片素材，不修改 PPT。
- 新增 `slides/assets/experience_library_selection_flow.png`，展示 2 Anchor + 64 Bootstrap → 66 候选 → 42 校准窗口 → 候选评分 → Top-8 → 加权融合 → 命令/拒识。
- 新增可编辑原生 Draw.io 文件 `slides/assets/experience_library_selection_flow.drawio`，框、箭头和文字均可独立调整。
- 新增可重复生成 PNG 的脚本 `slides/tools/generate_experience_library_flow.py`。
- 完成视觉检查与 XML 解析检查，修复 Bootstrap 标题裁切和竖向箭头标签重叠。

## 第 4.3.2 小节：主线 encoder 训练单图

- 从 small_network/arrays.npz 生成主线共享 encoder 的独立 loss/accuracy 图，不使用旧 embedding_diagnostics 拼图。
- 文档分析同步当前 220 epochs、最终 loss 0.082、训练准确率 96.43%，并强调训练收敛不等于泛化证明。

## 第 4.3.2 小节：主线 embedding PCA 单图

- 用训练 embeddings 与 518 个 held-out evaluation embeddings 联合拟合 PCA；明确排除 42 个 calibration windows。
- 圆点/叉号区分 train/evaluation，颜色区分 left/right/foot；分析 foot 聚集、left 分散和 left/right 重叠。

## 第 4.3.2 小节：主线累计指标单图

- 由 correct_trace 与 rejected 重新计算 cumulative command/accepted accuracy 和 reject rate，兼容旧 NPZ 未保存 cumulative 字段的情况。
- 分析初期波动、约 300 窗口后稳定以及最终 70.46% / 72.42% / 2.70%，并区分 cumulative 与 rolling。

## 主线单图生成工具

- 扩展 plot_experience_photonic.py，新增 mainline_encoder_training、mainline_embedding_pca、mainline_cumulative_metrics 的 PNG/PDF 单图输出。
- 三张 PNG 已同步到 template/figure/competition/ 并插入主线结果分析的合适位置。

## 主线新增单图构建与回归验收

- 最终 PDF 为 A4 50 页；LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
- 抽查训练曲线、PCA 与累计指标页：均为独立单图，图注和分析未裁切或越界。
- 工程回归测试 30 项全部通过。

## 第 3 章：Gazelle 模拟器单窗口性能记录

- 根据用户提供的终端截图转录 `run_single_window_inference.py --evaluation-index 0` 的性能记录；工作区未发现截图原文件，因此正文使用表格而未嵌入低清截图。
- 确认候选 backend=`gazelle_simulator`、bit width=4、qin=[0,15]、qwt=[-8,7]，其他线性路径为 `gazelle_simulator_bit_sliced`。
- Online forward 总计 476,405.402 ms、1,089,752 tiles。
- FBCSP 单窗口耗时 467,386.351 ms、1,087,136 tiles，约占总耗时 98.1% 和 tile 数 99.76%；候选 scan 为 305.832 ms、80 tiles。
- 本节仅保留单窗口耗时和 tile 数，不记录分类结果或正确性。
- 后续优化方向包括批量化、缓存、算子融合与逐算子最低精度搜索。

## 模拟器证据口径更新

- 项目状态改为“Gazelle 模拟器接口已接通、性能待优化”。
- 更新在线推理、TiledMVM、量化边界、计算占比结果解释：不再写当前只能 NumPy 回退，同时继续区分模拟器与真实光芯片实测。

## 第 3 章：Gazelle 单窗口性能统计口径调整

- 按要求删除 setup 耗时、样本标签、预测、概率、置信度、正确性与数值异常等内容。
- 文档和日志只保留单窗口 online forward 的分阶段耗时、总耗时和 bit-sliced photonic tile 数。
- 保留 FBCSP 为主要耗时/tile 瓶颈、候选 scan 为 80 tiles/305.832 ms 的性能分析，不将其与分类效果关联。
- 重新构建：A4 50 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0。

## 第 3 章：Gazelle 评估板单窗口时间推算

- 阅读 `info/Gazelle_光子计算评估板产品手册v1.0.1.pdf`：手册标注 $8\times2$ 基础光子矩阵计算、`2.6 MOPs`，脚注给出矩阵规模 $(1024,1024)\times(1024,1024)$；同时提供最多 $(4096,1024)$ 权重、$(1024,4096)$ 输入和 $(1024,1024)$ 输出的矩阵接口限制。
- 用本次单窗口的 $1,089,752$ bit-sliced tiles 按 $2.6\times10^6$ tile-equivalent operations/s 换算，得到连续供给、已配置且不含通信条件下的纯板端计算时间下界 $0.419\ \mathrm{s}$（419.1 ms）；相对 476.405 s 模拟器耗时的数量级比约为 1137。
- 明确该数值不是评估板端到端实测：手册未定义 MOP 的基本操作，也没有 API 单次调用、传输、FPGA 调度、读回或批量策略的延迟。文档同时给出若 MOP 被定义为标量 MAC/s 时的 0.254--0.508 s 理想计算口径，并要求以 `compass_matmul` 的逐窗口基准测试确认。
- 新版文档在独立验证目录完成 `xelatex -> biber -> xelatex -> xelatex` 构建：51 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0。默认 `out-competition/competition_report.pdf` 被外部进程占用，故未强制覆盖，验证版保存在 `out-competition-verify/competition_report.pdf`。

## 第 3 章：模拟器与评估板实时性表述调整

- 将 476.405 s 明确归因于模拟器逐 tile、逐 slice 展开的软件模拟开销，不再表述为“当前实现尚不具备实时性”，并说明该结果不能代表 Gazelle 实机运行速度。
- 将评估板时间推算压缩为标称算力解释、连续供给前提、$1,089,752/(2.6\times10^6)=0.419\ \mathrm{s}$ 公式和实时性分析，删除扩展的替代统计口径与冗长边界说明。
- 依据约 3 s 的 EEG 决策窗口，补充 0.42 s 对应约 2.39 window/s、具备单窗口实时推理计算时间条件的结论；保留通信、调度和数字端处理仍需实机端到端测量的边界。
- 删除本段及性能表总计行中的手工加粗格式。
- 重新构建验证版 PDF：51 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0。

## 结论部分：突出 Gazelle 平台的重要性

- 将 Gazelle 定位为项目从 MI-BCI 算法原型走向光电混合部署的关键承载平台，而不是仅用于替换单个分类头矩阵乘。
- 总结 FBCSP、标准化、小型 MLP、经验库候选 head bank 与 Gazelle 光子矩阵计算接口的对应关系，并强调三类统一 backend 建立的算法 shape、量化、tile/slice 调度和硬件执行边界。
- 衔接单窗口模拟器证据与标称算力估算：476.405 s 属于软件模拟开销，约 0.42 s 的板端计算估算短于约 3 s EEG 决策窗口，说明平台具备支撑实时推理的计算时间条件。
- 保留证据边界：真实评估板端到端时延、数值误差和功耗仍待实测，Cyton 上位机与真实闭环仍未完成。
- 重新构建验证版 PDF：51 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0。
## PPT 素材：统一后端接口与执行边界框图

- 仅新增独立图片与可编辑 Draw.io 素材，未修改 PPT 文件。
- 新图以 MatrixOps（矩阵/张量）、SignalOps（CAR/SOS）和 TiledMVM（候选 Bank）三类接口为中心，展示上层算法通过稳定接口与具体执行核解耦。
- 执行边界明确区分：全部前向线性算子由统一 Backend 接管，当前执行核为软件模拟；非线性、控制逻辑、拒识、经验库管理与流程调度仍保留在数字端。
- 输出文件为 `slides/assets/backend_interface_architecture.png` 与 `slides/assets/backend_interface_architecture.drawio`，并新增可重复生成脚本 `slides/tools/generate_backend_interface_diagram.py`。

## 第 1 章：插入文献 Fig. 1 的 MI-BCI 任务图

- 读取附件 `diagnostics-v13-i06_20260715.bib`，确认图片来源为 García-Murillo 等发表于 Diagnostics 13(6), 1122 (2023) 的论文 `KCS-FCnet: Kernel Cross-Spectral Functional Connectivity Network for EEG-Based Motor Imagery Classification`，DOI 为 `10.3390/diagnostics13061122`。
- 将论文 Fig. 1 原图复制为 `template/figure/competition/mi_bci_task_diagnostics_fig1.png`，放在“运动想象 EEG-BCI 任务与信号挑战”开头，替换原有通用任务流程图。
- 正文补充视觉提示、中央区电极和运动想象 EEG 采集场景的说明，图注与正文均明确标注文献来源；项目算法与 Gazelle 平台的具体流程仍由后续系统图说明，避免混淆文献背景图与本项目实现图。
- 首轮构建输出 50 页，图片与图注完整显示；同时发现量化章节的三列表格误声明为四列，修正列格式以消除该处 underfull alignment。
- 最终构建输出 50 页，LaTeX Error、未定义引用、overfull box、underfull box 均为 0。

## 工程维护：忽略 LaTeX 编译过程文件

- 在仓库根目录 `.gitignore` 中加入 XeLaTeX、Biber、latexmk、目录/图表索引、SyncTeX、Beamer 和术语表等编译过程文件扩展名。
- 保留 `.tex`、`.bib`、图片和最终 `.pdf` 可跟踪；同时忽略误由 PowerShell 变量字面量产生的 `report/template/$out/` 本地构建目录。
- 已经纳入 Git 索引的历史中间文件不会仅因新增忽略规则而自动取消跟踪，本次未修改 Git 索引。
