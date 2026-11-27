# Peter：环境修复与集群开发验证交付（2026-11-26）

本次交付包含公共环境接口修复、CPU 验证及两条真实模型开发回合的结果摘要。**不代表正式实验全部完成。** 依据任务分配对应的 [cluster_jobs README](../../README.md)，基线代码为 `3cd42280f8a3209cea3488d38bfa4149d309fa0a`。

## 分支与修改

分支：`hpml/environment-tool-parity`。没有合入 main，没有调整 agent 提示词或诊断策略，也没有改变物理状态转移、维修效果、价格、reward 或 verifier 判定。

- `28490cc86d13e83d3fa3dd0e9da7b54235b8d144`：有无额外 CausalDiagnosis MCP，公共 IoT/FMSR/WO/task 工具保持一致；等待和查时间等能力由环境层级决定。
- `40e5d90f5084db9bed8ea506df3c2c4e49eabf1f`：工单创建回复在数据库同步后返回实际生命周期状态和公共维修时间；时钟回执显示未完成工单及完成通知。工单完成不等于设备恢复，仍需修后观测验证。所有 agent 共享这些反馈。
- 入口位置：[计量及工具适配](../../../frameworks/AssetOpsBench/tool_servers/metering_layer.py)、[公共工单反馈](../../../frameworks/AssetOpsBench/tool_servers/work_order_progress.py)、[回归测试](../../../frameworks/AssetOpsBench/tool_servers/tests/test_work_order_progress.py)。

## 与任务分配对应的进度

| 分配部分 | 已验证 | 尚未完成 |
| --- | --- | --- |
| Before the first job | 固定 commit、数据校验、Python 环境、CouchDB 及生成表准备；原生启动摘要检查 572 项一致 | 正式新版本运行需建立该 pushed commit 的新 clone，不能原地覆盖旧 clone |
| Serving a model | L40S 上成功运行 Granite 3.3 8B，BF16、131072 context；真实 API 和 agent 请求成功 | 新补丁的真实 agent 回合尚未重跑 |
| One development row | 按 README 的指定起点示例，完成 CausalDiagnosis agent 带/不带额外诊断 MCP 两条 base 回合，真实 LLM、真实 CouchDB | 此批未完成真实 PE/AAT 对照；不是全部 development combinations 或八场景全跑 |
| After a run | 保存原始轨迹、评分、模型调用；导出逐步 HTML；提取状态变化与失败原因 | 未完成正式 judge、run certification、正式表格及整套 after-run 流程 |
| grid_agents.txt | 核对原生生成网格：107 条实验运行行，主表是三类 agent 各带/不带诊断 MCP 共六配置 | 未跑全量网格；不是旧的 3×8 演示清单 |
| test_run_review_pack.py | CPU 作业 14088650：10 passed | 开发回合不应绕过正式 review-pack 的 development 排除规则；此处用 explorer 导出 HTML |

## 验证证据

机器可读摘要：[checks.json](checks.json)。这些记录是开发验收，不是论文实验结果。

- CPU 14088708：真实 CouchDB 与内存后端一致性，2 passed。
- CPU 14088832：原生 agent 接口，以预设模型回复验证，8 passed；不算真实模型成绩。
- CPU 14111050：最新公共反馈补丁，工具服务器测试和原生 Causal/PE/AAT 相关流程，共 532 passed、107 warnings，退出码 0。覆盖即时完成、延迟维修、agent 无动作期间推进、依赖不满足导致维修无效，以及有无诊断 MCP 的公共反馈一致性。此处原生 driver 测试不调用真实 LLM。7 个修改文件与远程受测源码 SHA-256 全部相等。
- 首轮 14111000 为 528 passed / 4 failed：新加测试误配时钟模式。保留失败记录，修正测试配置以使用原生 preset 后重跑；没有通过改变环境状态转移来使断言通过。

## 真实模型回合与问题分类

真实模型回合运行于 `28490cc`，**在最新工单反馈修复之前**。两条都采用 base、seed 0、`--start-state 3/100/0/130@2`。base 不按思考耗时演化，不能据此声称动态场景已跑通。二者可能共享模型推理资源，因此也不作为严格延迟比较。

| 配置 | 实际行为 | 判定 |
| --- | --- | --- |
| CausalDiagnosis agent + CausalDiagnosis MCP | 诊断正确，维修冷却器后状态由 `[3,100,0,130]` 变成 `[100,100,0,130]`；随后对健康阀和蓄能器各开一单 | recovered=True、Top-1=1、unnecessary=2、pass=False；7 次模型调用、14 步 |
| 同一 agent，无额外 CausalDiagnosis MCP | 提交错误诊断，没有开维修单，之后重复读取已有读数和提交相同诊断 | recovered=False、Top-1=0、pass=False，60 步结束；59 次模型调用 |

环境问题是“工单回执的状态落后于已同步的设备/数据库状态”，现已修复并通过 CPU 回归。前者的多余维修和后者的误诊/重复动作是实际 agent 行为，不能通过禁止合法错误操作或自动选择正确维修来掩盖。两条任务未通过，但都不属于模型 API 无法连接或程序异常退出。

原始记录和逐步 HTML 已在 Peter 本地归档，集群原始运行目录不修改。此提交仅包含开发摘要及源码哈希，不含 API key、私有 env.sh、原始模型服务日志或可冒充正式认证的结果包。摘要不替代完整 ledger 认证。

## 下一步

在新 pushed commit 的独立 clone 中，按 README 重新准备表和摘要，先串行运行静态和动态开发场景的三种演示配置（我们的 agent 带/不带诊断 MCP、PE 带诊断 MCP）。核对实际公共反馈和状态轨迹后再决定是否扩大全量。正式主表仍按原生六配置网格执行，并完成 judge、认证与 after-run 检查。agent 任务失败也应保留，不能写成接口故障或删除。
