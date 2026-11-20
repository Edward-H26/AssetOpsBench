# v004_dynamic_scn6 / 20261120T081347_985606Z

状态：completed。开发试跑，不计入正式主表。

| 模式 | pass | 诊断全对 | 恢复 | 工具调用 | 诊断MCP调用 | 时间cycle | 结束原因 |
|---|---|---|---|---|---|---|---|
| mcp | False | True | False | 14 | 9 | 60 | horizon |

## 证据与限制

完整原始记录在各模式的 episodes/；命令、进程日志、拒绝原因和计数单独保留。
进程返回0不等于任务通过；回合超时/API失败与诊断失败分别记录。空读计数只识别明确的空数组，需结合原始返回核实。
±MCP 沿用正式工具契约，两组的等待等工具可能不同，不能把差异全部归于诊断信息。

## mcp

工具计数：`{"task.get_asset_profile": 1, "causal_diagnosis.rank_root_causes": 9, "fmsr.get_failure_modes": 4, "wo.list_workorders": 1, "task.get_current_time": 1, "iot.stream_extent": 1, "iot.history": 4, "task.submit_diagnosis": 1, "wo.generate_work_order": 1}`

- 2 次：the plan's prices (330) exceed the budget left (45)

## 分析与下一版本

自动报告只描述可观测结果。人工分析在 ANALYSIS.md 追加，并引用实际步骤；未经实测的设计不得写成收益。
