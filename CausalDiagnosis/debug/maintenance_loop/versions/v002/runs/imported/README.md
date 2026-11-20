# v002 / imported

状态：completed。开发试跑，不计入正式主表。

| 模式 | pass | 诊断全对 | 恢复 | 工具调用 | 诊断MCP调用 | 时间cycle | 结束原因 |
|---|---|---|---|---|---|---|---|
| causal_mcp | False | False | False | 48 | 41 | 0 | max_steps |
| causal_no_mcp | False | False | False | 18 | 0 | 40 | finish |

## 证据与限制

完整原始记录在各模式的 episodes/；命令、进程日志、拒绝原因和计数单独保留。
进程返回0不等于任务通过；回合超时/API失败与诊断失败分别记录。空读计数只识别明确的空数组，需结合原始返回核实。
±MCP 沿用正式工具契约，两组的等待等工具可能不同，不能把差异全部归于诊断信息。

## causal_mcp

工具计数：`{"task.get_asset_profile": 1, "causal_diagnosis.rank_root_causes": 41, "fmsr.get_failure_modes": 4, "wo.list_workorders": 1, "task.get_current_time": 1, "iot.stream_extent": 1, "iot.history": 40}`

- 6 次：the plan needs 8 tool calls on the action server but 6 steps are left
- 3 次：step 5 (finish): finish comes before the reading that verifies the repair, so read tier3 for 1 cycle at or after cycle 6, after the last close, then finish
- 2 次：step 6 (finish): finish comes before the reading that verifies the repair, so read tier3 for 1 cycle at or after cycle 6, after the last close, then finish
- 1 次：the plan needs 9 tool calls on the action server but 8 steps are left
## causal_no_mcp

工具计数：`{"task.get_asset_profile": 1, "fmsr.get_failure_modes": 4, "wo.list_workorders": 1, "iot.stream_extent": 1, "iot.history": 9, "task.submit_diagnosis": 1}`

- 37 次：step 1 (diagnose): a diagnosis or a declaration was already made; the environment refuses a second one
- 1 次：step 2 (diagnose): a diagnosis or a declaration was already made; the environment refuses a second one
- 1 次：the plan's prices (185) exceed the budget left (165)
- 1 次：the plan reaches cycle 80 but the episode ends at the horizon of 60 cycles before finish can run

运行错误：WallLimit: the episode passed 1800 s; APITimeoutError: Request timed out.

## 分析与下一版本

自动报告只描述可观测结果。人工分析在 ANALYSIS.md 追加，并引用实际步骤；未经实测的设计不得写成收益。
