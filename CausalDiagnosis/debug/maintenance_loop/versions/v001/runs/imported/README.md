# v001 / imported

状态：completed。开发试跑，不计入正式主表。

| 模式 | pass | 诊断全对 | 恢复 | 工具调用 | 诊断MCP调用 | 时间cycle | 结束原因 |
|---|---|---|---|---|---|---|---|
| causal_mcp | False | False | False | 8 | 1 | 0 | max_steps |
| causal_no_mcp | False | True | False | 10 | 0 | 0 | max_steps |

## 证据与限制

完整原始记录在各模式的 episodes/；命令、进程日志、拒绝原因和计数单独保留。
进程返回0不等于任务通过；回合超时/API失败与诊断失败分别记录。空读计数只识别明确的空数组，需结合原始返回核实。
±MCP 沿用正式工具契约，两组的等待等工具可能不同，不能把差异全部归于诊断信息。

## causal_mcp

工具计数：`{"task.get_asset_profile": 1, "causal_diagnosis.rank_root_causes": 1, "fmsr.get_failure_modes": 4, "wo.list_workorders": 1, "task.get_current_time": 1, "iot.stream_extent": 1}`

- 28 次：step 5 (finish): finish precedes the close of a work order (the last closes at cycle 5, the plan reaches cycle 4); wait to the close so that the controller can verify it first
- 8 次：step 5 (finish): finish precedes the close of a work order (the last closes at cycle 5, the plan reaches cycle 1); wait to the close so that the controller can verify it first
- 6 次：step 5 (finish): finish precedes the close of a work order (the last closes at cycle 6, the plan reaches cycle 1); wait to the close so that the controller can verify it first
- 2 次：step 5 (finish): finish comes before the reading that verifies the repair, so read tier3 for 1 cycle at or after cycle 6, after the last close, then finish
- 2 次：step 7 (finish): finish precedes the close of a work order (the last closes at cycle 6, the plan reaches cycle 4); wait to the close so that the controller can verify it first
- 1 次：step 6 (diagnose): a diagnosis or a declaration was already made; the environment refuses a second one
- 1 次：step 7 (finish): finish precedes the close of a work order (the last closes at cycle 5, the plan reaches cycle 4); wait to the close so that the controller can verify it first
- 1 次：step 6 (finish): finish precedes the close of a work order (the last closes at cycle 5, the plan reaches cycle 4); wait to the close so that the controller can verify it first
- 1 次：step 8 (diagnose): a diagnosis or a declaration was already made; the environment refuses a second one
- 1 次：step 9 (finish): finish precedes the close of a work order (the last closes at cycle 6, the plan reaches cycle 4); wait to the close so that the controller can verify it first
- 1 次：step 5 (finish): finish comes before the reading that verifies the repair, so read tier3 for 1 cycle at or after cycle 5, after the last close, then finish
- 1 次：step 8 (finish): finish precedes the close of a work order (the last closes at cycle 6, the plan reaches cycle 1); wait to the close so that the controller can verify it first
## causal_no_mcp

工具计数：`{"task.get_asset_profile": 1, "fmsr.get_failure_modes": 4, "wo.list_workorders": 1, "iot.stream_extent": 1, "iot.history": 1, "task.submit_diagnosis": 1, "wo.generate_work_order": 1}`

- 48 次：step 1 (diagnose): a diagnosis or a declaration was already made; the environment refuses a second one
- 48 次：step 2 (order): a work order on the pump is still open at this step; the environment refuses an order on a component with an open order
- 1 次：step 2 (diagnose): a diagnosis or a declaration was already made; the environment refuses a second one

## 分析与下一版本

自动报告只描述可观测结果。人工分析在 ANALYSIS.md 追加，并引用实际步骤；未经实测的设计不得写成收益。
