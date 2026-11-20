# 第一场景三组对照：提交时的固定快照

来源为review_snapshot_status.json中的3个已完成回合。整轮当时仍在运行，不是最终24回合成绩。配置：Scenario2007，seed0，同一Granite3.3-8B BF16/131072服务，memory backend，judge none。

| 模式 | pass | 诊断正确 | 恢复 | cycle | 结束方式 | 关键证据 |
|---|---|---|---|---|---|---|
| 我们的v4+MCP | True | True | True | 30 | max_steps | 4次history、1次维修后verify_outcome，核验consistent=True；工单closed；仍反复等待 |
| 我们的v4无MCP | False | False | False | 5 | harness在1800秒时限结束 | 48次LLM调用、2次history；18次sensor_group数组错误、21次空steps错误；没有实际诊断/工单 |
| PE+MCP | False | False | False | 0 | finish | 使用SiteX等参数；两次工单返回unknown component；声明引用非本设备真实读数的证据 |

## 能得出的结论

v4+MCP这一例已完成“诊断→维修→重新读取→核验”，但没有及时主动结束。无MCP开局的debug代码异常已修复，当前卡点是后续输出格式与提示/接口对接。PE这一例也存在工具参数及证据定位问题，不能把其失败概括成根因分析不准。

无MCP的错误字符串含APITimeoutError，但原评分同时记录cut_end=wall、failed_model_attempts=0、finish_by=harness；主要已观察到的问题是反复输出无效计划并耗尽整回合时间，不能只据错误字符串宣布模型服务宕机。

## 回查位置

- [v4+MCP结果](runs/scn01_causal_mcp/attempt_01/result.json)
- [v4无MCP结果](runs/scn01_causal_no_mcp/attempt_01/result.json)
- [PE+MCP结果](runs/scn01_plan_execute_mcp/attempt_01/result.json)

各自episodes/下的llm_calls.json、trajectory.json、plans.json（自有agent）、scores.csv保留原始数据。运行中的本地SUMMARY.md会被同步器更新，故本次推送使用此固定快照作为结论依据。
