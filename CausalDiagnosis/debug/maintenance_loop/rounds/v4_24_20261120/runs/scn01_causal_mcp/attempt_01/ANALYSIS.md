# scn01 / causal_mcp

开发试跑，单seed，不作为正式显著性结论。

| 指标 | 结果 |
|---|---|
| pass | True |
| diagnosis_all_correct | True |
| recovered | True |
| clock_mode | declared |
| time_used | 30 |
| cost | 281 |
| downtime_total | 16 |
| end_reason | max_steps |
| error |  |

## 观测到的问题类型

- 格式/协议：存在解析失败。模型输出与prompt/schema共同影响，不能仅凭解析失败断言模型能力不足。
  - 2次：the plan needs a non empty list under steps
  - 2次：step 1: kind must be one of read, diagnose, declare, order, wait, finish, profile, failure_modes, fix_history, time, extent, recall, got 'wait_for_notification'
  - 1次：step 2: kind must be one of read, diagnose, declare, order, wait, finish, profile, failure_modes, fix_history, time, extent, recall, got 'get_failure_modes'
  - 1次：step 2: kind must be one of read, diagnose, declare, order, wait, finish, profile, failure_modes, fix_history, time, extent, recall, got 'wait_for_notification'
- 控制器：存在执行前拒绝。具体原因见result.json和evidence_examples.json，不等同于环境物理规则。

## 原始证据

完整输入/输出、工具反馈、计划和原评分在episodes/。本报告只做证据分类，不自动宣称唯一根因。
