# scn01 / causal_no_mcp

开发试跑，单seed，不作为正式显著性结论。

| 指标 | 结果 |
|---|---|
| pass | False |
| diagnosis_all_correct | False |
| recovered | False |
| clock_mode | declared |
| time_used | 5 |
| cost | 110 |
| downtime_total | 10 |
| end_reason | finish |
| error | WallLimit: the episode passed 1800 s; APITimeoutError: Request timed out. |

## 观测到的问题类型

- 格式/协议：存在解析失败。模型输出与prompt/schema共同影响，不能仅凭解析失败断言模型能力不足。
  - 21次：the plan needs a non empty list under steps
  - 18次：step 1 (read): sensor_group must be a non empty string, got ['VS1_Vibration_mm_per_s_1Hz', 'FS1_VolumeFlow_l_per_min_10Hz', 'CP_CoolingPower_kW_1Hz']
  - 1次：step 2: kind must be one of read, diagnose, declare, order, wait, finish, profile, failure_modes, fix_history, time, extent, recall, got 'get_failure_modes'
  - 1次：mode must be one of test_first, act_now, repair_as_test, declare, got 'diagnose'
- 控制器：存在执行前拒绝。具体原因见result.json和evidence_examples.json，不等同于环境物理规则。
- 运行异常：WallLimit: the episode passed 1800 s; APITimeoutError: Request timed out.。与普通任务失败分开，不直接计作推理能力结论。

## 原始证据

完整输入/输出、工具反馈、计划和原评分在episodes/。本报告只做证据分类，不自动宣称唯一根因。
