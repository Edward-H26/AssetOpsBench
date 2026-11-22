# 环境与 agent 增量 debug 记录

目标：让 agent 在真实运行的环境中完成取证、诊断、维修、等待完工及复查，并分析诊断 MCP 是否帮助决策。失败保留，不通过修改标准制造成功。

## 记录约定

- 每次行为修改建立新版本，source/ 冻结；同版本可有多个 runs/，每次运行单独目录，不覆盖旧结果。
- 每个版本的 ANALYSIS.md 记录问题、证据、修改和下一版设计；每次运行的 README.md、summary.json 记录实际结果，原始输入输出、工具交互、计划及评分保留在下层。
- 正式环境、工具契约及 verifier 不变。debug 的单步控制器属于新变体，不能冒充原 agent 的正式成绩。
- 进程成功、工具能调用、诊断正确、维修恢复和最终 pass 分开记录。回合超时、网络/API错误单列。
- 当前 ±MCP 对照还改变了部分工具可用性，诊断信息的独立收益尚不能据此确认。

## 版本索引

| 版本 | 修改 | 已有结果 | 分析 |
|---|---|---|---|
| 原版 V0 | 正式代码 |  |  |
| [v001](versions/v001/README.md) | 分阶段开局、明确DSL |  | [分析与下一版](versions/v001/ANALYSIS.md) |
| [v002](versions/v002/README.md) | 首次单独读取、提示分段 |  | [分析与下一版](versions/v002/ANALYSIS.md) |
| [v003](versions/v003/README.md) | 实际只执行首个模型动作，再重规划 |  | [设计与分析](versions/v003/ANALYSIS.md) |
| [v004](versions/v004/README.md) | 修开局异常、保留计划进度、显式可用数据与空读反馈 |  | [设计与分析](versions/v004/ANALYSIS.md) |
| [v004_dynamic_scn6](versions/v004_dynamic_scn6/README.md) | agent不变，只换实时时钟/定时劣化场景 |  | [分析](versions/v004_dynamic_scn6/ANALYSIS.md) |
| [v005](versions/v005/README.md) | 明确输出字段、当前可用信息、核验后结束提示、逐轮原文日志 |  | [分析](versions/v005/ANALYSIS.md) |
