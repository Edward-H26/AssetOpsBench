# v5定向debug：独立GPU

本轮独立于v4整轮，用于定位和修复问题。本轮是3场景×2配置的6回合小规模测试，不是全量实验。

运行时run_round.py把状态写入status.json、汇总写入SUMMARY.md，本地同步器写sync.json；这些文件与runs/都不提交。v4整轮的冻结代码保持不变；本轮使用自己的服务地址与密钥。

## 先测什么

| 场景 | 配置 | 要检查的问题 |
|---|---|---|
| Scenario1，seed0 | v5带/不带诊断MCP | 输出参数是否符合接口；诊断、维修、核验之后的结束步骤 |
| Scenario2，seed0 | v5带/不带诊断MCP | 空读后的下一步，以及所需维修是否完成 |
| Scenario6，seed1 | v5带/不带诊断MCP | 思考计时与劣化发生后的取证与维修调整 |

本轮不含PE基线，PE的未修改基线在v4整轮的清单里；不同GPU的时延不能混成正式对照，本轮只定位本目录agent的执行问题。

## 如何查看和迭代

代码和配置冻结在source/、configs/，SHA256与6任务清单在manifest.json。每个回合的episodes/debug_trace.jsonl逐轮写出模型请求、原始回复、计划解析/检查结果、执行反馈；拒绝原因在当轮写入。

发现问题时先引用具体原文与拒绝原因，区分程序、接口、提示、决策及环境按规则失败；当前版本保留，再以新版本修复。不自动转换模型错误参数、不替模型给答案、不通过放宽评分制造成功。

GPU时限4小时，6项记录完成后自动打包归档并停服释放，不常驻。本地只读同步程序每120秒同步进度，完整归档完成后校验SHA256并下载。

集群上的启动方式与[rounds/v4_24](../v4_24/README.md)相同：`RUN_AFTER_SERVE`指向的脚本先运行`python <本目录>/source/test_protocol.py`，再对3份配置运行`python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables --config <配置> --check`，最后运行`python <本目录>/source/run_round.py --round <本目录> --base-url "$BASE_URL"`。
