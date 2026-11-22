# v4完整debug轮次

8个场景，每个依次运行：本目录的v4+诊断MCP、v4无诊断MCP、原版Plan-and-Execute+诊断MCP，共24回合。运行时run_round.py把状态写入status.json、汇总写入SUMMARY.md，本地同步器写sync.json；这些文件与runs/都不提交。

agent代码与8份配置已冻结，清单和SHA256见manifest.json。模型Granite3.3-8B BF16、131072上下文；不运行LLM judge。场景使用此前重建的debug配置，非正式完整测试集；单seed不作显著性结论，±MCP仍有部分工具可用性差异。

每个任务保存在runs/<场景_模式>/attempt_XX/：

- command.json、stdout.log：实际命令及运行错误。
- episodes/：原模型输入输出、工具轨迹、计划、环境记录与原评分。
- result.json、ANALYSIS.md、evidence_examples.json：计数、初步分类和具体失败证据。

代码在整轮运行中不修改。分析区分环境是否按规则更新、agent是否看到更新、格式/协议错误、控制器拒绝、诊断/维修决策和服务异常；prompt与模型能力不能凭单个失败直接分离。

完成24个任务后自动生成完整SUMMARY.md及压缩包，校验信息在archive.json。GPU服务随RUN_AFTER_SERVE结束而停止并释放。单回合失败会保留并继续下一项，不能把进程返回0写成任务成功。

集群上的启动方式：用`CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch`启动模型服务，`RUN_AFTER_SERVE`指向的脚本先对8份配置运行`python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables --config <配置> --check`，再运行`python <本目录>/source/run_round.py --round <本目录> --base-url "$BASE_URL"`。

本地隐藏同步进程只读查询集群，每120秒更新状态。完成后下载压缩包、验证SHA256、解压到本目录，再确认GPU作业结束后退出。VPN断开时自动重试；集群归档与GPU关闭不依赖本地同步程序。同步日志在sync_stdout.log/sync_stderr.log。
