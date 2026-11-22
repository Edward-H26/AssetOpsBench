# 环境与agent调试

先看[正常流程、问题位置和修复状态](maintenance_loop/FLOW_AND_ISSUES.md)，再看[版本记录](maintenance_loop/EXPERIMENT_LOG.md)。

本目录是独立debug路径。正式agent、环境状态转移、维修规则和verifier未修改；普通`python -m CausalDiagnosis ...`不启用本目录的控制器替换。

下一版见[v005](maintenance_loop/versions/v005/README.md)：明确输出格式、当前可用信息和完成提示，并逐轮保存拒绝原因；v4整轮的冻结代码不变。

- [如何启动和冻结版本](maintenance_loop/README.md)
- [v4八场景×三配置的清单与运行方式](maintenance_loop/rounds/v4_24/README.md)

运行目录（`runs/`）、运行中的同步状态和整轮汇总不随仓库发布；仓库保存版本源码、配置和运行方式。整轮全部回合完成后自动归档并停止GPU服务。这里只是开发调试，不能替代完整测试集统计。
