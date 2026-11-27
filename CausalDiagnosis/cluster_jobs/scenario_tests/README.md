# 八场景开发测试：先固定运行模式，再选场景

状态：代码与测试定义已编写。按用户要求暂停集群验证；最新版本尚未通过完整回归，也没有重跑真实 agent 的 24 回合。先前部分测试结果不作为本版整体验收。

本目录定义新的测试条件，不覆盖旧 v4 配置或既有运行记录，也不改变正式六层阶梯实验。旧配置未声明模式时维持原行为；旧的 `static` preset 仍代表“无自主劣化但维修有时间”，不能与这里的严格静态混用。

| 条件 | `static` | `dynamic` |
| --- | --- | --- |
| 模型思考、空动作 | 不推进时间 | 累计思考秒数，按资产的 cycle 折算推进 |
| 检查、备件、维修耗时 | 全部为零 | 使用资产声明的时长 |
| 等待工具 | 公共 task server 不提供；直接调用引擎也拒绝且不推进时间 | 所有 agent 都可使用，等待下一个通知或 timeout |
| 自主劣化、失效、临时维修过期 | 关闭 | 劣化和失效沿用场景配置，不凭模式凭空增加故障 |
| 生产时间损失 | 关闭 | 沿用场景成本开关；价格不由运行模式决定 |
| 维修依赖、优先级、证据、收费、预算 | 保留场景原有规则 | 保留场景原有规则 |

静态维修不需要等待。连续工单仍按既有“同次上门”规则归组，接下来的非工单调用会在同一 cycle 结算；这保证依赖和维修顺序仍有效。开单受理不代表依赖必然满足，维修后仍须验证读数。静态中也保留步骤上限，不允许无限免费循环。

数据边界：AOB 的 `history` 返回已记录的时间历史。冻结时钟后，它不会自动追加修后录音；本轮静态验收验证的是工单结算与设备隐藏状态更新，不能把重新读取开局历史称为修后遥测验证。独立的修后观测快照接口仍待完善，不能伪造真实连续维修录音。旧配置省略模式时，序列化中不增加字段，保持旧回放语义。

## 一个参数切换

```bash
python -m CausalDiagnosis experiments run-episodes --config environment_test_scn1 --environment-mode static --agent causal --servers aob --store couchdb --starts seeds --seeds 1 --first-seed 0 --scenario-ids 2007 --model granite-3.3-8b-instruct --base-url "$BASE_URL" --judge none --out "$OUT"
```

将 `--environment-mode static` 改为 `--environment-mode dynamic` 即可切换同一个场景的运行方式。JSON 中同名字段 `environment_mode` 设置默认值，CLI 优先。模式在价格表覆盖之后再次统一约束，避免价格表悄悄恢复静态维修延迟。与模式冲突的 `--clock-mode` 会报错。完整生效配置及模式写入 `episode.json`/`run.json`，公共资产档案也包含模式。显式模式的回合标记为开发条件，不进入现有正式表格。

## 八场景与三配置

[manifest.json](manifest.json) 固定场景、seed、AOB scenario ID 和三种原生 agent 配置。1、2、7、8 为静态，3–6 为动态。诊断 MCP 只提供额外建议，公共环境工具对三配置一致。

这些场景从旧 debug 配置改编，保留初始状态、题目锚点及事件参数。不能冒充队友未发布的原始 preset。尤其场景 5、7 的“不可分”需要在新条件下重新核对参考标签及 E18，不能仅凭名字宣布其应当弃权。

```bash
python CausalDiagnosis/cluster_jobs/scenario_tests/write_grids.py --out "$H/grids"
# 先跑场景 1（seed 0）和 6（seed 1），分别三配置；不要把八场景拼成同一个全局 seed。
E=ALL,RUN_KIND=development,RUN_AFTER_SERVE=$H/3_run/run_lines.sh
E=$E,GRID_FILE=local_environment_scn1.txt,GRID_STARTS=seeds,GRID_SEEDS=1,GRID_FIRST_SEED=0,GRID_PARALLEL=1,GRID_REFERENCE=0
bash "$H/submit.sh" serve_model.sbatch --export="$E" --gres=gpu:l40s:1
```

按原生 README 在 CPU 先为各 grid 计算/检查生成表、准备 clone 摘要。场景 6 使用其 grid 和 `GRID_FIRST_SEED=1`。真实时钟回合串行执行。使用最新 pushed commit 的新 clone；不能将新结果覆盖旧记录。agent 决策失败保留，环境/接口故障单独归类。每个场景的 grid、日志和查看页面单独存放，导出脚本必须显式 source env.sh，不能依赖子 shell 导出 ROOT。此处命令使用原生服务作业的结束策略；需要长期保留 GPU 时使用已记录的保持服务包装脚本。
