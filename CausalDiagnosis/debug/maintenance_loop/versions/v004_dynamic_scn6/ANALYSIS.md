# v004_dynamic_scn6 分析

## 设计

保持v004 agent字节不变，改测已有Scenario6：实时时钟与cycle3冷却器定时劣化，初始阀/泵轻度故障，无失效截止；先只测MCP一回合，验证agent思考期间的状态变化。

## 实测证据

运行记录不随仓库发布；experiment.py run-pair 为每次运行在 runs/ 下生成记录。

## 场景与控制变量

使用重建的Scenario6配置：初始[100,90,1,130]，seed1选中冷却器，第3个cycle首次劣化，之后每10个cycle再下降一级；failure开关关闭。配置包含real_time层，因此应把思考秒数折算成周期，具体运行参数以原生episode.config为准。

run.py、protocol.py、prompts_v2.py、test_protocol.py四个文件SHA256与v004完全一致，不在本轮偷偷修改agent。只替换场景配置并选择seed1；配置标注为重建的debug配置，不冒充原有的preset。

## 要检查的实际证据

- 原生clock_mode与think_seconds_per_cycle是否生效。
- agent尚未下维修单时，实际模型思考是否使cycle前进。
- 冷却器是否在规定周期从100变为20，是否产生对应观测。
- agent是否重新取证、调整诊断/维修，还是继续按开局计划行动。
- 思考和环境变化导致的生产损失、最终状态与评分。

先区分环境演化是否正确和agent是否成功；agent失败不能直接解释为环境没有变化。

## 结果解释与下一版

完成后引用运行目录和实际步骤追加分析。
