# Maintenance loop debug

先看[正常流程、问题位置与修复状态](FLOW_AND_ISSUES.md)。其中将评分通过、维修后核验和主动结束分别记录，避免把不同问题混成一个成功/失败数字。

独立实验入口，基于 `9fc86eabcd7c53b8fdde34c40b00043cf1876cae`。正式 agent、环境、工具契约和 verifier 文件均未修改。普通 `python -m CausalDiagnosis ...` 不启用本目录。

## 文件

- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)：按版本追加的总记录，包含结果、问题归因及下一版设计。
- `versions/<版本>/source/`：本版冻结代码和场景配置，哈希保存在 version.json。
- `versions/<版本>/ANALYSIS.md`：本版设计、实际证据、解释和下一版方案。
- `versions/<版本>/runs/<时间戳>/`：每次运行单独保存，含运行文档、summary/progress、每个模式的命令、stdout、原始episode和评分。
- `experiment.py`：建立新版本并冻结文件；按版本运行MCP与无MCP两组，生成每次运行报告。新版本和未来运行会向总文档追加事件。
- `run.py`：在当前进程安装替换逻辑，再调用原实验入口。
- `prompts_v2.py`：之前 prompt-only V2 的冻结版本，保留开局分阶段、首次真实读取等修改。
- `protocol.py`：当前开发版本的规划替换逻辑；具体运行必须以versions/中的冻结快照为准。v003丢弃尾部；v004保留待执行步骤并逐步重检，在观测/时间变化或surprise后要求模型重规划。完整模型回复由原llm_calls.json保存。
- `test_protocol.py`：验证原检查器仍拒绝非法第一步、重复诊断、重复工单、未完成维修就结束，并验证控制器执行合法前缀。
- `scenario1.json`：冻结之前已试跑的 Scenario 1 配置，源自本地 `batch_24_20261119/configs/scn1.json`；这是按队友描述重建的开发配置，不能冒充队友未提交的原 preset。

## 维修标准

工单创建成功不等于维修完成；工单关闭不等于设备恢复。遵守维修依赖，等待实际完成，在配置允许时取得维修后的真实读数并验证。诊断应覆盖整个设备。健康设备不应无故维修；证据不足声明仍按原 verifier 判定。MCP 的候选、概率、检查建议和干预预测都是建议，不能当隐藏真相。

环境的状态转移、依赖、恶化、费用、预算、时限、pass/reward 定义均沿用原代码。没有放宽证据要求，也不注入正确答案。

## 运行

先冻结版本，再运行冻结文件；不要边跑边改同一版本的source/。例如后续有新修改时：

```bash
python CausalDiagnosis/debug/maintenance_loop/experiment.py prepare \
  --version v005 --design "根据上一版的实际证据说明本次修改与预期"
python CausalDiagnosis/debug/maintenance_loop/experiment.py run-pair \
  --version v005 --base-url "$MODEL_BASE_URL" --model granite-3.3-8b-instruct
```

版本已存在时prepare会拒绝覆盖；run-pair先校验冻结文件哈希，每次创建新的UTC时间戳目录。v001/v002是历史结果迁入，保留原模型输出和评分，不伪装成新跑。

本地可直接检查协议（无需模型、GPU）：

```text
python CausalDiagnosis/debug/maintenance_loop/test_protocol.py
```

完整 episode 仍使用仓库准备好的 Linux Python 环境、数据和模型服务。Windows 原生运行器依赖 `fcntl`，本目录没有伪造替代文件锁。先在原集群环境中运行同一条已准备好的 Scenario 1 命令，只替换入口：

```bash
# 原入口：python -m CausalDiagnosis experiments run-episodes ...
python CausalDiagnosis/debug/maintenance_loop/run.py experiments run-episodes \
  --config CausalDiagnosis/debug/maintenance_loop/scenario1.json --scenario-ids 2007 \
  --agent causal --servers aob --store memory \
  --starts seeds --seeds 1 --first-seed 0 \
  --model ibm-granite/granite-3.3-8b-instruct \
  --base-url "$MODEL_BASE_URL" --judge none --out results/debug_maintenance_loop_v3/mcp
```

使用同一份配置、模型及 seed，添加 `--no-diagnostics`，并改用独立 `--out results/debug_maintenance_loop_v3/no-mcp`，运行另一组。API 认证继续使用原部署配置，不将密钥放进此目录。两组串行运行，避免干扰思考计时。不要在客户端完成/报错时释放 GPU allocation 或关闭模型服务。入口在输出目录写入 `debug_variant.json`，记录版本及替换文件哈希，避免与正式结果混淆。

这是新的控制器变体，不是原 agent 的正式实验结果；不要混入之前的 V0/V1/V2 或主表。先确认两组能够读取、诊断、下单、推进到完工、复查，再扩展到其他场景。

## 比较的已知限制

原 `aob_tool_contract.py` 的 `tools_for()` 在 diagnostics_server=False 时，task server 只保留 `FRAMEWORK_LEVEL_ACTIONS`；等待、查时间、维修手册等也受此开关影响。因此原生 ±MCP 对照同时改变了诊断信息和部分工具可用性，不能把差异全部解释为诊断模型的效果。本目录不暗中修改这项正式设计。提示根据实际公开工具列表说明 wait/forward-read 路径；缺工具时不能虚构等待动作。

记录至少包括：实际传感器读取数、空读数、诊断 MCP 调用及结果、计划拒绝原因、工单创建/关闭、维修后的读数与验证、最终 pass/诊断正确/恢复、时间与成本。出现改善也要检查是流程修复还是诊断建议起效；不预设带 MCP 一定获胜。

## 当前验证状态（2026-11-20）

v003已完成并归档；v004基础场景按用户要求切换而中断，不计完整成绩。v004动态Scenario6预试已完成：诊断正确但未修好，cycle60结束。

用户现要求固定v4跑8场景×3配置24回合，再归档并停止GPU。整轮GPU作业14027301已提交，状态与结果见[整轮记录](rounds/v4_24_20261120/README.md)。原GPU14011256已达到12小时调度时限；新整轮不再常驻，全部记录打包后自动关闭服务并释放。完整进度以SUMMARY.md和sync.json为准，不能把排队或正在运行写成已通过。
