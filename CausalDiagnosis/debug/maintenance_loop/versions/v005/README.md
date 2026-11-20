# v005：提示格式与逐轮记录

状态：10项离线检查通过，**未进行真实LLM试跑**。当前24回合继续使用冻结v004；不在本轮插入v005，不追加GPU，原轮次完成后仍自动归档停服。

## 改了什么

- 用明确字段契约替代固定的“先read”示例：mode/steps必填，sensor_group是一个实际组名称字符串，候选位于根对象，kind是控制器动作名。
- 从公开状态说明哪些组还有数据、是否能等待、是否能查询时间。只反映v004和原检查器已有条件，不增加环境规则。
- 把维修后核验结果、剩余预算和读数价格单独列出。提醒模型在证据支持整机恢复时主动finish，但不替它选择finish，也不把一次局部维修核验当成全机健康。
- 运行时逐轮写debug_trace.jsonl，包含model_request、model_reply、plan_decision、execution_result及异常。无需等整回合结束才能看拒绝原因。

## 没改什么

v4_protocol.py与v004/source/protocol.py完全相同，保留其逐步检查、pending计划、原生parser/checker与空读检查。环境、维修依赖、价格、评分、数据及MCP实现不变。不自动修正模型参数，不替模型选诊断或维修，不自动结束回合。

## 回查

- [设计依据、验证与下一步](ANALYSIS.md)
- [冻结文件哈希与状态](version.json)
- source/protocol.py是新提示和记录包装；source/v4_protocol.py是未修改的父版执行逻辑。

离线检查：在仓库根目录运行`python CausalDiagnosis/debug/maintenance_loop/versions/v005/source/test_protocol.py`。

以后若实测，应以独立输出目录运行`experiment.py run-pair --version v005 ...`，用相同场景、seed、模型比较v004，不覆盖旧记录。目前没有提交这项GPU任务。
