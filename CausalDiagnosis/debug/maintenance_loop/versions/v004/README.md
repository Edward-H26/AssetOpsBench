# v004

设计：修复无profile/无wait开局崩溃；保留并逐步原生检查模型未执行步骤，观测/时钟变化或surprise后重规划；短提示展示可读周期与最近返回数，拒绝已知空静态窗口。环境与verifier不变。

代码冻结于 source/，哈希见 version.json。每次运行在 runs/ 独立保存；同版本不得修改 source/，新修改必须建立新版本。

原工具契约、环境及 verifier 不变；±MCP 的工具可用性差异仍是比较限制。
