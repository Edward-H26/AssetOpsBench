# v003

设计：每轮执行模型提出的第一个合法动作；公开工单进度；沿用原维修与验证标准；Scenario2007 seed0 对照 MCP 与 no-MCP

代码冻结于 source/，哈希见 version.json。每次运行在 runs/ 独立保存；同版本不得修改 source/，新修改必须建立新版本。

原工具契约、环境及 verifier 不变；±MCP 的工具可用性差异仍是比较限制。

本次试跑已完成：[运行结果](runs/20261120T064603_717907Z/README.md)、[最终分析和下一版设计](ANALYSIS.md)。MCP组未通过；无MCP组因debug代码开局异常，不能用于模型能力对照。
