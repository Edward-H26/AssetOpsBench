# v004_dynamic_scn6

设计：保持v004 agent字节不变，改测已有Scenario6：实时时钟与cycle3冷却器定时劣化，初始阀/泵轻度故障，无失效截止；先只测MCP一回合，验证agent思考期间的状态变化。

代码冻结于 source/，哈希见 version.json。每次运行在 runs/ 独立保存；同版本不得修改 source/，新修改必须建立新版本。

原工具契约、环境及 verifier 不变；±MCP 的工具可用性差异仍是比较限制。
