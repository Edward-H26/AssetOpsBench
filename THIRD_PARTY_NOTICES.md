# Third party notices

No dataset file is part of this repository. [CausalDiagnosis/docs/dataset.md](CausalDiagnosis/docs/dataset.md) lists every dataset that the code fetches, with its official page, its license as its source states it and its fetch command.

## Meta Agents Research Environments (ARE)

- Source: https://github.com/facebookresearch/meta-agents-research-environments at commit a9fd2edccb4bc7e22fbf45c8482a5fef30a2a5a4.
- License: MIT License, Copyright (c) Meta Platforms, Inc. and affiliates. The full text follows this list.
- `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/time_tools.py` is adapted from `are/simulation/apps/system.py` (the System app, its tools `get_current_time` and `wait_for_notification`, and `WaitForNotificationTimeout`) and from `Environment.wait_for_next_notification` in `are/simulation/environment.py` (a wait ends at the next notification or at its timeout).
- `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/notifications.py` is adapted from `are/simulation/notification_system.py` (the message types `ENVIRONMENT_NOTIFICATION` and `ENVIRONMENT_STOP`, the message, and the timeout message that only a wait without another notification receives).
- The adaptation counts time in the integer cycles of a deterministic clock instead of ARE's wall clock, so that an episode replays byte for byte, describes the tools in the terms of the hydraulic rig, sets the default timeout to the cycles left before the horizon, runs a wait one cycle at a time because every cycle of the rig runs processes, drops attachments and the user message type, and carries the notifications on every tool answer, since the AssetOpsBench frameworks have no message queue. The time tools of the action server follow ARE's System app in their names and semantics.

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ReActXen

- Source: https://github.com/IBM/ReActXen at commit da40e4ba3b93c8c9dde100717323a0c6cab11edc.
- License: Apache License, Version 2.0. The full text is in `CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/LICENSE`. The repository has no NOTICE file.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/` holds the 17 files of `src/reactxen/` that `ReactReflectAgent`, its reviewer and `JSONReader` import, for the port of the Agent-As-Tool of the AssetOpsBench paper in `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/`. Each file opens with a header that names its source file and the changes, which are mechanical: the imports of `reactxen` renamed to `CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen`, `PromptTemplate` imported from `langchain_core.prompts`, colorama replaced by blank stand-ins in `_colors.py`, mdextractor imported when it is called, and the import of `utils.utils` removed from `utils/__init__.py`, since it needs `langchain_ibm`.
- One additive switch, with ReActXen's own behaviour as its default (`agents/react/agents.py`): `stop_when`, a callable, stops an agent at its first step after it returns true, as the end of an episode of the environment does. Every other setting of the agents is ReActXen's and the AssetOpsBench paper's code.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/utils/model_inference.py` replaces ReActXen's file of that name. It keeps the names and signatures of `watsonx_llm`, `count_tokens` and `get_context_length` and sends the calls to SGLang's raw completion route instead of watsonx.ai.

## IBM AssetOpsBench, branch archive/legacy-tmp

- The source is the branch `archive/legacy-tmp` of https://github.com/IBM/AssetOpsBench at commit 27552e9b58a3b54cc1e48ccac42298e7d037dd7c.
- The license is the Apache License, Version 2.0, the license of this repository, whose full text is in `LICENSE`.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/in_context_examples.py` holds `PAPER_SUPERVISOR`, the supervisor's examples of the five paper-era agents, byte for byte from the FewShots files of `src/tmp/meta_agent/agents/` of that branch, and the eight examples of the IoT agent adapted from `src/tmp/IoTAgent/BMSFewShots.py`.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/sub_agent_registry.py` holds `PAPER`, the descriptions and the request fields of the paper's tools, from the tool files of `src/tmp/meta_agent/agents/` of that branch.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/prompts.py` holds the planner, reviewer and summarizer prompts of AgentHive byte for byte, from `src/tmp/agent_hive/workflows/planning_review.py`, `src/tmp/agent_hive/agents/plan_reviewer_prompt.py` and `src/tmp/agent_hive/agents/summarization_agent.py` of that branch. `CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/tests/data/paper_prompts.json` holds the three prompts as the functions of those files fill them, and its `source` field names the repository, the branch, the commit, the three files and the license.

## IBM AssetOpsBench, branch trajectory_refactor

- The source is the branch `trajectory_refactor` of https://github.com/IBM/AssetOpsBench at commit 097b1d89982e59a256dce764d635cf6815d2de18.
- The license is the Apache License, Version 2.0, the license of this repository, whose full text is in `LICENSE`.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/tests/data/paper_supervisor_prompt.json` holds the fields `system_prompt` and `demonstration` of the sample trajectory `src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large/0001` of that branch, and its `source` field names the branch, the commit, the file and the license.
- `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/in_context_examples.py` holds the six examples of the TSFM agent adapted from the demonstration of the sample trajectory `src/trajectory_analysis/failure_mode/sample_trajectories/claude-4-sonnet/0420` of that branch.

## Granite 3.3 8B Instruct

- The source is the file `tokenizer_config.json` of https://huggingface.co/ibm-granite/granite-3.3-8b-instruct at revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b.
- The license is the Apache License, Version 2.0, the license of the model repository.
- `CausalDiagnosis/agents/utils/tests/data/granite-3.3-8b-instruct_tokenizer_config.json` holds the chat template of that file, and its `_source` field names the file and the revision.
