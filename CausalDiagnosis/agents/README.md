# agents

The agents that play episodes through the tool servers.

## What lives here

| Folder | What it holds |
| --- | --- |
| `Causal_Diagnosis_Agent/` | the CausalDiagnosis agent: the runner of each step, the situation board that records the evidence and the candidate plans, the plan and its checks, the schedule of reads shared by several candidate plans, the binding of each action to a tool, and the prompts |
| `utils/` | what every agent shares: one MCP session per server, the parsing of tool answers, and the table of model adapters keyed by the served model |

The two agents of the AssetOpsBench paper, Plan-and-Execute and Agent-as-Tool, live in `frameworks/AssetOpsBench/agents/`.

## How to use it

An agent plays through the episode runner, for example the CausalDiagnosis agent on the base preset with a model endpoint:

```bash
python -m CausalDiagnosis experiments run-episodes --config base --agent causal --seeds 1 --base-url <endpoint> --model <model>
python -m CausalDiagnosis experiments run-episodes --help
```

`CausalDiagnosis/docs/experiments.md` documents the runner, its options and what it writes.

## How to add to it

A new agent of this project gets its own folder here, a name among the agents of the episode runner, and a list of its files among the agent lists of `experiments/table_definitions/`, so that the code it loads enters the identity of every run it plays.
An agent that comes with another benchmark is declared by that benchmark under `frameworks/`, and no file here changes.
