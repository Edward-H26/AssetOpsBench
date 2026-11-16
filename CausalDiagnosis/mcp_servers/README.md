# mcp_servers

The tool servers that expose an episode to an agent through the Model Context Protocol (MCP). Each server is a standalone process bound to one episode.

## What lives here

| Folder | What it holds |
| --- | --- |
| `Environment_Actions_MCP/` | the action server: the twelve actions of the environment as MCP tools over one episode state file |
| `Causal_Diagnosis_MCP/` | the CausalDiagnosis MCP: four read-only tools that list the evidence, rank the possible fault combinations, predict the effect of a repair and check an outcome; it contains no language model and never opens the hidden state |
| `Causal_Diagnosis_MCP/offline/` | the programs that prepare the CausalDiagnosis MCP before any episode: the fitted models, the diagnosability grades, the calibration report and the offline table |
| `utils/` | the transcript of the calls of an episode, shared by both servers |

## How to use it

The servers need no manual start. `python -m CausalDiagnosis experiments run-episodes` writes one launcher per server beside the files of each episode, and the launcher binds the server process to that episode.
The offline programs run by hand or on the cluster:

```bash
python -m CausalDiagnosis mcp_servers fit-models
python -m CausalDiagnosis mcp_servers diagnosability-grade --help
python -m CausalDiagnosis mcp_servers calibration-report --help
python -m CausalDiagnosis mcp_servers offline-table --help
```

`CausalDiagnosis/docs/environment-actions-mcp.md` and `CausalDiagnosis/docs/causal-diagnosis-mcp.md` document both servers tool by tool.

## How to add to it

A tool of the action server is declared in the tool contract of the environment, `environment/Causal_Diagnosis_Environment/tool_contract.json`, which gives its class, price and duration. The text of every tool reaches the agent, so a change to it changes what the agent reads.
The tool servers of another benchmark belong to its folder under `frameworks/`.
