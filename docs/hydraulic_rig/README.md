# Hydraulic Rig Replay Environment

This fork of AssetOpsBench adds four parts.

**Environment.** The environment turns the public UCI hydraulic rig recordings into episodes of finding a fault and repairing the asset. Each episode hides one combination of the condition levels of four components. The agent pays to read sensors at three tiers, submits a diagnosis first, and replaces components afterward. After a replacement, the environment continues with the real recordings of the repaired combination. The score comes from the log alone: whether the diagnosis was right, whether the asset recovered, and what was spent.

**Action server.** The action server wraps one episode as five MCP tools that read sensors, submit a diagnosis, declare the evidence insufficient, replace a component, and finish. The agent sees only the measurements it has paid for.

**Causal diagnosis server.** The causal diagnosis server offers seven tools that help an agent reason: asset information, failure modes, the symptom report that opens the episode, the released evidence, root causes ranked by probability, the predicted effect of replacing a component, and a check of that prediction against the recordings after the replacement. It never touches the hidden answer and contains no language model.

**Drivers and studies.** The drivers play an episode with one command, as a wiring check without a model or with the Plan-Execute agent of AssetOpsBench, and write an audit report for every episode. Two studies are implemented, and their results are not in the repository.

| Folder | What it holds | Page |
| --- | --- | --- |
| `src/envs/hydraulic_rig/` | The environment: data loading, episode rules, scoring, and the reference policies with their classifiers | [environment.md](environment.md) |
| `src/servers/hydraulic_rig/` | The action server: five MCP tools over one episode | [action-server.md](action-server.md) |
| `src/servers/causal_diagnosis/` | The causal diagnosis server: seven MCP tools and the fitting of their models | [causal-diagnosis-server.md](causal-diagnosis-server.md) |
| `benchmarks/hydraulic_rig/` | The episode driver, the audit reports, the two studies, the data scripts, and the evaluation scripts of the reference policies | [benchmarks.md](benchmarks.md) |

## Quick start

Run every command from the fork root.

```bash
# 0. dependencies
uv sync

# 1. the rig data file: the IndustryAssetEQA facts file of the public EmbodiedQA repository, at a fixed commit
mkdir -p ~/rig-data
curl -L -o ~/rig-data/hyd_facts.jsonl \
  https://raw.githubusercontent.com/ChathurangiShyalika/EmbodiedQA/d7f42a703751cba077d99bae34e8c5fa4bcfa9fe/data/outputs/hydraulic/hyd_facts.jsonl
echo "7912b1b641254d54daffa40ea2fe84c64dedf50edb2d996951ad1f3d9a433e47  $HOME/rig-data/hyd_facts.jsonl" | shasum -a 256 -c
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 2. tests of the environment, both servers, and the drivers
uv run pytest src/envs/hydraulic_rig src/servers/hydraulic_rig src/servers/causal_diagnosis benchmarks/hydraulic_rig -q

# 3. one scripted episode through both servers, without a model
uv run python benchmarks/hydraulic_rig/run_episodes.py --scenario S1 --agent scripted --per-record 1 --max-records 1
```

The checksum line prints `OK` when the file matches. Without the data file, the tests that need it are skipped rather than failed, so a passing run without the file does not exercise the environment. The scripted episode writes a run folder under `benchmarks/hydraulic_rig/runs/` with `run.json`, `scores.csv`, `summary.md`, and one folder per episode that holds its audit report `report.md`.

## Pages

- [environment.md](environment.md) describes the data file and its rebuild from the UCI archive, the episode rules and prices, the scorer, the scenario classes, and the reference policies.
- [action-server.md](action-server.md) describes how a runner starts the server and binds it to one episode, the episode files, the briefing that opens an episode, and the five tools with their answers and refusals.
- [causal-diagnosis-server.md](causal-diagnosis-server.md) describes the design rules, the models file, the contract and results of the seven tools, the methods behind them, and how to run and test the server.
- [benchmarks.md](benchmarks.md) describes the two runtimes, the driver with its agents and options, the files a run writes with the audit report, the two studies, and the other scripts of the folder.

The tool tables of both servers also appear beside the other AssetOpsBench servers in [docs/mcp-servers.md](../mcp-servers.md).

## Data and results

The fork ships no rig data file. The code reads the recorded cycles from the file that `HYD_FACTS_PATH` names, or from `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` when the variable is unset. The causal diagnosis server reads its models from `src/servers/causal_diagnosis/models.json`. The drivers and the tests fit that file from the data file on first use, `uv run python -m servers.causal_diagnosis.fit` refits it, and git ignores it. Run folders under `benchmarks/hydraulic_rig/runs/` and result tables under `benchmarks/hydraulic_rig/results/` are ignored by git as well, so no result ships with the repository. No result is reported until the planned runs on Columbia's HPC cluster.

## Model access

The model agents read one key, `API_KEY`. By default they reach TokenRouter at `https://api.tokenrouter.com/v1` with the model `MiniMax-M3`, and `--base-url` and `--model` select another OpenAI-compatible endpoint and model. No model is connected yet, so the quick start uses the scripted agent, which needs no key.
