# Hydraulic rig drivers and result tables

This folder holds the episode driver `run_episodes.py`, which plays hydraulic rig episodes with a model agent, a scripted plan or a reference policy through the two MCP servers and verifies every episode from its log, the configuration presets and price tables under `configs/`, the pilot scenarios with their anchors under `pilot/`, the statistics in `stats.py`, the ablation tables of `aggregate.py`, the judge adapter `judge.py`, the prompt optimizer `gepa_optimize.py`, the publisher of run folders `publish_runs.py`, and the scripts that fetch, rebuild and check the rig data.

Quick start, from the fork root with the fork's virtual environment first on `PATH`:

```bash
# 1. the rig data file, fetched as in docs/hydraulic_rig/README.md
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 2. one scripted episode through both servers, without a model
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent scripted --seeds 1

# 3. the oracle policy and the do nothing policy through the action server, the ordering check of OpenEnv RFC 008
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent oracle --seeds 3
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent do_nothing --seeds 3
```

The model agents and the judge read one key, `API_KEY`, and reach TokenRouter by default. `--base-url` and `--model` select another OpenAI compatible endpoint and model, and `--judge-model` with `--judge-base-url` a judge other than the agent model.

Run folders go to `benchmarks/hydraulic_rig/runs/` and tables to `benchmarks/hydraulic_rig/results/`. Git ignores both folders, so no result ships with the repository. The runs happen on Columbia's cluster through the scripts of `CausalDiagnosis/Code/hpc/` in the project repository, and the published subset of their run folders with the tables and the results README lands under `benchmarks/hydraulic_rig/results/` once they exist.

[docs/hydraulic_rig/benchmarks.md](../../docs/hydraulic_rig/benchmarks.md) documents every script, option and output file of this folder, and [docs/hydraulic_rig/README.md](../../docs/hydraulic_rig/README.md) gives the overview of the environment and both servers.
