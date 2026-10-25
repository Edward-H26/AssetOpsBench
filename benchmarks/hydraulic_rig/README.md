# Hydraulic rig drivers and studies

This folder holds the episode driver `run_episodes.py`, which plays hydraulic rig episodes with a model agent, a scripted plan, or a reference policy and writes an audit report for every episode. It also holds the two studies in `verify_predictions.py` and `diagnosability.py`, the scripts that fetch, rebuild, and check the rig data, and the scripts that evaluate the reference policies.

Quick start, from the fork root:

```bash
# 1. the rig data file, fetched as in docs/hydraulic_rig/README.md
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 2. one scripted episode through both servers, without a model
uv run python benchmarks/hydraulic_rig/run_episodes.py --scenario S1 --agent scripted --per-record 1 --max-records 1

# 3. the same through run.sh, which pins the runtime and needs no project sync
bash benchmarks/hydraulic_rig/run.sh run_episodes.py --scenario S1 --agent scripted --per-record 1 --max-records 1
```

The model agents read one key, `API_KEY`, and reach TokenRouter by default. `--base-url` and `--model` select another OpenAI-compatible endpoint and model. No model is connected yet.

Run folders go to `benchmarks/hydraulic_rig/runs/` and study tables to `benchmarks/hydraulic_rig/results/`. Git ignores both folders, so no result ships with the repository.

[docs/hydraulic_rig/benchmarks.md](../../docs/hydraulic_rig/benchmarks.md) documents every script, option, and output file of this folder, and [docs/hydraulic_rig/README.md](../../docs/hydraulic_rig/README.md) gives the overview of the environment and both servers.
