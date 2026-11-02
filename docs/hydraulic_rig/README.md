# Hydraulic Rig Gym

This fork of AssetOpsBench adds four parts around the public UCI hydraulic rig recordings.

**Environment.** `src/envs/hydraulic_rig/` turns the recordings into episodes of finding the faults of an asset and restoring it on a clock. Each episode hides one combination of the condition levels of four components and runs three hidden processes on the clock: a degradation schedule, an external replacement by a technician, and a failure rule. The agent pays for sensor readings in price and in time, advances time, submits one diagnosis or one declaration of insufficient evidence, creates work orders that a technician executes after a lead time, and finishes. Every change of the hidden state is answered with the real recordings of the new combination, so nothing is simulated. A verifier reads the log and the hidden state, never the answer text, and gives the scores, the failure class and the reward of policy learning. Gymnasium and OpenEnv wrappers expose the same episodes for training.

**Action server.** `src/servers/hydraulic_rig/` wraps one episode as nine Model Context Protocol (MCP) tools, five reads and four writes, declared in a tool contract shared with the diagnostics server. The agent sees the briefing, the asset profile and the answers to its own calls, and nothing else.

**Diagnostics server.** `src/servers/causal_diagnosis/` offers four read only tools over the evidence an episode has released: the evidence with anomaly scores, the candidate faults ranked under an interventional effect model with a decision rule for insufficient evidence, the predicted effect of a repair with conformal intervals, and the check of a prediction against the recordings after the repair. Every threshold traces to a price, a duration, a downtime rate or the declared error level. The server contains no language model and never touches the hidden state.

**Drivers and result tables.** `benchmarks/hydraulic_rig/` plays episodes with agents and reference policies through the two servers, verifies every episode from its log, judges the closing summary with the AssetOpsBench judge, and aggregates the ablation tables of the paper from the run folders. The runs themselves happen on Columbia's cluster through the Slurm scripts under `CausalDiagnosis/Code/hpc/` of the project repository beside this fork, which has its own README.

| Folder or file | What it holds | Page |
| --- | --- | --- |
| `src/envs/hydraulic_rig/config.py` | `EpisodeConfig`, every parameter of an episode with its default and its validation | [environment.md](environment.md) |
| `src/envs/hydraulic_rig/clock.py`, `src/envs/hydraulic_rig/env.py`, `src/envs/hydraulic_rig/oracle.py` | the clock with its hidden processes, the environment with the nine actions, and the oracle of an episode | [environment.md](environment.md) |
| `src/envs/hydraulic_rig/verifier.py` | the checks, the failure classes, the scores, the reward and pass^k | [environment.md](environment.md) |
| `src/envs/hydraulic_rig/gym_env.py`, `src/envs/hydraulic_rig/openenv_env.py` | the Gymnasium and OpenEnv wrappers | [environment.md](environment.md) |
| `src/envs/hydraulic_rig/policies.py`, `src/envs/hydraulic_rig/greedy.py`, `src/envs/hydraulic_rig/classifiers.py`, `src/envs/hydraulic_rig/start_states.py` | the reference policies, the classifiers of the greedy policy and the start state tables | [environment.md](environment.md) |
| `src/envs/hydraulic_rig/data.py`, `src/envs/hydraulic_rig/knowledge.py` | the loader of the data file and the static knowledge of the rig | [environment.md](environment.md) |
| `src/servers/hydraulic_rig/` | the action server: nine MCP tools over one episode state file, the briefing and the released evidence | [action-server.md](action-server.md) |
| `src/servers/contract.py`, `src/servers/tool_contract_v2.json` | the tool contract of both servers | [action-server.md](action-server.md) |
| `src/servers/causal_diagnosis/` | the diagnostics server: four tools, the fit of its models, the grade table and the calibration report | [causal-diagnosis-server.md](causal-diagnosis-server.md) |
| `benchmarks/hydraulic_rig/run_episodes.py`, `benchmarks/hydraulic_rig/configs/`, `benchmarks/hydraulic_rig/pilot/` | the episode driver, the configuration presets with the price tables, and the pilot scenarios with their anchors | [benchmarks.md](benchmarks.md) |
| `benchmarks/hydraulic_rig/stats.py`, `benchmarks/hydraulic_rig/aggregate.py`, `benchmarks/hydraulic_rig/judge.py`, `benchmarks/hydraulic_rig/gepa_optimize.py` | the statistics, the ablation tables, the judge and the prompt optimizer | [benchmarks.md](benchmarks.md) |
| `benchmarks/hydraulic_rig/fetch_uci.py`, `benchmarks/hydraulic_rig/build_blocks.py`, `benchmarks/hydraulic_rig/verify_data_facts.py` | the scripts that fetch, rebuild and check the rig data | [environment.md](environment.md) |
| `examples/hydraulic_rig/rollout.py` | one rollout of the oracle policy and one of the do nothing policy through the gym | [environment.md](environment.md) |
| `CausalDiagnosis/Code/hpc/` | the Slurm scripts that set up the fork, serve the model and run the pilot and the ablation grid on Columbia's cluster, in the project repository beside this fork | its own README |

## Quick start

Run every command from the fork root. The commands use a slim virtual environment at `.venv` that holds the packages the rig imports at the versions `uv.lock` pins. `uv sync` installs the whole fork instead, and `uv run python` then replaces the `PATH` prefix of every command.

```bash
# 0. the slim environment: numpy, scipy, pandas, pydantic, mcp, fastmcp, openai, pytest and the small packages the runner imports
uv venv --python 3.12 .venv
uv export --frozen --no-hashes --no-emit-project --format requirements.txt \
  | grep -E "^(numpy|scipy|pandas|pydantic|mcp|fastmcp|openai|pyyaml|requests|pendulum|python-dotenv|opentelemetry-api|pytest|anyio)==" \
  | sed 's/ ;.*//' > /tmp/requirements-slim.txt
uv pip install --python .venv/bin/python -r /tmp/requirements-slim.txt

# 1. the rig data file: the IndustryAssetEQA facts file of the public EmbodiedQA repository, at a fixed commit
mkdir -p ~/rig-data
curl -L -o ~/rig-data/hyd_facts.jsonl \
  https://raw.githubusercontent.com/ChathurangiShyalika/EmbodiedQA/d7f42a703751cba077d99bae34e8c5fa4bcfa9fe/data/outputs/hydraulic/hyd_facts.jsonl
echo "7912b1b641254d54daffa40ea2fe84c64dedf50edb2d996951ad1f3d9a433e47  $HOME/rig-data/hyd_facts.jsonl" | shasum -a 256 -c
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 2. the tests of the environment, both servers and the drivers
PATH=$PWD/.venv/bin:$PATH python -m pytest src/envs/hydraulic_rig src/servers/hydraulic_rig src/servers/causal_diagnosis benchmarks/hydraulic_rig -q

# 3. one scripted episode through both servers, without a model
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent scripted --seeds 1

# 4. one rollout of the oracle policy and one of the do nothing policy through the gym
PATH=$PWD/.venv/bin:$PATH python examples/hydraulic_rig/rollout.py --seed 0
```

`PATH` has to start with the environment's `bin`, since the Plan-Execute runner starts every tool server as `python`. The checksum line prints `OK` when the data file matches. Without the data file, the tests that need it are skipped rather than failed, so a passing run without the file does not exercise the environment. The scripted episode writes a run folder under `benchmarks/hydraulic_rig/runs/` with `run.json`, `scores.csv`, `summary.md` and one episode folder that holds the public files of the episode, its audit report `report.md`, and under `hidden/` the oracle, the verdict, the scores and the full log. The rollout prints which of the two policies earned the higher reward and no value.

## Pages

- [environment.md](environment.md) describes the data file and its rebuild from the UCI archive, `EpisodeConfig`, the clock with its three modes and its hidden processes, the nine actions with their refusals, the log and the information barrier, the oracle, the reference policies, the verifier with its checks, failure classes, scores and reward, and the Gymnasium and OpenEnv wrappers.
- [action-server.md](action-server.md) describes how a runner starts the action server and binds it to one episode, the state file and the released evidence, the briefing with its three guidance levels, the nine tools with their parameters, classes, prices and refusals, the tool contract, and the clock modes as the server charges them.
- [causal-diagnosis-server.md](causal-diagnosis-server.md) describes the design rules, the models file and the grade table, the four tools with their results, the formulas E1 to E19 with the module that implements each, the ranking models and the abstention rules, the calibration report, and how to run and test the server.
- [benchmarks.md](benchmarks.md) describes the episode driver with every option and agent, the configuration presets and price tables, the pilot with its two stated assumptions, the files a run writes, the statistics, the ablation tables A1 to A12, the judge and the prompt optimizer.

The tool tables of both servers also appear beside the other AssetOpsBench servers in [docs/mcp-servers.md](../mcp-servers.md).

## Data and results

The fork ships no rig data file. The code reads the recorded cycles from the file that `HYD_FACTS_PATH` names, or from `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` when the variable is unset, the file that the release path of [environment.md](environment.md) rebuilds from the CC BY 4.0 archive. The diagnostics server reads its models from `src/servers/causal_diagnosis/models.json` and its grade table from `src/servers/causal_diagnosis/grades.json`, both generated from the data file on first use and ignored by git.

Run folders under `benchmarks/hydraulic_rig/runs/` and tables under `benchmarks/hydraulic_rig/results/` are ignored by git as well, so no result ships with the repository. The run folders and the result tables are produced by the planned runs on Columbia's cluster and will be published under `benchmarks/hydraulic_rig/results/` with a README once they exist. Until then no value from any run appears in the fork.

## Model access

The model agents of the driver, `plan_execute` and `static`, and the judge read one key, `API_KEY`, from the shell environment. By default they reach TokenRouter at `https://api.tokenrouter.com/v1` with the model that `--model` defaults to in `benchmarks/hydraulic_rig/run_episodes.py`, and `--base-url` and `--model` select another OpenAI compatible endpoint and model, such as the model served on the cluster. `--judge-model` and `--judge-base-url` select the judge, which has to be another model than the agent, since the fork's evaluator refuses self judging. The scripted agent and the reference policies need no key.
