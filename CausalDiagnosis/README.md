# CausalDiagnosis

CausalDiagnosis is an industrial environment for language-model agents, built inside a fork of IBM's AssetOpsBench.
An episode hides one combination of fault levels of the components of an asset, today the hydraulic test rig of UCI dataset 447, and the agent has to find the faults and restore the asset by calling tools that cost money and take time.
Every reading the agent receives is a recorded cycle of the real rig, and a verifier scores the episode from its log and the hidden state.
The same episodes reach model agents through Model Context Protocol (MCP) tool servers and reach training code through Gymnasium and OpenEnv wrappers.
A separate tool server, the CausalDiagnosis MCP, gives read-only diagnostic advice, so every agent can be compared with and without that advice.

## How the parts fit

An episode is built from left to right:

dataset (`datasets/`) → environment (`environment/`) → tool servers (`mcp_servers/`) → agent (`agents/`) → runs (`experiments/`) → scores and tables (`evaluation/`)

The dataset folder describes an asset as data, the environment turns that description into episodes, the tool servers expose an episode as tools, an agent plays it, the experiment code decides which runs make up each table and records them, and the evaluation code turns finished runs into tables.
Training (`training/`) and the episode viewer (`episode_viewer/`) read the same episodes and runs.
Another benchmark connects through `frameworks/`: it declares what it brings (agents, tool servers, a judge, scenario sets, commands), and the code of this fork finds it by name without importing it.
AssetOpsBench itself is connected that way, and on the rows of the paper's tables a model agent calls AssetOpsBench's own iot, fmsr and wo servers behind the metering layer of CausalDiagnosis.

| Folder | What lives there |
| --- | --- |
| `datasets/` | one folder per dataset (its `asset.json`, data script and tests), the shared loader that reads any dataset, and the rules that split starts into development, training and test sets |
| `environment/` | the episode engine: hidden faults, the six layers, the clock, the twelve actions, the tool contract, the verifier, the Gymnasium and OpenEnv wrappers, and the reference policies that use no model |
| `mcp_servers/` | the action server, which serves the twelve actions of an episode as MCP tools, and the CausalDiagnosis MCP, four read-only advice tools with the programs that fit their models offline |
| `agents/` | the CausalDiagnosis agent and the helpers every agent shares |
| `frameworks/` | other benchmarks, each in its own folder behind a declaration; `frameworks/AssetOpsBench/` holds its paper agents, tool servers, judge, scenarios and reproduction |
| `experiments/` | which runs make up each table, the episode runner, the job lists of the cluster, and the checks that a run is real and can be replayed |
| `evaluation/` | the metrics and the programs that build the result tables |
| `training/` | rejection-sampling data, LoRA fine-tuning and the online prompt optimization loop |
| `episode_viewer/` | pages that show one episode to a person |
| `configs/` | the presets of episodes, grouped by kind, with the price tables and the training settings |
| `cluster_jobs/` | the Slurm scripts, in folders numbered by run order, with a runbook in their README |
| `docs/` | the reference pages: environment, both servers, experiments, metrics, dataset, adding a dataset |
| `examples/` | one episode in a few lines, without a model |
| `results/` | the tables, written by programs and never by hand; run folders go to `results/runs/` and are not tracked |
| `utils/` | small helpers every folder may use, among them `paths.py`, the one file that knows where things are |

Tests sit in a `tests/` folder beside the code they test.
Each folder has a README that says what lives there, how to use it and how to add to it.

## Quick start

Run every command from the repository root. The commands need `uv`, whose install line stands under Prerequisites in `INSTRUCTIONS.md` at the root.

```bash
# 0. a slim environment with the packages the environment imports, at the versions uv.lock pins
uv venv --python 3.12 .venv
PINS=$(mktemp)
uv export --frozen --no-hashes --no-emit-project --format requirements.txt > "$PINS"
uv pip install --python .venv/bin/python -c "$PINS" \
  numpy scipy pandas pydantic mcp fastmcp openai pyyaml requests pendulum python-dotenv \
  opentelemetry-api pytest anyio couchdb3 langchain-core jinja2
export PATH=$PWD/.venv/bin:$PATH

# 1. the data of hyd_1: fetch the UCI archive from its official page, check it and build the data file
python -m CausalDiagnosis datasets download hyd_1 --download

# 2. one scripted episode through both tool servers, without a model
python -m CausalDiagnosis experiments run-episodes --config static --agent scripted --seeds 1

# 3. one episode of the oracle policy and one of the do-nothing policy through the Gymnasium wrapper
python -m CausalDiagnosis examples run-one-episode --seed 0
```

Step 1 ends with a line that starts with `DATA_OK` when every downloaded and built file has its pinned SHA-256.
Step 2 writes a run folder under `results/runs/` with `run.json`, `scores.csv`, `summary.md`, the run ledger `ledger.json` and one episode folder.
Step 3 prints which of the two policies earned the higher reward.
`uv sync` installs the whole fork instead of the slim environment.

The tool servers need no manual start. The runner writes a launcher file beside the files of each episode, and the launcher binds one server process to that episode.
`uv run causal-diagnosis-mcp-server` starts the CausalDiagnosis MCP without an episode, and then every tool answers with an error.

## Commands

Every program a person starts has one command: `python -m CausalDiagnosis <group> <command> [arguments]`.
A group is a folder name, and a command is the program's file name with `-` in place of `_`.
`python -m CausalDiagnosis` lists the groups, `python -m CausalDiagnosis <group>` lists the commands of a group, and `--help` after a command prints that program's own options.
The table only starts the program, so `python -m CausalDiagnosis.<folder>.<module>` works the same way.

| Group | Command | What it does |
| --- | --- | --- |
| tables | `list` | the name of every result table and what varies in it |
| tables | `<table name>` | build one table alone into a folder outside results/ (--runs, --out) |
| tables | `all` | build every table, the index and the pages |
| datasets | `download` | fetch the raw data of a dataset and build its data file (the dataset id first, such as hyd_1) |
| datasets | `new-dataset-scaffold` | write a first asset.json for a new dataset of recorded levels |
| datasets | `build-features-and-blocks` | recompute the per-cycle features and stable blocks of hyd_1 from its raw files |
| datasets | `verify-data-facts` | check the data facts of docs/environment.md on the hyd_1 files |
| environment | `diagnosability-label` | fill the diagnosability label of every start with the greedy reference policy |
| mcp_servers | `fit-models` | fit models.json of the CausalDiagnosis MCP |
| mcp_servers | `diagnosability-grade` | grade every recorded fault combination under a budget (grades.json) |
| mcp_servers | `offline-table` | the offline table of the CausalDiagnosis MCP |
| mcp_servers | `calibration-report` | the calibration report of the ranking model |
| agents | `model-adapters` | rewrite the chat template of the served model before serving |
| experiments | `run-episodes` | play episodes with an agent or a reference policy and verify each one |
| experiments | `run-reference-policies` | play the reference policies in process, without the servers |
| experiments | `grid-job-lists` | write, check and query the job lists of the tables |
| experiments | `precompute-grid-tables` | compute the tables a job list reads before its lines start |
| experiments | `pending-grid-lines` | the lines of a job list that lack a finished run |
| experiments | `cluster-run-checks` | the checks of the cluster run order |
| experiments | `write-training-split` | write the registered training split of a dataset |
| experiments | `run-ledger` | the ledger of a run folder, the platform record, the record check and the serving file of record |
| experiments | `replay-check` | replay a finished run on a clone of its commit and certify it |
| experiments | `verify-results-folder` | check that the tables of a results folder follow from their runs |
| experiments | `recompute-verdicts` | recompute the verdict of every episode under a runs folder |
| experiments | `rescore-under-verifier-scope` | score finished runs again under the verifier scope all |
| experiments | `invariant-digests` | digests of every text an agent reads and of the invariants of the environment |
| experiments | `check-grid-complete` | check that a finished job list ran in full |
| experiments | `check-synced-runs` | the laptop side of a sync of run folders from the cluster |
| experiments | `upload-manifest` | the manifest of an upload to the cluster and the commit it matches |
| experiments | `publish-check` | check run folders and results folders before they go to the branch of the runs |
| evaluation | `leaderboard` | the leaderboard page |
| evaluation | `difficulty-table` | the difficulty table of the datasheet |
| evaluation | `datasheet-statistics` | the counts and statistics of docs/dataset.md |
| evaluation | `publish-runs` | copy the readable part of the runs the tables read into one folder |
| training | `rejection-sampling-examples` | select the good episodes and write the fine-tuning examples |
| training | `train-lora` | LoRA fine-tuning of the agent model |
| training | `gepa-prompt-optimization` | the online GEPA loop over one block of prompt text |
| episode_viewer | `walkthrough` | the illustrated walkthrough of one episode |
| episode_viewer | `step-through-page` | the step-through page of one episode, one slide per tool call |
| episode_viewer | `run-review-pack` | the review pack of a run folder and the fixed starts of the walkthroughs |
| examples | `run-one-episode` | play one episode of the oracle and of the do-nothing policy, no model needed |
| utils | `paths` | print a location of the repository by its name |
| assetopsbench | `episode-judge` | The second scorer of an episode: the AssetOpsBench judge on the trajectory and the closing answer of the runner (CausalDiagnosis/docs/experiments.md). usage, from the fork root: python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge --model MODEL [--base-url URL] [--protocol single|paper] [--agent-model ID] EPISODE_DIR [EPISODE_DIR ...] The judge is the rubric scorer of src/evaluation/scorers/llm_judge.py, run through the fork's own Evaluator rather than a copy, on the input of the judge of AssetOpsBench: the scenario's question is the episode's task text, its characteristic answer is composed from the hidden oracle, the trajectory is trajectory.json, the list of the tool calls that the runner of the agent keeps, one shape for every agent, and the final response is the closing answer of the runner, answer.txt (the registered judge.record) |
| assetopsbench | `judge-finished-runs` | The AssetOpsBench judge over the episodes of finished runs whose judge did not answer, with scores.csv refreshed. A run's judge column stays empty when the judge could not run at all while the episode ended (run_episodes.run_judge records the failure instead of failing the episode), or when the run started without the in-run judge, and it reads False when every call of the judge failed, as when the endpoint was down, since a failed call counts as false (judge.summary_of) |
| assetopsbench | `download-scenarios` | Fetch the hydraulic scenarios of the AssetOpsBench dataset into the data folder of the asset (CausalDiagnosis/docs/experiments.md). usage: python -m CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios [--out PATH] [--from-file PATH] The scenarios are the records of data/asset/hydrolicpump_utterance.jsonl (spelled as in the dataset) of the dataset ibm-research/AssetOpsBench on Hugging Face, taken at the pinned revision REVISION |
| assetopsbench | `latency-probe` | Latency probe of the metering layer: what one metered call costs before any agent runs. A Plan-Execute call starts a Python process, imports AssetOpsBench's servers, replays the episode and calls one tool, so the probe times each part on the laptop with the memory store: a bare interpreter start, the start with the three server imports, every tool of the base row over a hyd_1 store of recorded cycles, and the replay of the rig engine for a growing number of stored actions |
| assetopsbench | `run-scenarios` | python -m run_scenarios: calls main of the submodule that holds it. |
| assetopsbench | `score-runs` | Score AssetOpsBench reproduction runs two ways (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md): the paper's rubric judge through the fork's Evaluator under two protocols, and the efficiency counts the driver recorded. usage, from the fork root with the fork's virtual environment first on PATH: python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.score_runs RUN_DIR [RUN_DIR ...] [--scenarios FILE] [--judge-model NAME --judge-base-url URL] [--judge-protocols paper single] [--workers 4] [--rejudge-failed] Judge |
| assetopsbench | `fetch-public-servers-and-data` | Layer B of the AssetOpsBench reproduction (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md): the paper-era ports over the servers and the data of IBM's public branches |
| assetopsbench | `acceptance-check` | The acceptance check of the two paper-era ports on AssetOpsBench's own scenarios: the Agent-As-Tool port (agent_as_tool_paper) and the Plan-Execute port (plan_execute_paper). usage, from the fork root with the fork's virtual environment first on PATH, after score_runs.py judged the runs: python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check RUN_DIR [RUN_DIR ...] [--port agent_as_tool_paper|plan_execute_paper] [--out DIR] A port is checked against the paper's values of its own design on runs that played it on the paper's scenarios, granite-3-3-8b-instruct on SGLang with the paper's decoding |

## Adding to it

| To add | Start from |
| --- | --- |
| a dataset | `datasets/README.md` and `docs/adding-a-dataset.md` |
| a layer of the environment | `environment/README.md` |
| an agent | `agents/README.md` |
| a metric or a table | `evaluation/README.md` |
| a preset | `configs/README.md` |
| another benchmark | `frameworks/README.md` |

## Tests

```bash
python -m pytest src CausalDiagnosis benchmarks -q -p no:cacheprovider
```

These are the roots that the suite job of the cluster collects. The tests need the data file of hyd_1.
