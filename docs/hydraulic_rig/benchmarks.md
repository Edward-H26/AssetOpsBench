# Hydraulic Rig Drivers and Result Tables

`benchmarks/hydraulic_rig/` holds the episode driver of the hydraulic rig gym, the audit report that every episode writes, the configuration presets and price tables, the pilot scenarios with their anchors, the statistics, the ablation tables, the judge, the prompt optimizer, the publisher of run folders, and the scripts that fetch, rebuild and check the rig data. One command plays episodes with an agent or a reference policy through the two MCP servers and verifies every episode from its log. No result is in the repository. This page covers the runtime, the options of every script, the files a run writes, and the columns of every table.

Related docs: [README.md](README.md) for the overview and the data file, [environment.md](environment.md) for the episode rules, the oracle and the verifier, [action-server.md](action-server.md) for the nine action tools, [causal-diagnosis-server.md](causal-diagnosis-server.md) for the four diagnostics tools, and [../running_benchmark.md](../running_benchmark.md) for the upstream benchmark run.

---

## Quick start

```bash
# 0. the rig data file, fetched and checked as in docs/hydraulic_rig/README.md
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 1. one scripted episode through both servers, without a model
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent scripted --seeds 1

# 2. the oracle policy and the do nothing policy through the action server on the same seeds
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent oracle --seeds 3
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent do_nothing --seeds 3

# 3. the ablation tables from every run folder, with their design and empty cells where no run contributes
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/aggregate.py

# 4. the output
ls benchmarks/hydraulic_rig/runs/ benchmarks/hydraulic_rig/results/aggregate/
```

Every command on this page runs from the fork root with the environment's `bin` first on `PATH`, since the Plan-Execute runner starts every tool server as `python`. A run folder is named after the configuration preset, the agent and the start time, as in `static_scripted_<YYYYmmdd-HHMMSS>`, with the dots and slashes of a model name turned into underscores. It holds `run.json`, `scores.csv`, `summary.md` and one folder per episode with its audit report `report.md` and its hidden files under `hidden/`.

---

## Runtime

The driver needs the environment and both servers, the MCP client and server, the OpenTelemetry API that the Plan-Execute runner imports, and the OpenAI SDK of its OpenAI compatible backend. The slim virtual environment of [README.md](README.md) holds them at the versions `uv.lock` pins, and `uv sync` installs the whole fork instead, in which `uv run python` replaces the `PATH` prefix. Every script of the folder puts `src/` and the folder itself on the import path, so no `PYTHONPATH` is needed. Before the first episode that registers the diagnostics server or looks a grade up, the driver fits `src/servers/causal_diagnosis/models.json` when it is missing, and the first episode under a new price table computes the grade table of that table, which takes a while.

---

## The driver

`run_episodes.py` plays the episodes of one configuration with one agent, verifies every episode from its log, judges the closing summary when asked, and writes the run folder.

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | `static` | a preset of `benchmarks/hydraulic_rig/configs/` (`static`, `timed`, `pilot`, `static_zero_durations`) or the path of a JSON file with the same fields |
| `--agent` | `scripted` | `scripted`, `plan_execute`, `static`, `oracle`, `do_nothing` or `greedy` |
| `--model` | the name in the file | the model name the model agents send to the endpoint |
| `--base-url` | `https://api.tokenrouter.com/v1` | the OpenAI compatible endpoint of the model agents |
| `--seeds`, `--first-seed` | 1, 0 | the number of seeds (per scenario for the pilot) and the first one |
| `--clock-mode`, `--guidance`, `--verifier-scope`, `--no-diagnostics` | the preset's values | overrides of the preset: the clock mode, the guidance level, the verifier scope, and the action server alone |
| `--prices` | the preset's prices | a price table of `benchmarks/hydraulic_rig/configs/prices/` (`placeholder`, `full_tier_high`, `sme`) or a JSON path |
| `--judge` | `none` | `local` runs the AssetOpsBench judge on every closing summary through `judge.py` |
| `--judge-protocol` | `single` | `single`, one call at temperature 0, or `paper`, five calls at temperature 0.3 |
| `--judge-model`, `--judge-base-url` | `--model`, `--base-url` | the judge's model and endpoint. A judge equal to the agent model is refused, since the fork's evaluator refuses self judging |
| `--ranking-model` | the server's default, `joint_block` | `joint_block`, `joint` or `discriminant_urn`, the ranking model of the diagnostics server for ablation A12, recorded in `run.json` and `scores.csv`. The driver and the server share the one default of `src/servers/causal_diagnosis/posterior.py` |
| `--abstention-rule`, `--abstention-cut` | `decision`, 0.95 | the abstention rule of the diagnostics server for ablation A8 and the cut of the `fixed_cut` rule |
| `--no-grade` | off | skips the grade lookup, so every declaration counts as correct |
| `--facts` | `HYD_FACTS_PATH`, else `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` | the rig data file |
| `--out` | `benchmarks/hydraulic_rig/runs` | the folder that receives the run folder, relative to the fork root. A folder inside the fork may not contain a dot, since the runner imports the per episode launchers as modules of that path |

`configure` lays the overrides over the preset and validates the result through the JSON round trip of `EpisodeConfig`, and the whole configuration lands in `run.json`. The diagnostics launcher of every episode names the ranking model the server runs, so the recorded model is the served one, and names the abstention rule with its cut only when the rule differs from `decision`.

| `--agent` | Plays through | What plays |
| --- | --- | --- |
| `scripted` | both servers | the Plan-Execute runner of AssetOpsBench with `ScriptedBackend`, a fixed plan that calls every tool of both servers with fixed arguments: the profile, the failure modes of the pump, one cycle of the cheapest group and one of the full group, the evidence, the ranking, the change summary, a fixed diagnosis, the predicted effect of replacing the pump, a work order on the pump citing that prediction, an advance until the order closes, one more cycle of the cheapest group, the verification of the prediction, a declaration after the diagnosis, which is the one refused call of the plan, and the finish. A wiring check without a model, whose verdict says nothing about diagnosis. With `--no-diagnostics` the diagnostics steps are dropped |
| `plan_execute` | both servers, or the action server with `--no-diagnostics` | the same runner, unchanged, with the model that `--model` names |
| `static` | no server | ablation A3: one model call over the question, the features of the recorded cycle the gym would have served first (read from a throwaway environment at the full sensor group, features only and never a label) and the failure modes. The answer is judged against the oracle's reference answer and parsed for the components it calls faulty and repaired |
| `oracle`, `do_nothing`, `greedy` | the action server | the reference policies of [environment.md](environment.md), each action sent as one tool call through the runner's own stdio helper (`play_policy`). A shadow environment in the driver process is reset with the same seed and configuration, the oracle reads the hidden state from it, and after every call the shadow steps the same action with the thinking cycles the server stored, so the served log is compared with the shadow's (`replay_matches`). The oracle looks the grade of its start up before it plays and declares insufficient evidence instead of diagnosing on an inseparable start, the correct abstention of the verifier. The greedy policy reads through `ServedClassifiers` over `TierClassifiers` fitted once per run |

A Plan-Execute run plans once, executes the plan and summarizes. An episode that the plan leaves open stays open and is verified as it stands, with the count gate failing on the missing finish. A run that raises, such as a plan with a malformed dependency or an API error that outlasts the client's retries, leaves its message in the `error` column, and so does a verifier that cannot read an episode. Before the first episode the driver lists the tools of every registered server once and stops when a server offers other tools than the contract names (`check_servers`).

### Model access

The model agents and the judge read one key, `API_KEY`, from the shell environment. Without it a model agent stops with:

```
API_KEY is not set; configure a key or run with --agent scripted
```

```bash
export API_KEY=<key>
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config static --agent plan_execute --model <model> --base-url <endpoint> --seeds 10
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_episodes.py --config pilot --agent plan_execute --model <model> --base-url <endpoint> --judge local --judge-model <judge> --judge-base-url <judge endpoint>
```

Inside its own process the driver copies `API_KEY` into `TOKENROUTER_API_KEY` and the endpoint into `TOKENROUTER_BASE_URL`, then builds AssetOpsBench's backend for the model id `tokenrouter/<model>` with `make_backend` from `src/llm/`, which sends each prompt to the chat completions of the endpoint through the OpenAI SDK, so `--base-url` accepts any OpenAI compatible endpoint, such as the model served on the cluster. `RecordingBackend` keeps every prompt, reply, token count and duration for `llm_calls.json`, and `run.json` records `model`, `model_id` and `endpoint`. The key never reaches the tool server processes, which the runner starts without an environment of their own. The tests run `plan_execute` and `static` against a local endpoint that answers with the scripted plan, so the real client path runs without a model.

---

## Configuration presets and price tables

A preset under `benchmarks/hydraulic_rig/configs/` lists only the fields it changes, and `load_config` in `common.py` lays them over the defaults of `EpisodeConfig` and validates the result. A field that `EpisodeConfig` does not have is refused.

| Preset | What it sets |
| --- | --- |
| `static.json` | no schedule and no external event: the hidden combination of the start is the only condition of the episode. Guidance level 2, the declared clock, the scope `writes`, the diagnostics server on |
| `timed.json` | the schedule and the external events in mode `seeded`, with `first_drop` ending early enough that a component with three recorded drops at the default interval reaches its worst level while a work order created then still closes before the horizon, after the longest default lead time and repair downtime, with one cycle left for the finish. The same seed gives the same hidden start as `static.json` |
| `pilot.json` | the pilot of the design: anchored episodes with `anchors.date0` the first recorded day, a fault rule that admits any number of faulty components, since an anchored day is not drawn under a rule, and `start` and `question` filled per scenario by the driver |
| `static_zero_durations.json` | the arm of ablation A6 in which every declared duration is zero: reads and advances take no time and work orders close in the cycle they are created, so the clock advances only through `advance_time` |

| Price table under `benchmarks/hydraulic_rig/configs/prices/` | Content |
| --- | --- |
| `placeholder.json` | the placeholder prices of the design, the defaults of `EpisodeConfig` |
| `full_tier_high.json` | ablation A9: the full sensor group priced far above the two cheaper groups, every other price as in the placeholder table |
| `sme.json` | a template for the values of the subject matter experts, every value empty. `load_price_table` refuses it while a value is empty, so the driver stops with the message `empty values` until the session with the experts fills it |

`apply_price_table` replaces the prices, and the durations and the downtime rates when the table gives them, and validates the configuration again. The grade table of the diagnostics server is keyed by these tables, so the first episode under a new table computes its grades.

---

## The pilot

`pilot/scenarios_17.json` holds the hydraulic scenarios of the AssetOpsBench dataset, the `hydrolic_pump` configuration of the dataset's utterance file, as released, with the dataset name, the clone commit, the SHA-256 of the source file and the number of records, so a reader can check the copy. `pilot/build_scenarios.py` rebuilds it from a local clone of the dataset, keeping the one date each scenario text names, which equals the date of its `failure_event` field. `tests/test_anchors.py` compares the file with a rebuild when the clone is present.

`pilot/anchors.py` maps a scenario date to a recording cycle and to the anchored start of an episode. `cycle_of` counts the days from `DATE0`, the first recorded day. `anchor` reads the recorded combination of that day's cycle and its position in the combination's stable block, and a day whose cycle is not stable is anchored at the nearest stable cycle of the same combination, which the result says. `start_of` turns the anchor into the `start` field of `EpisodeConfig`. The mapping rests on two assumptions, `ASSUMPTIONS` of the scenario file, which the mentor has not confirmed: A1, the asset of the scenarios is the `hyd_1` file with one recorded cycle per day from the first recorded day, and A2, the dates were chosen for the faults the texts name, so the hidden condition of an anchored episode is the recorded condition of the day and not the fault the text names. Every pilot `run.json`, `summary.md` and `report.md` prints both.

`--config pilot` plays one episode per scenario under every seed, in folders named `scenario<id>_seed<NNNNNN>`, with the scenario's anchored start and its text as the question.

---

## What a run writes

### The run folder

| File | Content |
| --- | --- |
| `run.json` | the label, the preset name and the whole configuration, the mode (`gym`, `static` or `gepa`), the price table and its source, the agent and its label, the servers, the model, the model id and the endpoint of a model agent, the judge settings, whether grades are looked up, the ranking model, the abstention rule and cut, the clock mode, the guidance level, the diagnostics flag, the verifier scope, the briefing version, the seeds, the number of episodes planned, the pilot note with its assumptions and the anchor of every scenario, the data file path and the prefix of its SHA-256, the fork commit with the SHA-256 prefixes of every environment, server, driver, configuration and pilot file, the Python, `mcp` and `openai` versions, and the start and end times |
| `scores.csv` | one row per episode, with the columns below |
| `summary.md` | the run settings, the counts of episodes, errors and files per episode, the pilot assumptions, and one row per episode with a link to its report. It states no score (`run_summary` in `report.py`) |
| `seed<NNNNNN>/`, `scenario<id>_seed<NNNNNN>/` | one episode folder per episode |

### The episode folder

| File | Written by | Content |
| --- | --- | --- |
| `episode.json` | action server | the state the server replays: the seed, the configuration, the data path, every action with its thinking cycles, every tool call with its answer |
| `released.json` | action server | everything the agent has been told, the only view of the episode that the diagnostics server and the verifier's provenance check read |
| `predictions.json` | diagnostics server | the predictions that `estimate_intervention_effect` stored |
| `hydraulic_rig_server.py`, `causal_diagnosis_server.py` | driver | one launcher per registered server, which binds the server process to this folder |
| `question.txt` | driver | the briefing the action server composed, the question the runner receives |
| `plan.txt`, `answer.txt`, `trajectory.json`, `llm_calls.json` | driver | Plan-Execute and scripted runs: the plan, the closing answer, every step with its server, tool, arguments, response, error and duration, and every model call. The static agent writes `prompt.txt`, `answer.txt` and `llm_calls.json` |
| the runner's trajectory record | Plan-Execute runner | the runner's own record of a completed run, since the driver points `AGENT_TRAJECTORY_DIR` at the episode folder |
| `report.md` | `report.py` | the audit report, from the public files only |
| `hidden/oracle.json`, `hidden/verdict.json`, `hidden/score.json`, `hidden/log.json` | driver, after the episode | the oracle of the episode, the verifier's verdict, the score columns with the reward, the oracle's cost, the grade source and the judge record, and the full log with its hidden state |
| `hidden/judge.json`, `hidden/static.json` | driver | the judge record when the judge ran, and the static agent's answer, parsed sets and scorings |

No agent facing tool reads `hidden/`, and the audit report never opens it. `episode.json` holds the seed, from which a replay recovers the hidden combination, so an agent with file or shell tools has to run with the episode and run folders out of its reach.

### The audit report

`report.md` comes from `episode_report` in `report.py`, which reads the public files of the folder and needs no model. It holds:

1. The preset, the agent and its model, the servers, the seed, the clock mode and the guidance level, and the task text.
2. The pilot assumptions and the anchor of the episode, for a pilot run, and a note on the static mode, for a static run.
3. The plan of a Plan-Execute or scripted run.
4. Every tool call of both servers with its arguments and a one line result, from `trajectory.json` when the runner wrote one, else from the calls the action server recorded.
5. The predictions the diagnostics tools stored.
6. The public end of the episode from `released.json`: the time, the budget and steps left, the work orders created and closed, the refused calls with their reasons, and the end reason, cost and downtime of the end entry.
7. The closing summary, or the static answer.
8. The number of model calls, their tokens and the time spent in the model.
9. A line that the oracle, the verdict and the scores sit under `hidden/`.

### Columns of scores.csv

| Columns | Content |
| --- | --- |
| `run`, `seed`, `scenario_id`, `date`, `config`, `agent`, `model`, `clock_mode`, `guidance_level`, `diagnostics_server`, `prices`, `verifier_scope`, `ranking_model`, `block`, `n_faulty`, `wall_s`, `error` | the run and episode identity and the factors of the ablations: the start combination as a JSON list, its number of faulty components, the wall time in seconds, and the error of a run that raised |
| `pass`, `success`, `separable`, `commitment_on_inseparable`, `diagnosis_all_correct`, `recovered`, `unnecessary`, `cost`, `cost_normalized`, `downtime_total`, `detection_delay`, `repair_delay`, `turns_to_correct_diagnosis`, `abstained`, `abstention_correct`, `prediction_verdict`, `failure_class`, `end_reason`, `time_used`, `steps`, `provenance_ok`, `reward`, `cost_oracle`, `grade`, `judge_pass` | the verifier's verdict of [environment.md](environment.md), with the two delays as the largest over components, the provenance check's flag, the reward, the oracle's cost, the grade and the judge's pass when the judge ran |
| `static_faulty_set_correct`, `static_repairs_match_oracle` | the static agent's parsed faulty set against the recorded start and its parsed repairs against the oracle's work orders |
| `tool_calls`, `refusals`, `causal_calls`, `llm_calls`, `input_tokens`, `output_tokens`, `plan_steps`, `replay_matches` | the bookkeeping: the action server's calls and refusals, the successful calls to the diagnostics server, the model calls and their tokens, the length of the plan, and whether the served log equals the shadow's for a reference policy |

`common.COLUMNS` fixes the order, and `aggregate.py` reads these names only.

---

## The judge

`judge.py` is the second scorer of an episode: the rubric scorer of `src/evaluation/scorers/llm_judge.py`, run through the fork's own `Evaluator` on a scenario file and a trajectory record written to a temporary folder, the only input shape the evaluation pipeline reads. The question is the task text of the episode, the characteristic answer is `reference_answer(oracle)`, which names every component's level and failure mode code, the faulty set, one work order per oracle order closed by its window end, and the restored rig, the answer is the closing summary of the `finish` event, and the trajectory is the released events, which hold nothing hidden.

| Protocol | Calls |
| --- | --- |
| `single` | one call at temperature 0, the default |
| `paper` | five calls at temperature 0.3, as the AssetOpsBench paper reports, with the average score and the pass by the majority of the calls |

`judge_episode(episode_dir, backend, protocol, agent_model, base_url)` writes `hidden/judge.json`, format 1: the protocol, the temperature, the judge and agent models, the task text, the reference answer, the answer, every call's pass, score, rationale, details and error, the pass rate, the mean score and the pass. A call whose scorer result is a backend error, an unparseable reply or a missing reference carries that rationale as its `error` and is left out of the pass rate, since it is a failure of the judge and not of the agent, and the pass is empty when no call produced a parsed review, so the agreement study counts such episodes apart. The Evaluator's rule against self judging stays active, so the judge model has to differ from the agent model.

```bash
export API_KEY=<key>
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/judge.py --model <judge> --base-url <endpoint> [--protocol single|paper] [--agent-model <id>] <episode folder> ...
```

The command line reads the agent model from the run folder's `run.json` unless `--agent-model` is given. `run_episodes.py --judge local` calls the same function after every episode, and the static agent is judged through the same pieces, since a static run has no finish event.

---

## Statistics

`stats.py` holds the statistics behind the tables. Every function takes plain sequences and returns plain numbers.

| Function | Content |
| --- | --- |
| `paired_cluster_bootstrap(a, b, clusters, resamples, seed, level)` | the mean of the paired differences with a percentile interval from a bootstrap that resamples whole clusters, the recorded blocks, since episodes that share a block share most of their cycles, and a two sided add one p value |
| `cluster_bootstrap(statistic, items, clusters, ...)` | any statistic of a list of items with the same clustered interval |
| `holm(p_values, alpha)` | the Holm step down over one family: adjusted p values and rejections |
| `wilson(successes, n)` | the Wilson score interval of a proportion |
| `rmst(turns, observed, cap)` | the restricted mean survival time over turns, the area under the Kaplan Meier curve of the turn of a correct diagnosis with censoring at the step cap |
| `cohens_kappa(a, b)` | the agreement of two verdict sequences, undefined when the chance agreement is one |
| `spearman(x, y)` | the rank correlation with average ranks |

---

## Ablation tables

`aggregate.py` writes one table per ablation A1 to A12 of the design, plus the GEPA table, from the run folders, as CSV and Markdown:

```bash
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/aggregate.py [--runs benchmarks/hydraulic_rig/runs] [--out benchmarks/hydraulic_rig/results/aggregate] [--readme benchmarks/hydraulic_rig/results/README.md] [--published benchmarks/hydraulic_rig/results/runs] [--cluster <name>] [--resamples N] [--alpha A] [--seed S] [--eqa-gold-direction <text>] [--eqa-gold-action <text>]
```

Every table is computed only from the run folders that exist. Without any folder the Markdown states the design of each table, what varies, the hypothesis, the metric and what a result would mean, with the header row and empty cells, and every CSV holds its header alone. Proportions get Wilson intervals, other means a bootstrap interval clustered by recorded block, and differences between the levels of a factor are paired by episode, the same seed, scenario and every other factor, with the p values of a table's primary metric corrected by Holm.

| Table | What varies | Metric |
| --- | --- | --- |
| A1 | the scoring head: the AssetOpsBench judge against the recorded pass on the same trajectories | the two by two table of judge pass against recorded pass, the false pass rate with a Wilson interval, Cohen's kappa with a clustered interval |
| A2 | the gold of the counterfactual and action questions of IndustryAssetEQA on this rig against the recordings | the agreement of the model gold, given as `--eqa-gold-direction` and `--eqa-gold-action`, with the recorded direction of every agent work order and with the recorded need for a work order |
| A3 | static answers against episodes in the gym | the judge pass rate and the parsed faulty set rate of the static answers against the recorded pass rate and the cost of the gym episodes, per model |
| A4 | the diagnostics server on or off | pass, diagnosis accuracy and cost per level, the paired difference against the server off |
| A5 | static against timed episodes with the same hidden start | pass, cost, normalized cost and detection delay, the paired difference against static |
| A6 | the clock mode, with the zero duration arm as its own level | pass, detection delay and repair delay per mode, the paired difference against declared |
| A7 | the verifier scope, writes against all | the pass rate of every agent under both scopes and Spearman's rank correlation of the agents between the scopes |
| A8 | the abstention rule of the diagnostics server, the fixed cut against the decision rule | abstention precision and recall against the grade, with Wilson intervals |
| A9 | the price table | cost, normalized cost and pass per table, the paired difference against the placeholder table |
| A10 | the guidance level | pass per level, the paired difference against level 1 |
| A11 | one against two faulty components at the start | pass and unnecessary replacements per number of faults, no pairing since the levels are different episodes |
| A12 | the ranking model of the diagnostics server | per number of readings, the Brier score of the top candidate's posterior over the `rank_root_causes` calls of the trajectories and the top candidate rate, with `A12_curves.csv` holding the calibration bins |
| GEPA | the untuned planner prompt against the best candidate of the prompt optimizer | the mean validation reward of the seed and the best candidate, the episodes played, and the untuned prompt's mean reward over the same number of episodes |

The outputs under `--out` are `A<k>.csv`, `GEPA.csv`, `A12_curves.csv` and `aggregate.md`. The script also writes the results README under `--readme`: the models and agents, the cluster, the dates and the presets of the runs, every table with its hypothesis and a one sentence reading rule, one example episode with its tool calls and its verdict linked from the published run folders, and the commands that reproduce the runs. Before any run exists the file says so and shows the empty designs.

`publish_runs.py --runs <synced run folders> [--out benchmarks/hydraulic_rig/results/runs] [--llm-cap-bytes N]` writes the subset of every run folder that goes with the results: `run.json` with every endpoint field set to null and every path reduced to its basename, `scores.csv`, `summary.md`, the GEPA result and best prompt of a GEPA run, the aggregate tables beside them, and per episode the report, the trajectory, the released evidence, the question, the plan, the answer, the static prompt, and under `hidden/` the verdict, the oracle, the scores, the static record and the judge record. `llm_calls.json` comes along only up to the cap, and a larger one is replaced by a note. The full log, the state file and the launchers stay out, and `INDEX.md` lists the published runs.

---

## Prompt optimization

`gepa_optimize.py` runs GEPA over the planner prompt of the Plan-Execute runner, the module constant that `Planner.generate_plan` of `src/agent/plan_execute/planner.py` reads at call time. Replacing that constant for the duration of an episode is the override, so no copy of the runner exists, and a candidate has to keep the placeholders the runner fills, else it scores zero with that fact as its feedback. `PlannerPromptAdapter` follows the adapter of the gepa package: `evaluate` plays one episode per metric call through `EpisodeRunner` and returns the verifier's reward as the score, zero when the verifier could not run, and `make_reflective_dataset` hands the reflection model the question, the plan, the closing answer and, as feedback, every check of the verdict with its outcome, the failure class, the end reason, the cost against the oracle's, the reward and the messages of the diagnostics tools in the trajectory.

```bash
export API_KEY=<key>
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/gepa_optimize.py --model <model> --base-url <endpoint> [--config static] [--train-seeds N] [--val-seeds N] [--first-seed S] [--max-metric-calls N] [--reflection-minibatch N] [--no-diagnostics] [--no-grade] [--judge none|local] [--out benchmarks/hydraulic_rig/runs]
```

The budget is `--max-metric-calls`, every episode counted, validation rollouts included. The run folder gets `run.json` with the mode `gepa`, `scores.csv` with one row per episode played, `gepa_result.json` with the candidates, the validation scores of the seed and the best candidate and the budget spent, and `best_prompt.txt`. The reflection model is the same endpoint as the agent, a limitation stated in the module: a prompt optimizer that reflects with the agent's own model measures what that model can teach itself. The gepa package is imported in `main` only and is not a dependency of the fork. The cluster setup script installs it into the cluster environment, and the tests exercise the adapter with a fake episode player.

---

## Reference policies in process

`run_policies.py` plays the four reference policies, `oracle`, `do_nothing`, `random` and `greedy`, in process on seeded episodes of one preset, verifies every episode with the released evidence composed by `released_from_env`, writes one CSV row per policy and seed to `benchmarks/hydraulic_rig/results/policies_<config>.csv`, and prints whether the ordering check of OpenEnv's RFC 008 holds, the oracle's reward above the do nothing policy's on every seed. `policy_table` builds a new policy for every episode, so any row reruns on its own.

```bash
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/run_policies.py [--config static] [--seeds N] [--first-seed S] [--facts PATH] [--out PATH]
```

---

## Data scripts

The fork ships no rig data file. [README.md](README.md) shows how to fetch the IndustryAssetEQA facts file and check its SHA-256, and [environment.md](environment.md) describes the release path and the data checks in full.

| Script | Purpose | Writes |
| --- | --- | --- |
| `verify_data_facts.py` | checks the facts file and the `hyd_1` file of AssetOpsBench against each other | nothing |
| `fetch_uci.py` | downloads dataset 447 of the UCI Machine Learning Repository under CC BY 4.0, with `--download` only and into an empty folder | the archive, its extracted files, and its SHA-256 in `src/envs/hydraulic_rig/data/CHECKSUMS.txt` |
| `build_blocks.py` | recomputes the labels and the served statistics of every cycle and the stable blocks from the raw UCI files, or compares them with the facts file under `--check` | `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` and `src/envs/hydraulic_rig/data/blocks.json` |

The three scripts use the standard library only and put `src/` on the import path themselves, so they also run under a plain `python3 -I`.

---

## Results policy and the cluster

No result ships with the repository. The run folders under `benchmarks/hydraulic_rig/runs/`, the tables under `benchmarks/hydraulic_rig/results/`, the fitted `models.json` and `grades.json` of the diagnostics server and the environment's data folder are listed in `.gitignore`. The runs happen on Columbia's cluster through the Slurm scripts of `CausalDiagnosis/Code/hpc/` in the project repository beside this fork, whose README lists the user's commands in order: the environment setup, the model download, the serving of the agent and judge models, the pilot, the ablation grid as a job array, the prompt optimization, and the copy of the run folders back with `sync_runs.sh`, which publishes their subset with `publish_runs.py` and rebuilds the tables and the results README with `aggregate.py`. Once the runs exist, the results folder with its README and the index of the published runs is added to the repository, and this page and [README.md](README.md) link it. Until then every table states its design with empty cells.

---

## Tests

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH python -m pytest benchmarks/hydraulic_rig -q
```

`tests/conftest.py` puts `src/` and the driver folder on the import path and fits `models.json` once before the tests when the data file exists. The end to end tests write their run folders under pytest's temporary folder, outside the repository, and assert relations, never a recorded value.

| Test file | Covers | Skipped when |
| --- | --- | --- |
| `test_run_episodes.py` | the scripted plan through both servers with every file written, the oracle and the do nothing policy through the action server with the served log equal to the shadow's and the oracle's reward above the do nothing policy's, the oracle's declaration on an inseparable start, the model agent through the OpenAI compatible client against a local stand in endpoint with a judge, the static agent judged and parsed, the pilot anchoring every scenario and printing the assumptions, and the options landing in the configuration with the price template refused | the data file or the mcp package is missing |
| `test_run_policies.py` | every policy plays every seed and is verified, and the oracle outranks the do nothing policy | the data file is missing |
| `test_report.py` | the audit report from synthetic public files shows no hidden value, the one line tool summaries, and the run summary | never |
| `test_judge.py` | both protocols with a fake backend, the reference answer, the refusal of self judging, an unparseable reply as a judge error, and the command line | never |
| `test_stats.py` | every statistic against hand computed expectations | never |
| `test_aggregate.py` | the empty designs without run folders, the paired factor tables with Holm, and the tables that read the hidden files | never |
| `test_configs.py` | the presets and price tables exist, load over the defaults and validate, the template is refused, the launchers bind each server and name the ablation arms only when asked | never |
| `test_anchors.py` | the scenario file, its rebuild from the clone, the date rule, the cycle of a date and the anchor on synthetic cycles and on the facts file | the clone or the data file is missing, for those tests |
| `test_gepa_optimize.py` | the seed candidate, the restored override, the adapter with a fake player, the placeholder guard, the reflective dataset, the feedback text and the result record | never |
| `test_publish_runs.py` | the published subset holds the listed files and no endpoint, the cap on model calls, the scrubbing at any depth | never |
| `test_data_facts.py`, `test_build_blocks.py`, `test_fetch_uci.py` | the data scripts, see [environment.md](environment.md) | the data file is missing, for the tests that read it |
| `test_paths.py` | every repository path that the two folder READMEs, the fork's sections of `docs/mcp-servers.md` and the pages under `docs/hydraulic_rig/` name exists | never |

Without the data file the tests that need it are skipped rather than failed, so a passing run without the file does not exercise the drivers.
