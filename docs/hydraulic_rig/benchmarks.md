# Hydraulic Rig Drivers and Studies

`benchmarks/hydraulic_rig/` holds the episode driver of the hydraulic rig environment, the audit report that every episode writes, two studies, the scripts that fetch and check the rig data, and the scripts that evaluate the reference policies. One command plays an episode, either as a wiring check without a model or with the Plan-Execute agent of AssetOpsBench. The code of both studies is in the folder, and no result is in the repository. This page covers the runtimes, the options of every script, the files a run writes, and the columns of every table.

Related docs: [README.md](README.md) for the overview and the data file, [environment.md](environment.md) for the episode rules and the scorer, [action-server.md](action-server.md) for the five action tools, [causal-diagnosis-server.md](causal-diagnosis-server.md) for the seven diagnosis tools, [../running_benchmark.md](../running_benchmark.md) for the upstream benchmark run.

---

## Quick start

```bash
# 0. the rig data file, fetched and checked as in docs/hydraulic_rig/README.md
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 1. the runtime: sync the project environment, or skip this step and use run.sh
uv sync

# 2. one scripted episode through both servers, without a model
uv run python benchmarks/hydraulic_rig/run_episodes.py --scenario S1 --agent scripted --per-record 1 --max-records 1

# 2b. the same episode in the pinned runtime, without a project sync
bash benchmarks/hydraulic_rig/run.sh run_episodes.py --scenario S1 --agent scripted --per-record 1 --max-records 1

# 3. the output: one run folder per command
ls benchmarks/hydraulic_rig/runs/
```

Every command on this page runs from the fork root. A run folder is named after the scenario, the agent, and the start time, as in `S1_scripted_<YYYYmmdd-HHMMSS>`. It holds `run.json`, `scores.csv`, `summary.md`, and one `seed<NNNNNN>/` folder per episode with its audit report `report.md`.

---

## Runtime

The driver needs the MCP client and server, the OpenTelemetry API that the Plan-Execute runner imports, and the OpenAI SDK of its OpenAI-compatible backend. Two runtimes provide them, and every script of the folder runs in either.

| Runtime | Setup | Command form |
| --- | --- | --- |
| Project environment | `uv sync` once | `uv run python benchmarks/hydraulic_rig/<script>.py [options]` |
| Pinned runtime | none | `bash benchmarks/hydraulic_rig/run.sh <script>.py [options]` |

`run.sh` wraps one uv command. It runs the named script with the remaining options in a temporary environment built from `requirements-smoke.txt`, without syncing the project:

```bash
uv run --no-project --python 3.12 --with-requirements benchmarks/hydraulic_rig/requirements-smoke.txt python benchmarks/hydraulic_rig/<script>.py [options]
```

`requirements-smoke.txt` lists the closure of `mcp`, `opentelemetry-api`, and `openai` at the versions that the fork's `uv.lock` pins, among them `mcp` 1.26.0, `openai` 2.31.0, and `opentelemetry-api` 1.39.1. Every script that imports the environment puts `src/` on the import path itself, so neither runtime needs `PYTHONPATH`. The two studies also need `pydantic` for the result models of the causal diagnosis server. The data scripts and the reference policy scripts use the standard library only, so they also run under a plain `python3 -I`.

---

## Scenarios and agents

`run_episodes.py` plays the episodes of one scenario with one agent, scores every episode from its log, and writes the run folder.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scenario` | `S1` | `S1`, `S2`, or `seeded` |
| `--agent` | `scripted` | who plays, one of the agents below |
| `--per-record` | every start state | `S1` and `S2`: the first K start states of every record |
| `--max-records` | every record | `S1` and `S2`: the first N records in sorted order |
| `--episodes` | `5` | `seeded`: the number of seeds |
| `--first-seed` | `0` | `seeded`: the first seed |
| `--model` | `MiniMax-M3` | the model name that the model agents send to the endpoint |
| `--base-url` | `https://api.tokenrouter.com/v1` | the OpenAI-compatible endpoint of the model agents |
| `--facts` | `HYD_FACTS_PATH`, else `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` | the rig data file |
| `--out` | `benchmarks/hydraulic_rig/runs/` | the folder that receives the run folder, relative to the fork root |

`S1` holds the 10 combinations with exactly one faulty component and `S2` the 37 with exactly two, one record per combination. In both, the briefing tells the agent the exact fault count, and the task text that `get_symptom_report` returns states it in words. `seeded` keeps the environment's default rule of one or two faulty components and lets each seed draw the combination, so its rows carry no record.

A start state is a combination, a window of its stable block, and an entry position in that window (`start_state_table` in `benchmarks/hydraulic_rig/scenario_anchors.py`). Each start state runs from the smallest seed whose `reset` reaches it, so every episode replays from its seed.

| `--agent` | Plays through | What plays |
| --- | --- | --- |
| `scripted` | both servers | the Plan-Execute runner of AssetOpsBench with a fixed fifteen-step plan that calls every tool of both servers with fixed arguments, a wiring check without a model |
| `plan_execute` | both servers | the same runner, unchanged, with the model that `--model` names |
| `plan_execute_rig_only` | the action server | the same runner without the diagnosis tools, the control |
| `random`, `greedy`, `greedy_prior_1to2` | the action server | the reference policies of `run_policies.py`, each action sent as one tool call through the runner's own stdio helper |
| `oracle` | no server | the oracle policy in process, since it reads the hidden state, which no tool releases |

The scripted plan fixes the diagnosis and the repair as well, so its score says nothing about diagnosis. It checks that every tool answers through the runner, that `verify_outcome` reads back the prediction that `estimate_intervention_effect` stored, and that a second submission after the diagnosis is refused.

A Plan-Execute run plans once, executes the plan, and summarizes. An episode that the plan leaves open stays open and is scored as it stands. A run that raises, such as a plan with a malformed dependency or an API error that outlasts the client's retries, leaves its message in the `error` column.

The question of every episode is the briefing of the action server (`briefing` in `src/servers/hydraulic_rig/briefing.py`), followed by a note on the seven diagnosis tools whenever the causal diagnosis server is registered. `run.json` records the version of each, `BRIEFING_VERSION` and `TOOLS_NOTE_VERSION`. Before the first episode the driver lists the tools of every registered server once and stops if a server offers other tools than expected (`check_servers`).

A reference policy sees the observation at reset and after that only the tool answers (`play` in `src/servers/hydraulic_rig/policy_client.py`). The greedy policies look their classifier predictions up by the measurements as the answers show them (`PublicClassifiers`), so they decide as they do in process.

---

## Model access

The model agents, `plan_execute` and `plan_execute_rig_only`, read one key, `API_KEY`, from the shell environment. By default they reach TokenRouter at `https://api.tokenrouter.com/v1` with the model `MiniMax-M3`, and `--base-url` and `--model` select another OpenAI-compatible endpoint and model. Unlike the command-line runners of AssetOpsBench, the driver does not load `.env`. Without the key it stops with:

```
API_KEY is not set; configure a key or run with --agent scripted
```

```bash
export API_KEY=<key>
uv run python benchmarks/hydraulic_rig/run_episodes.py --scenario S1 --agent plan_execute --per-record 1
uv run python benchmarks/hydraulic_rig/run_episodes.py --scenario S2 --agent plan_execute_rig_only --model <model> --base-url <endpoint>
```

Inside its own process the driver copies `API_KEY` into `TOKENROUTER_API_KEY` and `--base-url` into `TOKENROUTER_BASE_URL`, then builds AssetOpsBench's backend for the model id `tokenrouter/<model>` with `make_backend` from `src/llm/`. That backend sends each prompt to the chat completions of the endpoint through the OpenAI SDK, so `--base-url` accepts any OpenAI-compatible endpoint. A recording wrapper, `RecordingBackend`, keeps every prompt, reply, token count, and duration for `llm_calls.json`, and `run.json` records `model`, `model_id`, and `endpoint`.

The key never reaches the tool server processes. The runner starts every server through the MCP stdio client without an environment of its own, and on Linux and macOS that client passes on only `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, and `USER`. The action server finds the data file through the path stored in `episode.json`.

No model is connected yet. The tests run `plan_execute` against a local endpoint that answers with the scripted plan, so the real client path runs without a model.

A model name keeps its form in the `agent` column, while the run folder name turns its dots and slashes into underscores. The runner starts each launcher file inside the repository as `python -m <dotted path>` from the fork root, so `--out` refuses a folder inside the repository whose name contains a dot.

---

## What a run writes

### The run folder

| File | Content |
| --- | --- |
| `run.json` | the label, scenario, agent, and servers, the model and endpoint of a model agent, the model id of the backend, the versions of the briefing and the tools note, the environment parameters, `per_record`, `max_records`, the number of episodes, the data file path, the fork commit with SHA-256 prefixes of the environment, server, model, and driver files, the Python, `mcp`, and `openai` versions, and the start and end times |
| `scores.csv` | one row per episode, with the columns below |
| `summary.md` | the rates per agent, the closing answers against the log where the agent answered, and one row per episode with a link to its report (`run_summary` in `report.py`) |
| `seed<NNNNNN>/` | one episode folder per seed |

### The episode folder

| File | Written by | Content |
| --- | --- | --- |
| `episode.json` | action server | the episode state: the seed, the environment parameters, the data file path, every action passed to the environment, and every tool call with its answer |
| `released.json` | action server | everything the agent has been told: the public rules, the counters, and the history as the tool answers showed it, which is the causal diagnosis server's only view of the episode |
| `predictions.json` | causal diagnosis server | the predictions that `estimate_intervention_effect` stored, which `verify_outcome` reads back |
| `hydraulic_rig_server.py`, `causal_diagnosis_server.py` | driver | one launcher file per registered server, which binds the server process to this folder (`write_launchers` in `common.py`) |
| `question.txt` | driver | the question of the episode |
| `plan.txt`, `answer.txt`, `trajectory.json`, `llm_calls.json` | driver | Plan-Execute and scripted runs: the plan, the closing answer, every step with its server, tool, arguments, response, error, and duration, and every model call |
| `<run label>-seed<NNNNNN>.json` | Plan-Execute runner | the runner's own trajectory record of a completed run, since the driver points `AGENT_TRAJECTORY_DIR` at the episode folder |
| `log.json`, `score.json` | driver, after the episode | the replayed log with the hidden state, and the scorer's reading of it |
| `report.md` | `report.py` | the audit report |

An oracle folder holds only `log.json`, `score.json`, and `report.md`. A reference policy folder holds no plan, answer, trajectory, or model calls.

An agent with file or shell tools must run with the run folder out of its reach. `episode.json` holds the seed, from which a replay recovers the hidden combination, and once an episode is scored, `log.json`, `score.json`, and `scores.csv` hold the hidden combination itself.

### The audit report

`report.md` comes from `episode_report` in `report.py`, which reads the files of the folder and needs no model. It holds:

1. The scenario, agent, servers, seed, and task text, with the hidden start combination that the scorer reveals after the episode.
2. The plan of a Plan-Execute or scripted run.
3. Every tool call of both servers with its arguments and a one-line result, from `trajectory.json` when the runner wrote one, else from the calls that the action server recorded.
4. The predictions that the causal tools stored.
5. The outcome from the log: the diagnosis against the truth per component, every replacement with whether the component was faulty and the combination after it, the refused calls, and the scorer's recovery, end reason, cost, steps, and measurements per tier.
6. The closing answer beside the log, with a heuristic reading by `answer_claims_recovery` in `common.py` that flags an answer whose claim of a recovery disagrees with the log, for the reader to judge.
7. The number of model calls, their tokens, and the time spent in the model.

### Columns of scores.csv

| Columns | Rows | Content |
| --- | --- | --- |
| `seed`, `scenario`, `record`, `agent`, `via`, `wall_s`, `error` | every row | the episode seed, the scenario, the start combination as a JSON list, empty for `seeded`, the agent label, `mcp` or `in_process`, the wall time in seconds, and the error of a run that raised |
| `llm_calls`, `input_tokens`, `output_tokens`, `plan_steps`, `plan_tools`, `tool_steps`, `causal_calls`, `calls_reaching_no_tool`, `answer_claims_recovery` | Plan-Execute and scripted runs | the model calls and their tokens, the length and the tools of the plan, the tool steps, the successful calls to the causal diagnosis server, the action server steps whose call reached no tool, and the heuristic reading of the closing answer |
| `tool_calls`, `calls_after_end`, `episode_ended` | every row, the two counts blank for the oracle | the tool calls that the action server recorded, the calls after the episode had ended, and whether the episode ended |
| `submitted` to `final` | every row | the scorer's columns, with dictionaries and tuples written as JSON |

The scorer's columns come from `score` in `src/envs/hydraulic_rig/scoring.py`, which [environment.md](environment.md) describes: `submitted`, `insufficient_evidence`, `diagnosis_per_component`, `diagnosis_all_correct`, `recovered`, `repairs_match_diagnosis`, `cost`, `cost_before_submit`, `steps`, `steps_before_submit`, `measures_tier1`, `measures_tier2`, `measures_tier3`, `replacements`, `unnecessary_replacements`, `refusals`, `end_reason`, `initial`, and `final`.

---

## Results policy

No result ships with the repository. The run folders under `benchmarks/hydraulic_rig/runs/` and the tables under `benchmarks/hydraulic_rig/results/` are listed in `.gitignore`, as are the fitted `models.json` of the causal diagnosis server and the environment's data folder, apart from its `CHECKSUMS.txt`. No result is reported until the planned runs on Columbia's HPC cluster. This page describes every table by its columns or its rows and names the script that writes it.

| File under `benchmarks/hydraulic_rig/results/` | Written by | One CSV row per |
| --- | --- | --- |
| `prediction_verification.csv`, `prediction_verification_summary.md` | `verify_predictions.py` | counterfactual replacement |
| `diagnosability.csv`, `diagnosability_summary.md` | `diagnosability.py` | start state and rung |
| `v0_scores.csv` | `run_policies.py` | policy and episode seed |
| `scenario_anchors_S1.csv`, `scenario_anchors_S2.csv` | `scenario_anchors.py` | anchor, and fold seed for anchor 5 |
| `v0_classifiers.csv`, `v0_classifier_folds.csv` | `eval_classifiers.py` | tier and component, and combination |
| `v0_confusion_row_check.csv` | `confusion_row_check.py` | policy and variant |
| `v0_feature_ablation.csv` | `feature_ablation.py` | feature set |
| `v0_greedy_fold_seeds.csv` | `greedy_fold_seeds.py` | configuration and statistic |

---

## Study A: predictions verified on recorded outcomes

`verify_predictions.py` asks whether the causal diagnosis server predicts what a replacement does. For every start state of a scenario class and every component, one episode runs in process, with no model and no MCP server:

1. One reading of each tier.
2. The ranking of the candidates and the predicted effect of replacing each of the four components.
3. The top candidate submitted as the diagnosis.
4. The replacement of the one component.
5. One more reading of tier 3 where the episode continues.
6. The verification of the prediction for that replacement.

The environment answers the replacement with the recorded cycles of the repaired combination, so the check is against recordings, not a simulation. The study then adds from the log what no tool releases: whether the replaced component was faulty and whether the rig recovered. A replacement that ends the episode leaves no reading to verify. The server's models were fitted once on the combinations outside `S2`, so on `S2` the ranking concerns combinations that the models never saw, while on `S1` it is in-sample. The effect model of every combination leaves that combination's own recordings out.

```bash
bash benchmarks/hydraulic_rig/run.sh verify_predictions.py
bash benchmarks/hydraulic_rig/run.sh verify_predictions.py --scenarios S1 --per-record 1 --max-records 2 --keep /tmp/study_a
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--scenarios` | `S1 S2` | the scenario classes |
| `--per-record` | every start state | the first K start states of every record |
| `--max-records` | every record | the first N records in sorted order |
| `--facts` | `HYD_FACTS_PATH`, else `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` | the rig data file |
| `--out-dir` | `benchmarks/hydraulic_rig/results/` | where the table and the summary go. The default is resolved from the folder of the script, and a path given here from the working directory |
| `--keep` | a temporary folder, deleted at the end | keep the episode folders under this folder |

`prediction_verification.csv` has one row per replacement:

| Columns | Content |
| --- | --- |
| `scenario`, `record`, `seed` | the class, the start combination as a JSON list, and the episode seed |
| `replaced`, `truly_faulty` | the replaced component and whether it was off its healthy level |
| `prediction_id`, `pred_addresses_fault`, `pred_recovery` | the prediction for that replacement: its id, the predicted probability that it addresses a fault, and the predicted probability of recovery |
| `recovered_actual`, `episode_over_after_replace` | the scorer's recovery flag and whether the replacement ended the episode |
| `verify_status`, `n_inside`, `n_outside`, `n_outside_tolerated`, `p_value_outside`, `consistent` | the answer of `verify_outcome`: its status, the observed channel means inside and outside their predicted bands, the tolerated count outside, the binomial tail probability of the count outside, and the verdict |
| `top_candidate`, `top_probability`, `top_is_truth`, `insufficient_evidence` | the ranking after one reading per tier: the top combination, its posterior, whether it is the truth, and whether the server reported insufficient evidence |
| `cost` | the episode cost from the scorer |

`consistent` holds when at most `n_outside_tolerated` channel means lie outside their bands and, where the episode shows whether the rig recovered, the outcome matches the predicted recovery probability rounded to 0 or 1. The tolerated count is the smallest one that a calibrated prediction exceeds with probability at most 0.05 under a binomial model in which each mean misses its band with probability 0.02. [causal-diagnosis-server.md](causal-diagnosis-server.md) gives the full rule.

`prediction_verification_summary.md` gives, per scenario, how often the top candidate is the truth and the evidence insufficient, and the mean predicted probabilities against the log with their Brier scores. It then splits the verified readings by whether the replaced component was faulty, which serves new recordings, or healthy, which keeps the same recordings, and gives for each group the share of channel means inside their bands and the share of verifications that the tool calls consistent. Records can hold unequal numbers of start states, so the main shares also appear with every record weighted equally.

---

## Study B: diagnosability of the start states

`diagnosability.py` grades every start state by the evidence that the causal diagnosis server needs to name its fault. It takes the same options as `verify_predictions.py` and writes `diagnosability.csv` and `diagnosability_summary.md`. Every start state climbs a ladder of four rungs, and after each rung the server ranks the candidates:

| `rung` | Readings added | Readings in total |
| --- | --- | --- |
| `tier 1` | one of tier 1 | 1 |
| `tiers 1 and 2` | one of tier 2 | 2 |
| `tiers 1 to 3` | one of tier 3 | 3 |
| `tiers 1 to 3, ten readings each` | nine more of each tier, in an order that has every tier read every cycle of the window once | 30 |

The environment serves every reading, whatever its tier, from the next cycle of one window of ten cycles and starts the window again after the tenth reading. A cycle read again at the same tier adds nothing to the ranking, so after the last rung, in which every tier has read every cycle of the window, no further reading can change it.

A start state's `grade` is the cheapest rung at which the top candidate is the true combination and the server reports sufficient evidence, which needs a top posterior of at least 0.95 (`CONFIDENCE` in `src/servers/causal_diagnosis/adapter.py`). A state that never gets there is graded `inseparable within the ladder`. The grade is the server's reading of the recorded cycles under models fitted once on a fixed pool, not a property proved about the rig. On `S2`, whose combinations the models never saw, it also measures how the models carry over.

```bash
bash benchmarks/hydraulic_rig/run.sh diagnosability.py
bash benchmarks/hydraulic_rig/run.sh diagnosability.py --scenarios S2 --keep /tmp/study_b
```

`diagnosability.csv` has one row per start state and rung:

| Columns | Content |
| --- | --- |
| `scenario`, `record`, `seed` | the class, the start combination as a JSON list, and the episode seed |
| `rung`, `readings`, `cost` | the rung, the readings so far, and their cost at the default prices |
| `top_candidate`, `top_probability`, `top_is_truth` | the server's top combination, its posterior, and whether it is the truth |
| `insufficient_evidence`, `unseparated` | whether the top posterior is below 0.95, and the components whose most probable level stays below 0.95 |
| `truth_probability` | the posterior of the true combination, or the product of its level probabilities per component when it is not among the three returned candidates |
| `grade` | the grade of the start state, repeated on each of its rows |

`diagnosability_summary.md` gives, per scenario, the number and share of start states at each grade, the readings, the cost, and the share of top candidates that are the truth at each rung, the components left unseparated in the inseparable states, and the grades per record. The share of each grade and the share of top candidates that are the truth also appear with every record weighted equally.

---

## Data scripts

The fork ships no rig data file. [README.md](README.md) shows how to fetch the IndustryAssetEQA facts file and check its SHA-256. The scripts below check that file and rebuild its labels and served statistics from the UCI archive.

| Script | Purpose | Writes |
| --- | --- | --- |
| `verify_data_facts.py` | checks the facts file and the hyd_1 file of AssetOpsBench against each other | nothing |
| `fetch_uci.py` | downloads dataset 447 of the UCI Machine Learning Repository, "Condition monitoring of hydraulic systems", under CC BY 4.0 | the archive, its extracted files, and its SHA-256 in `CHECKSUMS.txt` |
| `build_blocks.py` | recomputes the labels and the served statistics of every cycle and the stable blocks from the raw UCI files | `hyd_cycles.jsonl` and `blocks.json` |

```bash
# check the facts file and the hyd_1 file, read-only
uv run python benchmarks/hydraulic_rig/verify_data_facts.py [--facts PATH] [--hyd1 PATH]

# rebuild the records from the UCI archive
uv run python benchmarks/hydraulic_rig/fetch_uci.py --download [--dest DIR]
uv run python benchmarks/hydraulic_rig/build_blocks.py <raw folder> --check [--facts PATH]
uv run python benchmarks/hydraulic_rig/build_blocks.py <raw folder> --std-ddof {0,1} [--out-dir DIR]
```

`verify_data_facts.py` reads the facts file, by default `HYD_FACTS_PATH` or `src/envs/hydraulic_rig/data/hyd_cycles.jsonl`, and the hyd_1 file, by default `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json`. It checks the SHA-256 of both files, and every other check is a relation between or within the files or a documented convention of the dataset, such as its level sets, so its code states no count of the data. [environment.md](environment.md) lists the checks. The script prints PASS or FAIL per check, exits with status 1 when one fails, downloads nothing, and writes nothing. It does not import the environment, so a test compares the environment's blocks with its recomputation.

`fetch_uci.py` downloads only with `--download`, into a new or empty folder, by default `uci_hydraulic` in the environment's data folder, `DATA_DIR` in `src/envs/hydraulic_rig/data.py`. It records the archive's SHA-256 in `CHECKSUMS.txt` of the data folder and raises an error when a later archive of the same name has another hash. Once `EXPECTED_SHA256` in the script is set, it also refuses every archive with another hash. The constant is unset in the code, so a run prints the hash to copy into it. The script then extracts the archive, refuses any member that would land outside the folder, and prints the folder that holds the 17 sensor files and `profile.txt`.

`build_blocks.py` reads that folder and writes `hyd_cycles.jsonl`, one record per cycle in the layout of the facts file with the mean, std, min, and max of every sensor, and `blocks.json`, the stable block of every combination as the environment's own block builder forms it. `--check` writes nothing. It compares the recomputed labels and statistics with the facts file for every cycle and reports which standard deviation convention, ddof 0 or 1, that file uses, and writing then needs that convention as `--std-ddof`. The default `--out-dir` is the environment's data folder, so the written `hyd_cycles.jsonl` becomes the default data file when `HYD_FACTS_PATH` is unset.

---

## Reference policy scripts

| Script | Purpose | Output in `benchmarks/hydraulic_rig/results/` |
| --- | --- | --- |
| `run_policies.py` | plays `random`, `oracle`, `greedy_prior_1to2`, and `greedy` on seeded episodes of the default rule, and defines `policy_table`, which the driver and `determinism_check.py` import | `v0_scores.csv`, with `policy`, `seed`, and the scorer's columns |
| `scenario_anchors.py` | the admission anchors of `S1` and `S2` for `oracle`, `do_nothing`, `replace_everything`, `random`, `greedy_prior_1to2` under each fold seed, and `symptom_only`, with every start state played once, every record weighted equally, and 95% percentile intervals over 10,000 record resamples, and defines `CLASSES`, `class_combinations`, and `start_state_table`, which `common.py` imports | `scenario_anchors_S1.csv`, `scenario_anchors_S2.csv` |
| `eval_classifiers.py` | the held-out accuracy of the greedy policy's classifiers at each tier over the fold seeds, with the split recorded | `v0_classifiers.csv`, `v0_classifier_folds.csv` |
| `split_checks.py` | prints the checks behind the split of the greedy policy's classifiers | nothing |
| `determinism_check.py` | runs the four policies of `run_policies.py` on episode seeds 0 to 9 in two fresh interpreter processes and compares the SHA-256 of every episode log | nothing, exit status 1 when a log differs |
| `causal_tools_example.py` | prints a worked example of the causal diagnosis computations on one `S2` episode, and the folds behind the case against a cross-fit per episode | nothing |
| `confusion_row_check.py` | compares the greedy policy's observation model with two variants built without the episode's own combination, as an analysis and not a policy | `v0_confusion_row_check.csv` |
| `feature_ablation.py` | the held-out accuracy of the greedy policy's classifiers on feature sets other than the served one | `v0_feature_ablation.csv` |
| `greedy_fold_seeds.py` | the greedy reference policy under the classifiers of every fold seed, and with other minimum rates and confidence thresholds | `v0_greedy_fold_seeds.csv` |

| Script | Options and defaults |
| --- | --- |
| `run_policies.py` | `--facts`, `--episodes` `100`, `--out` `benchmarks/hydraulic_rig/results/v0_scores.csv` |
| `scenario_anchors.py` | `--facts`, `--out-dir` `benchmarks/hydraulic_rig/results/`, `--classes` `S1 S2` |
| `eval_classifiers.py` | `--facts`, `--out-dir` `benchmarks/hydraulic_rig/results/` |
| `split_checks.py` | `--facts` |
| `determinism_check.py` | `--facts`, and `--emit` for the run inside each of the two processes |
| `causal_tools_example.py` | none, it reads `HYD_FACTS_PATH` or the default data file |
| `confusion_row_check.py` | `--facts`, `--episodes` `1000`, `--out` `benchmarks/hydraulic_rig/results/v0_confusion_row_check.csv` |
| `feature_ablation.py` | `--facts`, `--out` `benchmarks/hydraulic_rig/results/v0_feature_ablation.csv` |
| `greedy_fold_seeds.py` | `--facts`, `--episodes` `100`, `--out` `benchmarks/hydraulic_rig/results/v0_greedy_fold_seeds.csv` |

`--facts` defaults to `HYD_FACTS_PATH`, else `src/envs/hydraulic_rig/data/hyd_cycles.jsonl`. Every default output in these tables is resolved from the folder of the script, so it lands in `benchmarks/hydraulic_rig/results/` from any working directory, while a path given on the command line is taken from the working directory.

```bash
uv run python benchmarks/hydraulic_rig/run_policies.py
uv run python benchmarks/hydraulic_rig/scenario_anchors.py
uv run python benchmarks/hydraulic_rig/eval_classifiers.py
uv run python benchmarks/hydraulic_rig/split_checks.py
uv run python benchmarks/hydraulic_rig/determinism_check.py
uv run python benchmarks/hydraulic_rig/causal_tools_example.py
uv run python benchmarks/hydraulic_rig/confusion_row_check.py
uv run python benchmarks/hydraulic_rig/feature_ablation.py
uv run python benchmarks/hydraulic_rig/greedy_fold_seeds.py
```

---

## Tests

```bash
# project environment
uv run pytest benchmarks/hydraulic_rig/tests -q

# pinned runtime, without a project sync
uv run --no-project --python 3.12 --with-requirements benchmarks/hydraulic_rig/requirements-smoke.txt --with pytest python -m pytest benchmarks/hydraulic_rig/tests -q
```

`conftest.py` puts `src/` and the driver folder on the import path, and fits `models.json` once before the tests when the data file exists.

| Test file | Covers | Skipped when |
| --- | --- | --- |
| `test_run_episodes.py` | the scripted plan through both servers, the model agent through the OpenAI-compatible client against a local stand-in endpoint, and a reference policy that scores through the action server as it does in process | the data file or the `mcp` package is missing |
| `test_report.py` | the audit report of an episode built in process, the tool summaries, the answer heuristic, and the run summary | the data file is missing |
| `test_verify_predictions.py` | Study A on two records of `S1` | the data file is missing |
| `test_diagnosability.py` | Study B on two records of `S1`, and the record weighting | the data file is missing |
| `test_run_policies.py` | `run_policies.py` on 100 seeded episodes per policy, each row rerun on its own, and the oracle as a bound | the data file is missing |
| `test_scenario_anchors.py` | the six admission anchors on `S1` and `S2`, with every start state played once | the data file is missing |
| `test_data_facts.py` | every check of `verify_data_facts.py`, and the environment's blocks against its recomputation | the data file is missing or is not the IndustryAssetEQA facts file |
| `test_build_blocks.py` | `build_blocks.py` on synthetic raw files, `verify_data_facts.py` on a rebuilt file, plus two tests on the facts file | the facts file is missing, for those two tests |
| `test_fetch_uci.py` | `fetch_uci.py` without the network, on local archives | never |
| `test_paths.py` | every repository path that the two folder READMEs, the fork's sections of `docs/mcp-servers.md`, and the pages under `docs/hydraulic_rig/` name exists | never |

Without the data file the tests that need it are skipped rather than failed, so a passing run without the file does not exercise the drivers.
