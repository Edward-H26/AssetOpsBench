# Causal Diagnosis Server

`src/servers/causal_diagnosis/` is an MCP server with seven tools that help an agent reason about one hydraulic rig episode: asset information, failure modes, the complaint that opened the episode, the evidence released so far, root causes ranked by probability, the predicted effect of replacing a component, and, after the replacement, a check of that prediction against the recordings. The tools read only what the action server has released and models fitted once outside the evaluation class, so no tool touches the hidden answer or contains a language model. This page covers the design rules, the models file, the tool contract, the methods behind the tools, and how to run and test the server.

Related docs: [README.md](README.md) for the index, [environment.md](environment.md) for the environment and its data file, [action-server.md](action-server.md) for the server that writes the released answers, [benchmarks.md](benchmarks.md) for the drivers and studies, and [../mcp-servers.md](../mcp-servers.md) for every MCP server of the repository.

---

## Design rules

- **Deterministic.** The same released answers, stored predictions, and models file give the same result, and no tool prompts a language model.
- **Only what the agent has seen.** The episode tools read `released.json`, which `src/servers/hydraulic_rig/episode.py` rewrites after every call of the action server. Besides that file, the server opens only `predictions.json` beside it, `models.json`, and the IoT asset profile of `hyd_1`. It never opens the state file `episode.json`, the scored log, or the data file, so it never sees the seed, the hidden combination, a cycle number, or a recording not yet served.
- **One server per episode.** A copy of the launcher beside `released.json` binds a process to one episode, so no tool takes an episode identifier. Every call replays the released events, so a process started for each tool call, as the Plan-Execute executor does, answers as a long-lived one would.
- **Models fitted once.** The models are fitted once on a fixed pool outside the evaluation class S2, never per episode, so they carry no information about the hidden combination. [The models file](#the-models-file) gives the reasoning.

`CausalTools` in `src/servers/causal_diagnosis/adapter.py` implements the tools as plain methods, and `build_server` in `src/servers/causal_diagnosis/main.py` registers them with FastMCP.

---

## The models file

`models.json` in `src/servers/causal_diagnosis/` is generated from the rig data by `src/servers/causal_diagnosis/fit.py` and ignored by git. `ensure_models()` fits it when it is missing. The drivers call it before they start the server or read the models, and the test conftests call it once per session when the data file is present. `load_models()` caches the file per process and raises `FileNotFoundError` with the fit command when the file is missing, so the server cannot start without it. The fit writes the file with `write_json_atomically` of `src/servers/json_files.py`, through a temporary file and a rename.

The fit uses the standard library only and reads the data file that `HYD_FACTS_PATH` names, or `src/envs/hydraulic_rig/data/hyd_cycles.jsonl` when the variable is unset. To refit from the fork root:

```bash
export HYD_FACTS_PATH=/path/to/hyd_facts.jsonl
uv run python -m servers.causal_diagnosis.fit
```

The pool is the stable cycles of every combination whose number of faulty components is not `EVALUATION_FAULTS`, which is two. No model sees a combination of class S2, which holds every combination with exactly two faulty components, while the combinations of class S1, with one faulty component, belong to the pool.

**Why one fixed pool.** A cross-fit per episode would keep the start combination out of its own models by fitting them on the folds that do not hold it. The models served would then change with the fold of the hidden combination, and wherever the folds give different bands, the bands that `get_episode_evidence` returns would tell the folds apart. An agent that plays many episodes, as in training, could learn which bands go with which fold and narrow its candidates with information that no measurement paid for. The last step of `benchmarks/hydraulic_rig/causal_tools_example.py` prints, for the folds that `stratified_folds` draws over all 144 combinations, the candidates of the default rule and the S1 combinations in each fold, and how many distinct sets of tier 1 bands those folds would serve. The fixed pool serves the same models in every episode, so they say nothing about the hidden combination, and an episode of class S2 still never meets its start combination in a model.

The cost is that the models have already seen part of the evidence, in two places. On class S1, the classifiers and bands have seen the recordings that an episode serves. In an episode of class S2, a replacement of a faulty component moves the unit to a combination of the pool, whose recordings the classifiers and bands have also seen. The effect model avoids both, since the prediction for each combination comes from a fit without the recordings of that combination, and that fit is the same in every episode.

The file holds:

- **`tiers`.** Per tier and component, a linear discriminant analysis (LDA) of `src/envs/hydraulic_rig/classifiers.py`, fitted on the pool and kept at full precision so that the classifications cannot change with the file. Beside it, the confusion rows, the urn concentrations `alpha` chosen on `ALPHA_GRID`, and the held-out accuracies of a cross-fit within the pool over `FOLDS` folds of `stratified_folds` at `FOLD_SEED`, counted on the pool combinations with one or three faulty components, `ROW_FAULTS`.
- **`bands`.** For every served summary, the 1st and 99th percentiles, `BAND`, over the pool cycles in which every component that the channel indicates is healthy, or over all pool cycles for a channel that indicates none.
- **`effect` and `effect_pool`.** An additive model of the 17 channel means. For each pool combination, the file keeps the prediction of the fit on the other pool combinations and its residual quantiles on `EFFECT_GRID`, and for the other combinations, the fit on the whole pool.
- **Provenance.** The `pool` rule and combinations, `facts_sha256` of the data file, its number of `records`, and the date `generated`.

Bands, quantiles, and predicted means are rounded to the six significant digits at which values are served. `test_models_json_is_the_fit_of_the_facts_file` refits and compares the result with the stored file.

---

## Contract shared by the tools

**Inputs.** The episode tools check the `format` of `released.json` against `RELEASED_FORMAT` and use its public rule from `min_faulty` to `max_faulty`, its budget, limits, prices, and counters, the optional `task_text`, and the `events`: accepted measurements with their released values, replacements, and refusals. The only file a tool writes is `predictions.json`.

**Parameters.** Every parameter is flat, typed, and required, because the Plan-Execute argument filler sees only the name, JSON type, and required flag of a parameter. The tool descriptions state the legal values.

**Results.** Each tool returns a pydantic model of `src/servers/causal_diagnosis/models.py` that extends `ToolResult`, or an `ErrorResult` whose field `error` explains a bad argument or a missing episode. A client receives the error as an ordinary answer, `{"error": ...}`.

| Field | Content |
| --- | --- |
| `evidence_ids` | `obs/{n}/{field}` for a summary of the n-th accepted measurement, `fmea/industryasseteqa_hydraulic_facts/{code}` for a failure-mode code, `assoc/helwig2015condition/{channel}_{component}` for an association, and `pred/{n}` for a stored prediction |
| `time_window` | `start`, `end`, and `unit`, which is `static` for the static tools and `measurement` for the others. Measurements count from 1, and refused calls do not count |
| `assumptions` | Plain statements of the modeling choices behind the answer |
| `uncertainty` | `kind`, `value`, and an optional `detail` |
| `message` | A one-line summary |

The uncertainty kind is `none` for the asset context, `held_out_accuracy_tier3` for the paths, `candidates` for the symptom report, `band_coverage` for the evidence and the verification, `posterior` for the ranking, and `recovery_probability` for the effect.

---

## Tools

| Tool | Arguments | Returns, besides the shared fields | Needs an episode |
| --- | --- | --- | --- |
| `get_asset_context` | `asset_id` | `asset_type`, `description`, `site`, the `components` with their levels and healthy level, the `channels` with unit, rate, group, and indicated components, `n_failure_codes`, and `n_fault_codes` | No |
| `query_fmea_paths` | `asset_id`, `node` | `node_kind` and the `paths`, each with code, severity, recommended action, indicating channels, and held-out accuracy per group | No |
| `get_symptom_report` | none | `task_text`, `report_time`, `fault_count_min`, `fault_count_max`, `n_candidates`, `budget`, `step_cap`, `repairs`, `prices`, and `status` | Yes |
| `get_episode_evidence` | `sensor_group` | `n_measurements` and the `items`, each with value, unit, release time, healthy band, and whether the value lies inside it | Yes |
| `rank_root_causes` | `top_k` | `n_candidates`, the top `candidates` with faults, probability, and supporting evidence, `component_fault_probability`, `level_probability`, `insufficient_evidence`, `unseparated`, and `separating_channels` | Yes |
| `estimate_intervention_effect` | `action` | `prediction_id`, `component`, `addresses_fault`, `recovery`, `price`, and the `predicted` channel means with their bands | Yes |
| `verify_outcome` | `prediction_id` | `status`, `observation`, the `residuals`, `n_inside`, `n_outside`, `n_outside_tolerated`, `p_value_outside`, `recovery_predicted`, `recovered`, `recovery_status`, and `consistent` | Yes |

- `asset_id` is `hyd_1`. Any other asset returns an error.
- `node` is a component, a failure-mode code such as `valve_switching_critical`, or a channel such as `PS1` or `P1_MotorPower_W_100Hz`. A channel returns the fault codes of every component that it indicates.
- `sensor_group` is `all`, `tier1`, `tier2`, `tier3`, or one channel. A tier keeps the channels that a measurement at that tier returns.
- `top_k` is a positive integer, `action` is `monitor` or `replace:<component>`, and `prediction_id` is an ID such as `pred/1`.

---

## Knowledge base

`src/servers/causal_diagnosis/knowledge.py` holds the static knowledge. `FAILURE_CODES` holds the 16 distinct failure-mode entries of the IndustryAssetEQA hydraulic facts file, each with a component, the condition level that carries it, a severity, and a recommended action. The ten with the severity `critical` or `degraded` form `FAULT_CODES`, four mark the healthy levels, such as `cooler_ok`, and `transient` and `steady` describe the operating regime. In `INDICATES`, a channel indicates a component when the channel, or the virtual sensor served as CE, CP, or SE, is among the features most correlated with the condition of that component in Table II of Helwig, Pignanelli, and Schütze, IEEE I2MTC 2015. The temperatures indicate the cooler, and PS4, PS5, PS6, FS2, and VS1 indicate no component. The table takes the levels from `LEVELS` and `HEALTHY` of `src/envs/hydraulic_rig/data.py`.

| Component | Condition levels | Healthy | Fault codes by level | Indicating channels |
| --- | --- | --- | --- | --- |
| cooler | 3, 20, 100 | 100 | `cooler_efficiency_critical` at 3, `cooler_efficiency_reduced` at 20 | CE, CP, TS1, TS2, TS3, TS4 |
| valve | 73, 80, 90, 100 | 100 | `valve_switching_critical` at 73, `valve_switching_severe_lag` at 80, `valve_switching_small_lag` at 90 | PS1, PS2, PS3, FS1, EPS1, SE |
| pump | 0, 1, 2 | 0 | `pump_internal_leakage_weak` at 1, `pump_internal_leakage_severe` at 2 | PS3, FS1, SE |
| accumulator | 90, 100, 115, 130 | 130 | `accumulator_pressure_low_critical` at 90, `accumulator_pressure_severely_reduced` at 100, `accumulator_pressure_reduced` at 115 | PS1, PS2, PS3, FS1, EPS1, SE |

`query_fmea_paths` returns the correlations of that table, `HELWIG_CORRELATIONS`, with the assumption that they describe interval features of the fixed working cycle rather than the served summaries. An association is diagnostic evidence, not a causal edge. The 17 channels of `SENSORS` carry the field names of `HYD1_FIELDS` in `src/servers/hydraulic_rig/names.py`, such as `P1_MotorPower_W_100Hz` for EPS1, each with the four `STATISTICS`: mean, std, min, and max. `group_of` names the cheapest tier that serves a channel, and the groups are nested: `tier1` serves six channels, `tier2` adds three, and `tier3` serves all 17. `asset_profile` reads the `hyd_1` entry of `src/couchdb/scenarios_data/shared/iot/asset_profile_sample.json`.

---

## Ranking root causes

`rank_root_causes` replays the released events into a `Belief`, whose prior is uniform over the combinations of `COMBOS` that the public rule allows. The classifiers of the measured tier read each accepted measurement from its released values, one level per component, and repeated readings of one served block combine by the Polya urn of the greedy reference policy in `src/envs/hydraulic_rig/greedy.py`. Each reading multiplies the weight of a hypothesis by one factor per component:

```
factor = (alpha * row[true level][level read] + earlier[level read]) / (alpha + number of earlier readings)
```

`row` is the confusion row at that tier, `alpha` the urn concentration, and `earlier` counts the levels read before from the same block at the same tier. A cycle read again at the same tier adds nothing. A released replacement moves each hypothesis in which the component was faulty to the combination with that component healthy and restarts its counts, since the environment then serves the recordings of the new combination. A hypothesis in which the component was healthy keeps its block. The environment ends an episode at once when every component is healthy, so when the episode continues after a replacement, the hypotheses that the replacement brings to the healthy combination drop out. Hypotheses stay per start combination, so two start combinations that a replacement brings together keep their own blocks.

The answer lists the `top_k` most probable combinations with their fault codes and the released fields whose channels indicate a faulty component, and the fault probability and level distribution of every component. `insufficient_evidence` is true while the top probability is below `CONFIDENCE`, which is 0.95, the default of the greedy reference policy. `unseparated` lists each component whose most probable level has a probability below `CONFIDENCE`, and `separating_channels` names the fields of the channels that indicate it.

---

## Estimating an intervention effect

For `replace:<component>`, `estimate_intervention_effect` reports the posterior probability that the component is faulty as `addresses_fault` and moves the posterior through the replacement as the ranking does. It then reports the moved probability of the combination with every component healthy as `recovery` and the part price plus the downtime charge as `price`. For `monitor`, `addresses_fault` and `price` are 0 and the posterior stays as it is.

`predicted` holds the means of all 17 channels, which a measurement at tier 3 returns. Each combination in the moved posterior contributes the additive model fitted without it when it belongs to the pool, or the fit on the whole pool otherwise:

```
channel mean = intercept + sum over components of effect[component][level], with no effect at a healthy level
```

The mean is weighted by the posterior, and the band runs from the 1st to the 99th percentile of the posterior-weighted mixture of prediction plus residual. `mixture` in `src/servers/causal_diagnosis/effect.py` interpolates each residual distribution between its 39 quantiles on `EFFECT_GRID`, every half percent in the outer 5 percent of each tail and every five percent between. It finds the band ends by bisection and drops the combinations whose weight does not exceed `NEGLIGIBLE`.

Each call appends the prediction to `predictions.json` as `pred/{n}`, with the number of events released so far, for `verify_outcome` to read later. A note in the assumptions flags an action that the environment would refuse: a repair before the diagnosis, a repair with none left, or any action after the episode is over.

---

## Verifying an outcome

`verify_outcome` compares a stored prediction with the first measurement released after the action, which is the first replacement of that component since the prediction, or the prediction itself for `monitor`.

| `status` | When |
| --- | --- |
| `not yet observable: the action has not been taken` | No replacement of the component since the prediction |
| `action refused: <reason>` | The environment refused that replacement |
| `not yet observable: no measurement since the action` | No measurement followed the action |
| `not comparable: <replacements> came between the prediction and the measurement` | Another component was replaced after the prediction and before that measurement, so the measurement can read another combination |
| `verified` | A measurement followed the action |

Each channel mean of that measurement, six, nine, or 17 at tier 1, 2, or 3, gets a residual, the observed value minus the predicted mean, and a flag for whether it lies inside the band. The environment ends an episode at once on recovery, so `recovered` is false when the episode continues after the replacement. It is null when the episode ends with the replacement, since a recovery, the last allowed repair, and the step cap look the same in the released answers. For `monitor`, recovery does not apply.

A calibrated prediction misses each band with probability `MISS_RATE`, one minus `BAND_COVERAGE`, which is 0.02. Requiring all 17 means inside would call a calibrated prediction inconsistent in more than one case out of four, so the verdict tolerates as many misses as a calibrated prediction exceeds with probability at most `FALSE_ALARM`, which is 0.05, under a binomial model of independent summaries:

```
X                   ~ Binomial(n, MISS_RATE), n = number of residuals
n_outside_tolerated = smallest k with P(X >= k + 1) <= FALSE_ALARM
p_value_outside     = P(X >= n_outside)
consistent          = n_outside <= n_outside_tolerated
                      and (recovered is null or recovered == (recovery_predicted >= 0.5))
```

For six, nine, or 17 residuals, one miss is tolerated. `misses_tolerated`, `binomial_tail`, and `outcome_consistent` in `src/servers/causal_diagnosis/adapter.py` implement the rule.

---

## A worked episode

`test_worked_s2_example_of_the_design_document` in `src/servers/causal_diagnosis/tests/test_server.py` plays this sequence on an episode of class S2, from the ranking in step 1 onward.

1. `get_asset_context`, `query_fmea_paths`, and `get_symptom_report` describe the asset, its failure modes, the task, and the public rule, and a ranking gives every candidate the same probability.
2. The action server's `measure` reads tier 1, `get_episode_evidence` lists the summaries with their bands, and `rank_root_causes` updates the posterior. While the evidence is insufficient, the agent measures again, for example at tier 3, or calls the action server's `submit_insufficient_evidence`.
3. `estimate_intervention_effect` with `replace:accumulator` stores `pred/1`, and the agent submits the top candidate through the action server's `submit`, since a repair needs a diagnosis first.
4. After the action server's `replace`, `verify_outcome` reports `pred/1` as not yet observable and the unit as not recovered. After a `measure` at tier 3, it returns the residuals and the verdict, and `rank_root_causes` keeps only combinations with a healthy accumulator.
5. `estimate_intervention_effect` with `replace:valve` stores `pred/2`. The `replace` that follows uses the last repair and ends the episode, so `verify_outcome` reports recovery as not observable.

`benchmarks/hydraulic_rig/causal_tools_example.py` fits the same models itself with the standard library only and prints the steps of this episode. Its bands come from the full residual lists rather than from the stored quantiles.

---

## Running the server

The `causal-diagnosis-mcp-server` script of `pyproject.toml` starts the server without an episode, which serves the two static tools only:

```bash
uv run causal-diagnosis-mcp-server
```

For the episode tools, a copy of `src/servers/causal_diagnosis/launcher.py` beside the `released.json` of an episode passes that file to `main`, with `src/` on the import path as in the project environment. The server speaks MCP over stdio and waits for a client:

```bash
cp src/servers/causal_diagnosis/launcher.py /path/to/episode/causal_server.py
uv run python /path/to/episode/causal_server.py
```

The AssetOpsBench runners register by default only the servers of `DEFAULT_SERVER_PATHS` in `src/agent/runner.py`, so this server reaches a runner through `server_paths`. `benchmarks/hydraulic_rig/run_episodes.py` calls `ensure_models()`, creates each episode with `start_episode`, and writes one launcher per server into its folder with `write_launchers` of `benchmarks/hydraulic_rig/common.py`. These launchers put the fork's `src/` on the import path themselves, since a server process receives almost no environment variables. `PlanExecuteRunner` gets the launcher paths, and its executor starts a server process for every tool call. The studies `benchmarks/hydraulic_rig/verify_predictions.py` and `benchmarks/hydraulic_rig/diagnosability.py` call `CausalTools` in process.

---

## Tests

Run the tests from the fork root with the data file set:

```bash
export HYD_FACTS_PATH=/path/to/hyd_facts.jsonl
uv run pytest src/servers/causal_diagnosis -q
```

`default_facts_path()` in `src/envs/hydraulic_rig/data.py` finds the data file. When it is present, the session fixture of `src/servers/causal_diagnosis/tests/conftest.py` calls `ensure_models()` once. When it is absent, `src/servers/causal_diagnosis/tests/test_server.py` is skipped as a whole, so a passing run without the file does not exercise the server. The stdio test also needs the mcp package. Most tool calls in the tests run inside `hidden_files_sealed`, which makes every open of `episode.json`, `log.json`, `score.json`, or the data file fail. No test states a value computed from the rig data.
