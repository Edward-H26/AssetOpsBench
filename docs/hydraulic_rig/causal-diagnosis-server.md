# Causal Diagnosis Server

`src/servers/causal_diagnosis/` is the diagnostics server of the hydraulic rig gym, registered under the MCP name `causal_diagnosis`. It offers four read only tools over the evidence an episode has released: `get_episode_evidence` lists the readings with their healthy bands and anomaly scores, `rank_root_causes` ranks the candidate start combinations under an interventional effect model and decides whether the evidence is insufficient, `estimate_intervention_effect` predicts what a repair would do with conformal intervals, and `verify_outcome` checks a stored prediction against the recordings after the repair. Every threshold in a result traces to a price, a duration, a downtime rate or the declared error level, and the assumptions of every result say how. The server contains no language model and never touches the hidden state. This page covers the design rules, the models file and the grade table, the contract of the tools, the four tools with their results, the formulas E1 to E19 with the module that implements each, the ranking models and the abstention rules, the calibration report, and how to run and test the server.

Related docs: [README.md](README.md) for the index, [environment.md](environment.md) for the environment and its data file, [action-server.md](action-server.md) for the server that writes the released evidence and the tool contract, [benchmarks.md](benchmarks.md) for the drivers and the ablations, and [../mcp-servers.md](../mcp-servers.md) for every MCP server of the repository.

---

## Design rules

- **Deterministic.** The same released evidence, stored predictions and models file give the same result. The Monte Carlo of the decision rule and of the grade is seeded with declared seeds, and no tool prompts a language model.
- **Only what the agent has seen.** The tools read `released.json`, which `src/servers/hydraulic_rig/episode.py` rewrites after every call of the action server, and the predictions the server stored itself in `predictions.json` beside it. They never open the state file `episode.json`, the hidden files of an episode or the data file, so they never see the seed, the hidden combination, a recording index or a recording not yet served.
- **One server per episode, no memory between calls.** A copy of the launcher beside `released.json` binds a process to one episode, so no tool takes an episode identifier. Every call replays the released observations and closed work orders into a fresh posterior, so a process started for each tool call, as the Plan-Execute executor does, answers exactly as a long lived one would. The models file is parsed once per process. The cost of this design is one process start and one replay per call, which the executor pays for every tool.
- **Models fitted once on a fixed pool.** Every model is fitted once by `src/servers/causal_diagnosis/fit.py` on the stable cycles of the combinations outside the evaluation split, never per episode, so no model has seen a combination an episode can start from and no model depends on the episode. [The models file](#the-models-file) gives the reasoning.
- **Every threshold traces to a cost or to alpha.** The decision whether the evidence is insufficient compares expected losses built from the released prices, durations and downtime rates. The intervals and the verdicts use the declared error level `alpha`. No fixed confidence cut enters a default result, and the one cut the server knows is an ablation arm named in the result that uses it.
- **Read only.** Every tool is registered with `readOnlyHint` true, since none changes the asset or the case record. `estimate_intervention_effect` stores its prediction in `predictions.json`, the server's own record, which is the only file the tools write.

`CausalTools` in `src/servers/causal_diagnosis/adapter.py` implements the tools as plain methods, testable without the mcp package, and `build_server` in `src/servers/causal_diagnosis/main.py` registers them with FastMCP.

---

## The models file

`models.json` in `src/servers/causal_diagnosis/` is generated from the rig data by `src/servers/causal_diagnosis/fit.py` and ignored by git. `ensure_models` fits it when it is missing or in an older format. The drivers call it before they start a server or look a grade up, and the test fixtures call it once per session when the data file is present. `load_models` parses the file once per process into a `Models` object (the effect model, its conformal calibration, the within standard deviations, the urn model and the pool) and raises `FileNotFoundError` with the fit command when the file is missing, so the server cannot start without it. To refit from the fork root:

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python -m servers.causal_diagnosis.fit [--interactions select|none] [--resamples N] [--seed S] [--out PATH]
```

**The pool.** The evaluation split is every combination the default public fault rule can start an episode from, the combinations with one or two faulty components. The pool is the stable cycles of every other combination: the healthy one and those with three or four faulty components. No model sees a combination of the evaluation split, and the same models serve every episode, so they carry no information about the hidden combination. A cross fit per episode would instead serve different models depending on the fold of the hidden combination, and an agent that plays many episodes could learn which answers go with which fold and narrow its candidates with information no reading paid for.

`models.json`, format 2, holds:

| Field | Content |
| --- | --- |
| `format`, `generated`, `facts_sha256`, `records` | the format, the fit date, the SHA-256 of the data file and its number of records |
| `pool` | the rule in words, the pool combinations, the evaluation combinations, the number of pool cycles, the number of folds and the fold seed |
| `components`, `levels`, `healthy` | the facts of the asset |
| `ranking_models`, `default_ranking_model` | the three ranking models and the default, `joint_block` |
| `effect` | the effect model E1: the 34 summary fields, the summaries of every sensor group, the main effect terms, the selected interaction pairs and their indicator terms, the coefficients, the within covariance, the block misfit covariance, the healthy median and median absolute deviation of every summary, the conformal calibration scores with their block weights, the resampled fits, and the provenance (pool, folds, selection log, shrinkage weights, resample seed) |
| `discriminant_urn` | the v1 ranking model refitted on the pool for ablation A12: per sensor group the discriminants, the confusion rows, the urn concentrations and the held out accuracies |

Resampled fits and calibration scores are rounded to the six significant digits at which values are served. `test_models_json_is_the_fit_of_the_facts_file_in_format_2` refits and compares the result with the stored file.

**The grade table.** `grades.json` beside `models.json`, format 1, holds one table of diagnosability grades per hash of the released prices, durations, downtime rates, horizon, fault rule, budget and ranking model (`rules_key` in `src/servers/causal_diagnosis/grade.py`). `ensure_grades(models, rules, budget=None, ranking_model=...)` computes a table on first use and `grade_of(start, models, rules, ...)` returns the grade of one start, or None when the start is inseparable under the budget. The drivers look the grade of every episode up and hand it to the verifier. A table for the default configuration is regenerated with

```bash
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python -m servers.causal_diagnosis.grade [--ranking-model joint_block|joint] [--budget B] [--rollouts R] [--samples N] [--seed S] [--force] [--out PATH]
```

Both files are git ignored. Generating a grade table runs rollouts to budget exhaustion for inseparable starts, so the first use of a new table takes a while.

---

## Contract shared by the tools

**Inputs.** The tools check the `format` of `released.json` against `RELEASED_FORMAT` and read its `rules` (the public fault rule, the sensor groups, the prices, the durations, the downtime rates, the horizon and the healthy bands), its `status` (`time`, `open_work_orders`, `episode_over`), its `observations` (the readings with `n`, `t`, `sensor_group` and `values`) and its `work_order_closed` events (`id`, `component`, `source`, `t_end`), which enter the posterior as closed orders whatever their source. Refused calls release nothing. Without a released file, or with a server started without an episode, every tool answers an error.

**Parameters.** Every parameter is flat, typed and required, because the Plan-Execute argument filler sees only the name, the JSON type and the required flag of a parameter. The tool descriptions state the legal values and carry no number.

**Results.** Each tool returns a pydantic model of `src/servers/causal_diagnosis/models.py` that extends `ToolResult`, or an `ErrorResult` whose field `error` explains a bad argument or a missing episode. A client receives the error as an ordinary answer, `{"error": ...}`.

| Field | Content |
| --- | --- |
| `evidence_ids` | `obs/{n}/{field}` for a released value, `wo/{id}` for a closed work order, and `pred/{n}` for a stored prediction, the identifiers the result rests on |
| `time_window` | `start`, `end` and `unit`, which is `cycle`: the first reading used and the current time, or for a prediction the current time and the time the action would end |
| `assumptions` | plain statements of the formulas behind the answer, each threshold traced to its price, duration, downtime rate or alpha, with the Monte Carlo count and seed where one is used |
| `uncertainty` | `kind`, `value` and an optional `detail` |
| `message` | a one line summary, which the audit report prints |

| Tool | Uncertainty kind |
| --- | --- |
| `get_episode_evidence` | `largest_anomaly_score` |
| `rank_root_causes` | `top_marginal_resampling_interval`, with the interval and the top share in `detail` |
| `estimate_intervention_effect` | `recovery_probability`, with its resampling interval in `detail` |
| `verify_outcome` | `smallest_p_value` |

---

## Tools

| Tool | Argument | Returns, besides the shared fields |
| --- | --- | --- |
| `get_episode_evidence` | `sensor_group`: `all`, a group name such as `tier1`, a channel such as `PS1`, or a field such as `PS1_Pressure_bar_100Hz` | `n_observations`, the `items` (per released value its evidence identifier, field, channel, statistic, value, unit, sampling rate, observation number, cycle, sensor group, the public band of the channel mean with `inside_band`, and the anomaly score E3), and `work_orders_closed` |
| `rank_root_causes` | `top_k`: a positive integer | `ranking_model`, `n_candidates`, the top `candidates` (rank, start combination, current combination, faults with their codes, probability and the released values whose channel indicates a faulty component), `component_fault_probability`, `current_fault_probability`, `level_probability`, the `attribution` per component E4, the `stability` E6, the decision fields E10 (`insufficient_evidence`, `separating_test`, `expected_saving`, `expected_loss_act`, `expected_loss_test`), `unseparated` and `separating_channels` E12, `monte_carlo` and the `costs` E7 |
| `estimate_intervention_effect` | `action`: `replace:<component>` or `monitor` | `prediction_id`, `action`, `component`, `addresses_fault`, `recovery` with its resampling interval, `price`, `duration`, `downtime_during_action`, the `predicted` summaries with their intervals E14 and `alpha`, and `risk_before`, `risk_after`, `direction` and `dead_band` E15 |
| `verify_outcome` | `prediction_id`: an identifier such as `pred/1` | `status`, the `observation` compared and its cycle, the `channels` with predicted and observed values, the interval, the score, the p value E16 and the Bonferroni and Holm flags E17, `n_channels`, `alpha`, `consistent`, the rejection counts, `recovery_predicted` with its interval, `recovery_posterior` and `recovery_status` |

---

## Formulas

The numbers are those of the design and of the paper's appendix. Every function takes the released rules and `alpha`, and nothing in them names this rig (E19).

| Formula | Module and function | Declared constants |
| --- | --- | --- |
| E1 effect model | `effect_model.fit_effect_model`, `select_interactions`, `cross_fitted_log_likelihood`, `within_covariance`, `ledoit_wolf_diagonal`, `EffectModel` | `FOLDS` 8, `FOLD_SEED` 0, `RESAMPLES` 200, `RESAMPLE_SEED` 0, `RESAMPLE_ATTEMPTS` 100 |
| E2 likelihood and posterior | `posterior.Posterior`, `segments`, `merge_observations`, `orders_before`, `segment_log_likelihood` | a uniform prior over the public rule |
| E3 anomaly score | `EffectModel.anomaly_scores` | none |
| E4 attribution | `EffectModel.component_terms`, `Posterior.attribution` | none |
| E5 marginals | `Posterior.component_marginals`, `level_probability`, `current_marginals`, `order` | none |
| E6 stability | `Posterior.stability`, `resampled_recovery`, `EffectModel.resampled` | `RESAMPLES` 200 |
| E7 costs | `decision.costs`, `test_cost`, `downtime_rate`, `expected_downtime_rate` | the released prices, durations, downtime rates and horizon |
| E8 loss of acting | `decision.loss_matrix`, `expected_loss_act` | none |
| E9 loss of testing | `decision.expected_loss_test`, `Posterior.sample_next`, `update_with` | `MONTE_CARLO_SAMPLES` 512, `MONTE_CARLO_SEED` 0 |
| E10 decision | `decision.decide` | the candidate tests, one cycle of each sensor group |
| E11 one component threshold | `decision.one_component_threshold` | none |
| E12 unseparated | `decision.unseparated` | none |
| E13 post repair posterior | `Posterior.post_repair`, `CausalTools.estimate_intervention_effect` | none |
| E14 conformal interval | `conformal.Calibration.quantile`, `conformal.interval`, `CausalTools._predict` | `DEFAULT_ALPHA` 0.05 |
| E15 risk and direction | `CausalTools.estimate_intervention_effect` | the dead band from E6 |
| E16 p value | `conformal.Calibration.p_value`, `CausalTools.verify_outcome` | the `alpha` of the stored prediction |
| E17 verdict and recovery check | `conformal.verdict`, `CausalTools.verify_outcome` | `alpha` |
| E18 grade | `grade.rollout`, `grade`, `compute_grades`, `ensure_grades`, `grade_of`, `expected_information_gain` | `ROLLOUTS` 10, `GRADE_SAMPLES` 128, `GRADE_SEED` 0, the budget of the rules |
| E19 transfer | every function above | none |
| Calibration test | `calibration.evaluate`, `report`, `run` | `READINGS` 10, `ROLLOUTS` 20, `SEED` 0, `BINS` 10 |

**E1, the interventional effect model.** The experimenters of the dataset set every component level, so the recordings under a combination are draws from the interventional distribution of the channel summaries under that combination, and the effect of a component condition on a channel is estimated from the recordings without causal discovery. The mean of each of the 34 summaries, the mean and the standard deviation of the 17 channels over one recorded cycle, is an intercept plus one main effect per non healthy level of each component plus the pairwise interaction indicators selected forward by the cross fitted log likelihood over the pool, with the folds of `stratified_folds` in `src/envs/hydraulic_rig/classifiers.py` restricted to the pool. A pair is admissible only when the pool design keeps full column rank, a pair that some training fold cannot identify is not selectable, and the selection log is stored in the provenance. The fit is weighted least squares on block means with the block lengths as weights, which equals least squares over cycles. The residual covariance is the pooled within combination covariance, every block centered on its own mean, shrunk toward its diagonal by the Ledoit and Wolf weight in the diagonal target form of Schaefer and Strimmer. The fit also keeps the block misfit covariance, the shrunk covariance of the cross fitted block mean residuals, the healthy reference of E3, the conformal calibration scores of E14, and the resampled fits of E6, one per block bootstrap draw of the pool, redrawn when a draw loses a level.

**E2, the posterior.** The candidates are the start combinations the public fault rule allows, with a uniform prior. The combination current at a reading is the start combination with every component of a work order closed at or before the reading's cycle set to its healthy level, so the readings are grouped into segments between closures and each segment is read under its own combination. A cycle served twice within a segment, because the recording window wrapped or two groups read the same cycle, enters once with the union of its summaries. The literal E2, the ranking model `joint`, treats the observations of a segment as independent Gaussian draws around the model mean with the within covariance restricted to the observed summaries. The default, `joint_block`, adds a latent block mean: the cycles of one recorded block share an offset from the model mean, since the block was recorded in one run of the rig under one setting and the effect model predicts it from the other blocks. The offset has the block misfit covariance, is integrated out by a Kalman step per observation, and makes repeated readings of one block add less than independent draws would, the same reason for which E6 resamples by block. Scheduled drops are not modeled, which the assumptions state.

**E3, the anomaly score** of a summary is its absolute deviation from the median of the healthy recordings over their median absolute deviation, with the within standard deviation as the scale when the deviation is zero. It enters no decision.

**E4, the attribution.** The predicted deviation of a summary under the top candidate splits into the main effects of its faulty components and halves of their interactions, the shares are normalized per summary, and a component's explained anomaly is the sum over summaries of the anomaly score times its share, for the top candidate at the latest segment with readings.

**E5, the marginals.** The fault probability of a component at the start, its level probabilities, and its fault probability now, after the released closures. Candidates are ordered by decreasing posterior.

**E6, the stability.** The posterior is recomputed under every resampled fit. The interval of the top component's marginal is its range over the resamples, and `top_share` is the share of resamples in which that component stays on top. No band constant is used. The same resamples give the recovery probability of E13 under every fit, from which E15 takes its dead band.

**E7, the costs**, from the released rules at the current cycle. An unnecessary replacement costs the part plus the repair charge plus the stopped downtime rate over the repair downtime. A missed fault costs its downtime rate over the remaining horizon. A test costs its read price for one cycle plus one cycle of the expected downtime rate of the current state.

**E8 and E9, the two expected losses.** The loss of acting on the faulty set of combination a when h is true sums, over components, an unnecessary replacement where a is faulty and h healthy and a missed fault where a is healthy and h faulty. E8 is the posterior expected loss of acting on the top candidate's current faulty set. E9 is the expected loss of first reading one cycle of a sensor group and then acting on the top candidate of the updated posterior, integrated over the predictive mixture of the next reading by Monte Carlo with the declared sample count and seed, plus the test cost. The standard error of the Monte Carlo mean is reported per test.

**E10, the decision.** The evidence is insufficient when some test lowers the expected loss below acting now. The separating test is the test that lowers it most, and the expected saving is the difference. The candidate tests are one cycle of each sensor group. This is the one step lookahead of Heckerman, Breese and Rommelse written for the rig's prices, durations and downtime rates and for more than one fault.

**E11, the one component case.** With one uncertain component, one resolving test and a replacement that is unnecessary with probability one minus p, acting now is cheaper than testing when p is at least one minus the test cost over the cost of an unnecessary replacement. It shows what a fixed confidence cut stands for.

**E12, the unseparated components.** A component is unseparated when the decision to replace it differs between the top two candidates and some test lowers the expected loss of that decision alone. Its separating channels are the fields of that test.

**E13, the post repair posterior.** The posterior moves every candidate's current combination through the replacement. The probability that the action addresses a fault is the posterior mass on which the component is faulty now, and the recovery probability is the moved mass of the healthy combination. `monitor` is one cycle without a repair: it addresses nothing, costs no price, lasts one advance cycle and accrues the expected downtime of that cycle, since no observation is free.

**E14, the conformal interval.** The predicted mean of a summary after the action is the posterior weighted mean of the model means of the reachable combinations. The interval is the mean plus or minus q times the within standard deviation of the summary, where q is the smallest held out calibration score such that at least (1 minus alpha)(n plus 1) units of calibration weight lie at or below it, n the number of pool blocks. The calibration set holds the held out cycles of the cross fit with one unit of weight per block shared by the block's cycles, so that the exchangeability behind the coverage guarantee is across blocks rather than across the correlated cycles of one block. When the target exceeds the calibration set, the largest score stands in and the result flags it. `alpha` is the one declared constant, printed in every result.

**E15, risk and direction.** The risk before the action is one minus the posterior of the healthy combination now, the risk after is one minus the recovery probability, and the direction is `lower`, `higher` or `no change` with a dead band equal to the half width of the resampling interval of the recovery probability, or `undetermined` without resamples.

**E16 and E17, the verification.** On the first reading at or after the closure of the component's work order, every predicted summary the reading holds gets a nonconformity score, the absolute residual over the within standard deviation, and a p value, one plus the calibration weight at or above the score over n plus 1. The outcome is inconsistent when the smallest p value is at or below alpha over the number of channels compared (Bonferroni, valid under any dependence), and the Holm step down is reported beside it. The recovery check compares the predicted recovery and its resampling interval with the posterior of the healthy combination computed from the post repair readings alone, under a uniform prior over the combinations the repair can reach.

**E18, the grade.** The diagnosability grade of a start under a budget is the cost of the cheapest test sequence after which the decision rule of E10 commits to the correct faulty set, in expectation over resampled cycles of the true combination. Each rollout reads the group with the largest expected information gain per unit price, the mutual information by Monte Carlo under the joint model, charges the price plus one cycle of the true downtime, draws the reading among the recorded cycles of the true block, and stops when E10 says the evidence is sufficient and the top candidate's faulty set equals the truth. A commitment to a wrong set does not stop the search. A start with no committing rollout within the budget is inseparable under that budget. Because the grade and the declaration use one rule, an abstention is correct exactly on the starts the grade marks inseparable, which is how the verifier scores abstentions.

---

## Ranking models

| `ranking_model` | What the posterior is |
| --- | --- |
| `joint_block` | the default: E2 with the latent block mean, chosen from the calibration report |
| `joint` | the literal E2, an arm of ablation A12 |
| `discriminant_urn` | the v1 ranking model, kept for ablation A12 in `src/servers/causal_diagnosis/discriminant_urn.py`: one linear discriminant per sensor group and component reads a level from the same 34 summaries, and repeated readings of one block combine by a Polya urn around the confusion row of a cross fit within the pool, with add one smoothing of the rows and the concentrations fitted on `ALPHA_GRID` of `src/envs/hydraulic_rig/classifiers.py`. It is fitted on the same pool, so the ablation compares the observation models and nothing else. Under this arm the posterior probabilities are the urn's, replayed in time order with each distinct cycle read once at its richest group and the closed orders applied at their closing cycle, while the lookahead and the predictions keep the joint model's Gaussian predictive around them, as the assumptions state |

`DEFAULT_RANKING_MODEL` in `src/servers/causal_diagnosis/posterior.py` is the single constant read by the tools, the server, the grade and the fit, which writes it into `models.json`. The grade table is computed under `joint` or `joint_block` only.

## Abstention rules

| `abstention_rule` | `insufficient_evidence`, `separating_test` and `unseparated` |
| --- | --- |
| `decision` | the default: E10 and E12 |
| `fixed_cut` | the v1 rule, kept for ablation A8: the evidence is insufficient when the top posterior is below `abstention_cut` (the parameter of the ablation, named in the result, with `DEFAULT_ABSTENTION_CUT` as its default), the separating test is the cheapest sensor group, and a component is unseparated when its most probable level is below the cut. The expected losses of E8 to E10 are still computed and reported, so an audit sees both rules on the same evidence, and the grade keeps E10, since the grade is the reference the abstention is scored against |

---

## Ranking root causes

`rank_root_causes` replays the released readings and closed orders into the posterior of the configured ranking model, computes the costs E7 at the current cycle and the decision E10, and returns the `top_k` most probable start combinations with their current combinations, their fault codes from the knowledge base and the released values whose channel indicates a faulty component of the candidate, the fault probability and the level distribution of every component at the start and now, the attribution E4, the stability E6, the decision fields with the separating test (its group, one cycle, its price, its duration, its cost and the expected loss after it), the unseparated components with their channels, the Monte Carlo count, seed and standard errors, and the costs the decision used. The assumptions name the candidate rule, the pool, E1, E2, the block variant or the urn when they apply, the deduplication of repeated cycles, E3, E4, E6, the cost rule with the prices behind it, the abstention rule and the time convention.

---

## Estimating an intervention effect

`estimate_intervention_effect` takes `replace:<component>` or `monitor`. It moves the posterior through the action (E13), predicts every summary with its conformal interval (E14), reports the risk before and after with the direction (E15), the price and the duration of the action from the released rules (the part plus the repair charge, the lead time plus the repair downtime, and the stopped rate over the repair downtime for a repair), and appends the prediction to `predictions.json` as `pred/{n}` with the number of events and observations released so far, the time, the posterior after the action and the predicted intervals with their alpha. The assumptions flag an action the environment would refuse: a work order on a component whose order is open, a repair before a diagnosis or a declaration, or any action after the episode is over.

---

## Verifying an outcome

`verify_outcome` compares a stored prediction with the first reading released at or after the closing cycle of the component's work order that closed after the prediction, or with the first reading after the prediction for `monitor`.

| `status` | When |
| --- | --- |
| `not yet observable: the work order is open and has not closed` | an order on the component is open |
| `not yet observable: no work order on the component has closed since the prediction` | no order on the component closed after the prediction and none is open |
| `action refused: <reason>` | the environment refused a work order on the component after the prediction and none closed |
| `not yet observable: no reading since the action` | the repair closed, or `monitor` was predicted, and no reading followed |
| `not comparable: <actions> closed between the prediction and the reading` | another component's order closed after the prediction and before the reading, so the reading can show another combination |
| `verified` | a reading followed the action, and the channels, the verdict and the recovery check are filled |

A reading that follows a work order of the technician counts like one that follows the agent's own order, since the released evidence lists both as closed orders. A second order on the same component after its closure is not flagged here. The verifier of [environment.md](environment.md) counts it as unnecessary when the component was healthy.

---

## The calibration report

`src/servers/causal_diagnosis/calibration.py` writes the calibration test of the component marginals as Markdown under `benchmarks/hydraulic_rig/results/`, which git ignores:

```bash
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python -m servers.causal_diagnosis.calibration [--readings N] [--rollouts R] [--seed S] [--out DIR]
```

For every evaluation combination, the script draws rollouts of readings with replacement from its recorded block at the full sensor group, feeds each ranking model, and reports the Brier score and the calibration curve of the E5 fault probabilities by number of readings, the share of rollouts whose top candidate equals the truth by number of readings, and the coverage of the E14 interval of the true combination on the drawn cycles, under the placeholder rules. The report states that nothing in it enters code, tests or docs, and a revision of a model that it prompts is a declared decision. The default ranking model `joint_block` is such a decision.

---

## Running the server

A copy of `src/servers/causal_diagnosis/launcher.py` beside the `released.json` of an episode binds a server process to that episode. The launcher accepts optional `key=value` arguments, `ranking_model=joint_block|joint|discriminant_urn`, `abstention_rule=decision|fixed_cut` and `abstention_cut=<probability>`, forwarded to `main`, so a driver selects the arms of ablations A12 and A8 per episode. The server speaks MCP over stdio and waits for a client:

```bash
cp src/servers/causal_diagnosis/launcher.py /path/to/episode/causal_server.py
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python /path/to/episode/causal_server.py
```

The `causal-diagnosis-mcp-server` script of `pyproject.toml` starts the server without an episode, in which every tool answers an error that says so. The AssetOpsBench runners register by default only the servers of `DEFAULT_SERVER_PATHS` in `src/agent/runner.py`, so this server reaches a runner through `server_paths`. `benchmarks/hydraulic_rig/run_episodes.py` calls `ensure_models`, creates each episode and writes one launcher per server into its folder with `write_launchers` of `benchmarks/hydraulic_rig/common.py`. These launchers put the fork's `src` on the import path themselves, since a server process receives almost no environment variables, name the ranking model the server runs, and name the abstention rule when it differs from the default. The driver takes its default ranking model from `DEFAULT_RANKING_MODEL` of this package, so the two cannot drift apart.

---

## Tests

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH python -m pytest src/servers/causal_diagnosis -q
```

`src/servers/causal_diagnosis/tests/conftest.py` calls `ensure_models` once per session when the data file is present, and `src/servers/causal_diagnosis/tests/fixture_episode.py` writes a `released.json` of format 2 by hand over the recorded blocks, so the tools are tested without the action server.

| Test file | Covers |
| --- | --- |
| `src/servers/causal_diagnosis/tests/test_effect_model.py` | the design matrix, the weighted fit against least squares over cycles, the Ledoit and Wolf shrinkage, a synthetic additive world recovered from the pool, the refusal of pairs a fold cannot identify, the fit on the data file (identified, deterministic, round trips), the split of interactions in halves |
| `src/servers/causal_diagnosis/tests/test_posterior.py` | the uniform posterior without readings, a reading at the model mean, closed orders and segments, the merge of repeated cycles, the block variant against the literal model, the stability, the predictive sampling, the override by another ranking model, a posterior over given starts |
| `src/servers/causal_diagnosis/tests/test_decision.py` | the costs against the rules, the loss matrix, the seeded lookahead, an informative reading, the one component threshold, the unseparated rule |
| `src/servers/causal_diagnosis/tests/test_conformal.py` | the finite sample quantile, one unit of weight per block, the Bonferroni and Holm verdicts |
| `src/servers/causal_diagnosis/tests/test_grade.py` | the information gain, rollouts within the budget, one table per rules key, the command line |
| `src/servers/causal_diagnosis/tests/test_calibration.py` | the report is written with every section |
| `src/servers/causal_diagnosis/tests/test_adapter.py` | a worked episode from the empty ranking to a verified prediction, a second closure before the reading, an external replacement as a closed order, every ranking model, the evidence tool, the errors, the models file, the default ranking model, the fixed cut rule |
| `src/servers/causal_diagnosis/tests/test_server.py` | the four tools are read only and required, the advertised tools match the contract, the server answers over stdio with the default and with the ablation arguments |

The tests that need the data file are skipped without it, and the stdio test needs the mcp package. No test states a value computed from the rig data.

---

## Known limits

- Under the default pool, every combination with one or two faulty components is held out, so a pairwise interaction is identified on the pool only one pair at a time and in no cross fit fold. The selection therefore keeps the main effects alone and the selection log in `models.json` records every pair as not identified. This follows from the split, not from the data. A wider pool would put start combinations into the fit.
- The conformal interval of E14 is calibrated for a known combination. The posterior weighted mean with one quantile does not carry the spread of a posterior over several combinations, which the assumptions state.
- The decision of E10 compares a Monte Carlo estimate with the loss of acting, so when the saving is within the reported standard error the flag can turn with the seed. The result carries the standard error per test, and the sample count is one declared constant.
- The discriminant and urn arm keeps the fixed constants of the v1 classifiers (`SHRINKAGE`, `ALPHA_GRID`, the fold count) and fits its discriminants on the whole pool while its rows come from a cross fit within the pool, as the v1 server did. It exists to be compared with the joint model, not to be tuned.
- The grade table depends on the price table, the durations, the downtime rates, the horizon, the fault rule, the budget and the ranking model, so an ablation that changes any of them computes its own table on first use.
- Every call replays the whole released history, and the Plan-Execute executor starts one process per call. The cost is one process start, one parse of `models.json` and one replay per tool call.
