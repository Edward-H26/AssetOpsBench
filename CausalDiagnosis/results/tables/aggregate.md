# Ablation and result tables (2026-11-16)

Run folders under `runs`: 0; episodes read: 0. Intervals: a bootstrap of 10000 resamples drawn by recorded block, proportions included; paired differences match episodes on seed, scenario and every other factor; Holm at alpha 0.05 over the primary metric of a table.

No run has been played on the current code, so every table below states its design with empty cells. The cells stay empty until the test set is played on Columbia's cluster.

## A3

- What varies: the mode of answering the same question, Sc3 on layer 0: statically from the question, the facts of the cycle and the failure modes in one model call (runs of --agent static, mode static in run.json), or executed in the gym with tools and recordings by the six rows on the preset of the agents table
- Hypothesis: static answers are graded correct more often than the outcome the verifier reads in the gym warrants
- Metric: the judge pass rate of the static answers, where a static answer without a judge record, an empty answer among them, counts as not passed when its run names a judge, and the share whose parsed faulty set equals the recorded start, where an answer that the parser cannot read names no faulty component, against the recorded pass rate and the cost of the gym episodes, per model, with intervals from a bootstrap clustered by block
- What a result would mean: the first evidence for why a gym is needed, on the questions the benchmark already poses

| model | static_n | static_judge_pass | static_judge_pass_low | static_judge_pass_high | static_judge_pass_se | static_faulty_set_correct | static_faulty_set_correct_low | static_faulty_set_correct_high | static_faulty_set_correct_se | gym_n | gym_recorded_pass | gym_recorded_pass_low | gym_recorded_pass_high | gym_recorded_pass_se | gym_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## A7

- What varies: the verifier scope on the preset of the six rows, layer 0: writes only against all actions (verifier_scope of run.json), the latter from the runs of the six rows scored again by rescore_under_verifier_scope.py
- Hypothesis: scoring reads changes the ranking of agents without changing their outcomes
- Metric: the pass rate of every agent under the two scopes, where under the scope all an episode counts only when it passes and its sensor reads per group stay within those of the greedy policy on the same start (the read_gate cell), and per model Spearman's rank correlation of the agents' pass rates between the scopes
- What a result would mean: the justification of scoring writes only

| agent | model | diagnostics_server | n_writes | pass_writes | pass_writes_low | pass_writes_high | pass_writes_se | n_all | pass_all | pass_all_low | pass_all_high | pass_all_se | rank_correlation | agents_in_both_scopes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## A8

- What varies: the abstention rule of the CausalDiagnosis MCP: the fixed posterior cut (abstention_rule fixed_cut of run.json, with abstention_cut) against the decision rule E10 (abstention_rule decision, the default), per preset and level of the MCP, every other setting at its main level
- Hypothesis: the decision rule declares on inseparable episodes and commits on separable ones more often than the constant
- Metric: abstention precision (correct declarations among declarations) and recall (correct declarations among the starts the label marks inseparable) with intervals from a bootstrap clustered by block, against the reference policy's separability (label separability, the separable cell) and against the E18 grade of the MCP (label e18)
- What a result would mean: whether the threshold should follow prices

| agent | model | config | diagnostics_server | abstention_rule | label | n | unscored | abstentions | inseparable_starts | precision | precision_low | precision_high | precision_se | recall | recall_low | recall_high | recall_se | surface |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## A9

- What varies: the price table on all six big layers, En6: the full tier priced far above the rest (full_tier_high) against the default table, for the three agents without the CausalDiagnosis MCP
- Hypothesis: conclusions on cost hold under the alternative tables
- Metric: cost, normalized cost and pass per table, the paired difference of normalized cost and pass against default, Holm over the normalized cost differences
- What a result would mean: the sensitivity of the conclusions on cost to the price table

| agent | model | diagnostics_server | prices | n | cost | cost_low | cost_high | cost_se | cost_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | pass | pass_low | pass_high | pass_se | pass_n | cost_normalized_diff | cost_normalized_diff_low | cost_normalized_diff_high | cost_normalized_diff_se | cost_normalized_p | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## A10

- What varies: the guidance level L1 and L3 against L2 on layer 0 with the six rows, En7 (guidance_level of run.json)
- Hypothesis: the level an agent needs to pass is a measured quantity and differs between agents
- Metric: pass per level, the paired difference against L2, the level of the main runs, Holm over the pass differences
- What a result would mean: the information a prompt supplies

| agent | model | diagnostics_server | guidance_level | n | pass | pass_low | pass_high | pass_se | pass_n | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## A12

- What varies: the ranking model: the discriminants with the urn (ranking_model discriminant_urn) against the joint interventional likelihood of E1 and E2 (ranking_model joint), per preset, every other setting at its main level
- Hypothesis: the joint model is calibrated where the discriminants are not, and its top candidate does not fall with more readings
- Metric: per number of readings and over all readings: the Brier score of the top candidate's posterior over the rank_root_causes calls with a bootstrap interval clustered by block, and the share of calls whose top candidate is the hidden combination at the moment of the call with such an interval; A12_curves.csv holds the calibration bins (ten posterior bins with their bounds, the mean posterior and the observed frequency)
- What a result would mean: the calibration of the CausalDiagnosis MCP

| agent | model | config | ranking_model | readings | calls | brier | brier_low | brier_high | brier_se | top_candidate_correct | top1_low | top1_high | top1_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## MISDIAGNOSIS_PRICE

- What varies: the misdiagnosis price of big layer 1 at the budget of the asset, L1, on all six big layers: its arm preset of design.ARM_PRESETS against the preset of training, for the three agents without the CausalDiagnosis MCP
- Hypothesis: none, since the table describes the agents when a wrong diagnosis costs the whole budget, and no test decides
- Metric: per agent, model and preset Top-1, pass, the share of episodes whose diagnosis is wrong, the share that declared insufficient evidence, the cycles read before the first diagnosis, the cost without the misdiagnosis charge, on one scale for both presets, and the charge, each with an interval from a bootstrap that resamples the units of the starts
- What a result would mean: what the price of a wrong diagnosis changes in how the agents commit to a diagnosis

| agent | model | config | n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | pass | pass_low | pass_high | pass_se | pass_n | diagnosis_wrong | diagnosis_wrong_low | diagnosis_wrong_high | diagnosis_wrong_se | diagnosis_wrong_n | abstained | abstained_low | abstained_high | abstained_se | abstained_n | cycles_read_before_diagnosis | cycles_read_before_diagnosis_low | cycles_read_before_diagnosis_high | cycles_read_before_diagnosis_se | cycles_read_before_diagnosis_n | cost_without_charge | cost_without_charge_low | cost_without_charge_high | cost_without_charge_se | cost_without_charge_n | misdiagnosis_charge | misdiagnosis_charge_low | misdiagnosis_charge_high | misdiagnosis_charge_se | misdiagnosis_charge_n | surface | diagnostics_server |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## SWITCHES

- What varies: the options of layers 2, 3 and 5 on all six big layers, each arm preset of design.ARM_PRESETS of those layers against the preset of training, for the three agents without the CausalDiagnosis MCP: the thinking time off (L2), the declared action durations off (L3), and the deterioration off, the fixed schedule of level drops against the random hazard rates and the failure rule off (L5)
- Hypothesis: each option of a layer changes the pass of the full environment, as the cumulative layer study ranks the layers
- Metric: pass, top-1, normalized cost, detection delay and repair delay per preset, the paired differences of pass and top-1 against the preset of training on the same starts, matched across the clock modes of the two presets, Holm over the pass differences
- What a result would mean: what each option of a layer adds on the full environment

| agent | model | diagnostics_server | config | n | pass | pass_low | pass_high | pass_se | pass_n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | detection_delay | detection_delay_low | detection_delay_high | detection_delay_se | detection_delay_n | repair_delay | repair_delay_low | repair_delay_high | repair_delay_se | repair_delay_n | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## GEPA

- What varies: the online loop of GEPA on the preset of training (design.TRAINING_PRESET by default) under the feedback of the verifier and under the control that sends the score alone, one row per finished run of gepa_prompt_optimization.py
- Hypothesis: none, since the table states what each loop did, and the gain of its final block over no block is read in TRAINING_pairs from the evaluation runs, which play the test set once
- Metric: per finished run of gepa_prompt_optimization.py (status finished in gepa_online.json), the feedback (the column card), the reflection model, the candidates that played on the stream, the stream episodes with every member's validation on the blocks counted, the stream episodes the loop left out since they reached a single fault of the fold its model holds out, and the episodes played until the final candidate was first chosen
- What a result would mean: what each loop spent and where its final block came from, the trainable interface of the environment for the task block

| run | model | config | card | reflection_model | candidates_played | stream_episodes | left_out | rollouts_to_best |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## GEPA_curve

- What varies: the round of the online loop of GEPA on the preset of training, under the feedback of the verifier and under the control that sends the score alone
- Hypothesis: the prompt in use scores higher on each new block with the feedback of the verifier than with the score alone, from the same first round on
- Metric: per GEPA run, feedback (the column card) and round, the episodes that every candidate played in the rounds before it, validation counted, so none at the first round, the mean reward of the prompt in use on the block of the round, scored before GEPA reflects on it, and the episodes of that mean
- What a result would mean: whether the verifier's feedback teaches the prompt more than its score does, the environment's online training signal

| run | model | config | card | reflection_model | round | rollouts | mean_reward | n |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## AGENTS

- What varies: the six rows of design.AGENT_ROWS: Plan-and-Execute, Agents-as-Tools and the CausalDiagnosis agent, each without and with the CausalDiagnosis MCP (the agent label and diagnostics_server of scores.csv in gym mode), each clock mode, tool surface and training variant of a preset a row of its own
- Hypothesis: the CausalDiagnosis agent does better than each AssetOpsBench agent and the CausalDiagnosis MCP helps every agent, on top-1 and pass, which AGENTS_pairs decides
- Metric: pass, top-1 accuracy (diagnosis_all_correct), success, which credits a correct declaration of insufficient evidence, the verifier's reward, the share of episodes whose submitted diagnosis is wrong (diagnosis_wrong), which no declaration and no episode without a diagnosis counts toward, the share of episodes that declared insufficient evidence (abstained), the share of episodes that cited an evidence identifier the released evidence does not hold (unmatched_evidence, the complement of provenance_ok), the model calls per episode (llm_calls), cost, normalized cost, the production loss within the cost (downtime_total), the time to work order and the time to repair in cycles over the episodes that have one (detection_delay and repair_delay) per condition (agent label, model, config, clock mode, tool surface, level of the MCP, guidance level, verifier scope, price table, ranking model, abstention rule and training variant), each with an interval from a bootstrap clustered by block, and the turns to a correct diagnosis as the restricted mean survival time over turns, censored at the step limit of the run (turns_rmst), with such an interval, and Cohen's kappa between the judge's majority pass under the five call protocol and the verified pass over the episodes that carry both (judge_kappa), with such an interval, empty when either rule puts every episode on one side, and pass^k of tau-bench for every k from one to the start states that the test set plays of a start combination (pass_k1 and on), the share of the k-subsets of the episodes of a combination in which every episode passes, averaged over the combinations with at least k episodes, with an interval from a bootstrap that resamples the combinations
- What a result would mean: the headline of the paper: the outcome of the six rows on which the hypotheses of AGENTS_pairs are decided

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | played | planned | unscored | unjudged | incomplete | refused | replicates | read | pass | pass_low | pass_high | pass_se | pass_n | pass_k1 | pass_k1_low | pass_k1_high | pass_k1_se | pass_k1_n | pass_k2 | pass_k2_low | pass_k2_high | pass_k2_se | pass_k2_n | pass_k3 | pass_k3_low | pass_k3_high | pass_k3_se | pass_k3_n | pass_k4 | pass_k4_low | pass_k4_high | pass_k4_se | pass_k4_n | pass_k5 | pass_k5_low | pass_k5_high | pass_k5_se | pass_k5_n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | faulty_set_right | faulty_set_right_low | faulty_set_right_high | faulty_set_right_se | faulty_set_right_n | restored_clean | restored_clean_low | restored_clean_high | restored_clean_se | restored_clean_n | graded_top1 | graded_top1_low | graded_top1_high | graded_top1_se | graded_top1_n | graded_pass | graded_pass_low | graded_pass_high | graded_pass_se | graded_pass_n | normalized_score | normalized_score_n | success | success_low | success_high | success_se | success_n | reward | reward_low | reward_high | reward_se | reward_n | diagnosis_wrong | diagnosis_wrong_low | diagnosis_wrong_high | diagnosis_wrong_se | diagnosis_wrong_n | abstained | abstained_low | abstained_high | abstained_se | abstained_n | unmatched_evidence | unmatched_evidence_low | unmatched_evidence_high | unmatched_evidence_se | unmatched_evidence_n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | cost | cost_low | cost_high | cost_se | cost_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | downtime_total | downtime_total_low | downtime_total_high | downtime_total_se | downtime_total_n | turns_rmst | turns_rmst_low | turns_rmst_high | turns_rmst_se | turns_rmst_n | detection_delay | detection_delay_low | detection_delay_high | detection_delay_se | detection_delay_n | repair_delay | repair_delay_low | repair_delay_high | repair_delay_se | repair_delay_n | judge_kappa | judge_kappa_low | judge_kappa_high | judge_kappa_se | judge_kappa_n | faulty_set_right_baseline_k | faulty_set_right_baseline_n | faulty_set_right_baseline | faulty_set_right_floor | faulty_set_right_at_floor | replayed | replays | pass_replays_failed | diagnosis_all_correct_replays_failed | faulty_set_right_replays_failed | restored_clean_replays_failed | success_replays_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## AGENTS_pairs

- What varies: the hypotheses of the six rows on the preset of the agents table, layer 0: H1a to H1d pair the CausalDiagnosis agent against each AssetOpsBench agent, both without the CausalDiagnosis MCP and both with it, H2a to H2c every agent with the MCP against itself without it (design.HYPOTHESES), and the scoring pairs every two of the six rows
- Hypothesis: every hypothesis comparison holds, its paired differences of top-1 and pass both above zero with both Holm decisions rejecting, and scoring by the verified pass reverses the judge's order of at least one pair of the six rows
- Metric: on the hypothesis rows the paired difference of top-1 accuracy and of pass, which decide, of success, which credits a correct declaration of insufficient evidence, and of the verifier's reward, the ratio of mean cost with a bootstrap interval clustered by block and the p value of the paired cost difference, and the paired difference of the turns to a correct diagnosis capped at the step limit, the difference of the two restricted mean survival times over turns (turns_rmst_diff), which no Holm decision covers, on the scoring rows the paired differences of the verified pass and of the judge's pass, and Holm within each of the two families of a model and tool surface with one decision per measure, the top-1 and pass tests of the hypotheses, 14 when every row has run, and apart from them the two pass tests of the scoring rows, with the number of tests in family_tests. holds says whether a hypothesis holds, empty while a cell it needs has no run. Episodes match on seed, scenario and every factor other than agent and diagnostics_server, among the runs at the main level of every setting (the clock mode, guidance level and verifier scope of the preset, the decision rule, the default prices, the default ranking model and the agent as released)
- What a result would mean: the decision on the hypotheses of the six rows, and on whether scoring by verified outcomes reorders the judge's ranking

| hypothesis | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | pairs | clusters | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | success_diff | success_diff_low | success_diff_high | success_diff_se | success_p | reward_diff | reward_diff_low | reward_diff_high | reward_diff_se | reward_p | judge_pass_diff | judge_pass_diff_low | judge_pass_diff_high | judge_pass_diff_se | judge_pass_p | cost_ratio | cost_ratio_low | cost_ratio_high | cost_ratio_se | cost_ratio_p | turns_rmst_diff | turns_rmst_diff_low | turns_rmst_diff_high | turns_rmst_diff_se | turns_rmst_p | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | pass_holm_p | pass_reject | judge_pass_holm_p | judge_pass_reject | reversed | holds | family_tests | reference_pass_k | reference_pass_n | at_floor | status | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## H1_GUIDANCE

- What varies: the guidance level of the briefing at level 3 instead of the main level, the En7 runs of the six rows on the preset of the main table (guidance_level of run.json), for the comparisons of H1 in design.HYPOTHESES
- Hypothesis: the advantage of the CausalDiagnosis agent over each AssetOpsBench agent does not depend on how much guidance the briefing gives, a robustness check
- Metric: the paired differences of top-1 and pass, matched on seed, scenario and every factor other than the agent and the level of the MCP, with intervals from a bootstrap clustered by block, sign flip p values over the blocks and a Holm of their own over these tests, apart from H1, with the floor of each baseline
- What a result would mean: whether H1 holds without the guidance of the main level

| hypothesis | agent | agent_server | reference | reference_server | model | config | surface | guidance_level | status | pairs | clusters | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | pass_holm_p | pass_reject | family_tests | holds | reference_pass_k | reference_pass_n | at_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## H2PRIME_pairs

- What varies: the CausalDiagnosis MCP on the top layer, all six big layers on: each agent of design.TRAINING_ROWS as released against itself without the MCP (design.TRAINING_BASELINES), the comparisons of design.H2_PRIME
- Hypothesis: the MCP helps every agent where every big layer of the environment is on, as H2 claims on layer 0
- Metric: the paired differences of pass and of the verifier's reward (design.H2_PRIME_MEASURES), matched on seed, scenario and every factor other than the agent and the level of the MCP, with intervals from a bootstrap clustered by block, sign flip p values over the blocks and Holm run once with the family h1_top over the registered size of their group agents_environment on the same model and tool surface
- What a result would mean: whether the advantage of the MCP survives the layers that make the environment realistic

| hypothesis | agent | agent_server | reference | reference_server | model | config | surface | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | reward_diff | reward_diff_low | reward_diff_high | reward_diff_se | reward_p | pass_holm_p | pass_reject | reward_holm_p | reward_reject | family_tests | holds | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## H0

- What varies: the scoring of an episode by the AssetOpsBench judge's majority under the registered protocol against the verified pass, on the six rows of design.AGENT_ROWS and the comparisons of design.HYPOTHESES on the preset of the main table
- Hypothesis: the judge passes a different share of episodes than the verifier and decides the comparisons between rows differently
- Metric: per row the mean of the judge's pass minus the verified pass over its episodes, and per comparison the judge's paired difference minus the verified one over the matched episodes, each with an interval from a bootstrap clustered by block and a sign flip p value over the blocks, Holm over the row and comparison tests in one family, an episode without a judge record counted as a failure of the judge, a test unrun on a row that misses a planned episode or is refused, on an episode judged under another protocol or without any judge record, with p equal to one and counted in the family, every other test with its computed p value, and beside each test, deciding nothing, Cohen's kappa, the discordance and the false pass and false fail rates of a row, and the two paired differences and the reversed flag of a comparison
- What a result would mean: how far the judge that scores the published benchmark agrees with what the agent did

| test | name | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | clusters | nonzero_starts | statistic | statistic_low | statistic_high | statistic_se | h0_p | status | h0_holm_p | h0_reject | family_tests | judge_kappa | judge_kappa_low | judge_kappa_high | judge_kappa_se | judge_kappa_n | discordance | false_pass | false_fail | judge_diff | verified_diff | reversed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## A13_A17

- What varies: the switches of the CausalDiagnosis agent, each turned alone on the same episodes and read from the label suffixes of CausalLoopOptions.label(): candidate troubleshooting plans off (Ag1, nobranch), plans for the top candidate only (Ag2, single), plan length one (Ag3, len1), and the read the board advises, by the discriminating test of E10 (Ag7, e10) or by one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray, not their algorithm (Ag7, ec2), while the model decides every step in each
- Hypothesis: every switch lowers pass or raises the model calls per episode, and no switch improves both over the full agent
- Metric: per switch with intervals, and tool surface at the main level of every setting as in AGENTS_pairs, pass, unnecessary replacements, model calls, thinking cycles (the sum of think_cycles over the hidden log), deviations, the share of test first plans, cost, the price of the reads (test_cost), top-1 accuracy on the starts whose first ranking puts two levels of one faulty set on top (level_accuracy), the repair sets ordered again and the realized elimination per read, and the paired difference of pass and model calls against the full agent with the agent label left out of the pairing key, Holm over the pass differences
- What a result would mean: what each element of the agent is worth on its own

| agent | switch | model | config | surface | n | pass | pass_low | pass_high | pass_se | pass_n | unnecessary | unnecessary_low | unnecessary_high | unnecessary_se | unnecessary_n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | deviations | deviations_low | deviations_high | deviations_se | deviations_n | test_first | test_first_low | test_first_high | test_first_se | test_first_n | cost | cost_low | cost_high | cost_se | cost_n | test_cost | test_cost_low | test_cost_high | test_cost_se | test_cost_n | level_accuracy | level_accuracy_low | level_accuracy_high | level_accuracy_se | level_accuracy_n | repeated_repair_sets | repeated_repair_sets_low | repeated_repair_sets_high | repeated_repair_sets_se | repeated_repair_sets_n | realized_elimination_per_read | realized_elimination_per_read_low | realized_elimination_per_read_high | realized_elimination_per_read_se | realized_elimination_per_read_n | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | llm_calls_diff | llm_calls_diff_low | llm_calls_diff_high | llm_calls_diff_se | llm_calls_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## CALLS

- What varies: the agent and the clock mode (the agent label, diagnostics_server and clock_mode of scores.csv), the tool surface and the other settings of the condition held fixed in a row
- Hypothesis: the CausalDiagnosis agent calls the model less often per episode and is charged fewer cycles for thinking than both rows of the agent that decides every step
- Metric: model calls per episode, cycles charged for thinking (the sum of think_cycles over the hidden log, nonzero in the generation_time mode only) and tokens per episode, input and output together, with bootstrap intervals clustered by block, the paired difference of the first two against Agents-as-Tools with the CausalDiagnosis MCP in the same clock mode and, under the prefix vs_bare_, against Agent-As-Tool without the MCP, Holm over the model call differences against the row with the MCP
- What a result would mean: whether planning over candidate troubleshooting plans saves the thinking time that a degrading asset charges as downtime

| agent | model | config | diagnostics_server | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | tokens | tokens_low | tokens_high | tokens_se | tokens_n | llm_calls_diff | llm_calls_diff_low | llm_calls_diff_high | llm_calls_diff_se | llm_calls_p | thinking_cycles_diff | thinking_cycles_diff_low | thinking_cycles_diff_high | thinking_cycles_diff_se | thinking_cycles_p | status | pairs | clusters | holm_p | reject | vs_bare_llm_calls_diff | vs_bare_llm_calls_diff_low | vs_bare_llm_calls_diff_high | vs_bare_llm_calls_diff_se | vs_bare_llm_calls_p | vs_bare_thinking_cycles_diff | vs_bare_thinking_cycles_diff_low | vs_bare_thinking_cycles_diff_high | vs_bare_thinking_cycles_diff_se | vs_bare_thinking_cycles_p | vs_bare_pairs | vs_bare_clusters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## THINKING

- What varies: the agent, the preset and the training variant among the conditions played in the generation_time mode, the only mode that charges the time between calls
- Hypothesis: none, since the table shows what the charged thinking time is made of, the model's generation, a sleep of the machine or a step of the wall clock, and the rest, serving, tools and the harness
- Metric: the cycles charged for thinking and the seconds charged, which the clock measures on the monotonic clock after the credits of the failed model attempts, with, of these, the wall seconds of the model calls inside the charged span (generation_seconds, from started_at and ended_at of llm_calls.json), the wall gap minus the monotonic gap (wall_minus_monotonic_seconds, which the monotonic clock leaves out when the machine sleeps or the wall clock is set, and which cannot see a stopped process, since both clocks run while a process is stopped) and what remains (rest_seconds), each the mean over the episodes with an interval from a bootstrap clustered by block, the episodes played before the split was recorded (unsplit) and the episodes of a trained variant read from the runs of no training on a fold without an adapter (aliased)
- What a result would mean: how much of the thinking time that a degrading asset charges comes from the model, and whether a run lost time to a sleep or a clock step, which reads as a suspect run

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | unsplit | aliased | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | seconds_charged | seconds_charged_low | seconds_charged_high | seconds_charged_se | seconds_charged_n | generation_seconds | generation_seconds_low | generation_seconds_high | generation_seconds_se | generation_seconds_n | wall_minus_monotonic_seconds | wall_minus_monotonic_seconds_low | wall_minus_monotonic_seconds_high | wall_minus_monotonic_seconds_se | wall_minus_monotonic_seconds_n | rest_seconds | rest_seconds_low | rest_seconds_high | rest_seconds_se | rest_seconds_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## STRATA

- What varies: the stratum of the episode: an episode with a deterioration transition, scheduled or drawn by the hazard (the hidden log), a start separable and a start inseparable under the reference policy's separability (the separable cell), the same under the E18 grade of the CausalDiagnosis MCP (the e18 record of the grade cell), a start that both mark inseparable, where a declaration earns its credit, and an episode whose separating test exceeded the budget left at some rank_root_causes call that said read first (the trajectory)
- Hypothesis: the advantage of the CausalDiagnosis agent concentrates on the episodes the design targets
- Metric: pass, success and the share of episodes that declared insufficient evidence (abstained) per condition (agent label, model, config, clock mode, tool surface, level of the MCP, guidance level, verifier scope, price table, ranking model, abstention rule and training variant) and stratum, each with an interval from a bootstrap clustered by block, and the number of episodes of the stratum
- What a result would mean: whether the mechanism of the agent, checks that catch a change and plans that act or repair as a test where the rule declares, is what produces its advantage

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | stratum | n | pass | pass_low | pass_high | pass_se | pass_n | success | success_low | success_high | success_se | success_n | abstained | abstained_low | abstained_high | abstained_se | abstained_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## JUDGE

- What varies: the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the protocol and the judge model of the judge records of its episodes, with the metrics of Figure 5 of AssetOpsBench on the trajectories that the verifier scores
- Hypothesis: none, since the paper reports the three scores beside the verified outcome and the scoring pairs read the judge's pass in AGENTS_pairs
- Metric: per condition, judge protocol and judge model, task completion, data retrieval accuracy and result verification, the first three criteria of the judge prompt, each as the mean over the episodes of the share of the planned calls that answered true, where a call without a parsed review and an episode without a record count as false, with an interval from a bootstrap clustered by block and the standard deviation over episodes (_std), and the share of the episodes whose majority of the planned calls answered true with such an interval (_majority), the episodes of the condition (n), the calls without a parsed review (failed_calls), the episodes without a record (episodes_without_record), Cohen's kappa of the judge's majority against four constructs of the verifier with the counts of each 2 by 2 table, the share of the judged episodes whose input was cut to the context of the served judge (judge_record_cut) and the share of the judged episodes that the judge passes while the asset is not recovered (judge_passed_unrecovered)
- What a result would mean: what answer scoring reads in the runs whose outcome the verifier scores

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | judge_protocol | judge_model | n | failed_calls | episodes_without_record | declared | task_completion | task_completion_low | task_completion_high | task_completion_se | task_completion_n | task_completion_std | task_completion_majority | task_completion_majority_low | task_completion_majority_high | task_completion_majority_se | task_completion_majority_n | data_retrieval_accuracy | data_retrieval_accuracy_low | data_retrieval_accuracy_high | data_retrieval_accuracy_se | data_retrieval_accuracy_n | data_retrieval_accuracy_std | data_retrieval_accuracy_majority | data_retrieval_accuracy_majority_low | data_retrieval_accuracy_majority_high | data_retrieval_accuracy_majority_se | data_retrieval_accuracy_majority_n | generalized_result_verification | generalized_result_verification_low | generalized_result_verification_high | generalized_result_verification_se | generalized_result_verification_n | generalized_result_verification_std | generalized_result_verification_majority | generalized_result_verification_majority_low | generalized_result_verification_majority_high | generalized_result_verification_majority_se | generalized_result_verification_majority_n | kappa_pass | kappa_pass_low | kappa_pass_high | kappa_pass_se | pass_both | pass_judge_only | pass_verifier_only | pass_neither | kappa_diagnosis_all_correct | kappa_diagnosis_all_correct_low | kappa_diagnosis_all_correct_high | kappa_diagnosis_all_correct_se | diagnosis_all_correct_both | diagnosis_all_correct_judge_only | diagnosis_all_correct_verifier_only | diagnosis_all_correct_neither | kappa_faulty_set_right | kappa_faulty_set_right_low | kappa_faulty_set_right_high | kappa_faulty_set_right_se | faulty_set_right_both | faulty_set_right_judge_only | faulty_set_right_verifier_only | faulty_set_right_neither | kappa_restored_clean | kappa_restored_clean_low | kappa_restored_clean_high | kappa_restored_clean_se | restored_clean_both | restored_clean_judge_only | restored_clean_verifier_only | restored_clean_neither | judge_record_cut | judge_record_cut_low | judge_record_cut_high | judge_record_cut_se | judge_record_cut_n | judge_passed_unrecovered | judge_passed_unrecovered_low | judge_passed_unrecovered_high | judge_passed_unrecovered_se | judge_passed_unrecovered_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## JUDGE_AGREEMENT

- What varies: the runs of the six rows together per model, config and settings at the main level of every setting, under a judge protocol with more than one call, with the metrics of Table 14 of AssetOpsBench
- Hypothesis: none, since the table checks how stable the judge is under repeated calls
- Metric: per criterion the mean over the judged episodes of the mean pairwise agreement of the parsed calls of an episode, with an interval from a bootstrap clustered by block, and the uncertainty, one minus the agreement, whose formula Table 14 of AssetOpsBench states while it prints none for the agreement
- What a result would mean: whether the five calls of the judge agree enough for their mean to be read as a score

| model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | judge_protocol | n | task_completion_agreement | task_completion_agreement_low | task_completion_agreement_high | task_completion_agreement_se | task_completion_agreement_n | task_completion_uncertainty | data_retrieval_accuracy_agreement | data_retrieval_accuracy_agreement_low | data_retrieval_accuracy_agreement_high | data_retrieval_accuracy_agreement_se | data_retrieval_accuracy_agreement_n | data_retrieval_accuracy_uncertainty | generalized_result_verification_agreement | generalized_result_verification_agreement_low | generalized_result_verification_agreement_high | generalized_result_verification_agreement_se | generalized_result_verification_agreement_n | generalized_result_verification_uncertainty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## EFFICIENCY

- What varies: the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the number of faulty components at the start, one or two, which take the place of the single-agent and multi-agent tasks of Table 3 of AssetOpsBench
- Hypothesis: none, since the paper reports the steps and the runtime as what a row costs to run
- Metric: per condition and fault count, the episodes, the steps per episode (tool_invocations, the tool calls that the state files of the action or metered servers record and the calls to the CausalDiagnosis MCP together) and the runtime per episode in seconds (wall_s, the wall clock around the agent, which stops before the verifier and the judge), each with an interval from a bootstrap clustered by block and the standard deviation over episodes (_std)
- What a result would mean: what a row costs to run, as AssetOpsBench reports its frameworks

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | faults | n | tool_invocations | tool_invocations_low | tool_invocations_high | tool_invocations_se | tool_invocations_n | tool_invocations_std | wall_s | wall_s_low | wall_s_high | wall_s_se | wall_s_n | wall_s_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## PLANS

- What varies: the row of AGENTS that makes a plan, Plan-and-Execute and the CausalDiagnosis agent, each clock mode, tool surface and training variant of a preset a row of its own, with the metrics of Table 12 of AssetOpsBench
- Hypothesis: none, since the paper reports the length of the plans as a description of the rows
- Metric: per condition the planned steps per episode (planned_steps, the steps of the plan of Plan-and-Execute and of all the plans of the CausalDiagnosis agent in an episode) with an interval from a bootstrap clustered by block, the standard deviation over episodes (_std), the smallest and largest value and the episodes that planned no step (_zero)
- What a result would mean: how far ahead a row plans, as AssetOpsBench reports its models

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | planned_steps | planned_steps_low | planned_steps_high | planned_steps_se | planned_steps_n | planned_steps_std | planned_steps_min | planned_steps_max | planned_steps_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## ELIM

- What varies: the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the reference policies of the action server, every row measured by one replay of the filter of the CausalDiagnosis MCP over the released evidence (realized_read_elimination.py), whether or not its run had the MCP
- Hypothesis: none, since the paper reports the realized elimination as a score beside the hypotheses, and its gap to the expected elimination as a check of the MCP's forecast
- Metric: per condition the episodes, the paid reads and the reads with a forecast of the belief they met, the digest of the models.json the replay read (yardstick), and with intervals from a bootstrap clustered by block the realized elimination per paid read and per unit of the cost of E21 as the episode charged it, the effective number of hypotheses of the belief predicted to the cycle of a read minus that after its update, and over the reads that the MCP forecast on the belief they met the mean expected elimination, the mean realized elimination and their gap
- What a result would mean: how many hypotheses the reads an agent pays for rule out, a yardstick of the MCP's model of the asset and not of the agent's belief

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | paid_reads | calibrated_reads | yardstick | realized_elimination_per_read | realized_elimination_per_read_low | realized_elimination_per_read_high | realized_elimination_per_read_se | realized_elimination_per_read_n | realized_elimination_per_cost | realized_elimination_per_cost_low | realized_elimination_per_cost_high | realized_elimination_per_cost_se | realized_elimination_per_cost_n | expected_elimination | expected_elimination_low | expected_elimination_high | expected_elimination_se | expected_elimination_n | realized_elimination_calibrated | realized_elimination_calibrated_low | realized_elimination_calibrated_high | realized_elimination_calibrated_se | realized_elimination_calibrated_n | calibration_gap | calibration_gap_low | calibration_gap_high | calibration_gap_se | calibration_gap_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## ELIM_pairs

- What varies: the comparisons of the hypotheses of AGENTS_pairs, on the realized elimination of the paid reads
- Hypothesis: none, since no hypothesis of the paper reads the realized elimination, so no Holm decision covers these differences
- Metric: the paired difference of the realized elimination per unit of cost and per paid read with an interval from a bootstrap clustered by block and its p value, matched on seed, scenario and every factor other than agent and diagnostics_server among the runs at the main level of every setting, as in AGENTS_pairs
- What a result would mean: whether the MCP and the agent buy their pass with reads that rule out hypotheses

| hypothesis | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | status | pairs | clusters | realized_elimination_per_cost_diff | realized_elimination_per_cost_diff_low | realized_elimination_per_cost_diff_high | realized_elimination_per_cost_diff_se | realized_elimination_per_cost_p | realized_elimination_per_read_diff | realized_elimination_per_read_diff_low | realized_elimination_per_read_diff_high | realized_elimination_per_read_diff_se | realized_elimination_per_read_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## MCP

- What varies: the agents with the CausalDiagnosis MCP, each on every preset, clock mode and tool surface, each condition a row of its own
- Hypothesis: none, since the measures of the MCP are descriptive
- Metric: per condition, with intervals from a bootstrap that resamples the starts: the uptake of the advice over the visits (the first plan and any plan), the harm of the followed first plan and the right deviations, the pass when the first visit follows or deviates, the uptake of the advised reads and of the first candidate as the diagnosis, the Brier skill and the bias of the recovery rate prediction at the reading that verifies a plan, on a row with prices the bias and the mean absolute error of the cost prediction until the asset is restored, scaled by the cost of the oracle, beside the price and the downtime of the plan alone and the skill over it, the coverage of the predicted intervals, the error of the predicted readings against holding the current readings, the false alarms and misses of the post repair check recomputed offline, AC@1, AC@k, the mean reciprocal rank, the precision and the recall of the ranking at the truth of each answer, the production lost while thinking with its share of the production lost, and its paired difference against the agent without the MCP (mcp_metrics.py)
- What a result would mean: what the MCP adds, what it charges and what it risks, beside the hypotheses

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | unaligned | unplaced | yardstick | uptake | uptake_low | uptake_high | uptake_se | uptake_n | uptake_any | uptake_any_low | uptake_any_high | uptake_any_se | uptake_any_n | harm | harm_low | harm_high | harm_se | harm_n | good_deviation | good_deviation_low | good_deviation_high | good_deviation_se | good_deviation_n | pass_follow | pass_follow_low | pass_follow_high | pass_follow_se | pass_follow_n | pass_deviate | pass_deviate_low | pass_deviate_high | pass_deviate_se | pass_deviate_n | read_uptake | read_uptake_low | read_uptake_high | read_uptake_se | read_uptake_n | diagnosis_uptake | diagnosis_uptake_low | diagnosis_uptake_high | diagnosis_uptake_se | diagnosis_uptake_n | recovery_bss | recovery_bss_low | recovery_bss_high | recovery_bss_se | recovery_bss_n | recovery_bias | recovery_bias_low | recovery_bias_high | recovery_bias_se | recovery_bias_n | cost_bias | cost_bias_low | cost_bias_high | cost_bias_se | cost_bias_n | cost_mae | cost_mae_low | cost_mae_high | cost_mae_se | cost_mae_n | price_mae | price_mae_low | price_mae_high | price_mae_se | price_mae_n | cost_skill | cost_skill_low | cost_skill_high | cost_skill_se | cost_skill_n | coverage | coverage_low | coverage_high | coverage_se | coverage_n | readings_skill | readings_skill_low | readings_skill_high | readings_skill_se | readings_skill_n | false_alarm | false_alarm_low | false_alarm_high | false_alarm_se | false_alarm_n | miss | miss_low | miss_high | miss_se | miss_n | ac1 | ac1_low | ac1_high | ac1_se | ac1_n | ack | ack_low | ack_high | ack_se | ack_n | mrr | mrr_low | mrr_high | mrr_se | mrr_n | precision | precision_low | precision_high | precision_se | precision_n | recall | recall_low | recall_high | recall_se | recall_n | think_loss_share | think_loss_share_low | think_loss_share_high | think_loss_share_se | think_loss_share_n | think_failures | think_failures_low | think_failures_high | think_failures_se | think_failures_n | think_loss | think_loss_low | think_loss_high | think_loss_se | think_loss_n | think_loss_diff | think_loss_diff_low | think_loss_diff_high | think_loss_diff_se | pairs | clusters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## POLICIES

- What varies: the reference policy of design.REFERENCE_POLICIES (oracle, greedy reference, random, do nothing and always declare) on every preset, the rows of the environment of design.LADDER among them, each clock mode and tool surface a row of its own
- Hypothesis: the oracle is the best of the five reference policies, with a reward at least that of every other reference policy, and doing nothing scores below every policy that the verifier passes, on every start of every layer
- Metric: per policy, preset, clock mode and tool surface, the pass, the verifier's success and the mean reward over the gym episodes at the main level of every setting, each with an interval from a bootstrap clustered by block
- What a result would mean: how far an agent stands from the policy that knows the hidden state and from the policies that need no model, row by row

| agent | config | clock_mode | surface | n | pass | pass_low | pass_high | pass_se | pass_n | success | success_low | success_high | success_se | success_n | reward | reward_low | reward_high | reward_se | reward_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## LAYERS_pairs

- What varies: the big layer of the environment that a row of design.LADDER adds to the row below it, played by the three rows with the CausalDiagnosis MCP (design.LADDER_ROWS)
- Hypothesis: every big layer changes the pass, pooled over the three agents
- Metric: per model, tool surface and layer, the paired difference of pass between the preset of the layer (config) and the preset below it (reference_config) for each of the three rows and pooled over their pairs (agent pooled), matched on seed, scenario and every factor other than the preset and the clock mode, with intervals from a bootstrap clustered by block, and Holm over the pooled tests of the five layers
- What a result would mean: whether each big layer of the environment measures something the layer below it does not

| model | surface | config | reference_config | agent | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | holm_p | reject | pass_holm_p | pass_reject | family_tests | reference_pass_k | reference_pass_n | at_floor | pass_sign | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## TRAINING

- What varies: the training variant of design.TRAINING_VARIANTS (as released, GEPA, GEPA on the score alone, fine tuning, fine tuning then GEPA, and fine-tuning on unfiltered trajectories) of the three rows with the CausalDiagnosis MCP (design.TRAINING_ROWS) on the preset of training, all six big layers (design.TRAINING_PRESET), from the training variant of run.json, and the three agents without the MCP and without training on that preset (design.TRAINING_BASELINES) as the baselines, the top layer on the same seeds
- Hypothesis: training in the environment raises pass and top-1 on fault combinations that the trained model never saw
- Metric: per model, tool surface, row, training variant and subset, all for every episode of the test set, each played under cross-fitting by the model that holds out the fold of its combination, single and two for the single fault and the two fault starts, the pass and the top-1 accuracy, the three scores of the judge under the five call protocol with an episode without a judge record counted as false, the share of episodes that cited an evidence identifier the released evidence does not hold (unmatched_evidence) and Cohen's kappa between the judge's majority pass under the five call protocol and the verified pass (judge_kappa), each with an interval from a bootstrap clustered by block
- What a result would mean: whether the environment supports learning that carries over to fault combinations never trained on

| model | config | surface | agent | diagnostics_server | training_variant | subset | n | pass | pass_low | pass_high | pass_se | pass_n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | judged | aliased | episodes_without_record | failed_calls | declared | task_completion | task_completion_low | task_completion_high | task_completion_se | task_completion_n | data_retrieval_accuracy | data_retrieval_accuracy_low | data_retrieval_accuracy_high | data_retrieval_accuracy_se | data_retrieval_accuracy_n | generalized_result_verification | generalized_result_verification_low | generalized_result_verification_high | generalized_result_verification_se | generalized_result_verification_n | unmatched_evidence | unmatched_evidence_low | unmatched_evidence_high | unmatched_evidence_se | unmatched_evidence_n | judge_kappa | judge_kappa_low | judge_kappa_high | judge_kappa_se | judge_kappa_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## TRAINING_pairs

- What varies: the training comparisons of design.TRAINING_COMPARISONS per row with the CausalDiagnosis MCP: GEPA, fine tuning and fine tuning then GEPA against the agent as released, GEPA against GEPA on the score alone, and fine tuning against its control, fine-tuning on unfiltered trajectories, on as many episodes
- Hypothesis: each trained variant passes and diagnoses more often than the agent as released, GEPA more often than GEPA on the score alone, and fine tuning more often than its control, fine-tuning on unfiltered trajectories, on fault combinations the trained model never saw, so a gain comes from the environment's signal and not from more exposure to the task
- Metric: the paired differences of pass and top-1, matched on seed, scenario and every factor, with intervals from a bootstrap clustered by block: over every episode of the test set (subset all) with their p values and Holm over the tests of a model and tool surface, 30 when every variant has run, with the number of tests in family_tests, and on the single fault and two fault starts (subsets single and two) and the episodes of each fold of the cross-fitting (fold_1 and fold_2) with no test
- What a result would mean: the decision on the hypotheses on training

| model | config | surface | agent | diagnostics_server | training_variant | reference_variant | comparison | subset | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | pass_holm_p | pass_reject | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | family_tests | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## TRAINING_RECORD

- What varies: the row with the CausalDiagnosis MCP on the preset of training, from run.json of its trained evaluation runs, where the training pipeline records the adapter and the task block a run plays
- Hypothesis: none, since the table states what each agent trained on, the same budget for the three
- Metric: per agent, model and fold of the cross-fitting the episodes sampled, the episodes the filter on the verified pass kept, the model calls used as examples, those the length cap left out and the optimizer steps of the filtered adapter, the trajectories and the same three counts of the control, fine-tuning on unfiltered trajectories, and the candidates and stream episodes of GEPA and of GEPA on the score alone on the agent without training and of GEPA on the fine tuned agent, and the training episodes each left out (design.TRAINING_RECORD_SOURCES)
- What a result would mean: whether the three agents trained on the same budget

| agent | model | fold | episodes_sampled | kept_by_filter | model_calls_as_examples | examples_over_max_length | optimizer_steps | control_trajectories | control_model_calls_as_examples | control_examples_over_max_length | control_optimizer_steps | gepa_candidates | gepa_stream_episodes | ft_gepa_candidates | ft_gepa_stream_episodes | gepa_score_only_candidates | gepa_score_only_stream_episodes | episodes_visiting_test_combinations | control_episodes_visiting_test_combinations | gepa_episodes_visiting_test_combinations | gepa_score_only_episodes_visiting_test_combinations | ft_gepa_episodes_visiting_test_combinations | gate_minimum | gate_training_states | gate_rounds_sampled | gate_episodes_sampled | gate_kept | gate_distinct_training_states | gate_trained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## FUNNEL

- What varies: the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table
- Hypothesis: none, since the table describes and tests nothing
- Metric: of the episodes played in each row, the share and the count that had a diagnosis accepted, had the faulty set right, had Top-1, had an order accepted on every faulty component, placed no order on a healthy component and finished; the components that the first accepted diagnosis names as faulty against those faulty at its cycle, as means and as the shares that name more, as many and fewer; the orders on healthy components per healthy component of the start; the share that a failure ended before any order; the share whose cost ratio stands at the cap of the reward; and the refused diagnoses by reason, the episodes finished by the harness, the episodes that the runner's own limit ended, the cut replies, the failed calls, the replays and the cuts at the wall limit, each read from the hidden log, the oracle and scores.csv
- What a result would mean: where each design drops out on the way from a diagnosis to a restored asset, and what the harness did to its episodes

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | status | played | planned | unscored | unjudged | incomplete | refused | replicates | read | diagnosis_accepted | diagnosis_accepted_k | faulty_set_right | faulty_set_right_k | diagnosis_all_correct | diagnosis_all_correct_k | orders_on_every_faulty | orders_on_every_faulty_k | no_order_on_healthy | no_order_on_healthy_k | finished | finished_k | named_faulty_mean | faulty_at_write_mean | named_more | named_as_many | named_fewer | orders_on_healthy_per_healthy_component | failure_before_order | at_cost_cap | refused_diagnoses | finished_by_harness | runner_limit | cut_replies | failed_calls | replays | wall_limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## FAILURES

- What varies: the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table
- Hypothesis: none, since the table describes and tests nothing
- Metric: of the episodes played in each row, the share and the count of each failure class of the verifier in its order and of none, the class of a success and of a failure that no class describes, with an episode without a verdict in the unscored count alone, and beside them the closing claims of recovery and the share of them that the hidden state contradicts, a summary that claims the asset recovered while it ended off its healthy state, with its 95 percent interval from a bootstrap drawn by recorded block
- What a result would mean: where each design fails, by the taxonomy of the verifier, and how often its closing summary claims a recovery that the hidden state does not show

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | status | played | planned | unscored | unjudged | incomplete | refused | replicates | read | premature_diagnosis | premature_diagnosis_k | unmatched_evidence | unmatched_evidence_k | unnecessary_replacement | unnecessary_replacement_k | repair_before_dependency | repair_before_dependency_k | temporary_repair_expired | temporary_repair_expired_k | abstention_without_reading | abstention_without_reading_k | abstention_on_separable | abstention_on_separable_k | abstention_on_e18_separable | abstention_on_e18_separable_k | summary_contradicts_log | summary_contradicts_log_k | repeated_malformed_call | repeated_malformed_call_k | anchoring_on_first_anomaly | anchoring_on_first_anomaly_k | recovery_without_diagnosis | recovery_without_diagnosis_k | none | none_k | recovery_claims | recovery_contradicted | recovery_contradicted_low | recovery_contradicted_high | recovery_contradicted_se | recovery_contradicted_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## SYSTEM_CONTRAST

- What varies: the common designs of the main table, on its preset, each against the CausalDiagnosis agent with the MCP
- Hypothesis: none, since the table describes and tests nothing
- Metric: per common design the paired difference of Top-1, pass and the exact match of the faulty set between the CausalDiagnosis agent with the MCP and the design without it, with its 95 percent interval and standard error, beside its two parts on the same matched episodes, the design part with the MCP on for both and the MCP part on the common design, which add up to it
- What a result would mean: how much of the difference between the CausalDiagnosis agent with the MCP and a common design comes from the design and how much from the MCP

| agent | design | model | surface | config | diagnosis_all_correct_pairs | diagnosis_all_correct_clusters | diagnosis_all_correct_total | diagnosis_all_correct_total_low | diagnosis_all_correct_total_high | diagnosis_all_correct_total_se | diagnosis_all_correct_design | diagnosis_all_correct_design_low | diagnosis_all_correct_design_high | diagnosis_all_correct_design_se | diagnosis_all_correct_mcp | diagnosis_all_correct_mcp_low | diagnosis_all_correct_mcp_high | diagnosis_all_correct_mcp_se | pass_pairs | pass_clusters | pass_total | pass_total_low | pass_total_high | pass_total_se | pass_design | pass_design_low | pass_design_high | pass_design_se | pass_mcp | pass_mcp_low | pass_mcp_high | pass_mcp_se | faulty_set_right_pairs | faulty_set_right_clusters | faulty_set_right_total | faulty_set_right_total_low | faulty_set_right_total_high | faulty_set_right_total_se | faulty_set_right_design | faulty_set_right_design_low | faulty_set_right_design_high | faulty_set_right_design_se | faulty_set_right_mcp | faulty_set_right_mcp_low | faulty_set_right_mcp_high | faulty_set_right_mcp_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## PAIRS_BY_FAULT

- What varies: the comparisons of the agents family on the preset of the main table, each on the starts in which a component is healthy and on the starts that hold a fault of a component and a faulty level of the asset file
- Hypothesis: none, since the table describes and tests nothing
- Metric: the paired difference of Top-1 and pass with its 95 percent interval and standard error, its pairs and its clusters, the difference alone below the registered smallest number of clusters
- What a result would mean: in which fault situations of the asset a comparison of the agents family holds its difference

| hypothesis | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | subset | component | level | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_pairs | diagnosis_all_correct_clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_pairs | pass_clusters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## POLICY_LAYERS

- What varies: the reference policies on each layer after the first, against the layer below
- Hypothesis: none, since the table describes and tests nothing
- Metric: per policy, tool surface and layer the paired difference of pass, success and reward between the layer and the layer below, matched as two layers of the cumulative layer study are matched, with its 95 percent interval, its standard error and its pairs
- What a result would mean: what each single layer of the environment does to the policies that need no model

| agent | surface | config | reference_config | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_pairs | success_diff | success_diff_low | success_diff_high | success_diff_se | success_pairs | reward_diff | reward_diff_low | reward_diff_high | reward_diff_se | reward_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## FIRST_DROP

- What varies: the conditions of the agents tables on the top layer, each split at the median of the first deterioration transition over the starts it played, with the starts without one as a group of their own
- Hypothesis: none, since the table describes and tests nothing
- Metric: per group the episodes, the median that splits them, and Top-1 and pass with their 95 percent intervals
- What a result would mean: whether an agent does worse where the asset deteriorates early

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | group | median_first_drop | n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | pass | pass_low | pass_high | pass_se | pass_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## HEALTHY_CONTROL

- What varies: the row of the main table on the preset of the healthy control, the start with no faulty component
- Hypothesis: none, since the table counts and tests nothing
- Metric: of the registered episodes of each row, how many accepted a diagnosis, how many raised a false alarm with the number of components falsely named, and how many placed a work order with the number of orders, each unnecessary on this start
- What a result would mean: how often an agent finds a fault or orders work where the monitor's flag is false

| agent | agent_server | model | surface | config | n | played | planned | unscored | unjudged | incomplete | refused | replicates | read | diagnosed | false_alarm_episodes | components_falsely_named | work_order_episodes | work_orders |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## RESTORED

- What varies: the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table
- Hypothesis: none, since the table describes and tests nothing
- Metric: the share of episodes restored with no unnecessary order, every component at its healthy level at the end and no order on a healthy component, as the mean with its interval and standard error and k of n, the share of pass beside it, and the share of those episodes whose diagnosis did not match the faulty set exactly
- What a result would mean: how often an agent repairs the asset without the exact diagnosis that pass asks for

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | restored_clean | restored_clean_low | restored_clean_high | restored_clean_se | restored_clean_n | restored_clean_k | pass | pass_low | pass_high | pass_se | pass_n | restored_clean_not_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## H1_TOP_pairs

- What varies: the comparisons of H1 on the preset of training, all six big layers, from cells that exist: the rows without the MCP of the top layer and the rows with it without training of the training table
- Hypothesis: the CausalDiagnosis agent does better than each common design on top-1 and pass where every big layer is on
- Metric: the paired differences of top-1 and pass with their 95 percent intervals, standard errors and exact sign flip p values, Holm run once with the family h2_prime over the registered size of their group agents_environment on the same model and tool surface, a test not run entering at one, and the floor of each reference
- What a result would mean: the family h1_top, primary with h2_prime in the group agents_environment, which asks the question of the agent again on the full environment

| hypothesis | agent | agent_server | reference | reference_server | model | config | surface | status | pairs | clusters | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | pass_holm_p | pass_reject | family_tests | reference_pass_k | reference_pass_n | at_floor | holds | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## FULL_ENV_pairs

- What varies: the environment of each row with the MCP: the full environment, its cell without training in the training table, against the base layer, its cell of the main table, on the same test combinations, entry offsets and seeds
- Hypothesis: none registered as one sided: the difference is expected below zero and tested two-sided
- Metric: the paired difference of pass with its 95 percent interval, standard error and exact two-sided sign flip p value, Holm over the registered size of the family full_environment, the floor of the base cell, and the sign of the difference
- What a result would mean: the layer study without a new run, the family full_environment

| test | agent | agent_server | model | surface | config | reference_config | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | pass_holm_p | pass_reject | family_tests | reference_pass_k | reference_pass_n | at_floor | pass_sign | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## PARALLEL

- What varies: the schedule of candidate plans of the CausalDiagnosis agent with the CausalDiagnosis MCP on the preset of training: on, the agent as released, against off, ablation Ag9, and beside the pair the other rows of the main table with the CausalDiagnosis MCP on the same preset
- Hypothesis: none, since the table describes the rows of the pair and the designs beside it
- Metric: per row over its registered episodes the mean and standard error of the time to repair in declared cycles, the time to repair with the thinking cycles, the cycles charged for thinking, the work orders, the downtime, pass, the cost over the oracle, the model calls and the calls of the ranking tool of the MCP, the batches of reads that the board advised to a plan the model wrote with the number the model kept unchanged and changed and the share kept with its interval, and on the row of another design the share of its gap to the row with the switch on that the switch explains, off less on over the design less on on the measure of the family parallel_time, where the design is behind
- What a result would mean: what the switch changes beside the registered measure, whether the model follows the advised batches, and how much of the distance to the other designs the switch accounts for

| agent | model | surface | config | switch | n | status | repair_declared | repair_declared_low | repair_declared_high | repair_declared_se | repair_declared_n | repair_delay | repair_delay_low | repair_delay_high | repair_delay_se | repair_delay_n | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | work_orders | work_orders_low | work_orders_high | work_orders_se | work_orders_n | downtime_total | downtime_total_low | downtime_total_high | downtime_total_se | downtime_total_n | pass | pass_low | pass_high | pass_se | pass_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | ranking_calls | ranking_calls_low | ranking_calls_high | ranking_calls_se | ranking_calls_n | advised_batches | kept_batches | changed_batches | kept_share | kept_share_low | kept_share_high | kept_share_se | kept_share_n | gap_explained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## PARALLEL_pairs

- What varies: the schedule of candidate plans on against off, on the preset of training, matched by seed
- Hypothesis: the agent with the schedule of candidate plans on repairs sooner: the paired difference of the time to repair in declared cycles lies below zero, its test rejects, and both margins are met
- Metric: the paired difference of the time to repair in declared cycles with its 95 percent interval, its exact p value and its Holm decision over the registered size of the family parallel_time, the paired differences of pass and of the cost over the oracle against their registered margins, and the mean difference over the pairs with and without a refuted repair, which no test reads
- What a result would mean: the family parallel_time

| agent | reference | model | surface | config | status | pairs | clusters | repair_declared_diff | repair_declared_diff_low | repair_declared_diff_high | repair_declared_diff_se | repair_declared_p | repair_declared_holm_p | repair_declared_reject | family_tests | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | margin_pass | margin_pass_met | cost_normalized_diff | cost_normalized_diff_low | cost_normalized_diff_high | cost_normalized_diff_se | margin_cost | margin_cost_met | cost_pairs_without_cost | refuted_pairs | refuted_diff | unrefuted_pairs | unrefuted_diff | reading | holds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## LABELS

- What varies: the label of a start: the reference policy's separability, the label of the greedy reference (CausalDiagnosis/environment/reference_policies/diagnosability_label.py) that the verifier reads, against the E18 grade of the CausalDiagnosis MCP, per label table and ranking model
- Hypothesis: the two labels agree beyond chance
- Metric: the two by two table over the starts that both labels grade, each start counted once, each label the majority of its rollouts, Cohen's kappa, undefined and empty when one label puts every start on one side, Spearman's rank correlation between the share of greedy rollouts that submit the start and the share of E18 rollouts whose first commitment names it, and beside it the same correlation with the share of E18 rollouts that ever commit
- What a result would mean: whether the MCP finds evidence that the reference policy misses, or the reverse, which the starts separable under one label only show

| rules_key | ranking_model | starts | both_separable | separable_label_only | separable_e18_only | both_inseparable | kappa | share_spearman | commit_share_spearman |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

## RUNS

- What varies: the run folder: one row per folder that this script reads outside the online loop of GEPA, whose runs enter the GEPA tables alone, and one row per line of the test set over the runs of its replicates, with the settings of its episodes, the set of episodes it played (starts, master_seed and replicate of run.json) and its start
- Hypothesis: none, since the table indexes the runs that the other tables pool
- Metric: pass, pass with the read gate under the verifier scope all (scope_pass), the judge's pass, the mean of the cost over the cost of the oracle policy on the same seed, the verifier's success and the mean reward, each with an interval from a bootstrap clustered by block, and the median over the episodes with a delay of the larger of the detection and the repair delay
- What a result would mean: every number of the leaderboard that leaderboard.py writes traces to the run folders of one row

| run | agent | model | config | mode | diagnostics_server | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | starts | master_seed | replicate | started | episodes_planned | judge_model | n | judge_verdicts | pass | pass_low | pass_high | pass_se | pass_n | scope_pass | scope_pass_low | scope_pass_high | scope_pass_se | scope_pass_n | judge_pass | judge_pass_low | judge_pass_high | judge_pass_se | judge_pass_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | success | success_low | success_high | success_se | success_n | reward | reward_low | reward_high | reward_se | reward_n | delay_median | delay_n | context_errors | input_tokens_per_call | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

No episode contributes to this table yet.

The registered analysis file of these tables has the SHA-256 56b566684da88ad8c398f315d683cead226d7f63117a3c61b3cfa2b4a7cba7e6.
