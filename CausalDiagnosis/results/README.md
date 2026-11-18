# Results of the hydraulic rig gym

No run has been played on the current code, so the tables below show the design of every ablation with empty cells. The cells stay empty until the test set is played.

`MAIN_TABLES.md` beside this file sets out the result tables, first those with the metrics of the AssetOpsBench paper, then the verified tables that paper lacks, the tables of training and the strata, and `main_tables.tex` holds them in LaTeX.

## A3: the mode of answering the same question, Sc3 on layer 0: statically from the question, the facts of the cycle and the failure modes in one model call (runs of --agent static, mode static in run.json), or executed in the gym with tools and recordings by the six rows on the preset of the agents table

Hypothesis: static answers are graded correct more often than the outcome the verifier reads in the gym warrants

Reading rule: Read the static judge pass rate, where a static answer without a judge record counts as not passed when its run names a judge, beside the share of static answers whose faulty set matches the recorded start, against the recorded pass rate of the gym episodes.

| model | static_n | static_judge_pass | static_judge_pass_low | static_judge_pass_high | static_judge_pass_se | static_faulty_set_correct | static_faulty_set_correct_low | static_faulty_set_correct_high | static_faulty_set_correct_se | gym_n | gym_recorded_pass | gym_recorded_pass_low | gym_recorded_pass_high | gym_recorded_pass_se | gym_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## A7: the verifier scope on the preset of the six rows, layer 0: writes only against all actions (verifier_scope of run.json), the latter from the runs of the six rows scored again by rescore_under_verifier_scope.py

Hypothesis: scoring reads changes the ranking of agents without changing their outcomes

Reading rule: Read the rank correlation of each model: near 1 means scoring reads leaves the ranking of agents as the write only scope has it.

| agent | model | diagnostics_server | n_writes | pass_writes | pass_writes_low | pass_writes_high | pass_writes_se | n_all | pass_all | pass_all_low | pass_all_high | pass_all_se | rank_correlation | agents_in_both_scopes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## A8: the abstention rule of the CausalDiagnosis MCP: the fixed posterior cut (abstention_rule fixed_cut of run.json, with abstention_cut) against the decision rule E10 (abstention_rule decision, the default), per preset and level of the MCP, every other setting at its main level

Hypothesis: the decision rule declares on inseparable episodes and commits on separable ones more often than the constant

Reading rule: Read precision and recall of the declarations against the grade; the better rule declares on inseparable starts and commits on separable ones.

| agent | model | config | diagnostics_server | abstention_rule | label | n | unscored | abstentions | inseparable_starts | precision | precision_low | precision_high | precision_se | recall | recall_low | recall_high | recall_se | surface |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## A9: the price table on all six big layers, En6: the full tier priced far above the rest (full_tier_high) against the default table, for the three agents without the CausalDiagnosis MCP

Hypothesis: conclusions on cost hold under the alternative tables

Reading rule: Read whether the ordering of agents by normalized cost holds under every price table.

| agent | model | diagnostics_server | prices | n | cost | cost_low | cost_high | cost_se | cost_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | pass | pass_low | pass_high | pass_se | pass_n | cost_normalized_diff | cost_normalized_diff_low | cost_normalized_diff_high | cost_normalized_diff_se | cost_normalized_p | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## A10: the guidance level L1 and L3 against L2 on layer 0 with the six rows, En7 (guidance_level of run.json)

Hypothesis: the level an agent needs to pass is a measured quantity and differs between agents

Reading rule: Read pass per guidance level; the lowest level at which an agent's interval reaches the next level's pass is the information its prompt needs.

| agent | model | diagnostics_server | guidance_level | n | pass | pass_low | pass_high | pass_se | pass_n | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## A12: the ranking model: the discriminants with the urn (ranking_model discriminant_urn) against the joint interventional likelihood of E1 and E2 (ranking_model joint), per preset, every other setting at its main level

Hypothesis: the joint model is calibrated where the discriminants are not, and its top candidate does not fall with more readings

Reading rule: Read the Brier score against zero and the top candidate rate by readings; a rate that falls with more readings is the symptom of a misfit of the ranking model.

| agent | model | config | ranking_model | readings | calls | brier | brier_low | brier_high | brier_se | top_candidate_correct | top1_low | top1_high | top1_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## MISDIAGNOSIS_PRICE: the misdiagnosis price of big layer 1 at the budget of the asset, L1, on all six big layers: its arm preset of design.ARM_PRESETS against the preset of training, for the three agents without the CausalDiagnosis MCP

Hypothesis: none, since the table describes the agents when a wrong diagnosis costs the whole budget, and no test decides

Reading rule: Read the shares of wrong diagnoses and of declarations and the cycles read before the diagnosis across the two presets, and the cost without the charge on its one scale.

| agent | model | config | n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | pass | pass_low | pass_high | pass_se | pass_n | diagnosis_wrong | diagnosis_wrong_low | diagnosis_wrong_high | diagnosis_wrong_se | diagnosis_wrong_n | abstained | abstained_low | abstained_high | abstained_se | abstained_n | cycles_read_before_diagnosis | cycles_read_before_diagnosis_low | cycles_read_before_diagnosis_high | cycles_read_before_diagnosis_se | cycles_read_before_diagnosis_n | cost_without_charge | cost_without_charge_low | cost_without_charge_high | cost_without_charge_se | cost_without_charge_n | misdiagnosis_charge | misdiagnosis_charge_low | misdiagnosis_charge_high | misdiagnosis_charge_se | misdiagnosis_charge_n | surface | diagnostics_server |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## SWITCHES: the options of layers 2, 3 and 5 on all six big layers, each arm preset of design.ARM_PRESETS of those layers against the preset of training, for the three agents without the CausalDiagnosis MCP: the thinking time off (L2), the declared action durations off (L3), and the deterioration off, the fixed schedule of level drops against the random hazard rates and the failure rule off (L5)

Hypothesis: each option of a layer changes the pass of the full environment, as the cumulative layer study ranks the layers

Reading rule: Read the pass difference of each arm against the preset of training; an interval that excludes zero shows what the switch adds to the full environment.

| agent | model | diagnostics_server | config | n | pass | pass_low | pass_high | pass_se | pass_n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | detection_delay | detection_delay_low | detection_delay_high | detection_delay_se | detection_delay_n | repair_delay | repair_delay_low | repair_delay_high | repair_delay_se | repair_delay_n | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## GEPA: the online loop of GEPA on the preset of training (design.TRAINING_PRESET by default) under the feedback of the verifier and under the control that sends the score alone, one row per finished run of gepa_prompt_optimization.py

Hypothesis: none, since the table states what each loop did, and the gain of its final block over no block is read in TRAINING_pairs from the evaluation runs, which play the test set once

Reading rule: Read the stream episodes and the episodes until the final candidate under each feedback, and TRAINING_pairs for what the final block brings.

| run | model | config | card | reflection_model | candidates_played | stream_episodes | left_out | rollouts_to_best |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

## GEPA_curve: the round of the online loop of GEPA on the preset of training, under the feedback of the verifier and under the control that sends the score alone

Hypothesis: the prompt in use scores higher on each new block with the feedback of the verifier than with the score alone, from the same first round on

Reading rule: Read the mean reward against the rollouts spent, the feedback of the verifier against the control that sends the score alone from their common first round.

| run | model | config | card | reflection_model | round | rollouts | mean_reward | n |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

## AGENTS: the six rows of design.AGENT_ROWS: Plan-and-Execute, Agents-as-Tools and the CausalDiagnosis agent, each without and with the CausalDiagnosis MCP (the agent label and diagnostics_server of scores.csv in gym mode), each clock mode, tool surface and training variant of a preset a row of its own

Hypothesis: the CausalDiagnosis agent does better than each AssetOpsBench agent and the CausalDiagnosis MCP helps every agent, on top-1 and pass, which AGENTS_pairs decides

Reading rule: Read top-1 and pass of every row against the row it is compared with in AGENTS_pairs, and success, the wrong diagnoses, the declarations and the model calls beside them.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | played | planned | unscored | unjudged | incomplete | refused | replicates | read | pass | pass_low | pass_high | pass_se | pass_n | pass_k1 | pass_k1_low | pass_k1_high | pass_k1_se | pass_k1_n | pass_k2 | pass_k2_low | pass_k2_high | pass_k2_se | pass_k2_n | pass_k3 | pass_k3_low | pass_k3_high | pass_k3_se | pass_k3_n | pass_k4 | pass_k4_low | pass_k4_high | pass_k4_se | pass_k4_n | pass_k5 | pass_k5_low | pass_k5_high | pass_k5_se | pass_k5_n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | faulty_set_right | faulty_set_right_low | faulty_set_right_high | faulty_set_right_se | faulty_set_right_n | restored_clean | restored_clean_low | restored_clean_high | restored_clean_se | restored_clean_n | graded_top1 | graded_top1_low | graded_top1_high | graded_top1_se | graded_top1_n | graded_pass | graded_pass_low | graded_pass_high | graded_pass_se | graded_pass_n | normalized_score | normalized_score_n | success | success_low | success_high | success_se | success_n | reward | reward_low | reward_high | reward_se | reward_n | diagnosis_wrong | diagnosis_wrong_low | diagnosis_wrong_high | diagnosis_wrong_se | diagnosis_wrong_n | abstained | abstained_low | abstained_high | abstained_se | abstained_n | unmatched_evidence | unmatched_evidence_low | unmatched_evidence_high | unmatched_evidence_se | unmatched_evidence_n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | cost | cost_low | cost_high | cost_se | cost_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | downtime_total | downtime_total_low | downtime_total_high | downtime_total_se | downtime_total_n | turns_rmst | turns_rmst_low | turns_rmst_high | turns_rmst_se | turns_rmst_n | detection_delay | detection_delay_low | detection_delay_high | detection_delay_se | detection_delay_n | repair_delay | repair_delay_low | repair_delay_high | repair_delay_se | repair_delay_n | judge_kappa | judge_kappa_low | judge_kappa_high | judge_kappa_se | judge_kappa_n | faulty_set_right_baseline_k | faulty_set_right_baseline_n | faulty_set_right_baseline | faulty_set_right_floor | faulty_set_right_at_floor | replayed | replays | pass_replays_failed | diagnosis_all_correct_replays_failed | faulty_set_right_replays_failed | restored_clean_replays_failed | success_replays_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## AGENTS_pairs: the hypotheses of the six rows on the preset of the agents table, layer 0: H1a to H1d pair the CausalDiagnosis agent against each AssetOpsBench agent, both without the CausalDiagnosis MCP and both with it, H2a to H2c every agent with the MCP against itself without it (design.HYPOTHESES), and the scoring pairs every two of the six rows

Hypothesis: every hypothesis comparison holds, its paired differences of top-1 and pass both above zero with both Holm decisions rejecting, and scoring by the verified pass reverses the judge's order of at least one pair of the six rows

Reading rule: Read a hypothesis as holding when holds is true, its top-1 and pass differences both above zero with both Holm decisions rejecting, and on the scoring rows a reversed pair whose two differences both reject.

| hypothesis | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | pairs | clusters | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | success_diff | success_diff_low | success_diff_high | success_diff_se | success_p | reward_diff | reward_diff_low | reward_diff_high | reward_diff_se | reward_p | judge_pass_diff | judge_pass_diff_low | judge_pass_diff_high | judge_pass_diff_se | judge_pass_p | cost_ratio | cost_ratio_low | cost_ratio_high | cost_ratio_se | cost_ratio_p | turns_rmst_diff | turns_rmst_diff_low | turns_rmst_diff_high | turns_rmst_diff_se | turns_rmst_p | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | pass_holm_p | pass_reject | judge_pass_holm_p | judge_pass_reject | reversed | holds | family_tests | reference_pass_k | reference_pass_n | at_floor | status | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## H1_GUIDANCE: the guidance level of the briefing at level 3 instead of the main level, the En7 runs of the six rows on the preset of the main table (guidance_level of run.json), for the comparisons of H1 in design.HYPOTHESES

Hypothesis: the advantage of the CausalDiagnosis agent over each AssetOpsBench agent does not depend on how much guidance the briefing gives, a robustness check

Reading rule: Read each comparison against its row of AGENTS_pairs: a robustness check that agrees supports H1, and one that does not narrows it, without deciding it.

| hypothesis | agent | agent_server | reference | reference_server | model | config | surface | guidance_level | status | pairs | clusters | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | pass_holm_p | pass_reject | family_tests | holds | reference_pass_k | reference_pass_n | at_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## H2PRIME_pairs: the CausalDiagnosis MCP on the top layer, all six big layers on: each agent of design.TRAINING_ROWS as released against itself without the MCP (design.TRAINING_BASELINES), the comparisons of design.H2_PRIME

Hypothesis: the MCP helps every agent where every big layer of the environment is on, as H2 claims on layer 0

Reading rule: Read an agent as helped when holds is true, its pass and reward differences above zero with both Holm decisions rejecting.

| hypothesis | agent | agent_server | reference | reference_server | model | config | surface | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | reward_diff | reward_diff_low | reward_diff_high | reward_diff_se | reward_p | pass_holm_p | pass_reject | reward_holm_p | reward_reject | family_tests | holds | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## H0: the scoring of an episode by the AssetOpsBench judge's majority under the registered protocol against the verified pass, on the six rows of design.AGENT_ROWS and the comparisons of design.HYPOTHESES on the preset of the main table

Hypothesis: the judge passes a different share of episodes than the verifier and decides the comparisons between rows differently

Reading rule: Read a test as rejecting when h0_reject is true, the sign of the statistic saying whether the judge overstates or understates the verified pass, a test that does not reject as inconclusive and never as agreement, and the kappa and rates beside it as description.

| test | name | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | clusters | nonzero_starts | statistic | statistic_low | statistic_high | statistic_se | h0_p | status | h0_holm_p | h0_reject | family_tests | judge_kappa | judge_kappa_low | judge_kappa_high | judge_kappa_se | judge_kappa_n | discordance | false_pass | false_fail | judge_diff | verified_diff | reversed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## A13_A17: the switches of the CausalDiagnosis agent, each turned alone on the same episodes and read from the label suffixes of CausalLoopOptions.label(): candidate troubleshooting plans off (Ag1, nobranch), plans for the top candidate only (Ag2, single), plan length one (Ag3, len1), and the read the board advises, by the discriminating test of E10 (Ag7, e10) or by one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray, not their algorithm (Ag7, ec2), while the model decides every step in each

Hypothesis: every switch lowers pass or raises the model calls per episode, and no switch improves both over the full agent

Reading rule: Read each switch row against the full row: a lower pass or more model calls is the price of the element the switch removes, and on the Ag7 rows the top-1 accuracy on level starts against the price of the reads.

| agent | switch | model | config | surface | n | pass | pass_low | pass_high | pass_se | pass_n | unnecessary | unnecessary_low | unnecessary_high | unnecessary_se | unnecessary_n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | deviations | deviations_low | deviations_high | deviations_se | deviations_n | test_first | test_first_low | test_first_high | test_first_se | test_first_n | cost | cost_low | cost_high | cost_se | cost_n | test_cost | test_cost_low | test_cost_high | test_cost_se | test_cost_n | level_accuracy | level_accuracy_low | level_accuracy_high | level_accuracy_se | level_accuracy_n | repeated_repair_sets | repeated_repair_sets_low | repeated_repair_sets_high | repeated_repair_sets_se | repeated_repair_sets_n | realized_elimination_per_read | realized_elimination_per_read_low | realized_elimination_per_read_high | realized_elimination_per_read_se | realized_elimination_per_read_n | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | llm_calls_diff | llm_calls_diff_low | llm_calls_diff_high | llm_calls_diff_se | llm_calls_p | status | pairs | clusters | holm_p | reject |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## CALLS: the agent and the clock mode (the agent label, diagnostics_server and clock_mode of scores.csv), the tool surface and the other settings of the condition held fixed in a row

Hypothesis: the CausalDiagnosis agent calls the model less often per episode and is charged fewer cycles for thinking than both rows of the agent that decides every step

Reading rule: Read the model calls and the thinking cycles of every agent against the Agents-as-Tools row with the MCP in the same clock mode.

| agent | model | config | diagnostics_server | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | tokens | tokens_low | tokens_high | tokens_se | tokens_n | llm_calls_diff | llm_calls_diff_low | llm_calls_diff_high | llm_calls_diff_se | llm_calls_p | thinking_cycles_diff | thinking_cycles_diff_low | thinking_cycles_diff_high | thinking_cycles_diff_se | thinking_cycles_p | status | pairs | clusters | holm_p | reject | vs_bare_llm_calls_diff | vs_bare_llm_calls_diff_low | vs_bare_llm_calls_diff_high | vs_bare_llm_calls_diff_se | vs_bare_llm_calls_p | vs_bare_thinking_cycles_diff | vs_bare_thinking_cycles_diff_low | vs_bare_thinking_cycles_diff_high | vs_bare_thinking_cycles_diff_se | vs_bare_thinking_cycles_p | vs_bare_pairs | vs_bare_clusters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## THINKING: the agent, the preset and the training variant among the conditions played in the generation_time mode, the only mode that charges the time between calls

Hypothesis: none, since the table shows what the charged thinking time is made of, the model's generation, a sleep of the machine or a step of the wall clock, and the rest, serving, tools and the harness

Reading rule: Read the generation against the seconds charged: the rest is serving, tools and the harness, and a clock step above zero names a run that lost time to a sleep or a clock step.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | unsplit | aliased | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | seconds_charged | seconds_charged_low | seconds_charged_high | seconds_charged_se | seconds_charged_n | generation_seconds | generation_seconds_low | generation_seconds_high | generation_seconds_se | generation_seconds_n | wall_minus_monotonic_seconds | wall_minus_monotonic_seconds_low | wall_minus_monotonic_seconds_high | wall_minus_monotonic_seconds_se | wall_minus_monotonic_seconds_n | rest_seconds | rest_seconds_low | rest_seconds_high | rest_seconds_se | rest_seconds_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## STRATA: the stratum of the episode: an episode with a deterioration transition, scheduled or drawn by the hazard (the hidden log), a start separable and a start inseparable under the reference policy's separability (the separable cell), the same under the E18 grade of the CausalDiagnosis MCP (the e18 record of the grade cell), a start that both mark inseparable, where a declaration earns its credit, and an episode whose separating test exceeded the budget left at some rank_root_causes call that said read first (the trajectory)

Hypothesis: the advantage of the CausalDiagnosis agent concentrates on the episodes the design targets

Reading rule: Read pass per stratum against the whole grid of AGENTS, the declarations on the separable strata against zero, and success and the declarations beside pass on the starts both labels mark inseparable, since a correct declaration counts as success there.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | stratum | n | pass | pass_low | pass_high | pass_se | pass_n | success | success_low | success_high | success_se | success_n | abstained | abstained_low | abstained_high | abstained_se | abstained_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## JUDGE: the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the protocol and the judge model of the judge records of its episodes, with the metrics of Figure 5 of AssetOpsBench on the trajectories that the verifier scores

Hypothesis: none, since the paper reports the three scores beside the verified outcome and the scoring pairs read the judge's pass in AGENTS_pairs

Reading rule: Read the mean of the three scores under the protocol paper with the metrics of Figure 5 of AssetOpsBench, and the majority columns against the values that paper prints.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | judge_protocol | judge_model | n | failed_calls | episodes_without_record | declared | task_completion | task_completion_low | task_completion_high | task_completion_se | task_completion_n | task_completion_std | task_completion_majority | task_completion_majority_low | task_completion_majority_high | task_completion_majority_se | task_completion_majority_n | data_retrieval_accuracy | data_retrieval_accuracy_low | data_retrieval_accuracy_high | data_retrieval_accuracy_se | data_retrieval_accuracy_n | data_retrieval_accuracy_std | data_retrieval_accuracy_majority | data_retrieval_accuracy_majority_low | data_retrieval_accuracy_majority_high | data_retrieval_accuracy_majority_se | data_retrieval_accuracy_majority_n | generalized_result_verification | generalized_result_verification_low | generalized_result_verification_high | generalized_result_verification_se | generalized_result_verification_n | generalized_result_verification_std | generalized_result_verification_majority | generalized_result_verification_majority_low | generalized_result_verification_majority_high | generalized_result_verification_majority_se | generalized_result_verification_majority_n | kappa_pass | kappa_pass_low | kappa_pass_high | kappa_pass_se | pass_both | pass_judge_only | pass_verifier_only | pass_neither | kappa_diagnosis_all_correct | kappa_diagnosis_all_correct_low | kappa_diagnosis_all_correct_high | kappa_diagnosis_all_correct_se | diagnosis_all_correct_both | diagnosis_all_correct_judge_only | diagnosis_all_correct_verifier_only | diagnosis_all_correct_neither | kappa_faulty_set_right | kappa_faulty_set_right_low | kappa_faulty_set_right_high | kappa_faulty_set_right_se | faulty_set_right_both | faulty_set_right_judge_only | faulty_set_right_verifier_only | faulty_set_right_neither | kappa_restored_clean | kappa_restored_clean_low | kappa_restored_clean_high | kappa_restored_clean_se | restored_clean_both | restored_clean_judge_only | restored_clean_verifier_only | restored_clean_neither | judge_record_cut | judge_record_cut_low | judge_record_cut_high | judge_record_cut_se | judge_record_cut_n | judge_passed_unrecovered | judge_passed_unrecovered_low | judge_passed_unrecovered_high | judge_passed_unrecovered_se | judge_passed_unrecovered_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## JUDGE_AGREEMENT: the runs of the six rows together per model, config and settings at the main level of every setting, under a judge protocol with more than one call, with the metrics of Table 14 of AssetOpsBench

Hypothesis: none, since the table checks how stable the judge is under repeated calls

Reading rule: Read the agreement against 1 and the uncertainty against zero, as Table 14 of AssetOpsBench reads them.

| model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | judge_protocol | n | task_completion_agreement | task_completion_agreement_low | task_completion_agreement_high | task_completion_agreement_se | task_completion_agreement_n | task_completion_uncertainty | data_retrieval_accuracy_agreement | data_retrieval_accuracy_agreement_low | data_retrieval_accuracy_agreement_high | data_retrieval_accuracy_agreement_se | data_retrieval_accuracy_agreement_n | data_retrieval_accuracy_uncertainty | generalized_result_verification_agreement | generalized_result_verification_agreement_low | generalized_result_verification_agreement_high | generalized_result_verification_agreement_se | generalized_result_verification_agreement_n | generalized_result_verification_uncertainty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## EFFICIENCY: the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the number of faulty components at the start, one or two, which take the place of the single-agent and multi-agent tasks of Table 3 of AssetOpsBench

Hypothesis: none, since the paper reports the steps and the runtime as what a row costs to run

Reading rule: Read the steps and the runtime of a row on single-fault against two-fault starts, as Table 3 of AssetOpsBench reads single-agent against multi-agent tasks.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | faults | n | tool_invocations | tool_invocations_low | tool_invocations_high | tool_invocations_se | tool_invocations_n | tool_invocations_std | wall_s | wall_s_low | wall_s_high | wall_s_se | wall_s_n | wall_s_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## PLANS: the row of AGENTS that makes a plan, Plan-and-Execute and the CausalDiagnosis agent, each clock mode, tool surface and training variant of a preset a row of its own, with the metrics of Table 12 of AssetOpsBench

Hypothesis: none, since the paper reports the length of the plans as a description of the rows

Reading rule: Read the planned steps of Plan-and-Execute and of the CausalDiagnosis agent, each with and without the MCP, as Table 12 of AssetOpsBench reads its models.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | planned_steps | planned_steps_low | planned_steps_high | planned_steps_se | planned_steps_n | planned_steps_std | planned_steps_min | planned_steps_max | planned_steps_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## ELIM: the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the reference policies of the action server, every row measured by one replay of the filter of the CausalDiagnosis MCP over the released evidence (realized_read_elimination.py), whether or not its run had the MCP

Hypothesis: none, since the paper reports the realized elimination as a score beside the hypotheses, and its gap to the expected elimination as a check of the MCP's forecast

Reading rule: Read the realized elimination per unit of cost of each agent against the reference policies, and the calibration gap of the runs with the MCP against zero.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | paid_reads | calibrated_reads | yardstick | realized_elimination_per_read | realized_elimination_per_read_low | realized_elimination_per_read_high | realized_elimination_per_read_se | realized_elimination_per_read_n | realized_elimination_per_cost | realized_elimination_per_cost_low | realized_elimination_per_cost_high | realized_elimination_per_cost_se | realized_elimination_per_cost_n | expected_elimination | expected_elimination_low | expected_elimination_high | expected_elimination_se | expected_elimination_n | realized_elimination_calibrated | realized_elimination_calibrated_low | realized_elimination_calibrated_high | realized_elimination_calibrated_se | realized_elimination_calibrated_n | calibration_gap | calibration_gap_low | calibration_gap_high | calibration_gap_se | calibration_gap_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## ELIM_pairs: the comparisons of the hypotheses of AGENTS_pairs, on the realized elimination of the paid reads

Hypothesis: none, since no hypothesis of the paper reads the realized elimination, so no Holm decision covers these differences

Reading rule: Read the difference of the realized elimination per unit of cost against zero on every row.

| hypothesis | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | status | pairs | clusters | realized_elimination_per_cost_diff | realized_elimination_per_cost_diff_low | realized_elimination_per_cost_diff_high | realized_elimination_per_cost_diff_se | realized_elimination_per_cost_p | realized_elimination_per_read_diff | realized_elimination_per_read_diff_low | realized_elimination_per_read_diff_high | realized_elimination_per_read_diff_se | realized_elimination_per_read_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## MCP: the agents with the CausalDiagnosis MCP, each on every preset, clock mode and tool surface, each condition a row of its own

Hypothesis: none, since the measures of the MCP are descriptive

Reading rule: Read the harm beside the uptake, the two prediction skills against zero, and the production lost while thinking against the same agent without the MCP.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | unaligned | unplaced | yardstick | uptake | uptake_low | uptake_high | uptake_se | uptake_n | uptake_any | uptake_any_low | uptake_any_high | uptake_any_se | uptake_any_n | harm | harm_low | harm_high | harm_se | harm_n | good_deviation | good_deviation_low | good_deviation_high | good_deviation_se | good_deviation_n | pass_follow | pass_follow_low | pass_follow_high | pass_follow_se | pass_follow_n | pass_deviate | pass_deviate_low | pass_deviate_high | pass_deviate_se | pass_deviate_n | read_uptake | read_uptake_low | read_uptake_high | read_uptake_se | read_uptake_n | diagnosis_uptake | diagnosis_uptake_low | diagnosis_uptake_high | diagnosis_uptake_se | diagnosis_uptake_n | recovery_bss | recovery_bss_low | recovery_bss_high | recovery_bss_se | recovery_bss_n | recovery_bias | recovery_bias_low | recovery_bias_high | recovery_bias_se | recovery_bias_n | cost_bias | cost_bias_low | cost_bias_high | cost_bias_se | cost_bias_n | cost_mae | cost_mae_low | cost_mae_high | cost_mae_se | cost_mae_n | price_mae | price_mae_low | price_mae_high | price_mae_se | price_mae_n | cost_skill | cost_skill_low | cost_skill_high | cost_skill_se | cost_skill_n | coverage | coverage_low | coverage_high | coverage_se | coverage_n | readings_skill | readings_skill_low | readings_skill_high | readings_skill_se | readings_skill_n | false_alarm | false_alarm_low | false_alarm_high | false_alarm_se | false_alarm_n | miss | miss_low | miss_high | miss_se | miss_n | ac1 | ac1_low | ac1_high | ac1_se | ac1_n | ack | ack_low | ack_high | ack_se | ack_n | mrr | mrr_low | mrr_high | mrr_se | mrr_n | precision | precision_low | precision_high | precision_se | precision_n | recall | recall_low | recall_high | recall_se | recall_n | think_loss_share | think_loss_share_low | think_loss_share_high | think_loss_share_se | think_loss_share_n | think_failures | think_failures_low | think_failures_high | think_failures_se | think_failures_n | think_loss | think_loss_low | think_loss_high | think_loss_se | think_loss_n | think_loss_diff | think_loss_diff_low | think_loss_diff_high | think_loss_diff_se | pairs | clusters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## POLICIES: the reference policy of design.REFERENCE_POLICIES (oracle, greedy reference, random, do nothing and always declare) on every preset, the rows of the environment of design.LADDER among them, each clock mode and tool surface a row of its own

Hypothesis: the oracle is the best of the five reference policies, with a reward at least that of every other reference policy, and doing nothing scores below every policy that the verifier passes, on every start of every layer

Reading rule: Read the reward of the oracle as that of the best of the five reference policies on each row, that of doing nothing below every policy that passes, and the rows of AGENTS beside them.

| agent | config | clock_mode | surface | n | pass | pass_low | pass_high | pass_se | pass_n | success | success_low | success_high | success_se | success_n | reward | reward_low | reward_high | reward_se | reward_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## LAYERS_pairs: the big layer of the environment that a row of design.LADDER adds to the row below it, played by the three rows with the CausalDiagnosis MCP (design.LADDER_ROWS)

Hypothesis: every big layer changes the pass, pooled over the three agents

Reading rule: Read the pooled difference of each layer against zero, decided by its Holm test, and the three rows beside it for agents that react differently.

| model | surface | config | reference_config | agent | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | holm_p | reject | pass_holm_p | pass_reject | family_tests | reference_pass_k | reference_pass_n | at_floor | pass_sign | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## TRAINING: the training variant of design.TRAINING_VARIANTS (as released, GEPA, GEPA on the score alone, fine tuning, fine tuning then GEPA, and fine-tuning on unfiltered trajectories) of the three rows with the CausalDiagnosis MCP (design.TRAINING_ROWS) on the preset of training, all six big layers (design.TRAINING_PRESET), from the training variant of run.json, and the three agents without the MCP and without training on that preset (design.TRAINING_BASELINES) as the baselines, the top layer on the same seeds

Hypothesis: training in the environment raises pass and top-1 on fault combinations that the trained model never saw

Reading rule: Read each trained variant against the row as released over every episode and on the single fault and two fault starts.

| model | config | surface | agent | diagnostics_server | training_variant | subset | n | pass | pass_low | pass_high | pass_se | pass_n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | judged | aliased | episodes_without_record | failed_calls | declared | task_completion | task_completion_low | task_completion_high | task_completion_se | task_completion_n | data_retrieval_accuracy | data_retrieval_accuracy_low | data_retrieval_accuracy_high | data_retrieval_accuracy_se | data_retrieval_accuracy_n | generalized_result_verification | generalized_result_verification_low | generalized_result_verification_high | generalized_result_verification_se | generalized_result_verification_n | unmatched_evidence | unmatched_evidence_low | unmatched_evidence_high | unmatched_evidence_se | unmatched_evidence_n | judge_kappa | judge_kappa_low | judge_kappa_high | judge_kappa_se | judge_kappa_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## TRAINING_pairs: the training comparisons of design.TRAINING_COMPARISONS per row with the CausalDiagnosis MCP: GEPA, fine tuning and fine tuning then GEPA against the agent as released, GEPA against GEPA on the score alone, and fine tuning against its control, fine-tuning on unfiltered trajectories, on as many episodes

Hypothesis: each trained variant passes and diagnoses more often than the agent as released, GEPA more often than GEPA on the score alone, and fine tuning more often than its control, fine-tuning on unfiltered trajectories, on fault combinations the trained model never saw, so a gain comes from the environment's signal and not from more exposure to the task

Reading rule: Read a difference over every episode above zero whose Holm decision rejects as a gain of the variant over its reference, and the two strata beside it.

| model | config | surface | agent | diagnostics_server | training_variant | reference_variant | comparison | subset | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | pass_holm_p | pass_reject | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | family_tests | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## TRAINING_RECORD: the row with the CausalDiagnosis MCP on the preset of training, from run.json of its trained evaluation runs, where the training pipeline records the adapter and the task block a run plays

Hypothesis: none, since the table states what each agent trained on, the same budget for the three

Reading rule: Read the counts of the three agents side by side.

| agent | model | fold | episodes_sampled | kept_by_filter | model_calls_as_examples | examples_over_max_length | optimizer_steps | control_trajectories | control_model_calls_as_examples | control_examples_over_max_length | control_optimizer_steps | gepa_candidates | gepa_stream_episodes | ft_gepa_candidates | ft_gepa_stream_episodes | gepa_score_only_candidates | gepa_score_only_stream_episodes | episodes_visiting_test_combinations | control_episodes_visiting_test_combinations | gepa_episodes_visiting_test_combinations | gepa_score_only_episodes_visiting_test_combinations | ft_gepa_episodes_visiting_test_combinations | gate_minimum | gate_training_states | gate_rounds_sampled | gate_episodes_sampled | gate_kept | gate_distinct_training_states | gate_trained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## FUNNEL: the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read each step against the one before it; a row with a missing episode prints incomplete, and its shares are over the episodes played.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | status | played | planned | unscored | unjudged | incomplete | refused | replicates | read | diagnosis_accepted | diagnosis_accepted_k | faulty_set_right | faulty_set_right_k | diagnosis_all_correct | diagnosis_all_correct_k | orders_on_every_faulty | orders_on_every_faulty_k | no_order_on_healthy | no_order_on_healthy_k | finished | finished_k | named_faulty_mean | faulty_at_write_mean | named_more | named_as_many | named_fewer | orders_on_healthy_per_healthy_component | failure_before_order | at_cost_cap | refused_diagnoses | finished_by_harness | runner_limit | cut_replies | failed_calls | replays | wall_limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## FAILURES: the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read the classes of a row as one split of its scored episodes, and the share of contradicted claims as a share of claims, not of episodes.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | status | played | planned | unscored | unjudged | incomplete | refused | replicates | read | premature_diagnosis | premature_diagnosis_k | unmatched_evidence | unmatched_evidence_k | unnecessary_replacement | unnecessary_replacement_k | repair_before_dependency | repair_before_dependency_k | temporary_repair_expired | temporary_repair_expired_k | abstention_without_reading | abstention_without_reading_k | abstention_on_separable | abstention_on_separable_k | abstention_on_e18_separable | abstention_on_e18_separable_k | summary_contradicts_log | summary_contradicts_log_k | repeated_malformed_call | repeated_malformed_call_k | anchoring_on_first_anomaly | anchoring_on_first_anomaly_k | recovery_without_diagnosis | recovery_without_diagnosis_k | none | none_k | recovery_claims | recovery_contradicted | recovery_contradicted_low | recovery_contradicted_high | recovery_contradicted_se | recovery_contradicted_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## SYSTEM_CONTRAST: the common designs of the main table, on its preset, each against the CausalDiagnosis agent with the MCP

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read the total against the sum of its two parts; the table is no test and prints no p value.

| agent | design | model | surface | config | diagnosis_all_correct_pairs | diagnosis_all_correct_clusters | diagnosis_all_correct_total | diagnosis_all_correct_total_low | diagnosis_all_correct_total_high | diagnosis_all_correct_total_se | diagnosis_all_correct_design | diagnosis_all_correct_design_low | diagnosis_all_correct_design_high | diagnosis_all_correct_design_se | diagnosis_all_correct_mcp | diagnosis_all_correct_mcp_low | diagnosis_all_correct_mcp_high | diagnosis_all_correct_mcp_se | pass_pairs | pass_clusters | pass_total | pass_total_low | pass_total_high | pass_total_se | pass_design | pass_design_low | pass_design_high | pass_design_se | pass_mcp | pass_mcp_low | pass_mcp_high | pass_mcp_se | faulty_set_right_pairs | faulty_set_right_clusters | faulty_set_right_total | faulty_set_right_total_low | faulty_set_right_total_high | faulty_set_right_total_se | faulty_set_right_design | faulty_set_right_design_low | faulty_set_right_design_high | faulty_set_right_design_se | faulty_set_right_mcp | faulty_set_right_mcp_low | faulty_set_right_mcp_high | faulty_set_right_mcp_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## PAIRS_BY_FAULT: the comparisons of the agents family on the preset of the main table, each on the starts in which a component is healthy and on the starts that hold a fault of a component and a faulty level of the asset file

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read a row as the mean paired difference over the starts of its subset, which the test starts sample as exchangeable fault situations of the asset.

| hypothesis | agent | agent_server | reference | reference_server | model | config | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | subset | component | level | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_pairs | diagnosis_all_correct_clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_pairs | pass_clusters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## POLICY_LAYERS: the reference policies on each layer after the first, against the layer below

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read the effect of a layer on the policies beside the layer effects of the agents; the table is no test.

| agent | surface | config | reference_config | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_pairs | success_diff | success_diff_low | success_diff_high | success_diff_se | success_pairs | reward_diff | reward_diff_low | reward_diff_high | reward_diff_se | reward_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## FIRST_DROP: the conditions of the agents tables on the top layer, each split at the median of the first deterioration transition over the starts it played, with the starts without one as a group of their own

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read the early group against the late group of each row; the starts without a drop are a group of their own.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | group | median_first_drop | n | diagnosis_all_correct | diagnosis_all_correct_low | diagnosis_all_correct_high | diagnosis_all_correct_se | diagnosis_all_correct_n | pass | pass_low | pass_high | pass_se | pass_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## HEALTHY_CONTROL: the row of the main table on the preset of the healthy control, the start with no faulty component

Hypothesis: none, since the table counts and tests nothing

Reading rule: Read the counts of each row against its registered episodes.

| agent | agent_server | model | surface | config | n | played | planned | unscored | unjudged | incomplete | refused | replicates | read | diagnosed | false_alarm_episodes | components_falsely_named | work_order_episodes | work_orders |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## RESTORED: the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table

Hypothesis: none, since the table describes and tests nothing

Reading rule: Read the share restored with no unnecessary order beside pass; the last column is the part of it that pass does not credit.

| agent | model | config | clock_mode | surface | diagnostics_server | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | abstention_cut | prices_sha256 | fraction | n | restored_clean | restored_clean_low | restored_clean_high | restored_clean_se | restored_clean_n | restored_clean_k | pass | pass_low | pass_high | pass_se | pass_n | restored_clean_not_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## H1_TOP_pairs: the comparisons of H1 on the preset of training, all six big layers, from cells that exist: the rows without the MCP from the reference lines without it on all six big layers and the rows with it from the part environment of the agents grid, both on a server without LoRA adapters

Hypothesis: the CausalDiagnosis agent does better than each common design on top-1 and pass where every big layer is on

Reading rule: Read the reading column: holds, reversed, does not hold, not run or uninformative.

| hypothesis | agent | agent_server | reference | reference_server | model | config | surface | status | pairs | clusters | diagnosis_all_correct_diff | diagnosis_all_correct_diff_low | diagnosis_all_correct_diff_high | diagnosis_all_correct_diff_se | diagnosis_all_correct_p | diagnosis_all_correct_holm_p | diagnosis_all_correct_reject | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | pass_holm_p | pass_reject | family_tests | reference_pass_k | reference_pass_n | at_floor | holds | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## FULL_ENV_pairs: the environment of each row with the MCP: the full environment, its cell without training in the training table, against the base layer, its cell of the main table, on the same test combinations, entry offsets and seeds

Hypothesis: none registered as one sided: the difference is expected below zero and tested two-sided

Reading rule: Read the reading column and the sign of the difference.

| test | agent | agent_server | model | surface | config | reference_config | status | pairs | clusters | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | pass_p | pass_holm_p | pass_reject | family_tests | reference_pass_k | reference_pass_n | at_floor | pass_sign | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## PARALLEL: the schedule of candidate plans of the CausalDiagnosis agent with the CausalDiagnosis MCP on the preset of training: on, the agent as released, against off, ablation Ag9, and beside the pair the other rows of the main table with the CausalDiagnosis MCP on the same preset

Hypothesis: none, since the table describes the rows of the pair and the designs beside it

Reading rule: Read the two rows of the pair side by side and each other design against the row with the switch on, and find the test in PARALLEL_pairs.

| agent | model | surface | config | switch | n | status | repair_declared | repair_declared_low | repair_declared_high | repair_declared_se | repair_declared_n | repair_delay | repair_delay_low | repair_delay_high | repair_delay_se | repair_delay_n | thinking_cycles | thinking_cycles_low | thinking_cycles_high | thinking_cycles_se | thinking_cycles_n | work_orders | work_orders_low | work_orders_high | work_orders_se | work_orders_n | downtime_total | downtime_total_low | downtime_total_high | downtime_total_se | downtime_total_n | pass | pass_low | pass_high | pass_se | pass_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | llm_calls | llm_calls_low | llm_calls_high | llm_calls_se | llm_calls_n | ranking_calls | ranking_calls_low | ranking_calls_high | ranking_calls_se | ranking_calls_n | advised_batches | kept_batches | changed_batches | kept_share | kept_share_low | kept_share_high | kept_share_se | kept_share_n | gap_explained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## PARALLEL_pairs: the schedule of candidate plans on against off, on the preset of training, matched by seed

Hypothesis: the agent with the schedule of candidate plans on repairs sooner: the paired difference of the time to repair in declared cycles lies below zero, its test rejects, and both margins are met

Reading rule: Read the reading and the two margins: the comparison holds when the reading is holds and both margins are met.

| agent | reference | model | surface | config | status | pairs | clusters | repair_declared_diff | repair_declared_diff_low | repair_declared_diff_high | repair_declared_diff_se | repair_declared_p | repair_declared_holm_p | repair_declared_reject | family_tests | pass_diff | pass_diff_low | pass_diff_high | pass_diff_se | margin_pass | margin_pass_met | cost_normalized_diff | cost_normalized_diff_low | cost_normalized_diff_high | cost_normalized_diff_se | margin_cost | margin_cost_met | cost_pairs_without_cost | refuted_pairs | refuted_diff | unrefuted_pairs | unrefuted_diff | reading | holds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## LABELS: the label of a start: the reference policy's separability, the label of the greedy reference (CausalDiagnosis/environment/reference_policies/diagnosability_label.py) that the verifier reads, against the E18 grade of the CausalDiagnosis MCP, per label table and ranking model

Hypothesis: the two labels agree beyond chance

Reading rule: Read kappa against 1 for agreement between the labels, and the two one label cells for the starts on which they disagree. When kappa is empty, the rank correlation of the shares carries the comparison.

| rules_key | ranking_model | starts | both_separable | separable_label_only | separable_e18_only | both_inseparable | kappa | share_spearman | commit_share_spearman |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

## RUNS: the run folder: one row per folder that this script reads outside the online loop of GEPA, whose runs enter the GEPA tables alone, and one row per line of the test set over the runs of its replicates, with the settings of its episodes, the set of episodes it played (starts, master_seed and replicate of run.json) and its start

Hypothesis: none, since the table indexes the runs that the other tables pool

Reading rule: Read one row per run folder, or per line of the test set over its replicate runs, which the leaderboard places in its tables by the agent label and the settings.

| run | agent | model | config | mode | diagnostics_server | clock_mode | surface | guidance_level | verifier_scope | prices | ranking_model | abstention_rule | training_variant | lora_adapters | starts | master_seed | replicate | started | episodes_planned | judge_model | n | judge_verdicts | pass | pass_low | pass_high | pass_se | pass_n | scope_pass | scope_pass_low | scope_pass_high | scope_pass_se | scope_pass_n | judge_pass | judge_pass_low | judge_pass_high | judge_pass_se | judge_pass_n | cost_normalized | cost_normalized_low | cost_normalized_high | cost_normalized_se | cost_normalized_n | success | success_low | success_high | success_se | success_n | reward | reward_low | reward_high | reward_se | reward_n | delay_median | delay_n | context_errors | input_tokens_per_call | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Example episode

No episode has been played yet. After the runs this section names one episode and shows its tool calls and its verdict.

## How to reproduce

On a laptop, a wiring check without a model (the fork's virtual environment first on PATH, the data file built once by `python -m CausalDiagnosis.datasets.hyd_1.download_and_build --download`):

```
python -m CausalDiagnosis.experiments.episode_runners.run_episodes --config static --agent scripted --seeds 1
python -m CausalDiagnosis.experiments.episode_runners.run_episodes --config static --agent oracle --seeds 3
```

On the cluster the runs follow the run order of `CausalDiagnosis/cluster_jobs/README.md`, from the two gates to the grid files and the training of the training table with `gepa_prompt_optimization.py`, the online loop of GEPA, then on a laptop `bash CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh <remote runs folder>`, which copies the run folders to the laptop, where they stay. The tables, this file and the leaderboard are rebuilt from those copies with:

```
python -m CausalDiagnosis.evaluation.result_tables.tables_from_runs --runs <synced run folders> --out CausalDiagnosis/results/tables
python -m CausalDiagnosis.evaluation.result_tables.leaderboard --since <agent>=<first valid start> --runs <synced run folders>
```
