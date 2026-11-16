# Metrics of the hydraulic rig gym

The cards below define every metric that the result tables of the gym print, the scores of the verifier and of the tables first and then the measures of the CausalDiagnosis MCP. Every card has the same fields in the same order. A card opens with its class among benefit, expense, downside and outcome, a definition in words, a formula for any asset, its inputs and the code that computes it. It then states the relations that the baseline policies, which call no model, and the configurations with the CausalDiagnosis MCP off and on must satisfy, how an agent could game it, whether it decides a pre-registered hypothesis, and the tables that print it. Every definition applies to any asset with an asset file, and the hydraulic rig appears only as the worked instance. `CausalDiagnosis/evaluation/metrics/tests/test_metrics_doc.py` pins every count written in digits on this page to the code, with the counts of big layers and of configurations written in words. The same test checks that every name that a card gives under Computed in appears in a module that the same field lists. It also checks that every table and every score column of the tree has its place on the page. No value on this page is the outcome of an experiment.

The related pages are [environment.md](environment.md) for the episode rules, the oracle and the verifier, [experiments.md](experiments.md) for the evaluation scripts, the tables and the statistics, [causal-diagnosis-mcp.md](causal-diagnosis-mcp.md) for the formulas of the CausalDiagnosis MCP, and [dataset.md](dataset.md) for the data behind every episode.

---

## Symbols and conventions

| Symbol or convention | Meaning |
| --- | --- |
| Asset | The asset file gives the components `j` in `J`, the level set `L_j` and the healthy level `h_0j` of each, the combinations `S = prod_j L_j`, the healthy combination `h_0` and the faulty set `F(h) = {j : h_j != h_0j}` of a combination `h`, which `faulty` of `LevelsTruth` in `CausalDiagnosis/datasets/dataset_loader/ground_truth.py` computes. |
| Episode | An episode `e` hides the combination `h(t)` at cycle `t`, from its initial combination `s_e` to its last cycle `T_e`, under the horizon `T`, the budget `B` and the step limit `K`. The scheduled and the random level drops, the expiry of emergency repairs and the closing of work orders change the combination. The released evidence is the set of readings that the agent has requested and received. |
| Writes | A write is a state-changing tool call, the diagnosis, the declaration, a work order or the finish. `h_hat_e` is the submitted diagnosis, `t_w` the cycle of its write and `h(t_w)` the truth at the write. `u_e` counts the work orders placed on a component that was healthy when the order was placed, and `h(T_e)` is the final combination. |
| Cost | `cost_e` is the cost of an episode in the unit of its configuration, which `cost_unit` sets from the price table. In money it adds the prices that the agent incurred, visit charges included, the production loss that the hidden record of costs accrued, the failure penalties and the misdiagnosis charge of a wrong diagnosis. On a configuration whose reading, part, repair and visit prices are all zero it counts the tool calls of the agent, refused calls included. `cost*_e` is the cost of the oracle policy on the same seed. `C_d` is the price of a wrong diagnosis, `prices.misdiagnosis` of the released rules, which the environment adds to the hidden record of costs at a wrong diagnosis. |
| Configurations and intervals | A configuration is one experimental condition, the agent, the CausalDiagnosis MCP off or on, the preset and every setting, over its episodes `E`, and a row of a table prints one configuration. Every rate is the mean over `E` with a 95% interval from a bootstrap of 10,000 resamples that draws whole test combinations, `cluster_bootstrap` in `CausalDiagnosis/evaluation/metrics/table_statistics.py`, since the episodes of one test combination share its recorded block. |
| Paired differences | A paired difference matches two configurations episode by episode on the seed and the scenario, which fix the initial combination and the entry offset, and on every factor but the one compared, as `paired` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/cell_statistics.py` does. |
| Baseline policies | The 5 policies of `REFERENCE_POLICIES` in `CausalDiagnosis/experiments/table_definitions/experiment_tables.py` call no model. The oracle knows the hidden state without being charged for a reading, submits the current combination and orders every faulty component once it will not drop further and its repair will take, and its cost normalizes the cost of every episode. The greedy baseline policy reads by expected information gain per unit price through cross-fitted classifiers and shares no model with the CausalDiagnosis MCP. The random policy reads, diagnoses and orders at random. Doing nothing waits until the horizon and is the weakest baseline policy. Always declare reads the cheapest sensor group once, cites that reading and declares, the most a constant declaration earns. Among the relations that `test_run_reference_policies.py` pins, the oracle earns at least the reward of every other baseline policy on every seed of every preset of `LADDER` and of both deterioration modes, and doing nothing earns less than every policy that the verifier passes. |
| CausalDiagnosis MCP off and on | The agents table runs the 6 configurations of `AGENT_ROWS`, every agent with the CausalDiagnosis MCP off and on, on layer 0, the preset `AGENTS_PRESET`, and the same configurations on the preset `TRAINING_PRESET` with all six big layers on, the lines that the two tables of the agents name the CausalDiagnosis environment. The training table opens each agent with its baseline without the CausalDiagnosis MCP and its configuration with the CausalDiagnosis MCP, both untrained, on all six big layers and on the same test episodes. |
| Worked instance | On the hydraulic rig, `J` contains the 4 components cooler, valve, pump and accumulator, with 3, 4, 3 and 4 levels, so `S` holds 144 combinations, and the test set consists of 5 replicates of each of the 47 test combinations with 1 or 2 faulty components, 235 episodes per configuration. Layer 0 prices nothing, so its cost counts the tool calls of the agent. |

---

## Pre-registered analysis

In every pre-registered comparison, the model, the tool servers and the environment budget are equal between its two configurations, and the setup lists the own settings of each agent design once. The p value of a pre-registered comparison comes from a permutation test that flips the signs of the paired differences of whole test combinations, `sign_flip_p` in `CausalDiagnosis/evaluation/metrics/table_statistics.py`, which is exact by convolution for a binary outcome. Holm's step-down procedure corrects the tests of one family of the correction, once per model and tool surface. It reads the p values as the tables print them, rounded to six digits. The registered analysis, `CausalDiagnosis/experiments/table_definitions/registered_analysis.json`, names three primary groups, and each group is one family of the correction, whose size is the sum of the registered sizes of its families. The group `judge` holds H0, the group `agents` holds H1 and H2 on the base layer, and the group `agents_environment` holds `h1_top` and H2' on the preset with every big layer. A family that no group names is secondary and is corrected alone. `correction_group` and `corrected_family` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/hypothesis_tables.py` run the correction by these groups.

| Family | What it tests | Size and rule | Code and tables |
| --- | --- | --- | --- |
| H1 and H2 | The CausalDiagnosis agent against each AssetOpsBench agent, and each agent with the CausalDiagnosis MCP against the same agent without it, on the base layer, on top-1 and on pass | A comparison is confirmed when both paired differences lie above zero and both Holm decisions reject over the 14 tests of `tab:hyp`, and it reads as reversed when both lie below zero and both decisions reject. | `HYPOTHESES` and `HYPOTHESIS_MEASURES` in `CausalDiagnosis/experiments/table_definitions/experiment_tables.py`, AGENTS_pairs |
| H1 at floor | Whether a comparison of H1 is informative | A baseline whose verified pass rate on the test runs lies below 5% is at floor. H1 against it is reported as uninformative, and its tests remain in the family. | `descriptive.floor_of_pass` in `CausalDiagnosis/experiments/table_definitions/registered_analysis.json`, `floor_cells` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/family_correction.py` |
| H1 under another guidance level | The comparisons of H1 at the guidance level `ROBUSTNESS_GUIDANCE_LEVEL` of the briefing | A robustness check with a Holm family of its own, which decides nothing of H1. | `table_h1_guidance` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/hypothesis_tables.py`, H1_GUIDANCE |
| H1 on every big layer | The comparisons of H1 again on the preset with every big layer, the configurations without the CausalDiagnosis MCP from the top layer and those with it from the part environment of the agents grid, both on a server without LoRA adapters, on top-1 and on pass | A comparison is confirmed as in H1 and H2, and a baseline at floor makes it uninformative. Its 8 tests share one Holm correction with the 6 tests of H2' in the group `agents_environment`. | `h1_top_cells` and `table_h1_top` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/hypothesis_tables.py`, H1_TOP_pairs |
| H2' | Each agent with the CausalDiagnosis MCP as released against the same agent without it, on the preset with every big layer, on pass and on the reward | A comparison is confirmed when both paired differences lie above zero and both Holm decisions reject. Its 6 tests share one Holm correction with the 8 tests of `h1_top` in the group `agents_environment`, 14 tests in all. | `H2_PRIME` and `H2_PRIME_MEASURES` in `CausalDiagnosis/experiments/table_definitions/experiment_tables.py`, H2PRIME_pairs, `tab:mcptop` |
| H0 | The majority of the judge against the verified pass, per row of the agents table and per comparison of H1 and H2, two-sided | One Holm family contains the 13 tests of the judge. A test rejects when Holm rejects, and the sign of its statistic says whether the judge overstates or understates the verified pass. An episode without a judge record counts as a failure of the judge. A test of a row that misses a planned episode, with an episode judged under another protocol or without any judge record, or with too few test combinations on which the two scorers disagree, counts with a p value of one and remains in the family. | `table_h0` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/judge_tables.py`, H0, `tab:judgerows`, `tab:judgecmp` |
| Layers | The effect of each big layer on pass, each rung of the ladder against the rung below it, a paired difference pooled over the agents with the CausalDiagnosis MCP, two-sided | A secondary family, with Holm over the 5 layers above the base. A test whose rung below lies below the floor of pass is reported as uninformative. | LAYERS_pairs, `tab:layers` |
| Training | Each trained variant against its reference variant on the preset with every big layer | Top-1 and pass are the two measures of the 30 training tests, which share one secondary Holm family. A test holds when its difference lies above zero and its Holm decision rejects. | `TRAINING_COMPARISONS` in `CausalDiagnosis/experiments/table_definitions/experiment_tables.py`, TRAINING_pairs, `tab:trainhyp` |
| Full environment | Each configuration with the CausalDiagnosis MCP on the preset with every big layer against its cell on the base layer, on pass, two-sided | The family `full_environment` is secondary, with a Holm correction of its own over its 3 tests. A difference below zero is the registered expectation, and a test whose cell on the base layer lies below the floor of pass is reported as uninformative. | `table_full_environment` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/layer_tables.py`, FULL_ENV_pairs, `tab:fullenv` |
| Schedule of candidate plans | The CausalDiagnosis agent with the CausalDiagnosis MCP on the preset with every big layer, its schedule of candidate plans on against off, ablation Ag9, on the time to repair in declared cycles | The family `parallel_time` is secondary, with one test. A comparison is confirmed when the difference lies below zero, its test rejects and both pre-registered margins, on pass and on the cost over the oracle, are met. | `table_parallel_pairs` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/plan_schedule_tables.py`, PARALLEL_pairs, `tab:parallelpair` |

---

## Scores of the verifier and the tables

### Top-1

- **Class.** Outcome.
- **Definition.** Whether the submitted diagnosis identifies the hidden level of every component at the cycle of its write.
- **Formula.** `top1_e = 1[h_hat_e = h(t_w)]`, zero for an episode that writes no diagnosis. The rate of a configuration is the mean over `E`.
- **Inputs.** The `diagnosis` of the `submit_diagnosis` entry of `hidden/log.json` and its place in the log, and the field `hidden` of the `reset` entry. The entries before the write that move the combination are included as well, `scheduled_drop`, `hazard_drop`, `work_order_closed` with its `combination_after`, and `emergency_expired`.
- **Computed in.** `verify` and `_hard_checks` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py`, `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py`, which follow the combination to the write with `_combination_after` and write the field `diagnosis_all_correct`. Tests in `test_verifier.py`.
- **Sanity relations.** The oracle submits the current combination, so it scores one in every episode. Doing nothing and always declare write no diagnosis and score zero. With the CausalDiagnosis MCP off and on, the hypotheses H2a to H2c test the paired difference of each agent.
- **Gaming.** The environment refuses a second diagnosis or declaration, compares the levels exactly with the hidden state and reads no answer text, so the only shortcut is a guess. A diagnosis with no reading before it falls in the failure class `premature_diagnosis` only when the episode does not succeed.
- **Hypothesis.** Decides H1 and H2 with pass. A comparison is confirmed when both paired differences lie above zero and both Holm decisions reject over the 14 tests of `tab:hyp`. Top-1 is also one of the two measures of the 30 training tests of `tab:trainhyp`, which share one Holm family. H1_GUIDANCE repeats the comparisons of H1 at another guidance level of the briefing, a robustness check that decides nothing of H1. H1_TOP_pairs asks the comparisons of H1 again on the preset with every big layer, in the family `h1_top`, which shares its Holm correction with H2' in the group `agents_environment`.
- **Table.** `tab:agents`, `tab:hyp`, `tab:ladder`, `tab:misdiagnosis`, `tab:training`, `tab:trainhyp` and `tab:trainfolds`, and AGENTS, AGENTS_pairs, H1_GUIDANCE, H1_TOP_pairs, MISDIAGNOSIS_PRICE, SWITCHES, TRAINING, TRAINING_pairs, FUNNEL, SYSTEM_CONTRAST, PAIRS_BY_FAULT and FIRST_DROP.

### Pass

- **Class.** Outcome.
- **Definition.** The correct diagnosis at the write, the healthy combination at the end, and no work order on a component that was healthy when the order was placed. Under the verifier scope `all`, the field `read_gate` also states whether no sensor group was read more often than the greedy baseline policy reads it from the same initial state.
- **Formula.** `r_e = 1[h_hat_e = h(t_w)] * 1[h(T_e) = h_0] * 1[u_e = 0]`. The pass rate of a configuration is the mean over `E`. The pass with the read gate is `r_e * read_gate_e`.
- **Inputs.** The inputs of top-1, the field `final` of the `end` entry, and the hidden flag `unnecessary` of every `work_order_created` entry. The read gate compares the `read` entries with the readings of the greedy baseline policy from the same initial state, which `hidden/oracle.json` contains.
- **Computed in.** `verify` and `_outcome` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py`, which write the fields `recovered`, `unnecessary` and `pass`, with `_count_gate` for the field `read_gate`. `scope_pass` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` gives the pass with the read gate. Tests in `test_verifier.py` and `test_run_reference_policies.py`.
- **Sanity relations.** The oracle passes on the smallest seed of every initial combination of every preset but the anchored one, which `test_run_reference_policies.py` asserts. Doing nothing and always declare never pass an episode that starts in a faulty combination. For a baseline policy, the column `replay_matches` states whether the log that the action server wrote equals the log of a second environment that the evaluation script steps with the same actions. With the CausalDiagnosis MCP off and on, the hypotheses H2a to H2c test the paired difference of each agent.
- **Gaming.** Ordering every component restores the asset and counts an unnecessary order for every healthy component, so the pass is zero in every episode whose initial state has a healthy component. A closing summary that claims a recovery earns nothing, since `h(T_e)` comes from the hidden state.
- **Hypothesis.** Decides H1 and H2 with top-1. Pass also decides the effect of each big layer, a paired difference pooled over the agents with the CausalDiagnosis MCP, with Holm over the 5 layers above the base. It is the other measure of the 30 training tests. With the reward, it is a measure of the family H2' on the preset with every big layer. On that preset H1_TOP_pairs asks the comparisons of H1 again, FULL_ENV_pairs tests each configuration with the CausalDiagnosis MCP against its cell on the base layer, two-sided, in the family `full_environment`, and PARALLEL_pairs reads the paired difference of pass against a pre-registered margin.
- **Table.** `tab:agents`, `tab:hyp`, `tab:ladder`, `tab:layers`, `tab:misdiagnosis`, `tab:fullenv`, `tab:training`, `tab:trainhyp`, `tab:trainfolds`, `tab:mcptop`, `tab:parallel`, `tab:parallelpair`, `tab:policylayers` and `tab:policies`, whose oracle column prints it, and AGENTS, AGENTS_pairs, H1_GUIDANCE, H1_TOP_pairs, FULL_ENV_pairs, H2PRIME_pairs, LAYERS_pairs, MISDIAGNOSIS_PRICE, POLICIES, POLICY_LAYERS, TRAINING, TRAINING_pairs, RESTORED, PARALLEL, PARALLEL_pairs, SYSTEM_CONTRAST, PAIRS_BY_FAULT, FIRST_DROP, RUNS, A3, A7, A9, A10, SWITCHES, A13_A17 and STRATA. A7 and RUNS print the pass with the read gate.

### Graded Top-1

- **Class.** Outcome.
- **Definition.** The share of the components faulty at the write to which the diagnosis assigns their true level. The score is zero without a diagnosis and zero when the diagnosis contains a false alarm, a healthy component assigned another level. Where no component is faulty at the write, it equals top-1. `graded_top1_n` is the number of components faulty at the write, or at the start of the episode when no diagnosis was accepted.
- **Formula.** `graded_top1_e = |{j in F(h(t_w)) : h_hat_ej = h(t_w)_j}| / |F(h(t_w))|` under the rules of the definition.
- **Inputs.** The inputs of top-1.
- **Computed in.** `_graded` and `verify` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py`, which write the fields `graded_top1` and `graded_top1_n`. Tests in `test_graded_verdict.py`.
- **Sanity relations.** From an initial state with at most one faulty component and no drop during the episode, the score equals top-1 for every episode, which `test_graded_verdict.py` asserts. No table pairs the score between the configurations with the CausalDiagnosis MCP off and on.
- **Gaming.** A diagnosis that assigns a faulty level to a healthy component scores zero, so assigning faulty levels to more components earns no partial credit.
- **Hypothesis.** Descriptive.
- **Table.** The columns `graded_top1` and `graded_top1_n` of `scores.csv`, and no table of the paper.

### Graded pass

- **Class.** Outcome.
- **Definition.** The share of the components faulty at any time of the episode that are healthy at the end, where a component faulty at the write counts only when the diagnosis assigns it its true level. A component is faulty at any time when it is faulty at the start of the episode, at the write, or after a drop. The score is zero when the diagnosis contains a false alarm or a work order was unnecessary. Where no component is faulty at any time, it equals pass. `graded_pass_n` is the number of components faulty at any time.
- **Formula.** `graded_pass_e = k_e / graded_pass_n_e`, with `k_e` the components among the `graded_pass_n_e` that are healthy at the end and, where faulty at the write, assigned their true level by the diagnosis.
- **Inputs.** The inputs of pass.
- **Computed in.** `_graded` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py`, which writes the fields `graded_pass` and `graded_pass_n`. Tests in `test_graded_verdict.py`.
- **Sanity relations.** From an initial state with at most one faulty component and no drop during the episode, the score equals pass for every episode, which `test_graded_verdict.py` asserts. No table pairs the score between the configurations with the CausalDiagnosis MCP off and on.
- **Gaming.** Ordering every component counts an unnecessary order and zeroes the score, as it zeroes pass.
- **Hypothesis.** Descriptive.
- **Table.** The columns `graded_pass` and `graded_pass_n` of `scores.csv`, and no table of the paper.

### Exact match of the faulty set

- **Class.** Outcome.
- **Definition.** Whether an accepted diagnosis puts off their healthy level exactly the components that are faulty at the write, the subset accuracy of multi-label classification. Levels are not compared. The key is `faulty_set_right`, and it is false without an accepted diagnosis.
- **Formula.** `em_e = 1[a diagnosis was accepted and F(h_hat_e) = F(h(t_w))]`. The rate of a configuration is the mean over `E`.
- **Inputs.** The inputs of top-1.
- **Computed in.** `_hard_checks` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py`, which sets the key `faulty_set_right`, with `faulty` of `LevelsTruth` in `CausalDiagnosis/datasets/dataset_loader/ground_truth.py`. Tests in `test_graded_verdict.py`. `exact_match_floor` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agents_table.py` gives the share that the constant answer stating the most frequent faulty set reaches and whether a configuration remains at that floor, with tests in `test_tables_from_runs_registered.py`.
- **Sanity relations.** A diagnosis that is right on every level has the right faulty set, so top-1 implies the exact match, which `test_graded_verdict.py` asserts. No table pairs the rate between the configurations with the CausalDiagnosis MCP off and on.
- **Gaming.** The set is compared with the hidden state at the write, and a superset or a subset of the faulty components scores zero.
- **Hypothesis.** Descriptive.
- **Table.** `tab:agentmeasures`, column Exact match, and `tab:exactmatch`, which prints the rate beside its majority-class baseline, its floor and the mark at floor. The column `faulty_set_right` of `scores.csv`, AGENTS, which contains the rate beside that baseline, RESTORED, whose last column is the share of the restored episodes without the exact match, FUNNEL and SYSTEM_CONTRAST.

### Restored with no unnecessary order

- **Class.** Outcome.
- **Definition.** Whether the asset ends with every component at its healthy level and no work order went to a component that was healthy when the order was placed, whatever the diagnosis. The key is `restored_clean`. It is empty without a verdict, which the counting rule reads as a failure.
- **Formula.** `rc_e = 1[h(T_e) = h_0] * 1[u_e = 0]`. The rate of a configuration is the mean over `E`.
- **Inputs.** The verdict keys `recovered` and `unnecessary`, which the inputs of pass give.
- **Computed in.** `restored_clean` and `add_derived_fields` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agent_descriptive_tables.py`, and `table_restored`, which also counts the restored episodes whose faulty set was not matched exactly. Tests in `test_tables_from_runs_registered.py`.
- **Sanity relations.** Pass is top-1 times this measure, so every episode that passes is restored with no unnecessary order and the rate of a configuration is at least its pass rate, which `test_tables_from_runs_registered.py` asserts.
- **Gaming.** Ordering every component restores the asset and counts an unnecessary order for every healthy component, so the measure is zero in every episode whose initial state has a healthy component. A closing summary that claims a recovery earns nothing, since `h(T_e)` comes from the hidden state.
- **Hypothesis.** Descriptive, with no test and no family.
- **Table.** `tab:judgeagree`, column Restored, and AGENTS and RESTORED, which prints the rate beside pass.

### Success and the credit of a declaration

- **Class.** Outcome.
- **Definition.** A pass, or a declaration of insufficient evidence that cites a reading released before it, in an episode whose initial combination both diagnosability labels mark not diagnosable within the budget.
- **Formula.** `succ_e = r_e or (declared_e and cited_e and sep(s_e) = 0 and sep_E18(s_e) = 0)`, with the labels `sep` and `sep_E18` of the card on the two diagnosability labels. An unknown label counts as diagnosable within the budget.
- **Inputs.** The `evidence_ids` of the `declare_insufficient_evidence` entry, the `read` entries before it with the observation numbers and fields that they returned, and the cell `grade` of the initial combination, which contains the label and the E18 grade.
- **Computed in.** `verify` and `_cites_reading` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py`, with the labels read by `_separable` and `_separable_e18`, which write the fields `success` and `separable`. Tests in `test_verifier.py`.
- **Sanity relations.** Always declare succeeds exactly in the episodes whose initial combination both labels mark not diagnosable within the budget, and doing nothing never succeeds. With the CausalDiagnosis MCP off and on, the credit depends on E18, a rule of the CausalDiagnosis MCP, so success can favor the configurations with the CausalDiagnosis MCP, and the measure is secondary for that reason.
- **Gaming.** A declaration without a reading released before it earns nothing, and neither does a declaration in an episode whose initial combination either label marks diagnosable within the budget. A constant declarer earns at most the share of episodes whose initial combination both labels mark not diagnosable within the budget.
- **Hypothesis.** Secondary, paired in AGENTS_pairs without a Holm decision.
- **Table.** `tab:agentmeasures`, `tab:strata` and `tab:policylayers`, and AGENTS, AGENTS_pairs, STRATA, POLICIES, POLICY_LAYERS and RUNS. `tab:policies` does not print it, since its cells are the pass of the oracle and the reward of every policy.

### Wrong diagnoses

- **Class.** Downside.
- **Definition.** The share of episodes whose submitted diagnosis is wrong. A declaration or an episode without a diagnosis is not wrong.
- **Formula.** `wrong_e = 1[a diagnosis was written and h_hat_e != h(t_w)]`. The share of a configuration is the mean over `E`.
- **Inputs.** The cell `diagnosis_all_correct` of `scores.csv` and the `submit_diagnosis` entry of `hidden/log.json`.
- **Computed in.** `wrong_diagnosis` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py`, which `add_derived_fields` stores as `diagnosis_wrong`. Tests in `test_tables_from_runs.py`.
- **Sanity relations.** The share is zero for the oracle, doing nothing and always declare. With the CausalDiagnosis MCP off and on, the share of each configuration is printed beside the hypotheses H2a to H2c without a test.
- **Gaming.** An agent lowers it by declaring or by writing nothing, so it is read beside the declared share and pass.
- **Hypothesis.** Descriptive.
- **Table.** `tab:agentmeasures` and `tab:misdiagnosis`, column Wrong, and AGENTS and MISDIAGNOSIS_PRICE.

### False alarms and the false-alarm control

- **Class.** Downside.
- **Definition.** `false_alarms` is the number of components at their healthy level at the cycle of the write that the first accepted diagnosis puts at another level, zero without a diagnosis. The false-alarm control runs each of the 6 configurations of the first experiment on the healthy combination at 5 entry offsets, 30 episodes, and reports two counts per configuration, the false alarms and the unnecessary work orders. The briefing opens with the same sentence and asks the same question as in every other episode. The episode is a false alarm of the monitor, and no agent can tell a control episode by its text. The healthy combination is a development combination, and the public normal ranges were set on its recordings, so the control is on the easy side. The page distinguishes three uses of the words by name, a false alarm of the monitor, the false alarms of a diagnosis, and the false alarm rate of the post repair check, the cell `false_alarm` of the table MCP.
- **Formula.** `fa_e = |{j : h(t_w)_j = h_0j and h_hat_ej != h_0j}|`, zero for an episode without a diagnosis. The control reports per configuration, with no rate and no test, the episodes with an accepted diagnosis, the episodes with a false alarm and the components to which they assign a faulty level, and the episodes with a work order and the orders they place, each unnecessary on the healthy combination.
- **Inputs.** The inputs of top-1 for the false alarms, the hidden flag `unnecessary` of the `work_order_created` entries, and the preset `CausalDiagnosis/configs/healthy_control.json` for the control.
- **Computed in.** `verify` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py`, which writes the keys `false_alarms` and `unnecessary`, and `table_healthy_control` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agent_descriptive_tables.py`, which counts the control. Tests in `test_healthy_start.py` and `test_tables_from_runs_registered.py`.
- **Sanity relations.** Every baseline policy runs the false-alarm control to a verified end, and the oracle passes it with both counts at zero, which `test_healthy_start.py` asserts. In any episode, a wrong level on a faulty component is no false alarm, and neither is a component that dropped before the write.
- **Gaming.** A constant healthy answer has no false alarm in the control and fails every episode that starts in a faulty combination, so the control is read beside pass.
- **Hypothesis.** Descriptive, with no test and no family.
- **Table.** The column `false_alarms` of `scores.csv`, and HEALTHY_CONTROL, which prints the counts of the control for each configuration.

### Declarations and the abstention flags

- **Class.** Outcome.
- **Definition.** Whether the agent declared insufficient evidence, whether the declaration cites a reading released before it, and whether it was correct under both labels. A further flag marks an agent that did not declare in an episode whose initial combination the label marks not diagnosable within the budget, an episode that wrote nothing included. A8 reports the precision and the recall of the declarations.
- **Formula.** `correct_e = declared_e and cited_e and sep(s_e) = 0 and sep_E18(s_e) = 0` and `commit_insep_e = not declared_e and sep(s_e) = 0`. Per label, precision is `sum correct / sum declared` and recall is `sum correct / sum 1[inseparable]`. A8 counts a declaration correct under the label when the verifier credits it, and under E18 when it cites a reading in an episode whose initial combination E18 marks not diagnosable within the budget.
- **Inputs.** The declaration entry, the `read` entries before it and the cell `grade` of the initial combination.
- **Computed in.** `verify` and `_cites_reading` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py`, which write the fields `abstained`, `abstention_cites_reading`, `abstention_correct` and `commitment_on_inseparable`, and `table_a8` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/ablation_tables.py`. Tests in `test_verifier.py` and `test_tables_from_runs.py`.
- **Sanity relations.** Always declare declares in every episode, so its precision is the share of episodes in which a declaration is correct, the most a constant declarer reaches. The oracle never declares. With the CausalDiagnosis MCP off and on, the declared share of each configuration appears beside the hypotheses without a test, and A8 reports the precision and the recall per abstention rule and per level of the CausalDiagnosis MCP.
- **Gaming.** As for success. The online loop of GEPA sets the credit of a declaration to zero, so training cannot climb by declaring.
- **Hypothesis.** Descriptive, with A8 as the ablation of the abstention rule.
- **Table.** `tab:agentmeasures` and `tab:misdiagnosis`, column Declared, and `tab:strata`, and AGENTS, MISDIAGNOSIS_PRICE, STRATA, A8 and JUDGE, which counts the declaring episodes.

### Reward

- **Class.** Outcome, a composite of outcome, expense and time.
- **Definition.** The outcome term minus a cost term and a delay term, computed from the verdict and never from a model.
- **Formula.** `R_e = o_e - w_c * min(cost_e / cost*_e, c_max) - w_t * min(delay_e / T, 1)`, with `o_e = 1` for a good episode, a pass whose cited evidence the released record holds (`good` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_reward.py`), `o_e = c_decl` for a correct declaration whose cited evidence the record holds, and `o_e = 0` otherwise. `delay_e` is the larger of the time to work order and the time to repair of the episode, and the horizon `T` when a fault occurrence of the episode was never restored by an agent close (the verdict's `unrestored`). The weights `w_c` and `w_t` are 0.2 and 0.1, the cap `c_max` is 1.5, the constant `COST_CAP`, and the credit `c_decl` is 0.5, the constant `ABSTENTION_CREDIT`. An episode that the verifier cannot read reads zero on the reward under the counting rule, and an episode whose cost over the oracle is undefined has no reward and is excluded from the mean. The five numbers stand in the block `reward` of `CausalDiagnosis/experiments/table_definitions/registered_analysis.json`, which `episode_reward` in `CausalDiagnosis/experiments/episode_runners/run_folder_layout.py` passes to the verifier.
- **Inputs.** The verdict, with its `pass`, `abstention_correct`, provenance check, `cost_normalized`, `detection_delay` and `repair_delay`, and the horizon of the configuration.
- **Computed in.** `reward`, `outcome_gap`, `COST_CAP` and `ABSTENTION_CREDIT` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_reward.py`, which `verify_episode` in `CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py` writes per episode as `reward`. Tests in `test_verifier.py` and `test_run_reference_policies.py`.
- **Sanity relations.** With nonnegative weights each penalty lies between zero and `w_c * c_max + w_t`. Two episodes whose outcome terms differ by at least the smallest gap between outcome levels retain their order whenever that bound lies below the gap, and `reward` refuses weights that break the bound or are negative. The oracle earns at least the reward of every baseline policy on every seed of every preset of `LADDER` and of both deterioration modes, and doing nothing earns less than every policy that the verifier passes, as `test_run_reference_policies.py` asserts. With the CausalDiagnosis MCP off and on, the reward is paired in AGENTS_pairs without a Holm decision.
- **Gaming.** A cited identifier without a match zeroes the outcome term while the cost and delay terms remain. The reward does not order doing nothing against a wrong answer that read. A correct declaration earns the credit `c_decl`, so an agent that rarely passes can gain by declaring in episodes whose initial combination both labels mark not diagnosable within the budget. The online loop of GEPA sets the credit to zero on its stream, while its held-out comparison uses the full reward.
- **Hypothesis.** With pass, it is a measure of the family H2' on the preset with every big layer, a family that shares its Holm correction with `h1_top` in the group `agents_environment`. It is also the training signal of the online loop of GEPA in `CausalDiagnosis/training/gepa_prompt_optimization/`, which scores its candidate instructions by their mean reward.
- **Table.** `tab:policies`, the reward of every policy, `tab:policylayers`, its change from a layer to the next, and `tab:mcptop`, its paired difference on the preset with every big layer, and AGENTS, AGENTS_pairs, H2PRIME_pairs, POLICIES, POLICY_LAYERS, GEPA, GEPA_curve and RUNS. The column `gepa_call` of `scores.csv` numbers the episodes of a loop for GEPA and GEPA_curve.

### Cost and its parts

- **Class.** Expense.
- **Definition.** The prices that the agent incurred, the visit charges, the production loss that the hidden record of costs accrued, the failure penalties and the misdiagnosis charge of a wrong diagnosis, in money. On a configuration that prices nothing, the cost is the number of tool calls of the agent.
- **Formula.** `cost_e = sum_a c_a + sum_f p_f + sum_t lambda_t` in money, with `c_a` the price charged for action `a`, its visit charge included, `p_f` the penalty of failure `f`, and `lambda_t` the production loss in cycle `t`. The production loss is the production loss rate of the stopped asset while a repair stops it and otherwise the rates of its degraded components. In calls, `cost_e` is the number of tool calls, refused calls included, and a turn that calls no tool, a `no_call` entry of the log, counts against the step limit alone.
- **Inputs.** The `cost` of every agent entry, the `cost` of the system entries, which are the penalties, `downtime_total` of the `end` entry, and the price table of the reset configuration or of the released rules.
- **Computed in.** `_costs` and `_reset_config` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_costs.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py`, with the rule `cost_unit` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_1_priced.py` over the prices of the reset entry, which write the fields `cost`, `cost_unit`, `cost_prices`, `penalties` and `downtime_total`. Tests in `test_verifier.py`.
- **Sanity relations.** Doing nothing incurs no price, so cost alone does not order the policies by quality and is read beside pass. With the CausalDiagnosis MCP off and on, a call of the CausalDiagnosis MCP has no price and writes no entry of the log. The CausalDiagnosis MCP moves the cost only through what the agent does and, where the clock charges the wall seconds of thinking, through the cycles that its calls add, while the `generated_tokens` mode charges its calls no cycle.
- **Gaming.** The environment charges every term from the hidden record of costs. An agent lowers the cost by acting less, which pass catches, or by reading and repairing sooner and at a lower price, which is no gaming.
- **Hypothesis.** Descriptive.
- **Table.** `tab:agentmeasures`, columns Cost and Production loss, AGENTS, A3 with the cost of the gym episodes, A9 and A13_A17, and through the cost over the oracle `tab:ladder`. `tab:misdiagnosis` and MISDIAGNOSIS_PRICE print the cost without the misdiagnosis charge and the charge apart, columns Cost without the charge and Charge.

### Cost over the oracle

- **Class.** Expense.
- **Definition.** The cost of the episode divided by the cost of the oracle policy on the same seed, and between two configurations the ratio of their mean cost over matched episodes. The oracle reads nothing, so an agent that reads before it makes the repairs of the oracle lies above one.
- **Formula.** `c_e = cost_e / cost*_e`, undefined when the cost of the oracle is zero or unknown, and `sum_pairs cost_a / sum_pairs cost_b` between the configurations `a` and `b`.
- **Inputs.** The cells `cost` and `cost_oracle` of `scores.csv`, the oracle replayed on the same seed.
- **Computed in.** `_costs` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_costs.py`, field `cost_normalized`, with the cost of the oracle from `oracle_reference` in `CausalDiagnosis/experiments/episode_runners/run_episodes/reference_policy_episodes.py`. `cost_ratio` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/hypothesis_tables.py` gives the ratio between two configurations, with an interval clustered by test combination and the p value of the paired cost difference. Tests in `test_verifier.py`, `test_run_reference_policies.py` and `test_tables_from_runs.py`.
- **Sanity relations.** The oracle scores one by construction. Doing nothing can fall below one, since it incurs no repair cost, so the ratio favors inaction and is read only beside pass. With the CausalDiagnosis MCP off and on, AGENTS_pairs prints the ratio of mean cost for the hypotheses H2a to H2c without a Holm decision.
- **Gaming.** Acting less lowers it while pass catches the missed repairs, and a sooner repair lowers it as a legitimate gain.
- **Hypothesis.** Printed beside the hypotheses H1 and H2, which it does not decide. On the preset with every big layer it is one of the two pre-registered margins of the family `parallel_time`.
- **Table.** `tab:agentmeasures` and `tab:ladder`, column Cost / oracle, `tab:parallel` and `tab:parallelpair`, and AGENTS, AGENTS_pairs as `cost_ratio`, A9, SWITCHES, RUNS, PARALLEL and PARALLEL_pairs.

### Model calls and tokens

- **Class.** Expense.
- **Definition.** The calls of the model in an episode and their input and output tokens. Beside them the harness records how the run of the agent ended, in bookkeeping columns that apply to every agent alike. `finish_by` says whether the agent or the harness wrote the finish, and it is empty when the environment ended the episode without one. `runner_end` is the end that the runner reports, one of finish, own_limit, episode_over, cancelled and error, and it is empty for an agent without a runner. `wall_limit` is one when the run was cut at the wall limit of an episode and zero otherwise. `error_class` is empty or one of agent, harness_error and verifier. `replays` is one when the episode was run once more after an attempt that the infrastructure failed, and zero otherwise. `diagnosis_refusals` maps each reason of a refused diagnosis to its count. The thinking clock adds its own columns. `credited_s` and `uncredited_s` give the seconds of the server starts and of the counted calls that the clock credited and that it charged, as a JSON object with the keys starts and calls. `failed_model_s` and `failed_model_attempts` give the seconds and the count of the model requests that the service failed. `uncharged_after_last_s` is the time from the last counted call to the end of the agent's part, which no call charges. `cut_end` is service or wall for an episode cut at the wall limit and empty otherwise. `clock_skew` is one where the wall and the monotonic lengths of the agent's part differ by more than a second. `server_cache_flushed` is one when the harness flushed the cache of the model server before an episode of a preset with the wall clock, and it is empty on another clock and for an agent without a model endpoint. A flush that the server refuses is a step of the harness that raised, so the episode does not start and the attempt is the infrastructure's.
- **Formula.** `calls_e = number of model calls` and `tokens_e = sum over calls of input tokens + output tokens`. `cut_replies` counts the model calls whose finish reason is length. `failed_calls` counts the model calls that ended in an error after the retries of their client, plus the tool calls without an answer.
- **Inputs.** `llm_calls.json`, one record per call with its tokens and its seconds. The bookkeeping columns read the same file, `trajectory.json`, the refused entries of `hidden/log.json`, the end that the runner reports and the record `cache_flush` of the state file `episode.json`.
- **Computed in.** `record_causal` in `CausalDiagnosis/experiments/episode_runners/run_episodes/agent_runs.py`, `record_agent_as_tool` in `CausalDiagnosis/frameworks/AssetOpsBench/agents/agent_runs.py` and `record_plan_execute` in `CausalDiagnosis/frameworks/AssetOpsBench/agents/scripted_wiring_check.py`, which write the cells `llm_calls`, `input_tokens` and `output_tokens`, and `tokens_used` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py`. Tests in `test_run_episodes.py` and `test_paper_tables.py`.
- **Sanity relations.** No baseline policy makes a model call. With the CausalDiagnosis MCP off and on, CALLS pairs every agent against Agents-as-Tools with and without the CausalDiagnosis MCP.
- **Gaming.** Fewer and longer calls lower the count, so the tokens stand beside it.
- **Hypothesis.** Descriptive. CALLS applies Holm over its differences of model calls, a table of the evaluation scripts that the paper does not print.
- **Table.** `tab:agentmeasures` and `tab:parallel`, column Model calls, and AGENTS, CALLS, A13_A17 and RUNS, which prints the input tokens per call.

### Steps and refusals

- **Class.** Expense.
- **Definition.** Every attempted tool call of the action or metered servers and of the CausalDiagnosis MCP, failed and refused calls included, under one rule for every agent, and the refused calls among them. The verdict contains three cells beside them. The environment steps, the cell `steps`, count the entries of the log that took a step of the environment, without the calls of the CausalDiagnosis MCP. `end_reason` is the reason of the `end` entry, and `time_used` is the time of that entry. `react_steps` is the step counter of the supervisor of Agent-As-Tool as its framework logs it, empty for every other agent.
- **Formula.** `steps_e = tool_calls_e + causal_calls_e`, and `refusals_e` is the number of `refused` entries of the hidden log, the refusals of the environment.
- **Inputs.** The state file `episode.json` with its `tool_calls`, and `trajectory.json` for the calls of the CausalDiagnosis MCP.
- **Computed in.** `tool_counts` and `diagnostics_calls` in `CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py` and `CausalDiagnosis/experiments/episode_runners/run_episodes/agent_runs.py`, which write the cells `tool_calls` and `causal_calls`. `verify_episode` in the same module replaces the cell `refusals` with the count of `refused` log entries, and `tool_invocations` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` adds the two cells. Tests in `test_run_episodes.py`, `test_tables_from_runs.py` and `test_paper_tables.py`.
- **Sanity relations.** `causal_calls` is zero on every configuration without the CausalDiagnosis MCP by construction, since no server of the CausalDiagnosis MCP is registered there, and no code checks it.
- **Gaming.** No objective reads it.
- **Hypothesis.** Descriptive.
- **Table.** `tab:efficiency`, column Steps, and EFFICIENCY.

### Runtime

- **Class.** Expense.
- **Definition.** The wall clock around the agent in seconds, which stops before the verifier and the judge.
- **Formula.** `wall_e`, the seconds from the start of the run of the agent to its end.
- **Inputs.** The timer of the episode runner.
- **Computed in.** `EpisodeRunner.play` in `CausalDiagnosis/experiments/episode_runners/run_episodes/`, which writes the cell `wall_s` of `scores.csv`. Tests in `test_tables_from_runs.py` and `test_paper_tables.py`.
- **Sanity relations.** For the baseline policies, which run without a model, the runtime includes the start-up of a server process for each call through the action server, which no test measures. With the CausalDiagnosis MCP off and on, the runtime includes the seconds that the CausalDiagnosis MCP computes, since its calls run inside the run of the agent.
- **Gaming.** No objective reads it. A suspension or a sleep of the machine inflates it.
- **Hypothesis.** Descriptive.
- **Table.** `tab:efficiency`, column Runtime, and EFFICIENCY.

### Thinking cycles

- **Class.** Expense.
- **Definition.** The cycles that the clock charges for thinking, one cycle per `Delta_think` seconds with the rest carried to the next call. The seconds are the wall seconds between two calls of the agent that reached the environment in the `generation_time` mode, and the tokens that the agent's model generated between them times `seconds_per_token` in the `generated_tokens` mode. Only these two clock modes charge them.
- **Formula.** `think_e = sum_a k_a` with `k_a = floor((r_a + theta(a)) / Delta_think)` and `r_(a+1) = (r_a + theta(a)) mod Delta_think`. Here `theta(a)` includes the model calls, their serving, the calls of the CausalDiagnosis MCP and everything else between two calls in the `generation_time` mode, and the tokens generated since the previous call times `seconds_per_token` in the `generated_tokens` mode. `Delta_think` is `think_seconds_per_cycle` of the configuration.
- **Inputs.** `think_cycles` of every entry of `hidden/log.json`, and the seconds behind them in `think_seconds` of `episode.json`.
- **Computed in.** `think_cycles` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/wall_clock.py` charges the cycles at every call, and `thinking_cycles` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agent_descriptive_tables.py` sums them per episode. `thinking_split` and `table_thinking` in the same module split the seconds charged. Tests in `test_tables_from_runs.py`.
- **Sanity relations.** The count is zero on the configurations whose clock charges no thinking and under the thinking switch off. With the CausalDiagnosis MCP off and on, CALLS pairs it as it pairs the model calls. Since `theta(a)` includes the calls of the CausalDiagnosis MCP, the compute time of the CausalDiagnosis MCP is charged here.
- **Gaming.** The harness measures the wall time and the agent cannot report it, while a suspension or a sleep of the machine inflates it.
- **Hypothesis.** Descriptive.
- **Table.** `tab:thinking`, the cycles and the seconds charged for thinking with their split into the generation of the model, the wall clock less the monotonic clock and the rest, and THINKING, CALLS and A13_A17.

### Price of the readings

- **Class.** Expense.
- **Definition.** The prices that the episode was charged for its readings.
- **Formula.** `test_e = sum over read entries of their cost`.
- **Inputs.** The `cost` of the `read` entries of `hidden/log.json`.
- **Computed in.** `read_price` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py`, which `add_derived_fields` stores as `test_cost`. Tests in `test_tables_from_runs.py`.
- **Sanity relations.** The price is zero for the oracle and for doing nothing, and it is the price of one cycle of the cheapest group for always declare. With the CausalDiagnosis MCP off and on, no table compares it, since A13_A17 alone prints it, for the switches of the CausalDiagnosis agent with the CausalDiagnosis MCP.
- **Gaming.** No objective reads it.
- **Hypothesis.** Descriptive.
- **Table.** A13_A17, column `test_cost`.

### Time to work order

- **Class.** Outcome.
- **Definition.** Per fault occurrence of a component, the cycles from its onset to the first work order the agent creates on that component at or after the onset and before the component returns to its healthy level, per component the largest over its occurrences, and per episode the largest over the components. The onset is the cycle at which the hidden record takes the component off its healthy level, by a deterioration transition, a scheduled or hazard drop, the expiry of an emergency repair or a new fault after a restoring close, and cycle zero for a fault present at the start.
- **Formula.** `d_order(e) = max_(j,k) (t_first_order(j,k) - onset(j,k))` over the occurrences k of the components j that have an order in their window `[onset(j,k), end(j,k))`, with `onset(j,k)` the cycle of the hidden entry that began the occurrence, zero for a fault present at the start, and `end(j,k)` the entry that ended it. An occurrence never ordered is excluded from the maximum.
- **Inputs.** The field `hidden` of the `reset` entry, every entry that changes a level (`scheduled_drop`, `hazard_drop`, `work_order_closed` and `emergency_expired` with their `combination_after`), and the `t_start` of the `work_order_created` entries.
- **Computed in.** `fault_occurrences`, `fault_onset` and `_detection_delays` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/delays.py`, field `detection_delay`, whose largest value over the components `verdict_row` in `CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py` writes to `scores.csv`. Tests in `test_verifier.py` and `test_tables_from_runs.py`.
- **Sanity relations.** The oracle orders a faulty component as soon as it will not drop further and its repair will take. A component never ordered has no value, so doing nothing has none, and the mean of a configuration runs over the episodes that have one. With the CausalDiagnosis MCP off and on, AGENTS prints the mean of each configuration without a test.
- **Gaming.** An order placed before the drop counts for no occurrence of that fault, and ordering everything early breaks pass. A second fault of a component and the expiry of an emergency repair start an occurrence of their own, so a late order after them is not hidden by the first one.
- **Hypothesis.** Descriptive. It is part of the delay term of the reward.
- **Table.** AGENTS, SWITCHES and RUNS, which prints the median of the larger of the two delays.

### Time to repair

- **Class.** Outcome.
- **Definition.** Per fault occurrence that an agent close ended, the cycles from its onset to that close, per component the largest over its occurrences, and per episode the largest over the components. The onset is the one onset of the time to work order, read from the hidden record and not from any reading of the agent. The time to repair in declared cycles, the key `repair_declared`, takes from each delay the cycles charged for thinking between the onset and the creation of the order whose close restored the component, counts a component that the agent never restored and an episode without a verdict at the horizon of the preset, and is zero when no component moved off its healthy level.
- **Formula.** `d_repair(e) = max_(j,k) (t_restoring_close(j,k) - onset(j,k))` over the occurrences that a restoring close ended. An occurrence never restored is excluded from the maximum. In declared cycles `d_declared(e) = max_(j,k) (t_restoring_close(j,k) - onset(j,k) - think(j,k))` over every occurrence, with `think(j,k)` the cycles charged for thinking from the onset to the creation of the restoring order and the horizon `H` for an occurrence never restored.
- **Inputs.** The entries that change a level, as for the time to work order, and the `work_order_closed` entries with their `source`. The declared cycles also read the `think_cycles` of the entries of `hidden/log.json`.
- **Computed in.** `fault_occurrences`, `fault_onset`, `restored_by_agent` and `_repair_delays` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/delays.py`, field `repair_delay`, whose largest value over the components `verdict_row` in `CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py` writes. `repair_declared` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` gives the time in declared cycles. Tests in `test_verifier.py` and `test_tables_from_runs_registered.py`.
- **Sanity relations.** As for the time to work order, with no value for a component never restored. With the CausalDiagnosis MCP off and on, AGENTS prints the mean of each configuration without a test.
- **Gaming.** Every fault counts from its onset in the hidden record, since a delay that waited for the reading of the agent would reward not reading or reading late. In declared cycles a fault never restored counts at the horizon, so a fault that remains unrepaired never shortens the time. Thinking after the restoring order stays in the count, also in a visit of several orders, where a later order can delay the close.
- **Hypothesis.** Descriptive on every configuration but one. The time in declared cycles is the measure of the family `parallel_time`, whose one test in PARALLEL_pairs compares the CausalDiagnosis agent with the CausalDiagnosis MCP on the preset of training against the same agent under ablation Ag9, every reading step alone with its own ranking, below zero, with pre-registered margins on pass and on the cost over the oracle. The time to repair is part of the delay term of the reward.
- **Table.** `tab:agentmeasures`, `tab:parallel` and `tab:parallelpair`, and AGENTS, SWITCHES, RUNS, PARALLEL and PARALLEL_pairs.

### Turns to a correct diagnosis

- **Class.** Benefit.
- **Definition.** The step of the diagnosis when it is all correct, summarized over a configuration as the restricted mean survival time over turns, censored at the step limit.
- **Formula.** `RMST = integral from 0 to K of S_hat(k) dk`, with `S_hat` the Kaplan-Meier curve of the turn of a correct diagnosis and the episodes without one censored at `K`, the smallest step limit among the runs of the configuration. Since an episode is censored only at the step limit, `RMST = mean_e min(T_e, K)` with `T_e` the turn of the correct diagnosis of episode `e`, or its step limit where it has none.
- **Inputs.** The `step` of the `submit_diagnosis` entry, the top-1 cell, and `max_steps` of the configuration in `run.json`.
- **Computed in.** `verify` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py`, field `turns_to_correct_diagnosis`, `turn_record`, `capped_turns`, `turns_summary` and `turns_pair` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py`, `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agents_table.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/hypothesis_tables.py`, with `cluster_bootstrap` and `ratio` of `CausalDiagnosis/evaluation/metrics/table_statistics.py`. Tests in `test_tables_from_runs.py` and `test_table_statistics.py`.
- **Sanity relations.** The oracle diagnoses at its first step, the smallest value, while doing nothing and always declare are censored at `K`, the largest. With the CausalDiagnosis MCP off and on, AGENTS_pairs prints the paired difference for the hypotheses H2a to H2c without a Holm decision.
- **Gaming.** A wrong early guess is censored, so only a correct diagnosis shortens it.
- **Hypothesis.** Descriptive, paired in AGENTS_pairs as `turns_rmst_diff` without a Holm decision.
- **Table.** AGENTS as `turns_rmst`, and AGENTS_pairs.

### Realized elimination and the forecast of the CausalDiagnosis MCP

- **Class.** Benefit.
- **Definition.** For every charged reading, the effective number of candidate root causes of the filter of the CausalDiagnosis MCP before the reading minus that after it, with the filter replayed over the released evidence. A configuration reports it per reading and per unit of the cost that the episode was charged for the reading. On the configurations with the CausalDiagnosis MCP it appears beside the expected elimination that the CausalDiagnosis MCP forecast for the same reading on the same belief, with the gap between the two.
- **Formula.** `N(b) = exp(H(b))` with `H` in nats, `elim(a) = N(b_before) - N(b_after)`, per reading `sum elim / reads`, per cost `sum elim / sum C_t(a)`, and `gap = mean realized - mean expected` over the readings with a forecast.
- **Inputs.** `released.json` with its readings, prices and cycles, the `elimination` rows of the `rank_root_causes` answers in `trajectory.json`, the initial combination, and `models.json`.
- **Computed in.** `read_records`, `calibrate` and `episode_measures` in `CausalDiagnosis/evaluation/metrics/realized_read_elimination.py`, and `elimination_fields`, `table_elim` and `table_elim_pairs` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/replayed_fields.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/mcp_tables.py`. Tests in `test_realized_read_elimination.py` and `test_tables_from_runs.py`.
- **Sanity relations.** The oracle reads nothing, so it has no value, and always declare reads one cycle of the cheapest group. With the CausalDiagnosis MCP off and on, the yardstick is one replay for every configuration, so ELIM_pairs pairs the comparisons of H1 and H2 on it.
- **Gaming.** The yardstick shares no state with the agent, so a cheap reading that tells nothing scores near zero. The forecast gap measures the CausalDiagnosis MCP and not the agent.
- **Hypothesis.** Descriptive, with no Holm decision.
- **Table.** ELIM, ELIM_pairs and A13_A17, which prints the elimination per reading, and no table of the paper.

### Calibration of the ranking in episodes

- **Class.** Benefit.
- **Definition.** Over every answer of `rank_root_causes` that the hidden log places, the squared error of the probability of the first candidate root cause against whether that candidate root cause is the hidden combination at the moment of the answer. Beside it stands the share of answers whose first candidate root cause is that combination, per number of readings before the answer, with 10 calibration bins. A reading is a call whose log entry is a `read`, so a refused reading request and a call that never reached the engine count nothing. On a configuration with a clock, the truth follows the log up to the last entry with a step that ended at or before the `belief_time` of the answer. On a configuration without a clock, it follows the entry of the last call before the answer that took a step, with the closings that an order logged inside its own call. An episode whose transcript and log do not agree places no answer, and an answer without a truth gives no point.
- **Formula.** `BS = (1/m) sum_q (p_q - y_q)^2`, with `p_q` the probability of the first candidate root cause of answer `q`, `y_q = 1[first candidate of q = h(t_q)]` and `h(t_q)` the hidden combination at the moment of answer `q`.
- **Inputs.** `transcript.jsonl` with the answers and the calls that took a step, and `hidden/log.json`.
- **Computed in.** `rankings` and `table_a12` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/ablation_tables.py`, with `ranked_answers` and `answer_truths` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`. Tests in `test_tables_from_runs.py` and `test_mcp_metrics.py`.
- **Sanity relations.** The baseline policies call no server of the CausalDiagnosis MCP and the configurations without the CausalDiagnosis MCP have no answer, so only the configurations with the CausalDiagnosis MCP are included in it. A calibrated ranking has an observed frequency equal to the mean prediction in every bin.
- **Gaming.** An agent changes the answers of the CausalDiagnosis MCP through the evidence it is charged for and through when it asks, which sets the answers that the configuration includes.
- **Hypothesis.** Descriptive, the ablation of the ranking model.
- **Table.** A12 and `A12_curves.csv`, and no table of the paper.

### Prediction verdict

- **Class.** Benefit.
- **Definition.** For every work order that cites a prediction, whether the prediction exists, its predicted probability of recovery, the signed error of that prediction against the recovery at the end of the episode, and the field `verification`. The end of the episode is a later target than the one that the prediction specifies, the first reading after the last of its work orders closes. A later repair, or a drop on a configuration with deterioration, moves the target away from what was predicted. The calibration of the predicted probability of recovery among the measures of the CausalDiagnosis MCP scores the same prediction at its verifying reading.
- **Formula.** `err_o = 1[h(T_e) = h_0] - R_o` for every order `o` that cites a prediction with the predicted probability of recovery `R_o`.
- **Inputs.** The `prediction_id` of the `work_order_created` entries and the `recovery` of the records of `predictions.json`.
- **Computed in.** `_prediction_verdict` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py`, which fills the column `prediction_verdict` of `SCORE_COLUMNS` in `CausalDiagnosis/experiments/episode_runners/run_folder_layout.py`. Tests in `test_verifier.py`.
- **Sanity relations.** The configurations without the CausalDiagnosis MCP have no prediction, and no baseline policy cites one. A predictor that knew the end state of the episode would err by zero, while a perfect predictor of the reading that the prediction specifies errs whenever a later repair or a drop changes the end state. The field `verification` is empty on every row, since `_store_prediction` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/tools/` writes no such key. The verdict of the post repair check has no part in this score, and the error rates of that check replay it offline.
- **Gaming.** An agent cannot change `R_o`, but it chooses which predictions it acts on, so the errors describe the plans an agent chose and not every candidate plan that the CausalDiagnosis MCP offered. Every order of a plan cites the one plan identifier, so a plan of several orders repeats its row once per order.
- **Hypothesis.** Descriptive.
- **Table.** None.

### Failure class

- **Class.** Downside.
- **Definition.** The first of 12 classes that describes an episode that did not succeed or whose provenance check failed, decided in a fixed order, or none. A repeated malformed call is 3 or more refused actions in one episode.
- **Formula.** A classification in the order of `FAILURE_CLASSES`, `premature_diagnosis`, `unmatched_evidence`, `unnecessary_replacement`, `repair_before_dependency`, `temporary_repair_expired`, `abstention_without_reading`, `abstention_on_separable`, `abstention_on_e18_separable`, `summary_contradicts_log`, `repeated_malformed_call`, `anchoring_on_first_anomaly` and `recovery_without_diagnosis`, and `none` for a success and for a failure that no class describes.
- **Inputs.** The log, the verdict, the public normal ranges, the grade and the closing summary.
- **Computed in.** `_failure_class`, `FAILURE_CLASSES` and `_summary_check` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/failure_class.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/summary_consistency.py`, field `failure_class`, with the threshold `REPEATED_REFUSALS`. Tests in `test_verifier.py`. `table_failures` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agent_descriptive_tables.py` counts the classes per configuration, with tests in `test_tables_from_runs_registered.py`.
- **Sanity relations.** The class of the oracle is none in every episode, and always declare falls in an abstention class in every episode whose initial combination a label marks diagnosable within the budget. With the CausalDiagnosis MCP off and on, no table sets the classes of two configurations against each other.
- **Gaming.** The class is diagnostic and scores nothing. The class of a summary that contradicts the log reads the text by a conservative keyword rule.
- **Hypothesis.** Descriptive. The feedback card of GEPA in `CausalDiagnosis/training/gepa_prompt_optimization/feedback_card.py` includes it.
- **Table.** FAILURES, the share of each class per configuration beside the closing claims of recovery that the hidden state contradicts.

### Provenance and the unmatched share

- **Class.** Downside.
- **Definition.** Whether every evidence identifier and prediction identifier that a write cites exists in the released evidence or among the stored predictions. The unmatched share of a configuration is the share of its episodes that cite at least one identifier without a match.
- **Formula.** `prov_e = 1[no cited identifier lacks a match]` and `um_e = 1 - prov_e`. The unmatched share of a configuration is the mean of `um_e` over `E`.
- **Inputs.** The evidence identifiers of the diagnosis, the declaration and the orders, the prediction identifiers of the orders, `released.json` and `predictions.json`.
- **Computed in.** `_provenance`, `_cited_identifiers` and `_released_identifiers` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py`, with the column `provenance_ok` that `verdict_row` in `CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py` writes. `unmatched_evidence` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` gives the complement per episode. Tests in `test_verifier.py` and `test_tables_from_runs.py`.
- **Sanity relations.** The check is satisfied for every baseline policy, which cite at most what they read. With the CausalDiagnosis MCP off and on, the prediction identifiers exist only on the configurations with the CausalDiagnosis MCP, so a prediction cited on a configuration without it has no match.
- **Gaming.** The check is the detector. A provenance failure zeroes the outcome term of the reward. A rise of the unmatched share and of the declared share without a gain in pass is the sign of an agent that games the verifier.
- **Hypothesis.** Descriptive.
- **Table.** `tab:judgeagree` and `tab:training`, column Unmatched, and AGENTS and TRAINING.

### The write count, timing and ordering checks

- **Class.** Downside.
- **Definition.** Whether the episode passed each of three checks of the verifier, one boolean per episode in the cells `write_count_ok`, `timing_ok` and `ordering_ok` beside `provenance_ok`. Each cell is the ok of one named check of the verdict, `write_count_ok` of the check `count_gate`, `timing_ok` of the check `timing` and `ordering_ok` of the check `causality`, so the ordering that the paper names and the causality check of the verifier are one check. The write count check holds when the log holds exactly one diagnosis or declaration, exactly one finish and no write through a tool that the oracle does not name, and under the scope `all` the read gate as well. The timing check holds when every work order on a component that degrades during the episode closes no later than the window end of the run at the worst level that is open at its close. The ordering check holds when the diagnosis or the declaration precedes every work order in the log. The cells are reported for every episode, and pass reads none of them.
- **Formula.** `wc_e`, `tm_e` and `or_e` are the indicators that the count gate, the timing check and the causality check of episode `e` hold, and the failure shares of a configuration are the means of `1 - wc_e`, `1 - tm_e` and `1 - or_e` over `E`, while the cells hold the ok indicators.
- **Inputs.** The checks of `verdict.json`, which the verifier builds from the hidden log of the episode, its work orders and the oracle of the episode.
- **Computed in.** `verify` with `_count_gate`, `_timing` and `_causality` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py` and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py`, with the columns that `verdict_row` in `CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py` writes through `CHECK_COLUMNS` and `_check_ok`. `OUTCOME_COUNTING` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/cell_statistics.py` counts an episode that the verifier did not score as a failure on each. Tests in `test_check_columns.py` and `test_verifier.py`.
- **Sanity relations.** Pass reads none of the three checks, so an episode that passes can fail any of them, and a failed check of one kind leaves the cells of the other two unchanged.
- **Gaming.** Neither the reward nor any objective reads them.
- **Hypothesis.** Descriptive.
- **Table.** None.

### pass^k

- **Class.** Outcome.
- **Definition.** The probability that `k` episodes of one test combination, drawn without replacement from the replicates that a configuration runs of it, all pass. The replicates of a combination start at different entry offsets, so pass^k reads how reliably a configuration passes a combination from each of its initial states, where the pass rate averages over them.
- **Formula.** `pass^k = (1 / |S_k|) * sum_{s in S_k} C(c_s, k) / C(n_s, k)`, the unbiased estimator of tau-bench, with `c_s` passes among the `n_s` episodes of test combination `s` and `S_k` the combinations with `n_s >= k`, for every `k` from one to `REPLICATES`.
- **Inputs.** The cell `pass` of every episode of a configuration and its `block`, the test combination.
- **Computed in.** `pass_hat_k` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agents_table.py`, which applies `pass_power_k` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_reward.py` to each combination and takes the interval from a bootstrap that resamples the test combinations, for every `k` up to `REPLICATES` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/start_state_seeds.py`. Tests in `test_verifier.py` and `test_tables_from_runs.py`.
- **Sanity relations.** On a configuration whose combinations all have the same number of episodes, pass^k at `k` equal to one is the pass rate, and pass^k does not rise with `k`. With the CausalDiagnosis MCP off and on, the agents table prints it beside pass without a test.
- **Gaming.** It reads the cell `pass` alone, so it has the shortcuts of pass and no other.
- **Hypothesis.** Descriptive. The agents table prints it at the `k` of `PASS_K_SHOWN` in `CausalDiagnosis/evaluation/result_tables/paper_tables/body_tables.py`.
- **Table.** `tab:agents`, column Pass^k, and AGENTS.

### Judge scores and the judge's pass

- **Class.** Outcome.
- **Definition.** The scores of the AssetOpsBench judge under its protocol of 5 calls at temperature 0.3. Per criterion, task completion TC, data retrieval accuracy DRA and result verification RV, the score is the share of the planned calls that answer true. The pass of the judge, the cell `judge_pass`, is the majority of the planned calls, with a failed call counted false. An episode without a judge record counts as false on every criterion and on the pass, as `counting.unjudged` of the registered analysis states, in every table of the judge and in the tests of H0.
- **Formula.** `TC_e = (1/n_calls) sum_calls 1[true]`, likewise DRA and RV, and `judge_pass_e = 1[(1/n_calls) sum_calls 1[passed] > 1/2]`, with `n_calls` the planned calls of the protocol.
- **Inputs.** `hidden/judge.json` with its protocol, its calls and the details of every call.
- **Computed in.** `judge_fields` and `table_judge` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py`, `CausalDiagnosis/evaluation/result_tables/tables_from_runs/judge_tables.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/hypothesis_tables.py`, with `holm_family` and `reversed_order` for the scoring pairs and `table_h0` for the tests of the judge against the verified pass, and `PROTOCOLS` and `summary_of` in `CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/episode_judge.py`. Tests in `test_paper_tables.py` and `test_episode_judge.py`.
- **Sanity relations.** The judge and the verifier score an episode by different rules, so an episode can pass the judge while the verifier fails it. The tests of H0 measure that difference per configuration and per comparison. With the CausalDiagnosis MCP off and on, the scoring pairs of AGENTS_pairs test whether the verified pass reverses the order that the judge gives two configurations.
- **Gaming.** A confident answer text can raise these scores, while it earns nothing from the verifier.
- **Hypothesis.** Secondary, with the scoring pairs in a Holm family of their own. The disagreement of the majority of the judge with the verified pass is the family H0, whose row tests and comparison tests share one Holm family.
- **Table.** `tab:agents`, `tab:ladder` and `tab:training`, columns TC, DRA and RV, `tab:judgerows` and `tab:judgecmp`, and JUDGE, TRAINING, AGENTS_pairs, RUNS, H0 and A3. A3 prints the pass of the judge on the answers of the static mode as `static_judge_pass`, where a static answer without a judge record, an empty answer among them, counts as not passed when its run names a judge, beside `static_faulty_set_correct`, whether the faulty set that a static answer states is the true one, where an answer that the parser cannot read names no faulty component. The static mode also writes `static_repairs_match_oracle`, whether the repairs that the answer lists are the work orders of the oracle, which no table prints.

### Agreement of the judge's calls, and the kappa against the verifier

- **Class.** Outcome.
- **Definition.** The mean pairwise agreement of the parsed calls of an episode per criterion and its complement, the uncertainty, as Table 14 of AssetOpsBench reports it. Beside it stands Cohen's kappa of the majority of the judge against the verified pass, over the episodes that have both, and against each of four constructs of the verifier, pass, top-1, the exact match of the faulty set and restored with no unnecessary order.
- **Formula.** `agree_e = share of the pairs of parsed calls that agree`, and `kappa = (p_o - p_e) / (1 - p_e)` over the episodes of a configuration.
- **Inputs.** `hidden/judge.json` and the pass cell.
- **Computed in.** `pairwise_agreement`, `table_judge_agreement`, `judge_kappa` and `judge_agreements` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/judge_tables.py`, and `cohens_kappa` in `CausalDiagnosis/evaluation/metrics/table_statistics.py`. Tests in `test_paper_tables.py`, `test_tables_from_runs.py` and `test_table_statistics.py`.
- **Sanity relations.** The kappa is undefined when either rater puts every episode on one side. With the CausalDiagnosis MCP off and on, JUDGE_AGREEMENT pools the six configurations of a model, so it does not compare them.
- **Gaming.** Not an objective.
- **Hypothesis.** Descriptive.
- **Table.** JUDGE_AGREEMENT for the agreement of the calls. `tab:judgeagree` prints the kappa against the four constructs, from JUDGE, and `tab:training` and `tab:judgerows` the kappa against the verified pass, from TRAINING and H0.

### Planned steps

- **Class.** Expense.
- **Definition.** The steps of the plan of Plan-and-Execute and of all the plans of the CausalDiagnosis agent in an episode, as Table 12 of AssetOpsBench reports them. Three further columns count the planning of the CausalDiagnosis agent. `plans` is the number of its plans, `revisions` the plans it made after a surprise ended the plan before, and `fallbacks` the turns that produced no usable plan and an orientation reply that could not be parsed on either attempt.
- **Formula.** `plan_steps_e = sum over plans of the number of steps`.
- **Inputs.** The `plan_steps` cell of `scores.csv` and `plans.json`.
- **Computed in.** `planned_steps` and `table_plans` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agent_descriptive_tables.py`. `record_causal` in `CausalDiagnosis/experiments/episode_runners/run_episodes/agent_runs.py` writes the three planning columns. Tests in `test_paper_tables.py`.
- **Sanity relations.** The count is undefined for Agents-as-Tools and the baseline policies, which make no plan. With the CausalDiagnosis MCP off and on, PLANS prints the configurations of Plan-and-Execute and of the CausalDiagnosis agent with and without the CausalDiagnosis MCP, without a test.
- **Gaming.** Not an objective.
- **Hypothesis.** Descriptive.
- **Table.** PLANS.

### Measures of the agent's switches

- **Class.** Outcome.
- **Definition.** The measures of the switches of the CausalDiagnosis agent with the CausalDiagnosis MCP. The level accuracy is the top-1 on the episodes whose first ranking puts two levels of one faulty set on top. The repeated repair sets are the visits whose component set equals that of an earlier visit. The deviations count the plans that an observation against their expectation ended, the column `surprises`. The last measure is the share of first plans that test first, read from the column `first_plan_mode`.
- **Formula.** Level accuracy `= mean top1_e` over the episodes so marked, and repeated sets `= the visits whose component set equals that of an earlier visit`.
- **Inputs.** `trajectory.json`, the `visit` of the `work_order_created` entries of `hidden/log.json`, and `plans.json`.
- **Computed in.** `first_ranking_shares_faulty_set`, `repeated_repair_sets`, `add_derived_fields`, `switch_of` and `table_a13_a17` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py`, `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_selection.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/ablation_tables.py`. Tests in `test_tables_from_runs.py`.
- **Sanity relations.** A13_A17 reads the CausalDiagnosis agent with the CausalDiagnosis MCP and its switches alone. The baseline policies and the configurations without the CausalDiagnosis MCP appear in no cell of it, while the fields exist for other episodes as well.
- **Gaming.** Not objectives.
- **Hypothesis.** Descriptive, with Holm over the differences of pass of the switches against the full agent.
- **Table.** A13_A17.

### Strata

- **Class.** Outcome.
- **Definition.** Flags that split a configuration. One marks a deterioration transition in the episode. One marks a separating test that cost more than the remaining budget at some answer whose `read_first` is true. The others are the strata of the two diagnosability labels.
- **Formula.** `deterioration_e = 1[the log holds a scheduled_drop or hazard_drop]`, `unaffordable_e = 1[some answer whose read_first is true names a separating test whose cost exceeds the budget left]`, and `sep(s_e)`, `sep_E18(s_e)` and `sep(s_e) = sep_E18(s_e) = 0` for the strata of the labels.
- **Inputs.** `hidden/log.json`, `trajectory.json`, `released.json` for the budget, and the cell `grade`.
- **Computed in.** `deterioration`, `separating_test_unaffordable`, `in_stratum` and `table_strata` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_fields.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agent_descriptive_tables.py`. Tests in `test_tables_from_runs.py`.
- **Sanity relations.** None beyond those of the labels. With the CausalDiagnosis MCP off and on, the strata of the labels contain the same initial combinations in every configuration of one rules key. The stratum of the unaffordable test needs answers of the CausalDiagnosis MCP and so contains episodes of the configurations with the CausalDiagnosis MCP only.
- **Gaming.** Not objectives.
- **Hypothesis.** Descriptive.
- **Table.** `tab:strata`, success and the declared share on 3 strata, and STRATA.

### The CausalDiagnosis MCP offline

- **Class.** Benefit.
- **Definition.** The CausalDiagnosis MCP offline, with no model and no agent, on the test combinations at every entry offset. An episode opens with the prehistory recordings of the block of the initial combination, followed by each count of live readings of `READINGS`, 0, 1, 2, 3, 5 and 10. A row contains the mean probability of the first candidate root cause, the share of episodes whose first candidate root cause is the initial combination, and the count of the others. It also contains the share of released live summaries inside the predicted interval of the true combination and, over the wrong repairs, the mean share of the cycles of the live window that the post repair check finds inconsistent. A reference fit of main effects alone stands beside the row, and so do the same rows without the prehistory, but for the row of no prehistory and no live reading, which contains no evidence and is not built. On the hydraulic rig the prehistory contains 5 recordings.
- **Formula.** Every figure of a row is a mean, a share or a count over the cells of initial combination and entry offset of one count of live readings, with the prehistory or without it. The share of inconsistent cycles is averaged over the wrong repairs.
- **Inputs.** `models.json`, the recordings and the sampler of initial states of the asset.
- **Computed in.** `table`, `score_episode` and `row_of` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py`. Tests in `test_offline_table.py`.
- **Sanity relations.** The fit of main effects alone is the reference that the CausalDiagnosis MCP must beat on the wrong count with a calibration gap no larger than its own. With the CausalDiagnosis MCP off and on, the table has no row without the CausalDiagnosis MCP, since it runs the CausalDiagnosis MCP offline.
- **Gaming.** It involves no agent.
- **Hypothesis.** Descriptive.
- **Table.** `tab:mcpoffline`.

### Calibration report of the CausalDiagnosis MCP

- **Class.** Benefit.
- **Definition.** Rollouts of readings from every recorded block, scored by the fits without that block. The report contains the Brier score of the component fault probabilities, the share whose top combination is the truth, the coverage of the predicted intervals and the calibration curve, per number of readings and ranking model.
- **Formula.** `BS = mean (p_j - y_j)^2` over the components and the rollouts.
- **Inputs.** `models.json` and the recordings.
- **Computed in.** `evaluate`, `brier` and `curve` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/calibration_report.py`, whose module text states that nothing from the report is used in code, tests or docs. Tests in `test_calibration_report.py`.
- **Sanity relations.** A calibrated model has an observed frequency equal to the mean prediction in every bin. With the CausalDiagnosis MCP off and on, the report runs the CausalDiagnosis MCP offline.
- **Gaming.** It involves no agent.
- **Hypothesis.** Descriptive, a report outside every table.
- **Table.** None.

### The two diagnosability labels and their agreement

- **Class.** Outcome.
- **Definition.** The label of an initial combination states whether the greedy baseline policy, after reading without charge the prehistory recordings of that combination, submits it at every level in more than half of its rollouts. The rollouts use seeds after the evaluation pool and run the static counterpart of the configuration, the rules key of the configuration with every hidden process off and the declared clock. A priced action that the budget cannot afford is refused and ends the rollout, and the share of correct rollouts is stored beside the label. The label depends on the initial combination and on the rules key of the configuration alone, which contains the prices, durations, production loss rates, horizon, budget, fault rule, step cap, failure card, sensor groups, asset and clock of the configuration and the text of the rule, every field that a rollout reads. The policy starts from a uniform prior over the combinations and reads no past work order, so the label is the same for every agent and the strata remain fixed where a configuration provides the repair history. The belief of the CausalDiagnosis MCP on such a configuration weighs the initial combination by the released past orders. The E18 grade comes from rollouts of the decision rule of the CausalDiagnosis MCP from the initial combination, and an initial combination is diagnosable within the budget under E18 when the first commitment is right in more than half of them. LABELS adds the agreement of the two labels.
- **Formula.** `sep(s) = 1[(1/n_roll) sum_i 1[d_i = s] > 1/2]` with `d_i` the first diagnosis of rollout `i`, `sep_E18(s) = 1[first_share(s) > 1/2]`, Cohen's kappa of the two labels, and Spearman's correlation of the two shares. The label runs 10 rollouts per initial combination, `ROLLOUTS` of `diagnosability_label.py`, and E18 as many, `ROLLOUTS` of `diagnosability_grade.py`.
- **Inputs.** The label table `labels.json` and the grade table `grades.json`, read through the cell `grade` of `scores.csv`.
- **Computed in.** `compute_table`, `label_of` and `start_grade` in `CausalDiagnosis/environment/reference_policies/diagnosability_label.py`, `grade` and `separable` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/diagnosability_grade.py`, and `table_labels` with `label_kappa` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/mcp_tables.py`. Tests in `test_diagnosability_label.py`, `test_diagnosability_grade.py` and `test_tables_from_runs.py`.
- **Sanity relations.** The labels belong to the initial combination under the rules of its configuration, the same for every agent, policy and level of the CausalDiagnosis MCP in that configuration, so they never differ between the CausalDiagnosis MCP off and on.
- **Gaming.** They involve no agent.
- **Hypothesis.** The labels are inputs of the verifier and of the strata, and LABELS is descriptive.
- **Table.** `tab:strata`, and STRATA, LABELS and A8.

---

## Measures of the CausalDiagnosis MCP

The cards of this section are descriptive measures of what the CausalDiagnosis MCP adds, what it charges and what it risks. They are read beside the hypotheses and decide none of them. `CausalDiagnosis/evaluation/metrics/mcp_metrics.py` computes them from files that the servers and the verifier write, `transcript.jsonl`, `hidden/log.json`, `released.json`, `predictions.json` and `hidden/verdict.json`, with the cells `pass` and `block` of `scores.csv` and the fits of `models.json`. No measure reads `trajectory.json` or any other record of a runner, so none depends on what the runner of the measured agent chose to record, and none needs a rerun.

The calls of an episode are the call lines of the transcript in line order, the order in which the servers answered them. A call took a step of the environment exactly when its line says so, and the k-th such call belongs to the one entry of the log with step k. What a call did is read from its log entry, an order for `work_order_created`, a reading for `read`, the diagnosis for `submit_diagnosis` and a refusal for `refused`. An episode agrees when the calls that took a step are as many as the steps of the log and every step has exactly one entry. An episode that does not agree counts in the cell `unaligned` of its configuration and gives no measure that needs the match.

Every rate of a configuration pools the items of its episodes before the ratio, the visits for the advice, the plans for the predictions and the answers for the ranking. It comes with a 95% interval from a bootstrap that resamples the test combinations. The table MCP of `CausalDiagnosis/evaluation/result_tables/tables_from_runs/agents_table.py` contains every cell, and `tab:mcpvalue` prints a part of them. The symbols below join those above. `q` is an answer of `rank_root_causes`, `c_kq` its candidate root cause of rank `k`, and `P_k(q)` the components that the work orders of its candidate plan of rank `k` repair, the candidate plans following the ranks of the candidate root causes. A visit `v` is a run of work orders with no call between them that took a step and whose tool creates no order. It starts at the cycle `t_v` of its first order and contains the components `C(v)`. A refused order lies between two orders without splitting them, and a call that took no step never splits a run. A visit is a decision of the agent and not the visit that the environment books for its visit charge, since on a configuration whose orders close in their own call every order books a visit of its own. `q(d)` is the last answer of `rank_root_causes` that the CausalDiagnosis MCP recorded before the call of the decision `d`.

### Uptake of the advice

- **Class.** Benefit, the channel through which the advice can act, with no value of its own.
- **Definition.** On a configuration with the CausalDiagnosis MCP, the share of the advised visits that order the candidate plan of the first candidate root cause of the last answer before them, beside the share that order some candidate plan of that answer. An advised visit is a visit whose last answer contains a candidate plan. With them stand the share of the charged readings taken under the flag `read_first` that read its separating test or the head of its read batch, and whether the diagnosis equals the first candidate root cause of the last answer before it. The advice stands in a reply as the flag `read_first` with `separating_test` and `read_batch`, as the `branches` in rank order, and as the `candidates`. Where the field `advice` of a reply specifies a `rank`, the advised rank is the first candidate plan. A step of reading or of declaring specifies no rank.
- **Formula.** `U = sum_v 1[C(v) = P_1(q(v))] / n_adv` and `U_any = sum_v 1[C(v) = P_k(q(v)) for some k] / n_adv`, with `n_adv` the number of advised visits. `U_read = sum_r 1[read_first(q(r)) and g(r) in {test(q(r)), batch_1(q(r))}] / sum_r 1[read_first(q(r))]` over the charged readings `r` with their sensor group `g(r)`, where a charged reading is a reading with a charged price, or any reading on a configuration that prices nothing. `U_diag = 1[h_hat_e = c_1q]` for the last answer `q` before the write.
- **Inputs.** The call lines of `transcript.jsonl` in their order with the answers of `rank_root_causes`, whose `branches`, `read_first`, `separating_test`, `read_batch` and `candidates` give the advice. From `hidden/log.json` come the `work_order_created` entries with their `component`, the `read` entries with their `sensor_group`, and the `submit_diagnosis` entry.
- **Computed in.** `calls`, `decisions`, `visits`, `advice_measures`, `read_uptake` and `diagnosis_uptake` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`, which `add_mcp_fields` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/replayed_fields.py` calls once per episode, with `is_paid` of `CausalDiagnosis/evaluation/metrics/realized_read_elimination.py`. The cells are `uptake`, `uptake_any`, `read_uptake` and `diagnosis_uptake` of `row_cells`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** A configuration without the CausalDiagnosis MCP has no answer and no value.
- **Gaming.** No score and no training signal reads it, and no record of the agent is used in the measure. A high uptake is no merit by itself, so the harm of the advice reads it with the outcome.
- **Hypothesis.** Descriptive. A null result on H2 with a low uptake reads as advice not taken, and with a high uptake as advice taken that did not help.
- **Table.** `tab:mcpvalue`, column Uptake, and MCP.

### Harm of the advice

- **Class.** Downside.
- **Definition.** Among the advised visits that order the candidate plan of the first candidate root cause, the share whose plan differs from the faulty set of the hidden combination at the first order of the visit. Beside it stands the share of the other advised visits that order that faulty set. The pass of the episodes whose first advised visit orders the first candidate plan is set against the pass of those whose first advised visit orders something else.
- **Formula.** `O(v) = F(h(t_v))`, the faulty set of the hidden combination followed through the log to the first order of the visit. `harm = sum_v 1[C(v) = P_1(q(v)) and C(v) != O(v)] / sum_v 1[C(v) = P_1(q(v))]` and `good = sum_v 1[C(v) != P_1(q(v)) and C(v) = O(v)] / sum_v 1[C(v) != P_1(q(v))]` over the advised visits. `pass_follow` and `pass_deviate` are the mean of `r_e` over the episodes by their first advised visit.
- **Inputs.** The inputs of the uptake, and the hidden combination at the first order of each visit, followed from the field `hidden` of the `reset` entry through the entries before that order that move the combination.
- **Computed in.** `advice_measures` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py` with `_combination_after` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py`, the rule by which the verifier follows the combination. The cells are `harm`, `good_deviation`, `pass_follow` and `pass_deviate` of `row_cells`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** A configuration without the CausalDiagnosis MCP has no value. The split of pass by the first advised visit is observational, since an agent follows more often where the belief is concentrated and where a pass comes easier.
- **Gaming.** No score reads it. The two pass rates describe the episodes and do not measure an effect of following.
- **Hypothesis.** Descriptive.
- **Table.** `tab:mcpvalue`, column Harm, and MCP.

### Calibration of the predicted probability of recovery

- **Class.** Benefit.
- **Definition.** For every plan an agent acted on, the predicted probability of recovery of the plan against whether the asset is healthy at the reading that verifies the plan. A configuration summarizes it by the Brier skill against a predictor at the base rate and by the mean signed error. The verifying reading is the observation of the replayed post repair check. Without a verdict of that check, it is the cycle at which the agent last closed a work order on a component of the plan after its visit, the first cycle in which the asset runs repaired. A plan with a component that no later order closes has no target.
- **Formula.** `Y_i = 1[h(t_i) = h_0]` at the cycle `t_i` of the verifying reading, `BS = (1/m) sum_i (R_i - Y_i)^2`, and `BSS = 1 - BS / (Ybar * (1 - Ybar))`, undefined when `Ybar` is zero or one. `bias = mean_i (R_i - Y_i)`. Each plan counts once, at the first visit that cites it, whatever its number of orders.
- **Inputs.** The `recovery` and `components` of the records of `predictions.json`, the `prediction_id` of the `work_order_created` entries, the `work_order_closed` entries of the agent, the hidden combination at a cycle from `hidden/log.json`, and the replayed answer of `verify_outcome`.
- **Computed in.** `prediction_measures`, `combination_at` and `verdicts` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`. The cells are `recovery_bss` and `recovery_bias` of `row_cells`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** A predictor that knew the hidden state at the verifying reading would have a Brier score of zero, and a constant predictor at the base rate has a skill of zero. A skill above zero means that the prediction contains information. A configuration without the CausalDiagnosis MCP has no prediction.
- **Gaming.** An agent cannot change the prediction, but it chooses the plans it acts on, so the value of a configuration mixes the predictor with that choice.
- **Hypothesis.** Descriptive.
- **Table.** `tab:mcpvalue`, the column of the Brier skill of the predicted probability of recovery, and MCP with the bias beside it.

### Error of the predicted cost

- **Class.** Benefit, the accuracy of the cost estimate that the CausalDiagnosis MCP offers.
- **Definition.** For every plan an agent acted on, the cost that the CausalDiagnosis MCP predicted for acting on it now against the cost that the episode then incurred from the first order of its visit until the asset is restored. The declared price of a wrong diagnosis is added when the diagnosis was wrong, and the error is scaled by the cost of the oracle on the same seed. A configuration reports the mean signed error, the mean absolute error, and the skill over a baseline that predicts the price and the downtime of the plan alone. The measure is defined on the configurations whose rules contain a price. On a configuration whose cost unit is the call, no plan has a cost error and the four cells are empty.
- **Formula.** `C_hat_i` is the `act_now` value of `cost_to_restored` of the candidate plan of the last answer before the visit whose plan has the identifier that the orders cite, the candidate plan of the written diagnosis when several have it. `C_i` adds the price or penalty of every log entry from the first order of the visit to the entry that restores the asset, the `end` entry aside. It also adds the production loss of every cycle `t` with `t_v <= t < t_r`, where `t_r` is the cycle of the restoring entry. The asset is restored when every component is at its healthy level and no order of the visit is still open, and a window in which the asset is never restored runs to the end of the episode. `C_d * wrong_e` is then added. `eps_i = (C_hat_i - C_i) / cost*_e`, `bias = mean_i eps_i` and `MAE = mean_i |eps_i|`. The baseline is `C0_i = price_i + downtime_cost_i` of the work orders of the plan with its own `MAE_0`, and the skill is `1 - MAE / MAE_0`.
- **Inputs.** The `branches` of the answers of `rank_root_causes` in `transcript.jsonl` with their `cost_to_restored` and `repair_plan`, and the entries of `hidden/log.json` with the `cost`, the `prediction_id` and the `id` of each and the closings. The measure also reads `downtime_by_cycle` of the `end` entry, the `misdiagnosis` price of the released rules, and `cost_oracle` and `diagnosis_all_correct` of `hidden/verdict.json`. The prediction is read from the field of the CausalDiagnosis MCP, and no number of a price table is pinned.
- **Computed in.** `prediction_measures`, `_branch_acted`, `restored_at`, `cost_to_restore` and `cost_unit_of` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`. The cells are `cost_bias`, `cost_mae`, `price_mae` and `cost_skill` of `row_cells`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** The window stops when the asset is restored, since the prediction covers the cost to a restored asset, so later readings, drops and their repairs fall outside it. The cost of a right plan does not move with a drop after the asset is restored, and the cost of a plan that does not repair every fault runs to the repair that restores the asset. A configuration without the CausalDiagnosis MCP has no value.
- **Gaming.** An agent cannot change the prediction, while its later actions are part of the realized cost up to the restoring entry. The scale by the cost of the oracle defines the error in every episode, since that cost is above zero wherever a fault is to be repaired.
- **Hypothesis.** Descriptive.
- **Table.** `tab:mcpvalue`, the columns of the bias, the absolute error and the skill over price of the predicted cost, and MCP with `price_mae` beside them.

### Coverage of the predicted intervals and the error of the predicted readings

- **Class.** Benefit.
- **Definition.** For every plan acted on whose predicted combination equals the hidden combination at its verifying reading, the share of the released summaries of that reading inside the predicted intervals of the plan, against the nominal level of the intervals. Beside it, over every verified plan, stands the error of the predicted means of the summaries against the error of repeating the last reading that the prediction saw, both scaled by the spread of the prediction.
- **Formula.** `cov = sum_i sum_f 1[low_if <= y_if <= high_if] / sum_i |F_i|` over the plans with `pred_i = h(t_i)`, and `skill_read = 1 - sum_i sum_f |y_if - mu_if| / sigma_if / sum_i sum_f |y_if - x_if| / sigma_if`. Here `y_if` is the summary `f` of the verifying reading, `mu_if` the predicted mean, `x_if` the last released value of `f` that the prediction saw, and `sigma_if = (high_if - mu_if) / z_if` with `z_if` the quantile that the record stores for summary `f`.
- **Inputs.** The `predicted` intervals and quantiles and the `observations` count of the records of `predictions.json`, the observations of `released.json`, and the replayed answer of `verify_outcome` with its `observation`, `observed_t` and `predicted_combination`.
- **Computed in.** `prediction_measures` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`. The cells are `coverage` and `readings_skill` of `row_cells`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** Split conformal prediction covers at least the nominal level over the calibration blocks when the predicted combination is the truth, the claim that `tab:mcpoffline` checks offline. A value far below that level in episodes flags a broken assumption. A prediction that knew the next reading would err by zero, so a skill above zero means that the predicted means beat repeating the current readings. A configuration without the CausalDiagnosis MCP has no value.
- **Gaming.** No agent changes the intervals or the means.
- **Hypothesis.** Descriptive, a check of a claim of the paper.
- **Table.** MCP, cells `coverage` and `readings_skill`.

### Error rates of the post repair check

- **Class.** Benefit and downside.
- **Definition.** For every plan acted on, the verdict of the post repair check at the verifying reading against whether the hidden combination at that reading equals the predicted combination of the plan. `verify_outcome` of the CausalDiagnosis MCP recomputes the verdict offline over the released evidence and the stored predictions. The false alarm rate is the share of the plans with a right prediction whose verdict is inconsistent. The miss rate is the share of the plans with a wrong prediction whose verdict is consistent.
- **Formula.** `FA = sum_i 1[not consistent_i and pred_i = h(t_i)] / sum_i 1[pred_i = h(t_i)]` and `MISS = sum_i 1[consistent_i and pred_i != h(t_i)] / sum_i 1[pred_i != h(t_i)]`.
- **Inputs.** `released.json`, `predictions.json`, the hidden combination at the reading, and the fits of `models.json` that leave out the block of the initial combination. The cell `yardstick` of the table MCP records the digest of that file.
- **Computed in.** `verdicts` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`, which calls `verify_outcome` of `CausalTools` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/tools/causal_tools.py` over the fits that `yardstick` in `CausalDiagnosis/evaluation/metrics/realized_read_elimination.py` loads, and `prediction_measures` with the cells `false_alarm` and `miss` of `row_cells`. Only the configurations with the CausalDiagnosis MCP replay the check. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** The check bounds its false alarm rate by its level `alpha` across held-out calibration blocks. The verifying reading of an episode comes from a block that the fits read, so the rate in episodes is a description to set beside `alpha` and no bound. The miss rate depends on the components that a wrong plan does not repair, since a small fault of some components moves no released summary. A configuration without the CausalDiagnosis MCP has no value.
- **Gaming.** The replay is the rule of the CausalDiagnosis MCP, which no agent changes, and it scores every verified plan whether or not the agent called the check.
- **Hypothesis.** Descriptive.
- **Table.** MCP, cells `false_alarm` and `miss`.

### Production loss while thinking

- **Class.** Expense and downside.
- **Definition.** On the configurations whose clock charges thinking, the production loss that the hidden record of costs charged in the cycles the clock spent on thinking, its share of the production loss of the episode, and the failures in those cycles. Beside them stands the paired difference of the production loss between an agent with the CausalDiagnosis MCP and the same agent without it on the same episodes, with an interval and no test.
- **Formula.** `PLT_e = sum_a sum_{t in think(a)} charge(t)`, with `think(a)` the cycles from `t_start(a) - think_cycles(a)` to `t_start(a) - 1` that the clock advanced before action `a`, never before the end of the action before it, and `charge(t)` the entry of cycle `t` in `downtime_by_cycle`. `share = sum_e PLT_e / sum_e downtime_total_e` over a configuration, `FLT_e` counts the failure entries in those cycles, and the difference is `mean (PLT_on - PLT_off)` over the matched episodes.
- **Inputs.** `think_cycles` and `t_start` of every entry of `hidden/log.json`, the `failure` entries, and `downtime_by_cycle` and `downtime_total` of the `end` entry.
- **Computed in.** `thinking_loss` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`, with the cells `think_loss_share` and `think_failures` of `row_cells`. `table_mcp` in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/mcp_tables.py` and `CausalDiagnosis/evaluation/result_tables/tables_from_runs/cell_statistics.py` pairs the cell `think_loss` against the agent without the CausalDiagnosis MCP through `paired`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** The value is zero on the configurations whose clock charges no thinking and under the thinking switch off, and it never exceeds the production loss of the episode. A positive difference between the CausalDiagnosis MCP on and off describes more production loss in thinking cycles with the CausalDiagnosis MCP, which its compute, a longer exchange with the model or a later repair can each produce.
- **Gaming.** The harness measures the wall time and the agent cannot report it, while a suspension or a sleep of the machine inflates it.
- **Hypothesis.** Descriptive.
- **Table.** `tab:mcpvalue`, last column, the difference on all six big layers, and MCP with the failures in thinking cycles.

### The ranking metrics in episodes

- **Class.** Benefit.
- **Definition.** Over every answer of `rank_root_causes` that the hidden log places, by the rule of the calibration of the ranking, metrics on the faulty set of the hidden combination at the moment of the answer. AC@1 is whether that set is the faulty set of the first candidate root cause, and AC@k whether it is that of some returned candidate root cause. The reciprocal rank is that of the first candidate root cause with that faulty set. Precision and recall compare the faulty set of the first candidate root cause with the true one, where a true negative earns nothing.
- **Formula.** `AC@1 = (1/m) sum_q 1[F(c_1q) = F(h(t_q))]`, `AC@k = (1/m) sum_q 1[F(h(t_q)) = F(c_kq) for some returned k]`, and `MRR = (1/m) sum_q 1/rank_q` with `1/rank_q = 0` when no candidate root cause has the true faulty set. Precision is `|F(c_1q) & F(h(t_q))| / |F(c_1q)|`, undefined for a healthy first candidate root cause, and recall is `|F(c_1q) & F(h(t_q))| / |F(h(t_q))|`, undefined when the truth is healthy. `h(t_q)` is the hidden combination at the moment of answer `q`.
- **Inputs.** The `candidates` and `belief_time` of the answers in `transcript.jsonl`, and `hidden/log.json` for the hidden combination at the moment of each answer.
- **Computed in.** `ranking_measures` and `answer_truths` in `CausalDiagnosis/evaluation/metrics/mcp_metrics.py`. The cells are `ac1`, `ack`, `mrr`, `precision` and `recall` of `row_cells`, with the answers that the log cannot place in the cell `unplaced`. Tests in `test_mcp_metrics.py`.
- **Sanity relations.** A configuration without the CausalDiagnosis MCP has no answer. On a configuration with a clock, an answer is scored against the combination before the cycle moves on. On a configuration without a clock, it counts the repairs of the orders before it.
- **Gaming.** An agent changes the answers through the evidence it is charged for and through when it asks. It also changes them through `top_k`, which sets how many candidate root causes an answer returns (5 when left out, with no maximum) and so raises AC@k and the reciprocal rank.
- **Hypothesis.** Descriptive.
- **Table.** MCP, cells `ac1`, `ack`, `mrr`, `precision` and `recall`.

---

## Tables that contain records and no metric

Every table of the tree is listed by a card above or appears in this list.

| Table | What it records |
| --- | --- |
| `tab:budget` | What each agent trained on, per fold. |
| `tab:hazard` | The training episodes of each trained variant whose hidden log reaches a test combination of the fold that its model holds out, which only the random hazard of deterioration can cause. |
| TRAINING_RECORD | The training record of every trained evaluation run, read from its `run.json`, the adapter of a fine-tuned run and the task block of a GEPA run. |
| RUNS | One row per run, or per line of the test set over its replicate runs, with the settings of its episodes. |
