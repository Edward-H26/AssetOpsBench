from CausalDiagnosis.experiments.table_definitions.experiment_tables import H2_PRIME_MEASURES, HYPOTHESIS_MEASURES, TRAINING_RECORD_FIELDS  # noqa: E402
from . import cell_statistics, episode_selection, ablation_tables, episode_fields, hypothesis_tables, judge_tables, training_tables, agent_descriptive_tables, agents_table, comparison_split_tables, layer_tables, mcp_tables, runs_table, plan_schedule_tables


# the episodes a table reads: every episode, the GEPA loop included; every played episode, the trained ones included; or the
# played episodes of the agents as released, which every table reads unless it keeps the trained runs apart by a column of its own
ALL_EPISODES, PLAYED, UNTRAINED = "all", "played", "untrained"
# the shared steps that only some tables need: the replay of the filter for the realized elimination, and the post repair check
ELIMINATION, MCP_MEASURES = "elimination", "mcp"
# the calibration bins of A12: the file stem, the key of ctx its builder fills, and the columns
A12_BINS = ("A12_curves", "a12_curves",
            ["agent", "model", "config", "ranking_model", "readings", "bin_lo", "bin_hi", "predicted_mean", "observed_frequency", "n"])


class Table:
    """One table of record: what it shows (varies, hypothesis, metric, reading, rule), its columns and its builder, the episodes it
    reads, the shared steps it needs beyond the derived fields, and the second file it writes, if any."""

    def __init__(self, id, varies, hypothesis, metric, reading, columns, build, rule="", episodes=UNTRAINED, needs=(), companion=None):
        self.id, self.varies, self.hypothesis, self.metric, self.reading, self.columns, self.build = id, varies, hypothesis, metric, reading, columns, build
        self.rule = rule  # the one sentence reading rule of the results README
        self.episodes, self.needs, self.companion = episodes, tuple(needs), companion


# ----- the tables of the supporting ablations, GEPA, the six rows, the environment, training and the fleet, LABELS and RUNS -----
def with_standard_errors(columns):
    """The columns of a table with <stem>_se after every <stem>_high, the standard error from the same bootstrap as the interval."""
    out = []
    for column in columns:
        out.append(column)
        if column.endswith("_high"):
            out.append(f"{column[:-len('_high')]}_se")
    return out


# ----- the registry: one declaration function per table of record, in the order of the CSV files, the pages and INDEX.json -----
REGISTRY = []  # the declaration functions in that order, filled by @registered_table as this module loads


def registered_table(declare):
    """Adds a declaration function to the registry: the order of the decorated definitions below is the order of the tables."""
    REGISTRY.append(declare)
    return declare


def tables():
    """Every table of record, each with the standard error of every interval among its columns (with_standard_errors)."""
    registry = [declare() for declare in REGISTRY]
    for table in registry:
        table.columns = with_standard_errors(table.columns)
    return registry


def table_ids():
    """The ids of the tables of record in registry order."""
    return [table.id for table in tables()]


def table_named(table_id):
    """The table of record with this id, or a stop that lists the ids."""
    for table in tables():
        if table.id == table_id:
            return table
    raise SystemExit(f"no table {table_id}: the tables are {', '.join(table_ids())}")


@registered_table
def declare_static_answers_vs_gym():
    return Table("A3", "the mode of answering the same question, Sc3 on layer 0: statically from the question, the facts of the cycle and the failure modes in one model call (runs of --agent static, mode static in run.json), or executed in the gym with tools and recordings by the six rows on the preset of the agents table",
                 "static answers are graded correct more often than the outcome the verifier reads in the gym warrants",
                 "the judge pass rate of the static answers, where a static answer without a judge record, an empty answer among them, counts as not passed when its run names a judge, and the share whose parsed faulty set equals the recorded start, where an answer that the parser cannot read names no faulty component, against the recorded pass rate and the cost of the gym episodes, per model, with intervals from a bootstrap clustered by block",
                 "the first evidence for why a gym is needed, on the questions the benchmark already poses",
                 ["model", "static_n", "static_judge_pass", "static_judge_pass_low", "static_judge_pass_high", "static_faulty_set_correct", "static_faulty_set_correct_low", "static_faulty_set_correct_high", "gym_n", "gym_recorded_pass", "gym_recorded_pass_low", "gym_recorded_pass_high", "gym_cost"], ablation_tables.table_a3,
                 rule="Read the static judge pass rate, where a static answer without a judge record counts as not passed when its run names a judge, beside the share of static answers whose faulty set matches the recorded start, against the recorded pass rate of the gym episodes.")


@registered_table
def declare_verifier_scope():
    return Table("A7", "the verifier scope on the preset of the six rows, layer 0: writes only against all actions (verifier_scope of run.json), the latter from the runs of the six rows scored again by rescore_under_verifier_scope.py",
                 "scoring reads changes the ranking of agents without changing their outcomes",
                 "the pass rate of every agent under the two scopes, where under the scope all an episode counts only when it passes and its sensor reads per group stay within those of the greedy policy on the same start (the read_gate cell), and per model Spearman's rank correlation of the agents' pass rates between the scopes",
                 "the justification of scoring writes only",
                 ["agent", "model", "diagnostics_server", "n_writes", "pass_writes", "pass_writes_low", "pass_writes_high", "n_all", "pass_all", "pass_all_low", "pass_all_high", "rank_correlation", "agents_in_both_scopes"], ablation_tables.table_a7,
                 rule="Read the rank correlation of each model: near 1 means scoring reads leaves the ranking of agents as the write only scope has it.")


@registered_table
def declare_abstention_rule():
    return Table("A8", "the abstention rule of the CausalDiagnosis MCP: the fixed posterior cut (abstention_rule fixed_cut of run.json, with abstention_cut) against the decision rule E10 (abstention_rule decision, the default), per preset and level of the MCP, every other setting at its main level",
                 "the decision rule declares on inseparable episodes and commits on separable ones more often than the constant",
                 "abstention precision (correct declarations among declarations) and recall (correct declarations among the starts the label marks inseparable) with intervals from a bootstrap clustered by block, against the reference policy's separability (label separability, the separable cell) and against the E18 grade of the MCP (label e18)",
                 "whether the threshold should follow prices",
                 [*ablation_tables.A8_KEY, "label", "n", "unscored", "abstentions", "inseparable_starts", "precision", "precision_low", "precision_high", "recall", "recall_low", "recall_high", "surface"], ablation_tables.table_a8,
                 rule="Read precision and recall of the declarations against the grade; the better rule declares on inseparable starts and commits on separable ones.")


@registered_table
def declare_price_table():
    return Table("A9", "the price table on all six big layers, En6: the full tier priced far above the rest (full_tier_high) against the default table, for the three agents without the CausalDiagnosis MCP",
                 "conclusions on cost hold under the alternative tables",
                 "cost, normalized cost and pass per table, the paired difference of normalized cost and pass against default, Holm over the normalized cost differences",
                 "the sensitivity the preregistration promised", ablation_tables.A9_COLUMNS, ablation_tables.table_a9,
                 rule="Read whether the ordering of agents by normalized cost holds under every price table.")


@registered_table
def declare_guidance_level():
    return Table("A10", "the guidance level L1 and L3 against L2 on layer 0 with the six rows, En7 (guidance_level of run.json)",
                 "the level an agent needs to pass is a measured quantity and differs between agents",
                 "pass per level, the paired difference against L2, the level of the main runs, Holm over the pass differences",
                 "the information a prompt supplies", ablation_tables.A10_COLUMNS, ablation_tables.table_a10,
                 rule="Read pass per guidance level; the lowest level at which an agent's interval reaches the next level's pass is the information its prompt needs.")


@registered_table
def declare_ranking_calibration():
    return Table("A12", "the ranking model: the discriminants with the urn (ranking_model discriminant_urn) against the joint interventional likelihood of E1 and E2 (ranking_model joint), per preset, every other setting at its main level",
                 "the joint model is calibrated where the discriminants are not, and its top candidate does not fall with more readings",
                 "per number of readings and over all readings: the Brier score of the top candidate's posterior over the rank_root_causes calls with a bootstrap interval clustered by block, and the share of calls whose top candidate is the hidden combination at the moment of the call with such an interval; A12_curves.csv holds the calibration bins (ten posterior bins with their bounds, the mean posterior and the observed frequency)",
                 "the calibration of the CausalDiagnosis MCP",
                 ["agent", "model", "config", "ranking_model", "readings", "calls", "brier", "brier_low", "brier_high", "top_candidate_correct", "top1_low", "top1_high"], ablation_tables.table_a12,
                 rule="Read the Brier score against zero and the top candidate rate by readings; a rate that falls with more readings is the symptom of a misfit of the ranking model.",
                 companion=A12_BINS)


@registered_table
def declare_misdiagnosis_price():
    return Table("MISDIAGNOSIS_PRICE", "the misdiagnosis price of big layer 1 at the budget of the asset, L1, on all six big layers: its arm preset of design.ARM_PRESETS against the preset of training, for the three agents without the CausalDiagnosis MCP",
                 "none, since the table describes the agents when a wrong diagnosis costs the whole budget, and no test decides",
                 "per agent, model and preset Top-1, pass, the share of episodes whose diagnosis is wrong, the share that declared insufficient evidence, the cycles read before the first diagnosis, the cost without the misdiagnosis charge, on one scale for both presets, and the charge, each with an interval from a bootstrap that resamples the units of the starts",
                 "what the price of a wrong diagnosis changes in how the agents commit to a diagnosis",
                 ["agent", "model", "config", "n", *(cell for m in ablation_tables.MISDIAGNOSIS_MEASURES for cell in (m, f"{m}_low", f"{m}_high", f"{m}_n")), "surface",
                  "diagnostics_server"], ablation_tables.table_misdiagnosis_price,
                 rule="Read the shares of wrong diagnoses and of declarations and the cycles read before the diagnosis across the two presets, and the cost without the charge on its one scale.")


@registered_table
def declare_layer_options():
    return Table("SWITCHES", "the options of layers 2, 3 and 5 on all six big layers, each arm preset of design.ARM_PRESETS of those layers against the preset of training, for the three agents without the CausalDiagnosis MCP: the thinking time off (L2), the declared action durations off (L3), and the deterioration off, the fixed schedule of level drops against the random hazard rates and the failure rule off (L5)",
                 "each option of a layer changes the pass of the full environment, as the cumulative layer study ranks the layers",
                 "pass, top-1, normalized cost, detection delay and repair delay per preset, the paired differences of pass and top-1 against the preset of training on the same starts, matched across the clock modes of the two presets, Holm over the pass differences",
                 "what each option of a layer adds on the full environment", ablation_tables.SWITCHES_COLUMNS, ablation_tables.table_switches,
                 rule="Read the pass difference of each arm against the preset of training; an interval that excludes zero shows what the switch adds to the full environment.")


@registered_table
def declare_gepa_loop_runs():
    return Table("GEPA", "the online loop of GEPA on the preset of training (design.TRAINING_PRESET by default) under the feedback of the verifier and under the control that sends the score alone, one row per finished run of gepa_prompt_optimization.py",
                 "none, since the table states what each loop did, and the gain of its final block over no block is read in TRAINING_pairs from the evaluation runs, which play the test set once",
                 "per finished run of gepa_prompt_optimization.py (status finished in gepa_online.json), the feedback (the column card), the reflection model, the candidates that played on the stream, the stream episodes with every member's validation on the blocks counted, the stream episodes the loop left out since they reached a single fault of the fold its model holds out, and the episodes played until the final candidate was first chosen",
                 "what each loop spent and where its final block came from, the trainable interface of the environment for the task block",
                 ["run", "model", "config", "card", "reflection_model", "candidates_played", "stream_episodes", "left_out", "rollouts_to_best"], training_tables.table_gepa,
                 rule="Read the stream episodes and the episodes until the final candidate under each feedback, and TRAINING_pairs for what the final block brings.",
                 episodes=ALL_EPISODES)


@registered_table
def declare_gepa_loop_curve():
    return Table("GEPA_curve", "the round of the online loop of GEPA on the preset of training, under the feedback of the verifier and under the control that sends the score alone",
                 "the prompt in use scores higher on each new block with the feedback of the verifier than with the score alone, from the same first round on",
                 "per GEPA run, feedback (the column card) and round, the episodes that every candidate played in the rounds before it, validation counted, so none at the first round, the mean reward of the prompt in use on the block of the round, scored before GEPA reflects on it, and the episodes of that mean",
                 "whether the verifier's feedback teaches the prompt more than its score does, the environment's online training signal",
                 ["run", "model", "config", "card", "reflection_model", "round", "rollouts", "mean_reward", "n"], training_tables.table_gepa_curve,
                 rule="Read the mean reward against the rollouts spent, the feedback of the verifier against the control that sends the score alone from their common first round.",
                 episodes=ALL_EPISODES)


@registered_table
def declare_agents():
    return Table("AGENTS", "the six rows of design.AGENT_ROWS: Plan-and-Execute, Agents-as-Tools and the CausalDiagnosis agent, each without and with the CausalDiagnosis MCP (the agent label and diagnostics_server of scores.csv in gym mode), each clock mode, tool surface and training variant of a preset a row of its own",
                 "the CausalDiagnosis agent does better than each AssetOpsBench agent and the CausalDiagnosis MCP helps every agent, on top-1 and pass, which AGENTS_pairs decides",
                 "pass, top-1 accuracy (diagnosis_all_correct), success, which credits a correct declaration of insufficient evidence, the verifier's reward, the share of episodes whose submitted diagnosis is wrong (diagnosis_wrong), which no declaration and no episode without a diagnosis counts toward, the share of episodes that declared insufficient evidence (abstained), the share of episodes that cited an evidence identifier the released evidence does not hold (unmatched_evidence, the complement of provenance_ok), the model calls per episode (llm_calls), cost, normalized cost, the production loss within the cost (downtime_total), the time to work order and the time to repair in cycles over the episodes that have one (detection_delay and repair_delay) per condition (agent label, model, config, clock mode, tool surface, level of the MCP, guidance level, verifier scope, price table, ranking model, abstention rule and training variant), each with an interval from a bootstrap clustered by block, and the turns to a correct diagnosis as the restricted mean survival time over turns, censored at the step limit of the run (turns_rmst), with such an interval, and Cohen's kappa between the judge's majority pass under the five call protocol and the verified pass over the episodes that carry both (judge_kappa), with such an interval, empty when either rule puts every episode on one side, and pass^k of tau-bench for every k from one to the start states that the test set plays of a start combination (pass_k1 and on), the share of the k-subsets of the episodes of a combination in which every episode passes, averaged over the combinations with at least k episodes, with an interval from a bootstrap that resamples the combinations",
                 "the headline of the paper: the outcome of the six rows on which the hypotheses of AGENTS_pairs are decided",
                 [*episode_selection.CONDITION, "n", *agents_table.COUNTS, "pass", "pass_low", "pass_high", "pass_n", *agents_table.PASS_K_COLUMNS, "diagnosis_all_correct", "diagnosis_all_correct_low", "diagnosis_all_correct_high", "diagnosis_all_correct_n",
                  *[f"{m}{part}" for m in ("faulty_set_right", "restored_clean", "graded_top1", "graded_pass") for part in ("", "_low", "_high", "_n")],
                  "normalized_score", "normalized_score_n", "success", "success_low", "success_high", "success_n", "reward", "reward_low", "reward_high", "reward_n", "diagnosis_wrong", "diagnosis_wrong_low", "diagnosis_wrong_high", "diagnosis_wrong_n", "abstained", "abstained_low", "abstained_high", "abstained_n", "unmatched_evidence", "unmatched_evidence_low", "unmatched_evidence_high", "unmatched_evidence_n", "llm_calls", "llm_calls_low", "llm_calls_high", "llm_calls_n", "cost", "cost_low", "cost_high", "cost_n", "cost_normalized", "cost_normalized_low", "cost_normalized_high", "cost_normalized_n", "downtime_total", "downtime_total_low", "downtime_total_high", "downtime_total_n", "turns_rmst", "turns_rmst_low", "turns_rmst_high", "turns_rmst_n", "detection_delay", "detection_delay_low", "detection_delay_high", "detection_delay_n", "repair_delay", "repair_delay_low", "repair_delay_high", "repair_delay_n", "judge_kappa", "judge_kappa_low", "judge_kappa_high", "judge_kappa_n",
                  "faulty_set_right_baseline_k", "faulty_set_right_baseline_n", "faulty_set_right_baseline", "faulty_set_right_floor", "faulty_set_right_at_floor",
                  *agents_table.REPLAY_COLUMNS], agents_table.table_agents,
                 rule="Read top-1 and pass of every row against the row it is compared with in AGENTS_pairs, and success, the wrong diagnoses, the declarations and the model calls beside them.",
                 episodes=PLAYED)


@registered_table
def declare_agents_hypotheses():
    return Table("AGENTS_pairs", "the hypotheses of the six rows on the preset of the agents table, layer 0: H1a to H1d pair the CausalDiagnosis agent against each AssetOpsBench agent, both without the CausalDiagnosis MCP and both with it, H2a to H2c every agent with the MCP against itself without it (design.HYPOTHESES), and the scoring pairs every two of the six rows",
                 "every hypothesis comparison holds, its paired differences of top-1 and pass both above zero with both Holm decisions rejecting, and scoring by the verified pass reverses the judge's order of at least one pair of the six rows",
                 "on the hypothesis rows the paired difference of top-1 accuracy and of pass, which decide, of success, which credits a correct declaration of insufficient evidence, and of the verifier's reward, the ratio of mean cost with a bootstrap interval clustered by block and the p value of the paired cost difference, and the paired difference of the turns to a correct diagnosis capped at the step limit, the difference of the two restricted mean survival times over turns (turns_rmst_diff), which no Holm decision covers, on the scoring rows the paired differences of the verified pass and of the judge's pass, and Holm within each of the two families of a model and tool surface with one decision per measure, the top-1 and pass tests of the hypotheses, 14 when every row has run, and apart from them the two pass tests of the scoring rows, with the number of tests in family_tests. holds says whether a hypothesis holds, empty while a cell it needs has no run. Episodes match on seed, scenario and every factor other than agent and diagnostics_server, among the runs at the main level of every setting (the clock mode, guidance level and verifier scope of the preset, the decision rule, the default prices, the default ranking model and the agent as released)",
                 "the decision on the hypotheses of the six rows, and on whether scoring by verified outcomes reorders the judge's ranking",
                 ["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", *episode_selection.PAIR_SETTINGS, "pairs", "clusters", "diagnosis_all_correct_diff", "diagnosis_all_correct_diff_low", "diagnosis_all_correct_diff_high", "diagnosis_all_correct_p", "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "success_diff", "success_diff_low", "success_diff_high", "success_p", "reward_diff", "reward_diff_low", "reward_diff_high", "reward_p", "judge_pass_diff", "judge_pass_diff_low", "judge_pass_diff_high", "judge_pass_p", "cost_ratio", "cost_ratio_low", "cost_ratio_high", "cost_ratio_p", "turns_rmst_diff", "turns_rmst_diff_low", "turns_rmst_diff_high", "turns_rmst_p", "diagnosis_all_correct_holm_p", "diagnosis_all_correct_reject", "pass_holm_p", "pass_reject", "judge_pass_holm_p", "judge_pass_reject", "reversed", "holds", "family_tests", "reference_pass_k", "reference_pass_n", "at_floor",
                  "status", "reading"], hypothesis_tables.table_agents_pairs,
                 rule="Read a hypothesis as holding when holds is true, its top-1 and pass differences both above zero with both Holm decisions rejecting, and on the scoring rows a reversed pair whose two differences both reject.",
                 episodes=PLAYED)


@registered_table
def declare_h1_at_guidance_level_3():
    return Table("H1_GUIDANCE", "the guidance level of the briefing at level 3 instead of the main level, the En7 runs of the six rows on the preset of the main table (guidance_level of run.json), for the comparisons of H1 in design.HYPOTHESES",
                 "the advantage of the CausalDiagnosis agent over each AssetOpsBench agent does not depend on how much guidance the briefing gives, a robustness check",
                 "the paired differences of top-1 and pass, matched on seed, scenario and every factor other than the agent and the level of the MCP, with intervals from a bootstrap clustered by block, sign flip p values over the blocks and a Holm of their own over these tests, apart from H1, with the floor of each baseline",
                 "whether H1 holds without the guidance of the main level",
                 ["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", "surface", "guidance_level", "status", "pairs", "clusters", "diagnosis_all_correct_diff", "diagnosis_all_correct_diff_low", "diagnosis_all_correct_diff_high", "diagnosis_all_correct_p", "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "diagnosis_all_correct_holm_p", "diagnosis_all_correct_reject", "pass_holm_p", "pass_reject", "family_tests", "holds", "reference_pass_k", "reference_pass_n", "at_floor"], hypothesis_tables.table_h1_guidance,
                 rule="Read each comparison against its row of AGENTS_pairs: a robustness check that agrees supports H1, and one that does not narrows it, without deciding it.")


@registered_table
def declare_h2_prime_full_environment():
    return Table("H2PRIME_pairs", "the CausalDiagnosis MCP on the top layer, all six big layers on: each agent of design.TRAINING_ROWS as released against itself without the MCP (design.TRAINING_BASELINES), the comparisons of design.H2_PRIME",
                 "the MCP helps every agent where every big layer of the environment is on, as H2 claims on layer 0",
                 "the paired differences of pass and of the verifier's reward (design.H2_PRIME_MEASURES), matched on seed, scenario and every factor other than the agent and the level of the MCP, with intervals from a bootstrap clustered by block, sign flip p values over the blocks and Holm run once with the family h1_top over the registered size of their group agents_environment on the same model and tool surface",
                 "whether the advantage of the MCP survives the layers that make the environment realistic",
                 ["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", "surface", "status", "pairs", "clusters",
                  *[f"{m}{part}" for m in H2_PRIME_MEASURES for part in ("_diff", "_diff_low", "_diff_high", "_p")],
                  *[f"{m}{part}" for m in H2_PRIME_MEASURES for part in ("_holm_p", "_reject")], "family_tests", "holds", "reading"], hypothesis_tables.table_h2_prime,
                 rule="Read an agent as helped when holds is true, its pass and reward differences above zero with both Holm decisions rejecting.")


@registered_table
def declare_judge_vs_verifier():
    return Table("H0", "the scoring of an episode by the AssetOpsBench judge's majority under the registered protocol against the verified pass, on the six rows of design.AGENT_ROWS and the comparisons of design.HYPOTHESES on the preset of the main table",
                 "the judge passes a different share of episodes than the verifier and decides the comparisons between rows differently",
                 "per row the mean of the judge's pass minus the verified pass over its episodes, and per comparison the judge's paired difference minus the verified one over the matched episodes, each with an interval from a bootstrap clustered by block and a sign flip p value over the blocks, Holm over the row and comparison tests in one family, an episode without a judge record counted as a failure of the judge, a test unrun on a row that misses a planned episode or is refused, on an episode judged under another protocol or without any judge record, with p equal to one and counted in the family, every other test with its computed p value, and beside each test, deciding nothing, Cohen's kappa, the discordance and the false pass and false fail rates of a row, and the two paired differences and the reversed flag of a comparison",
                 "how far the judge that scores the published benchmark agrees with what the agent did",
                 ["test", "name", "agent", "agent_server", "reference", "reference_server", "model", "config", *episode_selection.PAIR_SETTINGS, "n", "clusters", "nonzero_starts", "statistic", "statistic_low", "statistic_high", "h0_p", "status", "h0_holm_p", "h0_reject", "family_tests", "judge_kappa", "judge_kappa_low", "judge_kappa_high", "judge_kappa_n", "discordance", "false_pass", "false_fail", "judge_diff", "verified_diff", "reversed"], judge_tables.table_h0,
                 rule="Read a test as rejecting when h0_reject is true, the sign of the statistic saying whether the judge overstates or understates the verified pass, a test that does not reject as inconclusive and never as agreement, and the kappa and rates beside it as description.")


@registered_table
def declare_agent_switches():
    return Table("A13_A17", "the switches of the CausalDiagnosis agent, each turned alone on the same episodes and read from the label suffixes of CausalLoopOptions.label(): candidate troubleshooting plans off (Ag1, nobranch), plans for the top candidate only (Ag2, single), plan length one (Ag3, len1), and the read the board advises, by the discriminating test of E10 (Ag7, e10) or by one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray, not their algorithm (Ag7, ec2), while the model decides every step in each",
                 "every switch lowers pass or raises the model calls per episode, and no switch improves both over the full agent",
                 "per switch with intervals, and tool surface at the main level of every setting as in AGENTS_pairs, pass, unnecessary replacements, model calls, thinking cycles (the sum of think_cycles over the hidden log), deviations, the share of test first plans, cost, the price of the reads (test_cost), top-1 accuracy on the starts whose first ranking puts two levels of one faulty set on top (level_accuracy), the repair sets ordered again and the realized elimination per read, and the paired difference of pass and model calls against the full agent with the agent label left out of the pairing key, Holm over the pass differences",
                 "what each element of the agent is worth on its own",
                 ["agent", "switch", "model", "config", "surface", "n", *[f"{m}{part}" for m in ablation_tables.A13_METRICS for part in ("", "_low", "_high", "_n")], "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "llm_calls_diff", "llm_calls_diff_low", "llm_calls_diff_high", "llm_calls_p", "status", "pairs", "clusters", "holm_p", "reject"], ablation_tables.table_a13_a17,
                 rule="Read each switch row against the full row: a lower pass or more model calls is the price of the element the switch removes, and on the Ag7 rows the top-1 accuracy on level starts against the price of the reads.",
                 needs=(ELIMINATION,))


@registered_table
def declare_model_calls():
    return Table("CALLS", "the agent and the clock mode (the agent label, diagnostics_server and clock_mode of scores.csv), the tool surface and the other settings of the condition held fixed in a row",
                 "the CausalDiagnosis agent calls the model less often per episode and is charged fewer cycles for thinking than both rows of the agent that decides every step",
                 "model calls per episode, cycles charged for thinking (the sum of think_cycles over the hidden log, nonzero in the generation_time mode only) and tokens per episode, input and output together, with bootstrap intervals clustered by block, the paired difference of the first two against Agents-as-Tools with the CausalDiagnosis MCP in the same clock mode and, under the prefix vs_bare_, against Agent-As-Tool without the MCP, Holm over the model call differences against the row with the MCP",
                 "whether planning over candidate troubleshooting plans saves the thinking time that a degrading asset charges as downtime",
                 ["agent", "model", "config", "diagnostics_server", "clock_mode", "surface", *episode_selection.SETTINGS, "n", "llm_calls", "llm_calls_low", "llm_calls_high", "llm_calls_n", "thinking_cycles", "thinking_cycles_low", "thinking_cycles_high", "thinking_cycles_n", "tokens", "tokens_low", "tokens_high", "tokens_n", "llm_calls_diff", "llm_calls_diff_low", "llm_calls_diff_high", "llm_calls_p", "thinking_cycles_diff", "thinking_cycles_diff_low", "thinking_cycles_diff_high", "thinking_cycles_p", "status", "pairs", "clusters", "holm_p", "reject", *[f"vs_bare_{c}" for c in ("llm_calls_diff", "llm_calls_diff_low", "llm_calls_diff_high", "llm_calls_p", "thinking_cycles_diff", "thinking_cycles_diff_low", "thinking_cycles_diff_high", "thinking_cycles_p", "pairs", "clusters")]], agent_descriptive_tables.table_calls,
                 rule="Read the model calls and the thinking cycles of every agent against the Agents-as-Tools row with the MCP in the same clock mode.",
                 episodes=PLAYED)


@registered_table
def declare_thinking_time():
    return Table("THINKING", "the agent, the preset and the training variant among the conditions played in the generation_time mode, the only mode that charges the time between calls",
                 "none, since the table shows what the charged thinking time is made of, the model's generation, a sleep of the machine or a step of the wall clock, and the rest, serving, tools and the harness",
                 "the cycles charged for thinking and the seconds charged, which the clock measures on the monotonic clock after the credits of the failed model attempts, with, of these, the wall seconds of the model calls inside the charged span (generation_seconds, from started_at and ended_at of llm_calls.json), the wall gap minus the monotonic gap (wall_minus_monotonic_seconds, which the monotonic clock leaves out when the machine sleeps or the wall clock is set, and which cannot see a stopped process, since both clocks run while a process is stopped) and what remains (rest_seconds), each the mean over the episodes with an interval from a bootstrap clustered by block, the episodes played before the split was recorded (unsplit) and the episodes of a trained variant read from the runs of no training on a fold without an adapter (aliased)",
                 "how much of the thinking time that a degrading asset charges comes from the model, and whether a run lost time to a sleep or a clock step, which reads as a suspect run",
                 [*episode_selection.CONDITION, "n", "unsplit", "aliased", *[f"{m}{part}" for m in ("thinking_cycles", *episode_fields.THINKING_SPLIT) for part in ("", "_low", "_high", "_n")]], agent_descriptive_tables.table_thinking,
                 rule="Read the generation against the seconds charged: the rest is serving, tools and the harness, and a clock step above zero names a run that lost time to a sleep or a clock step.",
                 episodes=PLAYED)


@registered_table
def declare_outcomes_by_stratum():
    return Table("STRATA", "the stratum of the episode: an episode with a deterioration transition, scheduled or drawn by the hazard (the hidden log), a start separable and a start inseparable under the reference policy's separability (the separable cell), the same under the E18 grade of the CausalDiagnosis MCP (the e18 record of the grade cell), a start that both mark inseparable, where a declaration earns its credit, and an episode whose separating test exceeded the budget left at some rank_root_causes call that said read first (the trajectory)",
                 "the advantage of the CausalDiagnosis agent concentrates on the episodes the design targets",
                 "pass, success and the share of episodes that declared insufficient evidence (abstained) per condition (agent label, model, config, clock mode, tool surface, level of the MCP, guidance level, verifier scope, price table, ranking model, abstention rule and training variant) and stratum, each with an interval from a bootstrap clustered by block, and the number of episodes of the stratum",
                 "whether the mechanism of the agent, checks that catch a change and plans that act or repair as a test where the rule declares, is what produces its advantage",
                 [*episode_selection.CONDITION, "stratum", "n", "pass", "pass_low", "pass_high", "pass_n", "success", "success_low", "success_high", "success_n", "abstained", "abstained_low", "abstained_high", "abstained_n"], agent_descriptive_tables.table_strata,
                 rule="Read pass per stratum against the whole grid of AGENTS, the declarations on the separable strata against zero, and success and the declarations beside pass on the starts both labels mark inseparable, since a correct declaration counts as success there.",
                 episodes=PLAYED)


@registered_table
def declare_judge_scores():
    return Table("JUDGE", "the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the protocol and the judge model of the judge records of its episodes, with the metrics of Figure 5 of AssetOpsBench on the trajectories that the verifier scores",
                 "none, since the paper reports the three scores beside the verified outcome and the scoring pairs read the judge's pass in AGENTS_pairs",
                 "per condition, judge protocol and judge model, task completion, data retrieval accuracy and result verification, the first three criteria of the judge prompt, each as the mean over the episodes of the share of the planned calls that answered true, where a call without a parsed review and an episode without a record count as false, with an interval from a bootstrap clustered by block and the standard deviation over episodes (_std), and the share of the episodes whose majority of the planned calls answered true with such an interval (_majority), the episodes of the condition (n), the calls without a parsed review (failed_calls), the episodes without a record (episodes_without_record), Cohen's kappa of the judge's majority against four constructs of the verifier with the counts of each 2 by 2 table, the share of the judged episodes whose input was cut to the context of the served judge (judge_record_cut) and the share of the judged episodes that the judge passes while the asset is not recovered (judge_passed_unrecovered)",
                 "what answer scoring reads in the runs whose outcome the verifier scores",
                 [*episode_selection.CONDITION, "judge_protocol", "judge_model", "n", "failed_calls", "episodes_without_record", "declared", *[f"{c}{part}" for c in cell_statistics.JUDGE_CRITERIA for part in ("", "_low", "_high", "_n", "_std", "_majority", "_majority_low", "_majority_high", "_majority_n")],
                  *[f"{name}{part}" for c in judge_tables.JUDGE_CONSTRUCTS for name, part in ((f"kappa_{c}", ""), (f"kappa_{c}", "_low"), (f"kappa_{c}", "_high"),
                                                                                (c, "_both"), (c, "_judge_only"), (c, "_verifier_only"), (c, "_neither"))],
                  *[f"{m}{part}" for m in ("judge_record_cut", "judge_passed_unrecovered") for part in ("", "_low", "_high", "_n")]], judge_tables.table_judge,
                 rule="Read the mean of the three scores under the protocol paper with the metrics of Figure 5 of AssetOpsBench, and the majority columns against the values that paper prints.",
                 episodes=PLAYED)


@registered_table
def declare_judge_call_agreement():
    return Table("JUDGE_AGREEMENT", "the runs of the six rows together per model, config and settings at the main level of every setting, under a judge protocol with more than one call, with the metrics of Table 14 of AssetOpsBench",
                 "none, since the table checks how stable the judge is under repeated calls",
                 "per criterion the mean over the judged episodes of the mean pairwise agreement of the parsed calls of an episode, with an interval from a bootstrap clustered by block, and the uncertainty, one minus the agreement, whose formula Table 14 of AssetOpsBench states while it prints none for the agreement",
                 "whether the five calls of the judge agree enough for their mean to be read as a score",
                 ["model", "config", *episode_selection.PAIR_SETTINGS, "judge_protocol", "n", *[f"{c}{part}" for c in cell_statistics.JUDGE_CRITERIA for part in ("_agreement", "_agreement_low", "_agreement_high", "_agreement_n", "_uncertainty")]], judge_tables.table_judge_agreement,
                 rule="Read the agreement against 1 and the uncertainty against zero, as Table 14 of AssetOpsBench reads them.")


@registered_table
def declare_steps_and_runtime():
    return Table("EFFICIENCY", "the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the number of faulty components at the start, one or two, which take the place of the single-agent and multi-agent tasks of Table 3 of AssetOpsBench",
                 "none, since the paper reports the steps and the runtime as what a row costs to run",
                 "per condition and fault count, the episodes, the steps per episode (tool_invocations, the tool calls that the state files of the action or metered servers record and the calls to the CausalDiagnosis MCP together) and the runtime per episode in seconds (wall_s, the wall clock around the agent, which stops before the verifier and the judge), each with an interval from a bootstrap clustered by block and the standard deviation over episodes (_std)",
                 "what a row costs to run, as AssetOpsBench reports its frameworks",
                 [*episode_selection.CONDITION, "faults", "n", *[f"{m}{part}" for m in agent_descriptive_tables.EFFICIENCY_METRICS for part in ("", "_low", "_high", "_n", "_std")]], agent_descriptive_tables.table_efficiency,
                 rule="Read the steps and the runtime of a row on single-fault against two-fault starts, as Table 3 of AssetOpsBench reads single-agent against multi-agent tasks.",
                 episodes=PLAYED)


@registered_table
def declare_plan_lengths():
    return Table("PLANS", "the row of AGENTS that makes a plan, Plan-and-Execute and the CausalDiagnosis agent, each clock mode, tool surface and training variant of a preset a row of its own, with the metrics of Table 12 of AssetOpsBench",
                 "none, since the paper reports the length of the plans as a description of the rows",
                 "per condition the planned steps per episode (planned_steps, the steps of the plan of Plan-and-Execute and of all the plans of the CausalDiagnosis agent in an episode) with an interval from a bootstrap clustered by block, the standard deviation over episodes (_std), the smallest and largest value and the episodes that planned no step (_zero)",
                 "how far ahead a row plans, as AssetOpsBench reports its models",
                 [*episode_selection.CONDITION, "n", "planned_steps", "planned_steps_low", "planned_steps_high", "planned_steps_n", "planned_steps_std", "planned_steps_min", "planned_steps_max", "planned_steps_zero"], agent_descriptive_tables.table_plans,
                 rule="Read the planned steps of Plan-and-Execute and of the CausalDiagnosis agent, each with and without the MCP, as Table 12 of AssetOpsBench reads its models.",
                 episodes=PLAYED)


@registered_table
def declare_read_elimination():
    return Table("ELIM", "the row of AGENTS, each clock mode, tool surface and training variant of a preset a row of its own, and the reference policies of the action server, every row measured by one replay of the filter of the CausalDiagnosis MCP over the released evidence (realized_read_elimination.py), whether or not its run had the MCP",
                 "none, since the paper reports the realized elimination as a score beside the hypotheses, and its gap to the expected elimination as a check of the MCP's forecast",
                 "per condition the episodes, the paid reads and the reads with a forecast of the belief they met, the digest of the models.json the replay read (yardstick), and with intervals from a bootstrap clustered by block the realized elimination per paid read and per unit of the cost of E21 as the episode charged it, the effective number of hypotheses of the belief predicted to the cycle of a read minus that after its update, and over the reads that the MCP forecast on the belief they met the mean expected elimination, the mean realized elimination and their gap",
                 "how many hypotheses the reads an agent pays for rule out, a yardstick of the MCP's model of the asset and not of the agent's belief",
                 [*episode_selection.CONDITION, "n", "paid_reads", "calibrated_reads", "yardstick", *[f"{m}{part}" for m in mcp_tables.ELIM_METRICS for part in ("", "_low", "_high", "_n")]], mcp_tables.table_elim,
                 rule="Read the realized elimination per unit of cost of each agent against the reference policies, and the calibration gap of the runs with the MCP against zero.",
                 episodes=PLAYED, needs=(ELIMINATION,))


@registered_table
def declare_read_elimination_pairs():
    return Table("ELIM_pairs", "the comparisons of the hypotheses of AGENTS_pairs, on the realized elimination of the paid reads",
                 "none, since no hypothesis of the paper reads the realized elimination, so no Holm decision covers these differences",
                 "the paired difference of the realized elimination per unit of cost and per paid read with an interval from a bootstrap clustered by block and its p value, matched on seed, scenario and every factor other than agent and diagnostics_server among the runs at the main level of every setting, as in AGENTS_pairs",
                 "whether the MCP and the agent buy their pass with reads that rule out hypotheses",
                 ["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", *episode_selection.PAIR_SETTINGS, "status", "pairs", "clusters", *[f"{m}{part}" for m in mcp_tables.ELIM_PAIRED for part in ("_diff", "_diff_low", "_diff_high", "_p")]], mcp_tables.table_elim_pairs,
                 rule="Read the difference of the realized elimination per unit of cost against zero on every row.",
                 needs=(ELIMINATION,))


@registered_table
def declare_mcp_measures():
    return Table("MCP", "the agents with the CausalDiagnosis MCP, each on every preset, clock mode and tool surface, each condition a row of its own",
                 "none, since the measures of the MCP are descriptive",
                 "per condition, with intervals from a bootstrap that resamples the starts: the uptake of the advice over the visits (the first plan and any plan), the harm of the followed first plan and the right deviations, the pass when the first visit follows or deviates, the uptake of the advised reads and of the first candidate as the diagnosis, the Brier skill and the bias of the recovery rate prediction at the reading that verifies a plan, on a row with prices the bias and the mean absolute error of the cost prediction until the asset is restored, scaled by the cost of the oracle, beside the price and the downtime of the plan alone and the skill over it, the coverage of the predicted intervals, the error of the predicted readings against holding the current readings, the false alarms and misses of the post repair check recomputed offline, AC@1, AC@k, the mean reciprocal rank, the precision and the recall of the ranking at the truth of each answer, the production lost while thinking with its share of the production lost, and its paired difference against the agent without the MCP (mcp_metrics.py)",
                 "what the MCP adds, what it charges and what it risks, beside the hypotheses",
                 mcp_tables.MCP_COLUMNS, mcp_tables.table_mcp,
                 rule="Read the harm beside the uptake, the two prediction skills against zero, and the production lost while thinking against the same agent without the MCP.",
                 needs=(MCP_MEASURES,))


@registered_table
def declare_reference_policies():
    return Table("POLICIES", "the reference policy of design.REFERENCE_POLICIES (oracle, greedy reference, random, do nothing and always declare) on every preset, the rows of the environment of design.LADDER among them, each clock mode and tool surface a row of its own",
                 "the oracle is the best of the five reference policies, with a reward at least that of every other reference policy, and doing nothing scores below every policy that the verifier passes, on every start of every layer",
                 "per policy, preset, clock mode and tool surface, the pass, the verifier's success and the mean reward over the gym episodes at the main level of every setting, each with an interval from a bootstrap clustered by block",
                 "how far an agent stands from the policy that knows the hidden state and from the policies that need no model, row by row",
                 ["agent", "config", "clock_mode", "surface", "n", "pass", "pass_low", "pass_high", "pass_n", "success", "success_low", "success_high", "success_n", "reward", "reward_low", "reward_high", "reward_n"], layer_tables.table_policies,
                 rule="Read the reward of the oracle as that of the best of the five reference policies on each row, that of doing nothing below every policy that passes, and the rows of AGENTS beside them.")


@registered_table
def declare_layer_effects():
    return Table("LAYERS_pairs", "the big layer of the environment that a row of design.LADDER adds to the row below it, played by the three rows with the CausalDiagnosis MCP (design.LADDER_ROWS)",
                 "every big layer changes the pass, pooled over the three agents",
                 "per model, tool surface and layer, the paired difference of pass between the preset of the layer (config) and the preset below it (reference_config) for each of the three rows and pooled over their pairs (agent pooled), matched on seed, scenario and every factor other than the preset and the clock mode, with intervals from a bootstrap clustered by block, and Holm over the pooled tests of the five layers",
                 "whether each big layer of the environment measures something the layer below it does not",
                 ["model", "surface", "config", "reference_config", "agent", "status", "pairs", "clusters", "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "holm_p", "reject",
                  "pass_holm_p", "pass_reject", "family_tests", "reference_pass_k", "reference_pass_n", "at_floor", "pass_sign", "reading"], layer_tables.table_layers_pairs,
                 rule="Read the pooled difference of each layer against zero, decided by its Holm test, and the three rows beside it for agents that react differently.")


@registered_table
def declare_trained_agents():
    return Table("TRAINING", "the training variant of design.TRAINING_VARIANTS (as released, GEPA, GEPA on the score alone, fine tuning, fine tuning then GEPA, and fine-tuning on unfiltered trajectories) of the three rows with the CausalDiagnosis MCP (design.TRAINING_ROWS) on the preset of training, all six big layers (design.TRAINING_PRESET), from the training variant of run.json, and the three agents without the MCP and without training on that preset (design.TRAINING_BASELINES) as the baselines, the top layer on the same seeds",
                 "training in the environment raises pass and top-1 on fault combinations that the trained model never saw",
                 "per model, tool surface, row, training variant and subset, all for every episode of the test set, each played under cross-fitting by the model that holds out the fold of its combination, single and two for the single fault and the two fault starts, the pass and the top-1 accuracy, the three scores of the judge under the five call protocol with an episode without a judge record counted as false, the share of episodes that cited an evidence identifier the released evidence does not hold (unmatched_evidence) and Cohen's kappa between the judge's majority pass under the five call protocol and the verified pass (judge_kappa), each with an interval from a bootstrap clustered by block",
                 "whether the environment supports learning that carries over to fault combinations never trained on",
                 ["model", "config", "surface", "agent", "diagnostics_server", "training_variant", "subset", "n", "pass", "pass_low", "pass_high", "pass_n", "diagnosis_all_correct", "diagnosis_all_correct_low", "diagnosis_all_correct_high", "diagnosis_all_correct_n", "judged", "aliased",
                  "episodes_without_record", "failed_calls", "declared",
                  *[f"{c}{part}" for c in cell_statistics.JUDGE_CRITERIA for part in ("", "_low", "_high", "_n")], "unmatched_evidence", "unmatched_evidence_low", "unmatched_evidence_high", "unmatched_evidence_n", "judge_kappa", "judge_kappa_low", "judge_kappa_high", "judge_kappa_n"], training_tables.table_training,
                 rule="Read each trained variant against the row as released over every episode and on the single fault and two fault starts.",
                 episodes=PLAYED)


@registered_table
def declare_training_hypotheses():
    return Table("TRAINING_pairs", "the training comparisons of design.TRAINING_COMPARISONS per row with the CausalDiagnosis MCP: GEPA, fine tuning and fine tuning then GEPA against the agent as released, GEPA against GEPA on the score alone, and fine tuning against its control, fine-tuning on unfiltered trajectories, on as many episodes",
                 "each trained variant passes and diagnoses more often than the agent as released, GEPA more often than GEPA on the score alone, and fine tuning more often than its control, fine-tuning on unfiltered trajectories, on fault combinations the trained model never saw, so a gain comes from the environment's signal and not from more exposure to the task",
                 "the paired differences of pass and top-1, matched on seed, scenario and every factor, with intervals from a bootstrap clustered by block: over every episode of the test set (subset all) with their p values and Holm over the tests of a model and tool surface, 30 when every variant has run, with the number of tests in family_tests, and on the single fault and two fault starts (subsets single and two) and the episodes of each fold of the cross-fitting (fold_1 and fold_2) with no test",
                 "the decision on the hypotheses on training",
                 ["model", "config", "surface", "agent", "diagnostics_server", "training_variant", "reference_variant", "comparison", "subset", "status", "pairs", "clusters", "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "diagnosis_all_correct_diff", "diagnosis_all_correct_diff_low", "diagnosis_all_correct_diff_high", "diagnosis_all_correct_p", "pass_holm_p", "pass_reject", "diagnosis_all_correct_holm_p", "diagnosis_all_correct_reject", "family_tests", "reading"], training_tables.table_training_pairs,
                 rule="Read a difference over every episode above zero whose Holm decision rejects as a gain of the variant over its reference, and the two strata beside it.",
                 episodes=PLAYED)


@registered_table
def declare_training_budget():
    return Table("TRAINING_RECORD", "the row with the CausalDiagnosis MCP on the preset of training, from run.json of its trained evaluation runs, where the training pipeline records the adapter and the task block a run plays",
                 "none, since the table states what each agent trained on, the same budget for the three",
                 "per agent, model and fold of the cross-fitting the episodes sampled, the episodes the filter on the verified pass kept, the model calls used as examples, those the length cap left out and the optimizer steps of the filtered adapter, the trajectories and the same three counts of the control, fine-tuning on unfiltered trajectories, and the candidates and stream episodes of GEPA and of GEPA on the score alone on the agent without training and of GEPA on the fine tuned agent, and the training episodes each left out (design.TRAINING_RECORD_SOURCES)",
                 "whether the three agents trained on the same budget",
                 ["agent", "model", "fold", *TRAINING_RECORD_FIELDS], training_tables.table_training_record,
                 rule="Read the counts of the three agents side by side.",
                 episodes=PLAYED)


@registered_table
def declare_diagnosis_to_repair_funnel():
    return Table("FUNNEL", "the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table",
                 "none, since the table describes and tests nothing",
                 "of the episodes played in each row, the share and the count that had a diagnosis accepted, had the faulty set right, had Top-1, had an order accepted on every faulty component, placed no order on a healthy component and finished; the components that the first accepted diagnosis names as faulty against those faulty at its cycle, as means and as the shares that name more, as many and fewer; the orders on healthy components per healthy component of the start; the share that a failure ended before any order; the share whose cost ratio stands at the cap of the reward; and the refused diagnoses by reason, the episodes finished by the harness, the episodes that the runner's own limit ended, the cut replies, the failed calls, the replays and the cuts at the wall limit, each read from the hidden log, the oracle and scores.csv",
                 "where each design drops out on the way from a diagnosis to a restored asset, and what the harness did to its episodes",
                 [*episode_selection.CONDITION, "n", "status", *agents_table.COUNTS, *agent_descriptive_tables.FUNNEL_COLUMNS], agent_descriptive_tables.table_funnel,
                 rule="Read each step against the one before it; a row with a missing episode prints incomplete, and its shares are over the episodes played.",
                 episodes=PLAYED)


@registered_table
def declare_failure_classes():
    return Table("FAILURES", "the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table",
                 "none, since the table describes and tests nothing",
                 "of the episodes played in each row, the share and the count of each failure class of the verifier in its order and of none, the class of a success and of a failure that no class describes, with an episode without a verdict in the unscored count alone, and beside them the closing claims of recovery and the share of them that the hidden state contradicts, a summary that claims the asset recovered while it ended off its healthy state, with its 95 percent interval from a bootstrap drawn by recorded block",
                 "where each design fails, by the taxonomy of the verifier, and how often its closing summary claims a recovery that the hidden state does not show",
                 [*episode_selection.CONDITION, "n", "status", *agents_table.COUNTS, *agent_descriptive_tables.FAILURE_COLUMNS], agent_descriptive_tables.table_failures,
                 rule="Read the classes of a row as one split of its scored episodes, and the share of contradicted claims as a share of claims, not of episodes.",
                 episodes=PLAYED)


@registered_table
def declare_design_and_mcp_parts():
    return Table("SYSTEM_CONTRAST", "the common designs of the main table, on its preset, each against the CausalDiagnosis agent with the MCP",
                 "none, since the table describes and tests nothing",
                 "per common design the paired difference of Top-1, pass and the exact match of the faulty set between the CausalDiagnosis agent with the MCP and the design without it, with its 95 percent interval and standard error, beside its two parts on the same matched episodes, the design part with the MCP on for both and the MCP part on the common design, which add up to it",
                 "how much of the difference between the CausalDiagnosis agent with the MCP and a common design comes from the design and how much from the MCP",
                 ["agent", "design", "model", "surface", "config",
                  *[f"{m}{cell}" for m in comparison_split_tables.CONTRAST_MEASURES for cell in ("_pairs", "_clusters", *(f"_{part}{s}" for part in ("total", "design", "mcp") for s in cell_statistics.INTERVAL))]],
                 comparison_split_tables.table_system_contrast,
                 rule="Read the total against the sum of its two parts; the table is no test and prints no p value.")


@registered_table
def declare_comparisons_by_fault():
    return Table("PAIRS_BY_FAULT", "the comparisons of the agents family on the preset of the main table, each on the starts in which a component is healthy and on the starts that hold a fault of a component and a faulty level of the asset file",
                 "none, since the table describes and tests nothing",
                 "the paired difference of Top-1 and pass with its 95 percent interval and standard error, its pairs and its clusters, the difference alone below the registered smallest number of clusters",
                 "in which fault situations of the asset a comparison of the agents family holds its difference",
                 ["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", *episode_selection.PAIR_SETTINGS, "subset", "component", "level",
                  *[f"{m}{cell}" for m in HYPOTHESIS_MEASURES for cell in (*(f"_diff{s}" for s in cell_statistics.INTERVAL), "_pairs", "_clusters")]], comparison_split_tables.table_pairs_by_fault,
                 rule="Read a row as the mean paired difference over the starts of its subset, which the test starts sample as exchangeable fault situations of the asset.")


@registered_table
def declare_layer_effects_on_policies():
    return Table("POLICY_LAYERS", "the reference policies on each layer after the first, against the layer below",
                 "none, since the table describes and tests nothing",
                 "per policy, tool surface and layer the paired difference of pass, success and reward between the layer and the layer below, matched as two layers of the cumulative layer study are matched, with its 95 percent interval, its standard error and its pairs",
                 "what each single layer of the environment does to the policies that need no model",
                 ["agent", "surface", "config", "reference_config", *[f"{m}{cell}" for m in layer_tables.POLICY_METRICS for cell in (*(f"_diff{s}" for s in cell_statistics.INTERVAL), "_pairs")]],
                 layer_tables.table_policy_layers,
                 rule="Read the effect of a layer on the policies beside the layer effects of the agents; the table is no test.")


@registered_table
def declare_outcomes_by_first_drop():
    return Table("FIRST_DROP", "the conditions of the agents tables on the top layer, each split at the median of the first deterioration transition over the starts it played, with the starts without one as a group of their own",
                 "none, since the table describes and tests nothing",
                 "per group the episodes, the median that splits them, and Top-1 and pass with their 95 percent intervals",
                 "whether an agent does worse where the asset deteriorates early",
                 [*episode_selection.CONDITION, "group", "median_first_drop", "n", *[f"{m}{part}" for m in HYPOTHESIS_MEASURES for part in ("", "_low", "_high", "_n")]],
                 agent_descriptive_tables.table_first_drop,
                 rule="Read the early group against the late group of each row; the starts without a drop are a group of their own.")


@registered_table
def declare_healthy_start_control():
    return Table("HEALTHY_CONTROL", "the row of the main table on the preset of the healthy control, the start with no faulty component",
                 "none, since the table counts and tests nothing",
                 "of the registered episodes of each row, how many accepted a diagnosis, how many raised a false alarm with the number of components falsely named, and how many placed a work order with the number of orders, each unnecessary on this start",
                 "how often an agent finds a fault or orders work where the monitor's flag is false",
                 ["agent", "agent_server", "model", "surface", "config", "n", *agents_table.COUNTS, "diagnosed", "false_alarm_episodes", "components_falsely_named",
                  "work_order_episodes", "work_orders"], agent_descriptive_tables.table_healthy_control,
                 rule="Read the counts of each row against its registered episodes.")


@registered_table
def declare_restored_clean():
    return Table("RESTORED", "the condition of the agents tables, every row of the main table, of the cumulative layer study and of the training table",
                 "none, since the table describes and tests nothing",
                 "the share of episodes restored with no unnecessary order, every component at its healthy level at the end and no order on a healthy component, as the mean with its interval and standard error and k of n, the share of pass beside it, and the share of those episodes whose diagnosis did not match the faulty set exactly",
                 "how often an agent repairs the asset without the exact diagnosis that pass asks for",
                 [*episode_selection.CONDITION, "n", "restored_clean", "restored_clean_low", "restored_clean_high", "restored_clean_n", "restored_clean_k", "pass", "pass_low",
                  "pass_high", "pass_n", "restored_clean_not_exact"], agent_descriptive_tables.table_restored,
                 rule="Read the share restored with no unnecessary order beside pass; the last column is the part of it that pass does not credit.",
                 episodes=PLAYED)


@registered_table
def declare_h1_full_environment():
    return Table("H1_TOP_pairs", "the comparisons of H1 on the preset of training, all six big layers, from cells that exist: the rows without the MCP of the top layer and the rows with it without training of the training table",
                 "the CausalDiagnosis agent does better than each common design on top-1 and pass where every big layer is on",
                 "the paired differences of top-1 and pass with their 95 percent intervals, standard errors and exact sign flip p values, Holm run once with the family h2_prime over the registered size of their group agents_environment on the same model and tool surface, a test not run entering at one, and the floor of each reference",
                 "the family h1_top, primary with h2_prime in the group agents_environment, which asks the question of the agent again on the full environment",
                 ["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", "surface", "status", "pairs", "clusters",
                  *[f"{m}{part}" for m in HYPOTHESIS_MEASURES for part in ("_diff", "_diff_low", "_diff_high", "_p", "_holm_p", "_reject")],
                  "family_tests", "reference_pass_k", "reference_pass_n", "at_floor", "holds", "reading"], hypothesis_tables.table_h1_top,
                 rule="Read the reading column: holds, reversed, does not hold, not run or uninformative.")


@registered_table
def declare_full_environment_vs_base():
    return Table("FULL_ENV_pairs", "the environment of each row with the MCP: the full environment, its cell without training in the training table, against the base layer, its cell of the main table, on the same test combinations, entry offsets and seeds",
                 "none registered as one sided: the difference is expected below zero and tested two-sided",
                 "the paired difference of pass with its 95 percent interval, standard error and exact two-sided sign flip p value, Holm over the registered size of the family full_environment, the floor of the base cell, and the sign of the difference",
                 "the layer study without a new run, the family full_environment",
                 ["test", "agent", "agent_server", "model", "surface", "config", "reference_config", "status", "pairs", "clusters", "pass_diff", "pass_diff_low",
                  "pass_diff_high", "pass_p", "pass_holm_p", "pass_reject", "family_tests", "reference_pass_k", "reference_pass_n", "at_floor",
                  "pass_sign", "reading"], layer_tables.table_full_environment,
                 rule="Read the reading column and the sign of the difference.")


@registered_table
def declare_candidate_plan_schedule():
    return Table("PARALLEL", "the schedule of candidate plans of the CausalDiagnosis agent with the CausalDiagnosis MCP on the preset of training: on, the agent as released, against off, ablation Ag9, and beside the pair the other rows of the main table with the CausalDiagnosis MCP on the same preset",
                 "none, since the table describes the rows of the pair and the designs beside it",
                 "per row over its registered episodes the mean and standard error of the time to repair in declared cycles, the time to repair with the thinking cycles, the cycles charged for thinking, the work orders, the downtime, pass, the cost over the oracle, the model calls and the calls of the ranking tool of the MCP, the batches of reads that the board advised to a plan the model wrote with the number the model kept unchanged and changed and the share kept with its interval, and on the row of another design the share of its gap to the row with the switch on that the switch explains, off less on over the design less on on the measure of the family parallel_time, where the design is behind",
                 "what the switch changes beside the registered measure, whether the model follows the advised batches, and how much of the distance to the other designs the switch accounts for",
                 ["agent", "model", "surface", "config", "switch", "n", "status", *[f"{m}{part}" for m in plan_schedule_tables.PARALLEL_METRICS for part in ("", "_low", "_high", "_n")],
                  *plan_schedule_tables.PARALLEL_BATCH_COLUMNS, "gap_explained"],
                 plan_schedule_tables.table_parallel, rule="Read the two rows of the pair side by side and each other design against the row with the switch on, and find the test in PARALLEL_pairs.")


@registered_table
def declare_candidate_plan_schedule_test():
    return Table("PARALLEL_pairs", "the schedule of candidate plans on against off, on the preset of training, matched by seed",
                 "the agent with the schedule of candidate plans on repairs sooner: the paired difference of the time to repair in declared cycles lies below zero, its test rejects, and both margins are met",
                 "the paired difference of the time to repair in declared cycles with its 95 percent interval, its exact p value and its Holm decision over the registered size of the family parallel_time, the paired differences of pass and of the cost over the oracle against their registered margins, and the mean difference over the pairs with and without a refuted repair, which no test reads",
                 "the family parallel_time",
                 ["agent", "reference", "model", "surface", "config", "status", "pairs", "clusters", "repair_declared_diff", "repair_declared_diff_low",
                  "repair_declared_diff_high", "repair_declared_p", "repair_declared_holm_p", "repair_declared_reject", "family_tests",
                  "pass_diff", "pass_diff_low", "pass_diff_high", "margin_pass", "margin_pass_met", "cost_normalized_diff", "cost_normalized_diff_low",
                  "cost_normalized_diff_high", "margin_cost", "margin_cost_met", "cost_pairs_without_cost", "refuted_pairs", "refuted_diff", "unrefuted_pairs",
                  "unrefuted_diff", "reading", "holds"], plan_schedule_tables.table_parallel_pairs,
                 rule="Read the reading and the two margins: the comparison holds when the reading is holds and both margins are met.")


@registered_table
def declare_diagnosability_label_agreement():
    return Table("LABELS", "the label of a start: the reference policy's separability, the label of the greedy reference (CausalDiagnosis/environment/reference_policies/diagnosability_label.py) that the verifier reads, against the E18 grade of the CausalDiagnosis MCP, per label table and ranking model",
                 "the two labels agree beyond chance",
                 "the two by two table over the starts that both labels grade, each start counted once, each label the majority of its rollouts, Cohen's kappa, undefined and empty when one label puts every start on one side, Spearman's rank correlation between the share of greedy rollouts that submit the start and the share of E18 rollouts whose first commitment names it, and beside it the same correlation with the share of E18 rollouts that ever commit",
                 "whether the MCP finds evidence that the reference policy misses, or the reverse, which the starts separable under one label only show",
                 list(mcp_tables.LABEL_COLUMNS), mcp_tables.table_labels,
                 rule="Read kappa against 1 for agreement between the labels, and the two one label cells for the starts on which they disagree. When kappa is empty, the rank correlation of the shares carries the comparison.")


@registered_table
def declare_run_folders():
    return Table("RUNS", "the run folder: one row per folder that this script reads outside the online loop of GEPA, whose runs enter the GEPA tables alone, and one row per line of the test set over the runs of its replicates, with the settings of its episodes, the set of episodes it played (starts, master_seed and replicate of run.json) and its start",
                 "none, since the table indexes the runs that the other tables pool",
                 "pass, pass with the read gate under the verifier scope all (scope_pass), the judge's pass, the mean of the cost over the cost of the oracle policy on the same seed, the verifier's success and the mean reward, each with an interval from a bootstrap clustered by block, and the median over the episodes with a delay of the larger of the detection and the repair delay",
                 "every number of the leaderboard that leaderboard.py writes traces to the run folders of one row",
                 runs_table.RUN_COLUMNS, runs_table.table_runs,
                 rule="Read one row per run folder, or per line of the test set over its replicate runs, which the leaderboard places in its tables by the agent label and the settings.",
                 episodes=PLAYED)
