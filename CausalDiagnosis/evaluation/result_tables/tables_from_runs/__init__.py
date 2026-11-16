"""One table per supporting ablation of the paper (A3, A7 to A10, A12, A13_A17 and SWITCHES,
CausalDiagnosis/docs/experiments.md) from the run folders of run_episodes/, as CSV and Markdown, plus the GEPA table with its online curve (GEPA_curve), the tables of the six rows of the agents
and the CausalDiagnosis MCP (AGENTS, AGENTS_pairs, A13_A17, CALLS, STRATA), the tables with the metrics of the AssetOpsBench paper
(JUDGE, JUDGE_AGREEMENT, EFFICIENCY, PLANS), the realized elimination of the paid reads (ELIM, ELIM_pairs), the measures of the CausalDiagnosis MCP (MCP, mcp_metrics.py), the reference policies
on the rows of the environment (POLICIES), the effect of each big layer of the environment (LAYERS_pairs), the tables of the trained
agents (TRAINING, TRAINING_pairs), LABELS, the agreement of the two separability labels, and RUNS,
one row per run folder or per line of the test set over its replicate runs, which leaderboard.py reads. The presets, the six rows, the
hypotheses and the training variants come from experiment_tables.py. The finished runs of the online loop of GEPA (gepa_prompt_optimization/, run.json mode gepa_online) fill GEPA and
GEPA_curve and no other table, RUNS included, and the runs that sample trajectories for fine tuning (run.json mode training) enter no
table. The agent tables need per episode quantities that scores.csv does not hold, so they are derived from the
episode files when the aggregate runs (add_derived_fields), and the realized elimination replays the filter of the
diagnostics server over the released evidence of every episode that ELIM and A13_A17 read (realized_read_elimination.py,
add_elimination_fields), which needs the models.json that CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/fit_models.py generates.

usage: python -m CausalDiagnosis.evaluation.result_tables.tables_from_runs [--runs CausalDiagnosis/results/runs] [--out CausalDiagnosis/results/tables]
       [--readme CausalDiagnosis/results/README.md] [--published CausalDiagnosis/results/published_runs] [--cluster "Columbia Insomnia"]
       [--resamples N] [--alpha A] [--seed S]

Every table is computed only from the run folders that exist; without any folder the Markdown states the design of each
table (what varies, the hypothesis, the metric, the reading) with the header row and empty cells, and every CSV holds its
header alone. Beside the tables the script writes the results README (--readme) for the results folder: the
models, the cluster, the dates and the presets of the runs, every table with a one sentence reading rule, one example
episode with its tool calls and verdict, and the commands that reproduce the runs; before any run exists it says so. From these tables
paper_tables/ writes the result tables beside the README, as a page and in LaTeX. Every interval is drawn
from a bootstrap that resamples recorded blocks, the start combinations, since the five episodes of a start share its block and are not
independent: the interval of a mean, of a proportion (proportion_cells) and of a difference between levels of a factor,
which are paired by episode (the same seed, scenario and every other factor). The p value of a paired difference comes from a
sign flip test over the blocks (stats.sign_flip_p), exact by convolution for a zero or one outcome whatever the number of blocks, and the
bootstrap (stats.cluster_bootstrap, one matrix product per statistic) gives the intervals alone. The
p values of a table's primary metric are corrected by the Holm step down (table_statistics.py). The run folders read are those of results_index.folders_read, so a run on the
memory backend of AssetOpsBench's servers, a run of the retired mode gepa and a model agent on a row of the
environment table over the rig's action server enter no table and no page, and INDEX.json names each with its
reason, as it names a run whose code.commit differs from the commit of the latest start, since runs of different commits
are never pooled, and the earlier of two finished runs of one condition that share a seed, a line played
again, of which the later start alone counts (results_index.pooling_refusals), both printed by main() as well. The
condition of the agent tables holds the clock mode, the tool surface, rig or aob, the abstention rule and the training variant of
run.json (design.TRAINING_VARIANT_KEY, a run.json without the key reads as the agent as released) and the serving setup of run.json,
the LoRA adapters of the server of the agent model (lora_adapters, 0 without the key), so a sensitivity line of the clock,
an abstention line, a trained agent and a line on a server with adapters are rows of their own, and AGENTS_pairs and A13_A17 read only the runs at the main level of
every setting (at_main_levels), so no ablation line of the grid and no trained agent pools into the hypotheses of the six rows or into
the full row of the agent's switches.
"""

from . import episode_files
from . import cell_statistics
from . import family_correction
from . import episode_selection
from . import ablation_tables
from . import episode_fields
from . import hypothesis_tables
from . import judge_tables
from . import training_tables
from . import agent_descriptive_tables
from . import agents_table
from . import comparison_split_tables
from . import layer_tables
from . import mcp_tables
from . import runs_table
from . import plan_schedule_tables
from . import table_declarations
from . import replayed_fields
from . import results_pages
from . import write_tables
from . import command_line

from .episode_files import DEFAULT_ASSET, _trajectory, episodes_of, get_profile, load_episodes, pooling_rules
from .cell_statistics import (
    Fraction, JUDGE_CRITERIA, JUDGE_OUTCOMES, OUTCOME_COUNTING, PAPER_PROTOCOL, PROPORTIONS, STATIC_OUTCOMES, _r,
    cluster_floor, completeness, counting_text, counts_as_failure, groups, paired_difference, registered,
    registered_of, row_status, run_configuration, summary, too_few_clusters, training, value)
from .family_correction import floor_cells, holm_joint, holm_over, reading, registered_floor, registered_size, untested
from .episode_selection import (
    ABLATION_PLACES, AGENTS_PRESET, CONDITION, EN4_PRESET, HEALTHY_CONTROL_PRESET, MAIN_LEVELS, PRESET_LEVELS,
    REFERENCE_LABELS, SWITCH_ABLATIONS, agent_episodes, at_main_settings, at_read, config_clock_mode, preset_config,
    read_of, switch_episodes, switch_of)
from .ablation_tables import (
    BINS, MISDIAGNOSIS_ABLATION, rankings, spearman, table_a12, table_misdiagnosis_price, verifier)
from .episode_fields import (
    DETERIORATION_EVENTS, GENERATION_TIME, THINKING_SPLIT, add_derived_fields, deterioration,
    first_ranking_shares_faulty_set, foreign_judges, judge_fields, merged_spans, pairwise_agreement, planned_steps,
    read_price, repair_declared, repeated_repair_sets, restored_clean, separating_test_unaffordable, thinking_cycles,
    thinking_split, tool_invocations, turn_record, unmatched_evidence)
from .hypothesis_tables import helps_on_every_measure, holds, holm_family
from .judge_tables import JUDGE_CONSTRUCTS, judge_kappa, table_judge_agreement
from .training_tables import (
    GEPA_ONLINE, GEPA_RESULT, TRAINING_SUBSETS, aliased, cross_fitted, episode_fold, fold_ids, fold_of, one_setup,
    table_training, table_training_pairs, training_pair_subsets, training_status)
from .agent_descriptive_tables import (
    EFFICIENCY_METRICS, FAILURE_CLASS_ORDER, FAILURE_END, FUNNEL_STEPS, HEALTHY_CONTROL_ROWS, STRATA, drop_groups,
    funnel_facts, funnel_row, in_stratum, table_failures, table_healthy_control, table_plans)
from .agents_table import (
    AGENT_METRICS, PASS_KS, REPLAY_COLUMNS, REPLAY_RATES, exact_match_floor, normalized_score, replay_cells,
    turns_summary)
from .comparison_split_tables import CONTRAST_MEASURES, fault_subsets, table_pairs_by_fault, table_system_contrast
from .layer_tables import POLICY_METRICS, table_policy_layers
from .mcp_tables import MCP_COLUMNS, cohens_kappa, label_kappa, table_mcp
from .runs_table import RUN_COLUMNS
from .plan_schedule_tables import table_parallel, table_parallel_pairs
from .table_declarations import tables, with_standard_errors
from .replayed_fields import elimination_fields
from .write_tables import aggregate, plan_judge_of_record
from .command_line import main
