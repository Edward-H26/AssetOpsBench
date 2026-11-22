"""The design of the experiment tables (CausalDiagnosis/docs/experiments.md): the one place that names
the presets, the rows, the hypotheses, the training comparisons and the places of the supporting ablations that the drivers read.
tables_from_runs/, leaderboard.py, paper_tables/, gepa_prompt_optimization/ and common.ROW_PRESETS take them from here, so a preset that is renamed
or regrouped changes in this file alone. The training variants, their run.json keys, the folds of the cross-fitting and the fold of a
combination come from training_variants/, their one definition."""
from fractions import Fraction
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import LADDER, layer_2_real_time as real_time  # noqa: F401
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import EPISODE_WALL_LIMIT_S  # noqa: F401  stated in both briefings, so the rule has one value
from CausalDiagnosis.experiments.table_definitions.training_variants import AS_RELEASED, TRAINING_FOLD_KEY, TRAINING_FOLDS, TRAINING_MODE, TRAINING_VARIANT_KEY, TRAINING_VARIANTS, VISITS, asset_of, fold_of
from CausalDiagnosis.experiments.table_definitions.training_variants.held_out_folds import fold_ids  # noqa: F401


# the agents and the CausalDiagnosis MCP play layer 0 alone, the base preset with no environment layer on
AGENTS_PRESET = "base"
# the environment is CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.LADDER, one big layer added per row in the order of the environment figure,
# and these are the names the tables print
LADDER_NAMES = ("Base", "+ priced reads and writes", "+ real-time clock", "+ advanced real-time clock", "+ repair rules", "+ deterioration and failure")
# training plays all six big layers with the random hazard and the failure card on
TRAINING_PRESET = "deterioration"
# the clock of record charges the wall seconds between two calls as thinking time; each rung of the ladder from the real-time clock
# on has a twin, its name with this suffix, that charges the tokens the agent's model generated between two calls in their place
# and differs in nothing else, a second condition on the same rows that enters no family: the layer study plays the twins of the
# rungs below all six big layers and the tables of the agents the twin of all six
TOKEN_SUFFIX = "_token_clock"
TOKEN_PRESETS = {f"{preset}{TOKEN_SUFFIX}": preset for preset in LADDER[LADDER.index(real_time.NAME):]}
# the runs in flight that a run of record plays beside it on every preset whose clock charges the wall seconds between calls,
# declared here and checked at the start (run_episodes/) and by the job (CausalDiagnosis/cluster_jobs/3_run/run_lines.sh); run.json records the number, which
# enters the group key of the tables (results_index.group_key), while a token twin charges no wall seconds and needs none
RUNS_IN_FLIGHT = 1
# the time limit of a serving job and of a grid job (#SBATCH -t of CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch and CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch), and the share
# of it that the optional pilot of cluster_run_checks.py projects each run against, the rest of a job kept for its start and for episodes
# slower than the pilot's; a run that passes one job continues in the next from its last finished episode (run_episodes/ --resume)
JOB_LIMIT_S = 12 * 3600
PROJECTION_SHARE = Fraction(3, 4)
# the harness rule of every agent beside the wall limit EPISODE_WALL_LIMIT_S of the runner's part of an attempt (1800 s, the value
# the upstream repository sets for one agent run, src/agent/opencode_agent/runner.py): one more attempt for an attempt the
# infrastructure broke; the run stops after that many harness errors in a row; and the timeout of the one health check
INFRASTRUCTURE_REPLAYS = 1
STOP_AFTER_FAILURES = 3
HEALTH_TIMEOUT_S = 10
# the small switches of layers 2, 3 and 5 that the supporting ablations set on all six big layers, each a preset of its own that
# differs from the preset of training by one setting, as (ablation, arm) under the ids of the layers: the
# thinking time of layer 2 off, the declared action durations of layer 3 off, and for layer 5 the deterioration off, the fixed
# schedule of level drops against the random hazard rates, and the failure card off; the two prices of thinking time, half and twice the price, join the
# thinking time off as the points of the time pressure curve, named by the price a second of thinking costs in cycles. For layer 1
# the misdiagnosis price at the budget of the asset (L1) has a table of its own, which describes it with no test
ARM_PRESETS = {"deterioration_thinking_off": ("L2", "the thinking time off"),
               "deterioration_thinking_price_double": ("L2", "the thinking time at twice its price"),
               "deterioration_thinking_price_half": ("L2", "the thinking time at half its price"),
               "deterioration_durations_off": ("L3", "the declared action durations off"),
               "deterioration_mode_none": ("L5", "the deterioration off"),
               "scheduled": ("L5", "the fixed schedule of level drops against the random hazard rates"),
               "deterioration_failure_off": ("L5", "the failure rule off"),
               "deterioration_misdiagnosis_budget": ("L1", "the misdiagnosis price at the budget")}
# the priced layer, the second rung of the ladder, where the arms of Ag7 play against the agent that follows the advice of
# the MCP on a row with prices
PRICED_PRESET = LADDER[1]
# the control on the start with no faulty component, played by the six rows of the main table: false alarms and unnecessary work
# orders when the monitor's flag is false, counts in a table of its own that enters no family and no other table
HEALTHY_CONTROL_PRESET = "healthy_control"
# the runs reported apart, and the presets that no row and no arm plays
OWN_TABLES = ("anchored", HEALTHY_CONTROL_PRESET)
UNREPORTED = {preset: "no row and no supporting ablation plays this preset" for preset in ("static", "timed", "static_zero_durations")}
# the server as a reader sees it, while the code keeps its package names
MCP = "CausalDiagnosis MCP"
ENVIRONMENT = "CausalDiagnosis environment"  # all six big layers on, the preset of training, as the tables of the agents name it
PLAN_EXECUTE, AGENT_AS_TOOL, OURS = "plan_execute", "agent_as_tool", "causal"  # the label prefixes of run_episodes.agent_label_of
# the six rows as (label prefix, with the MCP, name), each agent without and then with the MCP, under the names of the designs
# that the tables of the paper print
PE = (PLAN_EXECUTE, False, "Plan-and-Execute")
PE_MCP = (PLAN_EXECUTE, True, f"Plan-and-Execute + {MCP}")
AAT = (AGENT_AS_TOOL, False, "Agents-as-Tools")
AAT_MCP = (AGENT_AS_TOOL, True, f"Agents-as-Tools + {MCP}")
CD = (OURS, False, "CausalDiagnosis agent")
CD_MCP = (OURS, True, f"CausalDiagnosis agent + {MCP}")
AGENT_ROWS = (PE, PE_MCP, AAT, AAT_MCP, CD, CD_MCP)
FRAMEWORK_ROWS = (PE, AAT, CD)  # the three agents without the MCP, which play the framework's tools alone
MCP_ROWS = (PE_MCP, AAT_MCP, CD_MCP)
# the environment ladder plays the three agents with the MCP, the level with the CausalDiagnosis MCP, so that every rung serves the tools its
# layers need: its first rung is their lines of the main table and its last their lines of the part environment
LADDER_ROWS = MCP_ROWS
TRAINING_ROWS = MCP_ROWS  # training plays the three agents with it
# the baselines of the training table, the three agents without the MCP and without training on the preset of training, on the same
# test episodes and in the order of TRAINING_ROWS: the reference lines without the MCP on all six big layers, which h1_top, h2_prime
# and the arms of the layers read too
TRAINING_BASELINES = FRAMEWORK_ROWS
HEALTHY_CONTROL_ROWS = AGENT_ROWS
# the lines of every design in the two tables of the agents, as (the name after the design, with the MCP, preset): the design on the
# base layer, then with the MCP, with the environment and with both, so the lines read the runs of the agents table, of the reference
# lines without the MCP on all six big layers and of the rows without training of the training table
AGENT_LINES = (("", False, AGENTS_PRESET), (f"+ {MCP}", True, AGENTS_PRESET), (f"+ {ENVIRONMENT}", False, TRAINING_PRESET),
               (f"+ {MCP} and environment", True, TRAINING_PRESET))
# the stable id of a row in the registered analysis file: the label prefix, and +mcp for a row with the MCP
MCP_SUFFIX = "+mcp"


def row_id(row):
    return f"{row[0]}{MCP_SUFFIX if row[1] else ''}"


ROWS_BY_ID = {row_id(row): row for row in AGENT_ROWS}
# the supporting ablations of the paper, each as the preset it plays and the rows of AGENT_ROWS that play it, beside the claim it
# supports: the switches of the CausalDiagnosis agent with the MCP on layer 0 (H1), the parts of the MCP on layer 0 with the three
# agents with it (H2), the guidance level on layer 0 with the six rows (the agents table), the price table and the small switches of
# layers 2, 3 and 5 on all six big layers with the three agents without the MCP (FRAMEWORK_ROWS), where an arm of a small
# switch plays its own preset of ARM_PRESETS against the preset of training, and the scoring method on layer 0, the verifier scope
# over the runs of the six rows and the static agent of Sc3, which plays no row
# the switches of Ag7 keep the read of their arm against the advice of the MCP, which on a row with prices is one read of
# the record and then the diagnosis of rank one, so they play the priced layer. Ag9 turns the schedule of candidate plans off on the top
# layer, the one layer of its registered pair. A place whose rows do not play its preset in the main table, the ladder, the
# reference lines without the MCP or the part environment gets the reference line of the rows as released from grid_lines.ablation()
AGENT_ABLATIONS = ("Ag1", "Ag2", "Ag3", "Ag10")
PRICED_ABLATIONS = ("Ag7",)
PARALLEL_ABLATION = "Ag9"
ABLATION_PLACES = {**{ablation: (AGENTS_PRESET, (CD_MCP,)) for ablation in AGENT_ABLATIONS},
                   **{ablation: (PRICED_PRESET, (CD_MCP,)) for ablation in PRICED_ABLATIONS}, PARALLEL_ABLATION: (TRAINING_PRESET, (CD_MCP,)),
                   "En5": (AGENTS_PRESET, MCP_ROWS), "En9": (AGENTS_PRESET, MCP_ROWS), "En7": (AGENTS_PRESET, AGENT_ROWS),
                   "En6": (TRAINING_PRESET, FRAMEWORK_ROWS), "L1": (TRAINING_PRESET, FRAMEWORK_ROWS), "L2": (TRAINING_PRESET, FRAMEWORK_ROWS),
                   "L3": (TRAINING_PRESET, FRAMEWORK_ROWS), "L5": (TRAINING_PRESET, FRAMEWORK_ROWS), "Sc3": (AGENTS_PRESET, ()), "En4": (AGENTS_PRESET, AGENT_ROWS)}
# the arm of that reference line, on the grid and on the leaderboard alike
REFERENCE_ARM = "the reference, the agent as released"


def played_cells():
    """The (preset, row) cells that a line of the main table, of the environment ladder, of the reference lines without the MCP or of
    the part environment plays already on the server of every line of grid_agents.txt, so that an ablation placed on one of them needs
    no reference line of its own and is compared with a reference on its own serving setup."""
    cells = {(AGENTS_PRESET, row) for row in AGENT_ROWS}
    cells |= {(preset, row) for preset in LADDER for row in LADDER_ROWS}
    return cells | {(TRAINING_PRESET, row) for row in (*TRAINING_ROWS, *TRAINING_BASELINES)}


# the hypotheses of the six rows as (name, row, reference): H1, the CausalDiagnosis agent does better than each AssetOpsBench agent,
# and H2, the MCP helps every agent. Each holds when the paired differences of HYPOTHESIS_MEASURES lie above zero and every Holm
# decision of the family of the 14 tests rejects.
HYPOTHESES = (("H1a", CD, PE), ("H1b", CD, AAT), ("H1c", CD_MCP, PE_MCP), ("H1d", CD_MCP, AAT_MCP),
              ("H2a", PE_MCP, PE), ("H2b", AAT_MCP, AAT), ("H2c", CD_MCP, CD))
HYPOTHESIS_MEASURES = ("diagnosis_all_correct", "pass")  # top-1 and pass
# H1 under this guidance level, the En7 runs, is reported as a robustness check with a Holm of its own
ROBUSTNESS_GUIDANCE_LEVEL = 3
# H2', the family h2_prime, primary with h1_top in one group of the registered file: on the top rung of the ladder, every agent
# with the MCP against itself without it, as released, so that the claim of H2 is also tested where every big layer is on. It holds
# on an agent when the differences of pass and of the reward (H2_PRIME_MEASURES) lie above zero and both Holm decisions reject. The
# rows with the MCP are the part environment of grid_agents.txt, on the server of the rows without it, the reference lines without
# the MCP on all six big layers.
H2_PRIME = tuple((f"H2'{chr(ord('a') + i)}", row, baseline) for i, (row, baseline) in enumerate(zip(TRAINING_ROWS, TRAINING_BASELINES)))
# the second measure of H2' is the verifier's reward in place of the cost over the oracle, since an agent that gives up pays little,
# while the reward orders two episodes by their outcome first
H2_PRIME_MEASURES = ("pass", "reward")
H2_PRIME_HELPS = {"pass": 1, "reward": 1}  # the sign of a difference on which the MCP helps
# the comparisons of H1 asked again where every big layer is on, the family h1_top, primary with h2_prime in one group of the registered
# file, on cells that exist: the rows without
# the MCP are the reference lines without it on all six big layers and the rows with it the part environment, both on the server of
# grid_agents.txt
H1_TOP_PRESET = TRAINING_PRESET
H1_TOP = tuple(hypothesis for hypothesis in HYPOTHESES if hypothesis[0].startswith("H1"))
# the contrast of the full environment against the base layer for each row with the MCP, a secondary family on the cells of the main
# table and of the part environment, both on the server of grid_agents.txt
FULL_ENVIRONMENT = tuple((f"F_{chr(ord('a') + i)}", row) for i, row in enumerate(MCP_ROWS))
# the pair of the schedule of candidate plans, the CausalDiagnosis agent with the MCP with the switch on against off on the top layer, a family
# of one test on the time to repair in declared cycles; the row with the switch off is ablation Ag9, whose label ends in
# the suffix of the switch
PARALLEL_PRESET = TRAINING_PRESET
PARALLEL_ROW = CD_MCP
PARALLEL_OFF_SUFFIX = "seq"
# the other rows of the main table at the level of the MCP of the pair, which the table of the pair sets beside it on the same preset
PARALLEL_CONTRAST_ROWS = tuple(row for row in AGENT_ROWS if row[1] == PARALLEL_ROW[1] and row != PARALLEL_ROW)
# the companions of the main table, which enter no family: ablation En9 with the arm that stands beside the main table, a plain
# classifier in place of the ranking of the MCP, and the control on the start with no faulty component
COMPANIONS = {"En9": "discriminant_urn", HEALTHY_CONTROL_PRESET: "the six rows on the start with no faulty component"}
# the reference policies of the environment, which need no model, as (agent label, name)
REFERENCE_POLICIES = (("oracle", "Oracle"), ("greedy", "Greedy reference"), ("random", "Random"), ("do_nothing", "Do nothing"),
                      ("declare", "Always declare"))
# the names the tables print for the training variants of training.TRAINING_VARIANTS, in the words of the paper
VARIANT_NAMES = {AS_RELEASED: "no training", "gepa": "GEPA", "gepa_score_only": "GEPA on the score alone", "ft": "fine-tuning",
                 "ft_gepa": "fine-tuning + GEPA", "ft_unfiltered": "fine-tuning on unfiltered trajectories"}
# the training hypotheses as (variant, reference variant, name): each trained variant against the agent as released, GEPA against GEPA
# on the score alone, which asks whether a gain comes from the verifier's card rather than the reflection model, and fine tuning
# against its control, fine-tuning on unfiltered trajectories, which asks whether a gain comes from the environment's signal, 30 tests over
# the three agents and the two measures
TRAINING_COMPARISONS = (("gepa", AS_RELEASED, "GEPA against no training"), ("gepa", "gepa_score_only", "GEPA against GEPA on the score alone"),
                        ("ft", AS_RELEASED, "fine-tuning against no training"), ("ft_gepa", AS_RELEASED, "fine-tuning + GEPA against no training"),
                        ("ft", "ft_unfiltered", "fine-tuning against fine-tuning on unfiltered trajectories"))
TRAINING_MEASURES = ("pass", "diagnosis_all_correct")
# under two-fold cross-fitting every test episode is played by a model that never saw its fault combination, so the training hypotheses
# decide on all of them, with the single fault and two fault starts and each fold reported beside the decision
# what each agent trained on (the budget table), from run.json of its trained evaluation runs, where the training pipeline records the
# adapter and the task block a run plays: each field as the variants whose runs carry it and the path of keys to
# it, the runs of one agent and model agreeing on every field, and the fields the budget table prints, in its order
TRAINING_RECORD_SOURCES = {
    "episodes_sampled": (("ft", "ft_unfiltered"), ("adapter", "training", "episodes_sampled")),
    "kept_by_filter": (("ft", "ft_unfiltered"), ("adapter", "training", "kept_by_filter")),
    "model_calls_as_examples": (("ft",), ("adapter", "training", "examples")),
    "examples_over_max_length": (("ft",), ("adapter", "training", "examples_over_max_length")),
    "optimizer_steps": (("ft",), ("adapter", "training", "optimizer_steps")),
    "control_trajectories": (("ft", "ft_unfiltered"), ("adapter", "training", "control_trajectories")),
    "control_model_calls_as_examples": (("ft_unfiltered",), ("adapter", "training", "examples")),
    "control_examples_over_max_length": (("ft_unfiltered",), ("adapter", "training", "examples_over_max_length")),
    "control_optimizer_steps": (("ft_unfiltered",), ("adapter", "training", "optimizer_steps")),
    "gepa_candidates": (("gepa",), ("task_block", "candidates_played")),
    "gepa_stream_episodes": (("gepa",), ("task_block", "stream_episodes")),
    "ft_gepa_candidates": (("ft_gepa",), ("task_block", "candidates_played")),
    "ft_gepa_stream_episodes": (("ft_gepa",), ("task_block", "stream_episodes")),
    "gepa_score_only_candidates": (("gepa_score_only",), ("task_block", "candidates_played")),
    "gepa_score_only_stream_episodes": (("gepa_score_only",), ("task_block", "stream_episodes")),
    # the episodes left out of the data of fine-tuning and of the stream of each loop of GEPA, since their hidden log reaches a single
    # fault of the fold their model holds out, which under repair-closed training only the hazard of layer 5 can cause
    VISITS: (("ft",), ("adapter", "training", VISITS)),
    f"control_{VISITS}": (("ft_unfiltered",), ("adapter", "training", VISITS)),
    f"gepa_{VISITS}": (("gepa",), ("task_block", VISITS)),
    f"gepa_score_only_{VISITS}": (("gepa_score_only",), ("task_block", VISITS)),
    f"ft_gepa_{VISITS}": (("ft_gepa",), ("task_block", VISITS)),
}
# the gate of rejection sampling fine tuning: per agent and fold the data manifest records whether the filter kept at least one
# episode per training start state of the model, after one more round of sampling when the first fell short, and run.json carries it
# under adapter.training.gate of a trained adapter and under adapter.gate of a marker run, the run of a fold that trained no adapter.
# On such a fold every variant that needs the adapter reads the runs of no training of that fold, and the tables mark its cells.
GATE_FIELDS = ("minimum", "training_states", "rounds_sampled", "episodes_sampled", "kept", "distinct_training_states", "trained")
GATE_PATHS = (("adapter", "training", "gate"), ("adapter", "gate"))
ADAPTER_VARIANTS = ("ft", "ft_gepa", "ft_unfiltered")
TRAINING_RECORD_FIELDS = (*TRAINING_RECORD_SOURCES, *(f"gate_{field}" for field in GATE_FIELDS))
BUDGET_FIELDS = ("gate_rounds_sampled", "gate_episodes_sampled", "gate_kept", "gate_distinct_training_states", "model_calls_as_examples",
                 "gepa_candidates")
# the training episodes of each trained variant left out since their hidden log reaches a single fault of the fold its model holds out, the field of
# TRAINING_RECORD per variant in the order of TRAINING_VARIANTS, which the table of these episodes prints beside the budget table:
# amended): the episodes left out of the data of fine-tuning and of its control and of the stream of the loop of GEPA on each model
VISIT_FIELDS = {"gepa": f"gepa_{VISITS}", "gepa_score_only": f"gepa_score_only_{VISITS}", "ft": VISITS, "ft_gepa": f"ft_gepa_{VISITS}", "ft_unfiltered": f"control_{VISITS}"}


def row_label(row, model):
    """The agent label of scores.csv that a row plays with a model."""
    return f"{row[0]}_{model}"


# the tables that precompute_grid_tables.py writes, which no upload ships and no list hashes (CausalDiagnosis/cluster_jobs/env.example.sh TABLES names the same)
GENERATED_TABLES = ("CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json", "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json", "CausalDiagnosis/datasets/hyd_1/data/labels.json")
