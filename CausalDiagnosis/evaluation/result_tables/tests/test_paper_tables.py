"""The result tables of the paper (paper_tables/) in the order and layout of the result
tables of the paper, on synthetic run folders made with the helpers of test_tables_from_runs.py, whose values are made up here so that no
recorded value enters: the tables of the body and of the appendix under their labels in the paper, the row names and
the order, every cell checked by hand where the bootstrap leaves no doubt (a share of none or of every episode, a difference that is
the same on every start, the p value of a test whose every resample lies on one side of zero), the protocol of the judge, the offline
table of the MCP from offline_mcp.csv, a page without a run that holds no number, the LaTeX form against the page, and the two files
in the results index and in the rebuild of verify_results_folder.py. The aggregate tables with the metrics of AssetOpsBench that no main
table prints keep a test of their own."""
import json
import math
import re
import shutil
import statistics
from pathlib import Path

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS, RIG_SERVER
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ABLATION_PLACES, ARM_PRESETS, AAT, AAT_MCP, AS_RELEASED, CD, CD_MCP, FULL_ENVIRONMENT, H2_PRIME, HYPOTHESES, LADDER, LADDER_NAMES, LADDER_ROWS, H2_PRIME_MEASURES, PARALLEL_CONTRAST_ROWS, PE, PE_MCP, REFERENCE_POLICIES, AGENTS_PRESET, AGENT_LINES, AGENT_ROWS, TRAINING_BASELINES, TRAINING_COMPARISONS, TRAINING_FOLD_KEY, TRAINING_FOLDS, TRAINING_PRESET, TRAINING_ROWS, TRAINING_VARIANT_KEY, TRAINING_VARIANTS, VARIANT_NAMES, VISIT_FIELDS, VISITS
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from evaluation.scorers import llm_judge
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _csv, _logged, _row, _run, _timed
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.tests.test_episode_judge import RUBRIC_PASS, FakeJudge, episode_folder

MODEL = "granite_m"  # with an underscore, which LaTeX has to escape
SEEDS = range(5)
N_FAULTY = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}  # two single fault starts and three two fault starts
NAMES = tuple(name for _, _, name in AGENT_ROWS)
FOLDERS = dict(zip(AGENT_ROWS, ("pe", "pe_mcp", "aat", "aat_mcp", "cd", "cd_mcp")))  # the run folder of each of the six rows
PM, DELTA, K = "\N{PLUS-MINUS SIGN}", "\N{GREEK CAPITAL LETTER DELTA}", "\N{GREEK SMALL LETTER KAPPA}"
ALL_TRUE = {"task_completion": 5, "data_retrieval_accuracy": 5, "generalized_result_verification": 5, "agent_sequence_correct": 5}
# the tables of the page by their labels in the paper, in its order, the body and then the appendix, with the columns that label their lines
LABELS = {"agents": 1, "agentmeasures": 1, "hyp": 2, "ladder": 2, "training": 1, "efficiency": 1, "exactmatch": 1, "thinking": 2, "layers": 1, "misdiagnosis": 2,
          "fullenv": 2,
          "policies": 1, "policylayers": 2, "trainhyp": 2, "trainfolds": 2, "budget": 2, "hazard": 2, "mcptop": 2, "judgerows": 1, "judgecmp": 2,
          "judgeagree": 1, "strata": 1, "mcpoffline": 1, "mcpvalue": 2, "parallel": 1, "parallelpair": 1}
HEADINGS = ("Tables of the body of the paper, one per experiment, the metrics of the AssetOpsBench paper first and the verified outcome after them",
            "Tables of the appendix of the paper")
# the tables whose LaTeX form drops the interval of a column, with the columns that keep it, the difference of the row test of the judge
# and the last column of the table of the MCP; the other tables print every interval alike in both forms, but the counts of the offline
# table and the episodes behind the baseline of exact match, which the page alone names
KEPT_INTERVALS = {"thinking": (), "policies": (), "strata": (), "judgerows": (0,), "mcpvalue": (6,)}
# the tables whose cells give a mean and its standard error, which the LaTeX form sets on one line in the two tables of the agents and
# under the mean in smaller type in the others
ONE_LINE_TABLES = ("agents", "agentmeasures")
STACKED_SE_TABLES = ("ladder", "training", "exactmatch", "judgeagree", "parallel", "misdiagnosis")
FIRST_ONLY_TABLES = ("ladder", "policylayers", "trainhyp", "trainfolds", "budget", "hazard", "mcpvalue", "misdiagnosis")  # the first label of a group names it on the group's first line alone, as the paper prints them
STACKED_TABLES = ("trainhyp",)  # that name stands on two lines, the agent above its MCP
INDENTED_TABLES = ("agents", "agentmeasures", "training")  # the lines after the first of a group are indented under it, the lines of a design or the trained variants under their agent
PASS_K = main_tables.PASS_K_SHOWN  # the k of the pass^k column of the agents table
PROFILE = get_profile(DEFAULT_ASSET, engine=False)


def _combination(seed):
    """The start combination of a seed in the component order of the asset, one of its own per seed: a fault of the first component on
    the single fault starts and of the first two on the two fault starts, as N_FAULTY counts them."""
    first, second = (sorted(level for level in PROFILE.levels[c] if level != PROFILE.healthy[c]) for c in PROFILE.components[:2])
    levels = {**PROFILE.healthy, PROFILE.components[0]: first[seed % 2]}
    if N_FAULTY[seed] == 2:
        levels[PROFILE.components[1]] = second[seed // 4]
    return [levels[c] for c in PROFILE.components]


# the fault combination of each start as scores.csv holds it, a JSON list of component levels and the block of the start, and the fold of
# the cross-fitting it lies in under the fold rule of these tests, in place of training.fold_of, which reads the folds of the real
# combinations: fold 1 holds the single fault start 0 and the two fault starts 2 and 3, fold 2 the others
COMBINATIONS = {seed: json.dumps(_combination(seed)) for seed in SEEDS}
FOLD_OF_SEED = {0: 1, 1: 2, 2: 1, 3: 1, 4: 2}


@pytest.fixture(autouse=True)
def _two_folds(monkeypatch):
    folds = {tuple(json.loads(COMBINATIONS[seed])): fold for seed, fold in FOLD_OF_SEED.items()}
    monkeypatch.setattr(aggregate.training_tables, "fold_of", lambda combination, asset_id=None, split=None: folds[tuple(combination)])


@pytest.fixture(autouse=True)
def _offline_in_the_results_of_the_tests(request, read_no_offline_table_of_record, tmp_path, monkeypatch):
    """The results folder of a test is tmp_path / "results", where _aggregated writes the pages: paper_tables/ reads the offline
    table there, as it reads the one of the results folder of the fork, and _offline writes it there."""
    if not request.node.get_closest_marker("offline_file"):
        monkeypatch.setattr(main_tables.read_sources, "OFFLINE_PATH", tmp_path / "results" / main_tables.OFFLINE)


# ----- run folders -----

def _meta(preset=AGENTS_PRESET, **fields):
    return {"config_name": preset, "servers": list(METERED_SERVERS), "model": MODEL, "starts": "sampled", "master_seed": 42,
            "config": {"durations": {"read_per_cycle": 1}, "max_steps": 10}, **fields}


def _episode(seed, row, preset=AGENTS_PRESET, **fields):
    """A row of scores.csv of one of the six rows at the main level of every setting in the clock mode of its preset, each start a
    recorded block of its own."""
    cells = {"agent": f"{row[0]}_{MODEL}", "model": MODEL, "config": preset, "clock_mode": aggregate.config_clock_mode(preset),
             "diagnostics_server": row[1], "ranking_model": DEFAULT_RANKING_MODEL, "block": COMBINATIONS[seed], "n_faulty": N_FAULTY[seed], **fields}
    return _row(seed, **cells)


def _judge_record(path, trues, protocol="paper", failed=0):
    """A judge record with one rubric answer per call, trues holding how many calls answer true per criterion, and the last
    failed calls without a parsed review."""
    planned = judge.PROTOCOLS[protocol]["calls"]
    calls = []
    for c in range(planned):
        no_review = c >= planned - failed
        calls.append({"passed": False, "score": 0.0, "rationale": "", "details": {"raw": "x"} if no_review else {key: c < trues[key] for key in trues},
                      "error": "judge returned unparseable JSON" if no_review else None})
    (path / "hidden" / "judge.json").write_text(json.dumps({"format": 1, "protocol": protocol, "calls_planned": planned, "judge_model": "judge", "calls": calls}),
                                                encoding="utf-8")


def _judged(folder, trues=ALL_TRUE, seeds=SEEDS, **record):
    for s in seeds:
        _judge_record(folder / f"seed{s:06d}", trues, **record)


def _plans(folder, *lengths):
    records = [{"index": i, "plan": {"steps": [{"kind": "read"}] * length}} for i, length in enumerate(lengths, 1)]
    (folder / "plans.json").write_text(json.dumps({"plans": records}), encoding="utf-8")


DECLARING = (1, 3)  # the positions in AGENT_ROWS of the two AssetOpsBench agents with the MCP, whose fifth start declares
LABEL_SEPARABLE, E18_SEPARABLE = (0, 1), (0, 2)  # the starts each label marks separable, so both mark starts 3 and 4 inseparable


def _grade(seed):
    """The grade cell of a start with both separability labels and the fields that the LABELS table reads."""
    label, e18 = seed in LABEL_SEPARABLE, seed in E18_SEPARABLE
    return json.dumps({"separable": label, "share": float(label), "rollouts": 10, "source": "greedy reference", "rules_key": "cd" * 32,
                       "e18": {"separable": e18, "commit_share": float(e18), "first_share": float(e18), "cost": 20.0 if e18 else None}})


def _six_rows(root):
    """The six rows of the agents table on its preset, five starts each. The row at position i of AGENT_ROWS passes the starts below i, gets the
    diagnosis right on the starts up to i, makes 3 + i model calls, costs 1 + i times the oracle, makes (i + 1) n tool calls on a start of
    n faults and one call to the MCP when it has it, and takes 100 and 120 seconds on the two single fault starts and 200 + 10 i on each
    two fault start. Every start submits its diagnosis but the fifth on a row that misses it, which declares insufficient evidence on the
    rows of DECLARING and writes nothing on the others, so the row at position i holds max(3 - i, 0) wrong diagnoses. Success is a pass
    or a declaration, and the separability labels are those of _grade."""
    folders = {}
    for i, row in enumerate(AGENT_ROWS):
        declares = i in DECLARING
        rows = [_episode(s, row, diagnosis_all_correct=s <= i, success=s < i or (declares and s == 4), abstained=declares and s == 4, llm_calls=3 + i,
                         cost_normalized=1 + i, tool_calls=(i + 1) * N_FAULTY[s], causal_calls=int(row[1]), wall_s={0: 100, 1: 120}.get(s, 200 + 10 * i),
                         separable=s in LABEL_SEPARABLE, grade=_grade(s), **{"pass": s < i}) for s in SEEDS]
        folders[row] = _run(root, FOLDERS[row], rows, _meta())
        _logged(folders[row], [s for s in SEEDS if s != 4 or s <= i])
        if declares:
            _logged(folders[row], [4], "declare_insufficient_evidence")
    return folders


def _environment_lines(root):
    """The six rows in the CausalDiagnosis environment, all six big layers on, the five episodes of a row from initial states of one
    test combination, so that pass^k has a combination to count. The row at position i of AGENT_ROWS passes, and so succeeds on, the
    episodes below i, makes 3 + i model calls and costs 100 + 10 i against an oracle cost of 100, with a production loss of 5 i and a time
    to repair of 3 + i cycles, on every episode. No judge reads these runs."""
    for i, row in enumerate(AGENT_ROWS):
        _run(root, f"env_{FOLDERS[row]}", [_episode(s, row, TRAINING_PRESET, block=COMBINATIONS[0], n_faulty=N_FAULTY[0], llm_calls=3 + i, cost=100 + 10 * i,
                                                    cost_oracle=100, cost_normalized=1 + i / 10, downtime_total=5 * i, repair_delay=3 + i, success=s < i,
                                                    **{"pass": s < i}) for s in SEEDS], _meta(TRAINING_PRESET))


# per row of AGENT_ROWS, whether it passes every start or none and whether it gets every diagnosis right or none, a pass needing the right
# diagnosis, which Agent-As-Tool with the MCP gets without restoring the rig
PASSES = {PE: False, PE_MCP: True, AAT: True, AAT_MCP: False, CD: True, CD_MCP: True}
TOP1 = {**PASSES, AAT_MCP: True}
# per hypothesis of HYPOTHESES on the rows of _all_or_none_rows: the paired difference of top-1 with its Holm p, then those of pass. Six
# of the 14 tests differ on every start, each with the exact sign flip p = 2 / 32 over the five blocks, which Holm over the 14 raises to
# 14 x 0.0625 = 0.875, and the others on none, with p = 1
HYPOTHESES_BY_HAND = ((1, "0.875", 1, "0.875"), (0, "1.000", 0, "1.000"), (0, "1.000", 0, "1.000"), (0, "1.000", 1, "0.875"),
                      (1, "0.875", 1, "0.875"), (0, "1.000", -1, "0.875"), (0, "1.000", 0, "1.000"))


def _all_or_none_rows(root, rows=AGENT_ROWS):
    """The rows on the preset of the agents table as PASSES and TOP1 say, so every paired difference of the hypotheses is the same on every start, and its
    exact sign flip p value over the five blocks is 1 for a difference of zero and 2 / 32 for any other, the two assignments of one sign."""
    for row in rows:
        _run(root, FOLDERS[row], [_episode(s, row, diagnosis_all_correct=TOP1[row], **{"pass": PASSES[row]}) for s in SEEDS], _meta())


LADDER_PASS = {PE_MCP: (0, 1, 0, 1, 0, 1), AAT_MCP: (0, 1, 1, 0, 0, 1), CD_MCP: (0, 1, 1, 1, 0, 1)}  # per preset of LADDER, whether the row passes every start or none
BASELINE_PASS = {PE: 1, AAT: 1, CD: 1}  # whether a row of TRAINING_BASELINES passes every start or none


def _rung_runs(root, p, rows, passes_of):
    """The rows on the preset at position p of LADDER, five starts each. The row at position a of rows passes every start or none as
    passes_of says, gets the diagnosis right wherever it passes and the CausalDiagnosis agent on every start, and costs p + 1 + a / 10
    times the oracle. The judge answers true on task completion in all five calls after Base and in none on it, on data retrieval in a
    of the five calls and on result verification in all of them."""
    preset = LADDER[p]
    for a, row in enumerate(rows):
        passes = bool(passes_of[row])
        run = _run(root, f"{preset}_{FOLDERS[row]}", [_episode(s, row, preset, diagnosis_all_correct=passes or row[0] == CD[0], cost_normalized=p + 1 + a / 10,
                                                            **{"pass": passes}) for s in SEEDS], _meta(preset))
        _judged(run, {**ALL_TRUE, "task_completion": 5 if p else 0, "data_retrieval_accuracy": a})


def _ladder_runs(root, presets=LADDER):
    """The three rows with the MCP on the presets of LADDER, as _rung_runs plays them with LADDER_PASS."""
    for p, preset in enumerate(LADDER):
        if preset in presets:
            _rung_runs(root, p, LADDER_ROWS, {row: LADDER_PASS[row][p] for row in LADDER_ROWS})


def _baseline_runs(root):
    """The baselines of the training table, the three rows without the MCP on all six big layers, as _rung_runs plays them with
    BASELINE_PASS."""
    _rung_runs(root, LADDER.index(TRAINING_PRESET), TRAINING_BASELINES, BASELINE_PASS)


# per big layer of LADDER after Base on the runs of _ladder_runs: the difference of pass of each row of LADDER_ROWS, the pooled difference
# and its Holm p. Every start pools the same three differences, so the pooled one has no spread either, four layers move it with the
# exact sign flip p = 2 / 32, which Holm over the five layers raises to 5 x 0.0625 = 0.3125, printed 0.312, and the advanced clock moves it
# by zero, with p = 1. No row passes on Base or with the repair rules, so the pooled Pass of the layer below the priced layer and below
# deterioration lies at the floor of Pass and their lines carry the mark of an uninformative test
FLOOR_MARK = " (layer below at floor)"
LAYERS_BY_HAND = ((f"+ priced reads and writes{FLOOR_MARK}", 1, 1, 1, 1, "0.312"), ("+ real-time clock", -1, 0, 0, -1 / 3, "0.312"),
                  ("+ advanced real-time clock", 1, -1, 0, 0, "1.000"), ("+ repair rules", -1, 0, -1, -2 / 3, "0.312"),
                  (f"+ deterioration and failure{FLOOR_MARK}", 1, 1, 1, 1, "0.312"))
REWARD = {"oracle": 10, "greedy": 8, "random": 4, "do_nothing": 0, "declare": 2}  # the reward of a reference policy on Base, one less per big layer added


def _policy_runs(root):
    """The reference policies of REFERENCE_POLICIES on every preset of LADDER over the rig's action server, five starts each and no model:
    the oracle passes every start and the others none, and a policy earns its REWARD less the position of the preset in LADDER."""
    for p, preset in enumerate(LADDER):
        for label, _ in REFERENCE_POLICIES:
            _run(root, f"{preset}_{label}", [_row(s, agent=label, model="", config=preset, clock_mode=aggregate.config_clock_mode(preset), diagnostics_server=False,
                                                 ranking_model=DEFAULT_RANKING_MODEL, block=f"block{s}", n_faulty=N_FAULTY[s], reward=REWARD[label] - p,
                                                 **{"pass": label == "oracle"}) for s in SEEDS], _meta(preset, model="", servers=[RIG_SERVER]))


VARIANTS = TRAINING_VARIANTS
# per row of TRAINING_ROWS and training variant, where the row passes and where it gets the diagnosis right: True on every start, False
# on none, "single" on the single fault starts alone. Agent-As-Tool with the MCP gets every diagnosis right under every variant, so pass and
# top-1 part there
TRAINED = {PE_MCP: {variant: (outcome, outcome) for variant, outcome in zip(VARIANTS, (False, True, False, True, True, False), strict=True)},
           AAT_MCP: {variant: (outcome, True) for variant, outcome in zip(VARIANTS, (False, False, False, True, True, True), strict=True)},
           CD_MCP: dict.fromkeys(VARIANTS, ("single", "single"))}
# per row of TRAINING_ROWS, the differences of pass and of top-1 of each comparison of TRAINING_COMPARISONS on the runs of _training_runs,
# the same on every start, since every outcome holds on all starts or on none or on the single fault starts under every variant of a row,
# so the same over all five, which decide, on each stratum and on each fold. Twelve of the 30 tests differ on each of their
# five blocks, each with the exact sign flip p = 2 / 32, which Holm over the 30 raises to one, as five blocks can decide no test of a
# family of 30, and the 18 others on none, with p = 1
TRAINING_PAIRS_BY_HAND = {PE_MCP: ((1, 1),) * 5, AAT_MCP: ((0, 0), (0, 0), (1, 0), (1, 0), (0, 0)), CD_MCP: ((0, 0),) * 5}
TRAINING_RESAMPLES, HOLM_DIFFERS = 200, "1.000"


def _trained(outcome, seed):
    return outcome is True or (outcome == "single" and N_FAULTY[seed] == 1)


# what the training pipeline records in run.json of a trained evaluation run (the budget table and the table of the training episodes
# through a test combination), per row of TRAINING_ROWS, variant and fold of the cross-fitting: the adapter of a fine tuned run, with the
# counts of the agent's one data manifest equal under fine-tuning and its control, and the task block of a GEPA run, each with its training
# episodes left out since they reach a fault of the held-out fold. The CausalDiagnosis agent's adapter names no optimizer steps and no such
# episodes, since that step has not run, its GEPA runs carry no block record, and Agent-As-Tool with the MCP carries no record at all.
def _adapter(fold, examples, visits, **counts):
    return {"episodes_sampled": 390 + 10 * fold, "kept_by_filter": 120, "examples": examples, VISITS: visits, **counts}


def _gate(fold, trained=True):
    return {"minimum": 70, "kept_per_training_state": 1, "training_states": 70, "rounds_sampled": fold, "episodes_sampled": 1190 * fold,
            "kept": 300 if trained else 40, "distinct_training_states": 68 if trained else 30, "second_round_due": fold == 2, "trained": trained}


RECORDS = {**{(PE_MCP, "ft", fold): {"adapter": {"training": {**_adapter(fold, 900, 6 - 3 * fold, optimizer_steps=30), "gate": _gate(fold)}}}
              for fold in TRAINING_FOLDS},
           **{(PE_MCP, "gepa", fold): {"task_block": {"candidates_played": 12, VISITS: fold}} for fold in TRAINING_FOLDS},
           **{(PE_MCP, "ft_gepa", fold): {"adapter": {"training": _adapter(fold, 900, 6 - 3 * fold, optimizer_steps=30)},
                                          "task_block": {"candidates_played": 9, VISITS: 2 + 2 * fold}} for fold in TRAINING_FOLDS},
           **{(PE_MCP, "ft_unfiltered", fold): {"adapter": {"training": _adapter(fold, 950, 4 + fold, control_trajectories=120)}} for fold in TRAINING_FOLDS},
           **{(CD_MCP, "ft", fold): {"adapter": {"training": {"episodes_sampled": 400, "kept_by_filter": 150, "examples": 1100}}} for fold in TRAINING_FOLDS},
           **{(PE_MCP, "gepa_score_only", fold): {"task_block": {"candidates_played": 12, "card": "score_only", VISITS: 7 + fold}} for fold in TRAINING_FOLDS}}
# the budget table of the runs of _record_runs and the table of the training episodes through a test combination of the runs of every
# trained variant, one line per row of TRAINING_ROWS and fold, its columns in the order GEPA, GEPA on the score alone, fine-tuning,
# fine-tuning + GEPA and the control
BUDGET_BY_HAND = [[NAMES[1], "1", "1", "1190", "300", "68", "900", "12"], [NAMES[1], "2", "2", "2380", "300", "68", "900", "12"],
                  [NAMES[3], "1", *["no record"] * 6], [NAMES[3], "2", *["no record"] * 6],
                  [NAMES[5], "1", *["none"] * 4, "1100", "none"], [NAMES[5], "2", *["none"] * 4, "1100", "none"]]
BUDGET_HEADS = (("rounds", "sampled"), ("episodes", "sampled"), ("kept by", "the filter"), ("distinct training", "initial states"), ("model calls", "as examples"),
                ("GEPA", "candidates"))
HAZARD_BY_HAND = [[NAMES[1], "1", "1", "8", "3", "4", "5"], [NAMES[1], "2", "2", "9", "0", "6", "6"], [NAMES[3], "1", *["no record"] * 5],
                  [NAMES[3], "2", *["no record"] * 5], [NAMES[5], "1", *["none"] * 5], [NAMES[5], "2", *["none"] * 5]]


def _training_runs(root, rows=TRAINING_ROWS, variants=None):
    """The rows with the MCP on the preset of training under every training variant, as TRAINED says, under two-fold cross-fitting:
    the agent as released plays the five starts in one run, which names no variant and no fold in run.json and so reads as
    released, and each trained variant plays them in two runs, the model that holds out a fold playing the starts of that fold,
    each run with the fold its model holds out and the record of RECORDS in run.json where the row, the variant and the fold
    have one. The judge answers true in every call on every episode, which the training table prints beside the verified outcome."""
    for row in rows:
        for variant in variants or VARIANTS:
            passes, right = TRAINED[row][variant]
            plays = [(None, SEEDS)] if variant == AS_RELEASED else [(fold, [s for s in SEEDS if FOLD_OF_SEED[s] == fold]) for fold in TRAINING_FOLDS]
            for fold, seeds in plays:
                fields = {} if fold is None else {TRAINING_VARIANT_KEY: variant, TRAINING_FOLD_KEY: fold, **RECORDS.get((row, variant, fold), {})}
                run = _run(root, f"train_{FOLDERS[row]}_{variant}" + ("" if fold is None else f"_f{fold}"),
                           [_episode(s, row, TRAINING_PRESET, block=COMBINATIONS[s], diagnosis_all_correct=_trained(right, s), **{"pass": _trained(passes, s)})
                            for s in seeds], _meta(TRAINING_PRESET, **fields))
                _judged(run, seeds=seeds)


def _record_runs(root):
    """The trained runs that carry a record, the fine tuned and the GEPA runs of Plan-Execute and of the CausalDiagnosis agent with the MCP."""
    _training_runs(root, (PE_MCP, CD_MCP), ("ft", "gepa"))


def _every_table(root):
    """Runs behind every table of the page: the six rows of _six_rows with a judge that answers true in every call, the environment of
    _ladder_runs between Base and the top rung, whose Base rows are those of the six rows and whose top rung is the agent as released
    of training, the baselines of training, the reference policies, training and its records."""
    for folder in _six_rows(root).values():
        _judged(folder)
    _ladder_runs(root, LADDER[1:-1])
    _baseline_runs(root)
    _policy_runs(root)
    _training_runs(root)


# ----- the page and its LaTeX form -----

def _aggregated(tmp_path, runs, resamples=50):
    out = tmp_path / "results" / "tables"
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, out, resamples=resamples, fraction="1")}
    return tables, (out.parent / "MAIN_TABLES.md").read_text(encoding="utf-8"), out.parent


def _lines(page, label):
    """The cells of the header and of every body line of a table of the page, as lists."""
    section = page.split(f"(`tab:{label}`)\n", 1)[1].split("\n#### ")[0].split("\n### ")[0].split("\n## ")[0]
    return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in section.splitlines() if line.startswith("| ")]


def _table(page, label):
    return _lines(page, label)[1:]


def _caption(page, label):
    """The caption of a table of the page."""
    return page.split(f"(`tab:{label}`)\n", 1)[1].split("\n| ", 1)[0].strip()


def _design_lines():
    """The line labels of the two tables of the agents: each design of design.AGENT_ROWS named on its first line, the agent without the
    MCP on the base layer, and its other lines of design.AGENT_LINES under it."""
    return [[suffix or name] for _, server, name in AGENT_ROWS if not server for suffix, _, _ in AGENT_LINES]


def _line_index(row, preset):
    """The position of a row of design.AGENT_ROWS on a preset among the lines of the two tables of the agents."""
    designs = [prefix for prefix, server, _ in AGENT_ROWS if not server]
    lines = [(server, line_preset) for _, server, line_preset in AGENT_LINES]
    return designs.index(row[0]) * len(lines) + lines.index((row[1], preset))


def _percent(share, error):
    """The cell of a share in the tables of the agents, a percentage with its standard error."""
    return f"{100 * share:.1f} {PM} {100 * error:.1f}"


def _st(mean):
    """The LaTeX cell of a mean whose standard error is zero, the error under the mean in smaller type."""
    zero = f"{0:.{len(mean.split('.')[1])}f}"
    return rf"\st{{{mean}}}{{$\pm$\,{zero}}}"


def _same(value, spec=".2f"):
    """The cell of an estimate that every resample repeats, so that its interval has no width."""
    return f"{value:{spec}} [{value:{spec}}, {value:{spec}}]"


def _latex(results):
    return (results / "main_tables.tex").read_text(encoding="utf-8")


def _tex(tex, label):
    """The column format, the header lines, the body lines and the sizes of the groups between rules or small spaces of a table of the
    LaTeX form, for the first model."""
    lines = tex.split(rf"\label{{tab:{label}}}" + "\n", 1)[1].split("\n" + r"\end{table}", 1)[0].splitlines()  # the heads of two lines hold a tabular of their own
    tabular = next(line for line in lines if line.startswith(r"\begin{tabular}{"))
    top, bottom = lines.index(r"\toprule"), lines.index(r"\bottomrule")
    head_end = lines.index(r"\midrule", top)
    groups = [[]]
    for line in lines[head_end + 1:bottom]:
        if line in (r"\midrule", r"\addlinespace[3pt]"):
            groups.append([])
        else:
            groups[-1].append(line)
    return {"format": tabular[len(r"\begin{tabular}{"):-1], "header": lines[top + 1:head_end], "body": [line for group in groups for line in group],
            "groups": [len(group) for group in groups]}


def _columns(tex_format):
    """The columns of a tabular format: l, c and p{...}, with the >{...} prefixes and the vertical rules left out."""
    return len(re.sub(r"p\{[^}]*\}", "p", re.sub(r">\{[^}]*\}", "", tex_format)).replace("|", ""))


def _width(line):
    """The columns a line of a tabular fills, a multicolumn cell counting its span, a two line cell counting once."""
    line = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "cell", line)
    cells = re.split(r"(?<!\\)&", line.removesuffix("\\\\"))
    return sum(int(m.group(1)) if (m := re.match(r"\s*\\multicolumn\{(\d+)\}", cell)) else 1 for cell in cells)


def _two_lines(top, bottom, align="t"):
    return rf"\begin{{tabular}}[{align}]{{@{{}}{'l' if align == 't' else 'c'}@{{}}}}{top}\\{bottom}\end{{tabular}}"


def _latex_cell(label, column, cell):
    """The LaTeX form of a cell of a table of the page: without the interval or the count that the page alone prints, and a mean with
    its standard error on one line in the two tables of the agents or the error under the mean in the tables that stack it."""
    mark = main_tables.DAGGER if cell.endswith(main_tables.DAGGER) else ""
    cell = cell.removesuffix(mark)
    if label in KEPT_INTERVALS and column not in KEPT_INTERVALS[label]:
        cell = cell.split(" [")[0]
    if label == "mcpoffline":
        cell = cell.split(" of ")[0]
    if label == "exactmatch":
        cell = cell.split(" (")[0]
    if f" {PM} " in cell and label in ONE_LINE_TABLES + STACKED_SE_TABLES:
        mean, error = cell.split(f" {PM} ")
        cell = rf"{mean}\,{{\scriptsize{PM}\,{error}}}" if label in ONE_LINE_TABLES else rf"\st{{{mean}}}{{{PM}\,{error}}}"
    return cell + mark


def _escaped(text):
    """A label or a cell with the characters that LaTeX escapes, the underscore of a test name among them."""
    for character in "_%&#$":
        text = text.replace(character, rf"\{character}")
    return text


def _latex_lines(label, rows):
    """The body lines that the LaTeX form prints for the lines of a table of the page."""
    out = []
    for index, row in enumerate(rows):
        names, cells = row[:LABELS[label]], row[LABELS[label]:]
        continued = index and rows[index - 1][0] == names[0]
        if label in FIRST_ONLY_TABLES and continued:
            names[0] = ""
        elif label in STACKED_TABLES and names[0].endswith(" + CausalDiagnosis MCP"):
            names[0] = _two_lines(names[0].removesuffix(" + CausalDiagnosis MCP"), "+ CausalDiagnosis MCP")
        if label in INDENTED_TABLES and names[0].startswith("+ "):
            names[0] = r"\quad " + names[0]
        line = " & ".join(_escaped(cell) for cell in names + [_latex_cell(label, column, cell) for column, cell in enumerate(cells)])
        for sign, tex in ((PM, r"$\pm$"), (K, r"$\kappa$"), (DELTA, r"$\Delta$"), (main_tables.DAGGER, r"$^\dagger$"), (main_tables.DASH, "$-$"), ("<", "$<$")):
            line = line.replace(sign, tex)
        out.append(re.sub(r"\bHolm p\b", r"Holm $p$", line) + r" \\")
    return out


# ----- the agents, the CausalDiagnosis MCP and the CausalDiagnosis environment, the metrics of the AssetOpsBench paper first -----

def test_the_agents_table_prints_the_judge_scores_as_mean_and_standard_error_over_start_combinations(tmp_path):
    runs = tmp_path / "runs"
    folders = _six_rows(runs)
    # the fifth call of every start of Plan-Execute gives no review and counts as false with the denominator kept, so task completion,
    # true in every other call of the first start alone, scores 0.8 there and 0 elsewhere, data retrieval, true in three calls, 0.6, and
    # result verification, true in every other call, 0.8
    for s in SEEDS:
        _judge_record(folders[PE] / f"seed{s:06d}", {**ALL_TRUE, "task_completion": 5 if s == 0 else 0, "data_retrieval_accuracy": 3}, failed=1)
    for row in AGENT_ROWS[1:]:
        _judged(folders[row])
    tables, page, _ = _aggregated(tmp_path, runs)
    rows = _table(page, "agents")
    assert [row[:1] for row in rows] == _design_lines()
    bare = next(r for r in tables["JUDGE"] if r["agent"] == f"plan_execute_{MODEL}" and r["diagnostics_server"] == "False")
    # the rates 0.8, 0, 0, 0, 0 have the mean 0.16 and a standard error from the bootstrap over the five start combinations, while a rate
    # that every start repeats has none
    assert bare["task_completion_se"] > 0
    assert rows[0][1:4] == [f"16.0 {PM} {100 * bare['task_completion_se']:.1f}", f"60.0 {PM} 0.0", f"80.0 {PM} 0.0"]
    assert all(rows[_line_index(row, AGENTS_PRESET)][1:4] == [f"100.0 {PM} 0.0"] * 3 for row in AGENT_ROWS[1:])
    assert all(rows[_line_index(row, TRAINING_PRESET)][1:] == ["no run"] * 6 for row in AGENT_ROWS)  # no run in the environment
    caption = _caption(page, "agents")
    assert f"with all six big layers on, {MODEL}, 5 episodes per line." in caption and "majority per scenario" not in page
    assert f"after the sign {PM}, its standard error from a bootstrap that resamples test combinations" in caption
    # the counts of what counted as false stand above the tables and in no caption, with the per row counts of the judge calls without a
    # parsed review and of the declaring episodes, one on each AssetOpsBench agent with the MCP here
    above = page.split("### Tables of the body")[0]
    assert "In `tab:agents`, 5 judge calls without a parsed review and 0 episodes without a judge record count as false on every criterion." in above
    assert ("Per row, the judge calls without a parsed review and the declaring episodes, which the judge scores as failures: "
            f"{NAMES[0]} 5 and 0, {NAMES[1]} 0 and 1, {NAMES[2]} 0 and 0, {NAMES[3]} 0 and 1, {NAMES[4]} 0 and 0, {NAMES[5]} 0 and 0.") in above
    # the Base lines of the ladder read the same runs of the rows with the MCP and count them for that table
    assert ("In `tab:ladder`, 0 judge calls without a parsed review and 0 episodes without a judge record count as false on every criterion. Per row, the "
            f"judge calls without a parsed review and the declaring episodes, which the judge scores as failures: {NAMES[1]}, Base 0 and 1, {NAMES[3]}, Base 0 "
            f"and 1, {NAMES[5]}, Base 0 and 0.") in above
    assert ({(r["agent"], r["diagnostics_server"]): r["declared"] for r in tables["JUDGE"]}
            == {(f"{row[0]}_{MODEL}", str(row[1])): int(i in DECLARING) for i, row in enumerate(AGENT_ROWS)})
    assert (bare["n"], bare["task_completion_n"], bare["failed_calls"], bare["episodes_without_record"], bare["judge_model"], bare["judge_protocol"]) == (5, 5, 5, 0, "judge", "paper")
    # both reductions stay in the table: the mean of the calls beside the majority of the calls, with its interval
    assert (bare["data_retrieval_accuracy"], bare["data_retrieval_accuracy_majority"], bare["generalized_result_verification_majority"]) == (0.6, 1.0, 1.0)
    assert bare["task_completion_majority"] == 0.2 and bare["task_completion_low"] != "" and bare["task_completion_majority_high"] != ""
    assert bare["task_completion_std"] == round(statistics.stdev([0.8, 0, 0, 0, 0]), 6)


def test_the_captions_of_the_tables_with_judge_cells_state_the_counting_rule_of_an_unjudged_episode(tmp_path):
    """The agents table, the cumulative layer study, the training table and the two tables of H0 print judge cells, and each caption
    states that an episode without a judge record counts as false on every criterion, the registered counting.unjudged."""
    (tmp_path / "runs").mkdir()
    _tables, page, _results = _aggregated(tmp_path, tmp_path / "runs")
    for label in ("agents", "ladder", "training", "judgerows", "judgecmp"):
        assert main_tables.UNJUDGED_RULE in _caption(page, label), label


def test_the_judge_columns_refuse_another_protocol_and_count_an_episode_without_a_record_as_false(tmp_path):
    runs = tmp_path / "runs"
    folders = {row: _run(runs, FOLDERS[row], [_episode(s, row) for s in SEEDS], _meta()) for row in AGENT_ROWS[:4]}
    _judged(folders[PE])
    _judged(folders[PE_MCP], protocol="single")  # one call at temperature zero fills no cell
    _judged(folders[AAT], seeds=range(4))  # the last start of this run has no record, which counts as false on every criterion
    tables, page, _ = _aggregated(tmp_path, runs)
    pending = next(r for r in tables["JUDGE"] if r["agent"] == f"agent_as_tool_{MODEL}")
    rows = {row: _table(page, "agents")[_line_index(row, AGENTS_PRESET)] for row in AGENT_ROWS}
    assert rows[PE][1].startswith("100.0") and rows[PE_MCP][1:4] == ["protocol single"] * 3
    assert rows[AAT][1:4] == [f"80.0 {PM} {100 * pending[f'{metric}_se']:.1f}" for metric in aggregate.JUDGE_CRITERIA]
    assert rows[AAT_MCP][1:4] == ["no judge"] * 3 and rows[CD][1:] == rows[CD_MCP][1:] == ["no run"] * 6
    assert "In `tab:agents`, 0 judge calls without a parsed review and 1 episode without a judge record count as false on every criterion." in page.split("### Tables of the body")[0]
    assert (pending["n"], pending["task_completion_n"], pending["episodes_without_record"], pending["task_completion"], pending["task_completion_majority"]) == (5, 5, 1, 0.8, 0.8)
    assert {(r["agent"], r["judge_protocol"]) for r in tables["JUDGE"]} == {(f"plan_execute_{MODEL}", "paper"), (f"plan_execute_{MODEL}", "single"), (f"agent_as_tool_{MODEL}", "paper")}
    assert [r["judge_protocol"] for r in tables["JUDGE_AGREEMENT"]] == ["paper"]  # one call has no pair to agree


def test_the_training_table_counts_an_episode_without_a_judge_record_as_false_and_names_it_above_the_tables(tmp_path):
    """The judge columns of the training table count an episode without a judge record as false on every criterion, as the agents
    table does (counting.unjudged), and the page names it in the sentence above the tables: Plan-Execute with the MCP as released
    loses the record of one of its five starts, so its three judge cells read 80.0 over all five."""
    runs = tmp_path / "runs"
    _training_runs(runs, (PE_MCP,), (AS_RELEASED,))
    (runs / f"train_{FOLDERS[PE_MCP]}_{AS_RELEASED}" / "seed000000" / "hidden" / "judge.json").unlink()
    tables, page, _ = _aggregated(tmp_path, runs, resamples=TRAINING_RESAMPLES)
    row = next(r for r in tables["TRAINING"] if r["subset"] == "all" and r["diagnostics_server"] in (True, "True"))
    assert (row["n"], row["judged"], row["episodes_without_record"], row["failed_calls"], row["task_completion"], row["task_completion_n"]) == (5, 4, 1, 0, 0.8, 5)
    line = next(cells for cells in _table(page, "training") if cells[0] == PE_MCP[2])
    assert [cell.split(f" {PM} ")[0] for cell in line[1:4]] == ["80.0"] * 3
    assert ("In `tab:training`, 0 judge calls without a parsed review and 1 episode without a judge record count as false on every criterion."
            in page.split("### Tables of the body")[0])


def test_an_episode_judged_under_another_protocol_fills_no_cell_of_its_row(tmp_path):
    """A row whose cells print mixed protocols enters no count of the sentence above the tables, so the sentence and the
    cells describe the same episodes, and here no judged row is left, in the agents table or on the Base line of the ladder, so there is
    no sentence."""
    runs = tmp_path / "runs"
    folder = _run(runs, FOLDERS[PE_MCP], [_episode(s, PE_MCP) for s in SEEDS], _meta())
    for s in SEEDS:
        _judge_record(folder / f"seed{s:06d}", ALL_TRUE, protocol="paper" if s else "single", failed=1 if s == 1 else 0)
    _, page, _ = _aggregated(tmp_path, runs)
    agents = _table(page, "agents")[_line_index(PE_MCP, AGENTS_PRESET)]
    assert agents[1:4] == ["mixed protocols"] * 3 and _table(page, "ladder")[0][2:5] == ["mixed protocols"] * 3
    assert "judge calls without a parsed review" not in page


def test_a_call_that_failed_counts_as_false_in_the_scores_and_stays_out_of_the_agreement(tmp_path):
    """The reproduction's rule, which episode_judge.py shares: the share of true answers over the five planned calls, so
    four parsed calls that all say true give 0.8 in the pass rate and in the scores and agree with each other."""
    ep = episode_folder(tmp_path)
    record = judge.judge_episode(ep, FakeJudge(["no json here", RUBRIC_PASS, RUBRIC_PASS, RUBRIC_PASS, RUBRIC_PASS]), "paper", agent_model="agent-model")
    fields = aggregate.judge_fields({"episode_dir": ep})
    assert record["pass_rate"] == 0.8 and (fields["task_completion"], fields["task_completion_agreement"], fields["judge_failed_calls"]) == (0.8, 1.0, 1)


def test_the_judge_fields_read_the_record_that_judge_py_writes(tmp_path):
    ep = episode_folder(tmp_path)
    judge.judge_episode(ep, FakeJudge([RUBRIC_PASS, "no json here", RUBRIC_PASS, RUBRIC_PASS, RUBRIC_PASS]), "paper", agent_model="agent-model")
    fields = aggregate.judge_fields({"episode_dir": ep})
    assert (fields["task_completion"], fields["task_completion_majority"], fields["task_completion_agreement"]) == (0.8, True, 1.0)
    assert (fields["judge_record_protocol"], fields["judge_record_model"], fields["judge_failed_calls"]) == ("paper", "fake-judge", 1)
    assert aggregate.judge_fields({"episode_dir": tmp_path}) == {} and aggregate.judge_fields({"episode_dir": None}) == {}
    # two of five against three: four of the ten pairs agree, 1 + 3, and the fifth answer is not there
    assert aggregate.pairwise_agreement([True, True, False, False, False]) == 0.4 and aggregate.pairwise_agreement([True]) is None


def test_the_first_three_criteria_of_the_judge_prompt_are_the_scores_of_figure_5():
    """The reproduction paragraph of the paper reads the three scores of Figure 5 of AssetOpsBench from the first three
    criteria of the six criterion prompt of its code, whose headings run in that order."""
    assert llm_judge._RUBRIC_KEYS[:3] == aggregate.JUDGE_CRITERIA and len(llm_judge._RUBRIC_KEYS) == 6
    assert re.findall(r"^(\d)\. \*\*(.+?):\*\*$", llm_judge._PROMPT_TEMPLATE, re.M)[:3] == [("1", "Task Completion"), ("2", "Data Retrieval & Accuracy"),
                                                                                         ("3", "Generalized Result Verification")]
    assert judge.PROTOCOLS["paper"] == {"calls": 5, "temperature": 0.3} and judge.JUDGE_NAME == "judge.json"


OUTCOME_BY_HAND = (  # Top-1, Pass, Success, Wrong, Declared and Model calls of the rows of _six_rows
    ["0.20", "0.00", "0.00", "0.60", "0.00", "3.0"],
    ["0.40", "0.20", "0.40", "0.40", "0.20", "4.0"],
    ["0.60", "0.40", "0.40", "0.20", "0.00", "5.0"],
    ["0.80", "0.60", "0.80", "0.00", "0.20", "6.0"],
    ["1.00", "0.80", "0.80", "0.00", "0.00", "7.0"],
    ["1.00", "1.00", "1.00", "0.00", "0.00", "8.0"],
)


def test_the_agents_table_holds_four_lines_per_design_with_the_judge_first_and_pass_k_after_top_1_and_pass(tmp_path):
    """The agents table of the paper: per design of design.AGENT_ROWS its line without and with the CausalDiagnosis MCP on the base layer and,
    under them, the same two in the CausalDiagnosis environment with all six big layers on (design.AGENT_LINES), and the columns TC, DRA
    and RV of the judge, then Top-1, Pass and pass^k at the k of main_tables.PASS_K_SHOWN, each cell a percentage, the mean with its
    standard error on one line. On the base layer every start is a combination of its own, so no combination holds the episodes that
    pass^k asks for and its cell says none, while the five episodes of a line in the environment share one combination (_environment_lines)."""
    runs = tmp_path / "runs"
    for folder in _six_rows(runs).values():
        _judged(folder)
    _environment_lines(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    rows = _table(page, "agents")
    assert [row[:1] for row in rows] == _design_lines()
    assert _lines(page, "agents")[0] == ["Agent", "TC (%)", "DRA (%)", "RV (%)", "Top-1 (%)", "Pass (%)", f"Pass^{PASS_K} (%)"]
    agents = {(r["agent"], r["diagnostics_server"], r["config"]): r for r in tables["AGENTS"]}
    for i, row in enumerate(AGENT_ROWS):
        bare = agents[(f"{row[0]}_{MODEL}", str(row[1]), AGENTS_PRESET)]
        # on the base layer the row at position i gets the diagnosis right on the starts up to i and passes the starts below i (_six_rows)
        top1, passes = float(OUTCOME_BY_HAND[i][0]), float(OUTCOME_BY_HAND[i][1])
        assert rows[_line_index(row, AGENTS_PRESET)][1:] == [f"100.0 {PM} 0.0"] * 3 + [_percent(top1, bare["diagnosis_all_correct_se"]),
                                                                                       _percent(passes, bare["pass_se"]), "none"]
        # in the environment the one combination of a line passes on its episodes below i, so pass^k is the share of the k subsets of its
        # five episodes that all pass, and one combination leaves the bootstrap no spread
        all_pass = math.comb(i, PASS_K) / math.comb(len(SEEDS), PASS_K)
        assert rows[_line_index(row, TRAINING_PRESET)][1:] == ["no judge"] * 3 + [_percent(1.0, 0), _percent(i / len(SEEDS), 0), _percent(all_pass, 0)]
    tex = _tex(_latex(results), "agents")
    assert tex["format"] == "l" + "c" * 6 and tex["groups"] == [len(AGENT_LINES)] * 3
    assert tex["header"] == [r" & \multicolumn{3}{c}{Judge metrics (\%)} & \multicolumn{3}{c}{Verified outcome} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
                             rf"Agent & TC & DRA & RV & Top-1 & Pass & Pass$^{{{PASS_K}}}$ \\"]
    assert tex["body"] == _latex_lines("agents", rows)
    one = r"\,{\scriptsize$\pm$\,0.0}"
    assert tex["body"][2] == rf"\quad + CausalDiagnosis environment & no judge & no judge & no judge & 100.0{one} & 0.0{one} & 0.0{one} \\"
    assert (f"Pass^{PASS_K} is the probability that {main_tables.in_words(PASS_K)} episodes of one test combination from different initial states all "
            "pass, by the estimator of tau-bench, averaged over the test combinations.") in _caption(page, "agents")
    assert [main_tables.in_words(n) for n in (1, 5, 9, 10, 12)] == ["one", "five", "nine", "10", "12"]


def test_the_measures_of_the_agents_print_the_lines_of_the_agents_table_with_a_dash_where_a_measure_needs_the_layers(tmp_path):
    """The table of the measures of the agents: the lines of the agents table and the columns of main_tables.AGENT_MEASURES, the verified
    outcome and the costs, then the cost in money, the production loss and the time to repair, which exist only with the layers on, so a
    line on the base layer prints a dash there. In the environment the row at position i of design.AGENT_ROWS costs 100 + 10 i with a
    production loss of 5 i and a time to repair of 3 + i cycles on every episode (_environment_lines), so these cells have no spread."""
    runs = tmp_path / "runs"
    _six_rows(runs)
    _environment_lines(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    rows = _table(page, "agentmeasures")
    assert [row[:1] for row in rows] == _design_lines()
    assert _lines(page, "agentmeasures")[0] == ["Agent", "Exact match (%)", "Success (%)", "Wrong (%)", "Declared (%)", "Model calls", "Cost / oracle", "Cost",
                                                "Production loss", "Time to repair"]
    agents = {(r["agent"], r["diagnostics_server"], r["config"]): r for r in tables["AGENTS"]}
    for i, row in enumerate(AGENT_ROWS):
        bare = agents[(f"{row[0]}_{MODEL}", str(row[1]), AGENTS_PRESET)]
        # on the base layer the success, the wrong diagnoses and the declarations of OUTCOME_BY_HAND, the model calls and the cost over the
        # oracle's that every start repeats, and a dash for each measure of the layers
        success, wrong, declared = (float(share) for share in OUTCOME_BY_HAND[i][2:5])
        assert rows[_line_index(row, AGENTS_PRESET)][2:] == [_percent(success, bare["success_se"]), _percent(wrong, bare["diagnosis_wrong_se"]),
                                                             _percent(declared, bare["abstained_se"]), f"{3 + i:.1f} {PM} 0.0", f"{1 + i:.2f} {PM} 0.00",
                                                             *[main_tables.DASH] * 3]
        assert rows[_line_index(row, TRAINING_PRESET)][5:] == [f"{3 + i:.1f} {PM} 0.0", f"{1 + i / 10:.2f} {PM} 0.00", f"{100 + 10 * i:.1f} {PM} 0.0",
                                                               f"{5 * i:.1f} {PM} 0.0", f"{3 + i:.1f} {PM} 0.0"]
    caption = _caption(page, "agentmeasures")
    assert caption.startswith(f"The lines of the agents table, {MODEL}, 5 episodes per line. Exact match is the share of episodes whose diagnosis names "
                              "exactly the faulty components.")
    majority, episodes = 3, 5  # three of the five starts fault the first two components
    assert f"Exact match stands against a majority-class baseline of {majority / episodes:.2f} ({majority} of {episodes} episodes)." in caption
    assert f"Cost, Production loss and Time to repair exist only with the layers on, so a line on the base layer prints {main_tables.DASH} there." in caption
    tex = _tex(_latex(results), "agentmeasures")
    assert tex["format"] == "l" + "c" * 9 and tex["groups"] == [len(AGENT_LINES)] * 3
    assert tex["header"] == [r" & \multicolumn{6}{c}{Verified outcome} & \multicolumn{3}{c}{With the layers on} \\", r"\cmidrule(lr){2-7}\cmidrule(l){8-10}",
                             "Agent & " + " & ".join([_two_lines("Exact", "match", "b"), "Success", "Wrong", "Declared", _two_lines("Model", "calls", "b"),
                                                      _two_lines("Cost /", "oracle", "b"), "Cost", _two_lines("Production", "loss", "b"),
                                                      _two_lines("Time to", "repair", "b")]) + r" \\"]
    assert tex["body"] == _latex_lines("agentmeasures", rows) and tex["body"][0].endswith(r" & $-$ & $-$ & $-$ \\")


def test_a_column_list_drives_the_table_of_the_measures_of_the_agents(tmp_path, monkeypatch):
    """The metrics review may change the columns of the table of the measures of the agents, so its head, its cells, its caption and the
    groups of its LaTeX head follow main_tables.AGENT_MEASURES: here a share of the verified outcome and a measure of the layers in a
    group of another name."""
    columns = (main_tables.Column(("Success",), "success", ".1f", "the share of episodes that pass", 100),
               main_tables.Column(("Production", "loss"), "downtime_total", ".1f", "the loss of the asset while it runs degraded", layers=True, group="Plant"))
    monkeypatch.setattr(main_tables.body_tables, "AGENT_MEASURES", columns)
    runs = tmp_path / "runs"
    _environment_lines(runs)
    _, page, results = _aggregated(tmp_path, runs)
    assert _lines(page, "agentmeasures")[0] == ["Agent", "Success (%)", "Production loss"]
    rows = _table(page, "agentmeasures")
    for i, row in enumerate(AGENT_ROWS):
        assert rows[_line_index(row, AGENTS_PRESET)][1:] == ["no run"] * 2
        assert rows[_line_index(row, TRAINING_PRESET)][1:] == [_percent(i / len(SEEDS), 0), f"{5 * i:.1f} {PM} 0.0"]
    assert _caption(page, "agentmeasures") == (
        f"The lines of the agents table, {MODEL}, 5 episodes per line. Success is the share of episodes that pass. Production loss is the loss of the "
        f"asset while it runs degraded. Success is a percentage. Production loss exists only with the layers on, so a line on the base layer prints "
        f"{main_tables.DASH} there. Every cell is the mean over the episodes of the line and, after the sign {PM}, its standard error from a bootstrap "
        "that resamples test combinations.")
    tex = _tex(_latex(results), "agentmeasures")
    assert tex["format"] == "lcc" and tex["header"] == [r" & \multicolumn{1}{c}{Verified outcome} & \multicolumn{1}{c}{Plant} \\",
                                                        r"\cmidrule(lr){2-2}\cmidrule(l){3-3}", "Agent & Success & " + _two_lines("Production", "loss", "b") + r" \\"]


def test_the_hypotheses_print_the_seven_comparisons_in_two_groups_with_the_differences_and_the_holm_p_of_agents_pairs(tmp_path):
    runs = tmp_path / "runs"
    _all_or_none_rows(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    rows = _table(page, "hyp")
    # the baselines that pass no start, Plan-Execute without the MCP and Agent-As-Tool with it, are at floor, H1a and H1d uninformative,
    # and the caption states the rule without a decision id
    at_floor = {"H1a", "H1d"}
    assert [row[:2] for row in rows] == [[name, f"{row[2]} against {reference[2]}" + (" (baseline at floor)" if name in at_floor else "")]
                                         for name, row, reference in HYPOTHESES]
    assert "is at floor, and H1 against it is uninformative" in page and "below 5%" in page
    assert [row[2:] for row in rows] == [[_same(top1, "+.2f"), holm_top1, _same(passes, "+.2f"), holm_pass] for top1, holm_top1, passes, holm_pass in HYPOTHESES_BY_HAND]
    decided = {r["hypothesis"]: r for r in tables["AGENTS_pairs"] if r["hypothesis"] != "scoring"}
    assert list(decided) == [name for name, _, _ in HYPOTHESES] and {r["family_tests"] for r in decided.values()} == {14}
    assert (decided["H2b"]["pass_p"], decided["H2b"]["pass_holm_p"], decided["H2b"]["diagnosis_all_correct_holm_p"]) == (0.0625, 0.875, 1.0)
    tex = _tex(_latex(results), "hyp")
    assert tex["groups"] == [4, 3]  # H1 above the rule, H2 below it
    assert tex["body"] == [" & ".join(row) + r" \\" for row in rows]  # the differences keep their interval in both forms


def test_a_holm_p_below_one_in_a_thousand_prints_as_a_bound(tmp_path):
    """With the CausalDiagnosis agent passing and diagnosing every start and Plan-Execute none, H1a is the one comparison with both arms,
    its two differences lie above zero on every start, and the exact sign flip p value over the five blocks is 2 / 32 whatever the number of
    resamples, which Holm over the fourteen tests of the registered family, the twelve without a run at p = 1, raises to 0.875. A Holm p
    below 0.001, which takes more blocks, prints as the bound. The comparisons without both arms say no run."""
    runs = tmp_path / "runs"
    _all_or_none_rows(runs, (PE, CD))
    tables, page, results = _aggregated(tmp_path, runs, resamples=5000)
    h1a, = [r for r in tables["AGENTS_pairs"] if r["hypothesis"] != "scoring"]
    assert (h1a["hypothesis"], h1a["family_tests"], h1a["pass_p"], h1a["pass_holm_p"], h1a["diagnosis_all_correct_holm_p"]) == ("H1a", 14, 0.0625, 0.875, 0.875)
    rows = _table(page, "hyp")
    assert rows[0][2:] == [_same(1.0, "+.2f"), "0.875", _same(1.0, "+.2f"), "0.875"] and all(row[2:] == ["no run"] * 4 for row in rows[1:])
    assert _tex(_latex(results), "hyp")["body"][0] == rf"H1a & {rows[0][1]} & +1.00 [+1.00, +1.00] & 0.875 & +1.00 [+1.00, +1.00] & 0.875 \\"
    assert [main_tables.holm_cell({"p": p}, "p").full for p in (0.0008, 0.001, 0.0625)] == ["< 0.001", "0.001", "0.062"]


# ----- the environment ladder and the training tables of the body -----

def test_the_ladder_holds_the_three_agents_with_the_mcp_on_six_settings_with_the_judge_first_and_the_verified_outcome_after_it(tmp_path):
    """The 18 lines of the ladder merge the judge and the verified outcome, TC, DRA and RV, then Top-1, Pass and the cost over the
    oracle's, each cell the mean and its standard error, which the LaTeX form sets under the mean, the agent named on its first line."""
    runs = tmp_path / "runs"
    _ladder_runs(runs)
    # the CausalDiagnosis agent without the MCP on the priced row is no line of the ladder, which plays the agents with it
    _run(runs, "priced_cd_bare", [_episode(s, CD, "priced", diagnosis_all_correct=False, cost_normalized=9, **{"pass": False}) for s in SEEDS], _meta("priced"))
    _, page, results = _aggregated(tmp_path, runs)
    assert all(server for _, server, _ in LADDER_ROWS)
    rows = _table(page, "ladder")
    assert [row[:2] for row in rows] == [[name, setting] for _, _, name in LADDER_ROWS for setting in LADDER_NAMES]
    # every start of a line repeats its outcome (_ladder_runs), so no cell has a spread
    expected = []
    for a, row in enumerate(LADDER_ROWS):
        for p, passes in enumerate(LADDER_PASS[row]):
            expected.append([f"{100.0 * bool(p):.1f} {PM} 0.0", f"{20.0 * a:.1f} {PM} 0.0", f"100.0 {PM} 0.0", f"{100.0 * (passes or row == CD_MCP):.1f} {PM} 0.0",
                             f"{100.0 * passes:.1f} {PM} 0.0", f"{p + 1 + a / 10:.2f} {PM} 0.00"])
    assert [row[2:] for row in rows] == expected
    assert _lines(page, "ladder")[0] == ["Agent", "Setting", "TC (%)", "DRA (%)", "RV (%)", "Top-1 (%)", "Pass (%)", "Cost / oracle"]
    # the Base lines are the runs of the agents table and the top layer those of its lines in the environment with the MCP, whose judge
    # cells they repeat, and no line without the MCP has a run
    first = _table(page, "agents")
    for a, row in enumerate(LADDER_ROWS):
        assert first[_line_index(row, AGENTS_PRESET)][1:4] == rows[len(LADDER) * a][2:5]
        assert first[_line_index(row, TRAINING_PRESET)][1:4] == rows[len(LADDER) * (a + 1) - 1][2:5]
    assert all(first[_line_index(row, preset)][1:] == ["no run"] * 6 for row in AGENT_ROWS if not row[1] for preset in (AGENTS_PRESET, TRAINING_PRESET))
    tex = _tex(_latex(results), "ladder")
    assert tex["groups"] == [6, 6, 6]
    assert tex["header"] == [r" & & \multicolumn{3}{c}{Judge metrics (\%)} & \multicolumn{3}{c}{Verified outcome} \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
                             r"Agent & Setting & TC & DRA & RV & Top-1 & Pass & Cost / oracle \\"]
    assert tex["body"] == _latex_lines("ladder", rows)
    assert tex["body"][6:8] == [rf"{NAMES[3]} & Base & {_st('0.0')} & {_st('20.0')} & {_st('100.0')} & {_st('0.0')} & {_st('0.0')} & {_st('1.10')} \\",
                                rf" & + priced reads and writes & {_st('100.0')} & {_st('20.0')} & {_st('100.0')} & {_st('100.0')} & {_st('100.0')} & {_st('2.10')} \\"]
    assert tex["body"][12] == rf"{NAMES[5]} & Base & {_st('0.0')} & {_st('40.0')} & {_st('100.0')} & {_st('100.0')} & {_st('0.0')} & {_st('1.20')} \\"


def test_the_training_tables_print_every_episode_the_two_strata_and_each_fold_under_cross_fitting_and_decide_on_every_episode(tmp_path):
    """Every test episode is played by the model of the fold that did not train on its combination, so the training table prints
    Top-1 and Pass over all five starts and on the two single fault and three two fault starts, the training hypotheses decide on all
    five with Holm over the 24 tests and print the two strata beside them without a test, and the table by fold prints the three starts
    of fold 1 and the two of fold 2 without a test, Top-1 before Pass in each. Each agent's group opens
    with its baseline without the MCP, its reference line on all six big layers on the same five starts."""
    runs = tmp_path / "runs"
    _training_runs(runs)
    _baseline_runs(runs)
    tables, page, results = _aggregated(tmp_path, runs, resamples=TRAINING_RESAMPLES)
    assert all(server for _, server, _ in TRAINING_ROWS)
    trained = _table(page, "training")
    # the baseline without the MCP and the agent with it as released head the group of an agent and its trained variants follow indented
    assert [row[:1] for row in trained] == [line for baseline, (_, _, name) in zip(TRAINING_BASELINES, TRAINING_ROWS)
                                            for line in [[baseline[2]], *([name if variant == AS_RELEASED else f"+ {VARIANT_NAMES[variant]}"] for variant in VARIANTS)]]
    assert "no training" not in page.split("(`tab:training`)")[1].split("####")[0]
    expected = []
    for baseline, row in zip(TRAINING_BASELINES, TRAINING_ROWS):
        passes = bool(BASELINE_PASS[baseline])  # _baseline_runs, passing every start or none, the diagnosis right wherever it passes
        # its judge answers true on task completion and result verification in all five calls and on data retrieval in a of them
        judged = ["100.0", f"{20 * TRAINING_BASELINES.index(baseline):.1f}", "100.0"]
        expected.append(judged + [f"{100.0 * outcome:.1f}" for outcome in (passes or baseline == CD, passes) for _ in range(3)])
        for variant in VARIANTS:
            passes, right = TRAINED[row][variant]
            cells = ["100.0"] * 3  # the judge of _training_runs answers true in every call
            for outcome in (right, passes):  # top-1, then pass, each over all five starts, the single fault starts 0 and 1, the two fault starts 2 to 4
                share = 1.0 if outcome is True else 0.4 if outcome == "single" else 0.0
                cells += [f"{100.0 * share:.1f}", f"{100.0 * (outcome is not False):.1f}", f"{100.0 * (outcome is True):.1f}"]
            expected.append(cells)
    # the judge of these runs passes every start, one side alone, so its kappa has no value, and no run records provenance
    expected = [cells[:3] + ["none"] + cells[3:] + ["none"] for cells in expected]
    assert [[cell.split(f" {PM} ")[0] for cell in row[1:]] for row in trained] == expected
    # a stratum holds every start or none of a row, so the bootstrap leaves its share no spread, while all of the CausalDiagnosis agent mixes them
    assert all(cell.endswith(f" {PM} 0.0") for row in trained for column, cell in enumerate(row[5:11]) if column % 3)
    assert _lines(page, "training")[0] == ["Agent", "TC (%)", "DRA (%)", "RV (%)", f"Judge {K}",
                                           *(f"{name} {head} (%)" for name in ("Top-1", "Pass") for head in ("all", "single-fault", "two-fault")), "Unmatched"]
    # the runs as released carry no training variant in run.json and still fill their rows, and each trained variant pools the runs of its two folds
    assert sorted(r["n"] for r in tables["TRAINING"] if r["training_variant"] == AS_RELEASED) == [2] * 6 + [3] * 6 + [5] * 6
    assert {r["n"] for r in tables["TRAINING"] if r["subset"] == "all"} == {5}
    pairs = _table(page, "trainhyp")
    assert [row[:2] for row in pairs] == [[name, comparison] for _, _, name in TRAINING_ROWS for _, _, comparison in TRAINING_COMPARISONS]
    expected = []
    for row in TRAINING_ROWS:
        for passes, right in TRAINING_PAIRS_BY_HAND[row]:
            expected.append([_same(right, "+.2f"), HOLM_DIFFERS if right else "1.000", _same(passes, "+.2f"), HOLM_DIFFERS if passes else "1.000",
                             _same(right, "+.2f"), _same(right, "+.2f"), _same(passes, "+.2f"), _same(passes, "+.2f")])
    assert [row[2:] for row in pairs] == expected
    assert _lines(page, "trainhyp")[0] == ["Agent", "Comparison", f"{DELTA}Top-1, all 5 episodes", "Holm p", f"{DELTA}Pass, all 5 episodes", "Holm p",
                                           f"{DELTA}Top-1 single-fault", f"{DELTA}Top-1 two-fault", f"{DELTA}Pass single-fault", f"{DELTA}Pass two-fault"]
    folds = _table(page, "trainfolds")
    assert [row[:2] for row in folds] == [row[:2] for row in pairs]
    assert [row[2:] for row in folds] == [[_same(right, "+.2f")] * 2 + [_same(passes, "+.2f")] * 2 for row in TRAINING_ROWS for passes, right in TRAINING_PAIRS_BY_HAND[row]]
    assert _lines(page, "trainfolds")[0] == ["Agent", "Comparison", f"{DELTA}Top-1 fold 1", f"{DELTA}Top-1 fold 2", f"{DELTA}Pass fold 1", f"{DELTA}Pass fold 2"]
    by_subset = {}
    for r in tables["TRAINING_pairs"]:
        by_subset.setdefault(r["subset"], set()).add((r.get("family_tests", ""), r["pairs"], r["clusters"]))
    assert by_subset == {"all": {(30, 5, 5)}, "single": {("", 2, 2)}, "two": {("", 3, 3)}, "fold_1": {("", 3, 3)}, "fold_2": {("", 2, 2)}}
    tex = _latex(results)
    training, hypotheses, by_fold = _tex(tex, "training"), _tex(tex, "trainhyp"), _tex(tex, "trainfolds")
    assert training["groups"] == [7, 7, 7] and hypotheses["groups"] == by_fold["groups"] == [5, 5, 5]
    assert training["header"] == [r" & \multicolumn{3}{c}{Judge metrics (\%)} & & \multicolumn{7}{c}{Verified outcome} \\", r"\cmidrule(lr){2-4}\cmidrule(l){6-12}",
                                  r" & & & & & \multicolumn{3}{c}{Top-1} & \multicolumn{3}{c}{Pass} & \\", r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
                                  r"Agent & TC & DRA & RV & Judge $\kappa$ & all & single-fault & two-fault & all & single-fault & two-fault & Unmatched \\"]
    assert training["body"] == _latex_lines("training", trained)
    judge3, ones, zeros = (" & ".join([_st(mean)] * 3) for mean in ("100.0", "100.0", "0.0"))
    assert training["body"][7] == rf"{NAMES[2]} & {_st('100.0')} & {_st('20.0')} & {_st('100.0')} & none & {ones} & {ones} & none \\"
    assert training["body"][8] == rf"{NAMES[3]} & {judge3} & none & {ones} & {zeros} & none \\"
    assert training["body"][9:11] == [rf"\quad + {variant} & {judge3} & none & {ones} & {zeros} & none \\" for variant in ("GEPA", "GEPA on the score alone")]
    assert training["body"][20].startswith(r"\quad + fine-tuning on unfiltered trajectories & ")
    strata = " & ".join(_two_lines(rf"$\Delta${name}", head, "c") for name in ("Top-1", "Pass") for head in ("single-fault", "two-fault"))
    assert hypotheses["header"] == [r" & & \multicolumn{4}{c}{all 5 episodes, the decision} & \multicolumn{4}{c}{strata, no test} \\", r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
                                    rf"Agent & Comparison & $\Delta$Top-1 & Holm $p$ & $\Delta$Pass & Holm $p$ & {strata} \\"]
    one, zero = "+1.00 [+1.00, +1.00]", "+0.00 [+0.00, +0.00]"
    first = _two_lines(NAMES[0], "+ CausalDiagnosis MCP")
    assert hypotheses["body"][0] == rf"{first} & GEPA against no training & {one} & 1.000 & {one} & 1.000 & {one} & {one} & {one} & {one} \\"
    assert hypotheses["body"][1] == rf" & GEPA against GEPA on the score alone & {one} & 1.000 & {one} & 1.000 & {one} & {one} & {one} & {one} \\"
    assert hypotheses["body"][4] == rf" & fine-tuning against fine-tuning on unfiltered trajectories & {one} & 1.000 & {one} & 1.000 & {one} & {one} & {one} & {one} \\"
    assert hypotheses["body"][7] == rf" & fine-tuning against no training & {zero} & 1.000 & {one} & 1.000 & {zero} & {zero} & {one} & {one} \\"
    assert by_fold["header"] == [r" & & \multicolumn{2}{c}{$\Delta$Top-1} & \multicolumn{2}{c}{$\Delta$Pass} \\", r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
                                 r"Agent & Comparison & fold 1 & fold 2 & fold 1 & fold 2 \\"]
    assert by_fold["body"][5] == rf"{NAMES[3]} & GEPA against no training & {zero} & {zero} & {zero} & {zero} \\"
    assert by_fold["body"][7] == rf" & fine-tuning against no training & {zero} & {zero} & {one} & {one} \\"


# ----- the appendix: execution efficiency, layer effects, reference policies, budget, strata, the MCP offline -----

def test_the_execution_efficiency_splits_the_steps_and_the_runtime_by_the_fault_count(tmp_path):
    runs = tmp_path / "runs"
    _six_rows(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    rows = _table(page, "efficiency")
    assert [row[0] for row in rows] == list(NAMES)
    spread = f"{statistics.stdev([100, 120]):.0f}"
    for i, (row, (_, server, _)) in enumerate(zip(rows, AGENT_ROWS)):
        mcp = int(server)  # the call to the MCP counts as a step
        assert row[1:] == [f"{i + 1 + mcp:.1f} {PM} 0.0", f"110 {PM} {spread}", f"{2 * (i + 1) + mcp:.1f} {PM} 0.0", f"{200 + 10 * i} {PM} 0"]
    assert _lines(page, "efficiency")[0] == ["Agent", "Single fault steps", "Single fault runtime (s)", "Two faults steps", "Two faults runtime (s)"]
    assert "model calls" not in page.split("(`tab:efficiency`)")[1].split("####")[0].lower()
    split = {(r["agent"], r["faults"], r["diagnostics_server"]): r for r in tables["EFFICIENCY"]}
    assert len(split) == 12
    single, two = split[(f"causal_{MODEL}", "single", "True")], split[(f"causal_{MODEL}", "two", "True")]
    assert (single["n"], single["tool_invocations"], single["wall_s"], two["n"], two["tool_invocations"], two["wall_s"]) == (2, 7.0, 110.0, 3, 13.0, 250.0)
    assert single["wall_s_std"] == round(statistics.stdev([100, 120]), 6) and two["wall_s_std"] == 0.0 and single["wall_s_low"] != ""
    tex = _tex(_latex(results), "efficiency")
    assert tex["header"] == [r" & \multicolumn{2}{c}{Single fault} & \multicolumn{2}{c}{Two faults} \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
                             r"Agent & Steps & Runtime (s) & Steps & Runtime (s) \\"]
    assert tex["body"][5] == rf"{NAMES[5]} & 7.0 $\pm$ 0.0 & 110 $\pm$ 14 & 13.0 $\pm$ 0.0 & 250 $\pm$ 0 \\"


def test_a_fault_count_without_episodes_keeps_its_row_with_empty_cells(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, FOLDERS[CD_MCP], [_episode(s, CD_MCP, wall_s=7, tool_calls=4) for s in (0, 1)], _meta())
    tables, page, _ = _aggregated(tmp_path, runs)
    split = {r["faults"]: r for r in tables["EFFICIENCY"]}
    assert (split["single"]["n"], split["single"]["wall_s"], split["two"]["n"], split["two"]["tool_invocations"], split["two"]["tool_invocations_n"]) == (2, 7.0, 0, "", 0)
    rows = _table(page, "efficiency")
    assert rows[5][1:] == [f"4.0 {PM} 0.0", f"7 {PM} 0", "none", "none"] and all(row[1:] == ["no run"] * 4 for row in rows[:5])


def test_the_layer_effects_print_per_layer_the_three_differences_the_pooled_one_and_the_holm_p_of_layers_pairs(tmp_path):
    runs = tmp_path / "runs"
    _ladder_runs(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    assert [setting.removesuffix(FLOOR_MARK) for setting, *_ in LAYERS_BY_HAND] == list(LADDER_NAMES[1:])
    assert _lines(page, "layers")[0] == ["Layer added", *(f"{DELTA}Pass {name}" for name in (NAMES[1], NAMES[3], NAMES[5])), f"pooled {DELTA}Pass", "Holm p"]
    assert _table(page, "layers") == [[setting, *(_same(d, "+.2f") for d in differences), holm] for setting, *differences, holm in LAYERS_BY_HAND]
    pooled = [r for r in tables["LAYERS_pairs"] if r["agent"] == "pooled"]
    assert [(r["config"], r["reference_config"], r["pairs"], r["clusters"]) for r in pooled] == [(above, below, 15, 5) for below, above in zip(LADDER, LADDER[1:])]
    tex = _tex(_latex(results), "layers")
    assert tex["header"] == ["Layer added & " + " & ".join([*(_two_lines(r"$\Delta$Pass", name, "c") for name in (NAMES[1], NAMES[3], NAMES[5])),
                                                             _two_lines("pooled", r"$\Delta$Pass", "c"), r"Holm $p$ \\"])]
    assert tex["body"][1] == r"+ real-time clock & -1.00 [-1.00, -1.00] & +0.00 [+0.00, +0.00] & +0.00 [+0.00, +0.00] & -0.33 [-0.33, -0.33] & 0.312 \\"


def test_the_reference_policies_print_on_every_setting_need_no_model_and_read_the_pass_floor(tmp_path):
    """Doing nothing scores below every policy that passes, the reading of the caption, while the oracle stays the best of the five reference policies."""
    runs = tmp_path / "runs"
    _policy_runs(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    assert len(tables["POLICIES"]) == len(LADDER) * len(REFERENCE_POLICIES)
    assert "## No model yet" in page and _table(page, "agents")[0][1:] == ["no run"] * 6
    assert _lines(page, "policies")[0] == ["Setting", "Oracle pass (%)", *(f"{name} reward" for _, name in REFERENCE_POLICIES)]
    assert _table(page, "policies") == [[setting, _same(100.0, ".1f"), *(_same(REWARD[label] - p) for label, _ in REFERENCE_POLICIES)]
                                        for p, setting in enumerate(LADDER_NAMES)]
    assert "doing nothing must score below every policy that passes, on every initial state of every row." in page and "floor" not in page.split("(`tab:policies`)")[1].split("\n#### ")[0]
    tex = _tex(_latex(results), "policies")
    assert tex["header"] == ["Setting & " + " & ".join(_two_lines(name, metric, "c") for name, metric in [("Oracle", "pass"), *((name, "reward") for _, name in REFERENCE_POLICIES)]) + r" \\"]
    assert tex["body"][5] == r"+ deterioration and failure & 100.0 & 5.00 & 3.00 & -1.00 & -5.00 & -3.00 \\"


def test_a_fold_without_an_adapter_reads_the_runs_of_no_training_and_its_cells_carry_a_dagger(tmp_path):
    """The CausalDiagnosis agent with the MCP kept too few episodes on fold 1, so no adapter was trained there and the
    training pipeline wrote a marker run for each variant that needs it, run.json with adapter.trained false and the gate, no episode.
    Fine-tuning, its control and fine-tuning + GEPA then read the runs of no training on the episodes of fold 1, the test episodes of the
    model that holds out fold 1, their cells carry a dagger, their differences against no training on that fold are zero, and the budget
    table prints the gate of the marker. Fold 2 trained its adapters as usual."""
    runs = tmp_path / "runs"
    _training_runs(runs, (CD_MCP,), (AS_RELEASED, "gepa", "gepa_score_only"))
    for variant in ("ft", "ft_gepa", "ft_unfiltered"):
        trained_fold = 2  # the model that holds out fold 2 plays the starts of fold 2 and trained its adapter
        _run(runs, f"train_cd_{variant}_f{trained_fold}", [_episode(s, CD_MCP, TRAINING_PRESET, block=COMBINATIONS[s], **{"pass": True})
                                                          for s in SEEDS if FOLD_OF_SEED[s] == trained_fold],
             _meta(TRAINING_PRESET, **{TRAINING_VARIANT_KEY: variant, TRAINING_FOLD_KEY: trained_fold}))
        _run(runs, f"marker_cd_{variant}_f1", [], _meta(TRAINING_PRESET, **{TRAINING_VARIANT_KEY: variant, TRAINING_FOLD_KEY: 1, "agent_label": f"causal_{MODEL}",
             "diagnostics_server": True, "adapter": {"name": None, "trained": False, "reason": "below_minimum", "alias": AS_RELEASED, "fold": 1, "gate": _gate(1, False)}}))
    tables, page, results = _aggregated(tmp_path, runs)
    training = {(r["training_variant"], r["subset"]): r for r in tables["TRAINING"] if r["agent"] == f"causal_{MODEL}" and r["diagnostics_server"] in (True, "True")}
    marked = [s for s in SEEDS if FOLD_OF_SEED[s] == 1]  # the test episodes of the model that holds out fold 1, read as released
    assert training[("ft", "all")]["aliased"] == len(marked) and training[("ft", "all")]["n"] == len(SEEDS) and training[("gepa", "all")]["aliased"] == 0
    rows = {row[0]: row for row in _table(page, "training")[14:]}  # the group of the CausalDiagnosis agent, the last of three groups of seven
    assert all(cell.endswith(main_tables.DAGGER) for variant in ("fine-tuning", "fine-tuning + GEPA", "fine-tuning on unfiltered trajectories") for cell in rows[f"+ {variant}"][4:])
    assert not any(main_tables.DAGGER in cell for cell in rows["+ GEPA"])
    pairs = {(r["training_variant"], r["reference_variant"], r["subset"]): r for r in tables["TRAINING_pairs"] if r["agent"] == f"causal_{MODEL}"}
    # the model that holds out fold 1 is tested on the episodes of fold 1, where every variant that needs the adapter reads no training
    assert pairs[("ft", "as_released", "fold_1")]["pass_diff"] == pairs[("ft", "ft_unfiltered", "fold_1")]["pass_diff"] == 0.0
    assert pairs[("ft_gepa", "as_released", "fold_1")]["diagnosis_all_correct_diff"] == 0.0
    record = next(r for r in tables["TRAINING_RECORD"] if r["agent"] == f"causal_{MODEL}" and r["fold"] == 1)
    assert (record["gate_trained"], record["gate_kept"], record["gate_rounds_sampled"]) == (False, 40, 1)
    assert r"$^\dagger$" in _tex(_latex(results), "training")["body"][18] and r"$^\dagger$" not in _tex(_latex(results), "training")["body"][16]
    # tab:thinking reads the trained variants through the same rule
    thinking = {r["training_variant"]: r for r in tables["THINKING"] if r["agent"] == f"causal_{MODEL}" and r["config"] == TRAINING_PRESET
                and r["diagnostics_server"] in (True, "True")}
    assert thinking["ft"]["aliased"] == len(marked) and thinking["gepa"]["aliased"] == 0 and thinking[AS_RELEASED]["aliased"] == 0
    lines = {tuple(row[:2]): row for row in _table(page, "thinking")}
    assert all(cell.endswith(main_tables.DAGGER) for variant in ("fine-tuning", "fine-tuning + GEPA", "fine-tuning on unfiltered trajectories")
               for cell in lines[(NAMES[5], variant)][2:])
    assert not any(main_tables.DAGGER in cell for cell in lines[(NAMES[5], "GEPA")])
    other = tmp_path / "other"
    _training_runs(other, (CD_MCP,), (AS_RELEASED,))
    _run(other, "train_cd_ft_f1", [_episode(s, CD_MCP, TRAINING_PRESET, block=COMBINATIONS[s]) for s in SEEDS if FOLD_OF_SEED[s] == 1],
         _meta(TRAINING_PRESET, **{TRAINING_VARIANT_KEY: "ft", TRAINING_FOLD_KEY: 1}))
    _run(other, "marker_cd_ft_f1", [], _meta(TRAINING_PRESET, **{TRAINING_VARIANT_KEY: "ft", TRAINING_FOLD_KEY: 1, "agent_label": f"causal_{MODEL}",
         "diagnostics_server": True, "adapter": {"trained": False, "gate": _gate(1, False)}}))
    with pytest.raises(ValueError, match="trained no adapter on fold 1, while an evaluation run of ft on that fold exists"):
        _aggregated(tmp_path / "o", other)


def test_the_budget_table_reads_run_json_of_the_trained_runs_per_fold_and_says_no_record_without_one(tmp_path):
    """The budget table reads what the training pipeline records in run.json of the trained evaluation runs, the adapter of a fine tuned
    run with its gate and the task block of a GEPA run (RECORDS), one line per agent and fold of the cross-fitting: a line
    without a trained run says no record, and a field that none of its trained runs carries says none. The optimizer steps stay in
    TRAINING_RECORD.csv and leave the table, as the paper prints it."""
    runs = tmp_path / "runs"
    _run(runs, FOLDERS[CD_MCP], [_episode(s, CD_MCP) for s in SEEDS], _meta())
    _record_runs(runs)
    tables, page, results = _aggregated(tmp_path, runs)
    assert [(r["agent"], r["fold"]) for r in tables["TRAINING_RECORD"]] == [(f"{agent}_{MODEL}", fold) for agent in ("causal", "plan_execute") for fold in TRAINING_FOLDS]
    assert _lines(page, "budget")[0] == ["Agent", "Fold", *(" ".join(head) for head in BUDGET_HEADS)]
    assert _table(page, "budget") == BUDGET_BY_HAND
    assert tables["TRAINING_RECORD"][2]["optimizer_steps"] == 30
    tex = _tex(_latex(results), "budget")
    assert tex["groups"] == [2, 2, 2]
    assert tex["body"][:2] == [rf"{NAMES[1]} & 1 & 1 & 1190 & 300 & 68 & 900 & 12 \\", r" & 2 & 2 & 2380 & 300 & 68 & 900 & 12 \\"]
    assert tex["header"] == ["Agent & Fold & " + " & ".join(_two_lines(a, b, "c") for a, b in BUDGET_HEADS) + r" \\"]
    assert (tables["TRAINING_RECORD"][2]["gate_minimum"], tables["TRAINING_RECORD"][2]["gate_trained"]) == (70, True)
    lines = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))["main_tables"]["lines"]
    records = _csv(results / "tables" / "TRAINING_RECORD.csv")
    assert sorted((records[line["row"] - 1]["agent"], records[line["row"] - 1]["fold"], line["csv"]) for line in lines if line["table"] == "budget") == [
        (f"{agent}_{MODEL}", str(fold), "TRAINING_RECORD.csv") for agent in ("causal", "plan_execute") for fold in TRAINING_FOLDS]
    assert sorted(line["line"] for line in lines if line["table"] == "budget") == sorted(f"{name}, fold {fold}" for name in (NAMES[1], NAMES[5]) for fold in TRAINING_FOLDS)


def test_the_training_episodes_through_a_test_combination_print_per_agent_and_fold_one_column_per_trained_variant(tmp_path):
    """Under repair-closed training only the random hazard of deterioration carries a training episode into a
    combination that holds a fault of the held-out fold, and the table beside the budget table prints these episodes for every trained variant, the stream of
    each loop of GEPA, the kept episodes of fine-tuning and the episodes of its control, from run.json of the trained runs (RECORDS), in the
    order of the paper and with the agent named on the line of fold 1."""
    runs = tmp_path / "runs"
    _training_runs(runs, (PE_MCP, CD_MCP), VARIANTS[1:])
    tables, page, results = _aggregated(tmp_path, runs)
    assert list(VISIT_FIELDS) == list(VARIANTS[1:]) and VISIT_FIELDS["ft"] == VISITS
    assert _lines(page, "hazard")[0] == ["Agent", "Fold", *(VARIANT_NAMES[variant] for variant in VISIT_FIELDS)]
    assert _table(page, "hazard") == HAZARD_BY_HAND
    pe = next(r for r in tables["TRAINING_RECORD"] if r["agent"] == f"plan_execute_{MODEL}" and r["fold"] == 1)
    assert (pe["control_model_calls_as_examples"], pe["ft_gepa_candidates"], pe["control_trajectories"]) == (950, 9, 120)
    tex = _tex(_latex(results), "hazard")
    assert tex["groups"] == [2, 2, 2] and tex["body"][:2] == [rf"{NAMES[1]} & 1 & 1 & 8 & 3 & 4 & 5 \\", r" & 2 & 2 & 9 & 0 & 6 & 6 \\"]
    assert tex["header"] == ["Agent & Fold & GEPA & " + _two_lines("GEPA on the", "score alone", "c") + " & fine-tuning & " + _two_lines("fine-tuning", "+ GEPA", "c")
                             + " & " + _two_lines("fine-tuning on", "unfiltered trajectories", "c") + r" \\"]
    lines = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))["main_tables"]["lines"]
    assert sorted(line["line"] for line in lines if line["table"] == "hazard") == sorted(f"{name}, fold {fold}" for name in (NAMES[1], NAMES[5]) for fold in TRAINING_FOLDS)


STRATA_BY_HAND = (  # success and declared of the rows of _six_rows on the strata of the separability labels: starts 0 and 1, 0 and 2, and 3 and 4
    ["0.00", "0.00", "0.00", "0.00", "0.00", "0.00"],
    ["0.50", "0.00", "0.50", "0.00", "0.50", "0.50"],
    ["1.00", "0.00", "0.50", "0.00", "0.00", "0.00"],
    ["1.00", "0.00", "1.00", "0.00", "0.50", "0.50"],
    ["1.00", "0.00", "1.00", "0.00", "0.50", "0.00"],
    ["1.00", "0.00", "1.00", "0.00", "1.00", "0.00"],
)


def test_the_strata_print_success_and_declared_on_the_strata_of_the_two_separability_labels(tmp_path):
    runs = tmp_path / "runs"
    _six_rows(runs)
    _, page, results = _aggregated(tmp_path, runs)
    assert _lines(page, "strata")[0] == ["Agent", *(f"{name} {part}" for name in ("Label separable", "E18 separable", "Both inseparable") for part in ("success", "declared"))]
    rows = _table(page, "strata")
    assert [row[0] for row in rows] == list(NAMES) and [[cell.split(" [")[0] for cell in row[1:]] for row in rows] == list(STRATA_BY_HAND)
    assert _tex(_latex(results), "strata")["body"] == [" & ".join([name, *cells]) + r" \\" for name, cells in zip(NAMES, STRATA_BY_HAND)]
    lines = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))["main_tables"]["lines"]
    strata = _csv(results / "tables" / "STRATA.csv")
    behind = {line["line"]: strata[line["row"] - 1] for line in lines if line["table"] == "strata"}
    assert ({line: (r["agent"], r["diagnostics_server"], r["stratum"], r["n"]) for line, r in behind.items()}
            == {f"{name}, {stratum}": (f"{prefix}_{MODEL}", str(server), stratum, "2") for prefix, server, name in AGENT_ROWS
                for stratum in ("separable", "separable_e18", "inseparable_both")})


OFFLINE_HEADER = ("prehistory,readings,episodes,top_mass,top1_rate,exact_wrong,e14_covers_truth,e17_rejects_wrong_repair,wrong_repairs,main_effects_top_mass,"
                  "main_effects_top1_rate,calibration_gap,main_effects_exact_wrong,main_effects_calibration_gap,fewer_wrong,gap_two_sided,gap_one_sided")


def _offline(results, rows=None):
    """offline_mcp.csv in a folder in the format of CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py, header alone without rows; paper_tables/
    reads the one in tmp_path / "results" (_offline_in_the_results_of_the_tests)."""
    results.mkdir(parents=True, exist_ok=True)
    (results / "offline_mcp.csv").write_text("\n".join([OFFLINE_HEADER, *(rows or [])]) + "\n", encoding="utf-8")


def test_the_mcp_offline_prints_the_rows_of_offline_mcp_csv_with_the_five_recordings_served(tmp_path):
    """The offline table of the MCP comes from offline_mcp.csv, which the cluster writes after the grid: the rows with the start's five
    recordings served, one line per readings count, the exact wrong episodes as a count, an empty E17 cell as none, and on the page the
    episodes and the wrong repairs behind the rates. The rows without the recordings stay in the file, and the index names each row read."""
    (tmp_path / "runs").mkdir()
    lines = [f"{prehistory},{readings},470,{0.5 + readings / 100:.6g},{0.4 + readings / 100:.6g},{20 - readings},0.9,{'' if readings == 10 else 0.25},{0 if readings == 10 else 8},0.3,0.2"
             for prehistory in (5, 0) for readings in main_tables.OFFLINE_READINGS]
    _offline(tmp_path / "results", lines)
    _, page, results = _aggregated(tmp_path, tmp_path / "runs")
    assert "No run has been played" not in page
    assert _lines(page, "mcpoffline")[0] == ["Live readings", "Predicted top-1 probability", "Top-1 rate", "Exact wrong", "E14 prediction interval covers the truth",
                                             "E17 rejects a wrong repair"]
    assert _tex(_latex(results), "mcpoffline")["header"] == ["Live readings & " + _two_lines("Predicted top-1", "probability", "c") + " & Top-1 rate & Exact wrong & "
                                                             + _two_lines("E14 prediction interval", "covers the truth", "c") + " & "
                                                             + _two_lines("E17 rejects", "a wrong repair", "c") + r" \\"]
    expected = [[str(r), f"{0.5 + r / 100:.2f}", f"{0.4 + r / 100:.2f} of 470 episodes", str(20 - r), "0.90", "none" if r == 10 else "0.25 of 8 wrong repairs"]
                for r in main_tables.OFFLINE_READINGS]
    assert _table(page, "mcpoffline") == expected
    assert _tex(_latex(results), "mcpoffline")["body"] == _latex_lines("mcpoffline", expected)
    cited = [line for line in json.loads((results / "INDEX.json").read_text(encoding="utf-8"))["main_tables"]["lines"] if line["table"] == "mcpoffline"]
    assert [(line["line"], line["csv"], line["row"]) for line in cited] == [(str(r), "offline_mcp.csv", i) for i, r in enumerate(main_tables.OFFLINE_READINGS, 1)]
    _offline(tmp_path / "results")
    (tmp_path / "header_only").mkdir()
    shutil.copyfile(results / "INDEX.json", tmp_path / "header_only" / "INDEX.json")
    main_tables.write(results / "tables", tmp_path / "header_only", tmp_path / "header_only" / "INDEX.json")
    alone = (tmp_path / "header_only" / "MAIN_TABLES.md").read_text(encoding="utf-8")
    assert all(row[1:] == ["no run"] * 5 for row in _table(alone, "mcpoffline"))


@pytest.mark.offline_file
def test_the_offline_table_reads_the_file_name_the_header_and_the_readings_counts_that_the_server_writes():
    """paper_tables/ copies the names of CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py, the file and its place, its header and the readings counts, so
    that it stays clear of the server package, and this test holds the copies to the writer once the stage of the server has landed.
    The five recordings of the rows it prints are the segment of the evidence module."""
    offline_table = pytest.importorskip("CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.offline_table")
    from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT
    assert offline_table.RESULTS_PATH.name == main_tables.OFFLINE and offline_table.COLUMNS == tuple(OFFLINE_HEADER.split(","))
    assert offline_table.RESULTS_PATH.resolve() == main_tables.read_sources.OFFLINE_PATH.resolve()  # the place it reads, as it is (mark offline_file)
    assert offline_table.READINGS == main_tables.OFFLINE_READINGS and SEGMENT == main_tables.OFFLINE_PREHISTORY


def test_verify_results_rebuilds_from_the_offline_table_the_pages_read_and_names_a_changed_one(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, FOLDERS[CD_MCP], [_episode(s, CD_MCP) for s in SEEDS], _meta())
    _offline(tmp_path / "results", [f"5,{r},470,0.6,0.5,3,0.9,0.25,8,0.3,0.2" for r in main_tables.OFFLINE_READINGS])
    _, page, results = _aggregated(tmp_path, runs)
    assert _table(page, "mcpoffline")[0][1:3] == ["0.60", "0.50 of 470 episodes"]
    assert verify_results.verify(results, [runs])[1]["tables"][1] == []
    offline = results / "offline_mcp.csv"
    offline.write_text(offline.read_text(encoding="utf-8").replace("0.6,", "0.7,"), encoding="utf-8")
    assert "MAIN_TABLES.md: the text differs" in verify_results.verify(results, [runs])[1]["tables"][1]


def test_a_read_that_writes_its_pages_elsewhere_prints_the_offline_table_of_the_results_folder(tmp_path):
    """paper_tables/ reads the offline table in the results folder, where CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py writes it, and
    never in the folder the pages go to, so a read that writes its pages elsewhere prints the offline table of a read in place. The
    control puts a header alone beside the pages of the other read, and the table stays the one of the results folder. Verify
    rebuilds such a read from the same file: it passes, and names MAIN_TABLES.md once the row of the results folder changes, where
    a rebuild that read the header beside the pages would name nothing."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _offline(tmp_path / "results", ["5,0,470,0.6,0.5,3,0.9,0.25,8,0.3,0.2"])
    _, in_place, _ = _aggregated(tmp_path, runs)
    _offline(tmp_path / "elsewhere")
    aggregate.aggregate(runs, tmp_path / "elsewhere" / "tables", resamples=50, fraction="1")
    elsewhere = (tmp_path / "elsewhere" / main_tables.PAGE).read_text(encoding="utf-8")
    assert _table(in_place, "mcpoffline")[0][1:3] == ["0.60", "0.50 of 470 episodes"]
    assert _table(elsewhere, "mcpoffline") == _table(in_place, "mcpoffline")
    assert verify_results.verify(tmp_path / "elsewhere", [runs])[1]["tables"][1] == []
    offline = tmp_path / "results" / "offline_mcp.csv"
    offline.write_text(offline.read_text(encoding="utf-8").replace("0.6,", "0.7,"), encoding="utf-8")
    assert "MAIN_TABLES.md: the text differs" in verify_results.verify(tmp_path / "elsewhere", [runs])[1]["tables"][1]


# ----- the rows of the main comparison, the models, the layout, the LaTeX form, the page before any run, the index and the rebuild -----

def test_an_ablation_line_and_another_clock_mode_stay_out_of_the_tables_of_the_main_comparison(tmp_path):
    runs = tmp_path / "runs"
    _six_rows(runs)
    _run(runs, "guidance1", [_episode(s, CD, guidance_level=1, **{"pass": True}) for s in SEEDS], _meta())
    _run(runs, "instant", [_episode(s, CD, clock_mode="instant", **{"pass": True}) for s in SEEDS], _meta())
    tables, page, _ = _aggregated(tmp_path, runs)
    assert len(tables["AGENTS"]) == 8
    assert _table(page, "agents")[_line_index(CD, AGENTS_PRESET)][5].startswith(f"80.0 {PM} ") and f"{MODEL}, 5 episodes per line." in _caption(page, "agents")


def test_a_row_without_a_run_says_so_and_the_episodes_of_the_caption_name_a_range(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, FOLDERS[CD_MCP], [_episode(s, CD_MCP) for s in SEEDS], _meta())
    _run(runs, FOLDERS[PE], [_episode(s, PE) for s in range(3)], _meta())
    _, page, _ = _aggregated(tmp_path, runs)
    rows, ours = _table(page, "agents"), _line_index(CD_MCP, AGENTS_PRESET)
    assert all(row[1:] == ["no run"] * 6 for i, row in enumerate(rows) if i not in (0, ours)) and rows[0][4] == f"100.0 {PM} 0.0"
    # a row with a run says none where its episodes hold no value, as for pass^k on starts that each hold a combination of their own,
    # for the model calls and the unmatched evidence here and for the strata of a start without a label, while an empty verified outcome
    # of a played episode counts as a failure, as the success does here
    assert rows[ours][1:] == ["no judge"] * 3 + [f"100.0 {PM} 0.0", f"100.0 {PM} 0.0", "none"]
    assert _table(page, "agentmeasures")[ours][2:6] == [f"0.0 {PM} 0.0"] * 3 + ["none"] and _table(page, "judgeagree")[5][-2] == "none"
    assert _table(page, "strata")[5][1:] == ["none"] * 6 and _table(page, "strata")[1][1:] == ["no run"] * 6
    assert f"{MODEL}, 3 to 5 episodes per line." in _caption(page, "agents")


def test_the_tables_repeat_per_model_with_granite_first_and_name_a_second_episode_set(tmp_path):
    runs = tmp_path / "runs"
    _six_rows(runs)
    _run(runs, "alpha", [_row(s, agent="causal_alpha", model="alpha", config=AGENTS_PRESET, clock_mode="declared", diagnostics_server=True,
                              ranking_model=DEFAULT_RANKING_MODEL, **{"pass": True}) for s in SEEDS], _meta(model="alpha"))
    _run(runs, "seeds", [_episode(s, CD_MCP, guidance_level=1) for s in SEEDS], _meta(starts="combinations"))
    _, page, results = _aggregated(tmp_path, runs)
    assert page.index(f"## {MODEL}\n") < page.index("## alpha\n") and page.count("(`tab:agents`)\n") == 2
    alpha = page.split("## alpha\n")[1]
    assert _table(alpha, "agents")[_line_index(CD_MCP, AGENTS_PRESET)][5] == f"100.0 {PM} 0.0" and _table(alpha, "agents")[0][5] == "no run"
    assert "The episode sets of the runs of the six rows are combinations and evaluation, master seed 42." in page
    assert "The episode set of the runs of the six rows is evaluation, master seed 42." in alpha
    assert _latex(results).count(r"\clearpage") == 1


TITLES = {"agents": "Agents, the CausalDiagnosis MCP and the CausalDiagnosis environment", "agentmeasures": "The measures of the agents",
          "hyp": "The hypotheses H1 and H2", "ladder": "The cumulative layer study", "training": "Training in the environment", "efficiency": "Execution efficiency",
          "exactmatch": "Exact match and its majority-class baseline", "thinking": "Thinking time charged and where its seconds went",
          "layers": "What each big layer changes", "misdiagnosis": "The misdiagnosis price at the budget",
          "fullenv": "The full environment against the base layer", "policies": "The reference policies on the layers",
          "policylayers": "What each big layer changes for the reference policies", "trainhyp": "The training hypotheses",
          "trainfolds": "The training hypotheses by fold", "budget": "What each agent trained on", "hazard": "Training episodes through a test combination",
          "mcptop": "The hypotheses H2' on the top layer", "judgerows": "The judge against the verified pass, per row",
          "judgecmp": "The judge against the verified pass, per comparison", "judgeagree": "The judge against four constructs of the verifier",
          "strata": "Success and declarations by stratum", "mcpoffline": "The CausalDiagnosis MCP offline", "mcpvalue": "What the CausalDiagnosis MCP adds and risks",
          "parallel": "The schedule of candidate plans", "parallelpair": "The schedule of candidate plans, on against off"}


def _expected_lines():
    """The line labels of every table of the page before any run, written out from the design in the order of the paper."""
    six = [[name] for name in NAMES]
    trained = [name for _, _, name in TRAINING_ROWS]
    comparisons = [[name, comparison] for name in trained for _, _, comparison in TRAINING_COMPARISONS]
    hypotheses = [[name, f"{row[2]} against {reference[2]}"] for name, row, reference in HYPOTHESES]
    return {"agents": _design_lines(), "agentmeasures": _design_lines(), "hyp": hypotheses,
            "ladder": [[name, setting] for _, _, name in LADDER_ROWS for setting in LADDER_NAMES],
            "training": [line for baseline, name in zip(TRAINING_BASELINES, trained)
                         for line in [[baseline[2]], *([name if variant == AS_RELEASED else f"+ {VARIANT_NAMES[variant]}"] for variant in VARIANTS)]],
            "efficiency": six, "exactmatch": six, "thinking": [line for group in _thinking_lines() for line in group],
            "layers": [[setting] for setting in LADDER_NAMES[1:]],
            "misdiagnosis": [[name, setting] for _, _, name in ABLATION_PLACES["L1"][1] for setting in ("Asset file", "Budget")],
            "fullenv": [[name, row[2]] for name, row in FULL_ENVIRONMENT],
            "policies": [[setting] for setting in LADDER_NAMES], "policylayers": [[name, setting] for _, name in REFERENCE_POLICIES for setting in LADDER_NAMES[1:]],
            "trainhyp": comparisons, "trainfolds": comparisons, "budget": [[name, str(fold)] for name in trained for fold in TRAINING_FOLDS],
            "hazard": [[name, str(fold)] for name in trained for fold in TRAINING_FOLDS],
            "mcptop": [[name, f"{row[2]} against {reference[2]}"] for name, row, reference in H2_PRIME], "judgerows": six, "judgecmp": hypotheses,
            "judgeagree": six, "strata": six, "mcpoffline": [[str(r)] for r in main_tables.OFFLINE_READINGS],
            "mcpvalue": [[setting, name] for setting in ("All six big layers", "Layer 0") for name in trained],
            "parallel": [[name] for name in main_tables.SWITCH_NAMES.values()] + [[row[2]] for row in PARALLEL_CONTRAST_ROWS],
            "parallelpair": [["Pairs"], [f"{DELTA}Repair, declared cycles"], ["Exact p"], ["Holm p"], ["Reading"], [f"{DELTA}Pass"], ["Margin on Pass"],
                             [f"{DELTA}Cost / oracle"], ["Margin on cost"], ["Pairs without a cost"], ["Holds"], ["With a refuted repair"], ["Without a refuted repair"]]}


def _thinking_lines():
    """The groups of tab:thinking as line labels, written out from the design as the paper describes them: the rungs of the ladder in
    the generation_time mode with the three agents with the MCP, the preset of training with the three agents with it under each
    variant, and the arms of the first layer in that mode with the agents of its ablation place."""
    timed = [(preset, setting) for preset, setting in zip(LADDER, LADDER_NAMES) if aggregate.config_clock_mode(preset) == "generation_time"]
    place = f"L{LADDER.index(timed[0][0])}"
    arms = [setting for arm, (layer, setting) in ARM_PRESETS.items() if layer == place and aggregate.config_clock_mode(arm) == "generation_time"]
    return ([[name, setting] for _, _, name in LADDER_ROWS for _, setting in timed],
            [[name, VARIANT_NAMES[variant]] for _, _, name in TRAINING_ROWS for variant in VARIANTS],
            [[name, setting] for _, _, name in ABLATION_PLACES[place][1] for setting in arms])


def test_the_thinking_table_reads_the_split_of_the_charged_seconds_on_the_rows_that_charge_thinking_time(tmp_path):
    """On the first rung of the ladder that charges thinking time, every episode of the Plan-Execute row with the MCP charged 25 s,
    18 s of them generation and 7 s the rest, beside 15 s of a sleep or clock step that the charge never holds (test_aggregate's
    episode), so its line shows those means with no interval width, and every other line of the table says that no run exists. The
    rungs before it charge no thinking time and have no line."""
    runs = tmp_path / "runs"
    first = next(preset for preset in LADDER if aggregate.config_clock_mode(preset) == "generation_time")
    run = _run(runs, f"{first}_pe", [_episode(s, PE_MCP, first) for s in SEEDS], _meta(first))
    for s in SEEDS:
        _timed(run / f"seed{s:06d}", 1000.0, [1010.0, 1030.0, 1040.0], [10.0, 20.0, 10.0], [10.0, 5.0, 10.0],
               [(1001.0, 1006.0), (1012.0, 1020.0), (1018.0, 1025.0)])
    _, page, results = _aggregated(tmp_path, runs)
    rows = _table(page, "thinking")
    setting = LADDER_NAMES[LADDER.index(first)]
    line = next(row for row in rows if row[:2] == [NAMES[1], setting])
    assert line[3:] == [_same(25.0, ".1f"), _same(18.0, ".1f"), _same(15.0, ".1f"), _same(7.0, ".1f")]
    assert all(set(row[2:]) == {"no run"} for row in rows if row[:2] != [NAMES[1], setting])
    assert not any(row[1] in LADDER_NAMES[:LADDER.index(first)] for row in rows)
    tex = _tex(_latex(results), "thinking")
    assert any(cell.startswith(f"{NAMES[1]} & {setting} & ") and " & 25.0 & 18.0 & 15.0 & 7.0" in cell for cell in tex["body"])
    sentence = f"Over the lines of `tab:thinking`, at most 28.0% of the seconds charged for thinking fall outside the model's generation, on the line {NAMES[1]}, {setting}."
    assert sentence in page
    index = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))["main_tables"]["thinking_outside_generation"]
    assert index == {MODEL: {"line": f"{NAMES[1]}, {setting}", "share": 0.28, "sentence": sentence}}


def test_every_table_holds_the_rows_of_design_in_the_order_and_under_the_labels_of_the_paper(tmp_path):
    """The layout of the result tables of the paper: the tables of the body, one per experiment with the metrics of AssetOpsBench
    first and the measures of the agents beside the first, and the tables of the appendix, each under its label in the paper and in its
    order, the lines of design.AGENT_LINES under each design of design.AGENT_ROWS by the names of the paper, the three agents with the
    MCP on the six settings of the ladder, the seven hypotheses, the five layers, the reference policies, the three agents with the MCP
    under their training variants and comparisons and per fold, and the readings counts of the offline table, each in the order of
    experiment_tables.py. The page before any run shows them all."""
    assert NAMES == ("Plan-and-Execute", "Plan-and-Execute + CausalDiagnosis MCP", "Agents-as-Tools",
                     "Agents-as-Tools + CausalDiagnosis MCP", "CausalDiagnosis agent", "CausalDiagnosis agent + CausalDiagnosis MCP")
    assert [suffix for suffix, _, _ in AGENT_LINES] == ["", "+ CausalDiagnosis MCP", "+ CausalDiagnosis environment", "+ CausalDiagnosis MCP and environment"]
    assert LADDER_NAMES == ("Base", "+ priced reads and writes", "+ real-time clock", "+ advanced real-time clock", "+ repair rules", "+ deterioration and failure")
    assert [name for _, _, name in LADDER_ROWS] == [name for _, _, name in TRAINING_ROWS] == [NAMES[1], NAMES[3], NAMES[5]]
    assert [name for _, _, name in TRAINING_BASELINES] == [name for _, _, name in ABLATION_PLACES["L1"][1]] == [NAMES[0], NAMES[2], NAMES[4]]
    assert [VARIANT_NAMES[variant] for variant in VARIANTS] == ["no training", "GEPA", "GEPA on the score alone", "fine-tuning", "fine-tuning + GEPA",
                                                                "fine-tuning on unfiltered trajectories"]
    assert [name for _, _, name in TRAINING_COMPARISONS] == ["GEPA against no training", "GEPA against GEPA on the score alone", "fine-tuning against no training",
                                                             "fine-tuning + GEPA against no training", "fine-tuning against fine-tuning on unfiltered trajectories"]
    (tmp_path / "runs").mkdir()
    _, page, _ = _aggregated(tmp_path, tmp_path / "runs")
    sections = page.split("\n### ")[1:]
    order = (["agents", "agentmeasures", "hyp", "ladder", "training"],
             ["efficiency", "exactmatch", "thinking", "layers", "misdiagnosis", "fullenv", "policies", "policylayers", "trainhyp", "trainfolds", "budget", "hazard", "mcptop",
              "judgerows", "judgecmp", "judgeagree", "strata", "mcpoffline", "mcpvalue", "parallel", "parallelpair"])
    assert [(section.splitlines()[0], re.findall(r"^#### .* \(`tab:(\w+)`\)$", section, re.M)) for section in sections] == list(zip(HEADINGS, order))
    assert all(f"#### {TITLES[label]} (`tab:{label}`)" in page for label in LABELS) and list(TITLES) == list(LABELS) == [*order[0], *order[1]]
    expected = _expected_lines()
    assert list(expected) == list(LABELS)
    for label, labels in expected.items():
        assert [row[:LABELS[label]] for row in _table(page, label)] == labels, label
    headers = {"agents": ["Agent", "TC (%)", "DRA (%)", "RV (%)", "Top-1 (%)", "Pass (%)", f"Pass^{PASS_K} (%)"],
               "agentmeasures": ["Agent", "Exact match (%)", "Success (%)", "Wrong (%)", "Declared (%)", "Model calls", "Cost / oracle", "Cost", "Production loss",
                                 "Time to repair"],
               "hyp": ["", "Comparison", f"{DELTA}Top-1", "Holm p", f"{DELTA}Pass", "Holm p"],
               "ladder": ["Agent", "Setting", "TC (%)", "DRA (%)", "RV (%)", "Top-1 (%)", "Pass (%)", "Cost / oracle"],
               "training": ["Agent", "TC (%)", "DRA (%)", "RV (%)", f"Judge {K}", "Top-1 all (%)", "Top-1 single-fault (%)", "Top-1 two-fault (%)",
                            "Pass all (%)", "Pass single-fault (%)", "Pass two-fault (%)", "Unmatched"],
               "parallel": ["", "Repair, declared", "Repair with thinking", "Work orders", "Downtime", "Pass (%)", "Cost / oracle", "Model calls",
                            "Ranking calls", "Batches kept", "Gap explained"],
               "exactmatch": ["Agent", "Exact match", "Majority-class baseline", "Floor", "At floor"],
               "layers": ["Layer added", *(f"{DELTA}Pass {name}" for name in (NAMES[1], NAMES[3], NAMES[5])), f"pooled {DELTA}Pass", "Holm p"],
               "misdiagnosis": ["Agent", "Misdiagnosis price", "Top-1 (%)", "Pass (%)", "Wrong (%)", "Declared (%)", "Cycles read",
                                "Cost without the charge", "Charge"],
               "fullenv": ["", "Agent", f"{DELTA}Pass", "p", "Holm p", "Reading"],
               "policylayers": ["Policy", "Layer added", f"{DELTA}Pass", f"{DELTA}Success", f"{DELTA}Reward"],
               "trainhyp": ["Agent", "Comparison", f"{DELTA}Top-1, all episodes", "Holm p", f"{DELTA}Pass, all episodes", "Holm p", f"{DELTA}Top-1 single-fault",
                            f"{DELTA}Top-1 two-fault", f"{DELTA}Pass single-fault", f"{DELTA}Pass two-fault"],
               "trainfolds": ["Agent", "Comparison", f"{DELTA}Top-1 fold 1", f"{DELTA}Top-1 fold 2", f"{DELTA}Pass fold 1", f"{DELTA}Pass fold 2"],
               "budget": ["Agent", "Fold", *(" ".join(head) for head in BUDGET_HEADS)],
               "hazard": ["Agent", "Fold", "GEPA", "GEPA on the score alone", "fine-tuning", "fine-tuning + GEPA", "fine-tuning on unfiltered trajectories"],
               "thinking": ["Agent", "Setting", "Cycles charged", "Seconds charged", "Generation (s)", "Wall minus monotonic (s)", "Rest (s)"],
               "mcptop": ["", "Comparison", f"{DELTA}Pass", "Holm p", f"{DELTA}Reward", "Holm p"],
               "judgerows": ["Agent", f"{DELTA}Judge - verified", "Holm p", f"Judge {K}", "Discordance", "False pass", "False fail"],
               "judgecmp": ["", "Comparison", f"{DELTA}Judge - verified", "Holm p", f"{DELTA}Pass judge", f"{DELTA}Pass verified", "Reversed"],
               "judgeagree": ["Agent", f"{K} Pass", f"{K} Top-1", f"{K} Exact match", f"{K} Restored", "Cut", "Passed unrecovered", "Unmatched", "Restored"],
               "parallelpair": ["", "On less off"]}
    for label, header in headers.items():
        assert _lines(page, label)[0] == header, label


def test_the_latex_form_follows_the_page_cell_by_cell_with_intervals_only_where_the_paper_prints_them(tmp_path):
    runs = tmp_path / "runs"
    _every_table(runs)
    _offline(tmp_path / "results", [f"5,{r},470,0.6,0.5,3,0.9,0.25,8,0.3,0.2" for r in main_tables.OFFLINE_READINGS])
    _, page, results = _aggregated(tmp_path, runs)
    tex = _latex(results)
    assert tex.startswith("% generated by CausalDiagnosis/evaluation/result_tables/paper_tables.py") and tex.endswith("\\end{document}\n")
    assert tex.count(r"\begin{table}[H]") == tex.count(r"\caption{") == tex.count(r"\toprule") == tex.count(r"\bottomrule") == len(LABELS)
    assert re.findall(r"\\label\{tab:(\w+)\}", tex) == list(LABELS) and r"\setcounter" not in tex
    assert re.findall(r"\\textit\{([^}]*)\}", tex) == list(HEADINGS)
    # the groups of a table part by a rule, or by a small space in the two tables of the agents
    groups = {"agents": [len(AGENT_LINES)] * 3, "agentmeasures": [len(AGENT_LINES)] * 3, "hyp": [4, 3], "ladder": [6, 6, 6], "training": [7, 7, 7],
              "thinking": [len(group) for group in _thinking_lines()], "policylayers": [len(LADDER_NAMES) - 1] * len(REFERENCE_POLICIES), "trainhyp": [5, 5, 5],
              "trainfolds": [5, 5, 5], "budget": [2, 2, 2], "hazard": [2, 2, 2], "mcpvalue": [3, 3], "misdiagnosis": [2, 2, 2],
              "parallel": [len(main_tables.SWITCH_NAMES), len(PARALLEL_CONTRAST_ROWS)]}
    for label in LABELS:
        table = _tex(tex, label)
        assert table["body"] == _latex_lines(label, _table(page, label)), label
        assert {_width(line) for line in table["header"] + table["body"] if not line.startswith(r"\cmidrule")} == {_columns(table["format"])}, label
        assert table["groups"] == groups.get(label, [len(table["body"])]), label
    one = r"\,{\scriptsize$\pm$\,0.0}"
    assert rf"\quad + CausalDiagnosis MCP & 100.0{one} & 100.0{one} & 100.0{one} & " in tex
    assert (r"\caption{\textbf{Agents, the CausalDiagnosis MCP and the CausalDiagnosis environment.} Each design without and with the CausalDiagnosis "
            r"MCP, on the base layer and in the CausalDiagnosis environment with all six big layers on, granite\_m, 5 episodes per line.") in tex
    assert r"\textbf{granite\_m}" in tex and r"mean $\pm$ standard deviation" in tex
    assert "No run" not in tex and "count as false" not in tex and "majority per scenario" not in tex


def test_a_page_without_a_run_holds_no_number_and_every_cell_says_no_run_exists(tmp_path):
    """Before any run the page and its LaTeX form state that no run exists, and no number and no sentence reads as a
    result. The digits left on the page name the tables and the figure of AssetOpsBench, layer 0, the protocol of the judge, the
    hypotheses with their counts of tests, the top-1 column, the k of pass^k, the 95 percent intervals, E14, E17, E18, the five
    recordings and the readings counts of the offline table, and the two folds of the cross-fitting."""
    (tmp_path / "runs").mkdir()
    _, page, results = _aggregated(tmp_path, tmp_path / "runs")
    assert "No run has been played on the current code, so the tables below hold no number." in page
    assert "of the seconds charged for thinking fall outside" not in page
    assert json.loads((results / "INDEX.json").read_text(encoding="utf-8"))["main_tables"]["thinking_outside_generation"] == {"": None}
    assert "until the test set is played" in page and "## No model yet" in page
    for phrase in ("preliminary", "come from the runs", "episode set", "granite"):
        assert phrase not in page
    for label, labels in LABELS.items():
        rows = _table(page, label)
        assert rows and all(set(row[labels:]) == {"no record" if label in ("budget", "hazard") else "no run"} for row in rows), label
    readings = "|".join(str(r) for r in main_tables.OFFLINE_READINGS)
    stripped = re.sub(rf"^\| (?:{readings}) \|", "|", page, flags=re.M)
    stripped = re.sub(r"\| [12] \| no record", "| no record", stripped)
    stripped = re.sub(r"Tables? \d+(?: of AssetOpsBench)?|Figure 5|[Ll]ayer 0|temperature 0\.3|[Tt]op-1|H[12][a-d]?|the (?:14|30) tests|E1[478]|five recordings|"
                      r"fold [12]|below 5%|H2'[a-c]?|H0|95 percent|Pass\^\d+", "", stripped)
    assert not re.search(r"\d", stripped)
    tex = _latex(results)
    assert r"\textbf{No run has been played on the current code" in tex and "granite" not in tex
    bodies = {label: _tex(tex, label)["body"] for label in LABELS}
    expected = _expected_lines()
    assert {label: len(body) for label, body in bodies.items()} == {label: len(lines) for label, lines in expected.items()}
    # every cell but the labels of a line says that no run or no record exists, and the labels are those of the design
    allowed = {"no run", "no record", *(cell for label, lines in expected.items() for line in _latex_lines(label, lines)
                                       for cell in line.removesuffix(r" \\").split(" & "))}
    for label, body in bodies.items():
        for line in body:
            assert not set(line.removesuffix(r" \\").split(" & ")) - allowed, line
    index = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))
    assert index["main_tables"] == {"files": ["MAIN_TABLES.md", "main_tables.tex"], "lines": [], "thinking_outside_generation": {"": None}}


def test_the_results_index_names_the_row_behind_every_line(tmp_path):
    runs = tmp_path / "runs"
    _every_table(runs)
    _, _, results = _aggregated(tmp_path, runs)
    index = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))
    lines = index["main_tables"]["lines"]
    # the pair of the schedule of candidate plans names no row, since no run plays the switch off, and the offline table none without its file
    sources = {("agents", "JUDGE.csv"), ("agents", "AGENTS.csv"), ("agentmeasures", "AGENTS.csv"), ("hyp", "AGENTS_pairs.csv"), ("ladder", "JUDGE.csv"),
               ("ladder", "AGENTS.csv"), ("training", "TRAINING.csv"), ("efficiency", "EFFICIENCY.csv"), ("exactmatch", "AGENTS.csv"), ("thinking", "THINKING.csv"),
               ("layers", "LAYERS_pairs.csv"), ("misdiagnosis", "MISDIAGNOSIS_PRICE.csv"), ("fullenv", "FULL_ENV_pairs.csv"), ("policies", "POLICIES.csv"), ("policylayers", "POLICY_LAYERS.csv"),
               ("trainhyp", "TRAINING_pairs.csv"), ("trainfolds", "TRAINING_pairs.csv"), ("budget", "TRAINING_RECORD.csv"), ("hazard", "TRAINING_RECORD.csv"),
               ("strata", "STRATA.csv"), ("mcptop", "H2PRIME_pairs.csv"), ("judgerows", "H0.csv"), ("judgecmp", "H0.csv"), ("judgeagree", "AGENTS.csv"),
               ("judgeagree", "JUDGE.csv"), ("judgeagree", "RESTORED.csv"), ("mcpvalue", "MCP.csv"), ("parallel", "PARALLEL.csv")}
    assert {(line["table"], line["csv"]) for line in lines} == sources and {line["model"] for line in lines} == {MODEL}
    csvs = {name: _csv(results / "tables" / name) for _, name in sources}
    assert all(1 <= line["row"] <= len(csvs[line["csv"]]) for line in lines)

    def behind(table, line, csv_name=None):
        """The distinct rows that the index names behind a line of a table, in the order of their file."""
        return [csvs[name][row - 1] for name, row in sorted({(entry["csv"], entry["row"]) for entry in lines
                                                             if (entry["table"], entry["line"]) == (table, line) and entry["csv"] in (csv_name or entry["csv"],)})]

    ours, = behind("agents", NAMES[5], "AGENTS.csv")
    assert (ours["agent"], ours["diagnostics_server"], ours["config"], ours["clock_mode"]) == (f"causal_{MODEL}", "True", AGENTS_PRESET, "declared")
    number = next(line["row"] for line in lines if (line["table"], line["line"], line["csv"]) == ("agents", NAMES[5], "AGENTS.csv"))
    assert index["tables"]["AGENTS.csv"][number - 1]["episodes"] == [f"cd_mcp/seed{s:06d}" for s in SEEDS]
    judged, = behind("agents", NAMES[3], "JUDGE.csv")
    assert (judged["agent"], judged["diagnostics_server"], judged["judge_protocol"]) == (f"agent_as_tool_{MODEL}", "True", "paper")
    assert [(r["agent"], r["faults"]) for r in behind("efficiency", NAMES[0])] == [(f"plan_execute_{MODEL}", "single"), (f"plan_execute_{MODEL}", "two")]
    base, = behind("ladder", f"{NAMES[5]}, Base", "JUDGE.csv")
    assert (base["agent"], base["diagnostics_server"], base["config"]) == (f"causal_{MODEL}", "True", AGENTS_PRESET)
    h2b, = behind("hyp", "H2b")
    assert (h2b["hypothesis"], h2b["agent"], h2b["agent_server"], h2b["reference"], h2b["reference_server"]) == ("H2b", f"agent_as_tool_{MODEL}", "True", f"agent_as_tool_{MODEL}", "False")
    advanced, = behind("ladder", f"{NAMES[3]}, {LADDER_NAMES[3]}", "AGENTS.csv")
    assert (advanced["agent"], advanced["diagnostics_server"], advanced["config"], advanced["clock_mode"]) == (f"agent_as_tool_{MODEL}", "True", "advanced_real_time", "generation_time")
    assert [(r["agent"], r["config"], r["reference_config"]) for r in behind("layers", LADDER_NAMES[1])] == [(f"{row[0]}_{MODEL}", "priced", "base") for row in LADDER_ROWS] + [("pooled", "priced", "base")]
    assert [(r["agent"], r["config"]) for r in behind("policies", f"{LADDER_NAMES[5]}, Do nothing")] == [("do_nothing", "deterioration")]
    tuned, = behind("training", f"{NAMES[3]}, fine-tuning, two")
    assert (tuned["agent"], tuned["training_variant"], tuned["subset"]) == (f"agent_as_tool_{MODEL}", "ft", "two")
    control = behind("trainhyp", f"{NAMES[1]}, fine-tuning against fine-tuning on unfiltered trajectories")
    assert [(r["agent"], r["training_variant"], r["reference_variant"], r["subset"]) for r in control] == [(f"plan_execute_{MODEL}", "ft", "ft_unfiltered", subset)
                                                                                                          for subset in ("all", "single", "two")]
    by_fold = behind("trainfolds", f"{NAMES[1]}, fine-tuning against fine-tuning on unfiltered trajectories")
    assert [(r["agent"], r["training_variant"], r["reference_variant"], r["subset"]) for r in by_fold] == [(f"plan_execute_{MODEL}", "ft", "ft_unfiltered", subset)
                                                                                                          for subset in ("fold_1", "fold_2")]
    record, = behind("budget", f"{NAMES[3]}, fold 2")
    assert (record["agent"], record["fold"], record["episodes_sampled"]) == (f"agent_as_tool_{MODEL}", "2", "")
    visits, = behind("hazard", f"{NAMES[1]}, fold 1")
    assert (visits["agent"], visits["fold"], visits[VISIT_FIELDS["ft_unfiltered"]]) == (f"plan_execute_{MODEL}", "1", "5")
    stratum, = behind("strata", f"{NAMES[0]}, inseparable_both")
    assert (stratum["agent"], stratum["diagnostics_server"], stratum["stratum"]) == (f"plan_execute_{MODEL}", "False", "inseparable_both")
    assert "CausalDiagnosis/evaluation/result_tables/paper_tables/__init__.py" in index["code"]["rendering"] and index["code"]["rendering"]["CausalDiagnosis/evaluation/result_tables/paper_tables/__init__.py"]


def test_verify_results_rebuilds_the_two_files_and_names_a_changed_one(tmp_path):
    """The rebuild writes both forms again from the runs, the budget table among them from run.json of the trained runs."""
    runs = tmp_path / "runs"
    for folder in _six_rows(runs).values():
        _judged(folder)
    _record_runs(runs)
    tables, text, results = _aggregated(tmp_path, runs)
    assert _table(text, "budget")[0] == BUDGET_BY_HAND[0]
    assert verify_results.verify(results, [runs])[1]["tables"][1] == []
    page, tex = results / "MAIN_TABLES.md", results / "main_tables.tex"
    page.write_text(text.replace("(`tab:agents`)", "(`tab:agent`)"), encoding="utf-8")
    tex.write_text(tex.read_text(encoding="utf-8") + "%\n", encoding="utf-8")
    efficiency = results / "tables" / "EFFICIENCY.csv"
    efficiency.write_text(efficiency.read_text(encoding="utf-8").replace("two", "three", 1), encoding="utf-8")
    pairs = results / "tables" / "AGENTS_pairs.csv"
    pairs.write_text(pairs.read_text(encoding="utf-8") + "made_up\n", encoding="utf-8")
    problems = verify_results.verify(results, [runs])[1]["tables"][1]
    assert "MAIN_TABLES.md: the text differs" in problems and "main_tables.tex: the text differs" in problems
    assert "tables/EFFICIENCY.csv: row 2 column faults differs" in problems
    assert f"tables/AGENTS_pairs.csv: {len(tables['AGENTS_pairs']) + 1} rows against {len(tables['AGENTS_pairs'])} in the rebuild" in problems
    page.unlink()
    assert "MAIN_TABLES.md: missing in the results folder" in verify_results.verify(results, [runs])[1]["tables"][1]


def test_verify_results_rebuilds_the_budget_table_from_the_trained_runs_that_the_index_lists(tmp_path):
    """verify_results links the run folders that INDEX.json lists into a scratch root, the trained runs among them, whose run.json holds
    the record the budget table reads, so its rebuild matches without any file at the runs root."""
    runs = tmp_path / "runs"
    _run(runs, FOLDERS[CD_MCP], [_episode(s, CD_MCP) for s in SEEDS], _meta())
    _record_runs(runs)
    assert not [path for path in runs.iterdir() if path.is_file()]
    _, page, results = _aggregated(tmp_path, runs)
    assert _table(page, "budget")[4] == BUDGET_BY_HAND[4]
    assert verify_results.verify(results, [runs])[1]["tables"][1] == []


def test_the_aggregate_tables_beside_the_main_tables_keep_the_planned_steps_the_tokens_and_the_agreement_of_the_judge(tmp_path):
    """PLANS, CALLS and JUDGE_AGREEMENT hold the metrics of Tables 12 to 14 of AssetOpsBench, which no main table prints, by hand
    on two rows: Plan-Execute plans 4 steps on four starts and none on the fifth, the CausalDiagnosis agent plans 3 and 2 steps on every
    start, the two read 1,000 input tokens and write 100 and 200 output tokens, and the judge answers true in three of the five calls on
    task completion, whose ten pairs agree 3 + 1 times, and in all five on the other criteria."""
    runs = tmp_path / "runs"
    pe = _run(runs, FOLDERS[PE], [_episode(s, PE, plan_steps=0 if s == 4 else 4, input_tokens=1000, output_tokens=100, llm_calls=3) for s in SEEDS], _meta())
    cd = _run(runs, FOLDERS[CD], [_episode(s, CD, input_tokens=1000, output_tokens=200, llm_calls=7) for s in SEEDS], _meta())
    for folder in (pe, cd):
        _judged(folder, {**ALL_TRUE, "task_completion": 3})
    for s in SEEDS:
        _plans(cd / f"seed{s:06d}", 3, 2)
    tables, page, _ = _aggregated(tmp_path, runs)
    plans = {r["agent"]: r for r in tables["PLANS"]}
    planner = plans[f"plan_execute_{MODEL}"]
    assert set(plans) == {f"plan_execute_{MODEL}", f"causal_{MODEL}"} and plans[f"causal_{MODEL}"]["planned_steps"] == 5.0
    assert ((planner["planned_steps"], planner["planned_steps_std"], planner["planned_steps_min"], planner["planned_steps_max"], planner["planned_steps_zero"])
            == (3.2, round(statistics.stdev([4, 4, 4, 4, 0]), 6), 0, 4, 1))
    calls = {r["agent"]: r for r in tables["CALLS"]}
    assert (calls[f"plan_execute_{MODEL}"]["tokens"], calls[f"causal_{MODEL}"]["tokens"], calls[f"causal_{MODEL}"]["llm_calls"]) == (1100.0, 1200.0, 7.0)
    agreement, = tables["JUDGE_AGREEMENT"]
    assert (agreement["model"], agreement["config"], agreement["judge_protocol"], agreement["n"]) == (MODEL, AGENTS_PRESET, "paper", 10)
    assert (agreement["task_completion_agreement"], agreement["task_completion_uncertainty"], agreement["data_retrieval_accuracy_uncertainty"]) == (0.4, 0.6, 0.0)
    assert set(re.findall(r"^#### .* \(`tab:(\w+)`\)$", page, re.M)) == set(LABELS)


def test_the_table_of_the_judge_prints_its_kappa_against_pass_beside_the_unmatched_share(tmp_path):
    """Judge kappa, the agreement of the judge's majority pass with the verified pass, stands in the table of the judge against four
    constructs of the verifier as its kappa against Pass, and Unmatched, the share of episodes citing evidence the released evidence does
    not hold, beside Restored there, so the scores of the judge read beside how far the judge agrees with the verifier."""
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, read_rows
    from CausalDiagnosis.utils.csv_rows import write_rows

    runs = tmp_path / "runs"
    folders = _six_rows(runs)
    for folder in folders.values():
        _judged(folder)
    # Agent-As-Tool with the MCP passes the first three starts; its judge passes the first two, and the fifth start cites unmatched evidence
    rows = read_rows(folders[AAT_MCP])
    for s, row in enumerate(rows):
        row.update({"judge_pass": str(s < 2), "provenance_ok": str(s < 4)})
    write_rows(folders[AAT_MCP] / "scores.csv", rows, COLUMNS)
    tables, page, results = _aggregated(tmp_path, runs)
    head = _lines(page, "judgeagree")[0]
    assert head == ["Agent", f"{K} Pass", f"{K} Top-1", f"{K} Exact match", f"{K} Restored", "Cut", "Passed unrecovered", "Unmatched", "Restored"]
    lines = _table(page, "judgeagree")
    line = dict(zip(head, lines[AGENT_ROWS.index(AAT_MCP)]))
    # observed agreement 4/5, chance 2/5 x 3/5 + 3/5 x 2/5 = 12/25, so kappa (4/5 - 12/25) / (13/25) = 8/13
    assert line[f"{K} Pass"].startswith(f"{8 / 13:.2f} {PM} ") and line["Unmatched"].startswith(f"0.20 {PM} ")
    # the judge passes every start of the other rows, one side alone, so kappa has no value there, and no row recorded provenance
    assert all(row[1] == "none" and row[7] == "none" for i, row in enumerate(lines) if AGENT_ROWS[i] != AAT_MCP)
    agents = next(r for r in tables["AGENTS"] if r["agent"] == f"agent_as_tool_{MODEL}" and r["diagnostics_server"] == "True")
    judged = next(r for r in tables["JUDGE"] if r["agent"] == f"agent_as_tool_{MODEL}" and r["diagnostics_server"] == "True")
    assert (agents["judge_kappa"], agents["judge_kappa_n"], agents["unmatched_evidence"], judged["kappa_pass"]) == (round(8 / 13, 6), 5, 0.2, round(8 / 13, 6))
    assert "Cohen's kappa against Pass, Top-1, exact match and restored with no unnecessary order" in _caption(page, "judgeagree")
    tex = _tex(_latex(results), "judgeagree")
    assert tex["header"] == ["Agent & " + " & ".join([*(_two_lines(r"$\kappa$", name, "c") for name in ("Pass", "Top-1", "Exact match", "Restored")), "Cut",
                                                     _two_lines("Passed", "unrecovered", "c"), "Unmatched", "Restored"]) + r" \\"]
    assert tex["body"] == _latex_lines("judgeagree", lines)


def test_the_training_table_prints_the_judge_kappa_and_the_unmatched_share_over_every_test_episode(tmp_path):
    runs = tmp_path / "runs"
    _training_runs(runs)
    _baseline_runs(runs)
    tables, page, results = _aggregated(tmp_path, runs, resamples=TRAINING_RESAMPLES)
    head = _lines(page, "training")[0]
    assert head == ["Agent", "TC (%)", "DRA (%)", "RV (%)", "Judge \N{GREEK SMALL LETTER KAPPA}",
                    *(f"{name} {part} (%)" for name in ("Top-1", "Pass") for part in ("all", "single-fault", "two-fault")), "Unmatched"]
    assert all(len(row) == len(head) for row in _table(page, "training"))
    assert all({"judge_kappa", "judge_kappa_n", "unmatched_evidence"} <= set(row) for row in tables["TRAINING"])
    tex = _tex(_latex(results), "training")
    assert tex["header"][:4] == [r" & \multicolumn{3}{c}{Judge metrics (\%)} & & \multicolumn{7}{c}{Verified outcome} \\",
                                 r"\cmidrule(lr){2-4}\cmidrule(l){6-12}",
                                 r" & & & & & \multicolumn{3}{c}{Top-1} & \multicolumn{3}{c}{Pass} & \\", r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}"]


def test_the_families_of_h2_prime_and_h0_print_in_the_appendix_before_the_strata(tmp_path):
    """tab:mcptop reads H2PRIME_pairs and tab:judgerows and tab:judgecmp read H0. On the six rows of _all_or_none_rows with a judge
    that passes every start, the row test of Plan-Execute, which passes nothing, has J - V = 1 on every start, and H1a, whose arms the
    judge passes alike while the verifier passes only the CausalDiagnosis agent, has a judge difference of zero against a verified one of one,
    reversed being no since zero has no sign."""
    runs = tmp_path / "runs"
    _all_or_none_rows(runs)
    for row in AGENT_ROWS:
        _judged(runs / FOLDERS[row])
    tables, page, results = _aggregated(tmp_path, runs, resamples=50)
    rows = {line[0]: line for line in _table(page, "judgerows")}
    assert rows[NAMES[0]][1] == _same(1.0, "+.2f") and rows[NAMES[0]][5:] == ["1.00", "none"]
    h1a = _table(page, "judgecmp")[0]
    assert h1a[:2] == ["H1a", f"{HYPOTHESES[0][1][2]} against {HYPOTHESES[0][2][2]}"] and h1a[4:] == ["+0.00", "+1.00", "no"]
    assert all(line[2:] == ["no run"] * 4 for line in _table(page, "mcptop"))
    assert page.index("(`tab:judgecmp`)") < page.index("(`tab:strata`)")


def test_the_h2_prime_table_reads_only_columns_that_h2prime_pairs_holds():
    """Every key that tab:mcptop reads off a row of H2PRIME_pairs is a column that aggregate writes there, so no cell of the table
    reads a measure that design.H2_PRIME_MEASURES dropped and prints no value on every line. The recording row answers every key
    with a number, so each cell reads all the keys its form needs."""
    read = set()

    class RecordingRow(dict):
        def get(self, key, default=None):
            read.add(key)
            return "0.5"

        def __getitem__(self, key):
            return self.get(key)

    class OneRowSheet:
        model = "m"

        def first(self, table, **cells):
            assert table == "H2PRIME_pairs"
            return RecordingRow()

        def cite(self, table_label, line, table, row):
            return row

        def counted(self, table_label, line, labels, cells):
            return labels, cells

    main_tables.h2_prime_table(OneRowSheet())
    columns = set(next(table for table in aggregate.tables() if table.id == "H2PRIME_pairs").columns)
    assert {f"{measure}{part}" for measure in H2_PRIME_MEASURES for part in ("_diff", "_holm_p")} <= read
    assert read <= columns, sorted(read - columns)


def test_the_page_names_every_source_table_but_the_runs_table(tmp_path):
    """RUNS feeds the footnotes on the seeds and no cell, so the sentence that names the sources of the cells leaves it out by name,
    and it names the two tables of the families H2' and H0, which stand before RUNS in SOURCES."""
    _, page, _ = _aggregated(tmp_path, tmp_path / "no_runs")
    sentence = next(line for line in page.splitlines() if "a cell of `tables/" in line)
    assert all(f"`tables/{name}.csv`" in sentence for name in main_tables.SOURCES if name != "RUNS")
    assert "`tables/H0.csv`" in sentence and "`tables/H2PRIME_pairs.csv`" in sentence and "tables/RUNS.csv" not in sentence
    assert main_tables.SOURCES[-1] == "RUNS"


def test_the_caption_of_the_hypotheses_table_states_the_floor_without_a_decision_id(tmp_path):
    """The caption of tab:hyp states the floor rule of H1 at the registered share, in the page and in the LaTeX, and holds no
    decision id, since both are published."""
    _, page, results = _aggregated(tmp_path, tmp_path / "no_runs")
    caption = page.split("(`tab:hyp`)\n", 1)[1].split("\n| ", 1)[0]
    tex_caption = next(line for line in _latex(results).split("\\label{tab:hyp}", 1)[0].splitlines()[::-1] if line.startswith("\\caption{"))
    for text in (caption, tex_caption):
        assert "is at floor, and H1 against it is uninformative" in text and not re.search(r"\bD\d+\b", text), text
    assert "below 5%" in caption and "below 5\\%" in tex_caption


def test_the_floor_captions_follow_the_registered_floor_and_its_strict_rule(tmp_path, monkeypatch):
    """The caption of tab:hyp prints the floor that decides at_floor, the registered floor of pass, and the captions of tab:layers and
    tab:fullenv state the rule of floor_cells, a reference strictly below the floor."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

    load = registered.load
    monkeypatch.setattr(registered, "load", lambda *a, **k: {**load(*a, **k), "descriptive": {**load(*a, **k)["descriptive"], "floor_of_pass": 0.1}})
    _, page, _ = _aggregated(tmp_path, tmp_path / "no_runs")
    assert "lies below 10% is at floor" in _caption(page, "hyp")
    for label in ("layers", "fullenv"):
        assert "lies below the floor of Pass is uninformative" in _caption(page, label) and "does not lie above" not in _caption(page, label)
