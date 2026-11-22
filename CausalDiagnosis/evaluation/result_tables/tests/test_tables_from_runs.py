"""CausalDiagnosis/evaluation/result_tables/tables_from_runs/: the design of every table on a folder without runs, the builders on synthetic
run folders whose rows are made up here (no recorded value enters), the tables of the six rows of the agents and the
CausalDiagnosis MCP with the per episode fields they derive from the hidden log and the trajectory, the hypotheses of AGENTS_pairs
with the scoring pairs beside them, the tables of the environment and of training, and the tables whose runs do not exist yet,
which hold their columns."""
from CausalDiagnosis.utils.paths import RESULTS
import csv
import itertools
import json
import re
import shutil

import pytest

from fractions import Fraction

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.evaluation.result_tables import pooling_rules
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_bootstrap, cluster_proportion, ratio
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY

HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]
FAULTY_LIST = [3, 100, 0, 130]
PROFILE = get_profile(DEFAULT_ASSET, engine=False)


def _start(index):
    """The start of its own for each index, a combination of the levels of the asset in its component order, in JSON as scores.csv
    holds the start of an episode."""
    levels = [PROFILE.levels[c] for c in PROFILE.components]
    return json.dumps(list(next(itertools.islice(itertools.product(*levels), index, None))))


def _row(seed, **fields):
    row = {c: "" for c in COLUMNS}
    row.update({"seed": seed, "scenario_id": "", "config": "static", "agent": "plan_execute", "model": "m", "clock_mode": "declared",
                "guidance_level": 2, "diagnostics_server": True, "prices": "default", "verifier_scope": "writes",
                "ranking_model": "joint", "block": json.dumps(FAULTY_LIST), "n_faulty": 1, "pass": True, "cost": 10,
                "diagnosis_all_correct": True, "detection_delay": 0, "unnecessary": 0, "judge_pass": True, "abstained": False})
    row.update(fields)
    return row


def _run(root, name, rows, meta=None, hidden=True):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"label": name, "config_name": "static", "model": "m", "started": "2026-12-01T00:00:00",
                                                   "ended": "2026-12-01T01:00:00", "config": {"durations": {"read_per_cycle": 1}}, **(meta or {})}), encoding="utf-8")
    for row in rows:
        row["run"] = name
        ep = run_dir / f"seed{int(row['seed']):06d}"
        ep.mkdir()
        (ep / "episode.json").write_text("{}", encoding="utf-8")
        (ep / "report.md").write_text("# Episode\n", encoding="utf-8")
        if hidden:
            (ep / "hidden").mkdir()
            (ep / "hidden" / "oracle.json").write_text(json.dumps({"diagnosis": dict(zip(COMPONENTS, FAULTY_LIST)), "work_orders": [{"component": "cooler", "window_end": 60}]}), encoding="utf-8")
            (ep / "hidden" / "log.json").write_text(json.dumps([
                {"event": "reset"}, {"event": "work_order_created", "component": "cooler", "unnecessary": False},
                {"event": "work_order_created", "component": "pump", "unnecessary": True}]), encoding="utf-8")
            (ep / "trajectory.json").write_text(json.dumps([
                {"server": "hydraulic_rig", "tool": "read_sensors", "response": "{}", "error": None},
                {"server": "causal_diagnosis", "tool": "rank_root_causes", "error": None,
                 "response": json.dumps({"candidates": [{"combination": FAULTY_LIST, "probability": 0.8}]})},
                {"server": "hydraulic_rig", "tool": "read_sensors", "response": "{}", "error": None},
                {"server": "causal_diagnosis", "tool": "rank_root_causes", "error": None,
                 "response": json.dumps({"candidates": [{"combination": HEALTHY_LIST, "probability": 0.6}]})}]), encoding="utf-8")
    write_rows(run_dir / "scores.csv", rows, COLUMNS)
    return run_dir


def _timed(ep, started, called, wall, monotonic, spans):
    """The clock fields of the thinking split in an episode's state file, the task handed over at its start and no failed model
    attempt, so that the charge of each call is its monotonic gap, and model calls in its llm_calls.json."""
    (ep / "episode.json").write_text(json.dumps({"episode_started_at": started, "handover_wall": started, "called_at": called,
                                                 "think_seconds": monotonic, "think_seconds_wall": wall,
                                                 "think_seconds_monotonic": monotonic, "failed_model_seconds": 0.0}), encoding="utf-8")
    (ep / "llm_calls.json").write_text(json.dumps([{"prompt": "p", "started_at": s, "ended_at": t} for s, t in spans]), encoding="utf-8")


def test_thinking_splits_the_charged_seconds_into_generation_clock_step_and_rest(tmp_path):
    """The task is handed over at 1000 and the calls enter at 1010, 1030 and 1040, 40 wall seconds; the monotonic clock moves
    15 s less in the second gap, a sleep, which the charge on the monotonic clock never holds, so 25 s are charged; the model calls
    of 1001 to 1006 and of 1012 to 1020 and 1018 to 1025, which overlap and count once as 13 s, add 18 s of generation, and the
    call after the last tool call is charged to nothing; so the rest is 25 - 18 = 7 s. A second episode played before the split
    was recorded counts as unsplit, and a declared run has no row."""
    runs = tmp_path / "runs"
    timed = _run(runs, "pe_timed", [_row(s, agent="plan_execute_m", config="real_time", clock_mode="generation_time") for s in (0, 1)])
    _timed(timed / "seed000000", 1000.0, [1010.0, 1030.0, 1040.0], [10.0, 20.0, 10.0], [10.0, 5.0, 10.0],
           [(1001.0, 1006.0), (1012.0, 1020.0), (1018.0, 1025.0), (1045.0, 1050.0)])
    _run(runs, "pe_declared", [_row(s, agent="plan_execute_m", config="base") for s in (0, 1)])
    out = tmp_path / "out"
    aggregate.aggregate(runs, out, resamples=50, fraction="1")
    rows = _csv(out / "THINKING.csv")
    assert [(r["agent"], r["config"], r["clock_mode"]) for r in rows] == [("plan_execute_m", "real_time", "generation_time")]
    row = rows[0]
    assert (row["n"], row["unsplit"], row["seconds_charged_n"]) == ("2", "1", "1")
    assert [float(row[m]) for m in aggregate.THINKING_SPLIT] == [25.0, 18.0, 15.0, 7.0]
    assert aggregate.merged_spans([(3, 4), (1, 2), (1.5, 3)]) == [[1, 4]]


def test_thinking_leaves_generation_and_rest_empty_when_a_model_call_lacks_its_span(tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    _timed(ep, 0.0, [5.0], [5.0], [5.0], [(1.0, 2.0)])
    calls = json.loads((ep / "llm_calls.json").read_text())
    (ep / "llm_calls.json").write_text(json.dumps(calls + [{"prompt": "older"}]), encoding="utf-8")
    split = aggregate.thinking_split({"episode_dir": ep, "clock_mode": "generation_time"})
    assert split == {"seconds_charged": 5.0, "generation_seconds": None, "wall_minus_monotonic_seconds": 0.0, "rest_seconds": None}
    assert set(aggregate.thinking_split({"episode_dir": ep, "clock_mode": "declared"}).values()) == {None}


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_without_run_folders_every_table_states_its_design_with_empty_cells(tmp_path):
    out = tmp_path / "out"
    tables = aggregate.aggregate(tmp_path / "no_runs", out, resamples=50, fraction="1")
    assert [t.id for t, _ in tables] == ["A3", "A7", "A8", "A9", "A10", "A12", "MISDIAGNOSIS_PRICE", "SWITCHES", "GEPA", "GEPA_curve", "AGENTS", "AGENTS_pairs", "H1_GUIDANCE", "H2PRIME_pairs", "H0", "A13_A17", "CALLS", "THINKING", "STRATA", "JUDGE",
                                         "JUDGE_AGREEMENT", "EFFICIENCY", "PLANS", "ELIM", "ELIM_pairs", "MCP", "POLICIES", "LAYERS_pairs", "TRAINING", "TRAINING_pairs", "TRAINING_RECORD",
                                         "FUNNEL", "FAILURES", "SYSTEM_CONTRAST", "PAIRS_BY_FAULT", "POLICY_LAYERS", "FIRST_DROP", "HEALTHY_CONTROL", "RESTORED", "H1_TOP_pairs", "FULL_ENV_pairs", "PARALLEL", "PARALLEL_pairs", "LABELS", "RUNS"]
    assert all(rows == [] for _, rows in tables)
    text = (out / "aggregate.md").read_text(encoding="utf-8")
    assert "Run folders under `no_runs`: 0" in text and str(tmp_path) not in text
    # public text carries no clock time: the title states the date of the build alone
    assert re.fullmatch(r"# Ablation and result tables \(\d{4}-\d{2}-\d{2}\)", next(line for line in text.splitlines() if line.startswith("# ")))
    assert "No run has been played on the current code" in text and text.count("No episode contributes to this table yet.") == len(tables)
    for t, _ in tables:
        assert f"## {t.id}\n" in text and f"- Hypothesis: {t.hypothesis}" in text
        rows = _csv(out / f"{t.id}.csv")
        assert rows == [] and (out / f"{t.id}.csv").read_text(encoding="utf-8").strip() == ",".join(t.columns)
    assert _csv(out / "A12_curves.csv") == []
    readme = (out.parent / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Results of the hydraulic rig gym") and "No run has been played on the current code" in readme
    for page in (text, readme):  # a page written from no run must not read as if results existed
        assert "until the test set is played" in page and "come from the runs" not in page and "preliminary" not in page
    assert readme.count("\n## ") == len(tables) + 2 and "## Example episode" in readme and "## How to reproduce" in readme and "`MAIN_TABLES.md`" in readme
    assert "No episode has been played yet" in readme and all(f"Reading rule: {t.rule}" in readme for t, _ in tables)


def test_factor_tables_pair_episodes_and_correct_with_holm(tmp_path):
    """A9, the price table of En6 on the preset of training with the three agents without the MCP: the level full_tier_high is
    paired against the reference default on matched episodes, pass differs by -0.5 on average and the normalized cost by 0.4 over
    4 pairs, every row sits in one block so the interval of a mean has zero width, and a run with the MCP or at another guidance
    level enters no row."""
    runs = tmp_path / "runs"
    aob = {"servers": list(METERED_SERVERS), "config_name": design.TRAINING_PRESET}
    main = {"agent": "plan_execute_m", "config": design.TRAINING_PRESET, "clock_mode": load_config(design.TRAINING_PRESET)[1].clock_mode,
            "diagnostics_server": False, "ranking_model": DEFAULT_RANKING_MODEL}
    _run(runs, "default", [_row(s, cost_normalized=1.0, **main) for s in range(4)], aob)
    _run(runs, "high", [_row(s, prices="full_tier_high", cost=14, cost_normalized=1.4, **{"pass": s % 2 == 0}, **main) for s in range(4)], aob)
    _run(runs, "high_mcp", [_row(s, prices="full_tier_high", **{**main, "diagnostics_server": True}) for s in range(4)], aob)
    _run(runs, "high_g1", [_row(s, prices="full_tier_high", guidance_level=1, **main) for s in range(4)], aob)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200, fraction="1")}
    a9 = {row["prices"]: row for row in tables["A9"]}
    assert set(a9) == {"default", "full_tier_high"} and a9["default"]["n"] == a9["full_tier_high"]["n"] == 4
    assert a9["default"]["pass"] == 1.0 and a9["full_tier_high"]["pass"] == 0.5 and a9["full_tier_high"]["cost"] == 14.0
    # the level full_tier_high is compared with the reference default: pass differs by -0.5 on average, the normalized cost by 0.4
    assert a9["full_tier_high"]["pass_diff"] == -0.5 and a9["full_tier_high"]["cost_normalized_diff"] == 0.4 and a9["full_tier_high"]["pairs"] == 4
    assert a9["default"]["pass_diff"] == "" and a9["full_tier_high"]["holm_p"] != "" and a9["full_tier_high"]["reject"] in (True, False)
    assert a9["full_tier_high"]["cost_low"] == a9["full_tier_high"]["cost_high"] == 14.0
    assert {row["diagnostics_server"] for row in tables["A9"]} == {"False"}


def test_the_scope_abstention_and_ranking_tables_read_the_hidden_files(tmp_path):
    runs = tmp_path / "runs"
    rows = [_row(0, judge_pass=True, **{"pass": True}), _row(1, judge_pass=True, **{"pass": False}),
            _row(2, judge_pass=False, **{"pass": False}), _row(3, judge_pass=False, **{"pass": True}, abstained=True, abstention_correct=True)]
    writes = _run(runs, "writes", [{**row, "ranking_model": DEFAULT_RANKING_MODEL} for row in rows], {"judge_protocol": "paper"})
    # A12 reads the transcript that the servers write and a hidden log whose reset holds the start and whose two reads took a step
    for ep in sorted(writes.glob("seed*")):
        _transcript(ep, json.loads((ep / "trajectory.json").read_text(encoding="utf-8")))
        log = json.loads((ep / "hidden" / "log.json").read_text(encoding="utf-8"))
        reads = [{"event": "read", "step": k, "t_start": k, "t_end": k, "cost": 0} for k in (1, 2)]
        (ep / "hidden" / "log.json").write_text(json.dumps([{**log[0], "hidden": FAULTY_LIST}, *reads, *log[1:]]), encoding="utf-8")
    _run(runs, "all_scope", [_row(s, verifier_scope="all", judge_pass="", **{"pass": s == 0}) for s in range(4)], hidden=False)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=100, fraction="1")}
    assert tables["A7"] == []  # no run of the six rows on the preset of the agents table
    a8 = tables["A8"][0]
    assert a8["abstentions"] == 1 and a8["precision"] == 1.0 and a8["abstention_rule"] == "decision" and a8["label"] == "separability"
    assert len(tables["A8"]) == 1  # no grade cell holds an E18 record, so no row under that label
    a12 = {row["readings"]: row for row in tables["A12"]}
    # two ranking calls per episode: after one reading the truth at 0.8, after two readings a wrong candidate at 0.6
    assert set(a12) == {1, 2, "all"} and {r["config"] for r in tables["A12"]} == {"static"}
    assert a12[1]["calls"] == 4 and a12[1]["top_candidate_correct"] == 1.0 and a12[1]["brier"] == round(0.2 ** 2, 4)
    assert a12[2]["calls"] == 4 and a12[2]["top_candidate_correct"] == 0.0 and a12[2]["brier"] == round(0.6 ** 2, 4)
    assert a12["all"]["calls"] == 8 and a12["all"]["top_candidate_correct"] == 0.5 and a12["all"]["brier"] == round(((0.2 ** 2) + (0.6 ** 2)) / 2, 4)
    assert a12["all"]["brier_low"] != "" and a12["all"]["top1_high"] != ""
    curves = {(c["readings"], c["bin_lo"]): c for c in _csv(tmp_path / "out" / "A12_curves.csv")}
    assert curves[("1", "0.8")]["observed_frequency"] == "1.0" and curves[("1", "0.8")]["predicted_mean"] == "0.8" and curves[("1", "0.8")]["n"] == "4"
    assert curves[("2", "0.6")]["observed_frequency"] == "0.0" and curves[("all", "0.6")]["n"] == "4" and curves[("all", "0.8")]["bin_hi"] == "0.9"
    text = (tmp_path / "out" / "aggregate.md").read_text(encoding="utf-8")
    assert "No run has been played" not in text and "| plan_execute | m |" in text
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Models and agents: m." in readme and "Config presets: static." in readme and "Run folders: 2" in readme
    assert "## Example episode" in readme and "Episode `seed000000` of the run folder `writes`, which stays out of the repository" in readme
    assert "report.md](" not in readme and "published run folders" not in readme and "1. hydraulic_rig.read_sensors({})" in readme
    assert '"checks"' in readme


def test_runs_holds_one_row_per_run_folder_even_when_two_folders_share_a_label(tmp_path):
    runs = tmp_path / "runs"
    first = [_row(s, detection_delay=s, repair_delay=2 * s if s else "", judge_pass=s % 2 == 0, **{"pass": s < 3}) for s in range(4)]
    second = [_row(s, diagnostics_server=False, judge_pass="", cost_normalized=1.5) for s in range(4)]
    first[1]["error"] = "BadRequestError: Error code: 400, this model's maximum context length is 32768 tokens"  # a context overflow
    first[2]["error"] = "verifier: RuntimeError: the verifier could not read the episode"
    _run(runs, "same_label", first, {"starts": "combinations", "master_seed": None, "judge_model": "judge"})
    _run(runs, "same_label-2", second, {"starts": "sampled", "master_seed": 42})
    for row in second:  # the driver names the folder of a second run started in the same second with -2, never its label
        row["run"] = "same_label"
    write_rows(runs / "same_label-2" / "scores.csv", second, COLUMNS)
    (runs / "same_label" / "seed000000" / "llm_calls.json").write_text(json.dumps([{"input_tokens": 100}, {"input_tokens": 300}]), encoding="utf-8")
    (runs / "same_label" / "seed000001" / "llm_calls.json").write_text(json.dumps([{"input_tokens": 500}]), encoding="utf-8")
    rows = {r["run"]: r for r in dict((t.id, rows) for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1"))["RUNS"]}
    assert set(rows) == {"same_label", "same_label-2"} and rows["same_label"]["n"] == rows["same_label-2"]["n"] == 4
    a, b = rows["same_label"], rows["same_label-2"]
    assert (a["diagnostics_server"], a["starts"], a["judge_model"], b["diagnostics_server"], b["starts"], b["master_seed"]) == ("True", "combinations", "judge", "False", "sampled", 42)
    # one of four episodes names a context overflow, the verifier's own failure is none, and the three model calls average 300 input tokens
    assert (a["context_errors"], a["input_tokens_per_call"], b["context_errors"], b["input_tokens_per_call"]) == (0.25, 300.0, 0.0, "")
    assert a["pass"] == 0.75 and a["pass_n"] == 4 and a["judge_pass"] == 0.5 and a["judge_pass_n"] == 4 and b["judge_pass"] == "" and b["judge_pass_n"] == 0
    # the delay of an episode is the larger of its two delays: 0, 2, 4 and 6 cycles, whose median is 3
    assert a["delay_median"] == 3.0 and a["delay_n"] == 4 and b["delay_median"] == 0.0 and b["cost_normalized"] == 1.5
    assert list(_csv(tmp_path / "out" / "RUNS.csv")[0]) == aggregate.with_standard_errors(aggregate.RUN_COLUMNS)


def test_the_five_replicate_runs_of_a_line_pool_in_every_table_and_in_one_row_of_runs(tmp_path):
    """A line of the grid is five runs of one condition, one per replicate, which share no seed, so
    no run is left out (results_index.pooling_refusals) and every table pools them: pairs match on the seed, one pair per start
    and replicate, and the bootstrap resamples the recorded blocks, so the five episodes of a start are one cluster. RUNS, which
    the leaderboard reads, holds the line in one row that names its five folders in replicate order, counts their episodes and
    takes the earliest start, while a run of another set keeps a row of its own. By hand on three starts: with the server on, the
    replicates 1 to 3 of every start pass, without it the first alone, so 9 of 15 against 3 of 15, a paired difference of 0.4 over
    15 pairs in 3 clusters."""
    runs = tmp_path / "runs"
    blocks = [json.dumps(c) for c in ([3, 100, 0, 130], [3, 100, 0, 90], [20, 100, 0, 130])]
    base = {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS)}
    for server in (True, False):
        for replicate in range(1, 6):
            seeds = [10 * replicate + start for start in range(3)]
            passed = replicate <= 3 if server else replicate == 1
            _run(runs, f"{'on' if server else 'off'}_r{replicate}",
                 [_row(s, block=blocks[s % 10], agent="plan_execute_m", config=design.AGENTS_PRESET, ranking_model=DEFAULT_RANKING_MODEL, diagnostics_server=server,
                       **{"pass": passed}) for s in seeds],
                 {**base, "diagnostics_server": server, "starts": "sampled", "master_seed": 42, "seed_pool": 8, "seeds": seeds, "replicate": replicate,
                  "replicate_rule": "the rule", "episodes_planned": 3, "started": f"2026-12-01T0{replicate}:00:00"})
    _run(runs, "on_development", [_row(s, block=blocks[s], agent="plan_execute_m", config=design.AGENTS_PRESET, ranking_model=DEFAULT_RANKING_MODEL, **{"pass": False})
                                  for s in range(3)],
         {**base, "diagnostics_server": True, "starts": "combinations", "seeds": [0, 1, 2], "episodes_planned": 3})
    assert pooling_rules.pooling_left_out(runs) == {} and len(pooling_rules.folders_read(runs)) == 11
    tables = dict((t.id, rows) for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200, fraction="1"))
    agents = {row["diagnostics_server"]: row for row in tables["AGENTS"]}
    assert (agents["True"]["n"], agents["False"]["n"], agents["True"]["pass"], agents["False"]["pass"]) == (18, 15, 0.5, 0.2)
    h2a, = (row for row in tables["AGENTS_pairs"] if row["hypothesis"] == "H2a")
    assert (h2a["pairs"], h2a["clusters"], h2a["pass_diff"]) == (15, 3, 0.4)
    rows = {row["run"]: row for row in tables["RUNS"]}
    line = rows[" ".join(f"on_r{r}" for r in range(1, 6))]
    assert set(rows) == {line["run"], " ".join(f"off_r{r}" for r in range(1, 6)), "on_development"}
    assert (line["n"], line["replicate"], line["episodes_planned"], line["pass"], line["started"]) == (15, "1 2 3 4 5", 15, 0.6, "2026-12-01T01:00:00")
    assert (rows["on_development"]["n"], rows["on_development"]["replicate"]) == (3, "")
    index = json.loads((tmp_path / "INDEX.json").read_text(encoding="utf-8"))
    assert {entry["run"]: entry["episode_set"]["replicate"] for entry in index["inputs"]}["on_r4"] == 4


# ----- the six rows of the agents and the CausalDiagnosis MCP (AGENTS, AGENTS_pairs, A13_A17, CALLS, STRATA) -----

BLOCK_A, BLOCK_B = json.dumps(FAULTY_LIST), json.dumps([3, 100, 0, 90])
AGENT_RUNS = {  # run name: (agent label, diagnostics_server, clock_mode, pass per seed, cost, llm_calls)
    "pe_bare": ("plan_execute_m", False, "declared", (False, False, False, False), 40, 4),
    "pe_server": ("plan_execute_m", True, "declared", (True, False, False, False), 30, 4),
    "aat_bare": ("agent_as_tool_m", False, "declared", (False, False, True, False), 36, 9),
    "aat_server": ("agent_as_tool_m", True, "declared", (True, True, False, False), 24, 9),
    "causal_bare": ("causal_m", False, "declared", (True, False, False, False), 26, 3),
    "causal": ("causal_m", True, "declared", (True, True, True, False), 20, 3),
    "causal_nobranch": ("causal_nobranch_m", True, "declared", (True, False, False, False), 22, 3),
    "aat_gen": ("agent_as_tool_m", True, "generation_time", (True, False, False, False), 24, 9),
    "causal_gen": ("causal_m", True, "generation_time", (True, True, False, False), 20, 3),
}
FIRST_PLAN = {"causal_m": ("test_first", "test_first", "act_now", "act_now"), "causal_nobranch_m": ("act_now",) * 4}
THINK = {"aat_gen": (2, 1), "causal_gen": (1, 0)}  # think_cycles of the read and the finish entry; zero outside the generation time mode


def _envelope(budget_left):
    return json.dumps({"time": 1, "budget_left": budget_left, "steps_left": 9, "open_work_orders": [], "episode_over": False, "result": {}})


def _ranking(read_first, cost):
    test = {"sensor_group": "tier3", "cycles": 1, "price": 10, "duration": 1, "cost": cost, "expected_loss": 1.0} if read_first else None
    return json.dumps({"candidates": [{"combination": FAULTY_LIST, "probability": 0.7}], "read_first": read_first, "separating_test": test})


def _agent_runs(root):
    """Run folders of the six rows and one switch of the causal agent on the preset of the agents table, base, four seeds each: seeds 0 and 1
    share one recorded block and 2 and 3 another; the hidden log holds a scheduled drop on seeds 0 and 1; seed 3 is a start that the
    reference policy's separability marks inseparable and that the causal agent with the MCP declares, and the MCP's E18 grade marks
    seeds 2 and 3 inseparable; a run with the MCP ranks once per episode, and on the even seeds the separating test costs more than
    the budget left in the envelope before it. Every run plays AssetOpsBench's servers, since a model agent on a row of the
    ladder over the rig's action server enters no table."""
    for name, (agent, server, clock, passes, cost, calls) in AGENT_RUNS.items():
        run_dir = root / name
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"label": name, "config_name": design.AGENTS_PRESET, "model": "m", "clock_mode": clock,
                                                      "servers": list(METERED_SERVERS)}), encoding="utf-8")
        rows = []
        for seed, passed in enumerate(passes):
            abstained = agent == "causal_m" and server and seed == 3
            row = {c: "" for c in COLUMNS}
            row.update({"run": name, "seed": seed, "scenario_id": "", "config": design.AGENTS_PRESET, "agent": agent, "model": "m", "clock_mode": clock,
                        "guidance_level": 2, "diagnostics_server": server, "prices": "default", "verifier_scope": "writes", "ranking_model": DEFAULT_RANKING_MODEL,
                        "block": BLOCK_A if seed < 2 else BLOCK_B, "n_faulty": 1, "pass": passed, "success": passed or abstained, "separable": seed != 3,
                        "diagnosis_all_correct": passed, "cost": cost, "cost_normalized": cost / 20, "unnecessary": 0, "llm_calls": calls, "abstained": abstained,
                        "surprises": 1 if agent == "causal_m" else 0, "first_plan_mode": FIRST_PLAN.get(agent, ("",) * 4)[seed],
                        "grade": json.dumps({"separable": seed != 3, "share": 0.0 if seed == 3 else 1.0, "rollouts": 10, "source": "greedy reference",
                                             "rules_key": "cd" * 32, "e18": {"separable": seed < 2, "commit_share": 1.0 if seed < 2 else 0.0, "first_share": 1.0 if seed < 2 else 0.0,
                                                                             "cost": 20.0 if seed < 2 else None}})})
            rows.append(row)
            ep = run_dir / f"seed{seed:06d}"
            (ep / "hidden").mkdir(parents=True)
            (ep / "episode.json").write_text("{}", encoding="utf-8")
            (ep / "released.json").write_text(json.dumps({"rules": {"budget": 100}}), encoding="utf-8")
            think = THINK.get(name, (0, 0))
            log = [{"event": "reset"}, {"event": "read", "think_cycles": think[0]}]
            if seed < 2:
                log.append({"event": "scheduled_drop", "component": "cooler"})
            log.append({"event": "finish", "think_cycles": think[1]})
            (ep / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
            trajectory = [{"server": "hydraulic_rig", "tool": "read_sensors", "response": _envelope(50), "error": None}]
            if server:
                trajectory.append({"server": "causal_diagnosis", "tool": "rank_root_causes", "response": _ranking(True, 60 if seed % 2 == 0 else 10), "error": None})
            (ep / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        write_rows(run_dir / "scores.csv", rows, COLUMNS)
    return root


def _logged(run_dir, seeds, event="submit_diagnosis"):
    """Adds an event to the hidden log of the episodes of the given seeds, the write of a diagnosis unless it names another."""
    for seed in seeds:
        path = run_dir / f"seed{seed:06d}" / "hidden" / "log.json"
        path.write_text(json.dumps([*json.loads(path.read_text(encoding="utf-8")), {"event": event}]), encoding="utf-8")


def test_wrong_counts_a_submitted_diagnosis_that_misses_and_no_declaration_or_episode_without_a_diagnosis(tmp_path):
    """By hand on nine episodes of one row, each on a block of its own: three right, three wrong, a declaration, an episode
    without a diagnosis, and one without a top-1 cell, which the counting rule counts as not right and, with no diagnosis written, as
    not wrong. So wrong is 3 of 9, with the interval of the bootstrap clustered by block over the nine episodes."""
    runs = tmp_path / "runs"
    correct = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False, 6: False, 7: False, 8: ""}
    run = _run(runs, "row", [_row(s, agent="plan_execute_m", block=_start(s), diagnosis_all_correct=cell, abstained=s == 6) for s, cell in correct.items()])
    _logged(run, range(6))
    _logged(run, [6], "declare_insufficient_evidence")
    row = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["AGENTS"][0]
    assert (row["n"], row["diagnosis_wrong_n"], row["diagnosis_all_correct_n"]) == (9, 9, 9)
    assert row["diagnosis_wrong"] == row["diagnosis_all_correct"] == aggregate._r(3 / 9)
    clustered = cluster_proportion([s in (3, 4, 5) for s in range(9)], [_start(s) for s in range(9)], resamples=50, seed=0)
    assert (row["diagnosis_wrong_low"], row["diagnosis_wrong_high"]) == (round(clustered["low"], 6), round(clustered["high"], 6))
    assert row["diagnosis_wrong_low"] < aggregate._r(3 / 9) < row["diagnosis_wrong_high"]


def test_pass_k_averages_over_the_start_combinations_the_share_of_their_k_subsets_that_all_pass(tmp_path):
    """pass^k of tau-bench by hand on two rows of made-up episodes. On the first, one start combination passes four of five episodes,
    one passes one of five and one passes both of its two, so per combination C(c, k) / C(n, k) is 4/5, 1/5 and 1 at k = 1, 6/10, 0 and
    1 at k = 2, and the combination of two episodes enters no k above two. The interval is the bootstrap over the combinations. On the
    second row every combination has five episodes, so pass^1 is the pass rate of the row and pass^k does not rise with k."""
    runs = tmp_path / "runs"
    first = {0: [1, 1, 1, 1, 0], 1: [1, 0, 0, 0, 0], 2: [1, 1]}
    second = {0: [1, 1, 0, 0, 0], 1: [1, 1, 1, 1, 1]}
    for label, passes in (("plan_execute_m", first), ("agent_as_tool_m", second)):
        seeds = iter(range(100))
        _run(runs, label, [_row(next(seeds), agent=label, block=_start(combination), **{"pass": bool(outcome)})
                           for combination, outcomes in passes.items() for outcome in outcomes])
    rows = {row["agent"]: row for row in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["AGENTS"]}
    row = rows["plan_execute_m"]
    shares = {1: [4 / 5, 1 / 5, 1.0], 2: [6 / 10, 0.0, 1.0], 3: [4 / 10, 0.0], 4: [1 / 5, 0.0], 5: [0.0, 0.0]}
    assert [(row[f"pass_k{k}"], row[f"pass_k{k}_n"]) for k in shares] == [(aggregate._r(sum(s) / len(s)), len(s)) for s in shares.values()]
    assert row["pass_k1"] != row["pass"] == aggregate._r(7 / 12)  # combinations of unequal size weigh alike in pass^k
    boot = cluster_bootstrap([[share, 1.0] for share in shares[2]], [_start(c) for c in first], ratio, 50, 0)
    assert (row["pass_k2_low"], row["pass_k2_high"], row["pass_k2_se"]) == (aggregate._r(boot["low"]), aggregate._r(boot["high"]), aggregate._r(boot["se"]))
    equal = rows["agent_as_tool_m"]
    assert equal["pass_k1"] == equal["pass"] == aggregate._r(7 / 10)
    assert [equal[f"pass_k{k}"] for k in aggregate.PASS_KS] == sorted((equal[f"pass_k{k}"] for k in aggregate.PASS_KS), reverse=True)
    assert equal["pass_k2"] == aggregate._r((1 / 10 + 1) / 2) and equal[f"pass_k{aggregate.PASS_KS[-1]}"] == aggregate._r(1 / 2)


def test_a_cell_read_back_prints_the_digits_of_the_share_itself():
    """The pages print a cell of these tables again with fewer digits, three decimals or a percent with one, so a cell keeps enough digits
    that no share of a row's episodes moves its last printed digit on the way."""
    for n in (47, 5 * 47, 1000):
        for k in range(n + 1):
            share = k / n
            assert f"{aggregate._r(share):.3f}" == f"{share:.3f}" and f"{100 * aggregate._r(share):.1f}" == f"{100 * share:.1f}", (k, n)


def test_wrong_stays_empty_without_a_hidden_log_since_a_diagnosis_cannot_be_told_from_none(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "row", [_row(s, agent="plan_execute_m", diagnosis_all_correct=False) for s in range(3)], hidden=False)
    row = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["AGENTS"][0]
    assert (row["diagnosis_wrong"], row["diagnosis_wrong_n"], row["diagnosis_all_correct_n"]) == ("", 0, 3)


def test_the_agent_tables_hold_every_clock_mode_while_the_pairs_keep_the_clock_mode_of_the_config(tmp_path):
    runs = _agent_runs(tmp_path / "runs")
    out = tmp_path / "out"
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, out, resamples=100, fraction="1")}
    columns = {t.id: t.columns for t in aggregate.tables()}
    condition = ["agent", "model", "config", "clock_mode", "surface", "diagnostics_server", "guidance_level", "verifier_scope", "prices", "ranking_model", "abstention_rule", "training_variant", "lora_adapters",
                 "abstention_cut", "prices_sha256", "fraction"]
    # the graded measures and the normalized score are descriptive columns after the verified outcome, and the replays close the row;
    # pass^k for every k up to the start states of a combination follows Pass, and the production loss follows the cost
    graded = ["pass", *(f"pass_k{k}" for k in aggregate.PASS_KS), "diagnosis_all_correct", "faulty_set_right", "restored_clean", "graded_top1", "graded_pass"]
    measured = ["success", "reward", "diagnosis_wrong", "abstained", "unmatched_evidence", "llm_calls", "cost", "cost_normalized", "downtime_total",
                "turns_rmst", "detection_delay", "repair_delay", "judge_kappa"]
    assert columns["AGENTS"] == [*condition, "n", "played", "planned", "unscored", "unjudged", "incomplete", "refused", "replicates", "read",
                                 *[f"{m}{part}" for m in graded for part in ("", "_low", "_high", "_se", "_n")], "normalized_score", "normalized_score_n",
                                 *[f"{m}{part}" for m in measured for part in ("", "_low", "_high", "_se", "_n")],
                                 "faulty_set_right_baseline_k", "faulty_set_right_baseline_n", "faulty_set_right_baseline", "faulty_set_right_floor",
                                 "faulty_set_right_at_floor", *aggregate.REPLAY_COLUMNS]
    assert columns["AGENTS_pairs"] == aggregate.with_standard_errors(["hypothesis", "agent", "agent_server", "reference", "reference_server", "model", "config", "clock_mode", "surface",
                                       "guidance_level", "verifier_scope", "prices", "ranking_model", "abstention_rule", "training_variant", "lora_adapters", "abstention_cut", "prices_sha256", "fraction", "pairs", "clusters",
                                       "diagnosis_all_correct_diff", "diagnosis_all_correct_diff_low", "diagnosis_all_correct_diff_high", "diagnosis_all_correct_p", "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "success_diff", "success_diff_low", "success_diff_high", "success_p", "reward_diff", "reward_diff_low", "reward_diff_high", "reward_p", "judge_pass_diff", "judge_pass_diff_low", "judge_pass_diff_high", "judge_pass_p", "cost_ratio", "cost_ratio_low", "cost_ratio_high", "cost_ratio_p",
                                       "turns_rmst_diff", "turns_rmst_diff_low", "turns_rmst_diff_high", "turns_rmst_p", "diagnosis_all_correct_holm_p", "diagnosis_all_correct_reject", "pass_holm_p", "pass_reject", "judge_pass_holm_p",
                                       "judge_pass_reject", "reversed", "holds", "family_tests", "reference_pass_k", "reference_pass_n", "at_floor", "status", "reading"])
    measures = ["pass", "unnecessary", "llm_calls", "thinking_cycles", "deviations", "test_first", "cost", "test_cost", "level_accuracy", "repeated_repair_sets",
                "realized_elimination_per_read"]
    assert columns["A13_A17"] == aggregate.with_standard_errors(["agent", "switch", "model", "config", "surface", "n", *[f"{m}{part}" for m in measures for part in ("", "_low", "_high", "_n")],
                                  "pass_diff", "pass_diff_low", "pass_diff_high", "pass_p", "llm_calls_diff", "llm_calls_diff_low", "llm_calls_diff_high",
                                  "llm_calls_p", "status", "pairs", "clusters", "holm_p", "reject"])
    assert columns["CALLS"] == aggregate.with_standard_errors(["agent", "model", "config", "diagnostics_server", "clock_mode", "surface", "guidance_level", "verifier_scope", "prices", "ranking_model", "abstention_rule", "training_variant", "lora_adapters", "abstention_cut", "prices_sha256", "fraction", "n", "llm_calls", "llm_calls_low", "llm_calls_high", "llm_calls_n",
                                "thinking_cycles", "thinking_cycles_low", "thinking_cycles_high", "thinking_cycles_n", "tokens", "tokens_low", "tokens_high", "tokens_n", "llm_calls_diff", "llm_calls_diff_low",
                                "llm_calls_diff_high", "llm_calls_p", "thinking_cycles_diff", "thinking_cycles_diff_low", "thinking_cycles_diff_high",
                                "thinking_cycles_p", "status", "pairs", "clusters", "holm_p", "reject", "vs_bare_llm_calls_diff", "vs_bare_llm_calls_diff_low",
                                "vs_bare_llm_calls_diff_high", "vs_bare_llm_calls_p", "vs_bare_thinking_cycles_diff", "vs_bare_thinking_cycles_diff_low",
                                "vs_bare_thinking_cycles_diff_high", "vs_bare_thinking_cycles_p", "vs_bare_pairs", "vs_bare_clusters"])
    assert columns["STRATA"] == aggregate.with_standard_errors([*condition, "stratum", "n", "pass", "pass_low", "pass_high", "pass_n", "success", "success_low", "success_high", "success_n", "abstained", "abstained_low", "abstained_high", "abstained_n"])
    for table in ("AGENTS", "AGENTS_pairs", "A13_A17", "CALLS", "STRATA"):
        assert list(_csv(out / f"{table}.csv")[0]) == columns[table] and len(_csv(out / f"{table}.csv")) == len(tables[table])

    # AGENTS: one row per agent, server level and clock mode, so the two generation time runs are rows of their own beside
    # the six declared ones, all on AssetOpsBench's servers, and the ablation labels belong to A13_A17
    agents = {(r["agent"], r["diagnostics_server"], r["clock_mode"]): r for r in tables["AGENTS"]}
    order = [("plan_execute_m", "False"), ("plan_execute_m", "True"), ("agent_as_tool_m", "False"), ("agent_as_tool_m", "True"), ("causal_m", "False"), ("causal_m", "True")]
    generation = [("agent_as_tool_m", "True", "generation_time"), ("causal_m", "True", "generation_time")]
    assert set(agents) == {(*k, "declared") for k in order} | set(generation) and all(r["n"] == r["pass_n"] == 4 for r in agents.values())
    assert {r["surface"] for r in agents.values()} == {"aob"} and {r["training_variant"] for r in agents.values()} == {design.AS_RELEASED}
    declared = [agents[(*k, "declared")] for k in order]
    assert [r["pass"] for r in declared] == [0.0, 0.25, 0.25, 0.5, 0.25, 0.75] and [r["cost"] for r in declared] == [40.0, 30.0, 36.0, 24.0, 26.0, 20.0]
    assert [agents[k]["pass"] for k in generation] == [0.25, 0.5] and [agents[k]["cost"] for k in generation] == [24.0, 20.0]
    causal = agents[("causal_m", "True", "declared")]
    assert causal["cost_low"] == causal["cost_high"] == 20.0 and causal["cost_normalized"] == 1.0
    # the cells of AGENTS: the declaration of seed 3 counts in success and in the share of declarations, the model calls per episode
    assert (causal["success"], causal["abstained"], causal["llm_calls"]) == (1.0, 0.25, 3.0)
    assert (agents[("causal_m", "False", "declared")]["success"], agents[("causal_m", "False", "declared")]["abstained"]) == (0.25, 0.0)

    # AGENTS_pairs: the seven hypotheses of design.HYPOTHESES, paired over the four seeds in two blocks, and the fifteen scoring
    # pairs of the six rows, of which no judge pass exists here. By hand from the passes per seed: H1a 0.25, H1b 0, H1c 0.5,
    # H1d 0.25, H2a 0.25, H2b 0.25, H2c 0.5
    pairs = {r["hypothesis"]: r for r in tables["AGENTS_pairs"] if r["hypothesis"] != "scoring"}
    assert sorted(pairs) == [name for name, _, _ in design.HYPOTHESES]
    named = {name: ((row[0] + "_m", row[1]), (reference[0] + "_m", reference[1])) for name, row, reference in design.HYPOTHESES}
    assert all(((r["agent"], r["agent_server"]), (r["reference"], r["reference_server"])) == named[name] for name, r in pairs.items())
    assert [pairs[name]["pass_diff"] for name in sorted(pairs)] == [0.25, 0.0, 0.5, 0.25, 0.25, 0.25, 0.5]
    assert all(r["diagnosis_all_correct_diff"] == r["pass_diff"] and r["pairs"] == 4 and r["clusters"] == 2 for r in pairs.values())
    assert pairs["H2a"]["cost_ratio"] == 0.75 and pairs["H2a"]["cost_ratio_low"] == pairs["H2a"]["cost_ratio_high"] == 0.75  # constant costs
    assert pairs["H2b"]["cost_ratio"] == round(24 / 36, 6) and pairs["H1d"]["cost_ratio"] == round(20 / 24, 6)
    assert pairs["H2c"]["success_diff"] == 0.75  # the correct declaration of seed 3 counts in success, not in pass
    # one Holm family of the 14 tests of top-1 and pass, which decide, and none on success or the cost ratio, which are printed
    assert {r["family_tests"] for r in pairs.values()} == {14}
    assert all(r[f"{m}_holm_p"] != "" and r[f"{m}_reject"] in (True, False) for r in pairs.values() for m in design.HYPOTHESIS_MEASURES)
    assert not any(f"{m}_holm_p" in r for r in pairs.values() for m in ("success", "reward", "cost_ratio"))
    # two clusters leave every adjusted p value of the 14 tests too large to reject, and H1b has no gain, so no hypothesis holds
    assert {r["holds"] for r in pairs.values()} == {False}
    scoring = [r for r in tables["AGENTS_pairs"] if r["hypothesis"] == "scoring"]
    assert len(scoring) == 15 and all(r["pass_holm_p"] != "" and r["judge_pass_diff"] == "" and r["reversed"] == "" and "holds" not in r for r in scoring)
    assert {r["family_tests"] for r in scoring} == {15}  # a family of its own, the pass test of every scoring pair
    assert {(r["agent"], r["agent_server"], r["reference"], r["reference_server"]) for r in scoring} >= {("causal_m", True, "plan_execute_m", False),
                                                                                                        ("causal_m", True, "causal_m", False)}

    # A13_A17: the switches from the label suffixes, paired against the full agent with the label left out of the key
    full, nobranch = tables["A13_A17"]
    assert [r["switch"] for r in (full, nobranch)] == ["full", "branches_off"]
    assert full["cost"] == 20.0 and full["test_cost"] == 0.0 and full["repeated_repair_sets"] == 0.0 and full["level_accuracy_n"] == 0
    assert full["realized_elimination_per_read"] == "" and full["realized_elimination_per_read_n"] == 0  # no released read to replay
    assert full["agent"] == "causal_m" and full["n"] == 4 and full["pass"] == 0.75 and full["pass_diff"] == "" and full.get("holm_p", "") == ""
    assert full["test_first"] == 0.5 and full["deviations"] == 1.0 and full["llm_calls"] == 3.0 and full["thinking_cycles"] == 0.0 and full["unnecessary"] == 0.0
    assert nobranch["agent"] == "causal_nobranch_m" and nobranch["pass_diff"] == -0.5 and nobranch["llm_calls_diff"] == 0.0
    assert nobranch["pairs"] == 4 and nobranch["clusters"] == 2 and nobranch["test_first"] == 0.0 and nobranch["holm_p"] != ""

    # CALLS: every clock mode, the differences against Agent-As-Tool with the server in the same mode
    calls = {(r["agent"], r["diagnostics_server"], r["clock_mode"]): r for r in tables["CALLS"]}
    assert len(calls) == 8 and calls[("causal_m", "False", "declared")]["llm_calls_diff"] == -6.0
    reference = calls[("agent_as_tool_m", "True", "declared")]
    assert reference["llm_calls"] == 9.0 and reference["llm_calls_diff"] == "" and reference["pairs"] == "" and reference.get("holm_p", "") == ""
    assert calls[("plan_execute_m", "False", "declared")]["llm_calls_diff"] == -5.0 and calls[("causal_m", "True", "declared")]["llm_calls_diff"] == -6.0
    assert calls[("causal_m", "True", "declared")]["thinking_cycles"] == 0.0 and calls[("causal_m", "True", "declared")]["thinking_cycles_diff"] == 0.0
    gen = calls[("causal_m", "True", "generation_time")]
    assert gen["thinking_cycles"] == 1.0 and gen["thinking_cycles_diff"] == -2.0 and gen["llm_calls_diff"] == -6.0 and gen["pairs"] == 4 and gen["holm_p"] != ""
    assert calls[("agent_as_tool_m", "True", "generation_time")]["thinking_cycles"] == 3.0
    # the second reference, Agent-As-Tool without the server, under the prefix vs_bare_
    assert calls[("causal_m", "True", "declared")]["vs_bare_llm_calls_diff"] == -6.0 and calls[("plan_execute_m", "False", "declared")]["vs_bare_pairs"] == 4
    assert reference["vs_bare_llm_calls_diff"] == 0.0 and calls[("agent_as_tool_m", "False", "declared")]["vs_bare_llm_calls_diff"] == ""

    # STRATA: seven rows per agent, server level and clock mode, the strata from the hidden log, the separable cell, the E18
    # record of the grade cell and the trajectory
    strata = {(r["agent"], r["diagnostics_server"], r["clock_mode"], r["stratum"]): r for r in tables["STRATA"]}
    assert len(strata) == 8 * len(aggregate.STRATA) == 56 and {r["surface"] for r in tables["STRATA"]} == {"aob"}
    drops = strata[("causal_m", "True", "declared", "deterioration")]  # the scheduled drops of seeds 0 and 1
    assert drops["n"] == drops["pass_n"] == 2 and drops["pass"] == 1.0 and strata[("plan_execute_m", "False", "declared", "deterioration")]["pass"] == 0.0
    assert strata[("agent_as_tool_m", "True", "generation_time", "deterioration")]["pass"] == 0.5
    separable, inseparable = (strata[("causal_m", "True", "declared", name)] for name in ("separable", "inseparable"))
    assert (separable["n"], separable["pass"], separable["abstained"]) == (3, 1.0, 0.0)
    assert (inseparable["n"], inseparable["pass"], inseparable["success"], inseparable["abstained"]) == (1, 0.0, 1.0, 1.0)
    separable_e18, e18 = (strata[("causal_m", "True", "declared", name)] for name in ("separable_e18", "inseparable_e18"))
    assert (separable_e18["n"], separable_e18["pass"]) == (2, 1.0)
    assert e18["n"] == 2 and e18["pass"] == 0.5 and e18["success"] == 1.0  # E18 also marks seed 2 inseparable, which the causal agent passes
    # the start both labels mark inseparable, seed 3 alone, where the declaration earns its credit
    both, bare_both = strata[("causal_m", "True", "declared", "inseparable_both")], strata[("causal_m", "False", "declared", "inseparable_both")]
    assert (both["n"], both["pass"], both["success"], both["abstained"]) == (1, 0.0, 1.0, 1.0)
    assert (bare_both["n"], bare_both["success"], bare_both["abstained"]) == (1, 0.0, 0.0)
    unaffordable = strata[("plan_execute_m", "True", "declared", "test_unaffordable")]
    assert unaffordable["n"] == 2 and unaffordable["pass"] == 0.5
    assert strata[("causal_m", "True", "declared", "test_unaffordable")]["pass"] == 1.0
    bare = strata[("plan_execute_m", "False", "declared", "test_unaffordable")]
    assert bare["n"] == 0 and bare["pass"] == "" and bare["pass_n"] == 0  # no ranking call without the MCP


def test_the_agent_tables_keep_every_condition_apart(tmp_path):
    runs = _agent_runs(tmp_path / "runs")
    # an ablation line of the grid: Plan-Execute with the server at guidance level 1, passing every episode
    shutil.copytree(runs / "pe_server", runs / "pe_server_guidance1")
    rows = [{**r, "run": "pe_server_guidance1", "guidance_level": 1, "pass": True} for r in _csv(runs / "pe_server" / "scores.csv")]
    write_rows(runs / "pe_server_guidance1" / "scores.csv", rows, COLUMNS)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    pe = {r["guidance_level"]: r for r in tables["AGENTS"] if (r["agent"], r["diagnostics_server"]) == ("plan_execute_m", "True")}
    assert set(pe) == {"1", "2"} and pe["2"]["n"] == pe["1"]["n"] == 4 and pe["2"]["pass"] == 0.25 and pe["1"]["pass"] == 1.0
    calls = {r["guidance_level"]: r for r in tables["CALLS"] if (r["agent"], r["diagnostics_server"], r["clock_mode"]) == ("plan_execute_m", "True", "declared")}
    # no Agent-As-Tool run shares guidance level 1, so that row has no reference to pair with
    assert calls["2"]["llm_calls_diff"] == -5.0 and calls["1"]["llm_calls_diff"] == "" and calls["1"]["pairs"] == ""
    strata = [r for r in tables["STRATA"] if (r["agent"], r["diagnostics_server"]) == ("plan_execute_m", "True")]
    assert len(strata) == 2 * len(aggregate.STRATA) and {r["guidance_level"] for r in strata} == {"1", "2"}


def test_labels_counts_each_start_once_against_a_hand_table_and_a8_reads_the_separable_cell(tmp_path):
    # eight starts with (greedy share, E18 first commitment share): four separable under both labels, one under the independent
    # label only, one under E18 only, two under neither. A rollout may commit to the truth after a wrong first commitment, so the
    # share that ever commits is larger, and it decides neither the binary label nor the first correlation
    shares = [(1.0, 1.0), (0.9, 0.8), (0.8, 0.9), (0.7, 0.6), (0.6, 0.0), (0.4, 0.7), (0.2, 0.0), (0.0, 0.0)]
    commits = [1.0, 0.9, 1.0, 1.0, 0.4, 0.9, 0.1, 0.0]

    def row(i, share, first):
        record = {"separable": share > 0.5, "share": share, "rollouts": 10, "source": "greedy reference", "rules_key": "ab" * 32,
                  "e18": {"separable": first > 0.5, "commit_share": commits[i], "first_share": first, "cost": 20.0 if commits[i] else None}}
        declared = i == 6  # a correct declaration on a start that the independent label marks inseparable
        return _row(i, block=json.dumps([i, 100, 0, 130]), grade=json.dumps(record), separable=share > 0.5, abstained=declared, ranking_model=DEFAULT_RANKING_MODEL,
                    abstention_cites_reading=declared, abstention_correct=declared)

    runs = tmp_path / "runs"
    _run(runs, "first", [row(i, *s) for i, s in enumerate(shares)])
    _run(runs, "second", [row(i, *s) for i, s in enumerate(shares)])  # a second run on the same starts adds no start
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    labels, = tables["LABELS"]
    assert labels["rules_key"] == "ab" * 6 and labels["ranking_model"] == DEFAULT_RANKING_MODEL and labels["starts"] == 8
    assert (labels["both_separable"], labels["separable_label_only"], labels["separable_e18_only"], labels["both_inseparable"]) == (4, 1, 1, 2)
    # by hand: agreement 6/8, chance (5/8)(5/8) + (3/8)(3/8) = 34/64, kappa (48/64 - 34/64) / (30/64) = 7/15
    assert labels["kappa"] == round(7 / 15, 6)
    # by hand: greedy share ranks 8 to 1, first commitment share ranks 8, 6, 7, 4, 2, 5, 2, 2 (three tied zeros), rho = 35 / sqrt(42 x 40)
    assert labels["share_spearman"] == round(35 / (42 * 40) ** 0.5, 6)
    # by hand: the share that ever commits ranks 7, 4.5, 7, 7, 3, 4.5, 2, 1, rho = 33 / sqrt(42 x 39.5)
    assert labels["commit_share_spearman"] == round(33 / (42 * 39.5) ** 0.5, 6)
    a8 = {r["label"]: r for r in tables["A8"]}
    assert a8["separability"]["inseparable_starts"] == 6 and a8["separability"]["abstentions"] == 2 and a8["separability"]["recall"] == round(2 / 6, 6)
    # under E18 the starts 4, 6 and 7 commit first to the truth on no rollout and are inseparable, whatever share ever commits on
    # them, start 6 declared in both runs
    e18 = a8["e18"]
    assert (e18["n"], e18["inseparable_starts"], e18["abstentions"], e18["precision"], e18["recall"]) == (16, 6, 2, 1.0, round(2 / 6, 6))


def test_labels_leaves_the_kappa_cell_empty_when_one_label_marks_every_start_separable(tmp_path):
    """On a made up split: one label marks every start separable and the other a part of them, so one margin of
    kappa is empty and the formula reads 0. The cell stays empty, and the rank correlation of the shares stays a number."""
    greedy_shares = [1.0 - i * 0.02 if i < 20 else 0.5 - (i - 20) * 0.01 for i in range(47)]
    first_shares = [1.0 - (i + 1) % 47 * 0.01 for i in range(47)]

    def row(i):
        record = {"separable": i < 20, "share": greedy_shares[i], "rollouts": 10, "source": "greedy reference", "rules_key": "ab" * 32,
                  "e18": {"separable": True, "commit_share": 1.0, "first_share": first_shares[i], "cost": 20.0}}
        return _row(i, block=json.dumps([i, 100, 0, 130]), grade=json.dumps(record), separable=i < 20)

    runs = tmp_path / "runs"
    _run(runs, "first", [row(i) for i in range(47)])
    labels, = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["LABELS"]
    assert (labels["both_separable"], labels["separable_label_only"], labels["separable_e18_only"], labels["both_inseparable"]) == (20, 0, 27, 0)
    assert labels["kappa"] == "" and labels["share_spearman"] == round(aggregate.spearman(greedy_shares, first_shares), 6)
    assert labels["commit_share_spearman"] == ""  # a constant share has no rank correlation
    ours, theirs = [i < 20 for i in range(47)], [True] * 47
    assert aggregate.cohens_kappa(ours, theirs) == 0.0  # the formula itself stays as A1 reads it
    assert aggregate.label_kappa(ours, theirs) is None and aggregate.label_kappa(theirs, ours) is None
    assert aggregate.label_kappa(ours, ours) == 1


def test_a_start_that_commits_at_last_but_mostly_wrong_first_fills_the_e18_stratum(tmp_path):
    """The binary label reads the first commitments, so a start on which every rollout ends on the truth and most begin elsewhere
    falls in the stratum and counts in the recall under E18, which the rule of any commitment would have left empty."""
    def row(i, first):
        record = {"separable": True, "share": 1.0 - i * 0.1, "rollouts": 10, "source": "greedy reference", "rules_key": "ab" * 32,
                  "e18": {"separable": first > 0.5, "commit_share": 1.0, "first_share": first, "cost": 20.0}}
        return _row(i, agent="causal_m", block=json.dumps([i, 100, 0, 130]), grade=json.dumps(record), separable=True, abstained=i == 2, abstention_cites_reading=i == 2,
                    ranking_model=DEFAULT_RANKING_MODEL)

    runs = tmp_path / "runs"
    _run(runs, "first", [row(0, 0.9), row(1, 0.6), row(2, 0.3), row(3, 0.5)])
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    labels, = tables["LABELS"]
    assert (labels["both_separable"], labels["separable_e18_only"], labels["separable_label_only"], labels["both_inseparable"]) == (2, 0, 2, 0)
    stratum = next(r for r in tables["STRATA"] if r["stratum"] == "inseparable_e18")
    assert stratum["n"] == 2
    e18 = next(r for r in tables["A8"] if r["label"] == "e18")
    assert (e18["inseparable_starts"], e18["abstentions"], e18["recall"]) == (2, 1, 0.5)


def test_a8_under_e18_counts_a_declaration_with_no_reading_as_a_declaration_and_not_as_a_correct_one(tmp_path):
    """The rule of a correct declaration under the server's label as under the independent one: two declarations on starts E18 marks inseparable, of which
    one cites a reading, so one of two is correct by precision and by recall."""
    def row(i, cites):
        record = {"separable": True, "share": 1.0, "rollouts": 10, "source": "greedy reference", "rules_key": "ab" * 32,
                  "e18": {"separable": False, "commit_share": 0.0, "first_share": 0.0, "cost": None}}
        return _row(i, agent="causal_m", block=json.dumps([i, 100, 0, 130]), grade=json.dumps(record), separable=True, abstained=True,
                    abstention_cites_reading=cites, ranking_model=DEFAULT_RANKING_MODEL)

    runs = tmp_path / "runs"
    _run(runs, "first", [row(0, True), row(1, False)])
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    e18 = next(r for r in tables["A8"] if r["label"] == "e18")
    assert (e18["inseparable_starts"], e18["abstentions"], e18["precision"], e18["recall"]) == (2, 2, 0.5, 0.5)


def test_the_derived_fields_read_the_hidden_log_and_the_trajectory(tmp_path):
    runs = _agent_runs(tmp_path / "runs")
    episodes = {(e["run"], e["seed"]): e for e in aggregate.add_derived_fields(aggregate.load_episodes(runs))}
    assert episodes[("aat_gen", "0")]["thinking_cycles"] == 3 and episodes[("causal", "0")]["thinking_cycles"] == 0
    assert episodes[("causal", "0")]["deterioration"] is True and episodes[("causal", "3")]["deterioration"] is False
    assert episodes[("causal", "2")]["deterioration"] is False  # a log without a drop holds no deterioration transition
    assert episodes[("causal", "0")]["separating_test_unaffordable"] is True and episodes[("causal", "1")]["separating_test_unaffordable"] is False
    assert episodes[("pe_bare", "0")]["separating_test_unaffordable"] is False  # no ranking call
    assert episodes[("causal", "0")]["test_first"] is True and episodes[("causal", "2")]["test_first"] is False and episodes[("pe_bare", "0")]["test_first"] is None
    # without an action server call before the ranking the budget of released.json is the budget left; a failed call counts nothing
    ep = tmp_path / "lone"
    ep.mkdir()
    episode = {"episode_dir": ep}
    assert aggregate.thinking_cycles(episode) is None and aggregate.deterioration(episode) is None and aggregate.separating_test_unaffordable(episode) is None
    (ep / "released.json").write_text(json.dumps({"rules": {"budget": 50}}), encoding="utf-8")
    ranking = [{"server": "causal_diagnosis", "tool": "rank_root_causes", "response": _ranking(True, 60), "error": None}]
    (ep / "trajectory.json").write_text(json.dumps(ranking), encoding="utf-8")
    assert aggregate.separating_test_unaffordable(episode) is True
    (ep / "released.json").write_text(json.dumps({"rules": {"budget": 70}}), encoding="utf-8")
    assert aggregate.separating_test_unaffordable(episode) is False
    (ep / "trajectory.json").write_text(json.dumps([dict(ranking[0], response="", error="boom")]), encoding="utf-8")
    assert aggregate.separating_test_unaffordable(episode) is False
    assert aggregate.switch_of({"agent": "causal_len1_my_model", "model": "my_model"}) == "plan_length_1"
    assert aggregate.switch_of({"agent": "causal_nobranch_noexp_m", "model": "m"}) == "branches_off+expectations_off"
    assert aggregate.switch_of({"agent": "causal_m", "model": "m"}) == "full"


def test_the_hidden_log_gives_the_price_of_the_reads_and_the_repeated_repair_sets(tmp_path):
    ep = tmp_path / "ep"
    (ep / "hidden").mkdir(parents=True)
    episode = {"episode_dir": ep}
    start = [3, 100, 0, 130]  # the cooler degraded from cycle zero, the rest healthy
    log = [{"event": "reset", "t_start": 0, "hidden": start},
           {"event": "read", "t_start": 0, "cost": 1}, {"event": "read", "t_start": 1, "cost": 3},
           {"event": "work_order_created", "t_start": 2, "component": "cooler", "visit": 1, "id": "wo-1"},
           {"event": "hazard_drop", "t_start": 5, "component": "pump"},
           {"event": "work_order_created", "t_start": 9, "component": "pump", "visit": 2, "id": "wo-2"},
           {"event": "work_order_created", "t_start": 12, "component": "cooler", "visit": 3, "id": "wo-3"}]
    (ep / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    assert aggregate.read_price(episode) == 4.0 and aggregate.deterioration(episode) is True
    assert aggregate.repeated_repair_sets(episode) == 1  # the third visit orders the cooler again
    (ep / "hidden" / "log.json").write_text(json.dumps([{"event": "reset", "t_start": 0, "hidden": [100, 100, 0, 130]}]), encoding="utf-8")
    assert aggregate.read_price(episode) == 0.0 and aggregate.repeated_repair_sets(episode) == 0
    # the recomputed time to work order of the verifier's delay left the tool, since no table read it
    assert not hasattr(aggregate, "time_to_work_order")


def test_the_trajectory_gives_the_level_starts_and_the_readings_on_either_kind_of_row(tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    episode = {"episode_dir": ep}

    def ranking(*combinations):
        candidates = [{"combination": dict(zip(["cooler", "valve", "pump", "accumulator"], c)), "probability": 0.5} for c in combinations]
        return {"server": "causal_diagnosis", "tool": "rank_root_causes", "error": None,
                "response": json.dumps({"candidates": candidates})}

    def call(server, tool):
        return {"server": server, "tool": tool, "error": None, "response": "{}"}

    trajectory = [ranking([3, 100, 0, 130], [20, 100, 0, 130]), call("hydraulic_rig", "read_sensors"), ranking([3, 100, 0, 130]),
                  call("iot", "history"), call("iot", "sites"), ranking([3, 100, 0, 130]),
                  call("wo", "generate_work_order"), call("iot", "history"), ranking([3, 100, 0, 130])]
    (ep / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    # the first ranking puts the cooler at levels 3 and 20 on top, one faulty set at two levels
    assert aggregate.first_ranking_shares_faulty_set(episode) is True
    # A12 reads the transcript that the servers wrote and the hidden log: a reading is a call whose log entry is a read, on the rig
    # and on the AssetOpsBench rows alike, and the free call of sites, whose entry is no read, counts nothing
    _transcript(ep, [ranking([3, 100, 0, 130], [20, 100, 0, 130]), call("hydraulic_rig", "read_sensors"), ranking([3, 100, 0, 130]),
                     call("iot", "history"), call("iot", "sites"), ranking([3, 100, 0, 130]), call("wo", "generate_work_order"),
                     call("iot", "history"), ranking([3, 100, 0, 130])])
    _log(ep, ["read", "read", "aob_read", "work_order_created", "read"])
    assert [(r, top, truth) for r, top, _, truth in aggregate.rankings(episode)] == [(0, FAULTY_LIST, FAULTY_LIST), (1, FAULTY_LIST, FAULTY_LIST),
                                                                                    (2, FAULTY_LIST, FAULTY_LIST), (3, FAULTY_LIST, FAULTY_LIST)]
    trajectory[0] = ranking([3, 100, 0, 130], [3, 90, 0, 130])  # another faulty set on top
    (ep / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    assert aggregate.first_ranking_shares_faulty_set(episode) is False
    (ep / "trajectory.json").write_text(json.dumps([call("hydraulic_rig", "read_sensors")]), encoding="utf-8")
    assert aggregate.first_ranking_shares_faulty_set(episode) is None


def test_the_scoring_pairs_mark_a_reversed_order_and_share_the_holm_family(tmp_path):
    runs = _agent_runs(tmp_path / "runs")
    # the judge passes every episode of Plan-Execute with the server and none of the CausalDiagnosis agent, so the pair the
    # verified pass orders one way the judge orders the other way
    for name, judged in (("pe_server", True), ("causal", False), ("pe_bare", False), ("aat_bare", False), ("aat_server", False), ("causal_bare", False)):
        rows = [{**r, "judge_pass": judged} for r in _csv(runs / name / "scores.csv")]
        write_rows(runs / name / "scores.csv", rows, COLUMNS)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    scoring = {(r["agent"], r["agent_server"], r["reference"], r["reference_server"]): r for r in tables["AGENTS_pairs"] if r["hypothesis"] == "scoring"}
    ours = scoring[("causal_m", True, "plan_execute_m", True)]
    assert ours["pass_diff"] == 0.5 and ours["judge_pass_diff"] == -1.0 and ours["reversed"] is True
    assert ours["judge_pass_holm_p"] != "" and ours["judge_pass_reject"] in (True, False)
    same = scoring[("agent_as_tool_m", True, "agent_as_tool_m", False)]
    assert same["judge_pass_diff"] == 0.0 and same["reversed"] is False
    # two families per model and config: the 14 tests of the seven hypotheses, and 2 per scoring pair with both differences
    tested = sum(1 for r in tables["AGENTS_pairs"] for m in ("diagnosis_all_correct", "pass", "judge_pass") if r.get(f"{m}_holm_p", "") != "")
    assert tested == 14 + 30
    assert {r["family_tests"] for r in tables["AGENTS_pairs"] if r["hypothesis"] == "scoring"} == {30}
    assert {r["family_tests"] for r in tables["AGENTS_pairs"] if r["hypothesis"] != "scoring"} == {14}
    assert all(rows == [] for table, rows in tables.items() if table in ("GEPA_curve", "TRAINING", "TRAINING_pairs", "FLEET", "FLEET_pairs"))


def test_a_run_of_the_retired_gepa_mode_never_enters_the_gepa_table(tmp_path):
    """GEPA.csv is the table of the online loop of GEPA, which gepa_prompt_optimization/ writes, and the offline optimizer of the
    mode gepa left the fork, so a stale run of it, run.json mode gepa with gepa_result.json beside it, enters no table, RUNS
    included, and INDEX.json names it with its reason."""
    runs = tmp_path / "runs"
    run_dir = _run(runs, "gepa_card", [_row(0, agent="gepa_m", reward=0.5)], {"mode": "gepa", "gepa": {"card": "score_only", "reflection_model": "reflector"}})
    (run_dir / "gepa_result.json").write_text(json.dumps({"seed_val_scores": [0.2, 0.4], "best_val_scores": [0.6, 0.8], "episodes_played": 6,
                                                           "rollouts_to_best": 4}), encoding="utf-8")
    out = tmp_path / "results" / "aggregate"
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, out, resamples=50, fraction="1")}
    assert tables["GEPA"] == [] and tables["GEPA_curve"] == [] and tables["RUNS"] == []
    index = json.loads((out.parent / "INDEX.json").read_text(encoding="utf-8"))
    assert index["excluded"] == [{"folder": "gepa_card", "reason": pooling_rules.RETIRED_GEPA_MODE}]


def test_a7_moves_the_ranking_under_the_read_gate_and_is_no_longer_one_by_construction(tmp_path):
    """En4: the same outcomes under both scopes, the pass of scores.csv equal, and a read gate that fails the agent with
    the highest pass on most episodes, so the rank correlation falls below one, while an agent's runs on another preset stay out.
    By hand: the writes rates 0.2, 0.4, 0.6, 0.8, 1.0 rank 1 to 5, the rates under the gate 0.2, 0.4, 0.6, 0.8, 0.2 rank 1.5, 3, 4, 5, 1.5,
    so rho = 2 / sqrt(10 x 9.5)."""
    runs = tmp_path / "runs"
    main = {"config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL}
    meta = {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS)}
    five = (("plan_execute_m", False), ("plan_execute_m", True), ("agent_as_tool_m", False), ("agent_as_tool_m", True), ("causal_m", True))
    held = (5, 5, 5, 5, 1)  # the episodes that hold the read gate per row
    for i, (agent, server) in enumerate(five):
        passed = [s <= i for s in range(5)]
        _run(runs, f"writes_{i}", [_row(s, agent=agent, diagnostics_server=server, **{"pass": passed[s]}, **main) for s in range(5)], meta)
        _run(runs, f"all_{i}", [_row(s, agent=agent, diagnostics_server=server, verifier_scope="all", read_gate=s < held[i], **{"pass": passed[s]}, **main) for s in range(5)], meta)
    _run(runs, "other_preset", [_row(s, agent="causal_m", config="priced", clock_mode="declared", ranking_model=DEFAULT_RANKING_MODEL, **{"pass": False}) for s in range(5)],
         {"config_name": "priced", "servers": list(METERED_SERVERS)})  # A7 plays the preset of the agents table alone
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    rows = {(r["agent"], r["diagnostics_server"]): r for r in tables["A7"] if r["agent"] != "all agents"}
    assert len(rows) == 5, "the rows with and without the server stay apart"
    assert [rows[(agent, str(server))]["pass_writes"] for agent, server in five] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert [rows[(agent, str(server))]["pass_all"] for agent, server in five] == [0.2, 0.4, 0.6, 0.8, 0.2]
    assert all(row["n_writes"] == row["n_all"] == 5 for row in rows.values())
    last = tables["A7"][-1]
    assert last["agent"] == "all agents" and last["model"] == "m" and last["agents_in_both_scopes"] == 5
    assert last["rank_correlation"] == round(2 / 95 ** 0.5, 6)
    runs_rows = {r["run"]: r for r in tables["RUNS"]}
    assert runs_rows["all_4"]["pass"] == 1.0 and runs_rows["all_4"]["scope_pass"] == 0.2 and runs_rows["writes_4"]["scope_pass"] == 1.0


def test_a7_holds_one_rank_correlation_per_model_and_pooling_the_models_would_read_higher(tmp_path):
    """Two models on different levels, where the read gate turns the ranking of the rows against the pass in each of them. By hand,
    in either model the ranks 1 to 5 against 3, 4, 5, 1.5, 1.5 give rho = -5.5 / sqrt(10 x 9.5), while the ten rows pooled give
    51.5 / sqrt(82.5 x 81.5) above zero, which only the distance between the two models produces."""
    runs = tmp_path / "runs"
    five = (("plan_execute", False), ("plan_execute", True), ("agent_as_tool", False), ("agent_as_tool", True), ("causal", True))
    for model, level, held in (("m", 0, (1, 2, 3, 0, 0)), ("n", 5, (6, 7, 8, 5, 5))):  # held: the seeds that hold the read gate per row
        meta = {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS), "model": model}
        for i, (prefix, server) in enumerate(five):
            fields = {"agent": f"{prefix}_{model}", "model": model, "diagnostics_server": server, "config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL}
            passed = [s <= i + level for s in range(10)]
            _run(runs, f"writes_{model}_{i}", [_row(s, **fields, **{"pass": passed[s]}) for s in range(10)], meta)
            _run(runs, f"all_{model}_{i}", [_row(s, **fields, verifier_scope="all", read_gate=s < held[i], **{"pass": passed[s]}) for s in range(10)], meta)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    per_model = {r["model"]: r for r in tables["A7"] if r["agent"] == "all agents"}
    within = round(-5.5 / 95 ** 0.5, 6)
    assert per_model["m"]["rank_correlation"] == per_model["n"]["rank_correlation"] == within
    assert per_model["m"]["agents_in_both_scopes"] == per_model["n"]["agents_in_both_scopes"] == 5
    rows = [r for r in tables["A7"] if r["agent"] != "all agents"]
    pooled = aggregate.spearman([r["pass_writes"] for r in rows], [r["pass_all"] for r in rows])
    assert pooled == pytest.approx(51.5 / (82.5 * 81.5) ** 0.5) and pooled > 0 > within


def test_the_primary_pairs_read_only_the_runs_at_the_main_levels(tmp_path):
    # an En4 line of the grid on the preset of the agents table, the verifier scope all beside the runs of the six rows: it keeps rows of
    # its own in AGENTS and stays out of the hypotheses, which pair the runs at the main levels alone
    runs = tmp_path / "runs"
    meta = {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS)}
    main = {"config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL}
    for name, server, scope, passed in (("pe_server", True, "writes", (True, False, False, False)), ("pe_bare", False, "writes", (False,) * 4),
                                        ("pe_server_all", True, "all", (True,) * 4), ("pe_bare_all", False, "all", (True,) * 4)):
        _run(runs, name, [_row(s, agent="plan_execute_m", diagnostics_server=server, verifier_scope=scope, **{"pass": p}, **main)
                          for s, p in enumerate(passed)], meta)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    assert {r["verifier_scope"] for r in tables["AGENTS"]} == {"writes", "all"}
    server, = (r for r in tables["AGENTS_pairs"] if r["hypothesis"] == "H2a")
    assert server["pairs"] == 4 and server["pass_diff"] == 0.25
    assert (server["clock_mode"], server["surface"], server["verifier_scope"], server["ranking_model"]) == ("declared", "aob", "writes", DEFAULT_RANKING_MODEL)
    assert {r["verifier_scope"] for r in tables["AGENTS_pairs"]} == {"writes"}


def test_an_abstention_line_keeps_a_row_of_its_own_and_stays_out_of_the_primary_pairs(tmp_path):
    # an En5 line of the grid on the preset of the agents table, the fixed posterior cut beside the decision rule of run.json:
    # the abstention rule is a setting of the condition, so the line keeps a row of its own in AGENTS and the hypotheses
    # pair the decision rule runs alone
    runs = tmp_path / "runs"
    main = {"config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL}
    for name, server, rule, passed in (("pe_server", True, "decision", (True, False, False, False)), ("pe_bare", False, "decision", (False,) * 4),
                                       ("pe_server_cut", True, "fixed_cut", (True,) * 4)):
        _run(runs, name, [_row(s, agent="plan_execute_m", diagnostics_server=server, **{"pass": p}, **main) for s, p in enumerate(passed)],
             {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS), "abstention_rule": rule})
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    assert sorted((r["abstention_rule"], r["n"], r["pass"]) for r in tables["AGENTS"]) == [("decision", 4, 0.0), ("decision", 4, 0.25),
                                                                                           ("fixed_cut", 4, 1.0)]
    server, = (r for r in tables["AGENTS_pairs"] if r["hypothesis"] == "H2a")
    assert (server["abstention_rule"], server["pairs"], server["pass_diff"]) == ("decision", 4, 0.25)


def test_the_switch_rows_read_only_the_runs_at_the_main_levels(tmp_path):
    # an En4 line of the full CausalDiagnosis agent on the preset of the agents table, the verifier scope all: it stays out of
    # the full row of A13_A17, against which every switch pairs, and the rows name their tool surface
    runs = tmp_path / "runs"
    main = {"config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL}
    for name, agent, scope, passed in (("causal", "causal_m", "writes", (True, True, False, False)), ("causal_all", "causal_m", "all", (True,) * 4),
                                       ("causal_nobranch", "causal_nobranch_m", "writes", (False,) * 4)):
        _run(runs, name, [_row(s, agent=agent, verifier_scope=scope, **{"pass": p}, **main) for s, p in enumerate(passed)],
             {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS)})
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    full, nobranch = tables["A13_A17"]
    assert (full["switch"], full["surface"], full["n"], full["pass"]) == ("full", "aob", 4, 0.5)
    assert (nobranch["switch"], nobranch["surface"], nobranch["pairs"], nobranch["pass_diff"]) == ("branches_off", "aob", 4, -0.5)


def test_agents_and_strata_keep_the_sensitivity_line_and_every_condition_names_its_surface(tmp_path):
    """Both clock_mode and the tool surface join the condition, so the declared
    durations line of the advanced real-time clock row is a row of AGENTS and STRATA beside the row's own clock, and the
    rows of AGENTS, STRATA, CALLS and RUNS name the surface of their runs, here AssetOpsBench's servers."""
    runs = tmp_path / "runs"
    aob = {"servers": list(METERED_SERVERS), "config_name": "advanced_real_time"}

    def played(clock):
        return [_row(s, config="advanced_real_time", agent="causal_m", clock_mode=clock) for s in range(2)]

    _run(runs, "own_clock", played("generation_time"), aob)
    _run(runs, "declared", played("declared"), aob)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    lines = [("declared", "aob"), ("generation_time", "aob")]
    assert sorted((r["clock_mode"], r["surface"]) for r in tables["AGENTS"]) == lines
    assert sorted({(r["clock_mode"], r["surface"]) for r in tables["STRATA"]}) == lines
    assert sorted((r["clock_mode"], r["surface"]) for r in tables["CALLS"]) == lines
    assert sorted(r["surface"] for r in tables["RUNS"]) == ["aob", "aob"]


def test_the_readme_names_gepa_online_and_rebuilds_the_tables_from_the_copies(tmp_path):
    """The README names the online loop by gepa_prompt_optimization/ and no
    longer run_gepa.sbatch, and it rebuilds the tables from the synced copies itself, since sync_runs.sh aggregates
    nothing."""
    out = tmp_path / "results" / "aggregate"
    aggregate.aggregate(tmp_path / "no_runs", out, resamples=50, fraction="1")
    readme = (out.parent / "README.md").read_text(encoding="utf-8")
    assert "`gepa_prompt_optimization.py`, the online loop of GEPA" in readme and "the online loop driver" not in readme and "run_gepa.sbatch" not in readme
    assert "where they stay. The tables, this file and the leaderboard are rebuilt from those copies with" in readme


MAIN_ROW = {"config": "real_time", "clock_mode": "generation_time", "ranking_model": DEFAULT_RANKING_MODEL}
MAIN_META = {"config_name": "real_time", "servers": list(METERED_SERVERS)}
HELDOUT = (0, 1, 2, 3)  # seeds 0 and 1 start with two faulty components, the unseen combinations, and seed 3 is not isolable


def _stream_row(seed, **fields):
    return _row(seed, **MAIN_ROW, **fields)


def _heldout_row(seed, **fields):
    return _row(seed, **MAIN_ROW, block=BLOCK_B if seed < 2 else BLOCK_A, n_faulty=2 if seed < 2 else 1, separable=seed != 3, **fields)


def _gepa_online_run(root, name, card, episodes, rounds, start, final, status="finished", episodes_played=0, rollouts_to_best=0):
    """A run folder of gepa_prompt_optimization/: run.json in its mode, scores.csv with the GEPA call of every episode, the episode
    folders call<n>_seed<s> and gepa_online.json with the rounds and the episodes. An episode is (call, round, candidate, left
    out, seed, row fields)."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({**MAIN_META, "label": name, "model": "m", "mode": "gepa_online",
                                                   "gepa": {"card": card, "reflection_model": "reflector"}}), encoding="utf-8")
    rows, record = [], []
    for call, round_no, candidate, left_out, seed, fields in episodes:
        row = _stream_row(seed, agent="causal_m", gepa_call=call, **fields)
        row["run"] = name
        rows.append(row)
        ep = run_dir / f"call{call:05d}_seed{seed:06d}"
        ep.mkdir()
        (ep / "episode.json").write_text("{}", encoding="utf-8")
        record.append({"call": call, "episode": ep.name, "phase": "stream", "round": round_no, "candidate": candidate, "left_out": left_out,
                       "seed": seed, "reward": fields.get("reward")})
    write_rows(run_dir / "scores.csv", rows, COLUMNS)
    (run_dir / "gepa_online.json").write_text(json.dumps({"format": 2, "status": status, "card": card, "reflection_model": "reflector",
                                                          "start_candidate": start, "final_candidate": final, "episodes_played": episodes_played,
                                                          "rollouts_to_best": rollouts_to_best, "rounds": rounds, "episodes": record}), encoding="utf-8")
    return run_dir


def _online_runs(root):
    """The five rows on the held-out seeds, where Plan-Execute with the server diagnoses one of the two unseen starts as
    Agent-As-Tool with the server does, so the paper's order makes it the best row; a run of the online loop under the
    verifier's card whose child c001 wins round 2, with one episode of the incumbent left out in that round; a run under the
    score-only card that keeps c000; and a run still marked running."""
    for name, agent, server, top1, abstains in (("pe_bare", "plan_execute_m", False, (False, False, True, False), False),
                                               ("pe_server", "plan_execute_m", True, (True, False, True, False), False),
                                               ("aat_bare", "agent_as_tool_m", False, (False, False, False, False), False),
                                               ("aat_server", "agent_as_tool_m", True, (True, False, True, False), False),
                                               ("causal", "causal_m", True, (False, False, True, False), True)):
        _run(root, name, [_heldout_row(s, agent=agent, diagnostics_server=server, diagnosis_all_correct=top1[s],
                                       abstained=abstains and s == 3, abstention_correct=abstains and s == 3) for s in HELDOUT], MAIN_META)
    verifier = [(1, 1, "c000", False, 10, {"reward": 0.2}), (2, 1, "c000", False, 11, {"reward": 0.4}),
                (3, 1, "c000", False, 12, {"reward": 0.6}), (4, 2, "c000", False, 13, {"reward": 0.0}),
                (5, 2, "c001", False, 13, {"reward": 0.5}), (6, 2, "c000", True, 14, {"reward": 0.2}),
                (7, 2, "c001", False, 14, {"reward": 0.7}), (8, 2, "c000", False, 15, {"reward": 0.4}),
                (9, 2, "c001", False, 15, {"reward": 0.9})]
    _gepa_online_run(root, "gepa_verifier", "verifier", verifier, [{"round": 1, "rollouts": 0, "incumbent": "c000"},
                                                                   {"round": 2, "rollouts": 3, "incumbent": "c000"}],
                     "c000", "c001", episodes_played=9, rollouts_to_best=9)
    score = [(1, 1, "c000", False, 10, {"reward": 0.1}), (2, 1, "c000", False, 11, {"reward": 0.2}),
             (3, 1, "c000", False, 12, {"reward": 0.3})]
    one_round = [{"round": 1, "rollouts": 0, "incumbent": "c000"}]
    _gepa_online_run(root, "gepa_score", "score_only", score, one_round, "c000", "c000", episodes_played=3, rollouts_to_best=0)
    _gepa_online_run(root, "gepa_running", "verifier", score, one_round, "c000", "c000", status="running", episodes_played=3)
    return root


def test_the_online_loop_fills_the_gepa_tables_and_no_other_table(tmp_path):
    """The tables GEPA and GEPA_curve read the finished runs of gepa_prompt_optimization/, a run still marked running enters no table, and the
    episodes of the online loop stay out of every other table, RUNS and the tables of training included, which read the evaluation
    runs of the trained agents, so the leaderboard never takes a loop's episode for a row of the paper."""
    runs = _online_runs(tmp_path / "runs")
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200, fraction="1")}
    gepa = {r["run"]: r for r in tables["GEPA"]}
    assert sorted(gepa) == ["gepa_score", "gepa_verifier"]
    fields = ("card", "reflection_model", "candidates_played", "stream_episodes", "left_out", "rollouts_to_best")
    assert [gepa["gepa_verifier"][field] for field in fields] == ["verifier", "reflector", 2, 9, 1, 9]
    assert [gepa["gepa_score"][field] for field in fields] == ["score_only", "reflector", 1, 3, 0, 0]
    # the point of a round is the incumbent's mean over its episodes of the round that the loop did not leave out
    curve = [(r["run"], r["round"], r["rollouts"], r["mean_reward"], r["n"]) for r in tables["GEPA_curve"]]
    assert curve == [("gepa_score", 1, 0, 0.2, 3), ("gepa_verifier", 1, 0, 0.4, 3), ("gepa_verifier", 2, 3, 0.2, 2)]
    assert tables["TRAINING"] == tables["TRAINING_pairs"] == []
    assert sorted(r["run"] for r in tables["RUNS"]) == ["aat_bare", "aat_server", "causal", "pe_bare", "pe_server"]
    assert sum(r["n"] for r in tables["AGENTS"] if r["agent"] == "causal_m") == 4


def test_the_online_mode_and_result_file_are_those_of_gepa_online():
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online
    assert (aggregate.GEPA_ONLINE, aggregate.GEPA_RESULT) == (gepa_online.MODE, gepa_online.RESULT)


def test_agents_and_their_pairs_report_the_restricted_mean_survival_time_over_turns(tmp_path):
    """The turns to a correct diagnosis of the paper's scores table enter AGENTS as the
    restricted mean survival time over turns, censored at the step limit of the run.json config, and the hypotheses of
    AGENTS_pairs as the paired difference of the capped turns, outside the Holm family. Plan-Execute with the MCP diagnoses
    seeds 0 and 2 at turns 2 and 4 of 10 and never seeds 1 and 3, and seed 4 ends without a verdict, which counts as a start never
    diagnosed since every played episode counts, so its mean is (2 + 10 + 4 + 10 + 10) / 5; without the MCP it never diagnoses, so
    every episode counts 10."""
    runs = tmp_path / "runs"
    meta = {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS), "config": {"durations": {"read_per_cycle": 1}, "max_steps": 10}}
    turns = {0: 2, 1: "", 2: 4, 3: ""}
    main = {"config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL}
    _run(runs, "pe_server", [_row(s, agent="plan_execute_m", diagnosis_all_correct=turns[s] != "", turns_to_correct_diagnosis=turns[s], **main)
                             for s in range(4)] + [_row(4, agent="plan_execute_m", diagnosis_all_correct="", **main)], meta)
    _run(runs, "pe_bare", [_row(s, agent="plan_execute_m", diagnostics_server=False, diagnosis_all_correct=False, **main) for s in range(4)], meta)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    by_server = {r["diagnostics_server"]: r for r in tables["AGENTS"]}
    assert (by_server["True"]["n"], by_server["True"]["turns_rmst"], by_server["True"]["turns_rmst_n"]) == (5, 7.2, 5)
    assert (by_server["False"]["turns_rmst"], by_server["False"]["turns_rmst_n"]) == (10.0, 4)
    assert _kaplan_meier_restricted_mean([(2.0, True), (10.0, False), (4.0, True), (10.0, False)], 10.0) == 6.5  # the cell is the area under the curve
    server, = (r for r in tables["AGENTS_pairs"] if r["hypothesis"] == "H2a")
    assert server["turns_rmst_diff"] == -3.5 and server["turns_rmst_p"] != "" and "turns_rmst_holm_p" not in server


def test_elim_measures_the_agents_and_the_reference_policies_and_pairs_the_effect_comparisons(tmp_path, monkeypatch):
    """ELIM holds the six rows under every clock mode and the reference policies, the switches of the causal agent stay
    in A13_A17, ELIM_pairs pairs the hypotheses of AGENTS_pairs, and A13_A17 reads the same replay, here a stand-in of
    elimination_fields that gives every run a constant per read and per cost."""
    runs = _agent_runs(tmp_path / "runs")
    _run(runs, "oracle", [_row(s, agent="oracle", model="", config=design.AGENTS_PRESET, clock_mode="declared", block=BLOCK_A if s < 2 else BLOCK_B,
                               ranking_model=DEFAULT_RANKING_MODEL) for s in range(4)], {"config_name": design.AGENTS_PRESET, "model": ""})
    per_run = {"pe_bare": 1.0, "pe_server": 2.0, "aat_bare": 1.5, "aat_server": 3.0, "causal_bare": 2.5, "causal": 4.0, "causal_nobranch": 3.5,
               "aat_gen": 3.0, "causal_gen": 4.0, "oracle": None}
    seen = []

    def fields(episode):
        seen.append(episode["run"])
        value = per_run[episode["run"]]
        return {"paid_reads": 0 if value is None else 2, "realized_elimination_per_read": value,
                "realized_elimination_per_cost": None if value is None else value / 4, "calibrated_reads": 1 if episode["run"] == "causal" else 0,
                "expected_elimination": 3.0 if episode["run"] == "causal" else None,
                "realized_elimination_calibrated": 4.0 if episode["run"] == "causal" else None,
                "calibration_gap": 1.0 if episode["run"] == "causal" else None, "yardstick": "" if value is None else "abc"}

    monkeypatch.setattr(aggregate.replayed_fields, "elimination_fields", fields)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=100, fraction="1")}
    assert sorted(set(seen)) == sorted(per_run) and len(seen) == 4 * len(per_run)
    elim = {(r["agent"], r["diagnostics_server"], r["clock_mode"]): r for r in tables["ELIM"]}
    assert len(elim) == 9 and ("oracle", "True", "declared") in elim and not any("nobranch" in a for a, _, _ in elim)
    oracle = elim[("oracle", "True", "declared")]
    assert (oracle["n"], oracle["paid_reads"], oracle["realized_elimination_per_read"], oracle["realized_elimination_per_read_n"]) == (4, 0, "", 0)
    causal = elim[("causal_m", "True", "declared")]
    assert (causal["paid_reads"], causal["calibrated_reads"], causal["yardstick"]) == (8, 4, "abc")
    assert causal["realized_elimination_per_read"] == 4.0 and causal["realized_elimination_per_cost"] == 1.0
    assert (causal["expected_elimination"], causal["realized_elimination_calibrated"], causal["calibration_gap"]) == (3.0, 4.0, 1.0)
    pairs = {r["hypothesis"]: r for r in tables["ELIM_pairs"]}
    assert sorted(pairs) == [name for name, _, _ in design.HYPOTHESES]
    ours = pairs["H1d"]  # the CausalDiagnosis agent against Agent-As-Tool, both with the MCP
    assert (ours["agent"], ours["reference"], ours["pairs"], ours["clusters"]) == ("causal_m", "agent_as_tool_m", 4, 2)
    assert (ours["realized_elimination_per_read_diff"], ours["realized_elimination_per_cost_diff"]) == (1.0, 0.25)
    assert pairs["H2a"]["realized_elimination_per_cost_diff"] == 0.25 and pairs["H2c"]["realized_elimination_per_read_diff"] == 1.5
    assert all("holm_p" not in r for r in tables["ELIM_pairs"])
    switches = {r["switch"]: r for r in tables["A13_A17"]}
    assert switches["full"]["realized_elimination_per_read"] == 4.0 and switches["branches_off"]["realized_elimination_per_read"] == 3.5
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    assert sorted(aggregate.REFERENCE_LABELS) == sorted(run_episodes.POLICY_AGENTS)


def test_elimination_fields_read_the_released_evidence_and_the_answers_of_the_trajectory(tmp_path, monkeypatch):
    """elimination_fields hands read_elimination the released evidence and the successful rank_root_causes answers in call
    order, and an episode without a read loads no model."""
    from CausalDiagnosis.evaluation.metrics import realized_read_elimination as read_elimination

    ep = tmp_path / "ep"
    ep.mkdir()
    released = {"rules": {"prices": {"read": {"g": 1}, "parts": {}, "repair_charge": 0}},
                "events": [{"event": "read", "t_start": 0, "t_end": 1, "cost": 1, "sensor_group": "g", "served": [{"n": 1, "t": 0, "values": {"f": 1.0}}]}]}
    (ep / "released.json").write_text(json.dumps(released), encoding="utf-8")
    answer = {"evidence_ids": [], "elimination": []}
    (ep / "trajectory.json").write_text(json.dumps([
        {"tool": "rank_root_causes", "error": None, "response": json.dumps(answer)},
        {"tool": "rank_root_causes", "error": "refused", "response": ""},
        {"tool": "read_sensors", "error": None, "response": "{}"}]), encoding="utf-8")
    got = {}

    class Stick:
        digest = "d1"

    monkeypatch.setattr(read_elimination, "read_records", lambda r, left_out: got.update(left_out=left_out) or got.setdefault("released", r) and [{"eliminated": 2.0, "cost": 4.0}])
    monkeypatch.setattr(read_elimination, "calibrate", lambda records, r, answers, left_out: got.setdefault("answers", answers))
    monkeypatch.setattr(read_elimination, "yardstick", lambda left_out: Stick())
    fields = aggregate.replayed_fields.elimination_fields({"episode_dir": ep, "block": "[100, 80, 0, 90]"})
    assert got["released"] == released and got["answers"] == [answer] and got["left_out"] == (100, 80, 0, 90)  # the start, whose block the fits leave out
    assert (fields["paid_reads"], fields["realized_elimination_per_read"], fields["realized_elimination_per_cost"], fields["yardstick"]) == (1, 2.0, 0.5, "d1")
    monkeypatch.setattr(read_elimination, "read_records", lambda r, left_out: [])
    monkeypatch.setattr(read_elimination, "yardstick", lambda entry_offset: pytest.fail("no read, so no model is loaded"))
    assert aggregate.replayed_fields.elimination_fields({"episode_dir": None})["paid_reads"] == 0


def test_switches_pair_each_arm_preset_against_the_preset_of_training_for_the_three_agents_without_the_mcp(tmp_path):
    """The small switches of layers 2, 3 and 5 are presets of all six big layers with one setting changed
    (design.ARM_PRESETS). SWITCHES pairs every arm preset against the preset of training per agent without the MCP, the thinking time
    off across the clock modes of the two presets, while a run with the MCP or at another guidance level enters no row, and an agent
    without a run on the preset of training keeps its row without a pairing. Every run plays the AssetOpsBench surface, since a model
    agent on a row over the rig is refused."""
    runs = tmp_path / "runs"
    training, scheduled, thinking_off = design.TRAINING_PRESET, "scheduled", "deterioration_thinking_off"

    def on(preset, seed, **fields):
        return _row(seed, config=preset, clock_mode=load_config(preset)[1].clock_mode, ranking_model=DEFAULT_RANKING_MODEL,
                    **{"agent": "plan_execute_m", "diagnostics_server": False, **fields})

    def meta(preset):
        return {"servers": list(METERED_SERVERS), "config_name": preset}

    _run(runs, "training", [on(training, s) for s in range(4)], meta(training))
    _run(runs, "scheduled", [on(scheduled, s, **{"pass": s < 2}) for s in range(4)], meta(scheduled))
    _run(runs, "thinking_off", [on(thinking_off, s, **{"pass": s < 3}) for s in range(4)], meta(thinking_off))
    _run(runs, "price_double", [on("deterioration_thinking_price_double", s, **{"pass": s < 1}) for s in range(4)], meta("deterioration_thinking_price_double"))
    _run(runs, "scheduled_mcp", [on(scheduled, s, diagnostics_server=True) for s in range(4)], meta(scheduled))
    _run(runs, "scheduled_g1", [on(scheduled, s, guidance_level=1) for s in range(4)], meta(scheduled))
    _run(runs, "aat_scheduled", [on(scheduled, s, agent="agent_as_tool_m") for s in range(4)], meta(scheduled))
    assert load_config(thinking_off)[1].clock_mode != load_config(training)[1].clock_mode  # the one switch the arm turns
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    switches = {(row["agent"], row["config"]): row for row in tables["SWITCHES"]}
    assert set(switches) == {("plan_execute_m", training), ("plan_execute_m", scheduled), ("plan_execute_m", thinking_off), ("agent_as_tool_m", scheduled),
                             ("plan_execute_m", "deterioration_thinking_price_double")}
    assert switches[("plan_execute_m", "deterioration_thinking_price_double")]["pass_diff"] == -0.75  # the thinking time at twice its price
    assert all(row["n"] == 4 for row in tables["SWITCHES"]) and {row["diagnostics_server"] for row in tables["SWITCHES"]} == {"False"}
    assert switches[("plan_execute_m", scheduled)]["pass_diff"] == -0.5 and switches[("plan_execute_m", thinking_off)]["pass_diff"] == -0.25
    assert switches[("plan_execute_m", thinking_off)]["pairs"] == 4 and switches[("plan_execute_m", thinking_off)]["holm_p"] != ""
    assert switches[("plan_execute_m", training)]["pass_diff"] == "" and switches[("agent_as_tool_m", scheduled)]["pass_diff"] == ""
    assert set(design.ARM_PRESETS) == {thinking_off, "deterioration_thinking_price_double", "deterioration_thinking_price_half", "deterioration_durations_off",
                                       "deterioration_mode_none", "deterioration_failure_off", scheduled, "deterioration_misdiagnosis_budget"}
    assert {ablation for ablation, _ in design.ARM_PRESETS.values()} == {"L1", "L2", "L3", "L5"} and aggregate.SWITCH_ABLATIONS == ("L2", "L3", "L5")


def test_training_has_no_column_for_a_start_solved_in_any_of_eight_samples(tmp_path):
    """The paper's training tables read pass and top-1 of every variant on the test set and sample no start eight times, so TRAINING
    holds the top-1 columns, followed by the counts of judged and aliased episodes, of the episodes without a judge record, of the
    failed judge calls and of the declaring episodes, and the judge metrics, and neither its header nor its metric names solved_any."""
    out = tmp_path / "out"
    table = next(t for t, _ in aggregate.aggregate(tmp_path / "no_runs", out, resamples=50, fraction="1") if t.id == "TRAINING")
    header = (out / "TRAINING.csv").read_text(encoding="utf-8").strip().split(",")
    top1 = header.index("diagnosis_all_correct")
    assert header == table.columns and header[top1:top1 + 7] == ["diagnosis_all_correct", "diagnosis_all_correct_low", "diagnosis_all_correct_high",
                                                                 "diagnosis_all_correct_se", "diagnosis_all_correct_n", "judged", "aliased"]
    assert header[top1 + 7:top1 + 10] == ["episodes_without_record", "failed_calls", "declared"]
    assert header[top1 + 10:top1 + 25] == [f"{c}{part}" for c in aggregate.JUDGE_CRITERIA for part in ("", "_low", "_high", "_se", "_n")]
    assert header[top1 + 25:] == [f"{m}{part}" for m in ("unmatched_evidence", "judge_kappa") for part in ("", "_low", "_high", "_se", "_n")]
    assert not any("solved_any" in text for text in (*header, table.metric, aggregate.table_training.__doc__))


# ----- the hypotheses of the six rows, the environment, training and the interval of every proportion -----

BASE_META = {"config_name": design.AGENTS_PRESET, "servers": list(METERED_SERVERS)}  # a model agent on a ladder row plays AssetOpsBench's servers


def _base_row(seed, **fields):
    """A row on the preset of the agents table at the main level of every setting, seeds 0 and 1 on one block and 2 and 3 on another."""
    return _row(seed, **{"config": design.AGENTS_PRESET, "clock_mode": "declared", "ranking_model": DEFAULT_RANKING_MODEL,
                         "block": BLOCK_A if seed < 2 else BLOCK_B, **fields})


def _hypothesis_rows(**changed):
    """The rows of the seven hypotheses and two scoring pairs with made up cells: every difference 0.2 and every p value 0.001, and the
    cells of changed[name] in place of those of a hypothesis."""
    rows = {name: {"hypothesis": name, "diagnosis_all_correct_diff": 0.2, "diagnosis_all_correct_p": 0.001, "pass_diff": 0.2, "pass_p": 0.001,
                   "success_p": 0.001, "reward_p": 0.001, **changed.get(name, {})} for name, _, _ in design.HYPOTHESES}
    rows["s1"] = {"hypothesis": "scoring", "pass_p": 0.5, "judge_pass_p": 0.6}
    rows["s2"] = {"hypothesis": "scoring", "pass_p": 0.01, "judge_pass_p": 0.02}
    return rows


def test_holm_counts_the_14_tests_of_the_hypotheses_and_the_scoring_pairs_in_a_family_of_their_own():
    """By hand: the 14 tests of top-1 and pass at p 0.001 adjust to 0.014 and all reject, while
    success and the reward enter no family, and the two scoring pairs form a family of four tests apart. A hypothesis holds when both of
    its differences lie above zero and both Holm decisions reject, fails on a difference at or below zero, and stays empty while a cell
    it needs is missing, which leaves 13 tests in the family."""
    rows = _hypothesis_rows()
    aggregate.holm_family(list(rows.values()), 0.05)
    hypotheses = [rows[name] for name, _, _ in design.HYPOTHESES]
    assert {r["family_tests"] for r in hypotheses} == {14} and {rows["s1"]["family_tests"], rows["s2"]["family_tests"]} == {4}
    assert {r["pass_holm_p"] for r in hypotheses} == {r["diagnosis_all_correct_holm_p"] for r in hypotheses} == {0.014}
    assert not any("success_holm_p" in r or "reward_holm_p" in r for r in hypotheses)
    assert (rows["s2"]["pass_holm_p"], rows["s2"]["judge_pass_holm_p"], rows["s1"]["judge_pass_holm_p"]) == (0.04, 0.06, 1.0)
    assert all(aggregate.holds(r) is True for r in hypotheses)
    rows = _hypothesis_rows(H1b={"diagnosis_all_correct_diff": -0.1}, H2c={"pass_p": None})
    aggregate.holm_family(list(rows.values()), 0.05)
    assert rows["H1a"]["family_tests"] == 13 and "pass_holm_p" not in rows["H2c"]
    assert (aggregate.holds(rows["H1a"]), aggregate.holds(rows["H1b"]), aggregate.holds(rows["H2c"])) == (True, False, "")
    rows = _hypothesis_rows(**{name: {"pass_p": 0.2} for name in ("H1a", "H2a")})  # 0.2 times 2 tests left, no rejection
    aggregate.holm_family(list(rows.values()), 0.05)
    assert rows["H1a"]["pass_reject"] is False and aggregate.holds(rows["H1a"]) is False and aggregate.holds(rows["H1c"]) is True


def test_the_hypotheses_play_on_the_preset_of_the_agents_table_alone_and_a_trained_variant_stays_out(tmp_path):
    """The hypotheses of the six rows are decided on layer 0, so AGENTS_pairs pairs the runs on the preset of the agents table alone, while the
    rows of the same agents on a later row of the ladder stay in AGENTS. A run of a trained agent (the run.json key of
    design.TRAINING_VARIANT_KEY) keeps a row of its own in AGENTS and enters no hypothesis, which pairs the agents without training."""
    runs = tmp_path / "runs"
    for name, server, passed in (("pe_bare", False, (False,) * 4), ("pe_mcp", True, (True, False, False, False))):
        _run(runs, name, [_base_row(s, agent="plan_execute_m", diagnostics_server=server, **{"pass": p}) for s, p in enumerate(passed)], BASE_META)
        _run(runs, f"{name}_priced", [_base_row(s, agent="plan_execute_m", diagnostics_server=server, config="priced", **{"pass": True}) for s in range(4)],
             {**BASE_META, "config_name": "priced"})
    _run(runs, "pe_mcp_gepa", [_base_row(s, agent="plan_execute_m", **{"pass": True}) for s in range(4)], {**BASE_META, design.TRAINING_VARIANT_KEY: "gepa"})
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    agents = sorted((r["config"], r["diagnostics_server"], r["training_variant"], r["pass"]) for r in tables["AGENTS"])
    assert agents == [(design.AGENTS_PRESET, "False", "as_released", 0.0), (design.AGENTS_PRESET, "True", "as_released", 0.25),
                      (design.AGENTS_PRESET, "True", "gepa", 1.0), ("priced", "False", "as_released", 1.0), ("priced", "True", "as_released", 1.0)]
    pair, = (r for r in tables["AGENTS_pairs"] if r["hypothesis"] != "scoring")
    assert (pair["hypothesis"], pair["config"], pair["training_variant"], pair["pass_diff"], pair["pairs"]) == ("H2a", design.AGENTS_PRESET, "as_released", 0.25, 4)
    assert {r["config"] for r in tables["AGENTS_pairs"]} == {design.AGENTS_PRESET}
    assert {r["training_variant"] for r in tables["RUNS"]} == {"as_released", "gepa"}


def test_every_proportion_takes_the_interval_of_the_bootstrap_clustered_by_block(tmp_path):
    """Four blocks of three episodes each, two blocks passing every episode and two failing every one, so
    the share is 0.5 and its interval comes from stats.cluster_proportion over the blocks, far wider than the Wilson interval of 6 of 12,
    which would take the twelve episodes for independent."""
    runs = tmp_path / "runs"
    _run(runs, "row", [_base_row(s, agent="plan_execute_m", block=_start(s // 3), **{"pass": s // 3 < 2}) for s in range(12)], BASE_META)
    row, = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200, fraction="1")}["AGENTS"]
    clustered = cluster_proportion([s // 3 < 2 for s in range(12)], [_start(s // 3) for s in range(12)], resamples=200, seed=0)
    assert (row["pass"], row["pass_low"], row["pass_high"]) == (0.5, round(clustered["low"], 6), round(clustered["high"], 6))
    wilson = aggregate.__dict__.get("wilson")
    assert wilson is None  # no table reads a Wilson interval any more
    from CausalDiagnosis.evaluation.metrics.table_statistics import wilson as wilson_interval
    narrow = wilson_interval(6, 12)
    assert row["pass_high"] - row["pass_low"] > narrow["high"] - narrow["low"]


def test_policies_hold_the_reference_policies_per_preset_with_pass_success_and_reward(tmp_path):
    """POLICIES: one row per reference policy and preset at the main level of every setting, the random policy among them, while a policy
    run at another guidance level and a model agent enter no row."""
    runs = tmp_path / "runs"
    for name, agent, config, passed, reward in (("oracle_base", "oracle", design.AGENTS_PRESET, True, 0.9), ("nothing_base", "do_nothing", design.AGENTS_PRESET, False, -0.2),
                                                ("random_priced", "random", "priced", False, 0.1)):
        _run(runs, name, [_base_row(s, agent=agent, model="", config=config, success=passed, reward=reward if s < 2 else reward - 0.2, **{"pass": passed})
                          for s in range(4)], {"config_name": config, "model": None})
    _run(runs, "oracle_g1", [_base_row(s, agent="oracle", model="", guidance_level=1) for s in range(4)], {"config_name": design.AGENTS_PRESET, "model": None})
    _run(runs, "pe", [_base_row(s, agent="plan_execute_m") for s in range(4)], BASE_META)
    rows = {(r["agent"], r["config"]): r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["POLICIES"]}
    assert set(rows) == {("oracle", design.AGENTS_PRESET), ("do_nothing", design.AGENTS_PRESET), ("random", "priced")}
    oracle, nothing, random_policy = rows[("oracle", design.AGENTS_PRESET)], rows[("do_nothing", design.AGENTS_PRESET)], rows[("random", "priced")]
    assert (oracle["n"], oracle["pass"], oracle["success"], oracle["reward"], oracle["clock_mode"]) == (4, 1.0, 1.0, 0.8, "declared")
    assert (nothing["pass"], nothing["reward"], random_policy["pass"], random_policy["reward"]) == (0.0, -0.3, 0.0, 0.0)
    assert {label for label, _ in design.REFERENCE_POLICIES} == set(aggregate.REFERENCE_LABELS) and "random" in aggregate.REFERENCE_LABELS


def test_layers_pairs_pair_each_layer_against_the_row_below_and_pool_the_three_agents(tmp_path):
    """LAYERS_pairs, by hand on four seeds in two blocks. Adding prices: Plan-Execute loses seed 0 (-0.25), Agent-As-Tool keeps its two passes
    (0), the CausalDiagnosis agent loses seeds 0 to 2 (-0.75), so the twelve pooled pairs differ by -4 / 12. Adding the real-time clock,
    Plan-Execute alone gains seeds 0 and 1 (+0.5). Holm runs over the two pooled tests, and a row without the MCP enters no pair, since the
    ladder plays the rows with it (design.LADDER_ROWS)."""
    runs = tmp_path / "runs"
    passes = {("plan_execute", design.AGENTS_PRESET): (True, False, False, False), ("agent_as_tool", design.AGENTS_PRESET): (True, True, False, False),
              ("causal", design.AGENTS_PRESET): (True, True, True, False), ("plan_execute", "priced"): (False,) * 4,
              ("agent_as_tool", "priced"): (True, True, False, False), ("causal", "priced"): (False,) * 4, ("plan_execute", "real_time"): (True, True, False, False)}
    for (prefix, config), passed in passes.items():
        clock = "generation_time" if config == "real_time" else "declared"
        _run(runs, f"{prefix}_{config}", [_base_row(s, agent=f"{prefix}_m", config=config, clock_mode=clock, **{"pass": p})
                                          for s, p in enumerate(passed)], {**BASE_META, "config_name": config})
    _run(runs, "pe_bare_priced", [_base_row(s, agent="plan_execute_m", diagnostics_server=False, config="priced", **{"pass": True}) for s in range(4)],
         {**BASE_META, "config_name": "priced"})
    rows = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["LAYERS_pairs"]
    by = {(r["config"], r["agent"]): r for r in rows}
    assert set(by) == {("priced", "plan_execute_m"), ("priced", "agent_as_tool_m"), ("priced", "causal_m"), ("priced", "pooled"),
                       ("real_time", "plan_execute_m"), ("real_time", "pooled")}
    assert [by[("priced", a)]["pass_diff"] for a in ("plan_execute_m", "agent_as_tool_m", "causal_m", "pooled")] == [-0.25, 0.0, -0.75, round(-4 / 12, 6)]
    assert (by[("priced", "pooled")]["pairs"], by[("priced", "pooled")]["clusters"], by[("priced", "pooled")]["reference_config"]) == (12, 2, design.AGENTS_PRESET)
    assert (by[("real_time", "pooled")]["pass_diff"], by[("real_time", "pooled")]["pairs"], by[("real_time", "pooled")]["reference_config"]) == (0.5, 4, "priced")
    assert all(by[(c, "pooled")]["holm_p"] != "" and by[(c, "pooled")]["reject"] in (True, False) for c in ("priced", "real_time"))
    assert all("holm_p" not in r for (_, agent), r in by.items() if agent != "pooled")
    assert [preset for preset in design.LADDER[1:]] == ["priced", "real_time", "advanced_real_time", "repair_rules", "deterioration"]


def _training_run(runs, name, agent, server, passed, variant=None, seeds=None, **meta):
    meta = {**BASE_META, "config_name": design.TRAINING_PRESET, **({design.TRAINING_VARIANT_KEY: variant} if variant else {}), **meta}
    seeds = list(seeds) if seeds is not None else list(range(len(passed)))
    return _run(runs, name, [_base_row(s, agent=agent, diagnostics_server=server, config=design.TRAINING_PRESET, clock_mode="generation_time",
                                       n_faulty=1 if s % 10 < 2 else 2, diagnosis_all_correct=p, **{"pass": p}) for s, p in zip(seeds, passed)], meta)


def _two_folds(monkeypatch):
    """The fold rule of the tests in place of training.fold_of: the combination of block A, seeds 0 and 1, lies in fold 1 and that of
    block B, seeds 2 and 3, in fold 2."""
    monkeypatch.setattr(aggregate.training_tables, "fold_of", lambda combination, asset_id=None, split=None: 1 if list(combination) == json.loads(BLOCK_A) else 2)


def _cross_fitted(runs, name, agent, server, passed, variant, by_fold=None):
    """A trained variant under two-fold cross-fitting as two runs, the model that holds out fold 1 playing seeds 0 and 1 of fold 1 and
    the model that holds out fold 2 playing seeds 2 and 3 of fold 2, each with the run.json fields that by_fold
    gives its fold."""
    for fold, seeds in ((1, (0, 1)), (2, (2, 3))):
        _training_run(runs, f"{name}_f{fold}", agent, server, [passed[s] for s in seeds], variant, seeds=seeds, training_fold=fold, **(by_fold or {}).get(fold, {}))


def test_the_training_hypotheses_decide_on_every_episode_and_print_the_strata_and_the_folds_beside_them(tmp_path, monkeypatch):
    """The training table and its hypotheses, by hand on seeds 0 and 1, single fault starts of fold 1 on one block, and 2 and 3,
    two fault starts of fold 2 on another: Plan-Execute with the MCP passes seed 0 without training and seeds 0 to 2 with GEPA, so all
    reads 0.25 and 0.75, single 0.5 and 1.0, two 0.0 and 0.5; the CausalDiagnosis agent with the MCP passes seeds 0 and 1 fine tuned and
    seed 0 under its control, fine-tuning on unfiltered trajectories. Each trained variant plays each fold with the model that holds it out. Each comparison holds
    a row over all four seeds, the decision, with Holm over the registered family of thirty tests, of which these rows hold four (GEPA
    differs by 0.5 on each of the two blocks, the exact sign flip p = 2 / 4, which Holm raises to one),
    and a row on each stratum and each fold beside it, with its interval and no test. The untrained row without the MCP on the preset of
    training enters TRAINING as the baseline and no comparison, and a run on another preset enters no table of training."""
    _two_folds(monkeypatch)
    runs = tmp_path / "runs"
    _training_run(runs, "pe", "plan_execute_m", True, (True, False, False, False))
    _cross_fitted(runs, "pe_gepa", "plan_execute_m", True, (True, True, True, False), "gepa")
    _cross_fitted(runs, "cd_ft", "causal_m", True, (True, True, False, False), "ft")
    _cross_fitted(runs, "cd_ft_unfiltered", "causal_m", True, (True, False, False, False), "ft_unfiltered")
    _training_run(runs, "pe_bare", "plan_execute_m", False, (True,) * 4)
    _run(runs, "pe_base", [_base_row(s, agent="plan_execute_m", **{"pass": True}) for s in range(4)], BASE_META)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    training = {(r["agent"], r["training_variant"], r["subset"]): r for r in tables["TRAINING"] if r["diagnostics_server"] in (True, "True")}
    assert len(training) == 12 and len(tables["TRAINING"]) == 15 and {r["config"] for r in tables["TRAINING"]} == {design.TRAINING_PRESET} and aggregate.TRAINING_SUBSETS == ("all", "single", "two")
    baseline = [r for r in tables["TRAINING"] if r["diagnostics_server"] in (False, "False")]
    assert [(r["agent"], r["training_variant"], r["subset"], r["pass"]) for r in baseline] == [("plan_execute_m", "as_released", s, 1.0) for s in ("all", "single", "two")]
    assert [training[("plan_execute_m", v, s)]["pass"] for v in ("as_released", "gepa") for s in ("all", "single", "two")] == [0.25, 0.5, 0.0, 0.75, 1.0, 0.5]
    assert training[("plan_execute_m", "gepa", "two")]["n"] == 2 and training[("causal_m", "ft", "single")]["diagnosis_all_correct"] == 1.0
    subsets = ("all", "single", "two", "fold_1", "fold_2")
    assert aggregate.training_tables.training_pair_subsets() == subsets
    pairs = {(r["agent"], r["training_variant"], r["reference_variant"], r["subset"]): r for r in tables["TRAINING_pairs"]}
    assert list(pairs) == [(agent, variant, reference, subset) for agent, variant, reference in (("plan_execute_m", "gepa", "as_released"), ("causal_m", "ft", "ft_unfiltered"))
                           for subset in subsets]
    gepa = {subset: pairs[("plan_execute_m", "gepa", "as_released", subset)] for subset in subsets}
    assert [(gepa[s]["pass_diff"], gepa[s]["diagnosis_all_correct_diff"], gepa[s]["pairs"], gepa[s]["clusters"]) for s in subsets] == [(0.5, 0.5, 4, 2)] + [(0.5, 0.5, 2, 1)] * 4
    assert {gepa[s]["comparison"] for s in gepa} == {"GEPA against no training"}
    assert (gepa["all"]["pass_p"], gepa["all"]["pass_holm_p"], gepa["all"]["diagnosis_all_correct_holm_p"], gepa["all"]["family_tests"]) == (0.5, 1.0, 1.0, 30)
    for subset in subsets[1:]:
        assert gepa[subset]["pass_diff_low"] != "" and all(gepa[subset].get(column, "") == "" for column in ("pass_p", "diagnosis_all_correct_p", "pass_holm_p", "pass_reject",
                                                                                                           "diagnosis_all_correct_holm_p", "family_tests"))
    ft = {subset: pairs[("causal_m", "ft", "ft_unfiltered", subset)] for subset in subsets}
    assert (ft["all"]["comparison"], *(ft[s]["pass_diff"] for s in subsets)) == ("fine-tuning against fine-tuning on unfiltered trajectories", 0.25, 0.5, 0.0, 0.5, 0.0)
    assert ft["all"]["pass_holm_p"] != "" and {r["family_tests"] for r in tables["TRAINING_pairs"] if r["subset"] == "all"} == {30}
    assert len(design.TRAINING_ROWS) * len(design.TRAINING_COMPARISONS) * len(design.TRAINING_MEASURES) == 30


def test_a_baseline_row_on_other_seeds_than_its_agent_with_the_mcp_stops_the_script(tmp_path, monkeypatch):
    """The baseline of the training table, Plan-Execute without the MCP on the preset of training, is read beside Plan-Execute with
    the MCP without training on the same seeds. On seeds 0 to 3 against 0 to 3 it prints, on seeds 1 to 4 the script stops and names it."""
    _two_folds(monkeypatch)
    runs = tmp_path / "runs"
    _training_run(runs, "pe", "plan_execute_m", True, (True, False, False, False))
    _training_run(runs, "pe_bare", "plan_execute_m", False, (False,) * 4)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    assert {(r["diagnostics_server"] in (True, "True"), r["subset"]) for r in tables["TRAINING"]} == {(s, k) for s in (True, False) for k in ("all", "single", "two")}
    other = tmp_path / "other"
    _training_run(other, "pe", "plan_execute_m", True, (True, False, False, False))
    _training_run(other, "pe_bare", "plan_execute_m", False, (False,) * 4, seeds=(1, 2, 3, 4))
    with pytest.raises(ValueError, match="the baseline row of plan_execute_m without the MCP plays 4 seeds and its untrained row with the MCP 4, 2 of them"):
        aggregate.aggregate(other, tmp_path / "out2", resamples=50, fraction="1")


def test_a_trained_run_that_names_no_fold_or_plays_outside_the_fold_its_model_holds_out_stops_the_script(tmp_path, monkeypatch):
    """Every test episode is played by a model that never saw its combination, so a trained run must name the
    fold its model holds out and play only that fold, and the agent as released, which trained on nothing, names none."""
    _two_folds(monkeypatch)
    runs = tmp_path / "runs"
    _training_run(runs, "pe", "plan_execute_m", True, (True,) * 4)
    _training_run(runs, "pe_gepa_f1", "plan_execute_m", True, (True, True), "gepa", seeds=(2, 3), training_fold=1)
    with pytest.raises(ValueError, match="run pe_gepa_f1 of the variant gepa names the held-out fold 1 and plays an episode of fold 2"):
        aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")
    unnamed = tmp_path / "unnamed"
    _training_run(unnamed, "pe", "plan_execute_m", True, (True,) * 4)
    _training_run(unnamed, "pe_ft", "plan_execute_m", True, (True,) * 4, "ft")
    with pytest.raises(ValueError, match="run pe_ft of the variant ft names the held-out fold None"):
        aggregate.aggregate(unnamed, tmp_path / "out2", resamples=50, fraction="1")


FT_RECORD = {"episodes_sampled": 400, "kept_by_filter": 120, "examples": 900, "examples_over_max_length": 7, "optimizer_steps": 30, "control_trajectories": 120}
GEPA_RECORD = {"candidates_played": 12, "stream_episodes": 300}


def test_the_training_record_reads_run_json_of_the_trained_runs_per_fold_and_refuses_runs_that_disagree(tmp_path, monkeypatch):
    """The budget table from run.json of the trained evaluation runs, where the training pipeline records the adapter and the task block
    that the model of a fold plays (design.TRAINING_RECORD_SOURCES): one row per agent and fold, the fine tuned runs carrying
    adapter.training and the GEPA runs task_block, a field that no run of an agent and fold carries stays empty, an agent without a
    trained run has no row, and two runs of one agent and fold that disagree on a field stop the script."""
    _two_folds(monkeypatch)
    runs = tmp_path / "runs"
    control = {**FT_RECORD, "kept_by_filter": 400, "examples": 950, "examples_over_max_length": 0, "optimizer_steps": 31}
    _training_run(runs, "cd", "causal_m", True, (True,) * 4)
    _cross_fitted(runs, "cd_ft", "causal_m", True, (True,) * 4, "ft", {1: {"adapter": {"training": FT_RECORD}}, 2: {"adapter": {"training": {**FT_RECORD, "episodes_sampled": 410}}}})
    _cross_fitted(runs, "cd_gepa", "causal_m", True, (True,) * 4, "gepa", {fold: {"task_block": {**GEPA_RECORD, "stream_episodes": 300 + fold}} for fold in (1, 2)})
    _cross_fitted(runs, "pe_control", "plan_execute_m", True, (True,) * 4, "ft_unfiltered", {fold: {"adapter": {"training": control}} for fold in (1, 2)})
    _training_run(runs, "aat", "agent_as_tool_m", True, (True,) * 4)
    rows = {(r["agent"], r["fold"]): r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["TRAINING_RECORD"]}
    assert set(rows) == {("causal_m", 1), ("causal_m", 2), ("plan_execute_m", 1), ("plan_execute_m", 2)}
    fields = ("episodes_sampled", "kept_by_filter", "model_calls_as_examples", "examples_over_max_length", "optimizer_steps", "control_trajectories", "gepa_candidates",
              "gepa_stream_episodes")
    assert [rows[("causal_m", 1)][field] for field in fields] == [400, 120, 900, 7, 30, 120, 12, 301]
    assert [rows[("causal_m", 2)][field] for field in fields] == [410, 120, 900, 7, 30, 120, 12, 302]
    assert rows[("causal_m", 1)]["control_model_calls_as_examples"] == rows[("causal_m", 1)]["ft_gepa_candidates"] == ""
    pe = rows[("plan_execute_m", 2)]
    assert [pe[field] for field in ("episodes_sampled", "kept_by_filter", "control_model_calls_as_examples", "control_optimizer_steps", "model_calls_as_examples")] == [400, 400, 950, 31, ""]
    assert set(design.BUDGET_FIELDS) <= set(design.TRAINING_RECORD_FIELDS) == set(design.TRAINING_RECORD_SOURCES) | {f"gate_{field}" for field in design.GATE_FIELDS}
    _training_run(runs, "cd_ft_other", "causal_m", True, (True, True), "ft", seeds=(12, 13), training_fold=2, adapter={"training": {**FT_RECORD, "episodes_sampled": 401}})
    with pytest.raises(ValueError, match="the runs of causal_m on fold 2 disagree on episodes_sampled of the budget table"):
        aggregate.aggregate(runs, tmp_path / "out2", resamples=50, fraction="1")


def test_the_sampling_runs_of_fine_tuning_enter_no_table(tmp_path):
    """A run of mode training samples trajectories for fine tuning, which no table reads, RUNS included, while a gym run beside it does."""
    runs = tmp_path / "runs"
    _run(runs, "sampling", [_base_row(s, agent="causal_m") for s in range(4)], {**BASE_META, "mode": design.TRAINING_MODE})
    _run(runs, "gym", [_base_row(s, agent="causal_m") for s in range(4)], BASE_META)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    assert [r["run"] for r in tables["RUNS"]] == ["gym"] and [r["n"] for r in tables["AGENTS"]] == [4]
    assert {r["n"] for r in tables["A8"]} == {4}
    assert pooling_rules.refused(runs / "sampling") == pooling_rules.TRAINING_SAMPLE and pooling_rules.refused(runs / "gym") is None


def test_a_run_of_chosen_starts_that_another_rule_keeps_out_takes_that_rule_and_gives_its_root_no_kind(tmp_path):
    """refused() names a run of chosen starts only when no other rule keeps the run out, so a run of --start-state on the memory
    backend stays out as a run on the memory backend does and gives its root no kind, while a run of chosen starts on CouchDB is
    one of chosen starts and gives its root the kind of its stamp."""
    chosen = {**BASE_META, "start_state": "1/2@0", "stamp": pooling_rules.STAMP}
    _run(tmp_path / "memory", "chosen", [_base_row(s) for s in range(2)], {**chosen, "backend": {"kind": "memory"}})
    _run(tmp_path / "couchdb", "chosen", [_base_row(s) for s in range(2)], chosen)
    assert pooling_rules.refused(tmp_path / "memory" / "chosen") == pooling_rules.MEMORY_BACKEND
    assert pooling_rules.root_kind(tmp_path / "memory") == "empty"
    assert pooling_rules.refused(tmp_path / "couchdb" / "chosen") == pooling_rules.CHOSEN_STARTS
    assert pooling_rules.root_kind(tmp_path / "couchdb") == "development"


def test_the_steps_of_the_execution_efficiency_count_every_attempted_tool_call_and_no_other_counter_of_a_runner():
    """The steps of the execution efficiency are the tool_calls cell plus the causal_calls cell under one rule for every agent, every
    attempted call, so a counter of a runner's own, such as the supervisor's ReAct steps of the ported Agent-As-Tool, which only the
    acceptance check of the reproduction reads, never enters the steps, whatever column or key carries it."""
    assert aggregate.tool_invocations({"tool_calls": "5", "causal_calls": "2"}) == 7
    assert aggregate.tool_invocations({"tool_calls": "5", "causal_calls": "2", "react_steps": "3", "supervisor_steps": "9"}) == 7
    assert aggregate.tool_invocations({"tool_calls": "5", "causal_calls": ""}) == 5 and aggregate.tool_invocations({"tool_calls": ""}) is None
    assert aggregate.EFFICIENCY_METRICS == ("tool_invocations", "wall_s")


def test_a_column_of_a_runner_s_own_counter_in_scores_csv_changes_no_table(tmp_path):
    """The ported Agent-As-Tool appends its supervisor's ReAct counter to scores.csv as react_steps, which
    only the acceptance check of the reproduction reads, so two runs that differ by that column alone give the same rows in every table."""
    tables = []
    for name, extra in (("plain", None), ("counted", "react_steps")):
        runs = tmp_path / name / "runs"
        rows = [_base_row(s, agent="agent_as_tool_m", tool_calls=3 + s, causal_calls=1, wall_s=100 + s, n_faulty=1 if s < 2 else 2) for s in range(4)]
        run_dir = _run(runs, "aat_mcp", rows, BASE_META)
        if extra:
            write_rows(run_dir / "scores.csv", [{**row, extra: 9} for row in rows], [*COLUMNS, extra])
        tables.append({t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / name / "out", resamples=50, fraction="1")})
    plain, counted = tables
    efficiency = {r["faults"]: r["tool_invocations"] for r in counted["EFFICIENCY"]}
    assert efficiency == {"single": 4.5, "two": 6.5}
    assert all(plain[table] == counted[table] for table in plain if table != "RUNS")


def test_the_judge_kappa_reads_the_judged_episodes_and_the_unmatched_share_is_the_complement_of_provenance():
    """Cohen's kappa of the judge's majority pass under the five call protocol against the verified pass, over the episodes that carry
    both, and empty when either rule puts every episode on one side; an episode cites unmatched evidence when its provenance check failed."""
    ctx = {"resamples": 20, "seed": 0}
    judged = [{"judge_record_protocol": "paper", "judge_pass": j, "pass": v, "block": f"b{i}"}
              for i, (j, v) in enumerate([("True", "True"), ("True", "False"), ("False", "False"), ("False", "False")])]
    # observed agreement 3/4, chance 1/2 x 1/4 + 1/2 x 3/4 = 1/2, so kappa (3/4 - 1/2) / (1 - 1/2) = 1/2
    assert {k: aggregate.judge_kappa(judged, ctx)[k] for k in ("judge_kappa", "judge_kappa_n")} == {"judge_kappa": 0.5, "judge_kappa_n": 4}
    others = [{"judge_record_protocol": "single", "judge_pass": "True", "pass": "False", "block": "x"},
              {"judge_record_protocol": "paper", "judge_pass": "True", "pass": "", "block": "y"}, {"pass": "True", "block": "z"}]
    assert aggregate.judge_kappa(judged + others, ctx)["judge_kappa"] == 0.5 and aggregate.judge_kappa(judged + others, ctx)["judge_kappa_n"] == 4
    one_sided = [{**e, "judge_pass": "True"} for e in judged]
    assert aggregate.judge_kappa(one_sided, ctx)["judge_kappa"] == "" and aggregate.judge_kappa([], ctx)["judge_kappa_n"] == 0
    assert [aggregate.unmatched_evidence({"provenance_ok": cell}) for cell in ("False", "True", "")] == [True, False, ""]
    assert "unmatched_evidence" in aggregate.PROPORTIONS and "unmatched_evidence" in aggregate.AGENT_METRICS


def test_every_p_value_of_a_paired_difference_is_the_sign_flip_p_over_the_blocks_and_the_bootstrap_gives_the_interval():
    """A paired difference of one sign on every one of five blocks has the exact sign flip p value
    2 / 32 whatever the number of resamples, which the bootstrap's add one share would have put at 2 / 51 under 50 resamples."""
    level = [{"seed": s, "block": f"b{s}", "pass": "True"} for s in range(5)]
    reference = [{"seed": s, "block": f"b{s}", "pass": "False"} for s in range(5)]
    pairs = list(zip(level, reference))
    for resamples in (50, 400):
        cells = aggregate.paired_difference(pairs, "pass", {"resamples": resamples, "seed": 0})
        assert (cells["pass_diff"], cells["pass_diff_low"], cells["pass_diff_high"], cells["pass_p"], cells["pairs"], cells["clusters"]) == (1.0, 1.0, 1.0, 0.0625, 5, 5)
    assert not hasattr(aggregate, "pooled_pairs")


def _h_row(seed, agent, server, block, config=design.TRAINING_PRESET, **cells):
    """A row at the main level of every setting of its preset, the clock mode of training on the top rung, each start a block of its own."""
    return _base_row(seed, agent=agent, diagnostics_server=server, config=config, block=block,
                     clock_mode="generation_time" if config == design.TRAINING_PRESET else "declared", **cells)


def test_h2_prime_pairs_each_agent_with_the_mcp_against_itself_without_it_on_the_top_rung_with_a_holm_of_its_own(tmp_path):
    """On the top rung, every agent with the MCP passes four starts on four blocks that it fails without the MCP and earns a
    higher reward, so each pass difference and each reward difference is one on every block, with the exact sign flip p 2 / 16, which
    Holm over the fourteen tests of its group of primary, its own six and the eight of h1_top, raises to one; holds needs both Holm
    decisions to reject, so no agent holds here."""
    runs = tmp_path / "runs"
    meta = {**BASE_META, "config_name": design.TRAINING_PRESET}
    for prefix in ("plan_execute", "agent_as_tool", "causal"):
        for server, passes, reward in ((True, True, 1.0), (False, False, 0.0)):
            _run(runs, f"{prefix}_{server}", [_h_row(s, f"{prefix}_m", server, _start(s), reward=reward, **{"pass": passes}) for s in range(4)], meta)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    h2p = {r["hypothesis"]: r for r in tables["H2PRIME_pairs"]}
    assert list(h2p) == ["H2'a", "H2'b", "H2'c"] and [name for name, _, _ in design.H2_PRIME] == list(h2p)
    for r in h2p.values():
        assert (r["pass_diff"], r["reward_diff"], r["pass_p"], r["reward_p"], r["pairs"], r["clusters"]) == (1.0, 1.0, 0.125, 0.125, 4, 4)
        # superseded pin: Holm over the six tests of h2_prime alone; the registered file corrects h2_prime together with h1_top
        assert (r["pass_holm_p"], r["family_tests"], r["holds"], r["config"]) == (1.0, 14, False, design.TRAINING_PRESET)
    assert {r["family_tests"] for r in tables["H1_TOP_pairs"]} == {14}
    assert aggregate.helps_on_every_measure({"pass_diff": 0.2, "reward_diff": 0.1, "pass_reject": True, "reward_reject": True}) is True
    assert aggregate.helps_on_every_measure({"pass_diff": 0.2, "reward_diff": -0.1, "pass_reject": True, "reward_reject": True}) is False
    assert aggregate.helps_on_every_measure({"pass_diff": 0.2, "pass_reject": True}) == ""


def test_h0_tests_rows_and_comparisons_in_one_family_with_the_unrun_rows_counted(tmp_path):
    """H0, by hand: the CausalDiagnosis agent without the MCP passes nothing while the judge passes every start, so
    J - V is one on each of its starts; Plan-Execute without the MCP agrees with the judge everywhere. The CausalDiagnosis agent with the
    MCP lacks a judge record on one start, which counts as a failure of the judge, so its row test reads J - V = -1 on that start alone,
    and the comparison H2c that uses it runs on every start: three row tests and the comparisons H1a and H2c, the two whose arms both
    ran, five of the thirteen tests of the registered family, each with its computed p value. Twelve starts of one sign give the exact
    sign flip p 2 / 4096, which Holm over the thirteen keeps below 0.05."""
    runs = tmp_path / "runs"
    meta = {**BASE_META, "config_name": design.AGENTS_PRESET}
    seeds = range(12)

    def judged_run(name, agent, server, passes, judge, missing=()):
        run = _run(runs, name, [_h_row(s, agent, server, _start(s), config=design.AGENTS_PRESET, judge_pass=judge, **{"pass": passes}) for s in seeds], meta)
        for s in seeds:
            if s not in missing:
                (run / f"seed{s:06d}" / "hidden" / "judge.json").write_text(json.dumps({"format": 1, "protocol": "paper", "calls_planned": 1, "judge_model": "j",
                                                                                         "calls": [{"passed": judge, "details": {"task_completion": judge}, "error": None}]}), encoding="utf-8")
    judged_run("cd", "causal_m", False, False, True)
    judged_run("pe", "plan_execute_m", False, True, True)
    judged_run("cd_mcp", "causal_m", True, True, True, missing=(3,))
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=5000, fraction="1")}
    h0 = {(r["test"], r["name"]): r for r in tables["H0"]}
    m, size = len(h0), len(design.AGENT_ROWS) + len(design.HYPOTHESES)
    assert m == 3 + 2 and {r["family_tests"] for r in h0.values()} == {size} == {aggregate.registered_size({}, "h0")}
    assert sorted(name for test, name in h0 if test == "comparison") == ["H1a", "H2c"]
    cd = h0[("row", design.CD[2])]
    assert (cd["statistic"], cd["nonzero_starts"], cd["status"], cd["false_pass"], cd["false_fail"], cd["discordance"]) == (1.0, 12, "tested", 1.0, "", 1.0)
    assert cd["h0_p"] == round(2 / 4096, 6) and cd["h0_reject"] is True and cd["judge_kappa"] == ""
    pe = h0[("row", design.PE[2])]
    assert (pe["statistic"], pe["nonzero_starts"], pe["status"], pe["h0_p"], pe["h0_reject"]) == (0.0, 0, "tested", 1.0, False)
    h1a = h0[("comparison", "H1a")]
    # the judge passes both rows everywhere, the verifier only Plan-Execute: theta = (1 - 1) - (0 - 1) = 1 on every pair, reversed is false
    # since the judge's difference is zero
    assert (h1a["statistic"], h1a["judge_diff"], h1a["verified_diff"], h1a["reversed"], h1a["status"]) == (1.0, 0.0, -1.0, False, "tested")
    # the start without a judge record reads J = 0, so J - V is -1 there and 0 on the eleven others, one start of a nonzero sum
    cd_mcp = h0[("row", design.CD_MCP[2])]
    assert (cd_mcp["statistic"], cd_mcp["n"], cd_mcp["nonzero_starts"], cd_mcp["status"], cd_mcp["h0_p"], cd_mcp["h0_reject"]) == (
        round(-1 / 12, 6), 12, 1, "tested", 1.0, False)
    # H2c, the agent with the MCP against itself without it: the judge's paired difference is 1 - 1 on eleven starts and 0 - 1 on the one
    # without a record, the verified one 1 - 0 everywhere, so theta is -1 on eleven pairs and -2 on one, and the two differences have
    # opposite signs
    h2c = h0[("comparison", "H2c")]
    assert (h2c["statistic"], h2c["judge_diff"], h2c["verified_diff"], h2c["reversed"], h2c["nonzero_starts"], h2c["status"], h2c["h0_reject"]) == (
        round(-13 / 12, 6), round(-1 / 12, 6), 1.0, True, 12, "tested", True)


def test_an_h0_test_is_unrun_on_a_row_that_misses_a_planned_episode_on_another_protocol_and_without_any_judge_record(tmp_path):
    """A row that misses a planned episode leaves every H0 test that reads it unrun, in the words of the tests of the agents
    (row_status), a row with an episode judged under another protocol and a row without any judge record leave their tests unrun
    too, and each such test counts with p = 1 in the family of thirteen."""
    runs = tmp_path / "runs"
    seeds = range(12)

    def run(name, agent, server, protocols, planned=None):
        meta = {**BASE_META, "config_name": design.AGENTS_PRESET, **({"episodes_planned": planned} if planned else {})}
        folder = _run(runs, name, [_h_row(s, agent, server, _start(s), config=design.AGENTS_PRESET, judge_pass=True, **{"pass": True}) for s in seeds], meta)
        for s, protocol in zip(seeds, protocols):
            (folder / f"seed{s:06d}" / "hidden" / "judge.json").write_text(json.dumps({"format": 1, "protocol": protocol, "calls_planned": 1, "judge_model": "j",
                                                                                        "calls": [{"passed": True, "details": {"task_completion": True}, "error": None}]}), encoding="utf-8")
    run("cd", "causal_m", False, ["paper"] * 12, planned=13)
    run("pe", "plan_execute_m", False, ["single"] + ["paper"] * 11)
    run("cd_mcp", "causal_m", True, [])
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200, fraction="1")}
    h0 = {(r["test"], r["name"]): r for r in tables["H0"]}
    assert {key: row["status"] for key, row in h0.items()} == {
        ("row", design.CD[2]): "incomplete, 12 of 13", ("row", design.PE[2]): "unrun, judged under another protocol",
        ("row", design.CD_MCP[2]): "unrun, no judge record", ("comparison", "H1a"): "incomplete, 12 of 13", ("comparison", "H2c"): "incomplete, 12 of 13"}
    assert {(row["n"], row["h0_p"], row["h0_reject"], row["family_tests"]) for row in h0.values()} == {(0, 1.0, False, len(design.AGENT_ROWS) + len(design.HYPOTHESES))}


def test_the_floor_of_h1_compares_the_baseline_pass_rate_with_the_registered_share_as_exact_fractions():
    """A baseline below the registered floor of pass is at floor; one exactly at it is not, which exact fractions keep for any n."""
    assert aggregate.registered_floor(None) == Fraction(1, 20) and not hasattr(design, "H1_FLOOR")
    rows = lambda k, n: [{"pass": str(i < k)} for i in range(n)]  # noqa: E731 - k passes among n episodes
    assert aggregate.floor_cells(rows(0, 47)) == {"reference_pass_k": 0, "reference_pass_n": 47, "at_floor": True}
    assert aggregate.floor_cells(rows(1, 20))["at_floor"] is False and aggregate.floor_cells(rows(11, 235))["at_floor"] is True
    assert aggregate.floor_cells(rows(12, 235))["at_floor"] is False and aggregate.floor_cells([])["at_floor"] == ""


def test_h1_under_guidance_level_three_is_a_robustness_check_with_a_holm_of_its_own(tmp_path):
    """The En7 runs at guidance level 3 of the CausalDiagnosis agent and of Plan-Execute, both without the MCP, the former
    passing and diagnosing four starts on four blocks that the latter fails, give H1a with differences of one, the exact sign flip p
    2 / 16 and a Holm over its two tests alone; the runs at the main level enter AGENTS_pairs and no row here."""
    runs = tmp_path / "runs"
    for agent, outcome in (("causal_m", True), ("plan_execute_m", False)):
        for level in (2, 3):
            _run(runs, f"{agent}_{level}", [_base_row(s, agent=agent, diagnostics_server=False, block=_start(s), guidance_level=level, diagnosis_all_correct=outcome,
                                                      **{"pass": outcome}) for s in range(4)], BASE_META)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    h1a, = tables["H1_GUIDANCE"]
    assert (h1a["hypothesis"], h1a["guidance_level"], h1a["pass_diff"], h1a["pass_p"], h1a["pass_holm_p"], h1a["family_tests"]) == ("H1a", 3, 1.0, 0.125, 0.25, 2)
    assert (h1a["reference_pass_k"], h1a["reference_pass_n"], h1a["at_floor"], h1a["holds"]) == (0, 4, True, False)
    main, = [r for r in tables["AGENTS_pairs"] if r["hypothesis"] == "H1a"]
    assert main["at_floor"] is True and main["family_tests"] == aggregate.registered_size({}, "agents")


def _kaplan_meier_restricted_mean(records, cap):
    """The area under the Kaplan Meier curve of the turn of a correct diagnosis from turn zero to cap, over (turn, observed)
    records, the reference that the turns of AGENTS are checked against."""
    pairs = sorted(records)
    at_risk, survival, area, previous, i = len(pairs), 1.0, 0.0, 0.0, 0
    while i < len(pairs) and pairs[i][0] < cap:
        t = pairs[i][0]
        events = sum(1 for turn, seen in pairs[i:] if turn == t and seen)
        count = sum(1 for turn, _ in pairs[i:] if turn == t)
        area += survival * (t - previous)
        survival *= 1 - events / at_risk
        at_risk -= count
        previous, i = t, i + count
    return area + survival * (cap - previous)


def test_the_mean_of_the_capped_turns_equals_the_restricted_mean():
    """An episode is censored only at the step limit of its run, so on records of several step limits the restricted mean survival
    time at the smallest of them, the area under the Kaplan Meier curve up to it, equals the mean of the turns capped at it, which
    turns_summary bootstraps as a sum over a count. A correct diagnosis after the smallest limit counts as that limit, as on the
    curve: (2 + 10 + 4 + 10 + 10 + 10 + 10 + 7) / 8."""
    plays = [(2, 10), (None, 10), (4, 12), (None, 12), (11, 15), (13, 15), (None, 15), (7, 10)]
    episodes = [{"max_steps": cap, "diagnosis_all_correct": turn is not None, "turns_to_correct_diagnosis": "" if turn is None else turn,
                 "block": f"b{i % 3}"} for i, (turn, cap) in enumerate(plays)]
    records = [aggregate.turn_record(e) for e in episodes]
    smallest = min(cap for _, _, cap in records)
    reference = _kaplan_meier_restricted_mean([(turn, observed) for turn, observed, _ in records], smallest)
    cells = aggregate.turns_summary(episodes, {"resamples": 200, "seed": 0})
    assert smallest == 10 and reference == pytest.approx(63 / 8, abs=1e-12)
    assert cells["turns_rmst"] == pytest.approx(reference, abs=1e-6) and cells["turns_rmst_n"] == len(plays)
    assert cells["turns_rmst_low"] <= cells["turns_rmst"] <= cells["turns_rmst_high"]


def test_the_p_value_of_a_zero_or_one_outcome_is_exact_and_does_not_move_with_the_resamples_or_the_seed():
    """Pass differs on five of six blocks with block sums 2, 1, -1, 1 and 1, and not at all on the sixth, which the test drops. Of the
    32 assignments of signs to the five, those whose total is at least 4 away from zero are 5 with the 2 positive and 5 with it
    negative, so the exact p value is 10 / 32 under any number of resamples and any seed."""
    sums = {"b0": (1, 1), "b1": (1, 0), "b2": (-1,), "b3": (1,), "b4": (0, 1, 0), "b5": (0, 0)}
    level, reference = [], []
    for block, differences in sums.items():
        for d in differences:
            level.append({"seed": len(level), "block": block, "pass": str(d == 1)})
            reference.append({"seed": len(reference), "block": block, "pass": str(d == -1)})
    pairs = list(zip(level, reference))
    p_values = {aggregate.paired_difference(pairs, "pass", {"resamples": resamples, "seed": seed})["pass_p"]
                for resamples, seed in ((50, 0), (400, 7), (2000, 123))}
    assert p_values == {round(10 / 32, 6)}
    assert aggregate.paired_difference(pairs, "pass", {"resamples": 50, "seed": 0})["clusters"] == 6


def test_an_episode_without_a_start_combination_gets_the_empty_elimination_measures(tmp_path):
    """A failed verification writes a row with an empty block cell, so the episode has no start for the fits to leave out: its
    elimination measures are those of an episode without a read, and tables_from_runs/ does not stop on the cell."""
    from CausalDiagnosis.evaluation.metrics import realized_read_elimination as read_elimination

    ep = tmp_path / "ep"
    ep.mkdir()
    released = {"events": [{"event": "read", "t_start": 0, "t_end": 1, "cost": 1, "sensor_group": "g", "served": []}]}
    (ep / "released.json").write_text(json.dumps(released), encoding="utf-8")
    assert aggregate.replayed_fields.elimination_fields({"episode_dir": ep, "block": ""}) == {**read_elimination.episode_measures([]), "yardstick": ""}


def test_the_fold_of_an_episode_without_a_start_combination_stops_with_the_run_and_the_seed():
    with pytest.raises(ValueError, match="run r1, seed 7: the episode has no start combination"):
        aggregate.episode_fold({"run": "r1", "seed": 7, "block": ""})


def test_the_tables_of_this_stage_print_no_decision_id():
    """The strings of a Table print into results/README.md and the aggregate page, which are published, so the three tables of the
    families hold no decision id in what varies, the hypothesis, the metric, the reading and the reading rule."""
    import re

    found = {t.id: t for t in aggregate.tables()}
    for name in ("H1_GUIDANCE", "H2PRIME_pairs", "H0"):
        table = found[name]
        for text in (table.varies, table.hypothesis, table.metric, table.reading, table.rule):
            assert not re.search(r"\bD\d+\b", text), (name, text)


def test_the_training_tables_take_the_folds_of_the_asset_of_their_episodes(tmp_path, monkeypatch):
    """A table of training is laid out by the folds of the registered split of the asset its episodes
    played, here a made-up asset of three folds beside the rig's two, never by the folds of the default asset."""
    _two_folds(monkeypatch)
    folds = {"tiny": (1, 2, 3)}
    monkeypatch.setattr(aggregate.training_tables, "fold_ids", lambda asset_id=aggregate.DEFAULT_ASSET, split=None: folds.get(asset_id, (1, 2)))
    assert aggregate.training_tables.training_pair_subsets("tiny")[-3:] == ("fold_1", "fold_2", "fold_3") and aggregate.training_tables.training_pair_subsets()[-1] == "fold_2"
    seen = []
    real = aggregate.training_tables.training_pair_subsets
    monkeypatch.setattr(aggregate.training_tables, "training_pair_subsets", lambda asset_id=aggregate.DEFAULT_ASSET: seen.append(asset_id) or real(asset_id))
    runs = tmp_path / "runs"
    # the made-up asset names its components and healthy levels in run.json, as a run of any asset records them, so no profile is read
    rig = aggregate.get_profile(aggregate.DEFAULT_ASSET, engine=False)
    tiny = {"config": {"durations": {"read_per_cycle": 1}, "asset_id": "tiny", "components": list(rig.components), "healthy": dict(rig.healthy)}}
    _training_run(runs, "pe", "plan_execute_m", True, (True, False, True, False), **tiny)
    _cross_fitted(runs, "pe_gepa", "plan_execute_m", True, (True, True, True, True), "gepa", by_fold={1: tiny, 2: tiny})
    aggregate.table_training_pairs(aggregate.add_derived_fields(aggregate.load_episodes(runs)), {"resamples": 20, "seed": 0, "alpha": 0.05})
    assert seen and set(seen) == {"tiny"}


def _transcript(ep, steps):
    """The transcript that the servers of an episode write, from records in the form of trajectory.json: every call of an action
    server took a step, and a call of the diagnostics server took none and keeps its answer."""
    from CausalDiagnosis.mcp_servers.utils import transcript

    transcript.begin(ep, wall=0.0)
    for step in steps:
        mcp = step["server"] == "causal_diagnosis"
        transcript.answered(ep, step["server"], step["tool"], {}, step["response"], False, stepped=not mcp, keep_text=mcp)


def _log(ep, events, start=FAULTY_LIST):
    """A hidden log of one stepped entry per event, cycle by cycle, from a reset at the start combination."""
    (ep / "hidden").mkdir(exist_ok=True)
    log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "cost": 0, "hidden": list(start)},
           *({"event": event, "step": k, "t_start": k, "t_end": k, "cost": 0} for k, event in enumerate(events, start=1))]
    (ep / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    return log


def test_a12_scores_the_top_candidate_against_the_hidden_combination_at_the_cycle_of_its_answer(tmp_path):
    """A drop moves the hidden combination away from the start, so the top candidate of an answer after the wait that ran the
    cycle of the drop is right when it names the dropped combination, which the start alone would score wrong."""
    ep = tmp_path / "seed000000"
    (ep / "hidden").mkdir(parents=True)
    dropped = [*FAULTY_LIST[:2], 1, FAULTY_LIST[3]]
    log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "cost": 0, "hidden": FAULTY_LIST},
           {"event": "hazard_drop", "step": None, "t_start": 3, "t_end": 3, "cost": 0, "component": "pump", "to_level": 1},
           {"event": "wait", "step": 1, "t_start": 0, "t_end": 5, "cost": 0}]
    (ep / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    answer = {"belief_time": 5, "candidates": [{"combination": dict(zip(COMPONENTS, dropped)), "probability": 0.7}]}
    _transcript(ep, [{"server": "hydraulic_rig", "tool": "wait_for_notification", "response": "{}"},
                     {"server": "causal_diagnosis", "tool": "rank_root_causes", "response": json.dumps(answer)}])
    preset = aggregate.preset_config("deterioration")  # A12 reads the episodes at the main level of every setting but the ranking model
    episode = {**aggregate.MAIN_LEVELS, "clock_mode": preset.clock_mode, "guidance_level": preset.guidance_level, "verifier_scope": preset.verifier_scope,
               "agent": "causal_m", "model": "m", "config": "deterioration", "ranking_model": "joint", "episode_dir": ep, "block": json.dumps(FAULTY_LIST)}
    [row] = [r for r in aggregate.table_a12([episode], {"resamples": 20, "seed": 0}) if r["readings"] == "all"]
    assert row["top_candidate_correct"] == 1.0


def test_a12_counts_the_accepted_reads_before_an_answer(tmp_path):
    """A read that the engine refused, a read that never reached the engine and an accepted read before one answer: one reading."""
    ep = tmp_path / "seed000000"
    ep.mkdir()
    from CausalDiagnosis.mcp_servers.utils import transcript

    transcript.begin(ep, wall=0.0)
    answer = {"candidates": [{"combination": dict(zip(COMPONENTS, FAULTY_LIST)), "probability": 0.5}]}
    # the rig server answers a refused read without an error, and only a read that never reached the engine fails
    for server, tool, text, stepped, failed in (("hydraulic_rig", "read_sensors", "{}", True, False), ("hydraulic_rig", "read_sensors", "{}", False, True),
                                                ("hydraulic_rig", "read_sensors", "{}", True, False),
                                                ("causal_diagnosis", "rank_root_causes", json.dumps(answer), False, False)):
        transcript.answered(ep, server, tool, {}, text, failed, stepped=stepped, keep_text=server == "causal_diagnosis")
    _log(ep, ["refused", "read"])
    assert [(r, truth) for r, _top, _p, truth in aggregate.rankings({"episode_dir": ep})] == [(1, FAULTY_LIST)]


def test_the_unaffordable_separating_test_reads_the_flag_under_its_new_name_alone(tmp_path):
    """A ranking whose separating test costs more than the budget counts as an unaffordable test under read_first, and the same
    ranking with the flag under the old key, spelled here from two parts, counts as none."""
    old = "insufficient" + "_evidence"
    new = json.loads(_ranking(True, 60))
    legacy = {(old if key == "read_first" else key): value for key, value in new.items()}
    for name, answer, unaffordable in (("new", new, True), ("legacy", legacy, False)):
        ep = tmp_path / name
        ep.mkdir()
        (ep / "released.json").write_text(json.dumps({"rules": {"budget": 50}}), encoding="utf-8")
        step = {"server": "causal_diagnosis", "tool": "rank_root_causes", "response": json.dumps(answer), "error": None}
        (ep / "trajectory.json").write_text(json.dumps([step]), encoding="utf-8")
        assert aggregate.separating_test_unaffordable({"episode_dir": ep}) is unaffordable


def test_a3_counts_a_static_answer_without_a_judge_record_as_not_passed_when_its_run_names_a_judge(tmp_path):
    """Table A3 takes the counting rule of the other tables of the judge: a static answer of a run that names a judge and holds no
    judge record, an empty answer among them, counts as not passed, so three passed answers of four read 0.75; the answers of a run
    without a judge leave the cell empty; and the verified outcome of a static answer stays out, since no verifier scores it."""
    runs = tmp_path / "runs"

    def static_run(name, model, judge, cells):
        meta = {"mode": "static", "agent": "static", "model": model, **({"judge_model": judge} if judge else {})}
        _run(runs, name, [_row(s, agent="static", model=model, block=_start(s), judge_pass=cell, static_faulty_set_correct=s == 0, **{"pass": ""})
                          for s, cell in enumerate(cells)], meta)

    static_run("judged", "m1", "j", [True, True, True, ""])
    static_run("unjudged", "m2", None, ["", "", "", ""])
    a3 = {row["model"]: row for row in dict((t.id, rows) for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1"))["A3"]}
    assert (a3["m1"]["static_n"], a3["m1"]["static_judge_pass"], a3["m1"]["static_faulty_set_correct"]) == (4, 0.75, 0.25)
    assert (a3["m2"]["static_n"], a3["m2"]["static_judge_pass"]) == (4, "")
    static = {"mode": "static", "run_judge_model": "j"}
    assert aggregate.counts_as_failure(static, "judge_pass") and not aggregate.counts_as_failure(static, "pass")
    assert not aggregate.counts_as_failure({**static, "run_judge_model": ""}, "judge_pass")


def test_an_h0_test_with_few_starts_of_a_nonzero_sum_keeps_its_computed_p_value(tmp_path):
    """Every H0 test that runs enters Holm's procedure with its computed p value, as counting.holm of the registered file states, so a
    test on few starts of a nonzero sum is not set to p = 1: the judge passes all twelve starts and the verifier nine, J - V is one on
    three starts of one sign, whose exact sign flip p value is 2 ** (1 - 3) = 0.25."""
    runs = tmp_path / "runs"
    meta = {**BASE_META, "config_name": design.AGENTS_PRESET}
    run = _run(runs, "cd", [_h_row(s, "causal_m", False, _start(s), config=design.AGENTS_PRESET, judge_pass=True, **{"pass": s >= 3}) for s in range(12)], meta)
    for s in range(12):
        (run / f"seed{s:06d}" / "hidden" / "judge.json").write_text(json.dumps({"format": 1, "protocol": "paper", "calls_planned": 1, "judge_model": "j",
                                                                                 "calls": [{"passed": True, "details": {"task_completion": True}, "error": None}]}), encoding="utf-8")
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200, fraction="1")}
    cd = {(r["test"], r["name"]): r for r in tables["H0"]}[("row", design.CD[2])]
    assert (cd["nonzero_starts"], cd["status"], cd["h0_p"]) == (3, "tested", 0.25) and "minimum_starts" not in cd


# ----- a row prints one configuration, and a test of an incomplete row is not run -----

def test_a8_and_a12_read_the_main_level_of_every_setting_they_do_not_vary(tmp_path):
    """On the preset of the agents table the CausalDiagnosis agent with the MCP plays four starts at the main level of every setting, the
    copy of that run that rescore_under_verifier_scope.py scores under the verifier scope all, a line at guidance level 1 and a line under the fixed
    cut. A row prints one configuration (metrics.md), so the decision rule of A8 counts the four episodes of the main run and the fixed
    cut its own four, and A12, which varies the ranking model alone, reads the two answers of each episode of the main run."""
    runs = tmp_path / "runs"
    lines = {"main": ({}, {}), "main_scope_all": ({"verifier_scope": "all"}, {"verifier_scope": "all"}),
             "guidance_1": ({"guidance_level": 1}, {"guidance_level": 1}), "fixed_cut": ({}, {"abstention_rule": "fixed_cut"})}
    for name, (cells, meta) in lines.items():
        folder = _run(runs, name, [_base_row(s, agent="causal_m", abstained=s == 0, abstention_correct=s == 0, **cells) for s in range(4)], {**BASE_META, **meta})
        for ep in sorted(folder.glob("seed*")):
            _transcript(ep, json.loads((ep / "trajectory.json").read_text(encoding="utf-8")))
            log = json.loads((ep / "hidden" / "log.json").read_text(encoding="utf-8"))
            reads = [{"event": "read", "step": k, "t_start": k, "t_end": k, "cost": 0} for k in (1, 2)]
            (ep / "hidden" / "log.json").write_text(json.dumps([{**log[0], "hidden": FAULTY_LIST}, *reads, *log[1:]]), encoding="utf-8")
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    assert {(r["abstention_rule"], r["label"]): r["n"] for r in tables["A8"]} == {("decision", "separability"): 4, ("fixed_cut", "separability"): 4}
    a12 = {(r["ranking_model"], r["readings"]): r["calls"] for r in tables["A12"]}
    assert a12[(DEFAULT_RANKING_MODEL, "all")] == 8 and {model for model, _ in a12} == {DEFAULT_RANKING_MODEL}


def test_runs_counts_the_verdicts_of_the_judge_beside_the_judge_rate(tmp_path):
    """A run that names a judge reads an episode without a verdict as a failure of the judge, so the judge rate has every episode
    behind it, while judge_verdicts counts the one episode the judge scored, which the leaderboard waits for."""
    runs = tmp_path / "runs"
    _run(runs, "named", [_row(s, judge_pass=True if s == 0 else "") for s in range(3)], {"judge_model": "j"})
    row, = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["RUNS"]
    assert (row["n"], row["judge_pass_n"], row["judge_verdicts"]) == (3, 3, 1)


def test_an_h2_prime_comparison_of_an_incomplete_row_is_not_run_and_enters_holm_at_one(tmp_path):
    """Plan-Execute with the MCP plans five episodes and plays four on the top rung: H2'a prints the status, no p value and the reading
    not run, while H2'b is tested in the family of fourteen, the registered counting.not_played."""
    runs = tmp_path / "runs"
    for prefix in ("plan_execute", "agent_as_tool", "causal"):
        for server, passes, reward in ((True, True, 1.0), (False, False, 0.0)):
            planned = {"episodes_planned": 5} if (prefix, server) == ("plan_execute", True) else {}
            _run(runs, f"{prefix}_{server}", [_h_row(s, f"{prefix}_m", server, _start(s), reward=reward, **{"pass": passes}) for s in range(4)],
                 {**BASE_META, "config_name": design.TRAINING_PRESET, **planned})
    h2p = {r["hypothesis"]: r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["H2PRIME_pairs"]}
    assert (h2p["H2'a"]["status"], h2p["H2'a"]["pass_p"], h2p["H2'a"]["reading"], h2p["H2'a"]["family_tests"]) == ("incomplete, 4 of 5", "", "not run", 14)
    assert (h2p["H2'b"]["status"], h2p["H2'b"]["pass_p"], h2p["H2'b"]["family_tests"]) == ("", 0.125, 14)


def test_a_layer_whose_rows_miss_a_planned_episode_is_not_tested(tmp_path):
    """Plan-Execute plans five episodes on the priced layer and plays four: its difference and the pooled test of the layer are not
    computed, the pooled test enters the Holm of five at one and reads not run, and Agent-As-Tool keeps its own difference."""
    runs = tmp_path / "runs"
    passes = {("plan_execute", design.AGENTS_PRESET): (True, False, False, False), ("agent_as_tool", design.AGENTS_PRESET): (True, True, False, False),
              ("plan_execute", "priced"): (False,) * 4, ("agent_as_tool", "priced"): (True, True, False, False)}
    for (prefix, config), passed in passes.items():
        planned = {"episodes_planned": 5} if (prefix, config) == ("plan_execute", "priced") else {}
        _run(runs, f"{prefix}_{config}", [_base_row(s, agent=f"{prefix}_m", config=config, **{"pass": p}) for s, p in enumerate(passed)],
             {**BASE_META, "config_name": config, **planned})
    by = {(r["config"], r["agent"]): r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["LAYERS_pairs"]}
    assert (by[("priced", "plan_execute_m")]["status"], by[("priced", "plan_execute_m")]["pass_diff"]) == ("incomplete, 4 of 5", "")
    assert (by[("priced", "agent_as_tool_m")]["status"], by[("priced", "agent_as_tool_m")]["pass_diff"]) == ("", 0.0)
    pooled = by[("priced", "pooled")]
    assert (pooled["status"], pooled["pass_p"], pooled["family_tests"], pooled["reading"]) == ("incomplete, 4 of 5", "", aggregate.registered_size({}, "layers"), "not run")


def test_h1_under_guidance_level_three_leaves_a_comparison_of_an_incomplete_row_unrun(tmp_path):
    """The CausalDiagnosis agent plans five episodes at guidance level 3 and plays four: H1a is not run and enters its family of two
    tests at one."""
    runs = tmp_path / "runs"
    for agent, outcome in (("causal_m", True), ("plan_execute_m", False)):
        for level in (2, 3):
            planned = {"episodes_planned": 5} if (agent, level) == ("causal_m", 3) else {}
            _run(runs, f"{agent}_{level}", [_base_row(s, agent=agent, diagnostics_server=False, block=_start(s), guidance_level=level, diagnosis_all_correct=outcome,
                                                      **{"pass": outcome}) for s in range(4)], {**BASE_META, **planned})
    h1a, = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["H1_GUIDANCE"]
    assert (h1a["status"], h1a["pass_p"], h1a["family_tests"], h1a["holds"]) == ("incomplete, 4 of 5", "", 2, "")


def test_a_training_comparison_of_an_incomplete_row_is_not_decided(tmp_path, monkeypatch):
    """Plan-Execute with the MCP as released plans five episodes and plays four: the decision of GEPA against it is not run and enters
    the family of thirty at one, and the rows of the folds print the status beside their differences, which no test reads."""
    _two_folds(monkeypatch)
    runs = tmp_path / "runs"
    _training_run(runs, "pe", "plan_execute_m", True, (True, False, False, False), episodes_planned=5)
    _cross_fitted(runs, "pe_gepa", "plan_execute_m", True, (True, True, True, False), "gepa")
    pairs = {r["subset"]: r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["TRAINING_pairs"]}
    assert (pairs["all"]["status"], pairs["all"]["pass_p"], pairs["all"]["family_tests"], pairs["all"]["reading"]) == (
        "incomplete, 4 of 5", "", aggregate.registered_size({}, "training"), "not run")
    assert (pairs["fold_1"]["status"], pairs["fold_1"]["pass_diff"]) == ("incomplete, 4 of 5", 0.5)


def test_a_factor_level_of_an_incomplete_row_is_not_run_and_counts_at_one_in_the_holm_of_its_table(tmp_path):
    """A10 on the En7 runs: the CausalDiagnosis agent plans five episodes at guidance level 3 and plays four, so its level prints the
    status and no paired cells, while Plan-Execute, which passes four starts at level 3 that it fails at level 2, is tested; the Holm
    of the table counts the unrun test at one, so the exact p 2 / 16 doubles."""
    runs = tmp_path / "runs"
    for agent, at_three in (("causal_m", True), ("plan_execute_m", True)):
        for level in (2, 3):
            planned = {"episodes_planned": 5} if (agent, level) == ("causal_m", 3) else {}
            _run(runs, f"{agent}_{level}", [_base_row(s, agent=agent, diagnostics_server=False, block=_start(s), guidance_level=level,
                                                      **{"pass": at_three and level == 3}) for s in range(4)], {**BASE_META, **planned})
    rows = {(r["agent"], str(r["guidance_level"])): r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["A10"]}
    cd, pe = rows[("causal_m", "3")], rows[("plan_execute_m", "3")]
    assert (cd["status"], cd["pass_p"], cd.get("holm_p", "")) == ("incomplete, 4 of 5", "", "")
    assert (pe["status"], pe["pass_p"], pe["holm_p"]) == ("", 0.125, 0.25)
    assert rows[("plan_execute_m", "2")]["status"] == ""


def test_a_scoring_pair_of_an_incomplete_row_is_not_run_and_counts_at_one_in_the_scoring_family(tmp_path):
    """The CausalDiagnosis agent without the MCP plans five episodes and plays four: the scoring pair against Plan-Execute prints the
    status and no p value, and the scoring family counts both of its tests."""
    runs = tmp_path / "runs"
    _run(runs, "cd", [_base_row(s, agent="causal_m", diagnostics_server=False, block=_start(s)) for s in range(4)], {**BASE_META, "episodes_planned": 5})
    _run(runs, "pe", [_base_row(s, agent="plan_execute_m", diagnostics_server=False, block=_start(s)) for s in range(4)], BASE_META)
    scoring, = [r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["AGENTS_pairs"] if r["hypothesis"] == "scoring"]
    assert (scoring["status"], scoring["pass_p"], scoring["judge_pass_p"], scoring["family_tests"]) == ("incomplete, 4 of 5", "", "", 2)


def test_funnel_and_failures_hold_the_rows_of_the_training_table(tmp_path, monkeypatch):
    """FUNNEL and FAILURES say they hold every row of the training table, so a trained variant has a row of its own in both."""
    _two_folds(monkeypatch)
    runs = tmp_path / "runs"
    _training_run(runs, "pe", "plan_execute_m", True, (True, False, False, False))
    _cross_fitted(runs, "pe_gepa", "plan_execute_m", True, (True, True, True, False), "gepa")
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    for table in ("FUNNEL", "FAILURES"):
        assert sorted(r["training_variant"] for r in tables[table]) == ["as_released", "gepa"], table


def test_elim_pairs_reads_the_preset_of_the_agents_pairs_alone(tmp_path):
    """ELIM_pairs prints the comparisons of AGENTS_pairs, which reads the preset of the main table alone, so runs on the top rung
    give no row."""
    runs = tmp_path / "runs"
    for agent in ("causal_m", "plan_execute_m"):
        _run(runs, agent, [_h_row(s, agent, False, _start(s)) for s in range(4)], {**BASE_META, "config_name": design.TRAINING_PRESET})
    assert {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["ELIM_pairs"] == []


def test_a8_counts_the_episodes_of_its_group_that_the_verifier_did_not_score(tmp_path):
    """The declarations and the label are cells of the verdict, so an unscored episode enters neither denominator of A8, and the
    row says how many it leaves out."""
    runs = tmp_path / "runs"
    rows = [_base_row(s, agent="causal_m", abstained=s == 0, abstention_correct=s == 0) for s in range(3)]
    rows.append(_base_row(3, agent="causal_m", abstained="", abstention_correct="", **{"pass": ""}))
    _run(runs, "main", rows, BASE_META)
    a8, = [r for r in {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["A8"] if r["label"] == "separability"]
    assert (a8["n"], a8["unscored"]) == (3, 1)


def test_two_runs_that_differ_in_the_abstention_cut_alone_are_two_rows_and_the_condition_is_one_list(tmp_path):
    """The condition of the tables is derived from the condition of the index, so two cuts of the fixed-cut rule never pool."""
    runs = tmp_path / "runs"
    for cut in (0.5, 0.7):
        _run(runs, f"cut_{cut}", [_base_row(s, agent="causal_m") for s in range(2)], {**BASE_META, "abstention_rule": "fixed_cut", "abstention_cut": cut})
    agents = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}["AGENTS"]
    assert sorted(str(r["abstention_cut"]) for r in agents) == ["0.5", "0.7"]
    assert aggregate.CONDITION == pooling_rules.ROW_FIELDS and set(pooling_rules.ROW_FIELDS) <= {f for _, f, _ in pooling_rules.CONDITION_FIELDS}


def test_a_call_that_names_its_out_writes_its_pages_beside_it_and_no_page_of_the_fork(tmp_path):
    """The command line writes the README, the main tables and INDEX.json into the parent of --out, where the results folder
    of the fork holds the same four pages beside its folder of tables, which the default --out writes. So a call that names its
    own --out writes every file under that parent and no page of the fork. The control names a README in another folder, which
    takes the README and the main tables out of the parent."""
    runs, results = tmp_path / "runs", RESULTS
    runs.mkdir()
    pages = ("INDEX.json", "MAIN_TABLES.md", "README.md", "main_tables.tex")
    stamps = {name: (results / name).stat().st_mtime_ns for name in pages}

    def outside(folder):
        return sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file() and folder not in p.parents)

    aggregate.main(["--runs", str(runs), "--out", str(tmp_path / "T" / "aggregate"), "--fraction", "1"])
    assert outside(tmp_path / "T") == [] and sorted(p.name for p in (tmp_path / "T").iterdir() if p.is_file()) == list(pages)
    assert {name: (results / name).stat().st_mtime_ns for name in pages} == stamps
    aggregate.main(["--runs", str(runs), "--out", str(tmp_path / "U" / "aggregate"), "--readme", str(tmp_path / "other" / "README.md"),
                    "--fraction", "1"])
    assert [p for p in outside(tmp_path / "U") if not p.startswith("T/")] == ["other/MAIN_TABLES.md", "other/README.md", "other/main_tables.tex"]
