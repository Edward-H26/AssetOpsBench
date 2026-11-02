"""benchmarks/hydraulic_rig/aggregate.py: the design of every table on a folder without runs, and the builders on
synthetic run folders whose rows are made up here (no recorded value enters)."""
import csv
import json

import aggregate
from common import COLUMNS, write_rows
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY

HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]
FAULTY_LIST = [3, 100, 0, 130]


def _row(seed, **fields):
    row = {c: "" for c in COLUMNS}
    row.update({"seed": seed, "scenario_id": "", "config": "static", "agent": "plan_execute", "model": "m", "clock_mode": "declared",
                "guidance_level": 2, "diagnostics_server": True, "prices": "placeholder", "verifier_scope": "writes",
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


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_without_run_folders_every_table_states_its_design_with_empty_cells(tmp_path):
    out = tmp_path / "out"
    tables = aggregate.aggregate(tmp_path / "no_runs", out, resamples=50)
    assert [t.id for t, _ in tables] == [f"A{i}" for i in range(1, 13)] + ["GEPA"] and all(rows == [] for _, rows in tables)
    text = (out / "aggregate.md").read_text(encoding="utf-8")
    assert "No run folder exists yet" in text and text.count("No episode contributes to this table yet.") == 13
    for t, _ in tables:
        assert f"## {t.id}\n" in text and f"- Hypothesis: {t.hypothesis}" in text
        rows = _csv(out / f"{t.id}.csv")
        assert rows == [] and (out / f"{t.id}.csv").read_text(encoding="utf-8").strip() == ",".join(t.columns)
    assert _csv(out / "A12_curves.csv") == []
    readme = (out.parent / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Results of the hydraulic rig gym") and "No run exists yet" in readme
    assert readme.count("\n## ") == 15 and "## Example episode" in readme and "## How to reproduce" in readme
    assert "No episode has been played yet" in readme and all(f"Reading rule: {t.rule}" in readme for t, _ in tables)


def test_factor_tables_pair_episodes_and_correct_with_holm(tmp_path):
    runs = tmp_path / "runs"
    on = [_row(s, pass_=None) for s in range(4)]
    for r in on:
        r.pop("pass_")
    off = [_row(s, diagnostics_server=False, cost=14, **{"pass": s % 2 == 0}) for s in range(4)]
    _run(runs, "diag_on", on)
    _run(runs, "diag_off", off)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=200)}
    a4 = {row["diagnostics_server"]: row for row in tables["A4"]}
    assert set(a4) == {"True", "False"} and a4["True"]["n"] == a4["False"]["n"] == 4
    assert a4["True"]["pass"] == 1.0 and a4["False"]["pass"] == 0.5 and a4["False"]["cost"] == 14.0
    # the level True is compared with the reference False: pass differs by 0.5 on average, cost by -4, over 4 pairs
    assert a4["True"]["pass_diff"] == 0.5 and a4["True"]["cost_diff"] == -4.0 and a4["True"]["pairs"] == 4
    assert a4["False"]["pass_diff"] == "" and a4["True"]["holm_p"] != "" and a4["True"]["reject"] in (True, False)
    # every row sits in one block, so the interval of the mean has zero width
    assert a4["True"]["cost_low"] == a4["True"]["cost_high"] == 10.0
    a11 = {row["n_faulty"]: row for row in tables["A11"]}
    assert set(a11) == {"1"} and a11["1"]["n"] == 8 and "pass_diff" not in a11["1"]


def test_judge_gold_scope_abstention_and_ranking_tables_read_the_hidden_files(tmp_path):
    runs = tmp_path / "runs"
    rows = [_row(0, judge_pass=True, **{"pass": True}), _row(1, judge_pass=True, **{"pass": False}),
            _row(2, judge_pass=False, **{"pass": False}), _row(3, judge_pass=False, **{"pass": True}, abstained=True, abstention_correct=True)]
    _run(runs, "writes", rows)
    _run(runs, "all_scope", [_row(s, verifier_scope="all", judge_pass="", **{"pass": s == 0}) for s in range(4)], hidden=False)
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=100)}
    a1 = tables["A1"][0]
    assert (a1["both_pass"], a1["judge_pass_only"], a1["recorded_pass_only"], a1["neither"]) == (1, 1, 1, 1)
    assert a1["false_pass_rate"] == 0.5 and a1["kappa"] == 0.0
    a2 = {row["question_type"]: row for row in tables["A2"]}
    # four episodes, each with one necessary and one unnecessary work order: half the orders change nothing
    assert a2["counterfactual"]["n"] == 8 and a2["counterfactual"]["agreement"] == 0.5
    assert a2["action"]["n"] == 4 and a2["action"]["agreement"] == 1.0
    a7 = tables["A7"]
    assert a7[0]["pass_writes"] == 0.5 and a7[0]["pass_all"] == 0.25 and a7[-1]["agent"] == "all agents" and a7[-1]["rank_correlation"] == ""
    a8 = tables["A8"][0]
    assert a8["abstentions"] == 1 and a8["precision"] == 1.0 and a8["abstention_rule"] == "decision"
    assert a1["kappa_low"] != "" and a1["kappa_high"] != ""
    a12 = {row["readings"]: row for row in tables["A12"]}
    # two ranking calls per episode: after one reading the truth at 0.8, after two readings a wrong candidate at 0.6
    assert set(a12) == {1, 2, "all"}
    assert a12[1]["calls"] == 4 and a12[1]["top_candidate_correct"] == 1.0 and a12[1]["brier"] == round(0.2 ** 2, 4)
    assert a12[2]["calls"] == 4 and a12[2]["top_candidate_correct"] == 0.0 and a12[2]["brier"] == round(0.6 ** 2, 4)
    assert a12["all"]["calls"] == 8 and a12["all"]["top_candidate_correct"] == 0.5 and a12["all"]["brier"] == round(((0.2 ** 2) + (0.6 ** 2)) / 2, 4)
    assert a12["all"]["brier_low"] != "" and a12["all"]["top1_high"] != ""
    curves = {(c["readings"], c["bin_lo"]): c for c in _csv(tmp_path / "out" / "A12_curves.csv")}
    assert curves[("1", "0.8")]["observed_frequency"] == "1.0" and curves[("1", "0.8")]["predicted_mean"] == "0.8" and curves[("1", "0.8")]["n"] == "4"
    assert curves[("2", "0.6")]["observed_frequency"] == "0.0" and curves[("all", "0.6")]["n"] == "4" and curves[("all", "0.8")]["bin_hi"] == "0.9"
    text = (tmp_path / "out" / "aggregate.md").read_text(encoding="utf-8")
    assert "No run folder exists yet" not in text and "| plan_execute | m |" in text
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Models and agents: m." in readme and "Config presets: static." in readme and "Run folders: 2" in readme
    assert "## Example episode" in readme and "(runs/writes/seed000000/report.md)" in readme and "1. hydraulic_rig.read_sensors({})" in readme
    assert '"checks"' in readme
