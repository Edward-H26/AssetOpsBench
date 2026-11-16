"""verify_results_folder.py on synthetic runs with ledgers (the run folders of test_leaderboard.py) and on scripted runs of the driver on
both tool surfaces: identical when nothing changed, a tampered cell reported by row and column without its value, a tampered
episode file reported by path and accepted only with its ledger event, the local events of a synced copy read as its ledger, a
tampered verdict or receipt caught by the replay through the servers, and the certificate of one run folder written without a
results folder."""
import csv
import hashlib
import json
import re

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.evaluation.result_tables import leaderboard
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, STATE_FILE
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _run
from CausalDiagnosis.evaluation.result_tables.tests.test_leaderboard import CUTOFF, DEVELOPMENT, _main_row, _runs
from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _meta, scratch_repository
from CausalDiagnosis.experiments.run_verification.tests.test_replay_check import _copy, _set_commit, scripted  # noqa: F401  the scripted runs of the session, made after their tables


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results(tmp_path, extra=None):
    """The run folders of test_leaderboard.py and what extra adds to them, with a ledger each, their tables, index and leaderboard."""
    runs = _runs(tmp_path / "runs")
    if extra:
        extra(runs)
    facts = tmp_path / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    for run_dir in sorted(p for p in runs.iterdir() if (p / "run.json").exists()):
        ledger.start(run_dir, {**_meta(facts), "label": run_dir.name}, ["--seeds", "4"])
        ledger.finish(run_dir)
    results = tmp_path / "results"
    aggregate.aggregate(runs, results / "tables", resamples=50, fraction="1")
    leaderboard.main(["--aggregate", str(results / "tables"), "--out", str(results / "LEADERBOARD.md"), "--since", CUTOFF, "--runs", str(runs)])
    return runs, results


def test_the_results_verify_identical_until_a_cell_is_tampered_with_and_the_report_names_no_value(tmp_path):
    runs, results = _results(tmp_path)
    assert verify_results.main(["--runs", str(runs), "--results", str(results)]) == 0
    text = (results / "VERIFY.md").read_text(encoding="utf-8")
    assert "| ledger | 9 runs checked, 0 without a ledger | identical |" in text and "| tables | " in text and text.endswith("## Differences\n\nNone.\n")
    assert "| paper |" not in text  # the index never held a paper cell, so the report has no such check
    assert re.search(r"^Checked on \d{4}-\d{2}-\d{2} by ", text, re.M)  # the date alone, no clock time
    path = results / "tables" / "RUNS.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][rows[0].index("pass")] = "0.123456"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    _, sections = verify_results.verify(results, [runs])
    assert sections["tables"][1] == ["tables/RUNS.csv: row 1 column pass differs"] and sections["ledger"][1] == []
    assert verify_results.main(["--runs", str(runs), "--results", str(results)]) == 1
    assert "0.123456" not in (results / "VERIFY.md").read_text(encoding="utf-8")


def test_a_tampered_cell_of_wrong_is_named_by_its_column(tmp_path):
    """The rebuild computes the share of wrong diagnoses again from the runs."""
    runs, results = _results(tmp_path)
    path = results / "tables" / "AGENTS.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][rows[0].index("diagnosis_wrong")] = "0.123456"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    _, sections = verify_results.verify(results, [runs])
    assert sections["tables"][1] == ["tables/AGENTS.csv: row 1 column diagnosis_wrong differs"]


def test_a_tampered_episode_file_is_reported_until_an_event_records_the_change(tmp_path):
    runs, results = _results(tmp_path)
    run = runs / "pe_mcp"
    report = run / "seed000000" / "report.md"
    old = _sha(report)
    report.write_text("edited after the run\n", encoding="utf-8")
    (run / "seed000001" / "notes.txt").write_text("added after the run\n", encoding="utf-8")
    assert verify_results.check_ledger(run) == ("checked", ["pe_mcp/seed000000/report.md: changed without a recorded event",
                                                            "pe_mcp/seed000001/notes.txt: added without a recorded event"])
    ledger.append_event(run, "a test", "rewrote a report and added notes", {"seed000000/report.md": (old, _sha(report)),
                                                                             "seed000001/notes.txt": (None, _sha(run / "seed000001" / "notes.txt"))})
    assert verify_results.check_ledger(run) == ("checked", [])
    # the index was built before the event, so the rebuild sees a newer ledger of that run
    _, sections = verify_results.verify(results, [runs])
    assert sections["tables"][1] == ["INDEX.json: the section inputs differs"]


def _trained_runs(runs):
    """A finished run of a trained agent beside the agent without training, under one agent label, and a run of another in progress."""
    _run(runs, "pe_gepa", [_main_row(s, agent="plan_execute_m", **{"pass": s < 2}) for s in range(4)], {**DEVELOPMENT, design.TRAINING_VARIANT_KEY: "gepa"})
    running = runs / "base_agent_as_tool_m_20261102-140000"
    running.mkdir()
    write_rows(running / "scores.csv", [_main_row(0, agent="agent_as_tool_m")], COLUMNS)


def test_the_results_verify_with_a_trained_run_beside_its_agent_without_training_and_a_run_json_that_loses_the_key_no_longer_rebuilds_them(tmp_path):
    """A trained agent and the agent without training carry one agent label, so the rebuild reads the training variant from run.json. A
    run.json that loses the key, even with its ledger event, pools the trained run with the agent without training, and the table check
    reports that."""
    runs, results = _results(tmp_path, _trained_runs)
    assert verify_results.main(["--runs", str(runs), "--results", str(results)]) == 0
    assert (results / "VERIFY.md").read_text(encoding="utf-8").endswith("## Differences\n\nNone.\n")
    run_json = runs / "pe_gepa" / "run.json"
    old, meta = _sha(run_json), json.loads(run_json.read_text(encoding="utf-8"))
    del meta[design.TRAINING_VARIANT_KEY]
    run_json.write_text(json.dumps(meta), encoding="utf-8")
    ledger.append_event(runs / "pe_gepa", "a test", "took the training variant out of run.json", {"run.json": (old, _sha(run_json))})
    _, sections = verify_results.verify(results, [runs])
    assert sections["ledger"][1] == []
    problems = sections["tables"][1]
    assert any(p.startswith("tables/RUNS.csv: row ") and p.endswith(" column training_variant differs") for p in problems)
    assert any(p.startswith("tables/AGENTS.csv: ") for p in problems)


def test_a_run_on_the_assetopsbench_servers_replays_through_its_servers_and_a_tampered_receipt_is_caught(scripted, tmp_path):
    """A scripted run of the AssetOpsBench rows on the memory store replays through the metered servers of this tree, and one changed
    character of the receipt that the state file stores for a call is named by the file and never by its value."""
    run_dir = _copy(scripted("aob"), tmp_path)
    assert verify_results.check_replay(run_dir) == ("1 episodes replayed", [])
    state_path = run_dir / "seed000000" / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    receipt = state["tool_calls"][0]["response"]
    place = next(i for i, char in enumerate(receipt) if char.isdigit())
    state["tool_calls"][0]["response"] = receipt[:place] + str((int(receipt[place]) + 1) % 10) + receipt[place + 1:]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert verify_results.check_replay(run_dir)[1] == [f"{run_dir.name}/seed000000: the state file differs from {STATE_FILE}"]


def test_a_scripted_run_replays_identical_and_a_tampered_verdict_is_caught(scripted, tmp_path):
    run_dir = _copy(scripted("rig"), tmp_path / "runs")
    results = tmp_path / "results"
    aggregate.aggregate(tmp_path / "runs", results / "tables", resamples=50, fraction="1")
    runs, sections = verify_results.verify(results, [tmp_path / "runs"], replay=True)
    assert runs[run_dir.name]["replay"] == "1 episodes replayed" and runs[run_dir.name]["ledger"] == "checked"
    assert all(problems == [] for _, problems in sections.values()), sections
    verdict_path = run_dir / "seed000000" / "hidden" / "verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    verdict["pass"] = not verdict["pass"]
    verdict_path.write_text(json.dumps(verdict), encoding="utf-8")
    assert verify_results.check_replay(run_dir)[1] == [f"{run_dir.name}/seed000000: the verdict differs from hidden/verdict.json"]
    assert verify_results.check_ledger(run_dir)[1] == [f"{run_dir.name}/seed000000/hidden/verdict.json: changed without a recorded event"]


def test_the_certify_option_needs_no_results_folder(scripted, tmp_path, monkeypatch):
    """--certify replays one run folder through the servers of this tree and writes its certificate into it, with no results folder
    and no index, and exits 0. A certificate binds the commit the run was played at, and a copy of the tree is no repository, so the
    tree is made a clone of a commit of its own (scratch_repository) that the copied run names; the control: before it the tree
    reads as no repository."""
    run_dir = _copy(scripted("rig"), tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no_repository"))
    assert ledger.code_identity.git_state([], ledger.run_data.ROOT)[0] is None
    commit = scratch_repository(monkeypatch, tmp_path)
    _set_commit(run_dir, commit)
    assert ledger.code_identity.git_state([], ledger.run_data.ROOT)[0] == commit
    assert verify_results.main(["--certify", str(run_dir), "--results", str(tmp_path / "no_results")]) == 0
    assert len(list(run_dir.glob("replay_*.json"))) == 1 and not (tmp_path / "no_results").exists()


def test_the_local_events_of_a_synced_copy_count_as_its_ledger(tmp_path):
    """A synced copy keeps the ledger of the cluster byte for byte and the events of the laptop beside it in local_events.json, which
    the ledger check reads as part of the ledger and never as a file added without an event."""
    runs, _results_folder = _results(tmp_path)
    run = runs / "pe_mcp"
    (run / ledger.LOCAL_EVENTS).write_text("[]", encoding="utf-8")
    report = run / "seed000000" / "report.md"
    old = _sha(report)
    report.write_text("edited on the laptop\n", encoding="utf-8")
    assert ledger.append_event(run, "a test", "rewrote a report on the laptop", {"seed000000/report.md": (old, _sha(report))})
    assert json.loads((run / ledger.LOCAL_EVENTS).read_text(encoding="utf-8")) and verify_results.check_ledger(run) == ("checked", [])


def test_the_rebuild_names_the_runs_it_leaves_out_instead_of_pooling_them(tmp_path):
    """A run.json that changed its commit after the index was built makes the rebuild leave out every other run of the index
    (results_index.pooling_refusals), and the table check reports each with its reason beside the rows that moved."""
    runs, results = _results(tmp_path)
    run_json = runs / "pe_mcp" / "run.json"
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    run_json.write_text(json.dumps({**meta, "code": {"commit": "1" * 40}, "started": "2026-12-09T00:00:00"}), encoding="utf-8")
    _, sections = verify_results.verify(results, [runs])
    problems = sections["tables"][1]
    reason = ("played commit none recorded while pe_mcp (the latest start, 2026-12-09T00:00:00) played " + "1" * 40
              + ", and runs of different commits are never pooled")
    assert f"aat_bare: left out of the rebuild, {reason}" in problems and f"causal_mcp: left out of the rebuild, {reason}" in problems
    assert any(p.startswith("tables/RUNS.csv") for p in problems)


def test_code_drift_names_a_listed_file_that_differs_from_the_record_of_the_run(tmp_path):
    """A run records its code as full hashes per list under code.files, so the drift check reads them, and still
    reads the prefixes of code.sha256 of a run written before the lists."""
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT

    path = "CausalDiagnosis/experiments/run_verification/verify_results_folder.py"
    digest = verify_results.ledger.sha256_of(ROOT / path)
    assert verify_results.code_drift({"code": {"files": {"analysis": {path: digest}}}}) == []
    assert verify_results.code_drift({"code": {"files": {"analysis": {path: "0" * 64}}}}) == [path]
    assert verify_results.code_drift({"code": {"sha256": {path: digest[:12]}}}) == [] and verify_results.code_drift({"code": {"sha256": {path: "0" * 12}}}) == [path]


def test_a_run_that_the_index_does_not_hold_stops_the_check(tmp_path):
    """--run checks the ledger and the replay of one run of the index, so a name that the index does not hold stops the check
    instead of checking nothing."""
    runs, results = _results(tmp_path)
    try:
        verify_results.verify(results, [runs], run="no_such_run")
    except SystemExit as stop:
        message = str(stop)
    else:
        message = None
    assert message is not None and "--run no_such_run is no run of" in message


def test_a_run_of_a_registered_root_without_its_ledger_or_its_transcripts_is_a_difference(tmp_path):
    """Every run of a registered root holds its ledger and its transcripts, so a copy that lost either reads as a difference there,
    while a development root keeps reporting such a run as not checked or not replayed."""
    runs, results = _results(tmp_path)
    run_dir = next(p for p in sorted(runs.iterdir()) if (p / ledger.LEDGER).exists())
    (run_dir / ledger.LEDGER).unlink()
    _, sections = verify_results.verify(results, [runs], replay=True)
    assert sections["ledger"][1] == [] and not [p for p in sections["replay"][1] if "transcript" in p]
    index = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))
    (results / "INDEX.json").write_text(json.dumps({**index, "root_kind": "record"}), encoding="utf-8")
    _, sections = verify_results.verify(results, [runs], replay=True)
    assert sections["ledger"][1] == [f"{run_dir.name}: no ledger, which every run of a registered root holds"]
    assert f"{run_dir.name}: no episode holds a transcript, which every run of a registered root holds" in sections["replay"][1]
