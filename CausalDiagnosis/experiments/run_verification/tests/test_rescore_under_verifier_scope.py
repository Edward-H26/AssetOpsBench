"""rescore_under_verifier_scope.py on two runs of the greedy policy on the real-time clock row, one scored under the default scope and
one started with --verifier-scope all: the rescored copy of the first holds the second's verdicts and score cells, the
run it read stays byte for byte, the copy's ledger checks and records the rescoring, the replay of verify_results_folder.py
agrees with the copy, the refusals hold, a judge record written after the run gives the copy its judge cell, and a copy under
the run's own scope records the scoring list of the tree. The two runs play the instant mode, since the generation time mode charges
the wall clock time between the policy's calls, so two plays of one seed under it need not agree. Run folders go under
pytest's tmp_path, outside the repository."""
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, HIDDEN, SCORE_COLUMNS, episode_folder, read_rows
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.run_verification import replay_check  # noqa: E402
from CausalDiagnosis.experiments.run_verification import rescore_under_verifier_scope as rescore_scope  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results  # noqa: E402
from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _meta  # noqa: E402

GREEDY = ["--config", "real_time", "--clock-mode", "instant", "--agent", "greedy", "--seeds", "2"]


def _run(args, out):
    run_episodes.main([*args, "--out", str(out)])
    runs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _hashes(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The run under the default scope, the run started under the scope all, the rescored copy of the first, and the
    hashes of the first's files before the rescoring."""
    root = tmp_path_factory.mktemp("rescore")
    writes = _run(GREEDY, root / "writes")
    direct = _run([*GREEDY, "--verifier-scope", "all"], root / "all")
    before = _hashes(writes)
    copy, n = rescore_scope.rescore_run(writes)
    return writes, direct, copy, n, before


def test_the_copy_holds_the_verdicts_and_cells_of_a_run_started_under_the_scope_all(runs):
    writes, direct, copy, n, _ = runs
    assert copy == writes.with_name(writes.name + "_scope_all") and n == 2
    rows, expected = read_rows(copy), {row["seed"]: row for row in read_rows(direct)}
    assert sorted(row["seed"] for row in rows) == sorted(expected)
    for row in rows:
        assert row["verifier_scope"] == "all" and row["run"] == copy.name
        assert {c: row[c] for c in SCORE_COLUMNS} == {c: expected[row["seed"]][c] for c in SCORE_COLUMNS}
    for ep_dir in sorted(p for p in copy.iterdir() if (p / HIDDEN).is_dir()):
        for name in ("verdict.json", "oracle.json", "score.json"):
            assert read_json(ep_dir / HIDDEN / name) == read_json(direct / ep_dir.name / HIDDEN / name), (ep_dir.name, name)
        assert read_json(ep_dir / HIDDEN / "verdict.json")["scope"] == "all" and read_json(ep_dir / HIDDEN / "oracle.json")["reads"]
        assert (ep_dir / HIDDEN / "log.json").read_bytes() == (writes / ep_dir.name / HIDDEN / "log.json").read_bytes()
    meta = read_json(copy / "run.json")
    assert meta["verifier_scope"] == meta["config"]["verifier_scope"] == "all" and meta["rescored"]["from"] == writes.name
    assert "verifier scope all" in (copy / "summary.md").read_text(encoding="utf-8")


def test_the_greedy_policy_holds_its_own_read_gate_and_a_read_more_than_the_reference_fails_it(runs, tmp_path):
    """The copy of a greedy run reads what its reference reads, so it holds the gate. One read above the reference in a group
    fails the gate of that episode and leaves the outcome cells as they are."""
    writes, _, copy, _, _ = runs
    rows = read_rows(copy)
    assert {row["read_gate"] for row in rows} == {"True"}
    ep_dir = copy / "seed000000"
    oracle = read_json(ep_dir / HIDDEN / "oracle.json")
    tighter = {group: count - 1 for group, count in oracle["reads"].items()}
    verdict = verifier.verify(read_json(ep_dir / HIDDEN / "log.json"), read_json(ep_dir / "released.json"), {**oracle, "reads": tighter}, read_json(ep_dir / HIDDEN / "verdict.json")["cost_oracle"],
                              grade=read_json(ep_dir / HIDDEN / "verdict.json")["grade"], scope="all", predictions=read_json(ep_dir / "predictions.json"))
    assert verdict["read_gate"] is False and verdict["pass"] == read_json(ep_dir / HIDDEN / "verdict.json")["pass"]


def test_the_run_it_read_stays_byte_for_byte_and_no_staging_folder_is_left(runs):
    writes, _, copy, _, before = runs
    assert _hashes(writes) == before
    assert sorted(p.name for p in writes.parent.iterdir()) == sorted([writes.name, copy.name])


def test_the_copy_ledger_checks_and_records_the_rescoring(runs):
    writes, _, copy, _, _ = runs
    assert verify_results.check_ledger(copy) == ("checked", [])
    event = ledger.load(copy)["events"][-1]
    assert event["tool"] == "rescore_scope.py" and writes.name in event["what"]
    assert {"run.json", "scores.csv", "summary.md"} <= set(event["files"])
    assert all(f"{ep}/{HIDDEN}/{name}" in event["files"] for ep in ("seed000000", "seed000001") for name in ("verdict.json", "oracle.json", "score.json"))
    assert set(event["code"]) == set(rescore_scope.CODE) and all(event["code"].values())


def test_the_replay_of_verify_results_agrees_with_the_copy(runs):
    _, _, copy, _, _ = runs
    state, problems = verify_results.check_replay(copy)
    assert state == "2 episodes replayed" and problems == []


def test_the_replay_composes_and_scores_a_copy_under_the_scope_of_its_rescoring(runs, tmp_path):
    """A copy holds the command of its run's ledger and the state file of every episode as the run played them, so the replay
    composes that command and scores every replayed episode under the scope that the block rescored of run.json names, and every
    answer, file and score cell agrees. The same record without that block composes the run's own scope, which differs from
    run.json in verifier_scope. A copy under the run's own scope replays as well."""
    writes, _, copy, _, _ = runs
    result = replay_check.replay_run(copy)
    assert result["episodes"] == 2 and result["config_equal"] and result["problems"] == []
    meta = {key: value for key, value in read_json(copy / "run.json").items() if key != "rescored"}
    blocks = load_world(None, replay_check.asset_of(meta))[2]
    assert replay_check.config_problem(copy, meta, blocks) == (f"{copy.name}: the configuration that the command of the run composes in "
                                                               "this tree differs from run.json in verifier_scope")
    run = tmp_path / writes.name
    shutil.copytree(writes, run)
    own, _ = rescore_scope.rescore_run(run, own_scope=True)
    assert replay_check.replay_run(own)["problems"] == []


def test_a_run_scored_under_the_scope_all_a_second_rescoring_and_a_changed_run_are_refused(runs, tmp_path):
    writes, direct, copy, _, _ = runs
    for folder, reason in ((direct, "scored under the scope all already"), (copy, "scored under the scope all already"),
                           (writes, "exists already")):
        with pytest.raises(SystemExit, match=reason):
            rescore_scope.rescore_run(folder)
    changed = tmp_path / writes.name
    shutil.copytree(writes, changed)
    rows = (changed / "scores.csv").read_text(encoding="utf-8")
    (changed / "scores.csv").write_text(rows.replace("True", "False", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="differ from its ledger"):
        rescore_scope.rescore_run(changed)
    unfinished = tmp_path / "unfinished" / writes.name
    shutil.copytree(writes, unfinished)
    meta = read_json(unfinished / "run.json")
    (unfinished / "run.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "ended"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="has not finished"):
        rescore_scope.rescore_run(unfinished)


def test_one_call_over_two_runs_loads_the_facts_file_once(runs, tmp_path, monkeypatch):
    """One call of the script over two runs of an asset loads its data file once and gives both runs the same cycles and blocks, so
    the classifiers that run_episodes caches by the identity of the cycles object fit once for the call."""
    writes = runs[0]
    loads, real = [], rescore_scope.load_world
    monkeypatch.setattr(rescore_scope, "WORLDS", {}, raising=False)
    monkeypatch.setattr(rescore_scope, "load_world", lambda *args: loads.append(args) or real(*args))
    for name in ("first", "second"):
        shutil.copytree(writes, tmp_path / name / writes.name)
    assert rescore_scope.main([str(tmp_path / name / writes.name) for name in ("first", "second")]) == 0
    assert len(loads) == 1


def test_a_rescored_copy_does_not_depend_on_the_place_of_its_run_in_the_call(runs, tmp_path, monkeypatch):
    """The copy of a run rescored alone and the copy of the same run rescored second of two in one call hold the same bytes in every
    file but run.json and the ledger, which record the time of the rescoring."""
    writes = runs[0]
    for name in ("alone", "first", "second"):
        shutil.copytree(writes, tmp_path / name / writes.name)
    monkeypatch.setattr(rescore_scope, "WORLDS", {}, raising=False)
    rescore_scope.main([str(tmp_path / "alone" / writes.name)])
    monkeypatch.setattr(rescore_scope, "WORLDS", {}, raising=False)
    rescore_scope.main([str(tmp_path / name / writes.name) for name in ("first", "second")])

    def files(parent):
        return {k: v for k, v in _hashes(tmp_path / parent / (writes.name + rescore_scope.SUFFIX)).items() if k not in ("run.json", ledger.LEDGER)}
    assert files("alone") == files("second") and len(files("alone")) > 2


def test_a_rescored_copy_keeps_the_judge_cell(runs, tmp_path):
    """A pass of the judge after the run, as judge_finished_runs.py makes, writes hidden/judge.json and the cell judge_pass and leaves
    score.json as the run wrote it, so the copy takes the judge of judge.json, summarized again, and the cell stays."""
    writes, _, _, _, _ = runs
    run = tmp_path / writes.name
    shutil.copytree(writes, run)
    before = ledger.snapshot(run)
    calls = [{"passed": True, "score": 1.0, "rationale": "", "details": {"criterion": 1}, "error": None}]
    record = {"protocol": "single", "judge_model": "stand-in", "calls_planned": 1, "calls": calls, "pass": False}
    rows = read_rows(run)
    judged = episode_folder(run, rows[0])
    (judged / HIDDEN / "judge.json").write_text(json.dumps(record), encoding="utf-8")
    write_rows(run / "scores.csv", [{**row, "judge_pass": True} if episode_folder(run, row) == judged else row for row in rows], COLUMNS)
    ledger.record_changes(run, before, "rejudge.py", "a pass of the judge after the run")
    copy, _ = rescore_scope.rescore_run(run)
    cells = {episode_folder(copy, row).name: row["judge_pass"] for row in read_rows(copy)}
    assert cells[judged.name] == "True" and read_json(copy / judged.name / HIDDEN / "score.json")["judge"]["pass"] is True
    assert {name for name, cell in cells.items() if cell} == {judged.name}


def test_a_copy_scored_again_under_its_own_scope_records_the_scoring_list_of_the_tree(runs, tmp_path):
    """--own-scope scores a run again under the scope it ran with: the copy keeps that scope and, since this tree scored the run,
    its cells, holds the run's lists with the list scoring of the tree and the identity computed again, and results_index reads
    it in place of its run, in the group of that scoring list. A copy under the scope all is no such copy."""
    writes, _, copy_all, _, _ = runs
    run = tmp_path / writes.name
    shutil.copytree(writes, run)
    copy, n = rescore_scope.rescore_run(run, own_scope=True)
    assert copy == run.with_name(run.name + "_own_scope") and n == 2
    meta, source = read_json(copy / "run.json"), read_json(run / "run.json")
    assert meta["verifier_scope"] == meta["config"]["verifier_scope"] == source["config"]["verifier_scope"] == "writes"
    assert meta["rescored"]["own_scope"] is True and meta["rescored"]["from"] == run.name and meta["rescored"]["scope"] == "writes"
    assert read_json(copy_all / "run.json")["rescored"]["own_scope"] is False
    files = meta["code"]["files"]
    assert files["scoring"] == ledger.code_identity.tree_files()["scoring"] and meta["code"]["identity"] == ledger.identity_of(files, meta["agent"])
    assert {k: v for k, v in files.items() if k != "scoring"} == {k: v for k, v in source["code"]["files"].items() if k != "scoring"}
    assert [{c: r[c] for c in SCORE_COLUMNS} for r in read_rows(copy)] == [{c: r[c] for c in SCORE_COLUMNS} for r in read_rows(run)]
    assert pooling_rules.rescored_refusal(meta, source, copy) is None
    assert pooling_rules.rescored_refusal(meta, source, run) == "its ledger holds no event of rescore_scope.py"
    other = {**source, "code": {**source["code"], "files": {**source["code"]["files"], "harness": {}}}}
    assert pooling_rules.rescored_refusal(meta, other) == "its lists harness differ from those of the run it came from"
    assert pooling_rules.rescored_refusal(read_json(copy_all / "run.json"), source) == "it is no copy scored again under its own scope"
    with pytest.raises(SystemExit, match="exists already"):
        rescore_scope.rescore_run(run, own_scope=True)
