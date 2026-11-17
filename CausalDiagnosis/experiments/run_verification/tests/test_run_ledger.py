"""The run ledger: a synthetic run keeps a running ledger until finish() and then the hash of every file it wrote, a file
changed after its episode and a data file changed during the run each leave an event, events keep the old and the new
hash, the copied environment never holds a key, and a scripted run of the driver writes the ledger before its first
episode and closes it at the end with the job, the command and the code it ran. An AssetOpsBench reproduction run keeps
a ledger from its own run.json with the CouchDB data it loads and every scoring as an event with the scorer code, and a
finished one without a ledger gets one written after the fact that says so. Run folders go under pytest's tmp_path."""
from CausalDiagnosis.utils.paths import PACKAGE
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import ledger_fields
from CausalDiagnosis.experiments.run_verification import public_record
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import STATE_FILE
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

SECRET = "sk-ledger-test-0123456789abcdef"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _meta(facts_path):
    """The run.json fields that start() reads, as run_meta writes them for a run without a model."""
    return {"label": "static_scripted_20261103-120000", "agent": "scripted", "agent_label": "scripted", "started": "2026-11-03T12:00:00",
            "code": {"commit": None, "sha256": {"CausalDiagnosis/experiments/run_verification/run_ledger/ledger_file.py": "0" * 12}}, "facts_path": str(facts_path),
            "model": None, "endpoint": None, "judge_model": None, "judge_endpoint": None, "config_name": "static",
            "config": {"horizon": 60, "clock_mode": "declared"}, "mode": "gym", "prices": "default", "price_table_source": None,
            "starts": "seeds", "master_seed": None, "seed_pool": None, "seeds": [0, 1], "episodes_planned": 2}


def _episode(run_dir, name, text):
    ep_dir = run_dir / name
    (ep_dir / "hidden").mkdir(parents=True)
    (ep_dir / STATE_FILE).write_text(json.dumps({"tool_calls": []}), encoding="utf-8")
    (ep_dir / "report.md").write_text(text, encoding="utf-8")
    (ep_dir / "hidden" / "verdict.json").write_text(json.dumps({"verified": True}), encoding="utf-8")
    return ep_dir


def _synthetic_run(tmp_path):
    """A run folder with a started ledger and two recorded episodes, and the facts file it names."""
    facts = tmp_path / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "static_scripted_20261103-120000"
    run_dir.mkdir(parents=True)
    ledger.start(run_dir, _meta(facts), ["--config", "static", "--agent", "scripted", "--seeds", "2"])
    for seed in (0, 1):
        name = f"seed{seed:06d}"
        _episode(run_dir, name, f"episode {seed}\n")
        ledger.ledger_file.add_episode(run_dir, name, {"seed": seed, "block": "[100,100,0,90]"})
    (run_dir / "run.json").write_text(json.dumps({"ended": "2026-11-03T12:05:00"}), encoding="utf-8")
    (run_dir / "scores.csv").write_text("run,seed\nx,0\nx,1\n", encoding="utf-8")
    (run_dir / "summary.md").write_text("# summary\n", encoding="utf-8")
    return run_dir, facts


def test_a_synthetic_run_keeps_a_running_ledger_until_finish_and_then_the_hash_of_every_file(tmp_path):
    run_dir, _ = _synthetic_run(tmp_path)
    running = ledger.load(run_dir)
    assert running["status"] == "running" and running["ended"] is None and list(running["episodes"]) == ["seed000000", "seed000001"]
    closed = ledger.finish(run_dir)
    assert closed == ledger.load(run_dir)
    assert closed["status"] == "finished" and closed["ended"] == "2026-11-03T12:05:00" and closed["events"] == []
    assert set(closed["run_files"]) == {"run.json", "scores.csv", "summary.md"}
    for path, entry in closed["run_files"].items():
        assert entry == {"sha256": _sha(run_dir / path), "bytes": (run_dir / path).stat().st_size}
    for name, episode in closed["episodes"].items():
        found = sorted(p.relative_to(run_dir / name).as_posix() for p in (run_dir / name).rglob("*")
                       if p.is_file() and not ledger.ignored(p.relative_to(run_dir)))
        assert sorted(episode["files"]) == found == ["episode.json", "hidden/verdict.json", "report.md"]
        assert all(episode["files"][f]["sha256"] == _sha(run_dir / name / f) for f in found)
    assert closed["episodes"]["seed000001"]["seed"] == 1 and closed["episodes"]["seed000001"]["block"] == "[100,100,0,90]"
    assert closed["configuration"]["config_sha256"] == ledger.canonical_sha256({"clock_mode": "declared", "horizon": 60})
    assert closed["configuration"]["training_variant"] == "as_released"  # a run.json without the key plays the agent as released
    assert closed["code"]["files"] == {"CausalDiagnosis/experiments/run_verification/run_ledger/ledger_file.py": _sha(PACKAGE / "experiments" / "run_verification" / "run_ledger" / "ledger_file.py")}


def test_a_run_folder_inside_the_checkout_passes_its_ledger_check_with_bytecode_and_finder_files(tmp_path):
    """A launcher started as python -m from an episode folder inside the checkout leaves __pycache__ there, and Finder leaves
    .DS_Store: no ledger records either, so the ledger check, which leaves both out, finds the run as it was written, and the file
    table of a walkthrough page, which the review pack matches with the ledger, lists neither."""
    from CausalDiagnosis.episode_viewer import walkthrough

    facts = tmp_path / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "static_scripted_20261103-120000"
    run_dir.mkdir(parents=True)
    ledger.start(run_dir, {**_meta(facts), "seeds": [0], "episodes_planned": 1}, ["--config", "static", "--agent", "scripted", "--seeds", "1"])
    ep_dir = _episode(run_dir, "seed000000", "episode 0\n")
    (ep_dir / "__pycache__").mkdir()
    (ep_dir / "__pycache__" / "launcher.cpython-312.pyc").write_bytes(b"\x00bytecode")
    for folder in (ep_dir, run_dir):
        (folder / ".DS_Store").write_bytes(b"finder")
    ledger.ledger_file.add_episode(run_dir, "seed000000", {"seed": 0, "block": "[100,100,0,90]"})
    (run_dir / "run.json").write_text(json.dumps({"ended": "2026-11-03T12:05:00"}), encoding="utf-8")
    closed = ledger.finish(run_dir)
    assert sorted(closed["episodes"]["seed000000"]["files"]) == ["episode.json", "hidden/verdict.json", "report.md"]
    assert set(closed["run_files"]) == {"run.json"}
    assert verify_results.check_ledger(run_dir) == ("checked", [])
    page = "\n".join(walkthrough.files_read(SimpleNamespace(folder=ep_dir)))
    assert "seed000000/report.md" in page and "__pycache__" not in page and ".DS_Store" not in page
    assert ledger.ignored("a/__pycache__/b.pyc") and ledger.ignored(".DS_Store") and not ledger.ignored("hidden/verdict.json")


def test_a_file_changed_after_its_episode_and_a_data_file_changed_during_the_run_each_leave_an_event(tmp_path):
    run_dir, facts = _synthetic_run(tmp_path)
    report = run_dir / "seed000000" / "report.md"
    old_report, old_facts = _sha(report), _sha(facts)
    report.write_text("edited after the episode\n", encoding="utf-8")
    facts.write_text('{"changed": true}\n', encoding="utf-8")
    events = ledger.finish(run_dir)["events"]
    assert [e["what"] for e in events] == ["episode files changed after their episode was recorded",
                                           "data files changed while the run was in progress"]
    assert events[0]["files"] == {"seed000000/report.md": {"old": old_report, "new": _sha(report)}}
    assert events[1]["files"] == {public_record.public_path(facts): {"old": old_facts, "new": _sha(facts)}}
    assert ledger.load(run_dir)["episodes"]["seed000000"]["files"]["report.md"]["sha256"] == _sha(report)


def test_an_event_keeps_the_old_and_the_new_hash_and_a_run_without_a_ledger_takes_none(tmp_path):
    run_dir, _ = _synthetic_run(tmp_path)
    ledger.finish(run_dir)
    assert ledger.append_event(run_dir, "rejudge.py", "judged again", {"seed000000/hidden/judge.json": (None, "ab" * 32)})
    event = ledger.load(run_dir)["events"][-1]
    assert event["tool"] == "rejudge.py" and event["what"] == "judged again" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", event["at"])
    assert event["files"] == {"seed000000/hidden/judge.json": {"old": None, "new": "ab" * 32}}
    bare = tmp_path / "runs" / "made_before_the_ledger"
    bare.mkdir()
    assert ledger.append_event(bare, "rejudge.py", "judged again") is False and not (bare / ledger.LEDGER).exists()
    assert ledger.ledger_file.add_episode(bare, "seed000000", {"seed": "0"}) is False and not (bare / ledger.LEDGER).exists()


def test_the_command_line_records_a_move_of_a_run_folder_and_skips_a_run_without_a_ledger(tmp_path, capsys):
    run_dir, _ = _synthetic_run(tmp_path)
    assert ledger.main(["event", str(run_dir), "archive_invalid_runs.sh", "moved into the archive, reason: a bug fixed later"]) == 0
    event = ledger.load(run_dir)["events"][-1]
    assert (event["tool"], event["what"], event["files"]) == ("archive_invalid_runs.sh", "moved into the archive, reason: a bug fixed later", {})
    bare = tmp_path / "made_before_the_ledger"
    bare.mkdir()
    assert ledger.main(["event", str(bare), "archive_invalid_runs.sh", "moved"]) == 0
    assert capsys.readouterr().out == f"{bare} has no ledger, nothing recorded\n" and not (bare / ledger.LEDGER).exists()


def test_the_copied_environment_never_names_a_secret():
    assert "API_KEY" not in ledger.ENV_NAMES
    # the runs in flight, the replicates, the reference task of a grid job and the hash of the CouchDB image are recorded
    assert {"GRID_PARALLEL", "GRID_REPLICATES", "GRID_REFERENCE", "COUCHDB_IMAGE_SHA256"} <= set(ledger.ENV_NAMES)
    assert not [name for name in ledger.ENV_NAMES if re.search(r"KEY|TOKEN|SECRET|PASSWORD|PASS\b", name)]


def test_a_fork_inside_another_repository_has_no_commit(tmp_path, monkeypatch):
    """On the cluster the fork is an upload without .git, so git must not report the commit of a repository around it."""
    outer = tmp_path / "outer"
    (outer / "fork").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    monkeypatch.setattr(ledger.run_data, "ROOT", outer / "fork")
    assert ledger.code_identity.git_state(["CausalDiagnosis/experiments/run_verification/run_ledger/ledger_file.py"]) == (None, None)


def _driver():
    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    return run_episodes


def test_a_scripted_run_writes_its_ledger_before_the_first_episode_and_closes_it_at_the_end(tmp_path, monkeypatch):
    run_episodes = _driver()
    grid = tmp_path / "grid.txt"
    grid.write_text("--config static --agent scripted\n", encoding="utf-8")
    job = {"SLURM_JOB_ID": "4242", "SLURM_ARRAY_JOB_ID": "4240", "SLURM_ARRAY_TASK_ID": "2", "SLURM_RESTART_COUNT": "1"}
    for name, value in {**job, "API_KEY": SECRET, "HYD_GRID_FILE": str(grid), "HYD_GRID_LINE": "0"}.items():
        monkeypatch.setenv(name, value)
    seen = []
    original = run_episodes.EpisodeRunner.play

    def play(self, ep_dir, *args, **kwargs):
        seen.append(ledger.load(Path(ep_dir).parent))
        return original(self, ep_dir, *args, **kwargs)

    monkeypatch.setattr(run_episodes.EpisodeRunner, "play", play)
    argv = ["--config", "static", "--agent", "scripted", "--seeds", "2", "--out", str(tmp_path / "runs")]
    run_dir = run_episodes.main(argv)
    assert [s["status"] for s in seen] == ["running", "running"] and seen[0]["episodes"] == {} and list(seen[1]["episodes"]) == ["seed000000"]
    record, meta = ledger.load(run_dir), read_json(run_dir / "run.json")
    assert record["status"] == "finished" and record["ended"] == meta["ended"] and record["started"] == meta["started"]
    assert record["run"] == run_dir.name and record["label"] == meta["label"]
    # the first episode writes the grade table of its rules when the checkout lacks it, and other runs of the same fork
    # share the grade and label tables, so the only event a fresh run may carry names those tables
    shared = {str(Path(record["data"][name]["path"])) for name in ("grades", "labels")}
    assert all(e["what"] == "data files changed while the run was in progress" and set(e["files"]) <= shared for e in record["events"])
    # the published ledger holds the command in its public form and names no job and no host; the private file of the run holds them
    assert record["command"]["argv"] == public_record.public_argv(argv) and record["command"]["grid_file"] == grid.name
    assert record["command"]["grid_line"] == "0" and record["command"]["grid_file_sha256"] == _sha(grid) and "API_KEY" not in record["command"]["env"]
    start = public_record.read_private(run_dir)["starts"][0]
    assert "job" not in record and start["job"] == {"slurm_job_id": "4242", "slurm_array_job_id": "4240", "slurm_array_task_id": "2",
                                                    "slurm_restart_count": "1", "host": start["job"]["host"]} and start["job"]["host"]
    assert start["command"]["argv"] == argv and start["command"]["grid_file"] == str(grid)
    assert SECRET not in (run_dir / ledger.LEDGER).read_text(encoding="utf-8") + (run_dir / public_record.PRIVATE_FILE).read_text(encoding="utf-8")
    # the code the run recorded, every listed file with its full hash per list and one identity, which the ledger hashes again at
    # the start, and the configuration and data by one value each
    flat = {path: digest for listed in meta["code"]["files"].values() for path, digest in listed.items()}
    assert record["code"]["files"] == flat and meta["code"]["identity"] == ledger.identity_of(meta["code"]["files"], "scripted")
    assert "CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py" in meta["code"]["files"]["harness"] and all(len(h) == 64 for h in flat.values())
    assert record["configuration"]["config_sha256"] == ledger.canonical_sha256(meta["config"])
    assert record["data"]["facts"]["sha256"] == _sha(meta["facts_path"]) and record["data"]["facts"]["sha256"].startswith(meta["facts_sha256"])
    assert record["seeds"]["seeds"] == meta["seeds"] == [0, 1] and record["model"] is None and record["judge"] is None
    # every file of every episode and of the run, as the run left them
    assert sorted(record["episodes"]) == ["seed000000", "seed000001"]
    for name, episode in record["episodes"].items():
        found = sorted(p.relative_to(run_dir / name).as_posix() for p in (run_dir / name).rglob("*")
                       if p.is_file() and not ledger.ignored(p.relative_to(run_dir)))
        assert sorted(episode["files"]) == found and all(episode["files"][f]["sha256"] == _sha(run_dir / name / f) for f in found)
        assert str(episode["seed"]) == name[-1]
    assert set(record["run_files"]) == {"run.json", "scores.csv", "summary.md"}
    assert all(record["run_files"][f]["sha256"] == _sha(run_dir / f) for f in record["run_files"])


ENDPOINT_FILE = """http://node001:40001/v1
job 13760001 node node001 model ibm-granite/granite-3.3-8b-instruct served_name granite-3.3-8b-instruct
started 2026-11-03T14:19:14-05:00
weights_commit 0123456789abcdef0123456789abcdef01234567
sglang 0.5.19
image /shared/images/sglang-v0.5.19-cu129.sif
image_sha256 90568174c5a27b1aa5a654014ddbbb1d4ed785c43a70657dfae15ed1f3d6addf
"""


def test_the_endpoint_file_of_the_serving_job_reaches_the_ledger_with_its_weights_commit_and_engine(tmp_path, monkeypatch):
    path = tmp_path / "13760001.txt"
    path.write_text(ENDPOINT_FILE, encoding="utf-8")
    record = ledger.endpoint_file(path)
    assert record["lines"] == ENDPOINT_FILE.splitlines() and record["base_url"] == "http://node001:40001/v1"
    assert {k: record[k] for k in ("job", "node", "model", "served_name", "started", "weights_commit", "sglang", "image_sha256")} == {
        "job": "13760001", "node": "node001", "model": "ibm-granite/granite-3.3-8b-instruct", "served_name": "granite-3.3-8b-instruct",
        "started": "2026-11-03T14:19:14-05:00", "weights_commit": "0123456789abcdef0123456789abcdef01234567", "sglang": "0.5.19",
        "image_sha256": "90568174c5a27b1aa5a654014ddbbb1d4ed785c43a70657dfae15ed1f3d6addf"}
    # a file of the serving jobs before the engine lines, and a line that is not a list of pairs, which stays in lines only
    old = tmp_path / "old.txt"
    old.write_text("\n".join(ENDPOINT_FILE.splitlines()[:3] + ["sglang"]) + "\n", encoding="utf-8")
    assert "weights_commit" not in ledger.endpoint_file(old) and "sglang" not in ledger.endpoint_file(old)
    assert ledger.endpoint_file(old)["job"] == "13760001" and ledger.endpoint_file(None) is None
    assert ledger.endpoint_file(tmp_path / "gone.txt") == {"file": str(tmp_path / "gone.txt"), "lines": None}
    # the run's model through that file, the model list asked of its endpoint only, and a file of another endpoint marked
    asked = []
    monkeypatch.setattr(ledger.serving_endpoints, "endpoint_models", lambda url: asked.append(url) or [{"id": "granite-3.3-8b-instruct"}])
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(path))
    model = ledger.served("granite-3.3-8b-instruct", "http://node001:40001/v1", "HYD_ENDPOINT_FILE")
    assert model["endpoint_file"]["matches_endpoint"] is True and model["endpoint_models"] == [{"id": "granite-3.3-8b-instruct"}]
    assert ledger.served("granite-3.3-8b-instruct", "http://node002:40002/v1", "HYD_ENDPOINT_FILE")["endpoint_file"]["matches_endpoint"] is False
    assert ledger.served(None, None, "HYD_ENDPOINT_FILE") is None
    monkeypatch.delenv("HYD_ENDPOINT_FILE")
    assert ledger.served("granite-3.3-8b-instruct", "http://node001:40001/v1", "HYD_ENDPOINT_FILE") == {
        "name": "granite-3.3-8b-instruct", "endpoint": "http://node001:40001/v1", "endpoint_file": None, "endpoint_models": None}
    assert asked == ["http://node001:40001/v1", "http://node002:40002/v1"]


AOB_RUN = "aob_ids_granite-3_3-8b-instruct_20261103-142023"
AOB_EPISODES = ("plan_execute_s005_r1", "agent_as_tool_s005_r1")


def _aob_meta(scenarios, ended=None):
    """The run.json fields of CausalDiagnosis/frameworks/AssetOpsBench/reproduction/run_scenarios/ that the ledger reads, with two code files of the fork."""
    code = {path: _sha(ledger.run_data.ROOT / path)[:12] for path in ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/run_scenarios/__init__.py", "src/agent/runner.py")}
    return {"format": 1, "kind": "aob_repro", "run": AOB_RUN, "set": "ids", "scenario_ids": [5], "repeats": 1,
            "agents": ["plan_execute", "agent_as_tool"], "model": "granite-3.3-8b-instruct", "base_url": "http://node001:40001/v1",
            "endpoint_models": [{"id": "granite-3.3-8b-instruct"}], "temperature": 0.0,
            "scenarios": {"path": str(scenarios), "sha256": _sha(scenarios), "pinned": False, "revision": None},
            "fork": {"git_describe": None, "code_sha256": code}, "host": "node001", "slurm_job_id": "13760002",
            "started": "2026-11-03T14:20:23", "ended": ended}


def _aob_run(tmp_path, ended=None):
    scenarios = tmp_path / "all_utterance.jsonl"
    scenarios.write_text('{"id": 5}\n', encoding="utf-8")
    run_dir = tmp_path / "runs" / AOB_RUN
    run_dir.mkdir(parents=True)
    meta = _aob_meta(scenarios, ended)
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir, meta


def _aob_episode(run_dir, name):
    (run_dir / name).mkdir()
    for file in ("episode.json", "llm_calls.json", f"{name}.json"):
        (run_dir / name / file).write_text(json.dumps({"file": file}), encoding="utf-8")


def _score(run_dir, judge):
    """What score_runs.py writes: a verdict per episode, the verdict of another judge renamed beside it, and scores.csv."""
    for name in AOB_EPISODES:
        verdict = run_dir / name / "judge_paper.json"
        if verdict.exists():
            verdict.rename(run_dir / name / f"judge_paper.{read_json(verdict)['judge_model']}.json")
        verdict.write_text(json.dumps({"judge_model": judge}), encoding="utf-8")
    (run_dir / "scores.csv").write_text(f"run,episode,paper_judge_model\n{AOB_RUN},x,{judge}\n", encoding="utf-8")


def test_an_assetopsbench_run_keeps_a_ledger_from_its_run_json_and_every_scoring_enters_as_an_event(tmp_path, monkeypatch):
    monkeypatch.delenv("HYD_ENDPOINT_FILE", raising=False)
    run_dir, meta = _aob_run(tmp_path)
    record = ledger.start(run_dir, meta, ["--ids", "5"])
    assert (record["label"], record["agent"], record["agent_label"]) == ("aob_repro", "plan_execute+agent_as_tool", "plan_execute+agent_as_tool")
    assert record["judge"] is None and record["seeds"] is None and "facts" not in record["data"]
    assert record["model"] == {"name": "granite-3.3-8b-instruct", "endpoint": "http://node001:40001/v1", "endpoint_file": None,
                               "endpoint_models": [{"id": "granite-3.3-8b-instruct"}]}
    assert record["code"]["files"] == {path: _sha(ledger.run_data.ROOT / path) for path in meta["fork"]["code_sha256"]}
    assert record["configuration"]["settings"]["agents"] == ["plan_execute", "agent_as_tool"]
    manifest = ledger_fields.couchdb_manifest()
    named = [p for value in json.loads(manifest.read_text(encoding="utf-8")).values() for p in (value if isinstance(value, list) else [value])]
    couchdb = {name.split(":", 1)[1]: entry for name, entry in record["data"].items() if name.startswith("couchdb:")}
    assert sorted(couchdb) == sorted(named) and all(entry["sha256"] == _sha(entry["path"]) for entry in couchdb.values())
    assert record["data"]["couchdb_manifest"]["sha256"] == _sha(manifest) and record["data"]["scenarios"]["sha256"] == meta["scenarios"]["sha256"]
    for name in AOB_EPISODES:
        _aob_episode(run_dir, name)
        assert ledger.ledger_file.add_episode(run_dir, name, {"episode": name, "scenario_id": 5, "repeat": 1})
    (run_dir / "run.json").write_text(json.dumps({**meta, "ended": "2026-11-03T14:25:57"}), encoding="utf-8")
    closed = ledger.finish(run_dir)
    assert closed["status"] == "finished" and set(closed["run_files"]) == {"run.json"} and closed["events"] == []
    # score_runs.py takes a snapshot after it loads the run and records what changed once scores.csv is written, twice here
    for judge, changed in (("judge-a", {"old": None}), ("judge-b", {})):
        before = ledger.snapshot(run_dir)
        _score(run_dir, judge)
        assert ledger.record_changes(run_dir, before, "score.py", f"scored with the arguments {AOB_RUN} --judge-model {judge}",
                                     code=ledger_fields.SCORER_CODE)
        event = ledger.load(run_dir)["events"][-1]
        assert event["tool"] == "score.py" and event["code"] == {path: _sha(ledger.run_data.ROOT / path) for path in ledger_fields.SCORER_CODE}
        assert all(event["files"][f"{name}/judge_paper.json"]["new"] == _sha(run_dir / name / "judge_paper.json") for name in AOB_EPISODES)
        assert all(event["files"][path]["old"] is None for path in event["files"]) if changed else all(
            event["files"][f"{name}/judge_paper.judge-a.json"]["old"] is None for name in AOB_EPISODES)
    assert ledger.record_changes(run_dir, ledger.snapshot(run_dir), "score.py", "nothing changed") is False
    assert verify_results.check_ledger(run_dir) == ("checked", [])


def test_the_ledger_of_an_assetopsbench_run_holds_the_serving_identity_of_the_endpoint_file_its_job_exports(tmp_path, monkeypatch):
    """CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/run_reproduction.sh exports HYD_ENDPOINT_FILE before the driver starts, and the ledger of the reproduction run then holds
    the endpoint file of the serving job, so the weights commit and the engine of the run's model stand in the run folder."""
    script = (PACKAGE / "frameworks" / "AssetOpsBench" / "cluster_jobs" / "run_reproduction.sh").read_text(encoding="utf-8")
    export = 'export HYD_ENDPOINT_FILE=$(readlink -f "${ENDPOINT_FILE:-$ENDPOINTS/$SERVED_NAME.txt}" 2>/dev/null)'
    assert script.count(export) == 1 and script.index(export) < script.index("python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_scenarios")
    path = tmp_path / "13760001.txt"
    path.write_text(ENDPOINT_FILE, encoding="utf-8")
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(path))
    monkeypatch.setattr(ledger.serving_endpoints, "endpoint_models", lambda url: [{"id": "granite-3.3-8b-instruct"}])
    run_dir, meta = _aob_run(tmp_path)
    ledger.start(run_dir, meta, ["--ids", "5"])
    model = ledger.load(run_dir)["model"]
    assert model["endpoint_file"]["matches_endpoint"] is True
    identity = ledger.serving_identity(model)
    assert (identity["weights_commit"], identity["engine"], identity["model"]) == (
        "0123456789abcdef0123456789abcdef01234567", "0.5.19", "granite-3.3-8b-instruct")


def test_a_finished_assetopsbench_run_without_a_ledger_gets_one_written_after_the_fact(tmp_path):
    run_dir, meta = _aob_run(tmp_path)
    with pytest.raises(SystemExit, match="has not finished"):
        ledger.write_after(run_dir)
    for name in AOB_EPISODES:
        _aob_episode(run_dir, name)
    _score(run_dir, "judge-a")
    (run_dir / "run.json").write_text(json.dumps({**meta, "ended": "2026-11-03T14:25:57"}), encoding="utf-8")
    listing = tmp_path / "scorer_sha256.txt"
    listing.write_text("".join(f"{'cd' * 32}  {path}\n" for path in ledger_fields.SCORER_CODE[:2]), encoding="utf-8")
    assert ledger.main(["after", str(run_dir), "--scorer-sha256", str(listing)]) == 0
    record = ledger.load(run_dir)
    assert "host" not in record["written_after_run"] and record["written_after_run"]["note"].startswith("written after the run")
    assert (record["status"], record["ended"], record["command"]) == ("finished", "2026-11-03T14:25:57", None)
    assert "job" not in record and public_record.read_private(run_dir)["starts"][0]["job"] == {"slurm_job_id": "13760002", "host": "node001"}
    assert record["code"]["files"] is None and record["code"]["recorded_prefixes"] == meta["fork"]["code_sha256"]
    assert record["model"]["endpoint_file"] is None and record["model"]["endpoint_models"] == [{"id": "granite-3.3-8b-instruct"}]
    # the judge records and scores.csv leave the episodes and the run's files for one event, as a scoring with a ledger has it
    assert [sorted(record["episodes"][name]["files"]) for name in AOB_EPISODES] == [
        sorted(["episode.json", "llm_calls.json", f"{name}.json"]) for name in AOB_EPISODES]
    assert set(record["run_files"]) == {"run.json"}
    (event,) = record["events"]
    assert event["tool"] == "ledger.write_after" and event["what"].endswith(f"with the scorer code as {listing} hashed it")
    assert sorted(event["files"]) == sorted([f"{name}/judge_paper.json" for name in AOB_EPISODES] + ["scores.csv"])
    assert event["code"] == {path: "cd" * 32 if path in ledger_fields.SCORER_CODE[:2] else None for path in ledger_fields.SCORER_CODE}
    assert verify_results.check_ledger(run_dir) == ("checked", [])
    with pytest.raises(SystemExit, match="already has a ledger"):
        ledger.write_after(run_dir)
    rig = tmp_path / "static_scripted_20261103-120000"
    rig.mkdir()
    (rig / "run.json").write_text(json.dumps({"label": "static_scripted", "ended": "2026-11-03T12:05:00"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no run of a kind whose ledger may be written after the fact"):
        ledger.write_after(rig)


# ----- the start fields of another benchmark, the record check, the serving file, the platform and the events of a synced copy -----

def test_a_driver_of_another_benchmark_starts_a_ledger_with_the_fields_it_gives(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fields = {"label": "arena", "agent": "a", "agent_label": "a_m", "model": None, "judge": None, "data": {"tasks": {"path": "t", "sha256": "0" * 64}},
              "configuration": {"settings": {}}, "seeds": None}
    record = ledger.start(run_dir, {"started": "2026-11-11T00:00:00", "code": {}}, ["--x"], fields=fields)
    assert {key: record[key] for key in ("label", "agent", "agent_label", "data", "configuration")} == {key: fields[key] for key in ("label", "agent", "agent_label", "data", "configuration")}
    assert "facts" not in record["data"]  # no data file of the rig is hashed for it
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="lack seeds"):
        ledger.start(other, {"started": "x", "code": {}}, [], fields={k: v for k, v in fields.items() if k != "seeds"})
    assert not (other / ledger.LEDGER).exists()


def _serving_file(tmp_path, monkeypatch, **changes):
    """A filled serving file of record in tmp_path, which ledger reads in place of the tracked one."""
    block = {"hub_id": "org/model", "served_name": "model", "weights_commit": "a" * 40, "dtype": "bfloat16", "engine_version": "0.5.19",
             "launch_args": {"alone": ["--dtype", "bfloat16"], "co_served": ["--dtype", "bfloat16", "--mem-fraction-static", "0.85"]}}
    record = {"format": 1, "agent": {**block, **changes}, "judge": {**block, "hub_id": "org/judge", "served_name": "judge"}}
    path = tmp_path / "serving_of_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(ledger.serving_endpoints, "SERVING_FILE", path)
    return record


def test_the_tracked_serving_file_is_complete_and_a_missing_field_or_a_config_commit_is_refused(tmp_path, monkeypatch):
    assert set(ledger.serving_endpoints.serving_of_record()) == {"format", "agent", "judge"}  # the file of the tree holds no open field
    _serving_file(tmp_path, monkeypatch, engine_version=None)
    with pytest.raises(ledger.OpenServingFile, match="agent.engine_version is missing"):
        ledger.serving_endpoints.serving_of_record()
    _serving_file(tmp_path, monkeypatch, weights_commit="config:" + "d" * 64)
    with pytest.raises(ledger.OpenServingFile, match="no pinned Hub commit"):
        ledger.serving_endpoints.serving_of_record()
    assert _serving_file(tmp_path, monkeypatch)["agent"] == ledger.serving_endpoints.serving_of_record()["agent"]


def test_serving_shell_prints_one_quoted_assignment_per_field_and_exits_3_while_the_file_is_open(tmp_path, monkeypatch, capsys):
    _serving_file(tmp_path, monkeypatch)
    assert ledger.main(["serving", "--shell", "--role", "agent"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "CD_AGENT_WEIGHTS_COMMIT=" + "a" * 40 in lines and "CD_AGENT_LAUNCH_ARGS_ALONE='--dtype bfloat16'" in lines
    assert len(lines) == 7 and not any(line.startswith("CD_JUDGE_") for line in lines)
    _serving_file(tmp_path, monkeypatch, dtype=None)
    assert ledger.main(["serving", "--shell"]) == 3 and "agent.dtype is missing" in capsys.readouterr().err


def test_serving_check_names_every_field_where_the_endpoint_file_differs(tmp_path, monkeypatch, capsys):
    _serving_file(tmp_path, monkeypatch)
    args = json.dumps(["--model-path", f"org/model@{'a' * 40}", "--served-model-name", "model", "--dtype", "bfloat16"], separators=(",", ":"))
    lines = ["http://node:1/v1", "model org/model served_name model", "weights_commit " + "a" * 40, "sglang 0.5.19",
             "dtype bfloat16 dtype_launch bfloat16 quantization fp8 kv_cache_dtype auto", f"launch_args {args}"]
    endpoint = tmp_path / "endpoint.txt"
    endpoint.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert ledger.main(["serving", "--check", str(endpoint), "--role", "agent", "--mode", "alone"]) == 0
    assert capsys.readouterr().out.splitlines() == ["SERVING_OK agent"]
    endpoint.write_text("\n".join(lines).replace("weights_commit " + "a" * 40, "weights_commit " + "e" * 40) + "\n", encoding="utf-8")
    assert ledger.main(["serving", "--check", str(endpoint), "--role", "agent", "--mode", "co_served"]) == 1
    out = capsys.readouterr().out.splitlines()
    # the quantization the server reports is recorded and never compared, so fp8 above names no difference
    assert out[0] == "SERVING_DIFFERS agent" and f"REASON weights_commit declared {'a' * 40} endpoint {'e' * 40}" in out
    assert any(line.startswith("REASON launch_args.co_served") for line in out)


def test_a_server_with_the_adapters_of_the_training_table_passes_and_its_adapter_flags_are_held_to_its_count(tmp_path, monkeypatch):
    """SERVE_ADAPTERS=1 adds the adapter flags, whose names and count only the training output knows: the comparison with a declared
    mode sets them aside, so either mode matches, and the lora_adapters line must agree with them."""
    _serving_file(tmp_path, monkeypatch)
    head = ["--model-path", f"org/model@{'a' * 40}", "--served-model-name", "model"]
    lora = ["--enable-lora", "--max-loras-per-batch", "3", "--lora-paths", "fold0", "fold1"]
    fields = {"model": "org/model", "served_name": "model", "weights_commit": "a" * 40, "sglang": "0.5.19", "dtype_launch": "bfloat16"}

    def differences(launch, adapters):
        endpoint = {**fields, "launch_args": json.dumps(launch), **({"lora_adapters": adapters} if adapters is not None else {})}
        return [field for field, _declared, _seen in ledger.serving_endpoints.serving_differences({"endpoint_file": endpoint}, "agent")]

    assert differences([*head, "--dtype", "bfloat16", *lora], "2") == []
    assert differences([*head, "--dtype", "bfloat16", "--mem-fraction-static", "0.85", *lora], "2") == []  # co-served with adapters
    assert differences([*head, "--dtype", "bfloat16"], None) == [] and differences([*head, "--dtype", "bfloat16"], "0") == []
    assert differences([*head, "--dtype", "bfloat16", *lora], "3") == ["lora_adapters"]  # the batch is not one more than the adapters
    assert differences([*head, "--dtype", "bfloat16", *lora], None) == ["lora_adapters"]
    assert differences([*head, "--dtype", "bfloat16", *lora[1:]], "2") == ["lora_adapters"]  # adapter flags without --enable-lora
    assert differences([*head, "--dtype", "bfloat16", "--enable-lora", "--max-loras-per-batch", "3", "--other", "x"], "2") == ["launch_args.alone|co_served"]


def test_the_card_binds_only_once_env_sh_names_the_card_of_record(monkeypatch):
    record = {"endpoint_file": {"file": "e.txt", "lines": ["u"], "base_url": "u", "matches_endpoint": True, "gpu_name": "NVIDIA_H100_NVL", "gpu_mib": "95830"}}
    monkeypatch.delenv("GPU_OF_RECORD", raising=False)
    assert ledger.serving_endpoints.card_differences({"endpoint_file": None}) == []
    monkeypatch.setenv("GPU_OF_RECORD", "NVIDIA H100 NVL")
    monkeypatch.setenv("GPU_MIB_OF_RECORD", "95830")
    assert ledger.serving_endpoints.card_differences(record) == []
    monkeypatch.setenv("GPU_OF_RECORD", "NVIDIA A100-SXM4-80GB")
    assert ledger.serving_endpoints.card_differences(record) == ["the endpoint serves on NVIDIA H100 NVL with 95830 MiB, and the card of record is NVIDIA A100-SXM4-80GB with 95830 MiB"]


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def scratch_repository(monkeypatch, folder):
    """Makes the tree under test a clone of a commit of its own for the rest of the test, as a run of record needs, since a copy
    of the tree is no repository: a scratch repository in folder holds a commit of every file of the tree, and git reaches it
    through GIT_DIR and GIT_WORK_TREE, so no file of the tree changes; git's identity and settings come from the environment of
    the test, never from the machine. Returns the commit."""
    import os

    environment = {"GIT_DIR": str(Path(folder) / "scratch.git"), "GIT_WORK_TREE": str(ledger.run_data.ROOT),
                   "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.org", "GIT_COMMITTER_NAME": "test",
                   "GIT_COMMITTER_EMAIL": "test@example.org", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "the tree under test"]):
        _git(ledger.run_data.ROOT, *args)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ledger.run_data.ROOT, check=True, capture_output=True, text=True).stdout.strip()


def test_the_scratch_repository_makes_the_tree_a_clone_of_its_commit_and_writes_no_file_into_it(tmp_path, monkeypatch):
    """The control of scratch_repository: with git pointed at no repository the tree is none and a loaded file is untracked
    (rule 3); after it the tree stands at the scratch commit with no path changed, the file is tracked, and the top folder of the
    tree holds the same entries as before."""
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no_repository"))
    loaded = ["CausalDiagnosis/experiments/run_verification/run_ledger/code_identity.py"]
    assert ledger.code_identity.git_state([], ledger.run_data.ROOT) == (None, None) and ledger.code_identity.untracked_loaded(loaded) == loaded
    entries = sorted(path.name for path in ledger.run_data.ROOT.iterdir())
    commit = scratch_repository(monkeypatch, tmp_path)
    assert ledger.code_identity.git_state(["benchmarks"], ledger.run_data.ROOT) == (commit, []) and ledger.code_identity.untracked_loaded(loaded) == []
    assert sorted(path.name for path in ledger.run_data.ROOT.iterdir()) == entries


def test_a_tree_plays_runs_of_record_only_as_a_clean_repository_whose_commit_stands_on_origin(tmp_path, monkeypatch):
    """Rules 1, 2 and 4: no repository, a changed file, a file of no list and a commit on no branch of origin each name their reason."""
    root = tmp_path / "clone"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    monkeypatch.setattr(ledger.code_identity, "tree_files", lambda root=None: {})
    assert ledger.code_identity.tree_record_reasons(root) == ["the tree is no git repository at its own root, as an upload of CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh is not"]
    _git(root, "init", "-q", "-b", "main")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "one")
    assert ledger.code_identity.tree_record_reasons(root)[-1].endswith("stands on no branch of origin, so no one else can check it out")
    _git(tmp_path, "init", "-q", "--bare", "origin.git")
    _git(root, "remote", "add", "origin", str(tmp_path / "origin.git"))
    _git(root, "push", "-q", "origin", "main")
    _git(root, "fetch", "-q", "origin")
    assert ledger.code_identity.tree_record_reasons(root) == []
    (root / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
    assert ledger.code_identity.tree_record_reasons(root)[0].startswith("1 paths differ from the commit")
    _git(root, "checkout", "-q", "--", "src/a.py")
    monkeypatch.setattr(ledger.code_identity, "tree_files", lambda root=None: {"": {"src/b.py": "0" * 64}})
    assert ledger.code_identity.tree_record_reasons(root) == ["1 files lie in no list of design.py, first src/b.py"]


def test_a_loaded_file_that_git_does_not_track_is_named(tmp_path):
    """Rule 3: a file that git ignores passes rule 2, so the files a process loaded are each asked of git ls-files."""
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    for name in ("kept.py", "ignored.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "kept.py", ".gitignore")
    assert ledger.code_identity.untracked_loaded(["kept.py", "ignored.py"], tmp_path) == ["ignored.py"]


def test_record_check_prints_record_with_the_commit_or_development_with_every_reason(monkeypatch, capsys):
    monkeypatch.setattr(ledger.code_identity, "record_reasons", lambda facts=None, asset=None: [])
    monkeypatch.setattr(ledger.code_identity, "git_state", lambda paths, root=None: ("f" * 40, []))
    assert ledger.main(["record-check"]) == 0 and capsys.readouterr().out.splitlines() == ["RECORD " + "f" * 40]
    monkeypatch.setattr(ledger.code_identity, "record_reasons", lambda facts=None, asset=None: ["one", "two"])
    assert ledger.main(["record-check"]) == 0 and capsys.readouterr().out.splitlines() == ["DEVELOPMENT", "REASON one", "REASON two"]


def test_no_file_of_the_tree_names_the_freeze_file_or_its_command():
    """No freeze file belongs to the design, so no code, script or page names it; uv sync --frozen and uv pip freeze are other words."""
    import re as regex

    pattern = regex.compile(r"FREEZE_|ledger\.py freeze|freeze file|frozen commit")
    hits = []
    for top in ("CausalDiagnosis", "hpc", "CausalDiagnosis/docs", "src", "README.md"):
        base = ledger.run_data.ROOT / top
        for path in [base] if base.is_file() else sorted(base.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".sh", ".sbatch", ".md", ".txt", ".json") and path.name != "test_run_ledger.py":
                text = path.read_text(encoding="utf-8", errors="replace")
                hits += [f"{path.relative_to(ledger.run_data.ROOT)}:{n}" for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
    assert hits == []


def test_the_registered_file_names_no_pilot():
    """No pilot belongs to the registered design; cluster_run_checks.py keeps a development subcommand of that name, so only the
    registered file is read."""
    text = (PACKAGE / "experiments" / "table_definitions" / "registered_analysis.json").read_text(encoding="utf-8")
    assert [n for n, line in enumerate(text.splitlines(), 1) if "pilot" in line.lower()] == []


def test_the_hash_of_the_fitted_models_ignores_the_date_of_the_build(tmp_path):
    one, two, three = (tmp_path / f"{n}.json" for n in ("one", "two", "three"))
    one.write_text(json.dumps({"generated": "2026-11-10", "fits": [1, 2]}), encoding="utf-8")
    two.write_text(json.dumps({"fits": [1, 2], "generated": "2026-11-11"}), encoding="utf-8")
    three.write_text(json.dumps({"generated": "2026-11-10", "fits": [1, 3]}), encoding="utf-8")
    assert ledger.models_sha256(one) == ledger.models_sha256(two) != ledger.models_sha256(three) and ledger.models_sha256(tmp_path / "none") is None


def test_the_installed_list_hash_is_equal_for_equal_distributions_and_moves_with_one_version(monkeypatch):
    from importlib import metadata

    class Dist:
        def __init__(self, name, version):
            self.metadata, self.version = {"Name": name}, version

    monkeypatch.setattr(metadata, "distributions", lambda: [Dist("numpy", "2.1.0"), Dist("mcp", "1.2")])
    first = ledger.installed_sha256()
    monkeypatch.setattr(metadata, "distributions", lambda: [Dist("mcp", "1.2"), Dist("numpy", "2.1.0")])
    assert ledger.installed_sha256() == first
    monkeypatch.setattr(metadata, "distributions", lambda: [Dist("mcp", "1.3"), Dist("numpy", "2.1.0")])
    assert ledger.installed_sha256() != first


def test_load_returns_the_events_of_the_ledger_and_of_the_local_file_in_time_order(tmp_path):
    run_dir, _ = _synthetic_run(tmp_path)
    before = (run_dir / ledger.LEDGER).read_bytes()
    (run_dir / ledger.LOCAL_EVENTS).write_text("[]\n", encoding="utf-8")
    assert ledger.append_event(run_dir, "rejudge.py", "judged one episode on the laptop")
    assert (run_dir / ledger.LEDGER).read_bytes() == before  # the ledger of a synced copy stays the cluster's
    events = ledger.load(run_dir)["events"]
    assert events[-1]["tool"] == "rejudge.py" and [e["at"] for e in events] == sorted(e["at"] for e in events)


def test_a_run_records_the_scenario_file_only_when_its_preset_reads_it(tmp_path, monkeypatch):
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HYD_SCENARIOS_PATH", str(scenarios))
    facts = tmp_path / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    meta = _meta(facts)
    plain = ledger.run_fields(meta, serve=lambda *args: None)
    anchored = ledger.run_fields({**meta, "config": {**meta["config"], "anchors": {"date0": "2022-07-01"}}}, serve=lambda *args: None)
    assert "scenarios" not in plain["data"] and set(anchored["data"]) == set(plain["data"]) | {"scenarios"}
    assert anchored["data"]["scenarios"] == {"path": str(scenarios), "sha256": _sha(scenarios)}


def test_finish_keeps_a_data_entry_without_a_path(tmp_path):
    """An asset table registered without a folder enters the ledger with no path and no hash, and finish closes the ledger
    beside it without an event."""
    run_dir, _ = _synthetic_run(tmp_path)
    path = run_dir / ledger.LEDGER
    record = json.loads(path.read_text(encoding="utf-8"))
    record["data"]["table_without_a_folder"] = {"path": None, "sha256": None}
    path.write_text(json.dumps(record), encoding="utf-8")
    closed = ledger.finish(run_dir)
    assert closed["status"] == "finished" and closed["events"] == []
