"""The laptop side of a sync: the plan fetches only the files whose hash the local copy lacks, a first copy leaves one
event per finished run with a ledger and replaces nothing, a later change on the cluster is fetched and named with its
old and new hash, a file of a finished run that arrives with another hash stops the sync with every ledger untouched,
and a run in progress or a file only in the local copy is reported, never fatal. Folders live under pytest's tmp_path."""
import hashlib
import json
import shutil
from pathlib import Path

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import check_synced_runs as sync_check
from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _episode, _meta


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(root, name, finished, with_ledger=True, text=None, meta=None):
    run_dir = root / "grid_1" / name
    run_dir.mkdir(parents=True)
    facts = root.parent / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    if with_ledger:
        ledger.start(run_dir, _meta(facts), ["--seeds", "1"])
    _episode(run_dir, "seed000000", text or f"{name}\n")
    if with_ledger:
        ledger.ledger_file.add_episode(run_dir, "seed000000", {"seed": 0, "block": "[100,100,0,90]"})
    if finished:
        (run_dir / "run.json").write_text(json.dumps({"ended": "2026-11-03T12:05:00", **(meta or {})}), encoding="utf-8")
        (run_dir / "scores.csv").write_text("run,seed\nx,0\n", encoding="utf-8")
        if with_ledger:
            ledger.finish(run_dir)
    return run_dir


def _cluster(tmp_path):
    """A runs folder as the cluster holds it: a finished run, a run in progress, a run from before the ledger, a grid log."""
    remote = tmp_path / "cluster" / "runs"
    _run(remote, "static_scripted_done", finished=True)
    _run(remote, "static_scripted_running", finished=False)
    _run(remote, "static_scripted_old", finished=True, with_ledger=False)
    (remote / "grid_1" / "line_0.log").write_text("### line 0\n", encoding="utf-8")
    return remote


def _manifest(remote, out):
    """The cluster's hashes as find . -type f | xargs sha256sum prints them."""
    lines = [f"{_sha(p)}  ./{p.relative_to(remote).as_posix()}\n" for p in sorted(remote.rglob("*")) if p.is_file()]
    out.write_text("".join(lines), encoding="utf-8")
    return out


def _fetch(remote, local, paths):
    for path in paths:
        (local / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / path, local / path)


def _sync(tmp_path, remote, local, name, during=None, damage=None):
    """One sync: the cluster's hashes, the plan, then during() on the cluster before the files travel, the fetch, then
    damage() to the local copy after it."""
    manifest = _manifest(remote, tmp_path / f"{name}.sha256")
    planned = sync_check.plan(manifest, local)
    if during:
        during()
    _fetch(remote, local, planned)
    if damage:
        damage()
    return planned, sync_check.check(manifest, local, planned, "insomnia:/runs")


def test_a_first_sync_fetches_everything_and_leaves_one_event_per_finished_run_with_a_ledger(tmp_path):
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    planned, (problems, warnings, summary) = _sync(tmp_path, remote, local, "first")
    assert set(planned) == {p.relative_to(remote).as_posix() for p in remote.rglob("*") if p.is_file()} and set(planned.values()) == {None}
    assert problems == [] and warnings == []
    assert summary == {"files": len(planned), "copied": len(planned), "runs": 3, "events": 1, "runs_in_progress": 1, "runs_without_ledger": 1}
    event = ledger.load(local / "grid_1" / "static_scripted_done")["events"][-1]
    assert event["tool"] == "sync_runs.sh" and event["files"] == {}
    # the source names a host of the cluster, which no published file holds; the private file of the run travels with it
    assert event["what"] == "copied 7 files from the cluster, every file of the run checked against the cluster's hashes"
    assert ledger.load(local / "grid_1" / "static_scripted_running")["events"] == []


def test_a_change_on_the_cluster_is_fetched_alone_and_named_with_its_old_and_new_hash(tmp_path):
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    _sync(tmp_path, remote, local, "first")
    report = remote / "grid_1" / "static_scripted_done" / "seed000000" / "report.md"
    old = _sha(report)
    report.write_text("judged again on the cluster\n", encoding="utf-8")
    planned, (problems, _, summary) = _sync(tmp_path, remote, local, "second")
    # the ledger of the copy stays the cluster's, with the event of the first sync beside it, so it travels no more
    assert set(planned) == {"grid_1/static_scripted_done/seed000000/report.md"}
    assert problems == [] and summary["copied"] == 1
    event = ledger.load(local / "grid_1" / "static_scripted_done")["events"][-1]
    assert event["files"] == {"seed000000/report.md": {"old": old, "new": _sha(report)}}


def test_a_finished_run_whose_copy_differs_stops_the_sync_and_leaves_every_ledger_untouched(tmp_path):
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    _sync(tmp_path, remote, local, "first")
    (remote / "grid_1" / "static_scripted_done" / "scores.csv").write_text("run,seed\ny,0\n", encoding="utf-8")
    copy = local / "grid_1" / "static_scripted_done" / "scores.csv"
    before = ledger.load(local / "grid_1" / "static_scripted_done")
    _, (problems, _, summary) = _sync(tmp_path, remote, local, "second", damage=lambda: copy.write_text("damaged\n", encoding="utf-8"))
    assert problems == ["grid_1/static_scripted_done/scores.csv: the copy differs from the cluster's hash in a finished run"]
    assert summary["events"] == 0
    # the ledger is the cluster's, the event of the first sync survives beside it, and this sync adds none
    assert ledger.load(local / "grid_1" / "static_scripted_done")["events"] == before["events"]


def test_two_runs_of_one_label_stay_two_runs_though_only_the_training_variant_and_the_start_time_tell_them_apart(tmp_path):
    """A trained agent and the agent without training carry one agent label, and their folder names differ by the start time alone. A
    sync reads no label and no setting: it takes a run by its folder, so neither run stands in for the other, even with files of the
    same bytes."""
    remote, local = tmp_path / "cluster" / "runs", tmp_path / "mac"
    released, trained = "real_time_causal_m_20261102-130000", "real_time_causal_m_20261102-130500"
    for name, variant in ((released, "as_released"), (trained, "gepa")):
        _run(remote, name, finished=True, text="the same report\n", meta={"agent_label": "causal_m", "training_variant": variant})
    planned, (problems, warnings, summary) = _sync(tmp_path, remote, local, "first")
    assert set(planned) == {p.relative_to(remote).as_posix() for p in remote.rglob("*") if p.is_file()}
    assert problems == [] and warnings == [] and (summary["runs"], summary["events"]) == (2, 2)
    for name, variant in ((released, "as_released"), (trained, "gepa")):
        assert json.loads((local / "grid_1" / name / "run.json").read_text(encoding="utf-8"))["training_variant"] == variant
        assert ledger.load(local / "grid_1" / name)["events"][-1]["tool"] == "sync_runs.sh"
    copy = local / "grid_1" / trained / "seed000000" / "report.md"
    (remote / "grid_1" / trained / "seed000000" / "report.md").write_text("judged again on the cluster\n", encoding="utf-8")
    _, (problems, _, _) = _sync(tmp_path, remote, local, "second", damage=lambda: copy.write_text("damaged\n", encoding="utf-8"))
    assert problems == [f"grid_1/{trained}/seed000000/report.md: the copy differs from the cluster's hash in a finished run"]


def test_a_run_without_a_ledger_is_finished_only_once_run_json_holds_its_end_time(tmp_path):
    """The AssetOpsBench reproduction rewrites run.json after every episode and sets ended last, so a rewrite caught between
    the cluster's hashes and the copy is a run in progress, never a mismatch."""
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    meta = remote / "grid_1" / "aob_repro_running" / "run.json"
    meta.parent.mkdir()

    def episodes_done(n, ended=None):
        meta.write_text(json.dumps({"kind": "aob_repro", "episodes_done": n, "ended": ended}), encoding="utf-8")
    episodes_done(1)
    _sync(tmp_path, remote, local, "first")
    episodes_done(2)
    _, (problems, warnings, _) = _sync(tmp_path, remote, local, "second", during=lambda: episodes_done(3))
    assert problems == [] and warnings == ["grid_1/aob_repro_running/run.json: changed during the transfer (a run in progress)"]
    assert not sync_check.finished(meta.parent)
    episodes_done(3, ended="2026-11-03T16:00:00")
    assert sync_check.finished(meta.parent)


def test_a_run_in_progress_and_a_file_only_in_the_local_copy_are_reported_and_never_fatal(tmp_path):
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    _sync(tmp_path, remote, local, "first")
    running = remote / "grid_1" / "static_scripted_running" / "seed000000" / "report.md"
    running.write_text("the run went on\n", encoding="utf-8")
    (local / "grid_1" / "static_scripted_done" / "notes.txt").write_text("kept on the laptop\n", encoding="utf-8")
    _, (problems, warnings, summary) = _sync(tmp_path, remote, local, "second",
                                              during=lambda: running.write_text("and on, while the files travelled\n", encoding="utf-8"))
    assert problems == [] and summary["events"] == 0  # the sync copied no file of the finished run, so it records no event there
    assert warnings == ["grid_1/static_scripted_running/seed000000/report.md: changed during the transfer (a run in progress)",
                        "grid_1/static_scripted_done/notes.txt: only in the local copy, the cluster's run folder lacks it"]


def test_a_second_sync_with_no_change_on_the_cluster_fetches_nothing(tmp_path):
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    _sync(tmp_path, remote, local, "first")
    ledger_file = local / "grid_1" / "static_scripted_done" / ledger.LEDGER
    assert ledger_file.read_bytes() == (remote / "grid_1" / "static_scripted_done" / ledger.LEDGER).read_bytes()
    planned, (problems, warnings, _) = _sync(tmp_path, remote, local, "second")
    assert planned == {} and problems == [] and not [w for w in warnings if ledger.LOCAL_EVENTS in w]


def test_an_event_written_on_the_laptop_survives_the_next_sync(tmp_path):
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    _sync(tmp_path, remote, local, "first")
    run = local / "grid_1" / "static_scripted_done"
    assert ledger.append_event(run, "archive", "played again by mistake")
    (remote / "grid_1" / "static_scripted_done" / "seed000000" / "report.md").write_text("judged again on the cluster\n", encoding="utf-8")
    _sync(tmp_path, remote, local, "second")
    tools = [event["tool"] for event in ledger.load(run)["events"]]
    assert "archive" in tools and tools.count("sync_runs.sh") == 2


def test_a_sync_that_copies_nothing_of_a_run_leaves_its_events_as_they_are(tmp_path):
    """A sync records an event in a finished run only when it copied files of that run, so the events of a run stay as they
    are through a sync that found nothing new on the cluster."""
    remote, local = _cluster(tmp_path), tmp_path / "mac"
    _sync(tmp_path, remote, local, "first")
    run = local / "grid_1" / "static_scripted_done"
    before = ledger.load(run)["events"]
    planned, (problems, _warnings, summary) = _sync(tmp_path, remote, local, "second")
    assert planned == {} and problems == [] and summary["events"] == 0 and ledger.load(run)["events"] == before
