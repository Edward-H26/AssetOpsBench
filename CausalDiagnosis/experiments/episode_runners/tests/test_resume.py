"""Taking up a run that a job left partly played (run_episodes/ --resume). The run keeps its finished episodes byte for byte and
plays the rest as an uninterrupted run of the same command would; an episode played without its row, a row without its ledger
record and a row half written are set aside and played again; the ledger check and the replay accept the run; a folder that the
command cannot take up is left as it is while the line plays from its first episode. pending_grid_lines.py names the run in progress,
the fillers drop --resume and check_grid_complete.py names the run in progress of a grid folder. A run of --start-state and a run of
--scenario-ids are taken up as any run, and a command of another selection takes up neither. The runs are the oracle on the
development starts, which needs no model, cut where a time limit or a lost node would stop the driver, and the runs of the selectors
the oracle at one start state and doing nothing on made-up scenarios of the anchored preset."""
import csv
import importlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.run_verification import check_grid_complete as check_grid
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
from CausalDiagnosis.experiments.cluster_runs import grid_job_lists as grid_lines
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name, read_rows
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.dataset_loader.recordings import public_path
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

LINE = "--config base --agent oracle"
DEVELOPMENT = ["--starts", "development", "--start-cap", "5"]  # the fewest starts that cover every faulty level of the rig
CUT = 2  # the episodes a job finishes before it is cut
LABEL = "resumed_run"  # the label of every run here, so that the folders and the rows of two plays of one command compare
PLACES = ("boundary", "row", "play", "torn")
# the keys that hold a time, a reading of a clock, a value computed from such readings or a process id, by file: the one list
# of what two plays of one command may differ in, which _normal leaves out, and nothing else. In the transcript these are the
# readings that its servers stamp, which a replay feeds back from the record and does not compare (replay_check.recorded_stamps,
# the handover and the ready lines); the state file, run.json and scores.csv hold the values that the thinking clock computes
# from them, and the state file the byte offset of the transcript read so far, which a replay leaves out as well
# (replay_check.LEFT_FILES). A key "a.b" names the key b of the object under a.
TIME_KEYS = {"episode.json": {"called_at", "episode_started_at", "episode_ended_at", "last_call_monotonic_time", "last_call_wall_time",
                              "handover_wall", "handover_mono", "charged_until_mono", "credited_seconds", "uncredited_seconds",
                              "open_service.e", "open_service.e_thread", "transcript_cursor"},
             "transcript.jsonl": {"pid", "wall", "mono", "thread_cpu", "end_wall", "end_mono", "end_thread_cpu",
                                  "s_wall", "s_mono", "s_thread_cpu", "r_wall", "r_mono", "r_thread_cpu"},
             "run.json": {"started", "ended", "load_averages", "clock.credited_seconds", "clock.uncredited_seconds",
                          "clock.uncharged_after_last_seconds"},
             "scores.csv": {"wall_s", "credited_s", "uncredited_s", "uncharged_after_last_s"}}


class Cut(Exception):
    """The end of a job, raised where a time limit or a lost node would stop the driver."""


def _play(*arguments, words=(*LINE.split(), *DEVELOPMENT)):
    return Path(run_episodes.main([*words, *arguments]))


def _cut(out, where, words=(*LINE.split(), *DEVELOPMENT)):
    """The folder of a run of LINE under out that a job left partly played. At the boundary the job ends right after the ledger
    record of episode CUT; at row after the row of episode CUT + 1 and before its ledger record; at play after episode CUT + 1
    played and before its row, with the scratch folder of the tool listing left beside it; and torn at the boundary with the row of
    episode CUT half written."""
    real_add, real_play, count = ledger.ledger_file.add_episode, run_episodes.EpisodeRunner.play, {"add": 0, "play": 0}

    def add_episode(run_dir, name, row):
        count["add"] += 1
        if where == "row" and count["add"] == CUT + 1:
            raise Cut
        real_add(run_dir, name, row)
        if where in ("boundary", "torn") and count["add"] == CUT:
            raise Cut

    def play(self, ep_dir, *arguments):
        count["play"] += 1
        row = real_play(self, ep_dir, *arguments)
        if where == "play" and count["play"] == CUT + 1:
            scratch = Path(ep_dir).parent / run_episodes.SERVER_CHECK_FOLDER
            scratch.mkdir()
            (scratch / "state.json").write_text("{}", encoding="utf-8")
            raise Cut
        return row

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ledger.ledger_file, "add_episode", add_episode)
        patch.setattr(run_episodes.EpisodeRunner, "play", play)
        with pytest.raises(Cut):
            _play("--out", str(out), words=words)
    run_dir = out / LABEL
    if where == "torn":
        data = (run_dir / "scores.csv").read_bytes()
        header, first, second = data.split(b"\r\n")[:3]
        (run_dir / "scores.csv").write_bytes(data[:len(header) + len(first) + 4 + len(second) // 2])
    return run_dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{"whole": the uninterrupted run, "copy": a copy of the run cut at the boundary in a grid folder, taken before it was taken up,
    "resumed": {place of the cut: the run taken up into another --out}}. The cut runs play first, so that the generated tables that
    the first run of a preset completes stand complete for the uninterrupted run and every run taken up."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports

    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        # the generated tables of these runs are copies of their own for the module, since a run here may add a table that the
        # session copies lack, and a run of another module may have recorded those copies
        for name, attribute in ports.generated_tables():
            module = importlib.import_module(name)
            shared = Path(getattr(module, attribute))
            own = root / "tables" / name / shared.name
            own.parent.mkdir(parents=True, exist_ok=True)
            if shared.exists():
                shutil.copyfile(shared, own)
            patch.setattr(module, attribute, own)
        precompute_tables.main(["--config", "base"])
        patch.setattr(run_episodes.run_record, "run_label", lambda config_name, agent_label: LABEL)
        progress = {where: _cut(root / f"cut_{where}", where) for where in PLACES}
        copy = root / "grid_7" / LABEL
        shutil.copytree(progress["boundary"], copy)
        whole = _play("--out", str(root / "whole"))
        resumed = {where: _play("--resume", str(progress[where]), "--out", str(root / f"resumed_{where}")) for where in PLACES}
        yield {"whole": whole, "copy": copy, "resumed": resumed}


@pytest.fixture(scope="module")
def selected(tmp_path_factory):
    """{selector: (the line, a copy of its run cut at the boundary in a grid folder, the run taken up into another --out, the
    uninterrupted run)} for a line of --start-state and one of --scenario-ids, played with tables of their own as the runs of the
    fixture runs are, the cut run first."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
    from CausalDiagnosis.experiments.episode_runners.tests.test_run_episodes import _scenario_file

    root = tmp_path_factory.mktemp("selected")
    with pytest.MonkeyPatch.context() as patch:
        for name, attribute in ports.generated_tables():
            module = importlib.import_module(name)
            shared = Path(getattr(module, attribute))
            own = root / "tables" / name / shared.name
            own.parent.mkdir(parents=True, exist_ok=True)
            if shared.exists():
                shutil.copyfile(shared, own)
            patch.setattr(module, attribute, own)
        precompute_tables.main(["--config", "base"])
        patch.setattr(run_episodes.run_record, "run_label", lambda config_name, agent_label: LABEL)
        ids = _scenario_file(root, patch, count=4)
        lines = {"start_state": f"{LINE} --start-state 3/100/0/130@1 --seeds 3",
                 "scenario_ids": f"--config anchored --agent do_nothing --seeds 1 --no-diagnostics --no-grade --scenario-ids {'/'.join(map(str, ids[1:]))}"}
        played = {}
        for selector, line in lines.items():
            cut = _cut(root / f"cut_{selector}", "boundary", line.split())
            copy = root / f"grid_{selector}" / LABEL
            shutil.copytree(cut, copy)
            whole = _play("--out", str(root / f"whole_{selector}"), words=line.split())
            resumed = _play("--resume", str(cut), "--out", str(root / f"resumed_{selector}"), words=line.split())
            played[selector] = (line, copy, resumed, whole)
        yield played


def _without_times(value, keys):
    """A JSON object without the keys of its file in TIME_KEYS, a key "a.b" taken out of the object under a."""
    value = {k: v for k, v in value.items() if k not in keys}
    for key in keys:
        outer, _, inner = key.partition(".")
        if inner and isinstance(value.get(outer), dict):
            value[outer] = {k: v for k, v in value[outer].items() if k != inner}
    return value


def _normal(path):
    """The content of a file of a run folder with every time and process id of TIME_KEYS taken out, and summary.md
    without its sentence on the start and the end."""
    kind = path.name
    if kind == "scores.csv":
        with open(path, newline="", encoding="utf-8") as f:
            return [{**row, **dict.fromkeys(TIME_KEYS[kind])} for row in csv.DictReader(f)]
    if kind == "summary.md":
        return re.sub(r" Started [^,]+, ended [^.]+\.", "", path.read_text(encoding="utf-8"))
    if kind.endswith(".jsonl") and kind in TIME_KEYS:
        return [_without_times(json.loads(line), TIME_KEYS[kind]) for line in path.read_text(encoding="utf-8").splitlines()]
    if kind in TIME_KEYS:
        return _without_times(json.loads(path.read_text(encoding="utf-8")), TIME_KEYS[kind])
    return path.read_bytes()


def _files(run_dir):
    """{path: content as _normal reads it} of every file of a run folder but its ledger and the folder of what was set aside."""
    run_dir = Path(run_dir)
    return {path.relative_to(run_dir).as_posix(): _normal(path) for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != ledger.LEDGER and ledger.INTERRUPTED not in path.relative_to(run_dir).parts
            and not ledger.ignored(path.relative_to(run_dir))}  # the private files of a run, which hold its own jobs and times


def _bytes(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(Path(folder).rglob("*")) if path.is_file()}


def _names(run_dir):
    """The episodes of a run in the order of its plan, as its scores.csv lists them."""
    return [episode_name(row["seed"], row.get("scenario_id")) for row in read_rows(run_dir)]


def _resume_event(run_dir):
    events = [event for event in read_json(Path(run_dir) / ledger.LEDGER)["events"] if event["tool"] == ledger.RESUME_TOOL]
    assert len(events) == 1
    return events[0]


def test_a_run_cut_at_the_boundary_of_an_episode_and_taken_up_equals_the_uninterrupted_run(runs):
    """The run that a job cut after its second episode, taken up by a later job into another --out, holds every file of the
    uninterrupted run of the same command, equal apart from the times and the process ids, keeps its first episodes byte for byte
    as the cut job left them, sets nothing aside, and closes a ledger that the ledger check accepts with the job named."""
    resumed, whole, copy = runs["resumed"]["boundary"], runs["whole"], runs["copy"]
    assert _files(resumed) == _files(whole)
    assert not (resumed / ledger.INTERRUPTED).exists()
    kept = _names(whole)[:CUT]
    assert sorted(read_json(copy / ledger.LEDGER)["episodes"]) == sorted(kept)
    assert all(_bytes(resumed / name) == _bytes(copy / name) for name in kept)
    record, other = read_json(resumed / ledger.LEDGER), read_json(whole / ledger.LEDGER)
    assert {key: record[key] for key in ("label", "agent", "agent_label", "configuration", "seeds", "status", "start_check")} == \
           {key: other[key] for key in ("label", "agent", "agent_label", "configuration", "seeds", "status", "start_check")}
    assert {name: sorted(e["files"]) for name, e in record["episodes"].items()} == {name: sorted(e["files"]) for name, e in other["episodes"].items()}
    assert sorted(record["run_files"]) == sorted(other["run_files"])
    event = _resume_event(resumed)
    # the event names the folder and the command in their public form, a path outside the tree by its last part
    assert event["what"] == f"taken up again after {CUT} finished episodes, moved from {public_path(copy.parent.parent / 'cut_boundary' / LABEL)}"
    assert event["files"] == {} and event["command"]["argv"][-4:] == ["--resume", public_path(copy.parent.parent / "cut_boundary" / LABEL),
                                                                     "--out", public_path(resumed.parent)]
    assert not (copy.parent.parent / "cut_boundary" / LABEL).exists()
    assert verify_results.check_ledger(resumed) == ("checked", [])


def test_the_comparison_of_two_plays_leaves_out_the_readings_of_a_clock_alone(runs, tmp_path):
    """TIME_KEYS leaves out of the comparison the readings of a clock and the values computed from them and nothing else: a copy
    of the uninterrupted run whose readings moved stays equal to it, while a copy with the answer of one call changed, or with one
    count of thinking cycles of an episode raised by one, differs from it."""
    whole = runs["whole"]
    name = _names(whole)[0]

    def copy_with(label, path, change):
        copy = tmp_path / label
        shutil.copytree(whole, copy)
        target = copy / name / path
        if path.endswith(".jsonl"):
            lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            change(next(line for line in lines if line["kind"] == "call"))
            target.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        else:
            state = json.loads(target.read_text(encoding="utf-8"))
            change(state)
            target.write_text(json.dumps(state, indent=1), encoding="utf-8")
        return _files(copy)

    files = _files(whole)
    assert copy_with("moved", "transcript.jsonl", lambda call: call.update(end_mono=call["end_mono"] + 1.0)) == files
    assert copy_with("answer", "transcript.jsonl", lambda call: call.update(answer_sha256="0" * 64)) != files
    assert copy_with("cycles", "episode.json", lambda state: state.update(think_cycles=[*state["think_cycles"][:-1], state["think_cycles"][-1] + 1])) != files


@pytest.mark.parametrize("where", PLACES[1:])
def test_an_unfinished_episode_is_set_aside_and_played_again(runs, where):
    """An episode played without its row, an episode with its row and without its ledger record, and a finished episode whose row
    the cut job left half written are moved into interrupted/ with the scratch folder of the tool listing, named with the hash of
    every file in the one event of the taking up, and played again from their start: the run equals the uninterrupted run apart
    from the times and the folder set aside, and the ledger check accepts it."""
    resumed, whole = runs["resumed"][where], runs["whole"]
    names = _names(whole)
    assert _files(resumed) == _files(whole)
    aside = {"row": [names[CUT]], "play": [names[CUT], run_episodes.SERVER_CHECK_FOLDER], "torn": [names[CUT - 1]]}[where]
    assert sorted(path.name for path in (resumed / ledger.INTERRUPTED).iterdir()) == sorted(aside)
    event = _resume_event(resumed)
    kept = CUT - 1 if where == "torn" else CUT
    assert event["what"].startswith(f"taken up again after {kept} finished episodes, moved from ")
    assert f"; set aside into {ledger.INTERRUPTED}: " + ", ".join(f"{name} as {name}" for name in aside) in event["what"]
    assert event["what"].endswith(f"; ledger records dropped: {names[CUT - 1]}") == (where == "torn")
    moved = {f"{ledger.INTERRUPTED}/{path}": {"old": None, "new": ledger.sha256_of(resumed / ledger.INTERRUPTED / path)}
             for path in _bytes(resumed / ledger.INTERRUPTED) if not ledger.ignored(path)}
    assert event["files"] == moved and moved
    assert verify_results.check_ledger(resumed) == ("checked", [])


def test_the_run_taken_up_replays_as_any_run(runs):
    """The replay check composes the run from the command its ledger recorded and plays every episode of the run folder again
    through the servers of this tree, the folder set aside left out, with every answer, file and verdict equal."""
    for run in (runs["resumed"]["boundary"], runs["resumed"]["play"]):
        state, problems = verify_results.check_replay(run)
        assert problems == [] and state.startswith(f"{len(_names(runs['whole']))} episodes replayed"), (run, state)


def _in_progress(run_dir, folder, **ledger_fields):
    """A copy of a finished run under folder as a job would leave it in progress: no run.json and its ledger running, with
    ledger_fields over the fields of its ledger."""
    copy = Path(folder) / Path(run_dir).name
    shutil.copytree(run_dir, copy)
    (copy / "run.json").unlink()
    record = {**read_json(copy / ledger.LEDGER), "status": "running", "ended": None, **ledger_fields}
    (copy / ledger.LEDGER).write_text(json.dumps(record), encoding="utf-8")
    return copy


def test_the_command_takes_up_only_the_run_it_plays_and_names_the_first_field_that_differs(runs, tmp_path):
    """ledger.resume_refusal holds a run in progress to the command that plays it: the fields of its ledger, the hashes of its
    recordings, the names of its model and its judge, every code file it hashed, and the start check, which registers both parts
    or stamps both. A run stopped after harness errors in a row is no run in progress, and neither is a finished run."""
    whole = runs["whole"]
    meta = read_json(whole / "run.json")
    reasons = meta["stamp_reasons"]
    copy = _in_progress(whole, tmp_path / "same")
    assert ledger.in_progress(copy) and not ledger.in_progress(whole) and reasons
    assert ledger.resume_refusal(copy, meta, reasons) is None
    record = read_json(copy / ledger.LEDGER)
    path = sorted(record["code"]["files"])[0]
    cases = [({**meta, "seeds": meta["seeds"][:-1]}, reasons, {}, "the field seeds of its ledger differs from the command's"),
             ({**meta, "agent_label": "other"}, reasons, {}, "the field agent_label of its ledger differs from the command's"),
             ({**meta, "model": "another-model"}, reasons, {}, "its model None differs from the command's another-model"),
             (meta, [], {}, "the start check stamped its first part and registers this one"),
             (meta, reasons, {"data": {**record["data"], "facts": {**record["data"]["facts"], "sha256": "0" * 64}}},
              "its data file facts differs from the command's"),
             (meta, reasons, {"code": {**record["code"], "files": {**record["code"]["files"], path: "0" * 64}}},
              f"1 of its code files changed since it started: {path}")]
    for n, (fields, given, changed, refusal) in enumerate(cases):
        assert ledger.resume_refusal(_in_progress(whole, tmp_path / f"case{n}", **changed), fields, given) == refusal, n
    unchecked = _in_progress(whole, tmp_path / "unchecked")
    without = read_json(unchecked / ledger.LEDGER)
    del without["start_check"]
    (unchecked / ledger.LEDGER).write_text(json.dumps(without), encoding="utf-8")
    assert ledger.resume_refusal(unchecked, meta, reasons) == "its ledger records no start check"
    stopped = _in_progress(whole, tmp_path / "stopped", events=[{"at": None, "tool": "run_episodes.py", "what": failures.STOP_EVENT, "files": {}}])
    assert not ledger.in_progress(stopped)


def test_a_folder_the_command_cannot_take_up_is_left_as_it_is_and_the_line_plays_from_its_first_episode(runs, tmp_path, capsys):
    """--resume of a folder that holds no run in progress, a finished run here, leaves every byte of it and plays the line from its
    first episode into --out, as without --resume, so that a pending line never stops on a run it cannot take up; a filler with
    --resume is refused."""
    whole = runs["whole"]
    before = _bytes(whole)
    run = _play("--resume", str(whole), "--out", str(tmp_path / "fresh"))
    assert _bytes(whole) == before
    assert f"--resume {whole}: {run_episodes.RESUME_NONE}; the line plays from its first episode" in capsys.readouterr().out
    assert run.parent == (tmp_path / "fresh").resolve() and len(read_rows(run)) == len(_names(whole)) and (run / "run.json").is_file()
    with pytest.raises(SystemExit) as stop:
        run_episodes.main([*LINE.split(), *DEVELOPMENT, "--resume", str(whole), "--filler"])
    assert str(stop.value) == run_episodes.RESUME_FILLER


def test_pending_lines_names_the_run_in_progress_with_resume_and_nothing_once_the_pair_is_done(runs, tmp_path):
    """A pending pair that a run in progress plays is printed with --resume and the run folder; a finished run of the pair leaves
    nothing pending; a run that started before the cutoff of its agent, a filler and a run stopped after harness errors in a row
    are not taken up; and the filler arguments of a pending line drop --resume with its folder."""
    copy = runs["copy"]
    grid = copy.parent
    assert pending_lines.pending([LINE], [grid], {}, starts="development") == [f"{LINE} --resume {copy}"]
    assert pending_lines.pending([LINE], [grid, runs["resumed"]["boundary"].parent], {}, starts="development") == []
    assert pending_lines.pending([LINE], [grid], {"oracle": "2999-01-01T00:00:00"}, starts="development") == [LINE]
    record = read_json(copy / ledger.LEDGER)
    for name, changed in (("filler", {"command": {**record["command"], "argv": [*record["command"]["argv"], "--filler"]}}),
                          ("stopped", {"events": [{"at": None, "tool": "run_episodes.py", "what": failures.STOP_EVENT, "files": {}}]})):
        folder = tmp_path / name
        shutil.copytree(copy, folder / copy.name)
        (folder / copy.name / ledger.LEDGER).write_text(json.dumps({**record, **changed}), encoding="utf-8")
        assert pending_lines.pending([LINE], [folder], {}, starts="development") == [LINE], name
    assert grid_lines.filler_arguments(f"{LINE} --replicate 2 --resume {copy}") == LINE


@pytest.mark.parametrize("selector", ["start_state", "scenario_ids"])
def test_a_run_of_a_selector_cut_at_the_boundary_and_taken_up_equals_the_uninterrupted_run(selected, selector):
    """A run of --start-state and a run of --scenario-ids that a job cut after their second episode, taken up by a later job into
    another --out, hold every file of the uninterrupted run of the same command, equal apart from the times and the process ids,
    record the same selection in run.json and close a ledger that the ledger check accepts."""
    _line, _copy, resumed, whole = selected[selector]
    assert len(_names(whole)) == 3 and _names(resumed) == _names(whole)
    assert _files(resumed) == _files(whole)
    assert read_json(resumed / "run.json")[selector] == read_json(whole / "run.json")[selector] is not None
    assert not (resumed / ledger.INTERRUPTED).exists() and verify_results.check_ledger(resumed) == ("checked", [])


def test_a_command_of_another_selection_takes_up_no_run_of_a_selector(selected, tmp_path):
    """The ledger holds the selection of a run, the start state in its configuration and the scenarios beside its seeds, so a
    command of another start state or of other scenarios is refused the run in progress with the field that differs, and
    pending_grid_lines.py names the run in progress with --resume for its own line alone."""
    other = {"start_state": {"combination": [3, 100, 0, 130], "entry_offset": 2}, "scenario_ids": [901]}
    for selector, (line, copy, _resumed, whole) in selected.items():
        meta = read_json(whole / "run.json")
        reasons = meta["stamp_reasons"]
        progress = _in_progress(whole, tmp_path / selector)
        assert ledger.resume_refusal(progress, meta, reasons) is None
        changed = {**meta, selector: other[selector]}
        if selector == "start_state":
            changed["config"] = {**meta["config"], "start": other[selector]}
        field = "configuration" if selector == "start_state" else "seeds"
        assert ledger.resume_refusal(progress, changed, reasons) == f"the field {field} of its ledger differs from the command's", selector
        elsewhere = line.replace("@1", "@2") if selector == "start_state" else line.rsplit("/", 1)[0]
        assert elsewhere != line
        assert pending_lines.pending([line, elsewhere], [copy.parent], {}, starts="seeds") == [f"{line} --resume {copy}", elsewhere], selector


def test_check_grid_names_a_run_in_progress_of_a_grid_folder_with_its_recorded_episodes(runs):
    """A run in progress of a grid folder is one problem line that names its recorded and its planned episodes and the submission
    that takes it up, in place of the lines of a run without run.json."""
    copy = runs["copy"]
    planned = len(_names(runs["whole"]))
    assert check_grid.check_run(copy, True, copy.parent / check_grid.JOB_LOGS) == [
        f"{copy.name}: {ledger.LEDGER}: a run in progress, with {CUT} of its {planned} planned episodes recorded; its part submitted "
        "again with GRID_PENDING=1 takes it up from its last finished episode"]
