"""The check of a complete grid folder (check_grid_complete.py): a grid folder of scripted development runs with made-up job logs passes,
and a missing item, a byte changed without an event, a sampler copy that differs from its record and another GPU card are each
named. The runs are the driver with the scripted plan, which needs no model, and an oracle run, played once for the module; a
test that edits a grid folder edits a copy under pytest's folders. The tests assert what the check names, never a recorded value."""
import hashlib
import importlib.util
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.run_verification import check_grid_complete as check_grid
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_dirs
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

MODEL_JOB, POLICY_JOB = "101", "102"
SAMPLER_LINES = b"timestamp, utilization.gpu [%]\n2026/11/12 00:00:00.000, 41 %\n2026/11/12 00:00:05.000, 43 %\n"
LATER_LINE = b"2026/11/12 00:00:10.000, 40 %\n"  # a sampler keeps appending after the end of a run
CARD = {"gpu_name": "MADE_UP_GPU", "gpu_mib": "1024"}
OTHER = {"gpu_name": "OTHER_GPU", "gpu_mib": "2048"}  # a made-up card of record (GPU_OF_RECORD)


def _record(run_dir, change, what):
    """Changes the files of a run as a tool does, with the change recorded as one event of its ledger."""
    before = ledger.snapshot(run_dir)
    change()
    ledger.record_changes(run_dir, before, "CausalDiagnosis/experiments/run_verification/tests/test_check_grid_complete.py", what)


@pytest.fixture(scope="module")
def grids(tmp_path_factory):
    """(model grid, policy grid): grid_101 holds the scripted run on the AssetOpsBench rows made into a run of a model agent, with
    the serving record of a made-up card and sampler, and the three logs of a serving job; grid_102 holds an oracle run and the
    Slurm log of a CPU job. Both carry job_record.txt. The runs are development runs, since the tree of a test is no commit of record."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    precompute_tables.main(["--config", "base"])
    played = tmp_path_factory.mktemp("played")
    scripted = Path(run_episodes.main(["--config", "base", "--servers", "aob", "--agent", "scripted", "--seeds", "1", "--out", str(played)]))
    oracle = Path(run_episodes.main(["--config", "base", "--agent", "oracle", "--seeds", "1", "--out", str(played)]))
    root = tmp_path_factory.mktemp("runs")
    model, policy = root / f"grid_{MODEL_JOB}", root / f"grid_{POLICY_JOB}"
    for grid, run_dir, logs in ((model, scripted, (f"cd-serve-{MODEL_JOB}.out", f"sglang-{MODEL_JOB}.log")),
                                (policy, oracle, (f"cd-grid-{POLICY_JOB}.out",))):
        shutil.copytree(run_dir, grid / run_dir.name)
        (grid / check_grid.JOB_RECORD).write_text("JobID|State\n1|COMPLETED\n", encoding="utf-8")
        (grid / check_grid.JOB_LOGS).mkdir()
        for name in logs:
            (grid / check_grid.JOB_LOGS / name).write_text("a log line\n", encoding="utf-8")
    sampler = model / check_grid.JOB_LOGS / check_grid.SAMPLER.format(job=MODEL_JOB)
    sampler.write_bytes(SAMPLER_LINES + LATER_LINE)
    run_dir = model / scripted.name
    meta = read_json(run_dir / "run.json")
    record = {"sampler": {"path": f"/made/up/logs/{sampler.name}", "sha256": hashlib.sha256(SAMPLER_LINES).hexdigest(),
                          "bytes": len(SAMPLER_LINES)}, **CARD}
    _record(run_dir, lambda: write_json(run_dir / "run.json", {**meta, "agent": "causal", "serving": {"agent": record, "judge": None}}),
            "the run reads as a run of a model agent with its serving record")
    return model, policy


def _copy(grid, folder):
    return Path(shutil.copytree(grid, Path(folder) / grid.name))


def _run(grid):
    return next(p for p in grid.iterdir() if (p / "run.json").exists())


def test_a_complete_grid_folder_passes(grids, capsys):
    model, policy = grids
    assert check_grid.main(["--starts", "development", str(model), str(policy)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "CHECK_GRID_OK 2 runs in 2 grid folders"


# each removed item, with the words that name it in the output of the check
MISSING = {"job_record.txt": "job_record.txt: missing", f"job_logs/cd-serve-{MODEL_JOB}.out": "Slurm logs",
           f"job_logs/sglang-{MODEL_JOB}.log": f"sglang-{MODEL_JOB}.log missing", f"job_logs/gpu-{MODEL_JOB}.csv": f"gpu-{MODEL_JOB}.csv missing",
           "RUN/ledger.json": "ledger.json: no ledger", "RUN/run.json": "run.json: missing or not a JSON object",
           "EP": " episode folders, ", "EP/question.txt": "question.txt: missing",
           "EP/hidden/verdict.json": "hidden/verdict.json: missing", "EP/iot_server.py": "iot_server.py: missing",
           "EP/transcript.jsonl": "transcript.jsonl: missing", "EP/harness.json": "harness.json: missing",
           "EP/llm_calls.json": "llm_calls.json: missing"}


@pytest.mark.parametrize("item", sorted(MISSING))
def test_each_missing_item_is_named(grids, tmp_path, capsys, item):
    grid = _copy(grids[0], tmp_path)
    run_dir = _run(grid)
    episode = episode_dirs(run_dir)[0]
    path = Path(item.replace("RUN", str(run_dir.relative_to(grid))).replace("EP", str(episode.relative_to(grid))))
    target = grid / path
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    assert check_grid.main(["--starts", "development", str(grid)]) == 1
    out = capsys.readouterr().out
    assert MISSING[item] in out and out.splitlines()[-1].startswith("CHECK_GRID_FAILED"), out


def test_a_sampler_cut_short_or_changed_is_named(grids, tmp_path):
    for n, damage in enumerate((lambda data: data[: len(SAMPLER_LINES) - 1], lambda data: b"X" + data[1:])):
        grid = _copy(grids[0], tmp_path / str(n))
        sampler = grid / check_grid.JOB_LOGS / check_grid.SAMPLER.format(job=MODEL_JOB)
        sampler.write_bytes(damage(sampler.read_bytes()))
        problems, _runs = check_grid.check_grid(grid, True)
        assert len(problems) == 1 and ": sampler: the first" in problems[0], problems


def test_a_file_changed_without_an_event_is_named(grids, tmp_path):
    grid = _copy(grids[1], tmp_path)
    scores = _run(grid) / "scores.csv"
    scores.write_bytes(scores.read_bytes() + b"\n")
    problems, _runs = check_grid.check_grid(grid, True)
    assert problems == [f"{grid.name}/{_run(grid).name}: ledger.json: scores.csv: changed without a recorded event"], problems


def test_a_card_other_than_the_card_of_record_is_named_with_both_cards_and_no_card_of_record_checks_none(grids, tmp_path, monkeypatch):
    grid = _copy(grids[0], tmp_path)
    monkeypatch.setenv("GPU_OF_RECORD", OTHER["gpu_name"].replace("_", " "))
    monkeypatch.setenv("GPU_MIB_OF_RECORD", OTHER["gpu_mib"])
    problems = check_grid.check_run(_run(grid), False, grid / check_grid.JOB_LOGS)
    card = [p for p in problems if ": card: " in p]
    assert card == [f"{_run(grid).name}: card: the endpoint serves on MADE UP GPU with {CARD['gpu_mib']} MiB, "
                    f"and the card of record is OTHER GPU with {OTHER['gpu_mib']} MiB"], problems
    monkeypatch.delenv("GPU_OF_RECORD")
    assert not [p for p in check_grid.check_run(_run(grid), False, grid / check_grid.JOB_LOGS) if ": card: " in p]


def test_the_files_beside_the_grid_folders_are_passed_over(grids, tmp_path):
    model, policy = grids
    beside = [tmp_path / f"grid_{MODEL_JOB}.sha256", tmp_path / f"grid_{MODEL_JOB}.tar.xz.part-aa"]
    for path in beside:
        path.write_text("x\n", encoding="utf-8")
    assert check_grid.main(["--starts", "development", str(model), str(policy), *map(str, beside)]) == 0
    assert check_grid.main(["--starts", "development", *map(str, beside)]) == 2


def test_a_policy_grid_needs_no_server_log_and_no_sampler(grids):
    policy = grids[1]
    assert sorted(p.name for p in (policy / check_grid.JOB_LOGS).iterdir()) == [f"cd-grid-{POLICY_JOB}.out"]
    assert check_grid.check_grid(policy, True) == ([], 1)


def test_a_registered_check_of_a_development_run_names_its_kind(grids, tmp_path):
    grid = _copy(grids[1], tmp_path)
    problems = check_grid.check_run(_run(grid), False, grid / check_grid.JOB_LOGS)
    assert any(": run.json: a development run" in p for p in problems), problems


def _in_flight(grid, rows):
    """Writes the record of the runs in flight of a grid folder, one line of check_grid.IN_FLIGHT_FIELDS per row."""
    (grid / check_grid.IN_FLIGHT).write_text("".join("\t".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")


# a job of three runs in flight on the wall clock: run 0 ends and the last run 3 starts at 105, runs 1 and 2 end at 110 and two
# fillers take their slots, then run 3, the last run on the wall clock, ends at 130 and the job stops the fillers
FULL = [(100, "start", "units", "run", "line_0_replicate_1", 11, 1), (100, "start", "units", "run", "line_1_replicate_1", 12, 1),
        (100, "start", "units", "run", "line_2_replicate_1", 13, 1), (105, "end", "units", "run", "line_0_replicate_1", 11, 1),
        (105, "start", "units", "run", "line_3_replicate_1", 14, 1), (110, "end", "units", "run", "line_1_replicate_1", 12, 1),
        (110, "end", "units", "run", "line_2_replicate_1", 13, 1), (110, "start", "units", "filler", "filler_1", 15, 1),
        (111, "start", "units", "filler", "filler_2", 16, 1), (130, "end", "units", "run", "line_3_replicate_1", 14, 1),
        (130, "end", "units", "filler", "filler_1", 15, 1), (130, "end", "units", "filler", "filler_2", 16, 1)]


def test_the_in_flight_record_of_a_grid_folder_on_the_wall_clock_is_checked(grids, tmp_path):
    """A grid folder of runs on another clock needs no record. Once its run charges the wall seconds of thinking, the folder fails
    without the record, passes with a record in which every slot that freed took a filler, two ends in a row among them, and names
    the slot that took none and a line of the record that is not seven fields separated by tabs."""
    grid = _copy(grids[1], tmp_path)
    assert check_grid.check_grid(grid, True) == ([], 1)
    run_dir = _run(grid)
    meta = read_json(run_dir / "run.json")
    _record(run_dir, lambda: write_json(run_dir / "run.json", {**meta, "clock_mode": check_grid.WALL_CLOCK}),
            "the run reads as a run on the wall clock")
    assert check_grid.check_grid(grid, True) == (
        [f"{grid.name}: in_flight.tsv: missing, and a run of the grid folder charges the wall seconds of thinking"], 1)
    _in_flight(grid, FULL)
    assert check_grid.check_grid(grid, True) == ([], 1)
    _in_flight(grid, [row for row in FULL if row[4] != "filler_2"])
    assert check_grid.check_grid(grid, True)[0] == [
        f"{grid.name}: in_flight.tsv: the slot that line_2_replicate_1 freed at 110 in the loop units took no filler before the last "
        "run on the wall clock ended at 130"]
    _in_flight(grid, [*FULL[:3], ("x", "start"), *FULL[3:]])
    assert check_grid.check_grid(grid, True)[0] == [
        f"{grid.name}: in_flight.tsv: line 4 is not epoch, event, loop, kind, name, pid, wall separated by tabs"]


def test_the_folder_of_an_earlier_attempt_of_a_requeued_job_is_checked(grids, tmp_path, capsys):
    """A requeued job moves its grid folder aside as grid_<job id>_attempt<n> and counts its finished runs as covered, so the
    check takes such a folder as a grid folder of that job and names what it misses."""
    attempt = Path(shutil.copytree(grids[1], tmp_path / f"{grids[1].name}_attempt1"))
    assert check_grid.main(["--starts", "development", str(attempt)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "CHECK_GRID_OK 1 runs in 1 grid folders"
    (attempt / check_grid.JOB_RECORD).unlink()
    assert check_grid.main(["--starts", "development", str(attempt)]) == 1
    assert f"{attempt.name}: job_record.txt: missing" in capsys.readouterr().out
