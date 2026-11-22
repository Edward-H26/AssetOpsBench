"""Filler runs: a grid job keeps its runs in flight with filler runs on the development starts until its last run on the wall
clock ends (CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch). The driver refuses a filler outside the development starts, with a judge or inside a grid
folder, and marks the run.json of a filler alone; no table and no pending list reads a filler; the filler arguments of every line
of the registered grids pass the driver's checks of the development starts. The runs in flight of a job count its loops of GEPA and
its sampling runs as well, so a loop and a sampling run on a wall-clock preset register at the number of record alone. With the tables of
the development starts computed beforehand, as the tables job computes them, a filler leaves the shared label and grade files byte-equal
and a registered start registers before and after it. The runs are the oracle on the development starts, which needs no model."""
import importlib.util
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online
from CausalDiagnosis.experiments.cluster_runs import grid_files, grid_job_lists as grid_lines, run_lines
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.table_definitions import training_variants as training
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import read_rows
from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade
from CausalDiagnosis.experiments.episode_runners.tests.conftest import own_table_copies

from CausalDiagnosis.experiments.run_verification.tests.test_gpu_card import JUDGE, JUDGE_URL, MODEL, URL, _endpoint, _of_record

LINE = "--config base --agent oracle"
DEVELOPMENT = ["--starts", "development", "--start-cap", "5"]  # enough starts to cover every faulty level of the rig


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """(run, filler): a run of LINE and a filler run of LINE on the development starts, in one folder of run folders."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables

    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        # the runs here may add a table that the session copies lack, after a run of another module recorded those copies
        own_table_copies(patch, tmp_path_factory.mktemp("tables"))
        precompute_tables.main(["--config", "base"])
        run = Path(run_episodes.main([*LINE.split(), *DEVELOPMENT, "--out", str(root)]))
        filler = Path(run_episodes.main([*LINE.split(), *DEVELOPMENT, "--filler", "--out", str(root)]))
    return run, filler


def test_a_filler_plays_the_development_starts_without_a_judge_outside_every_grid_folder(played, tmp_path):
    for extra, refusal in ((["--starts", "sampled"], run_episodes.FILLER_STARTS),
                           ([*DEVELOPMENT, "--judge", "local"], run_episodes.FILLER_JUDGE),
                           ([*DEVELOPMENT, "--out", str(tmp_path / "grid_7" / "fillers")], f"{run_episodes.FILLER_OUT} grid_7")):
        with pytest.raises(SystemExit) as stop:
            run_episodes.main([*LINE.split(), *extra, "--filler"])
        assert str(stop.value) == refusal
    run, filler = played
    meta = read_json(filler / "run.json")
    assert meta["filler"] is True and meta["stamp"] == pooling_rules.STAMP and meta["stamp_reasons"][-1] == run_episodes.FILLER_REASON
    assert "filler" not in read_json(run / "run.json") and run_episodes.FILLER_REASON not in read_json(run / "run.json")["stamp_reasons"]


def test_no_table_reads_a_filler_run(played):
    run, filler = played
    root = run.parent
    assert index_file.excluded(root) == [{"folder": filler.name, "reason": pooling_rules.FILLER_RUN}]
    assert [folder.name for folder in pooling_rules.folders_read(root)] == [run.name]


def test_a_filler_covers_no_line_of_a_grid_of_the_development_starts(played, tmp_path):
    """The filler plays its line on the development starts, so on a grid of those starts only its mark keeps the line pending."""
    for folder, pending in zip(played, ([], [LINE])):
        root = tmp_path / folder.name
        shutil.copytree(folder, root / folder.name)
        assert pending_lines.pending([LINE], [root], {}, starts="development", run_options=DEVELOPMENT[2:]) == pending


def test_the_filler_of_every_line_of_the_registered_grids_passes_the_checks_of_the_development_starts():
    """A line of a trained variant or of a sampling run would stop on the development starts as it stands, and its filler
    arguments drop what those starts refuse."""
    lines = [line for text in grid_files.files().values() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert any("--training-variant" in line.split() for line in lines) and any("--training-sample" in line.split() for line in lines)
    for line in lines:
        arguments = grid_lines.filler_arguments(line).split()
        assert not set(arguments) & set(grid_lines.FILLER_DROPPED), line
        a = run_episodes.build_parser().parse_args([*arguments, "--starts", "development", "--filler"])
        _name, config, _prices = run_episodes.configure(a)
        run_episodes.training_checks(a, config)


def test_the_command_fillers_prints_one_line_per_line_of_the_grid_file_in_its_order(tmp_path, capsys):
    grid = tmp_path / "grid.txt"
    grid.write_text("# a comment\n\n--config deterioration --agent causal --training-variant gepa --training-fold 2\n"
                    "--config real_time --agent oracle --replicate=3 --judge local\n", encoding="utf-8")
    assert grid_lines.main(["fillers", str(grid)]) == 0
    assert capsys.readouterr().out.splitlines() == ["--config deterioration --agent causal", "--config real_time --agent oracle"]


def test_the_filler_of_a_line_drops_its_start_state_and_its_scenarios():
    """A filler plays the development starts of its line, so its arguments drop the selectors of the line in either form of an
    option, with their values."""
    assert grid_lines.filler_arguments("--config base --agent causal --start-state 3/100/0/130@2 --no-diagnostics") == \
        "--config base --agent causal --no-diagnostics"
    assert grid_lines.filler_arguments("--config anchored --agent oracle --scenario-ids=2001/2003") == "--config anchored --agent oracle"


def test_the_filler_of_a_line_whose_preset_pins_a_start_plays_the_development_starts(tmp_path, monkeypatch):
    """A pin names the start of its line's own play: the line of a preset whose field start pins a start of the test set plays that
    start, and its filler, on the development starts, drops the pin and plays the development starts that its seeds draw, with no
    pin in its run.json. The runs are the oracle, which needs no model, on tables of their own, since they may add a table."""
    import json

    from CausalDiagnosis.datasets.splits.development_split import starts_of_test_set
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.utils.paths import preset_path

    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    own_table_copies(monkeypatch, tmp_path / "tables")
    _facts, _cycles, blocks = load_world()
    pinned = list(starts_of_test_set(run_episodes.get_profile("hyd_1"), blocks)[0])
    preset = tmp_path / "pinned.json"
    preset.write_text(json.dumps({**read_json(preset_path("base")), "name": "pinned", "start": {"combination": pinned, "entry_offset": 0}}),
                      encoding="utf-8")
    line = f"--config {preset} --agent oracle"
    own = Path(run_episodes.main([*line.split(), "--seeds", "1", "--out", str(tmp_path / "line")]))
    filler = Path(run_episodes.main([*grid_lines.filler_arguments(line).split(), *DEVELOPMENT, "--filler", "--out", str(tmp_path / "fillers")]))
    assert [json.loads(row["block"]) for row in read_rows(own)] == [pinned]
    assert read_json(own / "run.json")["config"]["start"] == {"combination": pinned, "entry_offset": 0}
    meta = read_json(filler / "run.json")
    development = {tuple(combination) for combination in meta["config"]["allowed"]}
    assert meta["filler"] is True and meta["config"]["start"] is None and tuple(pinned) not in development
    assert sorted(tuple(json.loads(row["block"])) for row in read_rows(filler)) == sorted(development)


def _runs_in_flight_of_record(monkeypatch, number):
    """Declares number runs in flight for every preset on the wall clock where registration() reads it, the name that run_episodes
    imports from design."""
    monkeypatch.setattr(run_episodes.start_check, "RUNS_IN_FLIGHT", number)


@pytest.mark.parametrize("kind", ["loop", "sampling"])
def test_a_loop_of_gepa_and_a_sampling_run_on_a_wall_clock_preset_register_at_the_runs_in_flight_of_record(kind, tmp_path, monkeypatch):
    """The runs in flight of a run are the processes that play episodes against its served model at once, the runs of grid lines,
    sampling runs and loops of GEPA alike, and its job hands that number down as GRID_PARALLEL, 1 to a run on the wall clock. A
    loop of GEPA and a sampling run of the fine tuning on the preset of training, whose clock charges wall seconds, are registered
    at the number of record in GRID_PARALLEL and carry the reason of the runs in flight at any other, as a run of a grid line
    does, with a made-up serving file, endpoint file and number of record."""
    monkeypatch.setattr(ledger.serving_endpoints, "endpoint_models", lambda url: [{"id": MODEL, "max_model_len": 32768}])
    sampler = _of_record(tmp_path, monkeypatch)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", sampler=sampler)))
    _runs_in_flight_of_record(monkeypatch, 4)
    if kind == "loop":
        a = gepa_online.driver_settings(gepa_online.build_parser().parse_args(
            ["--model", MODEL, "--base-url", URL, "--training-fold", str(training.held_out_folds.fold_ids()[0]), "--reflection-model", JUDGE,
             "--reflection-base-url", JUDGE_URL]))
    else:
        a = run_episodes.build_parser().parse_args([*run_lines.sampling()[0][1].split(), "--model", MODEL, "--base-url", URL, "--starts", "training"])
    name, config, _prices = run_episodes.configure(a)
    assert name == design.TRAINING_PRESET and config.clock_mode == "generation_time"
    for parallel in ("4", "1", None):
        if parallel is None:
            monkeypatch.delenv("GRID_PARALLEL", raising=False)
        else:
            monkeypatch.setenv("GRID_PARALLEL", parallel)
        reasons = run_episodes.start_check.registration(a, config, name, default_facts_path())
        played = None if parallel is None else int(parallel)
        assert reasons == ([] if parallel == "4" else [f"the preset {name} has a clock and plays {played} runs in flight, not the registered 4"]), parallel


def test_a_filler_beside_a_registered_start_leaves_the_shared_label_and_grade_files_byte_equal(tmp_path, monkeypatch):
    """The tables job computes the tables of every line of the grid files on their own start set and on the development starts
    before the platform record (grid_tables of CausalDiagnosis/cluster_jobs/env.example.sh), so a filler, which plays its line on the development starts,
    finds the label and the grade tables of its rules and writes nothing into the files that the runs of record read. One filler of
    a line, played without a start cap as a grid job plays it, leaves both files byte-equal, and a registered start, a loop of
    GEPA, registers before and after it. The two files are this test's own, with a made-up serving file, endpoint file and number
    of record."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")
    from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables

    tables = tmp_path / "tables"
    tables.mkdir()
    files = (tables / "labels.json", tables / "grades.json")
    monkeypatch.setattr(separability, "LABELS_PATH", files[0])
    monkeypatch.setattr(grade, "GRADES_PATH", files[1])
    grid = tmp_path / "grid_one.txt"
    grid.write_text(f"{LINE}\n", encoding="utf-8")
    for starts in ([], ["--starts", "development"]):
        precompute_tables.main(["--grid", str(grid), *starts])
    before = [path.read_bytes() for path in files]
    monkeypatch.setattr(ledger.serving_endpoints, "endpoint_models", lambda url: [{"id": MODEL, "max_model_len": 32768}])
    sampler = _of_record(tmp_path, monkeypatch)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", sampler=sampler)))
    _runs_in_flight_of_record(monkeypatch, 4)
    monkeypatch.setenv("GRID_PARALLEL", "4")
    a = gepa_online.driver_settings(gepa_online.build_parser().parse_args(
        ["--model", MODEL, "--base-url", URL, "--training-fold", str(training.held_out_folds.fold_ids()[0]), "--reflection-model", JUDGE,
         "--reflection-base-url", JUDGE_URL]))
    name, config, _prices = run_episodes.configure(a)
    assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == []
    filler = Path(run_episodes.main([*grid_lines.filler_arguments(LINE).split(), "--seeds", "1", "--starts", "development", "--filler",
                                     "--out", str(tmp_path / "fillers")]))
    assert read_json(filler / "run.json")["filler"] is True and len(read_rows(filler)) == read_json(filler / "run.json")["episodes_planned"]
    assert [path.read_bytes() for path in files] == before
    assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == []

