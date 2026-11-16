"""The fit, grade and calibration command lines on another asset than the default (CausalDiagnosis/docs/adding-a-dataset.md, step 4),
here the tiny asset of the asset layer: each command takes --asset-id, reads the recordings that the profile names and
writes only the files it is given, and a command of another asset refuses the files of the default asset. No value of a
fit, a grade or the report is checked."""
import json

import pytest

from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data, write_recordings  # noqa: F401
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import calibration_report as calibration, fit_models as fit, diagnosability_grade as grade
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import asset_view, development_blocks, recorded_blocks


@pytest.fixture
def tiny_cycles(tmp_path, tiny, monkeypatch):
    """The tiny recordings, named by the environment variable of the tiny profile as a porting user names theirs."""
    path = write_recordings(tmp_path / "tiny_cycles.jsonl", tiny)
    monkeypatch.setenv(tiny.records["data_env"], str(path))
    return path


def test_the_three_command_lines_fit_grade_and_report_the_tiny_asset(tmp_path, tiny, tiny_cycles):
    models, grades, reports = tmp_path / "models.json", tmp_path / "grades.json", tmp_path / "reports"
    fit.main(["--asset-id", tiny.asset_id, "--resamples", "20", "--out", str(models)])
    assert json.loads(models.read_text(encoding="utf-8"))["space"]["components"] == list(tiny.components)
    grade.main(["--asset-id", tiny.asset_id, "--models", str(models), "--rollouts", "1", "--samples", "8", "--out", str(grades)])
    table = next(iter(json.loads(grades.read_text(encoding="utf-8"))["tables"].values()))
    assert len(table["grades"]) == len(tiny.combinations())
    calibration.main(["--asset-id", tiny.asset_id, "--models", str(models), "--readings", "2", "--rollouts", "1", "--out", str(reports)])
    report, = reports.glob("calibration_*.md")
    view = asset_view(tiny.asset_id)
    development = calibration.starts_of(view, recorded_blocks(view, tiny_cycles)[0], "development")
    assert f"the {len(development)} combinations of the start set development" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("command, given, kept", [
    (fit.main, [], "--out"),
    (grade.main, ["--out", "grades.json"], "--models"),
    (calibration.main, ["--models", "models.json"], "--out"),
], ids=("fit", "grade", "calibration"))
def test_another_asset_refuses_the_files_of_the_default_asset(tiny, capsys, command, given, kept):
    with pytest.raises(SystemExit):
        command(["--asset-id", tiny.asset_id, *given])
    assert f"needs its own {kept}," in capsys.readouterr().err


def test_ensure_models_fits_the_asset_it_is_asked_for(tmp_path, tiny, tiny_cycles):
    """ensure_models fits the models file of the asset it is told, on the data file that the profile of that asset names, and
    leaves the file as it is once it is current, so that a port keeps a models file of its own."""
    path = tmp_path / "models.json"
    fit.ensure_models(path, asset_id=tiny.asset_id)
    first = path.read_bytes()
    stored = json.loads(first)
    assert stored["space"]["components"] == list(tiny.components) and stored["facts_sha256"] == fit.facts_sha256(tiny_cycles)
    fit.ensure_models(path, asset_id=tiny.asset_id)
    assert path.read_bytes() == first


def test_the_calibration_report_reads_the_development_starts_unless_asked_for_all(tmp_path, tiny, tiny_cycles, monkeypatch):
    """The calibration report scores the development blocks of the asset by default and every recorded combination with
    --starts all, so a report rests on no combination of the test set unless it is asked to."""
    view = asset_view(tiny.asset_id)
    blocks, _records = recorded_blocks(view, tiny_cycles)
    development = set(development_blocks(view, blocks))
    assert development and set(blocks) - development
    assert calibration.starts_of(view, blocks, "development") == [h for h in view.combinations if h in development]
    assert calibration.starts_of(view, blocks, "all") == [h for h in view.combinations if h in blocks]
    models = tmp_path / "models.json"
    fit.main(["--asset-id", tiny.asset_id, "--resamples", "20", "--out", str(models)])
    scored, evaluate = [], calibration.evaluate
    monkeypatch.setattr(calibration, "evaluate", lambda *args, starts=None, **options: scored.append(starts) or evaluate(*args, starts=starts, **options))
    for starts in ([], ["--starts", "all"]):
        out = tmp_path / ("all" if starts else "default")
        calibration.main(["--asset-id", tiny.asset_id, "--models", str(models), *starts, "--readings", "2", "--rollouts", "1", "--out", str(out)])
    assert scored == [development, set(blocks)]
