"""gpu_sampler.py: the rows of a made-up sampler file placed on the epoch clock, the busy share and the energy of an episode from
the rows inside its span, and the sampler of a run read only when its copy is the one run.json recorded."""
import pytest

from CausalDiagnosis.performance import gpu_sampler
from CausalDiagnosis.performance.tests import made_up

ZONE = gpu_sampler.zone_of(made_up.STARTED)


def rows_of(tmp_path, rows):
    path = tmp_path / "gpu-77.csv"
    path.write_text(made_up.sampler_text(rows), encoding="utf-8")
    return gpu_sampler.read_rows(path, ZONE)


def test_the_local_times_of_the_sampler_are_placed_by_the_offset_of_the_started_line(tmp_path):
    rows, unread = rows_of(tmp_path, [(0.0, 50, 100.0), (5.0, 70, 110.0)])
    assert [row["t"] for row in rows] == [made_up.wall(0.0), made_up.wall(5.0)] and unread == 0
    later = gpu_sampler.read_rows(tmp_path / "gpu-77.csv", gpu_sampler.zone_of("2026-01-05T10:05:00+01:00"))[0]
    assert later[0]["t"] == made_up.wall(0.0) - 6 * 3600  # the control: another offset moves every row by its difference
    assert gpu_sampler.zone_of("2026-01-05T10:05:00") is None and gpu_sampler.zone_of(None) is None


def test_the_busy_share_and_the_energy_of_an_episode_read_the_rows_inside_its_span(tmp_path):
    rows, _unread = rows_of(tmp_path, [(-5.0, 0, 50.0), (0.0, 20, 100.0), (5.0, 40, 120.0), (10.0, 60, 140.0), (15.0, 80, 160.0),
                                       (20.0, 0, 50.0)])
    step = gpu_sampler.period(rows)
    figures = gpu_sampler.episode_figures(rows, made_up.wall(0.0), made_up.wall(16.5), step)
    assert step == 5.0 and figures["sampler_rows"] == 4
    assert figures["gpu_busy_share"] == pytest.approx(0.5) and figures["gpu_energy_j"] == pytest.approx(520 * 5.0)
    shifted = gpu_sampler.episode_figures(rows, made_up.wall(0.5), made_up.wall(17.0), step)
    assert shifted["sampler_rows"] == 3 and shifted["gpu_energy_j"] == pytest.approx(420 * 5.0)  # the control: the row at 0 is out


def test_a_value_the_driver_does_not_report_is_no_number_and_a_line_that_is_no_row_is_counted(tmp_path):
    path = tmp_path / "gpu-77.csv"
    path.write_text(made_up.sampler_text([(0.0, 20, None), (5.0, 40, 100.0)]) + "GPU_SAMPLER_FAILED made up\n", encoding="utf-8")
    rows, unread = gpu_sampler.read_rows(path, ZONE)
    figures = gpu_sampler.episode_figures(rows, made_up.wall(0.0), made_up.wall(5.0), 5.0)
    assert unread == 1 and rows[0]["power"] is None
    assert figures["gpu_energy_j"] == pytest.approx(500.0) and figures["gpu_busy_share"] == pytest.approx(0.3)


def test_the_period_is_the_median_gap_between_rows(tmp_path):
    rows, _unread = rows_of(tmp_path, [(0.0, 1, 1.0), (5.0, 1, 1.0), (15.0, 1, 1.0), (20.0, 1, 1.0)])
    assert gpu_sampler.period(rows) == 5.0 and gpu_sampler.period(rows[:1]) is None


def test_the_sampler_of_a_run_is_read_only_when_its_copy_begins_with_the_recorded_bytes(tmp_path):
    grid = made_up.write_grid(tmp_path / "grid_77", "77", sampler=made_up.sampler_text([(0.0, 50, 100.0), (5.0, 50, 100.0)]))
    meta = made_up.model_meta()
    meta["serving"]["agent"]["sampler"] = made_up.recorded_sampler(grid / "job_logs" / "gpu-77.csv")
    run = made_up.write_run(grid / "made_up_run", [(7, made_up.known_episode())], meta=meta)
    rows, step, note = gpu_sampler.run_sampler(run, meta)
    assert len(rows) == 2 and step == 5.0 and note == ""
    meta["serving"]["agent"]["sampler"]["sha256"] = "0" * 64
    rows, step, note = gpu_sampler.run_sampler(run, meta)
    assert rows == [] and note == "gpu-77.csv does not begin with the bytes whose SHA-256 run.json records"
    (grid / "job_logs" / "gpu-77.csv").unlink()
    assert gpu_sampler.run_sampler(run, meta)[2] == "job_logs holds no gpu-77.csv; job_record.sh copies it after the job"


def test_the_note_of_a_sampler_counts_the_lines_that_are_no_row(tmp_path):
    """Two lines of the copy are no row beside its three rows: the note counts the two; the control is the copy without them,
    whose note is empty."""
    text = made_up.sampler_text([(0.0, 50, 100.0), (5.0, 50, 100.0), (10.0, 50, 100.0)])
    grid = made_up.write_grid(tmp_path / "grid_77", "77", sampler=text + "GPU_SAMPLER_FAILED made up\nnot a row\n")
    meta = made_up.model_meta()
    meta["serving"]["agent"]["sampler"] = made_up.recorded_sampler(grid / "job_logs" / "gpu-77.csv")
    run = made_up.write_run(grid / "made_up_run", [(7, made_up.known_episode())], meta=meta)
    rows, _step, note = gpu_sampler.run_sampler(run, meta)
    assert len(rows) == 3 and note == "2 lines of gpu-77.csv are no row"
    (grid / "job_logs" / "gpu-77.csv").write_text(text, encoding="utf-8")
    meta["serving"]["agent"]["sampler"] = made_up.recorded_sampler(grid / "job_logs" / "gpu-77.csv")
    assert gpu_sampler.run_sampler(run, meta)[2] == ""  # the control


def test_without_the_started_line_or_outside_a_grid_folder_the_sampler_is_not_read(tmp_path):
    grid = made_up.write_grid(tmp_path / "grid_77", "77", sampler=made_up.sampler_text([(0.0, 50, 100.0)]))
    meta = made_up.model_meta()
    meta["serving"]["agent"]["sampler"] = made_up.recorded_sampler(grid / "job_logs" / "gpu-77.csv")
    run = made_up.write_run(grid / "made_up_run", [(7, made_up.known_episode())], meta=meta, started=None)
    assert gpu_sampler.run_sampler(run, meta)[2] == ("the ledger holds no started line of the endpoint file, which would place the local "
                                                     "times of the sampler")
    unrecorded = made_up.model_meta()
    assert gpu_sampler.run_sampler(run, unrecorded)[2] == "run.json records no sampler of a model endpoint"
    alone = made_up.write_run(tmp_path / "alone", [(7, made_up.known_episode())], meta=meta)
    assert gpu_sampler.run_sampler(alone, meta)[2] == "the run lies in no grid folder of a cluster job"
