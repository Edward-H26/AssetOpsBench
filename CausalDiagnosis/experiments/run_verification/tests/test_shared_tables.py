"""The session copies of the generated tables stay as a session run recorded them."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.utils.paths import ROOT
from CausalDiagnosis.experiments.run_verification.tests.test_replay_check import scripted  # noqa: F401  the scripted runs of the session
# the shortest orders that showed the fault: the AssetOpsBench run of the fixture scripted is made first and records the
# label table, test_resume's runs play a configuration without a table of its own, test_fillers' runs add the labels of the
# development starts of the preset base, and the cached run is replayed
SEQUENCE = ("CausalDiagnosis/experiments/run_verification/tests/test_verify_results_folder.py::test_a_run_on_the_assetopsbench_servers_replays_through_its_servers_and_a_tampered_receipt_is_caught",
            "CausalDiagnosis/experiments/episode_runners/tests/test_resume.py::test_a_run_cut_at_the_boundary_of_an_episode_and_taken_up_equals_the_uninterrupted_run",
            "CausalDiagnosis/experiments/episode_runners/tests/test_fillers.py::test_a_filler_plays_the_development_starts_without_a_judge_outside_every_grid_folder",
            "CausalDiagnosis/experiments/episode_runners/tests/test_fillers.py::test_the_filler_of_a_line_whose_preset_pins_a_start_plays_the_development_starts",
            "CausalDiagnosis/experiments/run_verification/tests/test_replay_check.py::test_a_scripted_run_replays_through_the_servers_on_both_surfaces")


def test_a_run_of_the_session_replays_after_a_module_that_adds_a_label_table():
    done = subprocess.run([sys.executable, "-m", "pytest", *SEQUENCE, "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "--tb=line"],
                          cwd=ROOT, capture_output=True, text=True, env=os.environ.copy(), timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]


def test_the_guard_names_a_shared_table_changed_after_a_run_recorded_it(table_guard, tmp_path):
    table = tmp_path / "labels.json"
    table.write_text("{}", encoding="utf-8")
    table_guard.record([table])
    try:
        assert table_guard.changed() == []
        table.write_text('{"tables": {}}', encoding="utf-8")
        assert table_guard.changed() == [table]
    finally:
        del table_guard.recorded[table]


def test_own_table_copies_point_the_test_process_and_the_processes_it_starts_at_copies_of_their_own(tmp_path, monkeypatch):
    """own_table_copies, which the runs that may add a table use (test_resume.py, test_fillers.py) and the recorded runs of
    test_replay_check.py, points every generated table at a copy of its own, taken from the session copy, in the test process and,
    through the variable of the copies, in a process that it starts."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.test_gymnasium_wrapper import child_tables
    from CausalDiagnosis.experiments.episode_runners.tests.conftest import own_table_copies

    def paths():
        return [Path(getattr(importlib.import_module(module), attribute)) for module, attribute in ports.generated_tables()]

    shared = paths()
    own_table_copies(monkeypatch, tmp_path)
    own = paths()
    assert all(path.is_relative_to(tmp_path) for path in own) and child_tables(os.environ) == own
    assert [path.read_bytes() if path.exists() else None for path in own] == [path.read_bytes() if path.exists() else None for path in shared]


def test_a_test_of_another_module_that_takes_scripted_reads_and_writes_the_tables_of_replay_tables(scripted, replay_tables):
    """scripted, a fixture of test_replay_check.py, points every generated table at replay_tables in each test that takes it, in the
    test process and in the processes it starts, so that a run made in one module is replayed in another on the tables it recorded,
    and the guard records those tables whichever module makes a run first."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.test_gymnasium_wrapper import child_tables

    tables = [Path(getattr(importlib.import_module(module), attribute)) for module, attribute in ports.generated_tables()]
    assert tables and all(path.is_relative_to(replay_tables) for path in tables) and child_tables(os.environ) == tables
