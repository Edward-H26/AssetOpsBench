"""The session copies of the generated tables stay as a session run recorded them."""
import os
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.utils.paths import ROOT
# the shortest order that showed the fault: the AssetOpsBench run of the session fixture scripted is made first and records the
# label table, test_resume's runs play a configuration without a table of its own, and the cached run is replayed
SEQUENCE = ("CausalDiagnosis/experiments/run_verification/tests/test_verify_results_folder.py::test_a_run_on_the_assetopsbench_servers_replays_through_its_servers_and_a_tampered_receipt_is_caught",
            "CausalDiagnosis/experiments/episode_runners/tests/test_resume.py::test_a_run_cut_at_the_boundary_of_an_episode_and_taken_up_equals_the_uninterrupted_run",
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
