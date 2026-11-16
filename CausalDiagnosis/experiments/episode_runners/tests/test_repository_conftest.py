"""The guards of the repository's conftest.py, each shown on a small repository in pytest's temporary folder: a copy of that
conftest.py, the package CausalDiagnosis whose modules define the two table paths and the two functions of the asset
layer that the guard calls, the asset layer with a copy of its dataset_packages.py and one asset package that declares the two tables, and
one test file. A pytest session runs there as a child process, so the guards are tested as the
file that every session loads, and no file of the tree under test is written. A session passes when a test writes a table
through the path of its module, fails at its end when a test writes a default table in place, and stops before its first test
when the test process, or a process that a test starts, would import a package of the repository from another tree, or when the asset layer refuses
the data file. A session that runs names the data file at the head of its log."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.utils.paths import ROOT
PYTESTS_OWN_PATH = '[tool.pytest.ini_options]\npythonpath = ["src"]\n'
NO_PATH_OF_PYTEST = "[tool.pytest.ini_options]\n"
TABLE_MODULE = "from pathlib import Path\n\n{name} = Path(__file__).with_name({file!r})\n"
WRITES_THROUGH_THE_MODULE = ("from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade\n\n\n"
                             "def test_case():\n    grade.GRADES_PATH.write_text('written by a test')\n")
WRITES_IN_PLACE = ("from pathlib import Path\n\nfrom CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade\n\n\n"
                   "def test_case():\n    Path(grade.__file__).with_name('grades.json').write_text('written by a test')\n")
LEAVES_A_FILE = "from pathlib import Path\n\n\ndef test_case():\n    Path(__file__).with_name('played').write_text('')\n"
RECORDS_MODULE = "def data_refusal():\n    return {refusal!r}\n\n\ndef data_header():\n    return {header!r}\n"
DECLARES = "TABLES = {tables!r}\n"
WRITES_A_DECLARED_TABLE_IN_PLACE = ("from pathlib import Path\n\nfrom CausalDiagnosis.datasets.made_up_2 import tables\n\n\n"
                                    "def test_case():\n    Path(tables.__file__).with_name('table.json').write_text('written by a test')\n")


def asset_package(repository, asset_id, tables, files=()):
    """An asset folder of the small repository whose package declares the tables, with further modules of its own."""
    folder = repository / "CausalDiagnosis/datasets" / asset_id
    folder.mkdir(parents=True)
    (folder / "asset.json").write_text("{}")
    (folder / "__init__.py").write_text(DECLARES.format(tables=tuple(tables)))
    for name, text in files:
        (folder / name).write_text(text)
    return folder


@pytest.fixture
def small_repository(tmp_path):
    """A repository with the package CausalDiagnosis at its root, upstream's package servers in src and the conftest.py of this one.
    Beside it, in the folder other, lies a second package servers, as the source folder of another copy holds one."""
    repository = tmp_path.resolve() / "repository"
    for package in ("CausalDiagnosis/environment/reference_policies", "CausalDiagnosis/datasets/dataset_loader",
                    "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline", "src/servers"):
        parts = package.split("/")
        for depth in range(1 + (parts[0] == "src"), len(parts) + 1):
            (repository.joinpath(*parts[:depth])).mkdir(parents=True, exist_ok=True)
            (repository.joinpath(*parts[:depth]) / "__init__.py").write_text("")
    (repository / "CausalDiagnosis/environment/reference_policies/diagnosability_label.py").write_text(TABLE_MODULE.format(name="LABELS_PATH", file="labels.json"))
    (repository / "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/diagnosability_grade.py").write_text(TABLE_MODULE.format(name="GRADES_PATH", file="grades.json"))
    (repository / "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json").write_text("as computed")
    (repository / "CausalDiagnosis/datasets/dataset_loader/recordings.py").write_text(RECORDS_MODULE.format(refusal=None, header=[]))
    (repository / "CausalDiagnosis/datasets/dataset_loader/dataset_packages.py").write_bytes((ROOT / "CausalDiagnosis" / "datasets" / "dataset_loader" / "dataset_packages.py").read_bytes())
    asset_package(repository, "made_up_1", ("CausalDiagnosis.environment.reference_policies.diagnosability_label:LABELS_PATH", "CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade:GRADES_PATH"))
    (repository / "conftest.py").write_bytes((ROOT / "conftest.py").read_bytes())
    (repository / "pyproject.toml").write_text(PYTESTS_OWN_PATH)
    (repository.parent / "other/servers").mkdir(parents=True)
    (repository.parent / "other/servers/__init__.py").write_text("")
    return repository


def session(repository, test, *python_path):
    """A pytest session over one test file of the small repository with PYTHONPATH as given, as (exit status, output)."""
    (repository / "test_case.py").write_text(test)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(str(folder) for folder in python_path)}
    done = subprocess.run([sys.executable, "-m", "pytest", "test_case.py", "-q", "-p", "no:cacheprovider"], cwd=repository,
                          env=environment, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_a_session_passes_when_a_test_writes_a_table_through_the_path_of_its_module(small_repository):
    status, output = session(small_repository, WRITES_THROUGH_THE_MODULE, small_repository / "src")
    assert status == 0, output
    assert (small_repository / "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json").read_text() == "as computed"


def test_a_session_fails_at_its_end_when_a_test_writes_a_default_table_in_place(small_repository):
    status, output = session(small_repository, WRITES_IN_PLACE, small_repository / "src")
    assert status != 0 and "the session wrote the default tables in place" in output
    assert str(small_repository / "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json") in output


def test_a_table_that_an_asset_folder_declares_is_guarded_with_no_edit_of_the_conftest(small_repository):
    """A second asset package with a table of its own: a test that writes that table in place fails the session at its end,
    since the conftest guards every table that an asset package declares and names none itself."""
    asset_package(small_repository, "made_up_2", ("CausalDiagnosis.datasets.made_up_2.tables:TABLE_PATH",),
                  [("tables.py", TABLE_MODULE.format(name="TABLE_PATH", file="table.json"))])
    status, output = session(small_repository, WRITES_A_DECLARED_TABLE_IN_PLACE, small_repository / "src")
    assert status != 0 and "the session wrote the default tables in place" in output
    assert str(small_repository / "CausalDiagnosis/datasets/made_up_2/table.json") in output


def test_a_session_stops_before_its_first_test_when_a_process_it_starts_would_import_another_tree(small_repository):
    other = small_repository.parent / "other"
    status, output = session(small_repository, LEAVES_A_FILE, other)
    assert status == 4 and f"a process that a test starts would import servers from {other / 'servers'}" in output
    assert f"PYTHONPATH={small_repository}{os.pathsep}{small_repository / 'src'}" in output and not (small_repository / "played").exists()


def test_a_session_stops_before_its_first_test_when_the_test_process_would_import_another_tree(small_repository):
    other = small_repository.parent / "other"
    (small_repository / "pyproject.toml").write_text(NO_PATH_OF_PYTEST)
    status, output = session(small_repository, LEAVES_A_FILE, other, small_repository / "src")
    assert status == 4 and f"the test process would import servers from {other / 'servers'}" in output
    assert not (small_repository / "played").exists()


def test_a_session_stops_before_its_first_test_when_the_asset_layer_refuses_the_data(small_repository):
    refusal = "The data file of a_1 is not built: data/cycles.jsonl does not exist."
    (small_repository / "CausalDiagnosis/datasets/dataset_loader/recordings.py").write_text(RECORDS_MODULE.format(refusal=refusal, header=[]))
    status, output = session(small_repository, LEAVES_A_FILE, small_repository / "src")
    assert status == 4 and refusal in output and not (small_repository / "played").exists()


def test_the_head_of_a_session_names_the_data_file(small_repository):
    line = "data file of a_1: data/cycles.jsonl, SHA-256 " + "0" * 64
    (small_repository / "CausalDiagnosis/datasets/dataset_loader/recordings.py").write_text(RECORDS_MODULE.format(refusal=None, header=[line]))
    status, output = session(small_repository, LEAVES_A_FILE, small_repository / "src")
    assert status == 0 and line in output, output
