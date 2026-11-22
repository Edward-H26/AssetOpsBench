"""Session fixtures and the session guard of every test of the repository. The generated tables that the asset packages declare
(CausalDiagnosis.datasets.dataset_loader.dataset_packages.generated_tables), for the hydraulic rig models.json and grades.json of the CausalDiagnosis MCP and
labels.json of the environment, are read and written as copies in pytest's temporary folder, so that no test writes the files a run in the same tree
reads, and a session that changes a default file fails at its end. The variable that dataset_packages.TABLE_COPIES names maps
every default file to its copy for the processes that a test starts with the environment of the session, whose modules take
their table paths through dataset_packages.table_path. No
test reaches the real uv: AssetOpsBench's runners launch a server named by its entry point with `uv run <name>` from the
repository root, which syncs the repository's .venv to uv.lock, and the MCP client hands a server only PATH and a few other
variables, so a stub named uv first on PATH, which refuses every launch and records it, is the guard that reaches it. A session
stops before its first test when a package of the repository would come from another tree, in the test process or in a process that a test
starts: the Python environment of one copy of the repository, used for the tests of a second copy, imports the source of the
first in every process that PYTHONPATH does not point at the second. The probe of a new process inherits the environment
of the session, so a server that a test starts through the MCP client with its reduced environment is not covered. A session
also stops before its first test when the data file of an asset is missing or is not the pinned one, unless the clone states with
ASSETS_WITHOUT_DATA=1 that it holds no data, in which case the tests that need recordings are skipped. The head of the log names
the data file of every asset with its SHA-256."""
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
# one package per path entry, since every package of an entry comes from the same place: CausalDiagnosis at the root, upstream's packages in src
PACKAGES = {"CausalDiagnosis": ROOT, "servers": SOURCE}
PATH = os.pathsep.join(str(entry) for entry in dict.fromkeys(PACKAGES.values()))
# prints the folder a process would import each package from, or an empty line when it finds none; it imports nothing
PROBE = ("import importlib.util as u, sys\nfor name in sys.argv[1:]:\n    s = u.find_spec(name)\n"
         "    print(next(iter(s.submodule_search_locations)) if s else '')")


def package_folders(python=None):
    """The folder that holds each package for this process, or with an interpreter for a new process that inherits the
    environment of this one, as a server that a test starts does. None for a package the process does not find."""
    if python is None:
        specs = [importlib.util.find_spec(name) for name in PACKAGES]
        found = [next(iter(spec.submodule_search_locations)) if spec else "" for spec in specs]
    else:
        found = subprocess.run([python, "-c", PROBE, *PACKAGES], capture_output=True, text=True).stdout.split("\n")[:len(PACKAGES)]
    return {name: Path(folder).resolve() if folder else None for name, folder in zip(PACKAGES, found)}


def pytest_sessionstart(session):
    for process, folders in (("the test process", package_folders()), ("a process that a test starts", package_folders(sys.executable))):
        for name, folder in folders.items():
            if folder is not None and folder != (PACKAGES[name] / name).resolve():
                pytest.exit(f"{process} would import {name} from {folder}, not from the tree under test; set PYTHONPATH={PATH} "
                            "so that every process of the session imports this tree", returncode=4)
    from CausalDiagnosis.datasets.dataset_loader import recordings as records  # imported here, after the probe above has shown that the packages come from this tree

    refusal = records.data_refusal()
    if refusal:
        pytest.exit(refusal, returncode=4)
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    for line in records.data_header():
        reporter.write_line(line)


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


class TableGuard:
    """The session copies of the generated tables and, once a fixture made a run kept for the session that recorded tables, the md5 of those
    tables at that moment: a test that changes a recorded table is named, since every run made before it would replay against
    another table."""

    def __init__(self):
        self.copies, self.recorded = [], {}

    def record(self, paths=None):
        for path in self.copies if paths is None else paths:
            self.recorded[Path(path)] = _md5(Path(path))

    def changed(self):
        return [path for path, digest in self.recorded.items() if _md5(path) != digest]


_GUARD = TableGuard()


@pytest.fixture(scope="session", autouse=True)
def table_copies(tmp_path_factory):
    """Every generated table that an asset package declares, as (module, attribute) of its path, pointed at a copy in a folder
    of its module, so that two tables of one file name stay apart."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports

    tables = {(module, attribute): importlib.import_module(module) for module, attribute in ports.generated_tables()}
    defaults = {key: Path(getattr(module, key[1])) for key, module in tables.items()}
    before = {key: _md5(path) for key, path in defaults.items()}
    folder = tmp_path_factory.mktemp("tables")
    with pytest.MonkeyPatch.context() as patch:
        copies = {}
        for (name, attribute), module in tables.items():
            copy = folder / name / defaults[name, attribute].name
            copy.parent.mkdir(exist_ok=True)
            if defaults[name, attribute].exists():
                shutil.copyfile(defaults[name, attribute], copy)
            patch.setattr(module, attribute, copy)
            _GUARD.copies.append(copy)
            copies[str(defaults[name, attribute].resolve())] = str(copy)
        # a process that a test starts imports the modules afresh, without the attributes set above
        patch.setenv(ports.TABLE_COPIES, json.dumps(copies))
        yield folder
    changed = [str(path) for key, path in defaults.items() if _md5(path) != before[key]]
    assert not changed, f"the session wrote the default tables in place: {changed}"


@pytest.fixture(scope="session")
def table_guard(table_copies):
    """The guard of the generated tables; a fixture whose runs the session keeps calls table_guard.record() with the tables its runs
    read (test_replay_check.py, scripted, records tables of its own)."""
    return _GUARD


@pytest.fixture(autouse=True)
def shared_tables_unchanged(request, table_copies):
    """Fails the test that changed a session copy of a generated table after a session run recorded it, and takes the new state
    as recorded, so that only the test that changed it is named."""
    yield
    changed = _GUARD.changed()
    if changed:
        _GUARD.record(changed)
        pytest.fail(f"{request.node.nodeid} changed the shared generated table(s) {[str(p) for p in changed]} after a session run "
                    "recorded them; work on a copy of its own (see test_resume.py, runs)", pytrace=False)


@pytest.fixture(scope="session", autouse=True)
def uv_stub(tmp_path_factory):
    """The record of every uv command a test started, one line each with its working directory."""
    folder = tmp_path_factory.mktemp("uv_stub")
    record = folder / "calls.txt"
    record.touch()
    stub = folder / "uv"
    stub.write_text(f'#!/bin/sh\necho "$PWD $*" >> "{record}"\necho "uv is a stub in the tests" >&2\nexit 97\n')
    stub.chmod(0o755)
    path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{folder}{os.pathsep}{path}"
    yield record
    os.environ["PATH"] = path
