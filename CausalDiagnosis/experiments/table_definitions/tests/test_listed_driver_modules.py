"""Every module the driver loads lies in a list of experiment_tables.py (else refuse_unlisted stops the run before run.json),
and no module of the servers writes a file when it is imported."""
import os
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.experiments.run_verification import run_identity_file_lists

from CausalDiagnosis.utils.paths import ROOT


def _fresh(code):
    """Run code in a fresh isolated process with the root and src on the path, as the cluster does."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


def test_every_module_of_the_driver_folder_lies_in_a_list():
    folders = [d for d in sorted((ROOT / "CausalDiagnosis" / "experiments").iterdir()) if d.is_dir() and d.name not in ("tests", "__pycache__")]
    unlisted = [p.relative_to(ROOT).as_posix() for folder in folders for p in sorted(folder.glob("*.py"))
                if not run_identity_file_lists.lists_of(p.relative_to(ROOT).as_posix())]
    assert unlisted == []


def test_the_driver_process_loads_no_file_outside_the_lists():
    done = _fresh("from CausalDiagnosis.experiments.episode_runners import run_episodes, wall_cut, cache_flush; "
                   "from CausalDiagnosis.agents.utils import idle_hook; from CausalDiagnosis.utils import model_request_retry; "
                   "print(run_episodes.loaded_outside_lists())")
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_importing_a_server_module_writes_no_file_into_the_package():
    before = {p for p in (ROOT / "CausalDiagnosis" / "mcp_servers").rglob("*") if p.is_file()}
    done = _fresh("import CausalDiagnosis.mcp_servers.Environment_Actions_MCP.launcher, CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.launcher")
    after = {p for p in (ROOT / "CausalDiagnosis" / "mcp_servers").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert done.returncode == 0, done.stderr[-3000:]
    assert {p for p in after - before} == set()
