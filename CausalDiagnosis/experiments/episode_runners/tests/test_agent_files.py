"""The files a run loads (CausalDiagnosis/experiments/episode_runners/run_episodes/ over the lists of run_identity_file_lists.py): the modules through which the
driver reaches each agent load only files of the list of that agent or of the harness list, the driver alone, all that a run of
a reference policy or of static loads, loads no file of an agent list, a run whose driver process loaded a file of the roots
outside every list stops before its run.json, and a run that ends well records the listed files it loaded. So a change of the
files of one agent never changes the code identity of a run of another agent or of a reference policy. Run folders go under
pytest's tmp_path, outside the repository."""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.run_verification import run_identity_file_lists
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.experiments.table_definitions.tests.test_code_lists import _fresh

needs_a_run = pytest.mark.skipif(not Path(default_facts_path()).exists() or importlib.util.find_spec("mcp") is None,
                                 reason="a run needs the data file of the rig and the mcp package")
# the folders of agent code in the tree: the causal agent, the agents of the framework, and upstream's agents and model backends
AGENT_FOLDERS = ("CausalDiagnosis/agents/", "CausalDiagnosis/frameworks/AssetOpsBench/agents/", "src/agent/", "src/llm/")
DRIVER = "CausalDiagnosis.experiments.episode_runners.run_episodes"


def _loaded_by(modules):
    """The files under the root that a fresh interpreter loads when it imports the modules, as paths relative to the root."""
    code = ("import importlib, json, sys\nfrom pathlib import Path\nroot = Path.cwd().resolve()\n"
            f"for name in {tuple(modules)!r}:\n    importlib.import_module(name)\n"
            "print(json.dumps(sorted({Path(m.__file__).resolve().relative_to(root).as_posix() for m in list(sys.modules.values())\n"
            "                         if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(root)})))\n")
    return json.loads(_fresh(code))


@pytest.mark.parametrize("agent", sorted(run_episodes.AGENT_ENTRY_MODULES))
def test_the_modules_an_agent_loads_lie_in_its_declared_list_or_in_the_harness_list(agent):
    """Every runner with code of its own, the scripted check among them, loads agent code of its own list or of the harness list
    alone, so it holds no file of the list of another agent."""
    agent_code = [path for path in _loaded_by(run_episodes.AGENT_ENTRY_MODULES[agent]) if path.startswith(AGENT_FOLDERS)]
    allowed = {f"{run_identity_file_lists.AGENT_LIST}{agent}", "harness"}
    assert agent_code and {path: run_identity_file_lists.lists_of(path) for path in agent_code if not set(run_identity_file_lists.lists_of(path)) & allowed} == {}


def test_the_driver_alone_loads_no_file_of_an_agent_list():
    """What the driver imports for every run, the whole code of a run of a reference policy or of static, holds no file of an
    agent list."""
    loaded = _loaded_by([DRIVER])
    assert DRIVER.replace(".", "/") + "/__init__.py" in loaded
    assert {path: run_identity_file_lists.lists_of(path) for path in loaded
            if any(name.startswith(run_identity_file_lists.AGENT_LIST) for name in run_identity_file_lists.lists_of(path))} == {}


def test_the_rule_names_a_loaded_file_of_the_roots_that_lies_in_no_list(tmp_path, monkeypatch):
    for k, path in enumerate(("CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py", "CausalDiagnosis/experiments/stray_module.py", "docs/elsewhere.py")):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("\n", encoding="utf-8")
        monkeypatch.setitem(sys.modules, f"probe_of_the_rule_{k}", SimpleNamespace(__file__=str(tmp_path / path)))
    assert run_episodes.run_record.loaded_outside_lists(tmp_path) == ["CausalDiagnosis/experiments/stray_module.py"]
    assert run_episodes.loaded_files(tmp_path) == ["CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py"]


@needs_a_run
def test_a_run_whose_process_loaded_a_file_outside_every_list_stops_without_run_json(tmp_path, monkeypatch):
    stray = "CausalDiagnosis/experiments/stray_module.py"
    monkeypatch.setattr(run_episodes.run_record, "loaded_outside_lists", lambda root=ROOT: [stray])
    with pytest.raises(SystemExit, match=re.escape(stray)):
        run_episodes.main(["--config", "static", "--agent", "scripted", "--seeds", "1", "--out", str(tmp_path / "stopped")])
    assert list((tmp_path / "stopped").rglob("ledger.json")) and not list((tmp_path / "stopped").rglob("run.json"))


@needs_a_run
def test_a_run_that_ends_well_records_the_listed_files_its_driver_process_loaded(tmp_path):
    """A fresh process, as on the cluster, since the process of the tests holds the modules of every test before it."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-m", "CausalDiagnosis.experiments.episode_runners.run_episodes", "--config", "static",
                           "--agent", "scripted", "--seeds", "1", "--out", str(tmp_path / "kept")],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    (run_dir,) = [p for p in (tmp_path / "kept").iterdir() if p.is_dir()]
    loaded = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["loaded_files"]
    assert loaded == sorted(loaded) and all(run_identity_file_lists.lists_of(path) for path in loaded)
    assert {"CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py", "src/agent/plan_execute/runner.py", "CausalDiagnosis/mcp_servers/utils/transcript.py"} <= set(loaded)
    # the code of the agent as tool port, which the driver imports only for that agent, stays out of a scripted run
    assert not [path for path in loaded if path.startswith("CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/")]
