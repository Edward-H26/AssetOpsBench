"""The file lists of a run (design.CODE_LISTS, AGENT_LISTS) and its identity (run_ledger/): every file of the tree
under the roots lies in one list or in agent lists only, the lists that an episode loads hold what it loads, the identity moves
with the files of a run's own agent, and the tables of record are built without any module of the list rendering."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.experiments.run_verification import run_identity_file_lists
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT

NOT_LOADED = ("scoring", "analysis", "training", "upstream", "rendering", "checks", "documents", "reproduction")


def _fresh(code, cwd=ROOT):
    """The standard output of a fresh interpreter that runs code with the fork's src and drivers folders first on the path."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(cwd)), str(Path(cwd) / "src")]),
           "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return done.stdout


def _fork_files(paths):
    """The files of the fork among the files under the root that a fresh interpreter loaded: the files of the interpreter's own
    environment outside the listed roots are left out, since the documented setup puts the .venv inside the root."""
    environment = Path(sys.prefix).resolve()
    return [path for path in paths if (ROOT / path).is_file()
            and (run_identity_file_lists.in_roots(path) or not (ROOT / path).resolve().is_relative_to(environment))]


def test_the_files_of_a_port_take_their_lists_by_rules_that_name_no_asset():
    """A port keeps everything in its folder under CausalDiagnosis/datasets: its data script by the name hyd_1 uses lies in
    harness with hyd_1's, its package modules and presets in episode, its tests in checks, its runs under results/runs in no
    list, and no rule of a list names the asset."""
    folder = "CausalDiagnosis/datasets/made_up_1/"
    assert run_identity_file_lists.lists_of(folder + "download_and_build.py") == run_identity_file_lists.lists_of("CausalDiagnosis/datasets/hyd_1/download_and_build.py") == ("harness",)
    assert run_identity_file_lists.lists_of(folder + "source.py") == run_identity_file_lists.lists_of(folder + "presets/made_up_1_layer_0.json") == ("episode",)
    assert run_identity_file_lists.lists_of(folder + "tests/test_made_up_fetch.py") == ("checks",) and run_identity_file_lists.lists_of(folder + "README.md") == ("documents",)
    assert run_identity_file_lists.lists_of("CausalDiagnosis/results/runs/made_up_1/run_1/run.json") == ()
    rules = [rule for listed in (*run_identity_file_lists.CODE_LISTS.values(), *run_identity_file_lists.AGENT_LISTS.values(), run_identity_file_lists.EXCLUDED, run_identity_file_lists.UPSTREAM)
             for rule in listed]
    assert not any("made_up_1" in rule or "hyd_1" in rule for rule in rules)


def test_every_file_under_the_roots_lies_in_exactly_one_list_or_in_agent_lists_only():
    files = ledger.code_identity.tree_files()
    assert files.get("", {}) == {}, sorted(files.get("", {}))
    homes = {}
    for name, listed in files.items():
        for path in listed:
            homes.setdefault(path, []).append(name)
    for path, names in homes.items():
        agents = [n for n in names if n.startswith(run_identity_file_lists.AGENT_LIST)]
        assert (len(names) == 1 and not agents) or names == agents, (path, names)
    # the four roots, the acceptance of the ports among them
    assert run_identity_file_lists.lists_of("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/acceptance_check.py") == ("reproduction",) and run_identity_file_lists.role_of("reproduction", "causal") == "bound"
    assert run_identity_file_lists.lists_of("uv.lock") == ("harness",) and run_identity_file_lists.lists_of("CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch") == ("cluster",)
    assert run_identity_file_lists.role_of("cluster", "causal") == "bound" and "cluster" not in ledger.identity_lists("causal")
    assert run_identity_file_lists.lists_of("CausalDiagnosis/experiments/table_definitions/serving_of_record.json") == ("analysis",)
    assert run_identity_file_lists.lists_of("CausalDiagnosis/results/tables/AGENTS.csv") == () and run_identity_file_lists.lists_of("CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json") == ()


def test_no_module_of_scoring_analysis_training_or_upstream_is_loaded_by_an_episode():
    """A fresh interpreter imports the two task servers, the MCP's main module, the environment and the driver, and every file of
    the fork it loaded lies in a list that an episode may load: episode, harness or an agent list."""
    code = ("import importlib, json, sys\nfrom pathlib import Path\nroot = Path.cwd().resolve()\n"
            "for name in ('CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main', 'CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server', 'CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main', 'CausalDiagnosis.environment.Causal_Diagnosis_Environment.env', 'CausalDiagnosis.experiments.episode_runners.run_episodes'):\n"
            "    importlib.import_module(name)\n"
            "print(json.dumps(sorted({Path(m.__file__).resolve().relative_to(root).as_posix() for m in list(sys.modules.values())\n"
            "                         if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(root)})))\n")
    loaded = _fork_files(json.loads(_fresh(code)))
    assert len(loaded) > 50
    wrong = {path: run_identity_file_lists.lists_of(path) for path in loaded if set(run_identity_file_lists.lists_of(path)) & set(NOT_LOADED) or not run_identity_file_lists.lists_of(path)}
    assert wrong == {}


def _small_tree(root):
    """A tree of one file per kind of list under the roots, enough for tree_files, the identity and the freeze check."""
    for path, text in (("CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py", "harness\n"), ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py", "episode\n"), ("CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/episode_judge.py", "scoring\n"),
                       ("CausalDiagnosis/evaluation/result_tables/tables_from_runs/__init__.py", "analysis\n"), ("CausalDiagnosis/evaluation/result_tables/paper_tables/__init__.py", "rendering\n"), ("CausalDiagnosis/agents/Causal_Diagnosis_Agent/controller.py", "causal\n"),
                       ("CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/supervisor_runner.py", "agent as tool\n"), ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/acceptance_check.py", "STEP_TOLERANCE = 1\n"),
                       ("CausalDiagnosis/experiments/tests/test_x.py", "check\n"), ("src/benchmark/other.py", "upstream\n"), ("uv.lock", "lock\n")):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


def test_the_identity_moves_with_a_file_of_the_run_s_own_agent_and_not_with_another_agent_s(tmp_path):
    root = _small_tree(tmp_path)
    before = {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in ("causal", "agent_as_tool", "oracle")}
    (root / "CausalDiagnosis/agents/Causal_Diagnosis_Agent/controller.py").write_text("causal, changed\n", encoding="utf-8")
    after = {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in before}
    assert after["causal"] != before["causal"] and after["agent_as_tool"] == before["agent_as_tool"] and after["oracle"] == before["oracle"]
    (root / "CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py").write_text("episode, changed\n", encoding="utf-8")
    moved = {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in before}
    assert all(moved[agent] != after[agent] for agent in before)  # an episode file is in every identity
    (root / ("CausalDiagnosis/evaluation/result_tables/paper_tables/__init__.py")).write_text("rendering, changed\n", encoding="utf-8")
    assert {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in before} == moved  # a free file is in none


def test_a_made_up_asset_adds_its_files_by_rules_and_no_rule_names_an_asset():
    rules = [rule for listed in (*run_identity_file_lists.CODE_LISTS.values(), *run_identity_file_lists.AGENT_LISTS.values(), run_identity_file_lists.EXCLUDED, run_identity_file_lists.UPSTREAM) for rule in listed]
    assert not any("hyd_1" in rule for rule in rules)
    assert run_identity_file_lists.lists_of("CausalDiagnosis/datasets/tiny/asset.json") == ("episode",) and run_identity_file_lists.lists_of("CausalDiagnosis/datasets/tiny/records.py") == ("episode",)


def test_each_agent_package_has_its_own_list_and_the_shared_files_of_the_agents_lie_in_harness():
    """The list of the present agent holds its package alone and the package of the general agent has a list of its own, so a file
    added under one package never joins the identity of the runs of the other; the shared files of the declaration and of the
    answers lie in harness, the rules of a kind in episode."""
    files = ledger.code_identity.tree_files()
    present = "CausalDiagnosis/agents/Causal_Diagnosis_Agent/"
    assert files["agent:causal"] and all(path.startswith(present) for path in files["agent:causal"])
    assert sorted(path for path in files["agent:causal_next"]) == sorted(
        path for name, listed in files.items() for path in listed if path.startswith("CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/")
        and name != "checks")
    for path, listed in (("CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/planted.py", ("agent:causal_next",)),
                         ("CausalDiagnosis/agents/Causal_Diagnosis_Agent/planted.py", ("agent:causal",)),
                         ("CausalDiagnosis/agents/utils/declaration.py", ("harness",)), ("CausalDiagnosis/agents/utils/answer.py", ("harness",)),
                         ("CausalDiagnosis/agents/utils/answer_forms.py", ("harness",)),
                         ("CausalDiagnosis/utils/public_rules/plan_rules.py", ("episode",))):
        assert run_identity_file_lists.lists_of(path) == listed, path


def test_a_file_of_the_general_agent_moves_the_identity_of_its_runs_and_not_that_of_the_present_agent(tmp_path):
    root = _small_tree(tmp_path)
    planted = root / "CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/controller.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("general\n", encoding="utf-8")
    agents = ("causal", "causal_next", "agent_as_tool")
    before = {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in agents}
    planted.write_text("general, changed\n", encoding="utf-8")
    after = {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in agents}
    assert after["causal_next"] != before["causal_next"] and after["causal"] == before["causal"] and after["agent_as_tool"] == before["agent_as_tool"]


def test_every_list_has_one_role_and_an_agent_list_is_identity_for_its_own_run_alone():
    assert set(run_identity_file_lists.LIST_ROLES) == {*run_identity_file_lists.CODE_LISTS, "upstream", run_identity_file_lists.FRAMEWORK_LIST} and set(run_identity_file_lists.LIST_ROLES.values()) <= {"identity", "bound", "free"}
    for agent in run_identity_file_lists.AGENT_LISTS:
        assert run_identity_file_lists.role_of(f"{run_identity_file_lists.AGENT_LIST}{agent}", agent) == "identity"
        assert {run_identity_file_lists.role_of(f"{run_identity_file_lists.AGENT_LIST}{other}", agent) for other in run_identity_file_lists.AGENT_LISTS if other != agent} == {"bound"}
    assert ledger.identity_lists("oracle") == ["scoring", "harness", "episode"]


def test_the_tables_of_record_are_built_with_every_rendering_module_blocked(tmp_path):
    """The tables of record are the CSV files of tables_from_runs/. A fresh interpreter whose finder refuses every module of the list
    rendering builds them from a small root and they equal a normal build, so no free file can move a registered number. Both builds
    read the copy of registered_analysis.json that the tests read, so that they differ in the rendering modules alone."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _row, _run

    runs = tmp_path / "runs"
    _run(runs, "pe", [_row(s) for s in range(4)])
    blocked = sorted(Path(path).stem for path in run_identity_file_lists.CODE_LISTS["rendering"] if path.endswith(".py"))
    code = ("import importlib.abc, sys\n"
            f"BLOCKED = {blocked!r}\n"
            "class Refuse(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[-1] in BLOCKED:\n"
            "            raise ImportError(f'the list rendering is blocked: {name}')\n"
            "sys.meta_path.insert(0, Refuse())\n"
            f"import pathlib\nfrom CausalDiagnosis.experiments.table_definitions import registered_analysis as registered\nregistered.PATH = pathlib.Path({str(registered.PATH)!r})\n"
            "from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate\n"
            f"aggregate.aggregate({str(runs)!r}, {str(tmp_path / 'blocked')!r}, resamples=50, pages=False, fraction='1')\n")
    _fresh(code)
    from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate

    aggregate.aggregate(runs, tmp_path / "normal", resamples=50, pages=False, fraction="1")
    tables = sorted(p.name for p in (tmp_path / "normal").glob("*.csv"))
    assert tables and all((tmp_path / "blocked" / name).read_bytes() == (tmp_path / "normal" / name).read_bytes() for name in tables)


def test_a_plan_execute_run_names_the_port_and_the_sub_agents_it_shares_with_agent_as_tool(tmp_path):
    """--agent plan_execute plays the paper-era port, whose steps run on the sub-agents of the Agent-As-Tool port over the vendored
    engine. A fresh interpreter that imports the port loads files of the harness, of episode (the tool servers of AssetOpsBench),
    both identity lists of every run, and of the list of plan_execute alone, and the identity of a plan_execute run moves with a
    file of the port and with a shared file, which moves that of agent_as_tool too."""
    code = ("import importlib, json, sys\nfrom pathlib import Path\nroot = Path.cwd().resolve()\nimportlib.import_module('CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper')\n"
            "print(json.dumps(sorted({Path(m.__file__).resolve().relative_to(root).as_posix() for m in list(sys.modules.values())\n"
            "                         if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(root)})))\n")
    loaded = _fork_files(json.loads(_fresh(code)))
    own = f"{run_identity_file_lists.AGENT_LIST}plan_execute"
    assert any(path.startswith("CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/") for path in loaded) and any(path.startswith("CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/") for path in loaded)
    assert {path: run_identity_file_lists.lists_of(path) for path in loaded if run_identity_file_lists.lists_of(path) not in (("harness",), ("episode",)) and own not in run_identity_file_lists.lists_of(path)} == {}
    root = _small_tree(tmp_path)
    port = root / "CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/runner.py"
    port.parent.mkdir(parents=True)
    port.write_text("plan execute\n", encoding="utf-8")

    def identities():
        return {agent: ledger.identity_of(ledger.code_identity.tree_files(root), agent) for agent in ("plan_execute", "agent_as_tool", "causal")}

    before = identities()
    port.write_text("plan execute, changed\n", encoding="utf-8")
    after = identities()
    others = ("agent_as_tool", "causal")
    assert after["plan_execute"] != before["plan_execute"] and [after[a] for a in others] == [before[a] for a in others]
    (root / "CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/supervisor_runner.py").write_text("agent as tool, changed\n", encoding="utf-8")
    shared = identities()
    assert shared["plan_execute"] != after["plan_execute"] and shared["agent_as_tool"] != after["agent_as_tool"] and shared["causal"] == after["causal"]
