"""The registry of the frameworks (utils/frameworks.py): declarations read as data, never imported; unknown fields ignored with
one message; known fields that are no literal or have the wrong shape refused by name; a name two frameworks declare refused."""
import subprocess
import sys

import pytest

from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.utils.paths import ROOT


@pytest.fixture
def planted(tmp_path, monkeypatch):
    """A frameworks folder of its own: plant(name, text) writes a declaration, and the registry reads only that folder."""
    monkeypatch.setattr(frameworks, "FRAMEWORKS", tmp_path)
    frameworks.declaration.cache_clear()
    monkeypatch.setattr(frameworks, "_warned", set())

    def plant(name, text):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(text, encoding="utf-8")
        frameworks.declaration.cache_clear()

    yield plant
    frameworks.declaration.cache_clear()


def test_the_declaration_of_assetopsbench_reads_without_importing_a_module_of_the_framework():
    code = ("import sys\nfrom CausalDiagnosis.utils import frameworks as f\n"
            "f.agents(); f.surfaces(); f.judges(); f.scenario_sets(); f.run_kinds(); f.commands()\n"
            "f.names('CONFIGS'); f.references('PROFILE_CHECKS'); f.lists('AssetOpsBench')\n"
            "loaded = [m for m in sys.modules if m.startswith('CausalDiagnosis.frameworks')]\n"
            "print(f.frameworks(), loaded)")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r})\n{code}"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "('AssetOpsBench',) []"


def test_the_agents_surfaces_and_commands_of_assetopsbench_are_declared():
    assert set(frameworks.agents()) == {"plan_execute", "agent_as_tool", "scripted", "mcp_follower"}
    assert frameworks.surfaces()["aob"]["servers"] == ("iot", "fmsr", "wo", "task")
    assert all(entry["framework"] == "AssetOpsBench" for entry in frameworks.agents().values())
    assert set(frameworks.commands()) == {"assetopsbench"}


def test_an_unknown_field_is_ignored_with_one_message_and_the_known_fields_still_resolve(planted, capsys):
    planted("Later", 'AGENTS = {"later": {"model_agent": True}}\nNEW_FIELD = {"a": 1}\n')
    assert frameworks.agents() == {"later": {"model_agent": True, "framework": "Later"}}
    frameworks.declaration.cache_clear()
    frameworks.agents()
    assert capsys.readouterr().err == "frameworks/Later: field NEW_FIELD is not known to this core and is ignored\n"


def test_a_known_field_that_is_no_literal_stops_the_registry_by_name(planted):
    planted("Broken", 'import os\nAGENTS = {"a": {"route": os.sep}}\n')
    with pytest.raises(frameworks.FrameworkError, match="frameworks/Broken: field AGENTS is not a literal"):
        frameworks.agents()


@pytest.mark.parametrize("text, shape", [('CONFIGS = "scenarios"\n', "a tuple of strings"),
                                         ('SURFACES = {"s": {"entry": "no_colon"}}\n', "'module:attribute'"),
                                         ('COMMANDS = {"c": "m:attr"}\n', "a dict of commands to module strings")])
def test_a_known_field_of_the_wrong_shape_stops_the_registry_by_name(planted, text, shape):
    planted("Broken", text)
    with pytest.raises(frameworks.FrameworkError, match=f"frameworks/Broken: field [A-Z]+ must be .*{shape}"):
        frameworks.declaration("Broken")


def test_a_name_that_two_frameworks_declare_stops_the_registry(planted):
    planted("One", 'AGENTS = {"twin": {}}\n')
    planted("Two", 'AGENTS = {"twin": {}}\n')
    with pytest.raises(frameworks.FrameworkError, match="AGENTS twin is declared by frameworks/One and frameworks/Two"):
        frameworks.agents()


def test_code_is_imported_only_when_an_entry_is_resolved(planted):
    planted("Stub", 'JUDGES = {"j": {"entry": "json:dumps"}}\nPROFILE_CHECKS = {"s": "json:loads"}\nSURFACES = {"s": {}}\n')
    import json

    assert frameworks.code("JUDGES", "j") is json.dumps
    with pytest.raises(frameworks.FrameworkError, match="names no code"):
        frameworks.code("SURFACES", "s")
    assert frameworks.surface_profile_problems("[]", "s") == []


def test_the_surface_of_assetopsbench_declares_what_it_serves_of_the_environment_actions():
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract

    aob = frameworks.surfaces()["aob"]
    env_actions = set(contract.tools_of(contract.ACTION_SERVER))
    assert aob["action_server_name"] == "task" and aob["levels"] == ("framework", "full")
    assert set(aob["covers"]) | set(aob["withheld_actions"]) <= env_actions - set(frameworks.ANSWER_ACTIONS)
    assert set(frameworks.ANSWER_ACTIONS) <= env_actions


@pytest.mark.parametrize("keys, name", [('"action_server_name": "rig"', "action_server_name"),
                                        ('"covers": {"finish": ("iot.history",)}', "covers"),
                                        ('"covers": {"read_sensors": ("elsewhere.history",)}', "covers read_sensors"),
                                        ('"covers": {"read_sensors": ("task.read",)}, "action_server_name": "task"', "covers read_sensors"),
                                        ('"covers": {"read_sensors": ("iot.history",)}, "withheld_actions": ("read_sensors",)', "withheld_actions"),
                                        ('"levels": ("L0",)', "levels"),
                                        ('"gateway": "no_colon"', "gateway")])
def test_a_surface_key_of_the_wrong_shape_stops_the_registry_by_its_name(planted, keys, name):
    planted("Broken", f'SURFACES = {{"s": {{"servers": ("iot", "task"), {keys}}}}}\n')
    with pytest.raises(frameworks.FrameworkError, match=f"frameworks/Broken: field SURFACES entry s: {name}"):
        frameworks.declaration("Broken")


def test_a_surface_that_serves_none_of_the_environment_actions_loads_without_those_keys(planted):
    planted("Plain", 'SURFACES = {"s": {"servers": ("x",)}}\n')
    assert frameworks.surfaces() == {"s": {"servers": ("x",), "framework": "Plain"}}
