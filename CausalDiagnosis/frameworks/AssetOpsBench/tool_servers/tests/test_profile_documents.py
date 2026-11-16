"""The AssetOpsBench documents of a profile: the hyd_1 registry entry equals the stock entry but
for its sensors, the children and the fmsr classes are distinct and follow fmsr's own key, a record carries the
installed names in profile order and nothing else, timestamps sort in time order across the epoch, and the tiny asset
yields documents that name no hyd_1 table."""
import ast
import dataclasses
import json
import re
from datetime import datetime

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import REPO, ProfileDocuments, class_key, class_keys, stock_entry
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.cycle_timestamps import Timeline

TIMELINE = Timeline(datetime(2026, 1, 1), 60)  # the producer's timeline, whole seconds


def _fmsr_key():
    """fmsr's own _asset_class_key, compiled from its source without importing the server, which connects at import."""
    source = (REPO / "src" / "servers" / "fmsr" / "main.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_asset_class_key")
    namespace = {"re": re}
    exec(compile(ast.Module([node], []), "fmsr_key", "exec"), namespace)
    return namespace["_asset_class_key"]


def served_values(profile, cycle):
    return {f"{profile.field_name(c)}_{s}": float(cycle) + i for i, c in enumerate(profile.channels) for s in profile.statistics["served"]}


def test_the_class_key_is_fmsr_s():
    fmsr = _fmsr_key()
    for name in ("Hydraulic_Pump", "HYDRAULIC_PUMP", "COOLER", "pump2", "Air-Handling  Unit 3", "", None):
        assert class_key(name) == fmsr(name)


def test_the_hyd_1_registry_is_the_stock_entry_with_the_installed_names_and_four_children():
    profile = get_profile("hyd_1")
    asset, *children = ProfileDocuments(profile, TIMELINE).registry()
    stock = stock_entry(profile)
    assert list(asset) == list(stock) and {k: v for k, v in asset.items() if k != "sensors"} == {k: v for k, v in stock.items() if k != "sensors"}
    # the stock entry also installs cycle, a field no record of an episode carries
    assert asset["sensors"] == [profile.field_name(c) for c in profile.channels]
    assert sorted(asset["sensors"]) == sorted(s for s in stock["sensors"] if s != "cycle")
    assert [c["assetnum"] for c in children] == ["hyd_1_cooler", "hyd_1_valve", "hyd_1_pump", "hyd_1_accumulator"]
    assert all(c["parent"] == "hyd_1" and c["sensors"] == [] and c["assettype"] == c["assetnum"][6:].upper() for c in children)


def test_the_hyd_1_failure_modes_file_each_code_once_under_a_distinct_class():
    profile = get_profile("hyd_1")
    docs = ProfileDocuments(profile, TIMELINE).failure_modes()
    assert [d["asset_class"] for d in docs] == ["cooler", "valve", "pump", "accumulator", "hydraulic pump"]
    assert sorted(code for d in docs for code in d["failure_modes"]) == sorted(e["code"] for e in profile.failure_modes)
    assert docs[-1]["failure_modes"] == ["transient", "steady"]
    assert docs[0]["failure_modes"] == ["cooler_efficiency_critical", "cooler_efficiency_reduced", "cooler_ok"]
    assert all(d["exhaustive"] and d["source"] == profile.knowledge["fmea_source"] for d in docs)


def test_a_record_carries_the_installed_names_in_profile_order_and_refuses_any_other_value():
    profile = get_profile("hyd_1")
    documents = ProfileDocuments(profile, TIMELINE)
    record = documents.telemetry(3, served_values(profile, 3))
    fields = [profile.field_name(c) for c in profile.channels]
    assert list(record) == ["asset_id", "timestamp", *[name for f in fields for name in (f, f"{f}_std")]]
    assert set(fields) <= set(record) and record["asset_id"] == "hyd_1"
    with pytest.raises(ValueError, match="no served fields"):
        documents.telemetry(3, {**served_values(profile, 3), "combination": [3, 80, 0, 90]})


def test_timestamps_sort_in_time_order_across_the_epoch():
    documents = ProfileDocuments(get_profile("hyd_1"), TIMELINE)
    stamps = [documents.telemetry(c, {})["timestamp"] for c in range(-12, 13)]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)


def test_the_tiny_asset_yields_documents_that_name_no_hyd_1_table(tiny):
    documents = ProfileDocuments(tiny, TIMELINE)
    text = json.dumps([documents.registry(), documents.failure_modes(), documents.telemetry(0, served_values(tiny, 0))])
    assert "hyd_1" not in text and "HYDRAULIC" not in text and "Pressure" not in text
    asset, *children = documents.registry()
    assert asset["status"] == "OPERATING" and asset["sensors"] == ["T1_Temperature_C_1Hz", "V1_Vibration_mm_per_s_10Hz", "I1_Current_A_10Hz"]
    assert [c["assetnum"] for c in children] == ["tiny_1_fan", "tiny_1_belt", "tiny_1_bearing"]
    assert [d["asset_class"] for d in documents.failure_modes()] == ["fan", "belt", "bearing", "test bench"]


def test_a_component_whose_name_differs_from_its_class_key_files_its_documents_under_the_key(tiny_data):
    data = json.loads(json.dumps(tiny_data).replace('"belt"', '"linear_drive"'))
    profile = AssetProfile.from_json(data)
    assert class_keys(profile)["linear_drive"] == "linear drive"
    documents = ProfileDocuments(profile, TIMELINE)
    assert [d["asset_class"] for d in documents.failure_modes()] == ["fan", "linear drive", "bearing", "test bench"]
    assert [c["assettype"] for c in documents.registry()[1:]] == ["FAN", "LINEAR_DRIVE", "BEARING"]


def test_classes_that_collide_under_the_fmsr_key_are_refused():
    profile = dataclasses.replace(get_profile("hyd_1"), components=("pump1", "pump2"))
    with pytest.raises(ValueError, match="share the fmsr class 'pump'"):
        class_keys(profile)


def test_the_class_key_check_runs_as_the_aob_surface_s_declared_profile_check_and_not_in_validate():
    """The fmsr class keys bind only a run on AssetOpsBench's servers: the core's validate() accepts a profile whose components
    share a key, and the aob surface's declared profile check refuses it by name, on the raw mapping and on a loaded profile."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR
    from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate
    from CausalDiagnosis.utils import frameworks

    raw = json.loads((ASSETS_DIR / "hyd_1" / "asset.json").read_text(encoding="utf-8"))
    twin = json.loads(json.dumps(raw).replace('"valve"', '"cooler2"'))
    message = "cooler and cooler2 share the fmsr class 'cooler', under which the fmsr server files their failure modes"
    assert frameworks.surface_profile_problems(raw, "aob") == [] and frameworks.surface_profile_problems(get_profile("hyd_1"), "aob") == []
    assert frameworks.surface_profile_problems(twin, "aob") == [message] and message not in validate(twin)
    loaded = dataclasses.replace(get_profile("hyd_1"), components=("pump1", "pump2"))
    assert frameworks.surface_profile_problems(loaded, "aob") == [
        "pump1 and pump2 share the fmsr class 'pump', under which the fmsr server files their failure modes"]


def test_a_run_on_the_aob_surface_refuses_a_profile_its_check_refuses_before_the_first_episode(monkeypatch):
    """EpisodeRunner asks the surface of the servers it registers for its profile check, so a run on the metered servers stops
    with the surface's problems, and a run on the action server never asks."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world

    _facts, cycles, blocks = load_world()
    name, config = load_config("static")
    colliding = dataclasses.replace(get_profile(config.asset_id), components=("pump1", "pump2"))
    monkeypatch.setattr(run_episodes.play_one_episode, "get_profile", lambda asset_id: colliding)
    settings = {"judge": None, "grade": False, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}
    with pytest.raises(SystemExit, match=f"the aob surface refuses the profile of {config.asset_id}: pump1 and pump2 share the fmsr class 'pump'"):
        run_episodes.EpisodeRunner(config, name, "scripted", METERED_SERVERS, cycles, blocks, _facts, "refused", settings, store={"kind": "memory"})
    run_episodes.EpisodeRunner(config, name, "scripted", run_episodes.registered_servers("rig", config.diagnostics_server), cycles, blocks,
                               _facts, "accepted", settings)
