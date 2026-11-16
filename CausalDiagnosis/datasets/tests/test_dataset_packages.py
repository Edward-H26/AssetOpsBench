"""The asset packages (CausalDiagnosis/datasets/dataset_loader/dataset_packages.py) and the refusals of validation that a port meets: discovery by folder, the
declarations of a package, the label and the generated tables an asset declares, a table the engine cannot run, the plain
statistic that carries the normal range and the alarm, a fixed start without a recording, the one definition of the test set,
and the features picked by their exact names. The tests name no asset folder of the tree: they read the folders from the
disk, and the made-up port of the folder ports is mounted where a test needs it."""
import copy
import dataclasses
import importlib
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence, dataset_packages as ports, recordings as records
from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, get_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

from CausalDiagnosis.datasets.tests.conftest import PORT_ID, PORTS, mount_port, write_recordings
from CausalDiagnosis.datasets.tests.test_asset_profile import SCRIPTED


def _episode(of, asset=None):
    """A stand in for an environment, for the declarations that read its asset id, its start and its profile."""
    return SimpleNamespace(config=SimpleNamespace(asset_id=of), initial=asset, profile=get_profile(of) if asset is not None else None)


def _renamed(data, old, new):
    """The raw profile with every name old of a component replaced by new, in every table that names it."""
    return json.loads(re.sub(rf'"{old}"', f'"{new}"', json.dumps(data)))


def test_every_asset_folder_is_found_imported_and_validated():
    folders = sorted(path.parent.name for path in ASSETS_DIR.glob(f"*/{ports.ASSET_FILE}"))
    assert folders and ports.asset_ids() == folders
    for asset_id in folders:
        assert get_profile(asset_id).asset_id == asset_id
        assert ports.package(asset_id) is sys.modules[f"{ports.PACKAGE}.{asset_id}"]
    assert ports.package("no_such_asset") is None and ports.declared("no_such_asset", "LABELS", "unset") == "unset"


def test_a_port_is_found_when_its_folder_joins_the_path_and_gone_when_it_leaves():
    assert PORT_ID not in ports.asset_ids()
    with mount_port():
        assert ports.asset_dirs()[PORT_ID] == PORTS / PORT_ID and PORT_ID in ports.asset_ids()
        assert get_profile(PORT_ID).evidence_kind == "demo_history"
    assert PORT_ID not in ports.asset_ids() and f"{ports.PACKAGE}.{PORT_ID}" not in sys.modules
    with pytest.raises(KeyError, match=f"no asset profile '{PORT_ID}'"):
        get_profile(PORT_ID)


def _copy_of_port(folder, init_text):
    """The made-up port under folder with another package text, for a declaration that the tree holds nowhere."""
    target = folder / PORT_ID
    shutil.copytree(PORTS / PORT_ID, target, ignore=shutil.ignore_patterns("__pycache__"))
    (target / "__init__.py").write_text(init_text, encoding="utf-8")
    return folder


def test_a_declaration_that_does_not_resolve_is_refused_with_the_asset_and_the_reference(tmp_path):
    init = (PORTS / PORT_ID / "__init__.py").read_text(encoding="utf-8")
    broken = init.replace('"CausalDiagnosis.datasets.demo_1.labels:label_of"', '"CausalDiagnosis.datasets.demo_1.labels:no_such_function"').replace(
        '("CausalDiagnosis.datasets.demo_1.labels:TABLE_PATH",)', '("CausalDiagnosis.datasets.demo_1.no_such_module:TABLE_PATH",)')
    assert broken.count("no_such") == 2
    with mount_port(_copy_of_port(tmp_path / "broken", broken)):
        with pytest.raises(LookupError, match=r"demo_1: LABELS names 'CausalDiagnosis\.datasets\.demo_1\.labels:no_such_function'"):
            ports.start_label(_episode(PORT_ID))
        with pytest.raises(LookupError, match=r"demo_1: TABLES names 'CausalDiagnosis\.datasets\.demo_1\.no_such_module:TABLE_PATH'"):
            ports.generated_tables()
    no_path = init.replace('("CausalDiagnosis.datasets.demo_1.labels:TABLE_PATH",)', '("CausalDiagnosis.datasets.demo_1.labels:label_of",)')
    with mount_port(_copy_of_port(tmp_path / "no_path", no_path)):
        with pytest.raises(LookupError, match=r"demo_1: TABLES names 'CausalDiagnosis\.datasets\.demo_1\.labels:label_of', which is no path"):
            ports.generated_tables()


def test_the_label_of_a_start_comes_from_the_declaration_of_its_asset(tiny, mounted_port):
    assert ports.start_label(_episode(tiny.asset_id)) is None
    faulty, healthy = ports.start_label(_episode(PORT_ID, (1,))), ports.start_label(_episode(PORT_ID, (0,)))
    assert (faulty["separable"], faulty["e18"]["separable"], healthy["separable"], healthy["e18"]["separable"]) == (False, False, True, True)
    for asset_id in ports.asset_ids():
        reference = ports.declared(asset_id, "LABELS")
        assert reference is None or callable(ports.resolve(reference)), asset_id


def test_every_declared_table_is_a_path_attribute_of_its_module(mounted_port):
    found = ports.generated_tables()
    assert (f"{ports.PACKAGE}.{PORT_ID}.labels", "TABLE_PATH") in found
    declared = [ref.partition(":")[::2] for a in ports.asset_ids() for ref in ports.declared(a, "TABLES", ())]
    assert found == declared
    for module, attribute in found:
        assert isinstance(getattr(importlib.import_module(module), attribute), Path), (module, attribute)


def test_two_components_under_one_class_key_are_refused_and_a_name_that_differs_from_its_key_is_not(tiny_data):
    """The aob surface's profile check refuses components, or a component and the asset type, under one fmsr class key; the
    core's validate() leaves that check to the surface, since only fmsr files by the key."""
    from CausalDiagnosis.utils import frameworks

    pair = _renamed(_renamed(tiny_data, "fan", "comp1"), "belt", "comp2")
    shared = "comp1 and comp2 share the fmsr class 'comp', under which the fmsr server files their failure modes"
    assert frameworks.surface_profile_problems(pair, "aob") == [shared] and shared not in validate(pair)
    asset_type = _renamed(tiny_data, "bearing", "test_bench")
    assert frameworks.surface_profile_problems(asset_type, "aob") == [
        "test_bench and the asset type share the fmsr class 'test bench', under which the fmsr server files their failure modes"]
    renamed = _renamed(tiny_data, "belt", "linear_drive")
    assert validate(renamed) == [] and frameworks.surface_profile_problems(renamed, "aob") == []


def test_a_component_with_one_level_is_refused_by_the_asset_file_and_by_the_configuration(tiny, tiny_data):
    data = copy.deepcopy(tiny_data)
    data["levels"]["bearing"], data["degradation"]["bearing"] = [1], [1]
    data["failure_modes"] = [m for m in data["failure_modes"] if (m["component"], m["level"]) != ("bearing", 2)]
    next(c for c in data["rules"] if c["kind"] == "downtime")["parameters"]["rates"]["bearing"] = {"1": 0}
    assert validate(data) == ["levels of bearing must hold a second level beside the healthy one"]
    config = tiny.default_config()
    with pytest.raises(ValueError, match="levels of bearing must hold a second level beside the healthy one"):
        dataclasses.replace(config, levels={**config.levels, "bearing": [1]}, downtime_rate={**config.downtime_rate, "bearing": {"1": 0}})


def test_an_asset_that_plays_episodes_needs_a_word_for_the_whole_asset(tiny_data):
    tiny_data["text"]["whole_asset_patterns"] = []
    assert validate(tiny_data) == ["text.whole_asset_patterns must hold at least one pattern, the words of the summary check for the "
                                   "whole asset"]
    assert SCRIPTED["text"]["whole_asset_patterns"] == [] and validate(SCRIPTED) == []


def test_the_optional_sentences_are_strings_and_the_sentence_on_the_readings_follows_the_kind(tiny_data, mounted_port):
    for key in ("question", "opening", "scoring", "level_meaning"):
        data = copy.deepcopy(tiny_data)
        data["text"][key] = "A sentence of the asset."
        assert validate(data) == [], key
        data["text"][key] = ""
        assert validate(data) == [f"text.{key} must be a non-empty string"], key
    data = copy.deepcopy(tiny_data)
    data["text"]["readings"] = "A reading is a summary of a cycle."
    assert validate(data) == ["text.readings is stated by the evidence kind recorded_levels, whose readings loop"]
    data["text"]["colour"] = "red"
    assert validate(data)[0].startswith("text must hold exactly ['noun', 'short_noun'")
    port = json.loads((mounted_port / ports.ASSET_FILE).read_text(encoding="utf-8"))
    assert validate(port) == []
    del port["text"]["readings"]
    assert validate(port) == ["text.readings must say what a reading of the asset is, since the evidence kind demo_history declares no loop"]


def _tiny_world(tiny, folder, keep=lambda combination: True, edit=None):
    """The records and blocks of tiny recordings written into folder, the combinations that keep refuses left out, and
    edit giving the features of every record."""
    folder.mkdir(exist_ok=True)
    path = write_recordings(folder / "tiny.jsonl", tiny)
    cycles = [c for c in records.load_cycles(path, tiny) if keep(c.combo)]
    if edit is not None:
        cycles = [dataclasses.replace(c, features=edit(c)) for c in cycles]
    return cycles, records.build_blocks(cycles)


def test_the_band_and_the_alarm_read_the_plain_statistic(tiny_data, tmp_path):
    data = copy.deepcopy(tiny_data)
    data["statistics"]["served"] = ["std", "mean"]
    profile = register_profile(AssetProfile.from_json(data))
    try:
        faulty = (1, 100, 1)
        cycles, blocks = _tiny_world(profile, tmp_path, edit=lambda c: {**c.features, "Temperature_T1_std": 99.0} if c.combo == faulty else c.features)
        env = HydraulicRigEnv(cycles, blocks, profile.default_config())
        assert set(env.normal_ranges) == {f"{profile.field_name(c)}_std" for c in profile.channels}
        alarms = env.source.alarms([(3, blocks[faulty][0])], 0, ("T1",))
        assert alarms == [{"channel": "T1_Temperature_C_1Hz_std", "first_t": 3}]
    finally:
        unregister_profile(profile.asset_id)


def test_a_fixed_start_without_a_recording_is_refused_when_the_environment_is_built(tiny, tmp_path):
    missing = (1, 100, 1)
    cycles, blocks = _tiny_world(tiny, tmp_path, keep=lambda combination: combination != missing)
    config = dataclasses.replace(tiny.default_config(), start={"combination": list(missing), "entry_offset": 0})
    with pytest.raises(ValueError, match=r"the start combination \[1, 100, 1\] has no recording in the data of tiny_1"):
        HydraulicRigEnv(cycles, blocks, config)


def _two_components(tiny_data):
    """The tiny asset without its bearing and with two levels of the belt: four combinations, all three faulty ones within
    a fault rule of one or two faulty components, so the only development start is a test combination."""
    data = copy.deepcopy(tiny_data)
    data["components"] = ["fan", "belt"]
    data["levels"] = {"fan": [0, 1], "belt": [75, 100]}
    data["healthy"], data["degradation"] = {"fan": 0, "belt": 100}, {"fan": [0, 1], "belt": [100, 75]}
    for table in ("parts", "fix_kind"):
        del data["defaults"]["prices"][table]["bearing"]
    for table in ("lead_time", "repair_downtime"):
        del data["defaults"]["durations"][table]["bearing"]
    cards = {c["kind"]: c["parameters"] for c in data["rules"]}
    cards["downtime"]["rates"] = {"fan": {"0": 0, "1": 2}, "belt": {"100": 0, "75": 1}}
    cards["dependencies"].update(dependencies={"fan": [], "belt": []}, priority={"fan": 1, "belt": 2})
    data["sampler"]["fault_rule"] = {"min_faulty": 1, "max_faulty": 2}
    data["failure_modes"] = [m for m in data["failure_modes"] if m["component"] != "bearing" and m["level"] != 50]
    data["knowledge"]["indicates"] = {"T1": ["fan"], "V1": ["belt"], "I1": ["fan", "belt"]}
    del data["knowledge"]["correlations"]["bearing"]
    del data["records"]["label_keys"]["bearing"]
    return data


def test_the_start_sampler_leaves_out_what_the_development_rule_moved(tiny, tiny_data, tmp_path):
    two = register_profile(AssetProfile.from_json(_two_components(tiny_data), "two components"))
    try:
        cycles, blocks = _tiny_world(two, tmp_path / "two")
        config = two.default_config()
        assert HydraulicRigEnv(cycles, blocks, config).candidates == [(0, 75), (1, 100)]
        assert HydraulicRigEnv(cycles, blocks, dataclasses.replace(config, allowed=[[1, 75]])).candidates == [(1, 75)]
    finally:
        register_profile(tiny)
    cycles, blocks = _tiny_world(tiny, tmp_path)
    rule = tiny.default_config()
    assert HydraulicRigEnv(cycles, blocks, rule).candidates == sorted(c for c in tiny.combinations() if rule.within_fault_rule(c))
    cycles, blocks = _tiny_world(tiny, tmp_path / "no_level", keep=lambda combination: combination[1] != 50)
    assert HydraulicRigEnv(cycles, blocks, rule).candidates == sorted(c for c in blocks if rule.within_fault_rule(c))


def test_an_asset_id_that_names_a_module_of_the_asset_layer_is_refused(tiny_data):
    for name in ("dataset_loader", "new_dataset_scaffold", "tests"):
        assert validate(dict(tiny_data, asset_id=name)) == [
            f"asset_id {name!r} is the name of a module of the asset layer, which a folder of that name would hide from the import "
            "of CausalDiagnosis.datasets"], name


def test_a_record_without_a_declared_feature_is_refused_at_load(tiny, tmp_path):
    path = write_recordings(tmp_path / "tiny.jsonl", tiny)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[3])
    record["features"] = [f for f in record["features"] if f["name"] != "Vibration_V1_std"]
    lines[3] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(records.DataFileError, match=r"the record of cycle 3 holds no feature Vibration_V1_std, the std of the channel V1"):
        records.load_cycles(path, tiny)


def test_a_feature_of_another_quantity_is_not_served_for_a_channel(tiny):
    cycle = records.Cycle(0, (0, 100, 1), True, {"Temperature_T1_mean": 20.0, "Temperature_T1_std": 0.5, "Zzz_T1_mean": 999.0})
    assert evidence.served_values(cycle, ("T1",), tiny) == {"T1_Temperature_C_1Hz_mean": 20.0, "T1_Temperature_C_1Hz_std": 0.5}
