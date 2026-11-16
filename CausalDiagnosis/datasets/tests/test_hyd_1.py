"""The hyd_1 asset file and its record loader against the rig modules they were taken from: the tables of hyd_1_tables.py,
former_field_names.py and knowledge_tables.py, which now serve the asset file and keep the repr digests of the baseline, the degradation
order of the deterioration layer, the grid of tier_classifiers.py, the quantities of build_features_and_blocks.py, the served statistics that the texts of the rig name and the
whole asset pattern of verifier/, the configuration digests stored at the baseline, and the loaders of hyd_1_tables.py on a
synthetic file and on the facts file, the environment that make_env builds over them, and the tool contract of hyd_1,
which is the contract file, with the clauses of the emergency card."""
import dataclasses
import hashlib
import importlib.util
import json
import re

import pytest

from CausalDiagnosis.utils.paths import preset_path  # noqa: E402
from CausalDiagnosis.utils.json_digest import digest
from CausalDiagnosis.datasets.hyd_1 import records as hyd_1
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.reference_policies import tier_classifiers as classifiers
from CausalDiagnosis.datasets.hyd_1 import hyd_1_tables as data
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import env
from CausalDiagnosis.datasets.hyd_1.tests import knowledge_tables as knowledge
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier, environment_factory as world
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.datasets.hyd_1 import former_field_names as names

from CausalDiagnosis.datasets.tests.conftest import ROOT, needs_data, write_recordings

PRESET_FIELDS = {"allowed", "start", "guidance_level", "question", "anchors", "verifier_scope", "diagnostics_server"}
# fields that the engine changed on purpose after the baseline, which the asset file follows (the coinciding windows of
# the fix history, and the price of a wrong diagnosis)
CHANGED_SINCE_BASELINE = {"fix_history", "prices"}
# fields that the engine gained after the baseline: the thinking rate, which the asset file decides,
# and the switch of the time tools of the real-time clock, which a composition sets
ADDED_SINCE_BASELINE = {"think_seconds_per_cycle", "time_tools", "seconds_per_token"}
# seconds_per_token, the rate of the generated_tokens clock, which the JSON of a configuration states in that mode alone
TOKEN_ONLY = {"seconds_per_token"}
# fields that the engine lost after the baseline
REMOVED_SINCE_BASELINE = {"external_events"}  # removed with the card of external events
# the repr digests of the tables and lookups of hyd_1_tables.py, former_field_names.py, knowledge_tables.py and tier_classifiers.py at the baseline, when these
# modules held the tables as literals, so that a change of a value, an order or a type of a served table fails here
BASELINE_TABLES = {
    "data.COMPONENTS": "4b030871ceb1a0ad", "data.LABEL_KEYS": "134def32c0a76f11", "data.LEVELS": "815e8e868dfbd6af",
    "data.HEALTHY": "a3862fe6971203d0", "data.SENSORS": "f43c87a9a8e4f2d1", "data.TIER_SENSORS": "415ed005e6fd00d1",
    "data.SUMMARY_SUFFIXES": "26595667e868b4b6", "data.STABLE_FLAG_STABLE": "5feceb66ffc86f38", "data.DATA_DIR": "0e4f0d5260107ee7",
    "names.ASSET_ID": "1ddc5c5f92dc58ff", "names.HYD1_FIELDS": "e1588cc4fda7988d",
    "knowledge.ASSOCIATION_SOURCE": "de54e6c07b7d1ae5", "knowledge.FMEA_SOURCE": "9414ad6b8510eb3e",
    "knowledge.STATISTICS": "898956f820fee22e", "knowledge.INDICATES": "ee5046972b255b5e",
    "knowledge.HELWIG_CORRELATIONS": "0eae10bcd2e61549", "knowledge.FAILURE_CODES": "9f144eae8aae42e0",
    "knowledge.CODES": "9673d3630c907bff", "knowledge.CODE_OF_LEVEL": "e387f579d6a716cb", "knowledge.FAULT_CODES": "052aeebebc9d85a6",
    "knowledge.PREFIX_TO_SENSOR": "78361e452476387e", "knowledge.ASSET_PROFILE": "0eb936c867ea2509",
    "classifiers.COMBOS": "e7efe13c1ab5381d", "classifiers.COMBO_INDEX": "097c1228c8232ac1",
}
BASELINE_LOOKUPS = {"fallback": "1e80f3d172ee5f16", "unit_and_rate": "8f65614f799eef2b", "public_name": "a3b2540afc49f82d",
                    "tier_fields": "d2f8acf7c9104921"}


def _build_blocks_module():
    spec = importlib.util.spec_from_file_location("hyd_1_build_blocks", ROOT / "CausalDiagnosis" / "datasets" / "hyd_1" / "build_features_and_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repr_digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def test_the_served_tables_keep_the_values_orders_and_types_of_the_baseline(monkeypatch, tmp_path):
    served = {"data.COMPONENTS": data.COMPONENTS, "data.LABEL_KEYS": data.LABEL_KEYS, "data.LEVELS": data.LEVELS,
              "data.HEALTHY": data.HEALTHY, "data.SENSORS": data.SENSORS, "data.TIER_SENSORS": data.TIER_SENSORS,
              "data.SUMMARY_SUFFIXES": data.SUMMARY_SUFFIXES, "data.STABLE_FLAG_STABLE": data.STABLE_FLAG_STABLE,
              "data.DATA_DIR": data.DATA_DIR.relative_to(ROOT), "names.ASSET_ID": names.ASSET_ID, "names.HYD1_FIELDS": names.HYD1_FIELDS,
              "knowledge.ASSOCIATION_SOURCE": knowledge.ASSOCIATION_SOURCE, "knowledge.FMEA_SOURCE": knowledge.FMEA_SOURCE,
              "knowledge.STATISTICS": knowledge.STATISTICS, "knowledge.INDICATES": knowledge.INDICATES,
              "knowledge.HELWIG_CORRELATIONS": knowledge.HELWIG_CORRELATIONS, "knowledge.FAILURE_CODES": knowledge.FAILURE_CODES,
              "knowledge.CODES": knowledge.CODES, "knowledge.CODE_OF_LEVEL": knowledge.CODE_OF_LEVEL,
              "knowledge.FAULT_CODES": knowledge.FAULT_CODES, "knowledge.PREFIX_TO_SENSOR": knowledge.PREFIX_TO_SENSOR,
              "knowledge.ASSET_PROFILE": knowledge.ASSET_PROFILE.relative_to(ROOT), "classifiers.COMBOS": classifiers.COMBOS,
              "classifiers.COMBO_INDEX": classifiers.COMBO_INDEX}
    assert {name: _repr_digest(value) for name, value in served.items()} == BASELINE_TABLES
    profile = get_profile("hyd_1")
    monkeypatch.setitem(profile.text, "registry_file", str(tmp_path / "absent.json"))
    # the lookups that every reader calls, the profile's own since the knowledge view's copies of them left
    lookups = {"fallback": profile.registry_entry(), "unit_and_rate": {s: profile.unit_and_rate(s) for s in data.SENSORS},
               "public_name": [names.public_name(f"{q}_{s}_{st}") for s in data.SENSORS for q in ("Pressure", "X")
                               for st in ("mean", "std", "min", "max")],
               "tier_fields": {t: names.tier_fields(t) for t in data.TIER_SENSORS}}
    assert {name: _repr_digest(value) for name, value in lookups.items()} == BASELINE_LOOKUPS


def test_the_tables_equal_the_rig_modules():
    profile = get_profile("hyd_1")
    assert profile.asset_id == names.ASSET_ID
    assert profile.components == data.COMPONENTS
    assert profile.levels == data.LEVELS and list(profile.levels) == list(data.LEVELS)
    assert profile.healthy == data.HEALTHY
    assert tuple(profile.channels) == data.SENSORS
    assert {name: tuple(spec["channels"]) for name, spec in profile.observation_kinds.items()} == \
        {f"tier{t}": sensors for t, sensors in data.TIER_SENSORS.items()}
    assert {c: spec["field"] for c, spec in profile.channels.items()} == names.HYD1_FIELDS
    assert {c: spec["quantity"] for c, spec in profile.channels.items()} == _build_blocks_module().QUANTITY
    assert tuple(f"_{s}" for s in profile.statistics["recorded"]) == data.SUMMARY_SUFFIXES
    assert profile.statistics["served"] == ["mean", "std"]  # the recorded mean and std that the texts of the rig name
    assert profile.records["label_keys"] == data.LABEL_KEYS and profile.records["stable_flag_stable"] == data.STABLE_FLAG_STABLE


def test_the_knowledge_and_the_texts_equal_the_rig_modules():
    profile = get_profile("hyd_1")
    assert profile.failure_modes == knowledge.FAILURE_CODES and profile.fault_modes() == knowledge.FAULT_CODES
    assert profile.knowledge == {"fmea_source": knowledge.FMEA_SOURCE, "association_source": knowledge.ASSOCIATION_SOURCE,
                                 "indicates": {k: list(v) for k, v in knowledge.INDICATES.items()},
                                 "correlations": knowledge.HELWIG_CORRELATIONS}
    assert ROOT / profile.text["registry_file"] == knowledge.ASSET_PROFILE
    whole_asset = verifier._asset_claims(profile.components, tuple(profile.text["whole_asset_patterns"]))[0]
    # the pattern that the verifier held as a module constant before the asset layer
    assert (whole_asset.pattern, whole_asset.flags & re.IGNORECASE) == \
        (r"\b(?:rig|asset|system|machine|hyd_1|all (?:four )?components|every component)\b", re.IGNORECASE)


def test_the_degradation_orders_and_the_grid_equal_the_engine():
    profile = get_profile("hyd_1")
    config = EpisodeConfig()
    assert {c: list(profile.degradation[c]) for c in profile.components} == {c: degradation_order(config, c) for c in data.COMPONENTS}
    assert profile.combinations() == classifiers.COMBOS
    assert profile.healthy_state() == tuple(data.HEALTHY[c] for c in data.COMPONENTS)


def test_the_contract_of_hyd_1_is_the_contract_file():
    """The contract of the hyd_1 asset is the file, down to the order of the parameters of submit_diagnosis, so every
    hyd_1 episode keeps the tool rows of the file."""
    assert contract.load_contract("hyd_1") == contract.load_contract()
    parameters = list(contract.load_contract("hyd_1")["tools"][contract.DIAGNOSIS_TOOL]["parameters"])
    assert parameters == list(contract.load_contract()["tools"][contract.DIAGNOSIS_TOOL]["parameters"])
    assert parameters == [*get_profile("hyd_1").components, "evidence_ids"]
    assert contract.public_table(asset_id="hyd_1") == contract.public_table()


def test_the_emergency_card_off_leaves_out_only_the_mode_and_its_clauses_of_the_work_order_row():
    """A card that is off takes its parameters and the clauses of its rules out of the public tool table and changes no
    other row. A clause that its rule text does not hold exactly once stops the table, so the rule texts and the
    card_clauses of the contract cannot drift apart unseen."""
    on = {row["name"]: row for row in contract.public_table(asset_id="hyd_1")}
    off = {row["name"]: row for row in contract.public_table(asset_id="hyd_1", cards_off=("emergency",))}
    assert [name for name in on if on[name] != off[name]] == ["create_work_order"]
    order_on, order_off = on["create_work_order"], off["create_work_order"]
    assert list(order_off["parameters"]) == [name for name in order_on["parameters"] if name != "mode"]
    assert "emergency" in order_on["duration"] and "emergency" in order_on["price"]
    assert "emergency" not in order_off["duration"] and "emergency" not in order_off["price"]
    clauses = contract.load_contract()["tools"]["create_work_order"]["card_clauses"]["emergency"]["price"]
    with pytest.raises(ValueError, match="occurs 0 times"):
        contract._without("create_work_order", order_off["price"], clauses)


def test_the_configuration_fields_keep_the_baseline_digests(baseline_digests):
    fields = get_profile("hyd_1").config_fields()
    stored = {key[len("config.default."):] for key in baseline_digests if key.startswith("config.default.")}
    assert set(fields) - ADDED_SINCE_BASELINE == stored - PRESET_FIELDS - REMOVED_SINCE_BASELINE
    for field, value in fields.items():
        if field not in CHANGED_SINCE_BASELINE | ADDED_SINCE_BASELINE:
            assert digest(value) == baseline_digests[f"config.default.{field}"], field


def test_the_history_card_carries_the_windows_of_the_engine_configuration(baseline_digests):
    card = get_profile("hyd_1").card("fix_history")["parameters"]
    assert card == EpisodeConfig().fix_history
    assert digest(card) != baseline_digests["config.default.fix_history"]  # the coinciding windows, not those of the baseline


def test_the_engine_configuration_gains_no_field_and_differs_from_the_asset_only_where_the_engine_changed(baseline_digests):
    built = get_profile("hyd_1").default_config().to_json()
    engine = EpisodeConfig().to_json()
    assert list(built) == list(engine) == [f.name for f in dataclasses.fields(EpisodeConfig) if f.name not in TOKEN_ONLY]
    baseline = {field for field in engine if field not in ADDED_SINCE_BASELINE}
    assert {f"config.default.{field}" for field in baseline | REMOVED_SINCE_BASELINE} == {
        key for key in baseline_digests if key.startswith("config.default.")}
    differing = {field for field in engine if digest(built[field]) != digest(engine[field])}
    changed_since_baseline = {field for field in baseline if digest(engine[field]) != baseline_digests[f"config.default.{field}"]}
    assert differing <= changed_since_baseline


def _assert_same_loaders(path):
    ours, theirs = hyd_1.load_cycles(path), data.load_cycles(path)
    assert [(c.index, c.combo, c.stable, c.features) for c in ours] == [(c.index, c.combo, c.stable, c.features) for c in theirs]
    assert hyd_1.build_blocks(ours) == data.build_blocks(theirs)


def test_the_loaders_equal_the_rig_loaders_on_a_synthetic_file(tmp_path):
    _assert_same_loaders(write_recordings(tmp_path / "hyd_cycles.jsonl", get_profile("hyd_1"), seed=3, stable=2))


@needs_data
def test_the_loaders_equal_the_rig_loaders_on_the_facts_file():
    _assert_same_loaders(hyd_1.default_facts_path())


def test_the_data_file_paths_equal_the_rig_paths(monkeypatch, tmp_path):
    assert hyd_1.release_path() == data.DATA_DIR / "hyd_cycles.jsonl"
    monkeypatch.setenv("HYD_FACTS_PATH", str(tmp_path / "facts.jsonl"))
    assert hyd_1.default_facts_path() == data.default_facts_path() == world.default_facts_path(EpisodeConfig()) == tmp_path / "facts.jsonl"
    monkeypatch.delenv("HYD_FACTS_PATH")
    assert hyd_1.default_facts_path() == data.default_facts_path() == world.default_facts_path(EpisodeConfig()) == hyd_1.release_path()


@needs_data
def test_make_env_builds_the_rig_environment_over_one_shared_world():
    config = EpisodeConfig()
    first, second = world.make_env(config), world.make_env(config)
    cycles = data.load_cycles(data.default_facts_path())
    reference = env.HydraulicRigEnv(cycles, data.build_blocks(cycles), config)
    assert first.blocks is second.blocks
    for seed in (0, 1, 7):
        assert first.reset(seed) == reference.reset(seed)


def _preset(name):
    """The configuration of a preset of CausalDiagnosis/configs as the drivers read it, the composition of its big
    layers."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers

    return layers.compose(json.loads(preset_path(name).read_text(encoding="utf-8")))


@needs_data
def test_a_return_to_the_start_resumes_its_window_on_the_rig_path():
    """On the repair rules row an emergency repair that expires takes the component back to its level, and the rig then
    continues the window of the start combination where it left it, as the engine of the AssetOpsBench rows does, so no
    recording the start window showed before the exit comes back before the window wraps. The start
    has one faulty component above its worst level, so the failure rule cannot end the episode first."""
    config = _preset("repair_rules")
    rig = world.make_env(config)

    def faulty(start):
        return [c for c, level in zip(config.components, start) if level != config.healthy[c]]

    def above_worst(start):
        return all(level != degradation_order(config, c)[-1] for c, level in zip(config.components, start))

    seed = next(s for s in range(200) if rig.reset(s) and len(faulty(rig.hidden)) == 1 and above_worst(rig.hidden))
    start, window = tuple(rig.hidden), rig.source.window
    rig.step({"type": "read_sensors", "sensor_group": next(iter(config.sensor_groups)), "cycles": 3})
    rig.step({"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, start)), "evidence_ids": []})
    rig.step({"type": "create_work_order", "component": faulty(start)[0], "prediction_id": "", "evidence_ids": [], "mode": "emergency"})
    for _ in range(2):  # to the close of the emergency order, then past the expiry of its hold
        rig.step({"type": "wait_for_notification", "timeout": 30})
    # the recordings the rig played, each under the combination of its cycle, from the clock's own record, so that the cycles
    # a timed read plays on a copy of the engine to learn its price do not count
    plays = [(tuple(rig.clock.combination_at(t)), index) for t, index in rig.clock.played if index is not None]
    visits = [combo for combo, _ in plays]
    exit_at = next(i for i, combo in enumerate(visits) if combo != start)
    return_at = next(i for i in range(exit_at, len(visits)) if visits[i] == start)
    before = [index for _, index in plays[:exit_at]]
    after = [index for combo, index in plays[return_at:] if combo == start]
    cap = len(window["cycles"])
    assert seed is not None and rig.source.window is window and before and after
    assert before + after == [window["cycles"][(window["offset"] + k) % cap] for k in range(len(before) + len(after))]
    assert not set(before) & set(after[:cap - len(before)])
