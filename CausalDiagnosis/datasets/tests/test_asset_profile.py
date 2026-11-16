"""The asset profile: validation as the only gate, listing every problem at once and the problems of the engine
configuration a profile builds, the round trip of the asset file, the configuration built from the tables and the cards,
the processes block, and the registry by asset id."""
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.utils.json_digest import canonical
from CausalDiagnosis.datasets.dataset_loader.profile_data import CONFIG_KINDS, PROCESS_FIELDS, AssetProfile, ProfileError, processes_block
from CausalDiagnosis.datasets.dataset_loader.profile_validation import CONFIG_PREFIX, validate
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, load_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.rule_cards import read_card
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import STOPPED, EpisodeConfig

SCRIPTED = {
    "format": 1, "asset_id": "cases_1", "evidence_kind": "scripted_cases", "components": [], "levels": {}, "healthy": {},
    "degradation": {}, "observation_kinds": {"photo": {}, "log": {}}, "channels": {}, "statistics": {}, "defaults": {},
    "rules": [],
    "sampler": {}, "failure_modes": [], "knowledge": {}, "records": {},
    "text": {"noun": "case", "short_noun": "case", "episode_noun": "fault case", "unit_noun": "unit", "whole_asset_patterns": [],
             "registry_file": None, "fallback": {"asset_type": "CASE", "description": "Scripted fault case", "site": "TEST"}},
    "scoring": "judge",
}


def test_the_hyd_1_asset_file_validates_and_round_trips(hyd_1_data):
    assert validate(hyd_1_data) == []
    profile = AssetProfile.from_json(hyd_1_data)
    assert profile.to_json() == hyd_1_data and canonical(profile.to_json()) == canonical(hyd_1_data)
    assert AssetProfile.from_json(profile.to_json()) == profile


def test_a_profile_lists_every_problem_at_once(hyd_1_data):
    data = hyd_1_data
    data["colour"] = "red"
    data["asset_id"] = "hyd 1"
    data["scoring"] = "vote"
    data["healthy"]["pump"] = 7
    data["degradation"]["valve"] = [90, 100, 80, 73]
    data["observation_kinds"]["tier1"]["channels"].append("XS9")
    data["rules"][0]["kind"] = "fix_histories"
    data["failure_modes"][0]["severity"] = "fatal"
    data["text"]["whole_asset_patterns"].append("(")
    problems = validate(data)
    expected = [
        "unknown fields ['colour']",
        "asset_id must be letters, digits and underscores",
        "scoring must be one of ['program', 'judge']",
        "the healthy level of pump must be one of its levels",
        "degradation of valve must start at the healthy level",
        "observation kind tier1 names channels ['XS9'] that channels does not declare",
        "failure_modes[0] has the severity 'fatal', not one of ['critical', 'degraded', 'healthy', 'info']",
    ]
    for message in expected:
        assert message in problems
    assert any(p.startswith("rules[0] has the unknown kind 'fix_histories'") for p in problems)
    assert any(p.startswith("text.whole_asset_patterns holds '(', which does not compile") for p in problems)
    assert len(problems) == len(expected) + 2
    with pytest.raises(ProfileError) as raised:
        AssetProfile.from_json(data, "draft.json")
    assert raised.value.problems == problems and str(raised.value).startswith("draft.json: invalid asset profile: ")


def test_the_problems_of_the_engine_configuration_come_back_through_validation(hyd_1_data):
    fields = AssetProfile.from_json(hyd_1_data).config_fields()
    fields["prices"]["visit"], fields["budget"] = -1, 0
    with pytest.raises(ValueError) as raised:
        EpisodeConfig(**fields)
    engine_problems = str(raised.value)[len(CONFIG_PREFIX):].split("; ")
    data = hyd_1_data
    next(c for c in data["rules"] if c["kind"] == "visits")["parameters"]["charge"] = -1
    data["defaults"]["limits"]["budget"] = 0
    assert len(engine_problems) == 2
    assert validate(data) == [f"the configuration built from the profile: {problem}" for problem in engine_problems]


def test_the_tables_load_without_the_engine_check_while_the_engine_is_imported(hyd_1_data):
    data = hyd_1_data
    data["defaults"]["limits"]["budget"] = 0
    problems = validate(data)
    assert problems and all(p.startswith("the configuration built from the profile") for p in problems)
    assert validate(data, engine=False) == []
    assert get_profile("hyd_1", engine=False) == get_profile("hyd_1")


def test_a_profile_states_each_kind_at_most_once_and_may_state_none(hyd_1_data):
    data = copy.deepcopy(hyd_1_data)
    data["rules"].append(copy.deepcopy(data["rules"][3]))
    assert validate(data) == ["rules name the kinds ['hazard'] more than once"]
    data = hyd_1_data
    data["rules"] = []
    assert validate(data) == []
    AssetProfile.from_json(data).default_config()


def test_a_kind_without_a_card_takes_its_off_parameters_in_the_configuration(hyd_1_data):
    visits = next(c for c in hyd_1_data["rules"] if c["kind"] == "visits")
    hyd_1_data["rules"] = [visits]
    profile = AssetProfile.from_json(hyd_1_data)
    config = profile.default_config()
    for name in CONFIG_KINDS:
        kind = rules.kind_of(name)
        assert read_card(config.to_json(), kind) == (visits["parameters"] if name == "visits" else kind.off(profile)), name
    assert config.schedule["mode"] == "none" and config.dependencies == {}
    assert config.hazard["collateral"] == config.fix_history["healthy_probability"] == config.fix_history["degraded_probability"] == 0
    assert not any(config.level_rate(c, level) for c in config.components for level in config.levels[c])
    assert config.downtime_rate[STOPPED] == 0 and not config.emergency_on() and not config.failure_on()


def test_the_configuration_follows_the_tables_and_the_cards_in_any_order(hyd_1_data):
    profile = AssetProfile.from_json(hyd_1_data)
    config = profile.default_config()
    assert isinstance(config, EpisodeConfig) and config.asset_id == "hyd_1"
    assert config.fault_rule == hyd_1_data["sampler"]["fault_rule"]
    shuffled = copy.deepcopy(hyd_1_data)
    shuffled["rules"].reverse()
    for card in shuffled["rules"]:
        card["parameters"] = dict(reversed(list(card["parameters"].items())))
    assert AssetProfile.from_json(shuffled).config_fields() == profile.config_fields()
    assert canonical(AssetProfile.from_json(shuffled).default_config().to_json()) == canonical(config.to_json())


def test_the_processes_block_holds_the_fields_of_the_d05_cards_with_the_values_an_episode_runs(hyd_1_data):
    profile = AssetProfile.from_json(hyd_1_data)
    assert PROCESS_FIELDS == ("schedule", "hazard", "emergency", "dependencies", "priority", "fix_history", "failure")
    seeded = {**hyd_1_data["rules"][2]["parameters"], "mode": "seeded"}
    episode = dataclasses.replace(profile.default_config(), schedule=seeded)
    fields = {}
    for card in profile.rules:
        rules.apply_card(fields, {"kind": card["kind"], "parameters": read_card(episode.to_json(), rules.kind_of(card["kind"]))})
    block = processes_block(episode)
    assert block == processes_block(episode.to_json()) == {field: fields[field] for field in PROCESS_FIELDS}
    assert block["schedule"] == seeded and "min_faulty" not in json.dumps(block)


def test_the_registry_serves_bundled_and_registered_profiles_and_refuses_unsafe_ids(hyd_1_data):
    bundled = get_profile("hyd_1")
    assert bundled is get_profile("hyd_1") and bundled == AssetProfile.from_json(hyd_1_data)
    draft = register_profile(dataclasses.replace(bundled, scoring="judge"))
    try:
        assert get_profile("hyd_1") is draft
    finally:
        unregister_profile("hyd_1")
    assert get_profile("hyd_1") is bundled
    for unsafe in ("../hyd_1", "hyd_1/../hyd_1", "hyd-1", "", None, 7):
        with pytest.raises(KeyError, match="letters, digits and underscores"):
            get_profile(unsafe)
    with pytest.raises(KeyError, match="no asset profile 'no_such_asset'"):
        get_profile("no_such_asset")


def test_a_scripted_cases_profile_validates_without_an_engine_configuration():
    assert validate(SCRIPTED) == []
    profile = AssetProfile.from_json(SCRIPTED)
    with pytest.raises(ValueError, match="since the engine plays no episode of that kind"):
        profile.config_fields()
    data = copy.deepcopy(SCRIPTED)
    data["rules"].append({"kind": "schedule", "parameters": {"mode": "none", "components": 1, "first_drop": [5, 40], "interval": 10}})
    assert validate(data) == ["rules[0] (schedule): the kind applies to ['recorded_levels'], not to scripted_cases"]


def test_loading_a_broken_file_names_the_file_and_every_problem(tmp_path, hyd_1_data):
    hyd_1_data["format"] = 2
    del hyd_1_data["knowledge"]
    path = tmp_path / "asset.json"
    path.write_text(json.dumps(hyd_1_data), encoding="utf-8")
    with pytest.raises(ProfileError) as raised:
        load_profile(path)
    assert str(raised.value).startswith(f"{path}: invalid asset profile: ")
    assert raised.value.problems == ["missing fields ['knowledge']", "format must be 1"]


def test_one_unusable_table_is_one_problem_and_not_a_cascade(hyd_1_data):
    data = copy.deepcopy(hyd_1_data)
    data["components"] = "cooler valve pump accumulator"
    assert validate(data) == ["components must be a list of distinct non-empty names"]
    data = copy.deepcopy(hyd_1_data)
    data["channels"] = []
    assert validate(data) == ["channels must map every channel to its public field and its quantity"]
    data = hyd_1_data
    data["evidence_kind"] = "recorded"
    assert validate(data) == ["evidence_kind must be one of ['recorded_levels', 'scripted_cases']"]


def test_names_that_the_tools_and_the_data_file_cannot_carry_are_refused(hyd_1_data):
    data = copy.deepcopy(hyd_1_data)
    data["components"][1] = "evidence_ids"
    for table in ("levels", "healthy", "degradation"):
        data[table]["evidence_ids"] = data[table].pop("valve")
    problems = validate(data)
    assert problems[0] == "components ['evidence_ids'] must be identifiers other than keywords and evidence_ids"
    data = hyd_1_data
    data["channels"]["PS_7"] = {"field": "PS7_Pressure_bar_100Hz", "quantity": "Pressure"}
    data["statistics"]["recorded"].append("p_95")
    assert validate(data) == [
        "channels ['PS_7'] must be letters and digits, as the feature names of the data file spell them",
        "statistics must give the recorded statistics, letters and digits, and the served ones among them",
    ]


def test_the_registry_entry_is_read_once_per_profile_and_belongs_to_it(hyd_1_data, monkeypatch):
    """The registry file is read on the first call of a profile alone, an entry that the caller changes leaves the next one
    as it was, and a second profile registered under the same asset id with another fallback and no registry file returns
    its own entry. The last assertion fails on an entry kept by the asset id."""
    registry = Path(hyd_1_data["text"]["registry_file"]).name
    opened, read_text = [], Path.read_text

    def counted(path, *args, **kwargs):
        if path.name == registry:
            opened.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    profile = AssetProfile.from_json(hyd_1_data)
    entries = [profile.registry_entry() for _ in range(5)]
    assert len(opened) == 1 and all(entry == entries[0] for entry in entries) and entries[0]["asset_id"] == profile.asset_id
    entries[0]["site"] = "changed"
    assert profile.registry_entry() == entries[1] != entries[0]
    fallback = {"asset_type": "TEST", "description": "another entry", "site": "ELSEWHERE"}
    other = register_profile(dataclasses.replace(profile, text={**profile.text, "registry_file": None, "fallback": fallback}))
    try:
        assert get_profile(profile.asset_id) is other
        assert other.registry_entry() == {"asset_id": profile.asset_id, **fallback} != profile.registry_entry()
    finally:
        unregister_profile(profile.asset_id)
    assert len(opened) == 1
