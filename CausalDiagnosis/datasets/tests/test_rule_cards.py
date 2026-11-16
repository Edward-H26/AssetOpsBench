"""The rule library: the built in card kinds, every problem of a card at once, the evidence kinds a kind applies to, the
plugin registry, and the binds that write a card into the episode configuration in the engine's key order."""
import pytest

from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.rule_cards import RuleKind, apply_card, card_problems, kind_of, kinds, read_card, register_kind, unregister_kind
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import STOPPED, EpisodeConfig

SCRIPTED = {"evidence_kind": "scripted_cases", "observation_kinds": {"photo": {}, "log": {}}}


@pytest.fixture
def plugin():
    kind = register_kind(RuleKind("wear_rate", ("rate", "floor")))
    yield kind
    unregister_kind(kind.name)


def test_the_built_in_kinds_are_the_eight_of_the_design_in_order():
    assert kinds() == ("fix_history", "emergency", "schedule", "hazard", "dependencies", "visits", "downtime", "failure")
    assert all(kind_of(name).name == name for name in kinds())
    assert kind_of("no_such_kind") is None


def test_a_card_lists_every_problem_of_its_keys_at_once():
    card = {"kind": "schedule", "parameters": {"mode": "none", "components": 1, "every": 3}}
    problems = card_problems(card, where="rules[2]")
    assert problems == ["rules[2] (schedule): missing parameters ['first_drop', 'interval']",
                        "rules[2] (schedule): unknown parameters ['every'], the keys of schedule are "
                        "['mode', 'components', 'first_drop', 'interval']"]


def test_a_card_of_the_wrong_shape_or_an_unknown_kind_names_what_is_allowed():
    assert card_problems({"kind": "schedule"}) == ["card must be an object with exactly the keys kind and parameters"]
    assert card_problems({"kind": "schedule", "parameters": [1]}) == ["card (schedule): parameters must be an object"]
    unknown = card_problems({"kind": "corrosion", "parameters": {}})
    assert len(unknown) == 1 and "'corrosion'" in unknown[0] and "a plugin registered by name" in unknown[0]


def test_a_kind_applies_only_to_the_evidence_kinds_it_declares():
    schedule = {"kind": "schedule", "parameters": {"mode": "none", "components": 1, "first_drop": [5, 40], "interval": 10}}
    assert card_problems(schedule, {"evidence_kind": "recorded_levels"}) == []
    assert card_problems(schedule, SCRIPTED) == ["card (schedule): the kind applies to ['recorded_levels'], not to scripted_cases"]


def test_the_check_of_a_kind_runs_once_its_keys_pass_and_names_the_card():
    def floor_below_rate(parameters, profile):
        return [] if parameters["floor"] < parameters["rate"] else ["floor must lie below rate"]

    register_kind(RuleKind("wear_rate", ("rate", "floor"), check=floor_below_rate))
    try:
        assert card_problems({"kind": "wear_rate", "parameters": {"rate": 1, "floor": 2}}) == ["card (wear_rate): floor must lie below rate"]
        assert card_problems({"kind": "wear_rate", "parameters": {"rate": 1}}) == ["card (wear_rate): missing parameters ['floor']"]
        assert card_problems({"kind": "wear_rate", "parameters": {"rate": 3, "floor": 2}}) == []
    finally:
        unregister_kind("wear_rate")


def test_a_plugin_registers_by_name_and_cannot_shadow_or_remove_a_built_in_kind(plugin):
    assert kind_of("wear_rate") is plugin and kinds()[-1] == "wear_rate"
    assert card_problems({"kind": "wear_rate", "parameters": {"rate": 0.1, "floor": 0}}, {"evidence_kind": "recorded_levels"}) == []
    with pytest.raises(ValueError, match="already declared"):
        register_kind(RuleKind("wear_rate", ("rate",)))
    with pytest.raises(ValueError, match="already declared"):
        register_kind(RuleKind("schedule", ("mode",)))
    with pytest.raises(ValueError, match="cannot be removed"):
        unregister_kind("schedule")
    with pytest.raises(TypeError, match="keys"):
        register_kind(type("Partial", (), {"name": "partial"})())
    assert read_card(EpisodeConfig().to_json(), plugin) is None  # a plugin without binds has no place in the configuration


def test_unregistering_a_plugin_removes_only_that_kind():
    register_kind(RuleKind("wear_rate", ("rate",)))
    unregister_kind("wear_rate")
    assert kind_of("wear_rate") is None and "wear_rate" not in kinds()
    unregister_kind("wear_rate")  # a second removal is a no-op


def test_the_stopped_rate_key_is_the_engine_key():
    assert rules.STOPPED == STOPPED


def test_every_kind_with_a_place_in_the_configuration_declares_its_off_parameters(hyd_1_data):
    profile = AssetProfile.from_json(hyd_1_data)
    for name in kinds():
        kind = kind_of(name)
        if kind.binds:
            assert tuple(kind.off(profile)) == kind.keys, name
        else:
            assert kind.off is None, name


def test_the_visit_charge_lands_after_the_repair_charge_and_the_stopped_rate_last():
    fields = {"prices": {"read": {"tier1": 1}, "parts": {"pump": 3}, "repair_charge": 20, "fix_kind": {"pump": "replace"}}}
    apply_card(fields, {"kind": "visits", "parameters": {"charge": 15}})
    assert list(fields["prices"]) == ["read", "parts", "repair_charge", "visit", "fix_kind"] and fields["prices"]["visit"] == 15
    apply_card(fields, {"kind": "downtime", "parameters": {"stopped": 5, "rates": {"pump": {"0": 0, "2": 2}}}})
    assert fields["downtime_rate"] == {"pump": {"0": 0, "2": 2}, STOPPED: 5} and list(fields["downtime_rate"])[-1] == STOPPED


def test_a_card_writes_its_parameters_in_the_order_of_its_kind():
    fields = {}
    apply_card(fields, {"kind": "hazard", "parameters": {"collateral": 0.0, "depth_factor": 1.0, "base_rate": 0.01, "mode": "scheduled"}})
    assert list(fields["hazard"]) == list(kind_of("hazard").keys)
    apply_card(fields, {"kind": "dependencies", "parameters": {"priority": {"a": 1}, "dependencies": {"a": []}}})
    assert fields["dependencies"] == {"a": []} and fields["priority"] == {"a": 1}


def test_reading_a_card_back_from_the_configuration_inverts_writing_it():
    config = EpisodeConfig().to_json()
    for name in kinds():
        kind = kind_of(name)
        parameters = read_card(config, kind)
        if not kind.binds:
            assert parameters is None
            continue
        fields = {"prices": {k: v for k, v in config["prices"].items() if k != "visit"}}
        apply_card(fields, {"kind": name, "parameters": parameters})
        for bind in kind.binds:
            assert fields[bind.field] == config[bind.field], name
