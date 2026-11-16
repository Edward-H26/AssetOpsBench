"""EpisodeConfig (CausalDiagnosis/environment/Causal_Diagnosis_Environment/episode_config.py): the defaults validate and round trip through JSON, a bad configuration
names every problem at once, and the degradation order of the clock is the one the design states."""
import json

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order, worst_level
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig, default_durations, default_prices, default_priority, default_sensor_groups
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS

DESIGN_ORDER = {"cooler": [100, 20, 3], "valve": [100, 90, 80, 73], "pump": [0, 1, 2], "accumulator": [130, 115, 100, 90]}
BAD = {  # one kind of problem each, with the wording the error must carry
    "negative read": ({"prices": {**default_prices(), "read": {"tier1": -1, "tier2": 3, "tier3": 10}}}, "read prices of ['tier1'] are not non-negative"),
    "group without a price": ({"prices": {**default_prices(), "read": {"tier1": 1, "tier2": 3}}}, "sensor groups ['tier3'] have no read price"),
    "priced group without sensors": ({"prices": {**default_prices(), "read": {**default_prices()["read"], "tier4": 2}}}, "groups ['tier4'] that have no sensors"),
    "component without a part": ({"prices": {**default_prices(), "parts": {"cooler": 20, "valve": 20, "pump": 30}}}, "part prices of ['accumulator']"),
    "negative repair charge": ({"prices": {**default_prices(), "repair_charge": -1}}, "repair_charge"),
    "string price": ({"prices": {**default_prices(), "parts": {**default_prices()["parts"], "pump": "30"}}}, "part prices of ['pump']"),
    "bool price": ({"prices": {**default_prices(), "repair_charge": True}}, "repair_charge"),
    "infinite budget": ({"budget": float("inf")}, "budget must be null, for an episode without a budget stop, or a positive finite number"),
    "zero horizon": ({"horizon": 0}, "horizon must be a positive integer"),
    "component with one level": ({"levels": {**EpisodeConfig().levels, "pump": [0]}}, "levels of pump must hold a second level beside the healthy one"),
    "no group of every channel": ({"sensor_groups": {g: default_sensor_groups()[g] for g in ("tier1", "tier2")},
                                   "prices": {**default_prices(), "read": {"tier1": 1, "tier2": 3}}},
                                  "sensor_groups must hold one group of every channel of hyd_1, the group that a read of every channel is charged to"),
    "float steps": ({"max_steps": 2.0}, "max_steps must be a positive integer"),
    "fault rule below zero": ({"fault_rule": {"min_faulty": -1, "max_faulty": 2}}, "0 <= min_faulty <= max_faulty"),
    "fault rule reversed": ({"fault_rule": {"min_faulty": 2, "max_faulty": 1}}, "0 <= min_faulty <= max_faulty"),
    "advance without time": ({"durations": {**default_durations(), "advance_per_cycle": 0}}, "advance_per_cycle must be 1"),
    "advance by two": ({"durations": {**default_durations(), "advance_per_cycle": 2}}, "advance_per_cycle must be 1"),
    "negative read duration": ({"durations": {**default_durations(), "read_per_cycle": -1}}, "read_per_cycle"),
    "lead time missing": ({"durations": {**default_durations(), "lead_time": {"cooler": 3}}}, "durations.lead_time of"),
    "rate without a level": ({"downtime_rate": {"cooler": {"100": 0, "20": 1}, "valve": {"100": 0, "90": 1, "80": 2, "73": 3},
                                                "pump": {"0": 0, "1": 1, "2": 2}, "accumulator": {"130": 0, "115": 1, "100": 2, "90": 3}, "stopped": 5}},
                             "downtime_rate of cooler must have exactly the keys"),
    "unknown clock mode": ({"clock_mode": "wall"}, "clock_mode must be one of"),
    "guidance level": ({"guidance_level": 4}, "guidance_level must be one of"),
    "schedule mode": ({"schedule": {"mode": "random", "components": 1, "first_drop": [5, 40], "interval": 10}}, "schedule.mode"),
    "schedule range": ({"schedule": {"mode": "seeded", "components": 1, "first_drop": [40, 5], "interval": 10}}, "schedule.first_drop"),
    "too many scheduled components": ({"schedule": {"mode": "seeded", "components": 5, "first_drop": [5, 40], "interval": 10}}, "schedule.components"),
    "patience": ({"failure": {"patience": -1, "strict": False, "penalty": 50}}, "failure.patience"),
    "strict not bool": ({"failure": {"patience": 10, "strict": "no", "penalty": 50}}, "failure.strict"),
    "start outside the rule": ({"start": {"combination": [100, 100, 0, 130], "entry_offset": 0}}, "start.combination must satisfy the fault rule"),
    "start with a bad level": ({"start": {"combination": [100, 100, 7, 130], "entry_offset": 0}}, "start.combination must give one valid level"),
    "allowed with a bad combination": ({"allowed": [[3, 100, 0, 130], [1, 2, 3]]}, "allowed must be null or"),
    "anchor without a date": ({"anchors": {"date0": "July 2022"}}, "anchors must be null or"),
    "verifier scope": ({"verifier_scope": "reads"}, "verifier_scope must be one of"),
    "unknown sensor": ({"sensor_groups": {"tier1": ["TS1", "XS9"]}, "prices": {**default_prices(), "read": {"tier1": 1}}}, "sensor group tier1 must be"),
    "negative visit charge": ({"prices": {**default_prices(), "visit": -1}}, "prices.visit must be"),
    "unknown fix kind": ({"prices": {**default_prices(), "fix_kind": {**default_prices()["fix_kind"], "pump": "rebuild"}}}, "prices.fix_kind must name"),
    "priority without a component": ({"priority": {"valve": 1, "accumulator": 2, "cooler": 3}}, "priority must map every component"),
    "dependency with a larger priority number": ({"priority": {**default_priority(), "valve": 5}}, "dependencies ['valve'] of pump need a smaller priority number"),
    "dependency on itself": ({"dependencies": {"cooler": ["cooler"], "valve": [], "pump": [], "accumulator": []}}, "dependencies of cooler must be"),
    "dependencies without a component": ({"dependencies": {"pump": ["valve"]}}, "dependencies must name every component once"),
    "emergency surcharge below one": ({"emergency": {"lead": 1, "surcharge": 0.5, "hold": 10}}, "emergency.surcharge"),
    "negative hold": ({"emergency": {"lead": 1, "surcharge": 2.0, "hold": -1}}, "emergency.hold"),
    "history probability": ({"fix_history": {"healthy_probability": 1.5, "healthy_ago": [5, 25], "degraded_probability": 0.3, "degraded_ago": [30, 80]}},
                            "fix_history.healthy_probability"),
    "history range reversed": ({"fix_history": {"healthy_probability": 0.4, "healthy_ago": [5, 25], "degraded_probability": 0.3, "degraded_ago": [80, 30]}},
                               "fix_history.degraded_ago"),
    "hazard mode": ({"hazard": {"mode": "random", "base_rate": 0.01, "depth_factor": 1.0, "collateral": 0.0}}, "hazard.mode must be one of"),
    "hazard rate above one": ({"hazard": {"mode": "stochastic", "base_rate": 1.5, "depth_factor": 1.0, "collateral": 0.0}}, "hazard.base_rate"),
    "hazard over a schedule": ({"hazard": {"mode": "stochastic", "base_rate": 0.01, "depth_factor": 1.0, "collateral": 0.0},
                                "schedule": {"mode": "seeded", "components": 1, "first_drop": [5, 40], "interval": 10}}, "schedule.mode must be none"),
}


def test_the_defaults_validate_and_round_trip_through_json():
    config = EpisodeConfig()
    data = json.loads(json.dumps(config.to_json()))
    assert EpisodeConfig.from_json(data) == config
    assert (config.budget, config.horizon, config.max_steps) == (300, 60, 60)
    assert config.schedule["mode"] == "none"
    assert (config.clock_mode, config.guidance_level, config.verifier_scope, config.diagnostics_server) == ("declared", 1, "writes", True)
    assert config.components == list(COMPONENTS) and config.fault_rule == {"min_faulty": 1, "max_faulty": 2}
    assert set(config.prices["read"]) == set(config.sensor_groups) == {"tier1", "tier2", "tier3"}
    assert config.read_price("tier2", 4) == 4 * config.prices["read"]["tier2"]
    assert config.order_price("pump") == config.prices["parts"]["pump"] + config.prices["repair_charge"]
    assert config.level_rate("valve", 80) == 2 and config.level_rate("pump", 0) == 0


def test_a_stored_configuration_is_validated_as_one_built_in_code():
    """A stored configuration keeps every key it holds and is validated like one built in code, so a price table without the
    visit charge is refused whichever way it arrives."""
    stored = EpisodeConfig().to_json()
    del stored["prices"]["visit"]
    with pytest.raises(ValueError, match="prices.visit"):
        EpisodeConfig.from_json(stored)
    with pytest.raises(ValueError, match="prices.visit"):
        EpisodeConfig(prices={k: v for k, v in default_prices().items() if k != "visit"})
    stored["prices"]["visit"] = 0
    assert EpisodeConfig.from_json(stored).prices["visit"] == 0 and "fix_kind" in EpisodeConfig.from_json(stored).prices


def test_from_json_refuses_unknown_keys_and_keeps_lists():
    data = EpisodeConfig(allowed=[(3, 100, 0, 130)]).to_json()
    assert data["allowed"] == [[3, 100, 0, 130]]
    with pytest.raises(ValueError, match="unknown episode config keys"):
        EpisodeConfig.from_json({**data, "tiers": 3})


@pytest.mark.parametrize("name", list(BAD))
def test_a_bad_configuration_names_the_problem(name):
    overrides, problem = BAD[name]
    with pytest.raises(ValueError) as error:
        EpisodeConfig(**overrides)
    assert problem in str(error.value)


def test_every_problem_is_listed_at_once():
    with pytest.raises(ValueError) as error:
        EpisodeConfig(budget=0, horizon=0, clock_mode="x")
    text = str(error.value)
    assert "budget must be" in text and "horizon must be" in text and "clock_mode must be" in text


def test_the_default_dependencies_validate_and_a_dependency_with_a_larger_priority_number_is_refused():
    config = EpisodeConfig()
    assert config.depends_on("pump") == ["valve", "accumulator"] and all(not config.depends_on(c) for c in ("cooler", "valve", "accumulator"))
    assert all(config.priority[d] < config.priority[c] for c in COMPONENTS for d in config.depends_on(c))
    assert EpisodeConfig(dependencies={}).depends_on("pump") == []  # the empty mapping switches the rule off, an ablation
    with pytest.raises(ValueError, match=r"dependencies \['valve', 'accumulator'\] of pump need a smaller priority number than pump"):
        EpisodeConfig(priority={"valve": 4, "accumulator": 4, "cooler": 1, "pump": 2})
    assert config.order_price("pump", "emergency") == config.order_price("pump") * config.emergency["surcharge"]
    assert config.lead_time("pump", "emergency") == config.emergency["lead"] and config.lead_time("pump") == config.durations["lead_time"]["pump"]


def test_the_degradation_order_is_the_one_the_design_states():
    config = EpisodeConfig()
    assert {c: degradation_order(config, c) for c in COMPONENTS} == DESIGN_ORDER
    assert {c: worst_level(config, c) for c in COMPONENTS} == {"cooler": 3, "valve": 73, "pump": 2, "accumulator": 90}


def test_anchored_dates_count_one_recorded_cycle_per_day():
    config = EpisodeConfig(anchors={"date0": "2022-07-01"})
    assert config.date_of(0) == "2022-07-01" and config.date_of(31) == "2022-08-01"
    assert EpisodeConfig().date_of(5) is None


def test_a_fault_rule_from_zero_is_valid_and_admits_the_healthy_start():
    """A fault rule may start at zero faulty components, which admits the combination with every component at its healthy
    level, the start of the control, while the default rule of the asset file stays as it is."""
    healthy = [EpisodeConfig().healthy[c] for c in COMPONENTS]
    control = EpisodeConfig(fault_rule={"min_faulty": 0, "max_faulty": 0})
    assert control.within_fault_rule(healthy) and EpisodeConfig(fault_rule={"min_faulty": 0, "max_faulty": 2}).within_fault_rule(healthy)
    assert not EpisodeConfig().within_fault_rule(healthy) and EpisodeConfig().fault_rule["min_faulty"] >= 1
    assert EpisodeConfig(start={"combination": healthy, "entry_offset": 0}, fault_rule={"min_faulty": 0, "max_faulty": 0}).start["combination"] == healthy
