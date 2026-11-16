"""The six big layers of the environment: the registry that gives every field of EpisodeConfig one owner,
the small switches of each layer with their off and on values, the composition of a preset from big layers and switch
settings, the switches that need a lower layer, and the rows of the ladder, each the row below it with one more big layer
on."""
import json
import re
from dataclasses import fields

import pytest

from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.reference_policies.policies import DoNothingPolicy, OraclePolicy, run_episode
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv

LADDER = ("base", "priced", "real_time", "advanced_real_time", "repair_rules", "deterioration")
# the small switches of every big layer, which the registry of the layers must name
SWITCH_NAMES = {"base": (), "priced": ("read_prices", "part_prices", "repair_charge", "budget", "misdiagnosis"),
                "real_time": ("clock_mode", "think_seconds_per_cycle", "time_tools"),
                "advanced_real_time": ("read_per_cycle", "lead_time", "repair_downtime", "downtime_rate"),
                "repair_rules": ("dependencies", "priority", "visit", "emergency", "fix_history"),
                "deterioration": ("deterioration", "collateral", "failure")}
# a setting away from the off value for the switches whose on value equals their off value for hyd_1
SETTINGS = {"think_seconds_per_cycle": 30, "collateral": 0.1}
SEED = 3


def _paths():
    """Every field of EpisodeConfig, and every key of the two tables whose keys belong to different layers."""
    default = EpisodeConfig().to_json()
    return {f"{f.name}.{key}" for f in fields(EpisodeConfig) if f.name in layers.SPLIT for key in default[f.name]} | \
        {f.name for f in fields(EpisodeConfig) if f.name not in layers.SPLIT}


def _value(data, path):
    field, _, key = path.partition(".")
    return data[field][key] if key else data.get(field)  # the JSON leaves seconds_per_token out where it is None


def _changed(a, b):
    """The paths of the registry where two JSONs differ, with the differing keys of a whole field that is a mapping, so that
    a switch that sets one key of a field shows as that key."""
    out = {p for p in _paths() if _value(a, p) != _value(b, p)}
    return out | {f"{p}.{k}" for p in out if "." not in p and isinstance(a.get(p), dict) for k in a[p] if a[p][k] != b[p].get(k)}


def _switches():
    return [(layer, name, switch) for layer in layers.LAYERS for name, switch in layer.SWITCHES.items()]


def _setting(name):
    return SETTINGS.get(name, "on")


# the switches whose cost unit the loader refuses alone, a priced read with free work orders or a hidden ledger under a cost in
# calls, each with the part prices, so that a composition of one switch keeps a cost unit that fits it
UNIT_NEEDS = {"read_prices": "part_prices", "downtime_rate": "part_prices", "failure": "part_prices", "misdiagnosis": "part_prices"}


def _alone(name):
    """The settings that turn one switch on over the base: the switch, with the lower switches it needs and the part prices
    where its cost unit needs them."""
    switch = layers.SWITCHES[name].SWITCHES[name]
    extra = {UNIT_NEEDS[name]: "on"} if name in UNIT_NEEDS else {}
    return {**extra, **{need: "on" for need, _reason in switch.needs}, name: _setting(name)}


def _owned(paths, switch_paths):
    """Whether every changed path lies under a path of the switch."""
    return all(any(p == s or p.startswith(f"{s}.") or s.startswith(f"{p}.") for s in switch_paths) for p in paths)


NEEDED = {need for _layer, _name, switch in _switches() for need, _reason in switch.needs}


def test_the_six_big_layers_follow_the_rows_of_the_environment_figure():
    assert tuple(layer.NAME for layer in layers.LAYERS) == layers.LADDER == LADDER
    assert not layers.LAYERS[0].SWITCHES  # the base is always on and has no switch


def test_every_layer_holds_the_small_switches_of_the_users_layer_sheet():
    assert {layer.NAME: tuple(layer.SWITCHES) for layer in layers.LAYERS} == SWITCH_NAMES


def test_every_field_of_the_configuration_has_exactly_one_owner():
    owned = [path for layer in layers.LAYERS for path in layer.FIELDS]
    assert len(owned) == len(set(owned)), "a path is owned twice"
    assert set(owned) == _paths(), "a field or key is unowned, or a path names no field"
    assert layers.OWNER == {path: layer.NAME for layer in layers.LAYERS for path in layer.FIELDS}
    for layer in layers.LAYERS[1:]:
        covered = [path for switch in layer.SWITCHES.values() for path in switch.paths]
        assert len(covered) == len(set(covered)), layer.NAME  # no path is set by two switches
        assert {path.split(".")[0] if path.split(".")[0] not in layers.SPLIT else path for path in covered} == set(layer.FIELDS), layer.NAME
        default = EpisodeConfig().to_json()
        for field in {path.split(".")[0] for path in covered if "." in path and path.split(".")[0] not in layers.SPLIT}:
            keys = {path.split(".")[1] for path in covered if path.startswith(f"{field}.")}
            assert keys == set(default[field]), (layer.NAME, field)  # a field set key by key has every key under a switch


def test_the_horizon_and_the_step_cap_belong_to_the_base():
    assert {"horizon", "max_steps"} <= set(layers.base.FIELDS)
    assert all((c.horizon, c.max_steps) == (60, 60) for c in (layers.compose({"layers": list(LADDER[1:k])}) for k in range(1, 7)))


def test_the_switch_names_are_unique_across_the_layers():
    names = [name for _layer, name, _switch in _switches()]
    assert len(names) == len(set(names))


def test_the_base_composition_holds_every_switch_at_its_off_value_and_the_full_one_at_its_on_value():
    profile = get_profile("hyd_1")
    base, full = (layers.compose({"layers": list(names)}).to_json() for names in ((), LADDER[1:]))
    for _layer, name, switch in _switches():
        for path, value in switch.off(profile).items():
            assert _value(base, path) == value, (name, path)
    for layer in layers.LAYERS[1:]:
        for name, switch in layer.SWITCHES.items():
            for path, value in switch.on(profile).items():
                assert _value(full, path) == value, (name, path)


def test_the_off_value_of_a_card_switch_is_the_off_parameters_of_the_rule_library():
    profile = get_profile("hyd_1")
    base = layers.compose({}).to_json()
    for card in ("fix_history", "emergency", "dependencies", "downtime", "schedule", "hazard", "failure"):
        kind = rules.kind_of(card)
        assert rules.read_card(base, kind) == kind.off(profile), card
    assert "external_replacement" not in rules.kinds()


def test_the_failure_card_of_layer_5_ends_the_episode_in_every_composition_while_the_asset_card_stays():
    card = get_profile("hyd_1").config_fields()["failure"]
    assert card == {"patience": 10, "strict": False, "penalty": 50}
    for names in (["priced", "deterioration"], list(LADDER[1:]), ["priced", "advanced_real_time", "deterioration"]):
        assert layers.compose({"layers": names}).failure == {"patience": 10, "strict": True, "penalty": 50}, names


@pytest.mark.parametrize("k", range(1, 6))
def test_turning_a_big_layer_off_on_its_ladder_row_gives_the_row_below_field_by_field(k):
    row, below = (layers.compose({"layers": list(LADDER[1:n + 1])}) for n in (k, k - 1))
    off = layers.without({"layers": list(LADDER[1:k + 1])}, LADDER[k])
    assert json.dumps(layers.compose(off).to_json()) == json.dumps(below.to_json())
    assert _owned(_changed(row.to_json(), below.to_json()), layers.LAYERS[k].FIELDS)


@pytest.mark.parametrize("k", range(1, 6))
def test_turning_every_small_switch_of_a_layer_off_on_its_ladder_row_gives_the_row_below_byte_for_byte(k):
    names = list(layers.LAYERS[k].SWITCHES)
    off = layers.compose({"layers": list(LADDER[1:k + 1]), "switches": {name: "off" for name in names}})
    below = layers.compose({"layers": list(LADDER[1:k])})
    assert json.dumps(off.to_json()) == json.dumps(below.to_json())


def test_without_drops_the_switch_settings_of_the_layer_it_turns_off():
    spec = {"layers": list(LADDER[1:]), "switches": {"failure": {"patience": 20}, "budget": 200}}
    off = layers.without(spec, "deterioration")
    assert off == {"layers": list(LADDER[1:5]), "switches": {"budget": 200}}


@pytest.mark.parametrize("name", [name for _layer, name, _switch in _switches()])
def test_every_small_switch_can_be_set_alone_over_the_base(name):
    owner = layers.SWITCHES[name]
    switch = owner.SWITCHES[name]
    base = layers.compose({}).to_json()
    alone = layers.compose({"switches": _alone(name)}).to_json()
    changed = _changed(base, alone)
    others = [*(need for need, _reason in switch.needs), *([UNIT_NEEDS[name]] if name in UNIT_NEEDS else [])]
    needed = [p for need in others for p in layers.SWITCHES[need].SWITCHES[need].paths]
    assert changed and _owned(changed, switch.paths + tuple(needed)), (name, changed)
    assert set(layers.layers_on(EpisodeConfig.from_json(alone))) == {owner.NAME, *(layers.SWITCHES[n].NAME for n in others)}


@pytest.mark.parametrize("name", [name for _layer, name, switch in _switches() if switch.needs])
def test_a_switch_without_the_switch_it_needs_is_refused_with_the_missing_layer(name):
    switch = layers.SWITCHES[name].SWITCHES[name]
    for need, _reason in switch.needs:
        missing = f"the small switch {name} of the big layer {layers.SWITCHES[name].NAME} needs the small switch {need} of the big layer {layers.SWITCHES[need].NAME}"
        others = {n: "on" for n, _r in switch.needs if n != need}
        for spec in ({"switches": {**others, name: "on"}}, {"layers": list(LADDER[1:]), "switches": {need: "off"}}):
            with pytest.raises(ValueError, match=re.escape(missing)):
                layers.compose(spec)


def test_the_dependencies_need_the_priorities_that_repair_a_dependency_first():
    with pytest.raises(ValueError, match="dependencies of the big layer repair_rules needs the small switch priority of the big layer repair_rules"):
        layers.compose({"switches": {"dependencies": "on"}})
    config = layers.compose({"switches": {"dependencies": "on", "priority": "on"}})
    assert config.dependencies == EpisodeConfig().dependencies and config.priority == EpisodeConfig().priority


def test_an_emergency_order_needs_the_lead_times_of_the_advanced_real_time_clock():
    assert layers.SWITCHES["emergency"].SWITCHES["emergency"].needs[0][0] == "lead_time"
    with pytest.raises(ValueError, match="needs the small switch lead_time of the big layer advanced_real_time"):
        layers.compose({"layers": ["priced", "real_time", "repair_rules"]})
    assert layers.compose({"layers": ["priced", "real_time", "repair_rules"], "switches": {"emergency": "off"}}).emergency_on() is False


@pytest.mark.parametrize("name", [name for _layer, name, _switch in _switches()])
def test_every_small_switch_can_be_set_off_alone_on_the_full_environment(name):
    owner = layers.SWITCHES[name]
    spec = {"layers": list(LADDER[1:]), "switches": {name: "off"}}
    if name in NEEDED:  # a switch that another one needs goes off only with it
        with pytest.raises(ValueError, match=f"needs the small switch {name} of the big layer {owner.NAME}"):
            layers.compose(spec)
        return
    full = layers.compose({"layers": list(LADDER[1:])}).to_json()
    assert _owned(_changed(full, layers.compose(spec).to_json()), owner.SWITCHES[name].paths)


@pytest.mark.parametrize("mode", ("none", "schedule", "hazard"))
def test_the_deterioration_mode_sets_the_schedule_and_the_hazard_cards_together(mode):
    config = layers.compose({"layers": list(LADDER[1:]), "switches": {"deterioration": mode}})
    assert (config.schedule["mode"], config.hazard["mode"]) == {"none": ("none", "scheduled"), "schedule": ("seeded", "scheduled"),
                                                               "hazard": ("none", "stochastic")}[mode]
    assert layers.deterioration.mode(config) == mode and config.hazard["collateral"] == 0.0


def test_the_collateral_chance_is_a_switch_of_its_own_at_0_in_every_composition_and_independent_of_the_mode():
    assert all(layers.compose({"layers": list(LADDER[1:k])}).hazard["collateral"] == 0.0 for k in range(1, 7))
    for mode in ("none", "schedule", "hazard"):
        for order in ((("collateral", 0.2), ("deterioration", mode)), (("deterioration", mode), ("collateral", 0.2))):
            config = layers.compose({"layers": list(LADDER[1:]), "switches": dict(order)})
            assert config.hazard["collateral"] == 0.2 and layers.deterioration.mode(config) == mode


def test_a_mapping_setting_overrides_the_parameters_of_its_switch():
    config = layers.compose({"layers": list(LADDER[1:]), "switches": {"failure": {"patience": 20},
                                                                      "deterioration": {"mode": "schedule", "first_drop": [5, 30]},
                                                                      "priority": {"cooler": 5}, "dependencies": {"pump": ["valve"]}}})
    assert config.failure == {"patience": 20, "strict": True, "penalty": 50}
    assert config.schedule == {**EpisodeConfig().schedule, "mode": "seeded", "first_drop": [5, 30]}
    assert config.priority == {**EpisodeConfig().priority, "cooler": 5} and config.dependencies == {**EpisodeConfig().dependencies, "pump": ["valve"]}
    hazard = layers.compose({"layers": list(LADDER[1:]), "switches": {"deterioration": {"mode": "hazard", "base_rate": 0.02}}}).hazard
    assert hazard == {"mode": "stochastic", "base_rate": 0.02, "depth_factor": 1.0, "collateral": 0.0}


@pytest.mark.parametrize("spec, problem", [
    ({"layers": ["stochastic"]}, "unknown big layer"),
    ({"layers": ["base"]}, "always on"),
    ({"layers": ["priced", "priced"]}, "more than once"),
    ({"switches": {"technician": "on"}}, "unknown small switch"),
    ({"switches": {"horizon": 40}}, "unknown small switch"),
    ({"switches": {"deterioration": "sometimes"}}, "deterioration"),
    ({"switches": {"deterioration": {"mode": "hazard", "collateral": 0.1}}}, "deterioration"),
    ({"switches": {"time_tools": 1}}, "time_tools"),
    ({"switches": {"think_seconds_per_cycle": {"scale": 0}}}, "think_seconds_per_cycle"),
    ({"switches": {"think_seconds_per_cycle": {"scale": -2}}}, "think_seconds_per_cycle"),
    ({"switches": {"think_seconds_per_cycle": {"scale": True}}}, "think_seconds_per_cycle"),
    ({"switches": {"think_seconds_per_cycle": {"scale": float("inf")}}}, "think_seconds_per_cycle"),
    ({"switches": {"think_seconds_per_cycle": {"factor": 2}}}, "think_seconds_per_cycle"),
    ({"switches": {"think_seconds_per_cycle": {"scale": 2, "seconds": 5}}}, "think_seconds_per_cycle"),
    ({"budget": 300}, "switches"),
    ({"clock_mode": "generation_time"}, "switches"),
    ({"time_tools": True}, "switches"),
    ({"external_events": {"mode": "none"}}, "external_events"),
])
def test_a_composition_names_known_layers_and_switches_and_sets_a_layer_field_only_through_a_switch(spec, problem):
    with pytest.raises(ValueError, match=problem):
        layers.compose(spec)


@pytest.mark.parametrize("scale", [0.5, 2, 0.25, 3])
def test_a_scale_sets_the_thinking_rate_to_a_multiple_of_the_asset_files_value(scale):
    """The thinking price conditions are a multiplier on the value of the asset file, which the pilot sets,
    never a fixed number of seconds, and the scaled rate passes the validation of the configuration."""
    rate = get_profile("hyd_1").config_fields()["think_seconds_per_cycle"]
    config = layers.compose({"layers": list(LADDER[1:]), "switches": {"think_seconds_per_cycle": {"scale": scale}}})
    assert config.think_seconds_per_cycle == scale * rate  # compose validates through EpisodeConfig.from_json
    full = layers.compose({"layers": list(LADDER[1:])})
    assert {k for k, v in config.to_json().items() if full.to_json()[k] != v} == {"think_seconds_per_cycle"}


def test_a_base_field_is_set_directly():
    config = layers.compose({"guidance_level": 2, "fault_rule": {"min_faulty": 1, "max_faulty": 4}, "horizon": 40})
    assert config.guidance_level == 2 and config.fault_rule == {"min_faulty": 1, "max_faulty": 4} and config.horizon == 40


@pytest.mark.parametrize("k", range(6))
def test_the_row_of_a_ladder_composition_is_its_ladder_name(k):
    config = layers.compose({"layers": list(LADDER[1:k + 1])})
    assert layers.row_of(config) == LADDER[k] and layers.layers_on(config) == LADDER[1:k + 1]


@pytest.mark.parametrize("k", range(6))
def test_the_time_tools_are_listed_with_the_clock_on_every_ladder_row_and_follow_their_switch_alone(k):
    config = layers.compose({"layers": list(LADDER[1:k + 1])})
    assert config.time_tools == layers.lists_tools("real_time", config) == config.has_clock() == (k >= 2)
    switched = layers.compose({"layers": list(LADDER[1:k + 1]), "switches": {"time_tools": "off" if k >= 2 else "on"}})
    assert layers.lists_tools("real_time", switched) == (k < 2) and switched.has_clock() == config.has_clock()


def test_the_repair_manual_is_listed_where_the_repair_rules_run_on_a_clock_of_layer_2_or_layer_3():
    full = list(LADDER[1:])
    assert layers.lists_tools("repair_rules", layers.compose({"layers": full}))
    assert layers.lists_tools("repair_rules", layers.compose({"layers": full, "switches": {"clock_mode": "off"}}))  # layer 3 runs the clock
    without_layer_2 = layers.compose({"layers": [n for n in full if n != "real_time"]})
    assert layers.lists_tools("repair_rules", without_layer_2) and without_layer_2.has_clock()
    no_clock = layers.compose({"layers": ["priced", "repair_rules"], "switches": {"emergency": "off"}})
    assert not no_clock.has_clock() and not layers.lists_tools("repair_rules", no_clock)
    assert not any(layers.lists_tools("repair_rules", layers.compose({"layers": list(LADDER[1:k])})) for k in range(1, 5))


def test_the_prices_keep_the_order_of_the_price_tables():
    assert list(layers.compose({"layers": ["priced", "repair_rules"], "switches": {"emergency": "off"}}).to_json()["prices"]) == [
        "read", "parts", "repair_charge", "visit", "misdiagnosis", "fix_kind"]
    assert list(EpisodeConfig().to_json()["prices"]) == ["read", "parts", "repair_charge", "visit", "misdiagnosis", "fix_kind"]


def test_the_outside_technician_has_left_the_configuration():
    assert "external_events" not in {f.name for f in fields(EpisodeConfig)}
    with pytest.raises(ValueError, match="external_events"):
        EpisodeConfig.from_json({**EpisodeConfig().to_json(), "external_events": {"mode": "none", "probability": 0.0, "window": [0, 0]}})


def test_no_outside_technician_replaces_a_component(world):
    """The reset draws no replacement, the cycle has no step for one and the log holds no replacement event, so a degraded
    component stays degraded until the agent repairs it."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.episode_clock import CYCLE_ORDER

    assert not [step for step in CYCLE_ORDER if "external" in step or "technician" in step]
    for name in ("base", "deterioration"):
        env = HydraulicRigEnv(*world, layers.compose({"layers": list(LADDER[1:LADDER.index(name) + 1])}))
        log = run_episode(env, DoNothingPolicy(), SEED)
        assert "external_events" not in log[0] and not hasattr(env.clock, "external")
        assert not [e for e in log if e["event"] == "external_replacement" or e.get("source") == "external"]


@pytest.mark.parametrize("engine", (HydraulicRigEnv, RecordedRigEnv))
@pytest.mark.parametrize("name", [name for _layer, name, _switch in _switches()])
def test_an_episode_plays_to_its_end_with_any_one_switch_set_alone(world, engine, name):
    config = layers.compose({"switches": _alone(name)})
    for policy in (lambda env: OraclePolicy(env), lambda env: DoNothingPolicy()):
        env = engine(*world, config)
        log = run_episode(env, policy(env), SEED)
        assert env.done and log[-1]["event"] == "end"


@pytest.mark.parametrize("k", range(1, 6))
def test_turning_every_switch_of_a_big_layer_off_gives_byte_identical_logs_of_the_row_below(world, k):
    off = layers.compose({"layers": list(LADDER[1:k + 1]), "switches": {name: "off" for name in layers.LAYERS[k].SWITCHES}})
    below = layers.compose({"layers": list(LADDER[1:k])})
    for seed in (0, SEED, 11):
        for make in (lambda env: OraclePolicy(env), lambda env: DoNothingPolicy()):
            logs = []
            for config in (off, below):
                env = HydraulicRigEnv(*world, config)
                logs.append(json.dumps(run_episode(env, make(env), seed)))
            assert logs[0] == logs[1], (k, seed)


def test_a_money_unit_with_a_free_work_order_is_refused():
    """A price above zero makes the cost unit money, and a work order that costs nothing would let the oracle policy, which reads
    nothing, pay nothing on a start with that fault. Priced reads alone are refused with the rule, and they load once the work
    orders have a price, the part prices, the repair charge or the visit charge."""
    free = ", ".join(EpisodeConfig().components)
    with pytest.raises(ValueError, match=re.escape(f"the cost unit is money, since a price is above zero, while a work order on {free} costs nothing")):
        layers.compose({"switches": {"read_prices": "on"}})
    for price in ("part_prices", "repair_charge", "visit"):
        assert layers.compose({"switches": {"read_prices": "on", price: "on"}}).cost_unit() == "money", price


def test_a_calls_unit_with_a_hidden_ledger_is_refused():
    """Every price at zero makes the cost unit calls, which leaves the hidden ledger out, so a downtime rate or the penalty of a
    failure rule that is on is refused, alone and in the composition of the advanced real-time clock with deterioration, and
    each loads once the work orders have a price."""
    for spec in ({"switches": {"downtime_rate": "on"}}, {"switches": {"failure": "on"}}, {"layers": ["advanced_real_time", "deterioration"]}):
        with pytest.raises(ValueError, match="the cost unit is calls, since every price is zero, while the downtime rates or the penalty"):
            layers.compose(spec)
        for price in ("part_prices", "repair_charge", "visit"):
            assert layers.compose({**spec, "switches": {**spec.get("switches", {}), price: "on"}}).cost_unit() == "money", (spec, price)


def test_under_every_single_switch_composition_the_oracle_has_a_cost_and_a_reward(world):
    """For every small switch set alone over the base with the switches it needs, a composition that the loader takes gives
    the oracle policy an episode whose cost is above zero and whose reward is a number, so the normalized cost of any agent is
    defined. The loader refuses the three switches whose cost unit does not fit them alone, and no other."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, reward, verify

    played, refused = [], []
    for _layer, name, switch in _switches():
        try:
            config = layers.compose({"switches": {**{need: "on" for need, _reason in switch.needs}, name: _setting(name)}})
        except ValueError as problem:
            assert "the cost unit is" in str(problem), name
            refused.append(name)
            continue
        cost = oracle_cost(HydraulicRigEnv(*world, config), SEED)
        env = HydraulicRigEnv(*world, config)
        log = run_episode(env, OraclePolicy(env), SEED)
        verdict = verify(log, released_from_env(env), env.oracle(), cost)
        assert cost > 0 and reward(verdict, config.horizon) is not None, name
        played.append(name)
    assert sorted(refused) == sorted(UNIT_NEEDS) and played
