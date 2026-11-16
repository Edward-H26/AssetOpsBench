"""The presets of the environment rows: base.json, priced.json, real_time.json,
advanced_real_time.json, repair_rules.json and deterioration.json each name the big layers of their row, the row below
with one more big layer, so turning the last layer off gives the preset below field by field and episode by episode. They
leave every rule card they do not select at the off value of the rule library, draw the hidden start of static.json for
every seed, move the clock as their row states and brief only the rules of the cards they select. scheduled.json, the arm
of a small switch ablation, is the deterioration row with the fixed schedule of level drops in place of the random hazard
rates. Every preset holds the off value of each process card its clock draws nothing from."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.utils.paths import preset_files, preset_path  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, apply_price_table, load_config, load_price_table
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.experiments.table_definitions.experiment_tables import TOKEN_PRESETS
from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.datasets.dataset_loader.profile_data import CONFIG_KINDS
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig, default_downtime_rate, default_durations, default_prices
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.reference_policies.policies import DeclarePolicy, DoNothingPolicy, OraclePolicy, RandomPolicy, run_episode

ROWS = ("base", "priced", "real_time", "advanced_real_time", "repair_rules", "deterioration")
ARMS = ("scheduled",)  # the arm of a small switch ablation, the preset beside the rows of the environment table
PRESETS = sorted(p.stem for p in preset_files(CONFIGS).values())
REPAIR_CARDS = {"dependencies", "visits", "fix_history", "emergency"}
SELECTED = {"advanced_real_time": {"downtime"}, "repair_rules": {"downtime"} | REPAIR_CARDS,
            "deterioration": {"downtime", "failure", "hazard"} | REPAIR_CARDS,
            "scheduled": {"downtime", "failure", "schedule"} | REPAIR_CARDS}
POLICIES = {"oracle": lambda env, seed: OraclePolicy(env), "do_nothing": lambda env, seed: DoNothingPolicy(),
            "random": lambda env, seed: RandomPolicy(seed), "declare": lambda env, seed: DeclarePolicy()}
# the presets of the small switch ablations, all six big layers with one setting changed,
# and the switches each names: the action durations of layer 3 take the emergency order with them, which needs their lead
# times, and the two thinking price presets scale the thinking rate of the asset file
THINKING_PRICE = {"deterioration_thinking_price_double": 0.5, "deterioration_thinking_price_half": 2}
SWITCH_PRESETS = {"deterioration_thinking_off": {"clock_mode"}, "deterioration_mode_none": {"deterioration"},
                   "deterioration_failure_off": {"failure"}, "scheduled": {"deterioration"},
                   "deterioration_durations_off": {"read_per_cycle", "lead_time", "repair_downtime", "emergency"},
                   **dict.fromkeys(THINKING_PRICE, {"think_seconds_per_cycle"})}
# every supporting ablation that changes the environment, with the row it is paired with: those presets, the price
# table of En6 on all six layers, and on the base row the guidance levels of En7 and the verifier scope of En4
SUPPORTING_ABLATIONS = (*(("deterioration", "preset", name) for name in SWITCH_PRESETS), ("deterioration", "prices", "full_tier_high"),
           ("base", "fields", {"guidance_level": 1}), ("base", "fields", {"guidance_level": 3}), ("base", "fields", {"verifier_scope": "all"}))
PROCESS_CARDS = {  # whether the clock draws anything from the card, by the rules of episode_clock.py
    "schedule": lambda card: card["mode"] != "none" and card["components"] > 0,
    "hazard": lambda card: (card["mode"] == "stochastic" and card["base_rate"] > 0) or card["collateral"] > 0,
}
needs_data = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")


def _json(name):
    return load_config(name)[1].to_json()


def _preset_file(name):
    return json.loads(preset_path(name).read_text(encoding="utf-8"))


def _changed(a, b):
    return {k for k in a if a[k] != b[k]}


@pytest.fixture(scope="module")
def world():
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, cycles, blocks = load_world()
    return cycles, blocks


def _env(world, name):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    return HydraulicRigEnv(*world, load_config(name)[1])


def test_each_ladder_preset_names_the_big_layers_of_its_row():
    """One preset per row of the ladder, named after the row, each naming its big layers above the base, so
    the row of a big layer is the row below it with that layer on, and no ladder preset sets a small switch."""
    assert ROWS == layers.LADDER
    for k, name in enumerate(ROWS):
        preset = _preset_file(name)
        assert preset["layers"] == list(ROWS[1:k + 1]) and "switches" not in preset, name
    assert "stochastic" not in PRESETS  # the former stochastic row is the deterioration row


@pytest.mark.parametrize("k", range(1, 6))
def test_turning_the_last_big_layer_of_a_ladder_preset_off_gives_the_preset_below(k):
    off = layers.compose(layers.without(_preset_file(ROWS[k]), ROWS[k]))
    assert json.dumps(off.to_json()) == json.dumps(_json(ROWS[k - 1]))


@needs_data
@pytest.mark.parametrize("k", range(1, 6))
def test_turning_the_last_big_layer_off_plays_byte_identical_episodes_of_the_preset_below(world, k):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    configs = (layers.compose(layers.without(_preset_file(ROWS[k]), ROWS[k])), load_config(ROWS[k - 1])[1])
    for policy, make in POLICIES.items():
        for seed in (0, 3, 11):
            logs = [json.dumps(run_episode(env, make(env, seed), seed)) for env in (HydraulicRigEnv(*world, c) for c in configs)]
            assert logs[0] == logs[1], (ROWS[k], policy, seed)


def _registry_changes(a, b):
    """The paths of the layer registry where two configurations differ, with the differing keys of a whole field that is a
    mapping, so that a switch that sets one key of a field shows as that key."""
    a, b = a.to_json(), b.to_json()

    def value(data, path):  # a field that the JSON leaves out while it is None, seconds_per_token outside its mode, reads None
        field, _, key = path.partition(".")
        return data[field][key] if key else data.get(field)

    changed = {p for p in layers.OWNER if value(a, p) != value(b, p)}
    return changed | {f"{p}.{k}" for p in changed if "." not in p and isinstance(a[p], dict) for k in a[p] if a[p][k] != b[p].get(k)}


def _within(changed, allowed):
    return all(any(p == s or s.startswith(f"{p}.") or p.startswith(f"{s}.") for s in allowed) for p in changed)


def _owner(path):
    return layers.OWNER.get(path) or layers.OWNER[path.partition(".")[0]]


def _task_tools(config):
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract

    return set(layer_contract.tools_for(layer_contract.TASK_SERVER, config))


def _tools_of_layer(name):
    """The task tools a big layer adds, by the rows of the contract of the AssetOpsBench rows."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract

    return {tool for tool, entry in layer_contract.entries(layer_contract.TASK_SERVER).items() if entry["row"] == name}


@pytest.mark.parametrize("k", range(1, 6))
def test_a_ladder_row_differs_from_the_row_below_in_the_fields_and_tools_of_its_big_layer_alone(k):
    """No step of the ladder changes two things at once, so a row differs
    from the row below in the paths of its big layer and the task tools that layer adds, and in nothing else."""
    row, below = load_config(ROWS[k])[1], load_config(ROWS[k - 1])[1]
    changed = _registry_changes(row, below)
    assert changed and {_owner(p) for p in changed} == {ROWS[k]}, changed
    assert _task_tools(below) <= _task_tools(row) and _task_tools(row) - _task_tools(below) == _tools_of_layer(ROWS[k])


@pytest.mark.parametrize("row, kind, change", SUPPORTING_ABLATIONS)
def test_a_supporting_ablation_changes_one_switch_or_field_of_the_row_it_is_paired_with(row, kind, change):
    """Each composition of the supporting ablations differs from the row it is paired with
    in the paths of the switches it names and in each of them, one switch or the durations of layer 3 with the emergency
    order that needs them, of the price table or of the base field it changes, and lists the same task tools."""
    paired, preset = load_config(row)[1], _preset_file(row)
    if kind == "preset":
        ablated, named = load_config(change)[1], _preset_file(change)
        assert named["layers"] == list(ROWS[1:]) and set(named["switches"]) == SWITCH_PRESETS[change], change
        allowed = {path for name in named["switches"] for path in layers.SWITCHES[name].SWITCHES[name].paths}
    elif kind == "prices":
        ablated, allowed = apply_price_table(paired, load_price_table(change)), {"prices.read"}
    else:
        ablated, allowed = layers.compose({**preset, **change}), set(change)
    changed = _registry_changes(paired, ablated)
    assert changed and _within(changed, allowed), (row, change, changed)
    if kind == "preset":
        assert all(any(_within({p}, layers.SWITCHES[name].SWITCHES[name].paths) for p in changed) for name in named["switches"]), change
    assert _task_tools(ablated) == _task_tools(paired)


def test_each_row_differs_from_the_row_below_only_in_the_keys_it_adds():
    base, priced, real_time, advanced, repair, deterioration = (_json(n) for n in ROWS)
    assert _changed(base, priced) == {"prices", "budget"}
    assert _changed(priced, real_time) == {"clock_mode", "time_tools"}
    assert _changed(real_time, advanced) == {"durations", "downtime_rate"}
    # the priced row adds the default read, part and repair prices and the budget, the visit charge stays with the repair rules
    assert {k: v for k, v in priced["prices"].items() if k != "visit"} == {k: v for k, v in default_prices().items() if k != "visit"}
    assert base["prices"]["visit"] == priced["prices"]["visit"] == 0 and priced["budget"] == EpisodeConfig().budget
    # the misdiagnosis price is the default visit charge wherever prices are on and zero on the free base row
    assert base["prices"]["misdiagnosis"] == 0 and priced["prices"]["misdiagnosis"] == default_prices()["visit"] == 15
    assert all(_json(name)["prices"]["misdiagnosis"] == 15 for name in ROWS[2:] + ARMS + ("static", "timed", "anchored", "static_zero_durations"))
    assert (priced["clock_mode"], real_time["clock_mode"]) == ("declared", "generation_time")
    assert advanced["durations"] == default_durations() and advanced["downtime_rate"] == default_downtime_rate()
    # the failure card belongs to big layer 5, so it stays off until the deterioration row
    assert advanced["failure"] == repair["failure"] == base["failure"] and not load_config("repair_rules")[1].failure_on()
    # the repair rules row switches on the four repair cards at the defaults of EpisodeConfig, the visit charge among the
    # prices, and the deterioration row adds the hazard at the default rates and the failure card with the strict end
    defaults = EpisodeConfig().to_json()
    assert _changed(advanced, repair) == {"prices", "dependencies", "priority", "emergency", "fix_history"}
    assert all(repair[k] == defaults[k] for k in ("dependencies", "priority", "emergency", "fix_history"))
    assert _changed(advanced["prices"], repair["prices"]) == {"visit"} and repair["prices"]["visit"] == defaults["prices"]["visit"]
    assert _changed(repair, deterioration) == {"hazard", "failure"} and deterioration["hazard"] == {**defaults["hazard"], "mode": "stochastic"}
    assert deterioration["failure"] == {**defaults["failure"], "strict": True}
    # the scheduled arm sits one deterioration mode from the deterioration row, the seeded schedule card of timed.json
    # in place of the hazard card, which takes the off value of the repair rules row, so the arm reads the separability
    # labels of the deterioration row
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability

    scheduled = _json("scheduled")
    assert _changed(deterioration, scheduled) == {"hazard", "schedule"} and _changed(repair, scheduled) == {"schedule", "failure"}
    assert scheduled["schedule"] == _json("timed")["schedule"] and scheduled["hazard"] == repair["hazard"]
    labelled = [separability.rules_of(separability.static_config(load_config(n)[1])) for n in ("deterioration", "scheduled")]
    assert labelled[0] == labelled[1]


def test_the_base_row_is_free_and_has_no_budget_and_no_declared_time():
    base = load_config("base")[1]
    prices = [*base.prices["read"].values(), *base.prices["parts"].values(), base.prices["repair_charge"], base.prices["visit"]]
    assert set(prices) == {0} and base.budget is None and base.clock_mode == "declared"
    durations = [base.durations["read_per_cycle"], *base.durations["lead_time"].values(), *base.durations["repair_downtime"].values()]
    assert set(durations) == {0} and base.durations["advance_per_cycle"] == 1  # a wait still lasts the cycles it waits


@pytest.mark.parametrize("name", ROWS + ARMS)
def test_every_row_leaves_the_cards_it_does_not_select_at_their_off_value(name):
    profile, config = get_profile("hyd_1"), _json(name)
    selected = SELECTED.get(name, set())
    for kind_name in CONFIG_KINDS:
        kind = rules.kind_of(kind_name)
        assert (rules.read_card(config, kind) == kind.off(profile)) is (kind_name not in selected), kind_name


@pytest.mark.parametrize("kind_name", sorted(PROCESS_CARDS))
@pytest.mark.parametrize("name", PRESETS)
def test_a_process_card_reads_its_off_value_wherever_the_clock_draws_nothing_from_it(name, kind_name):
    """The processes block publishes the schedule and hazard cards as read_card reads them, so a
    preset whose clock draws nothing from a card holds the off value of the rule library and publishes no parameter that
    it does not run."""
    kind = rules.kind_of(kind_name)
    card = rules.read_card(_json(name), kind)
    assert (card == kind.off(get_profile("hyd_1"))) is not PROCESS_CARDS[kind_name](card)


def test_the_misdiagnosis_price_is_validated_and_a_configuration_stored_without_it_is_refused():
    """prices.misdiagnosis is a declared price of the price table, zero allowed and below zero refused,
    and a stored configuration without the key is refused as one built in code is."""
    priced = _json("priced")
    with pytest.raises(ValueError, match="prices.misdiagnosis"):
        EpisodeConfig.from_json({**priced, "prices": {**priced["prices"], "misdiagnosis": -1}})
    assert EpisodeConfig.from_json({**priced, "prices": {**priced["prices"], "misdiagnosis": 0}}).prices["misdiagnosis"] == 0
    stored = {k: v for k, v in priced["prices"].items() if k != "misdiagnosis"}
    with pytest.raises(ValueError, match="prices.misdiagnosis"):
        EpisodeConfig.from_json({**priced, "prices": stored})


def test_the_switches_of_the_configuration_read_the_off_values():
    base, advanced, deterioration, static = (load_config(n)[1] for n in ("base", "advanced_real_time", "deterioration", "static"))
    assert not (base.downtime_on() or base.failure_on() or base.emergency_on())
    assert advanced.downtime_on() and not advanced.failure_on()
    assert not (advanced.emergency_on() or advanced.dependencies or advanced.prices["visit"])
    assert deterioration.failure_on() and deterioration.failure["strict"] and static.downtime_on() and static.failure_on()


@needs_data
def test_every_row_draws_the_hidden_start_of_static_for_every_seed(world):
    envs = [_env(world, n) for n in ("static",) + ROWS + ARMS]
    for seed in range(200):
        starts = set()
        for env in envs:
            env.reset(seed)
            starts.add((tuple(env.initial), env.window_start, env.entry_offset))
        assert len(starts) == 1, seed


@needs_data
def test_the_rows_without_plant_time_move_the_clock_only_by_waits_and_thinking(world):
    for name, thinking in (("priced", 0), ("real_time", 2)):
        env = _env(world, name)
        env.reset(3)
        envelope, *_ = env.step({"type": "wait_for_notification", "timeout": 3})
        assert (env.time, envelope["result"]["ended_by"]) == (3, "timeout")  # no card is on and no group is paid for
        envelope, *_ = env.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 2})
        assert env.time == 3 and len(envelope["result"]["served"]) == 2  # a read takes no time and still serves its cycles
        env.step({"type": "get_current_time"}, think_cycles=2)
        assert env.time == 3 + thinking  # the declared mode ignores thinking, the generation_time mode runs the clock with it


@needs_data
@pytest.mark.parametrize("name", ("base", "priced", "real_time"))
def test_an_order_of_zero_time_closes_inside_its_call_on_the_rows_without_a_clock(world, name):
    """Without a clock no cycle runs unless the agent waits, so an agent that does not know to wait would
    never see an order of zero lead time and zero repair downtime close, and on the rows whose repair card is off
    create_work_order closes it. A row with a clock, real_time, closes it in the next cycle the rig runs, as before."""
    env = _env(world, name)
    env.reset(3)
    config = env.config
    faulty = min((c for c, level in zip(config.components, env.initial) if level != config.healthy[c]), key=lambda c: config.priority[c])
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, env.initial)), "evidence_ids": []})
    envelope, *_ = env.step({"type": "create_work_order", "component": faulty, "prediction_id": "", "evidence_ids": []})
    order = envelope["result"]["work_order"]
    assert env.time == 0 and config.has_clock() is (name == "real_time")
    if config.has_clock():
        assert order["status"] == "open" and "work_order_closed" not in [n["event"] for n in envelope["notifications"]]
        envelope, *_ = env.step({"type": "wait_for_notification", "timeout": 1})
    else:
        assert order["status"] == "closed" and order["closed_t"] == 0
    assert [n["id"] for n in envelope["notifications"] if n["event"] == "work_order_closed"] == [order["id"]]
    assert env.clock.levels[faulty] == config.healthy[faulty]


@needs_data
def test_on_static_zero_durations_a_visit_settles_in_priority_order_at_the_next_call_that_is_not_an_order(world):
    """Under the repair card of static_zero_durations: the orders created since the last call that was not
    an order form one visit with one charge, and the next such call closes them in priority order, so an order created
    before its degraded dependency's order still takes. An order after that call opens a visit of its own."""
    from dataclasses import replace

    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    config = load_config("static_zero_durations")[1]
    dependent = next(c for c in config.components if config.dependencies[c])
    dependency = config.dependencies[dependent][0]
    combination = [config.healthy[c] for c in config.components]
    for c in (dependent, dependency):
        combination[config.components.index(c)] = next(level for level in config.levels[c] if level != config.healthy[c])
    env = HydraulicRigEnv(*world, replace(config, start={"combination": combination, "entry_offset": 0}))
    env.reset(0)
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, combination)), "evidence_ids": []})
    for c in (dependent, dependency):  # the dependent first, against the priority order
        envelope, *_ = env.step({"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []})
        assert envelope["result"]["work_order"]["status"] == "open"
    created = [e for e in env.log if e["event"] == "work_order_created"]
    assert [e["visit"] for e in created] == [1, 1] and [e["visit_charge"] for e in created] == [config.prices["visit"], 0]
    envelope, *_ = env.step({"type": "get_current_time"})
    closed = [n["component"] for n in envelope["notifications"] if n["event"] == "work_order_closed"]
    assert env.time == 0 and closed == sorted((dependent, dependency), key=config.priority.get)
    assert not [e for e in env.log if e["event"] == "order_ineffective"] and env.hidden == [config.healthy[c] for c in config.components]
    env.step({"type": "create_work_order", "component": dependent, "prediction_id": "", "evidence_ids": []})
    later = [e for e in env.log if e["event"] == "work_order_created"][-1]
    assert (later["visit"], later["visit_charge"]) == (2, config.prices["visit"])


@needs_data
def test_on_the_base_row_orders_and_a_finish_leave_the_rig_restored(world):
    """The base row has no clock and its repair card is off, so every order closes inside its call and
    an agent that orders and finishes without a wait leaves the rig restored."""
    env = _env(world, "base")
    env.reset(3)
    config = env.config
    faulty = [c for c, level in zip(config.components, env.initial) if level != config.healthy[c]]
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, env.initial)), "evidence_ids": []})
    for c in faulty:
        env.step({"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []})
    *_, done, info = env.step({"type": "finish", "summary": "done"})
    assert faulty and done and env.time == 0 and info["log"][-1]["final"] == [config.healthy[c] for c in config.components]


def _diagnosed(world, name, levels, **overrides):
    """An environment of the preset that starts from the healthy rig with the given components at the given levels and
    has already submitted the true diagnosis, so that the next call may order."""
    from dataclasses import replace

    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    config = load_config(name)[1]
    combination = [levels.get(c, config.healthy[c]) for c in config.components]
    env = HydraulicRigEnv(*world, replace(config, start={"combination": combination, "entry_offset": 0}, **overrides))
    env.reset(0)
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, combination)), "evidence_ids": []})
    return env


def _order(env, component):
    return env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})


@needs_data
@pytest.mark.parametrize("name", ("real_time", "advanced_real_time"))
def test_on_a_row_with_a_clock_finish_runs_the_clock_to_the_last_close_and_counts_its_downtime(world, name):
    """On a row with a clock finish settles the open work orders. With no further call the rig runs to
    the last close and the downtime of the lead time and of the repair counts, so an agent that orders and finishes
    without a wait leaves the rig restored and pays what the wait costs. real_time has no lead time, no repair downtime
    and no downtime rate, so its order closes at the cycle of the finish at no charge."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import STOPPED
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order

    config = load_config(name)[1]
    level = degradation_order(config, "cooler")[1]  # degraded but not at its worst level, so no failure can start
    env = _diagnosed(world, name, {"cooler": level})
    envelope, *_ = _order(env, "cooler")
    order = envelope["result"]["work_order"]
    assert order["status"] == "open" and env.time == 0
    envelope, _cost, done, info = env.step({"type": "finish", "summary": "cooler ordered"})
    lead, repair = config.lead_time("cooler"), config.durations["repair_downtime"]["cooler"]
    end = info["log"][-1]
    assert done and info["reason"] == "finish" and env.time == order["expected_close_t"] == lead + repair
    assert envelope["open_work_orders"] == [] and end["final"] == [config.healthy[c] for c in config.components]
    assert end["downtime_total"] == lead * config.level_rate("cooler", level) + repair * config.downtime_rate[STOPPED]
    after = [e["event"] for e in info["log"][[e["event"] for e in info["log"]].index("finish") + 1:]]
    assert after[-2:] == ["work_order_closed", "end"]


@needs_data
def test_on_a_row_with_a_clock_finish_closes_a_dependency_plan_in_priority_order(world):
    """Under the repair card of static.json: a dependent ordered before its degraded dependency in one
    cycle forms one visit, and the finish runs the clock through both repairs in priority order, so the dependent takes
    after its dependency has closed and the end entry holds the restored rig."""
    config = load_config("static")[1]
    dependent = next(c for c in config.components if config.dependencies[c])
    dependency = config.dependencies[dependent][0]
    levels = {c: next(level for level in config.levels[c] if level != config.healthy[c]) for c in (dependent, dependency)}
    env = _diagnosed(world, "static", levels)
    for c in (dependent, dependency):  # the dependent first, against the priority order
        _order(env, c)
    assert env.time == 0 and len(env.clock.open_orders()) == 2
    _envelope, _cost, done, info = env.step({"type": "finish", "summary": "both ordered"})
    first, second = sorted((dependent, dependency), key=config.priority.get)
    lead = max(config.lead_time(c) for c in (first, second))
    closed = [e for e in info["log"] if e["event"] == "work_order_closed"]
    assert done and [(e["component"], e["t_end"]) for e in closed] == [
        (first, lead + config.durations["repair_downtime"][first]),
        (second, lead + config.durations["repair_downtime"][first] + config.durations["repair_downtime"][second])]
    assert env.time == closed[-1]["t_end"] and not [e for e in info["log"] if e["event"] == "order_ineffective"]
    assert info["log"][-1]["final"] == [config.healthy[c] for c in config.components]


@needs_data
@pytest.mark.parametrize("last", ("get_current_time", "finish"))
def test_the_step_cap_leaves_an_order_open_while_a_finish_at_the_cap_settles_it(world, last):
    """The settle of the finish keeps the meaning of the step cap: an episode that the cap ends leaves its open order open and
    is scored on the unrepaired rig, while a finish that is itself the last allowed call still settles the order."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order

    config = load_config("advanced_real_time")[1]
    level = degradation_order(config, "cooler")[1]
    env = _diagnosed(world, "advanced_real_time", {"cooler": level}, max_steps=3)
    _order(env, "cooler")
    action = {"type": "finish", "summary": "cooler ordered"} if last == "finish" else {"type": last}
    envelope, _cost, done, info = env.step(action)
    final = info["log"][-1]["final"]
    assert done and info["reason"] == ("finish" if last == "finish" else "max_steps")
    if last == "finish":
        assert envelope["open_work_orders"] == [] and final == [config.healthy[c] for c in config.components]
    else:
        assert [o["component"] for o in envelope["open_work_orders"]] == ["cooler"] and env.time == 0
        assert final[config.components.index("cooler")] == level


@needs_data
def test_each_row_briefs_the_rules_of_its_cards_and_of_no_other(world):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import DETERIORATION_SENTENCES, briefing

    texts = {}
    for name in ROWS + ARMS:
        env = _env(world, name)
        env.reset(0)
        texts[name] = briefing(env.public_profile(), env.config)
    inactive = ("emergency", "visit charge", "depends on", "The fix history, get_fix_history")
    for name in ROWS[:4]:
        assert not [w for w in inactive if w in texts[name]], name
    for name in ("repair_rules", "deterioration", "scheduled"):  # the four repair cards are briefed
        assert [w for w in inactive if w in texts[name]] == list(inactive), name
    # the hazard card adds its one sentence on the deterioration row, beside the failure card of big layer 5
    hazard, schedule = DETERIORATION_SENTENCES["hazard"], DETERIORATION_SENTENCES["schedule"]
    assert hazard in texts["deterioration"] and hazard not in texts["repair_rules"]
    assert not [name for name in ROWS[:5] if hazard in texts[name] or schedule in texts[name]]
    # the scheduled arm briefs the schedule card where the deterioration row briefs the hazard card, and nothing else
    assert hazard not in texts["scheduled"] and texts["scheduled"] == texts["deterioration"].replace(hazard, schedule, 1)
    assert "budget" not in texts["base"].lower() and "budget of 300" in texts["priced"]
    for name in ("base", "priced", "real_time"):
        assert "Downtime accrues" not in texts[name] and "downtime rate" not in texts[name] and "failure that adds" not in texts[name]
    seconds = load_config("real_time")[1].think_seconds_per_cycle
    assert f"every {seconds} seconds move the clock one cycle" in texts["real_time"] and "response time" not in texts["priced"]
    for name in ("advanced_real_time", "repair_rules", "deterioration", "scheduled"):
        text, failing = texts[name], name in ("deterioration", "scheduled")  # the failure card joins with big layer 5
        assert "Downtime accrues" in text and ("failure that adds 50 to the cost and ends the episode" in text) is failing, name
        notified = ("the alarms of the sensor groups already read and the failures since" if failing
                    else "the work orders executed and closed and the alarms of the sensor groups already read since")
        assert notified in text, name
    for name, text in texts.items():  # the finish settle is stated on the rows with a clock alone
        assert ("runs until the open work orders close" in text) is (name in ROWS[2:] + ARMS), name
    # the misdiagnosis price is briefed where it is above zero, on every row but the base row
    assert "misdiagnosis" not in texts["base"]
    assert all("costs the misdiagnosis price of 15," in texts[name] for name in ROWS[1:] + ARMS)


@needs_data
def test_the_briefing_names_a_deterioration_card_only_while_it_is_on(world):
    """Briefing v2.6: one sentence per deterioration card the episode switches on, the schedule card on timed and on the
    scheduled arm and the hazard card on the deterioration row among the presets, and the collateral damage of the
    hazard card counts as on."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import DETERIORATION_SENTENCES, briefing, deterioration_sentences

    on = {name: deterioration_sentences(load_config(name)[1]) for name in PRESETS}
    assert on.pop("timed") == on.pop("scheduled") == [DETERIORATION_SENTENCES["schedule"]]
    assert on.pop("deterioration") == [DETERIORATION_SENTENCES["hazard"]]
    for twin, wall in TOKEN_PRESETS.items():  # a preset of the token clock states the cards of the preset it repeats
        assert on.pop(twin) == deterioration_sentences(load_config(wall)[1]), twin
    for name in ("deterioration_thinking_off", "deterioration_durations_off", "deterioration_failure_off", "deterioration_misdiagnosis_budget", *THINKING_PRICE):  # the hazard of the switch ablations
        assert on.pop(name) == [DETERIORATION_SENTENCES["hazard"]], name
    assert not [name for name, sentences in on.items() if sentences]
    timed = _env(world, "timed")
    timed.reset(0)
    assert DETERIORATION_SENTENCES["schedule"] in briefing(timed.public_profile(), timed.config)
    repair = _json("repair_rules")
    collateral = EpisodeConfig.from_json({**repair, "hazard": {**repair["hazard"], "collateral": 0.3}})
    assert deterioration_sentences(collateral) == [DETERIORATION_SENTENCES["hazard"]]


@needs_data
def test_the_manual_offers_the_emergency_option_only_when_the_emergency_mode_is_on(world):
    assert "emergency" not in _env(world, "advanced_real_time").fix_manual("pump")
    repair = _env(world, "repair_rules")
    assert repair.fix_manual("pump")["emergency"]["hold"] == repair.config.emergency["hold"]
    static = _env(world, "static")
    assert static.fix_manual("pump")["emergency"]["hold"] == static.config.emergency["hold"]


@needs_data
@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_publishes_the_price_and_the_duration_of_every_tool(world, name):
    """The tool table of the asset profile gives every tool its price and its declared duration under the preset,
    from the configuration's own tables: a price per cycle by sensor group and one cycle per cycle read for
    read_sensors, the standard price of every component with the visit charge for create_work_order and the emergency
    price only while the emergency mode is on, one cycle per cycle waited for wait_for_notification, and zero elsewhere."""
    env = _env(world, name)
    env.reset(0)
    config = env.config
    standard = {c: config.prices["parts"][c] + config.prices["repair_charge"] for c in config.components}
    order = {"standard": standard, "visit": config.prices["visit"]}
    if config.emergency["hold"] > 0:
        order["emergency"] = {c: price * config.emergency["surcharge"] for c, price in standard.items()}
    prices = {"read_sensors": config.prices["read"], "create_work_order": order}
    durations = {"read_sensors": config.durations["read_per_cycle"], "wait_for_notification": 1}
    profile = env.public_profile()
    rows = {row["name"]: row for row in profile["tools"]}
    assert {"read_sensors", "create_work_order", "wait_for_notification", "get_current_time"} <= set(rows)
    for tool, row in rows.items():
        assert (row["price_value"], row["duration_value"]) == (prices.get(tool, 0), durations.get(tool, 0)), (name, tool)
    # the clock block states the same durations with the mode, the values released.json carries under rules["clock"]
    assert profile["clock"]["mode"] == config.clock_mode and profile["clock"]["seconds_per_cycle"] == config.seconds_per_cycle
    assert profile["clock"]["durations"] == {tool: rows[tool]["duration_value"] for tool in profile["clock"]["durations"]}


def test_every_preset_takes_the_two_values_of_the_clock_from_the_asset_file():
    """One knob for the thinking rate: no preset names seconds_per_cycle or a number
    of thinking seconds, so every preset resolves both to the values of the asset file, the thinking rate that the pilot of
    CausalDiagnosis/docs/environment.md sets by a one line change, and the recorded cycle that spaces the AssetOpsBench rows,
    while the two thinking price presets state only a multiplier of that rate."""
    clock = json.loads((ROOT / "CausalDiagnosis" / "datasets" / "hyd_1" / "asset.json").read_text(encoding="utf-8"))["defaults"]["clock"]
    fields = get_profile("hyd_1", engine=False).config_fields()
    for key in ("think_seconds_per_cycle", "seconds_per_cycle"):
        assert not [name for name in PRESETS if key in json.loads(preset_path(name).read_text(encoding="utf-8"))], key
    assert {name for name in PRESETS if "think_seconds_per_cycle" in _preset_file(name).get("switches", {})} == set(THINKING_PRICE)
    for name, scale in THINKING_PRICE.items():
        assert _preset_file(name)["switches"] == {"think_seconds_per_cycle": {"scale": scale}}, name
    for key in ("think_seconds_per_cycle", "seconds_per_cycle"):
        resolved = {name: getattr(load_config(name)[1], key) for name in PRESETS}
        expected = {name: clock[key] * (THINKING_PRICE.get(name, 1) if key == "think_seconds_per_cycle" else 1) for name in PRESETS}
        assert resolved == expected and clock[key] == fields[key], key


@needs_data
@pytest.mark.parametrize("name, scale", THINKING_PRICE.items())
def test_a_thinking_price_preset_briefs_and_publishes_the_scaled_rate_and_charges_by_it(world, name, scale):
    """The briefing sentence of the generation time clock, the public profile and the clock charge all use the
    scaled rate, so the agent is told the price of thinking it pays, and nothing else of the briefing differs from the
    deterioration row."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import wall_clock
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

    rate = load_config("deterioration")[1].think_seconds_per_cycle
    texts = {}
    for preset in ("deterioration", name):
        env = _env(world, preset)
        env.reset(0)
        texts[preset] = briefing(env.public_profile(), env.config)
        assert env.public_profile()["think_seconds_per_cycle"] == env.config.think_seconds_per_cycle
    seconds = load_config(name)[1].think_seconds_per_cycle
    assert seconds == scale * rate and seconds != rate
    old, new = f"every {rate} seconds move the clock one cycle", f"every {seconds} seconds move the clock one cycle"
    assert old in texts["deterioration"] and new in texts[name] and texts[name] == texts["deterioration"].replace(old, new)
    assert wall_clock.think_cycles(load_config(name)[1], 0.0, 4 * rate)[0] == int(4 / scale)


FAILURE_OFF = [name for name in PRESETS if load_config(name)[1].failure["patience"] == 0]  # the presets without the failure rule


def test_the_failure_rule_is_off_on_the_presets_below_the_deterioration_layer_and_on_its_failure_ablation():
    assert {"base", "priced", "real_time", "advanced_real_time", "repair_rules", "deterioration_failure_off"} <= set(FAILURE_OFF)
    assert "deterioration" not in FAILURE_OFF


@needs_data
@pytest.mark.parametrize("name", FAILURE_OFF)
def test_without_the_failure_rule_every_window_of_the_oracle_ends_at_the_horizon(world, name):
    """The oracle's window end is the cycle at which a component would fail. With the failure rule off (patience 0) no
    component fails, so every window ends at the horizon, at the start and after an episode with its drops, the judge's
    reference answer names no cycle before the horizon, and the timing check of the verifier marks no order of the
    oracle's own play late."""
    import re

    from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

    env = _env(world, name)
    horizon = env.config.horizon
    for seed in range(60):
        env.reset(seed)
        assert {o["window_end"] for o in env.oracle()["work_orders"]} <= {horizon}, (name, seed)
        run_episode(env, OraclePolicy(env), seed)
        oracle = env.oracle()
        assert {o["window_end"] for o in oracle["work_orders"]} <= {horizon}, (name, seed)
        reference = judge.reference_answer(oracle)
        assert re.findall(r"closed by cycle (-?\d+)", reference) == [str(horizon)] * len(oracle["work_orders"]), (name, seed)
        assert verify(env.log, released_from_env(env), oracle, cost_oracle=None)["checks"]["timing"]["ok"], (name, seed)


@needs_data
def test_with_the_failure_rule_on_a_window_ends_patience_cycles_after_the_worst_level_or_at_the_horizon(world):
    """On the deterioration row the failure rule is on, so a component at its worst level at t0 has the window end
    patience - 1 and every other window lies between that and the horizon."""
    env = _env(world, "deterioration")
    patience, horizon = env.config.failure["patience"], env.config.horizon
    for seed in range(60):
        env.reset(seed)
        for order in env.oracle()["work_orders"]:
            assert patience - 1 <= order["window_end"] <= horizon, (seed, order)


@needs_data
@pytest.mark.parametrize("name", PRESETS)
def test_the_profile_and_the_briefing_name_the_same_public_rules(world, name):
    """Every rule card of the asset reaches get_asset_profile, and the briefing, which says that
    get_asset_profile returns its rules, states a card exactly where the profile shows the card's on parameters. A card the
    briefing leaves out shows its off parameters, those of the rule library, in the profile."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_3_advanced_real_time as advanced_real_time, layer_5_deterioration_and_failure as deterioration, layer_4_repair_rules as repair_rules
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import DETERIORATION_SENTENCES, REPAIR_SENTENCES, briefing

    env = _env(world, name)
    env.reset(0)
    profile, config = env.public_profile(), env.config
    text = briefing(profile, config)
    on = {"failure": config.failure_on(), "downtime": advanced_real_time.downtime_on(config),
          **{card: card in deterioration.cards(config) for card in ("schedule", "hazard")},
          **{card: card in repair_rules.cards(config) for card in ("fix_history", "emergency", "dependencies", "visits")}}
    stated = {"failure": "only a repair on it that has closed by then averts it", "downtime": "Downtime accrues",
              **DETERIORATION_SENTENCES, **REPAIR_SENTENCES}
    assert set(on) == set(CONFIG_KINDS) == set(stated)
    asset = get_profile(config.asset_id)
    for kind in CONFIG_KINDS:
        assert (stated[kind] in text) is on[kind], (name, kind)
        off = rules.apply_card({}, {"kind": kind, "parameters": rules.kind_of(kind).off(asset)})
        for field, value in off.items():
            published, running = profile["processes"].get(field, profile.get(field)), getattr(config, field)
            if field == "prices":  # the visit charge is the one entry of the price table that the visits card binds
                published, running = {k: published[k] for k in value}, {k: running[k] for k in value}
            assert published == running, (name, kind, field)
            assert (published != value) is on[kind], (name, kind, field)


def test_the_time_section_of_the_environment_page_states_the_declared_thinking_rate():
    """The thinking rate as a declared value of the asset file that no run measures, written where the clock is described,
    with the failure card's patience that decides whether a model agent can act in time."""
    page = (ROOT / "CausalDiagnosis" / "docs" / "environment.md").read_text(encoding="utf-8")
    time = page.split("\n## Time\n", 1)[1].split("\n## ", 1)[0]
    for phrase in ("declared values of the asset file", "no run measures or sets them", "think_seconds_per_cycle",
                   "CausalDiagnosis/datasets/hyd_1/asset.json", "never pool", "failure card", "patience"):
        assert phrase in time, phrase
