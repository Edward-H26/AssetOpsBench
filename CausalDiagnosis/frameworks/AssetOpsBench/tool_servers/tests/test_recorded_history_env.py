"""The engine of the AssetOpsBench rows: the prehistory and the live cycles from the two segments of the start, the
reads over the record, the free AssetOpsBench read and the order that closes in its call where no clock runs."""
from CausalDiagnosis.utils.paths import CONFIGS, preset_files
import copy

import pytest

from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT, segment_positions
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig, default_prices
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, SENSORS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import BLOCK_CAP, HIDDEN_FIELDS, HydraulicRigEnv, served_values
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import HISTORY_CYCLES, UNRECORDED_REASON, RecordedRigEnv
from CausalDiagnosis.datasets.hyd_1.former_field_names import HYD1_FIELDS

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, submit_truth, summary

ZERO_DURATIONS = {"read_per_cycle": 0, "advance_per_cycle": 1, "lead_time": {c: 0 for c in COMPONENTS},
                  "repair_downtime": {c: 0 for c in COMPONENTS}}
FREE = {**default_prices(), "read": {"tier1": 0, "tier2": 0, "tier3": 0}, "parts": {c: 0 for c in COMPONENTS}, "repair_charge": 0, "visit": 0}
# the clock and the prices of the four rows, each on the defaults of EpisodeConfig, with the repair cards off on
# the two rows without a clock as their presets have them, since a visit there closes at the next call
ROWS = {"base": {"durations": ZERO_DURATIONS, "prices": FREE, "budget": None, "dependencies": {}},
        "priced": {"durations": ZERO_DURATIONS, "prices": {**default_prices(), "visit": 0}, "dependencies": {}},
        "real_time": {"durations": ZERO_DURATIONS, "clock_mode": "generation_time"},
        "advanced_real_time": {"clock_mode": "generation_time"}}
SEED = 7


@pytest.fixture
def make(world):
    cycles, blocks = world

    def build(row, cls=RecordedRigEnv, seed=SEED, **overrides):
        env = cls(cycles, blocks, EpisodeConfig(**{**ROWS[row], **overrides}))
        env.reset(seed)
        return env
    return build


def fields_of(group, config, statistic="mean", count=None):
    return [f"{HYD1_FIELDS[s]}_{statistic}" for s in config.sensor_groups[group][:count]]


def read(env, fields, group="tier1", tool="history", start=None, end=None, offset=0, limit=1000, think=0):
    action = {"type": "read_recorded", "tool": tool, "sensor_group": group, "fields": list(fields)}
    if tool == "history":
        action.update(start=start, end=end, offset=offset, limit=limit)
    return env.step(action, think_cycles=think)


def prehistory(env):
    return [t for t in env.recorded_cycles() if t < 0]


def test_the_prehistory_is_the_segment_at_the_entry_offset_and_the_start_is_the_rigs(make, world):
    _cycles, blocks = world
    env, rig = make("base"), make("base", cls=HydraulicRigEnv)
    assert (env.initial, env.window_start, env.entry_offset) == (rig.initial, rig.window_start, rig.entry_offset)
    block = blocks[env.initial]
    assert prehistory(env) == list(range(-HISTORY_CYCLES, 0))
    assert [env._recorded[t] for t in prehistory(env)] == [(block[p], False) for p in segment_positions(env.entry_offset)]
    # the live cycles play the other segment as the rig does, then wrap among themselves with the repeats marked
    live = [rig.clock.next_cycle() for _ in range(SEGMENT)]
    assert [index for index, _ in live] == [block[p] for p in segment_positions(env.entry_offset + SEGMENT)]
    assert [env.clock.next_cycle() for _ in range(2 * len(live))] == live + [(index, True) for index, _ in live]


def test_the_nineteen_recording_block_serves_its_first_ten_recordings_five_as_prehistory_and_five_live(make, world):
    _cycles, blocks = world
    combination = next(k for k, block in blocks.items() if len(block) == 19)
    for entry in range(BLOCK_CAP):
        env = make("real_time", start={"combination": list(combination), "entry_offset": entry})
        assert env.window_start == 0 and env.entry_offset == entry
        history = [env._recorded[t][0] for t in prehistory(env)]
        live = [env.clock.next_cycle() for _ in range(2 * SEGMENT)]
        fresh, repeats = live[:SEGMENT], live[SEGMENT:]
        assert len(set(history)) == HISTORY_CYCLES and sorted(history + [index for index, _ in fresh]) == sorted(blocks[combination][:BLOCK_CAP])
        assert not any(repeated for _, repeated in fresh) and repeats == [(index, True) for index, _ in fresh]


def test_a_change_of_combination_enters_its_live_segment(make, world):
    _cycles, blocks = world
    env = make("real_time", dependencies={})
    submit_truth(env)
    component = next(c for c, level in zip(COMPONENTS, env.initial) if level != HEALTHY[c])
    env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})
    env.step({"type": "aob_read", "server": "iot", "tool": "sites"}, think_cycles=SEGMENT + 1)
    combination, window = tuple(env.hidden), env.clock._window
    # the order closed in cycle 0 before the rig played it, so every live cycle comes from the new combination's window
    assert combination != env.initial and window["t"] == 0
    played = [env._recorded[t] for t in range(SEGMENT + 1)]
    assert [index for index, _ in played[:SEGMENT]] == [blocks[combination][p] for p in segment_positions(env.entry_offset + SEGMENT)]
    assert [repeated for _, repeated in played] == [False] * SEGMENT + [True]


def test_a_read_of_the_prehistory_serves_negative_cycles_with_the_asked_fields_only(make):
    env = make("base")
    asked = fields_of("tier1", env.config, count=2)
    obs, cost, done, info = read(env, asked)
    result = obs["result"]
    assert result["cycles"] == prehistory(env) and result["paid"] == prehistory(env) and result["price"] == cost == 0
    entry = events(env.log, "read")[0]
    assert [o["t"] for o in entry["served"]] == prehistory(env)
    assert all(list(o["values"]) == asked for o in entry["served"])
    assert all(o["values"] == {f: env.record_values(o["t"])[f] for f in asked} for o in entry["served"])
    assert result["observations"] == [o["n"] for o in entry["served"]]
    public = env.public_events()[0]["served"][0]
    assert not set(public) & set(HIDDEN_FIELDS)


def test_a_re_read_is_free_and_only_an_uncovered_group_is_charged_again(make):
    env = make("priced")
    first, second = fields_of("tier1", env.config, count=1), fields_of("tier1", env.config, statistic="std", count=1)
    window = {"start": -3, "end": 0}
    obs, cost, _, _ = read(env, first, **window)
    assert obs["result"]["price"] == cost == 3 * env.config.prices["read"]["tier1"] and obs["result"]["paid"] == [-3, -2, -1]
    obs, cost, _, _ = read(env, first, **window)
    assert (obs["result"]["price"], obs["result"]["paid"], obs["result"]["observations"]) == (0, [], [])
    obs, cost, _, _ = read(env, second, **window)  # another field of the paid group, released now and free
    assert obs["result"]["price"] == 0 and len(obs["result"]["observations"]) == 3
    only_tier3 = [f"{HYD1_FIELDS[s]}_mean" for s in env.config.sensor_groups["tier3"] if s not in env.config.sensor_groups["tier1"]][:1]
    obs, cost, _, _ = read(env, only_tier3, group="tier3", **window)
    assert obs["result"]["price"] == 3 * env.config.prices["read"]["tier3"]
    assert obs["budget_left"] == env.config.budget - 3 * (env.config.prices["read"]["tier1"] + env.config.prices["read"]["tier3"])


def test_latest_reading_serves_the_newest_recorded_cycle(make):
    env = make("base")
    assert read(env, fields_of("tier1", env.config, count=1), tool="latest_reading")[0]["result"]["cycles"] == [-1]
    clocked = make("real_time")
    assert read(clocked, fields_of("tier1", clocked.config, count=1), tool="latest_reading", think=5)[0]["result"]["cycles"] == [4]


def test_on_the_real_time_clock_thinking_records_cycles_and_a_read_stays_in_the_present(make):
    env = make("real_time")
    env.step({"type": "aob_read", "server": "iot", "tool": "sites"}, think_cycles=4)
    assert env.time == 4 and env.recorded_cycles() == prehistory(env) + [0, 1, 2, 3]
    obs, _, _, _ = read(env, fields_of("tier1", env.config, count=1), start=0, end=10)
    assert obs["result"]["cycles"] == [0, 1, 2, 3] and env.time == 4


def test_on_the_advanced_real_time_clock_a_window_beyond_the_clock_plays_its_cycles_first(make):
    env = make("advanced_real_time")
    obs, cost, _, _ = read(env, fields_of("tier1", env.config, count=1), start=0, end=3)
    assert env.time == 3 * env.config.durations["read_per_cycle"] and obs["result"]["cycles"] == [0, 1, 2]
    # the result carries the price alone, while the step's cost also holds the downtime of the cycles the read played
    assert obs["result"]["price"] == 3 * env.config.prices["read"]["tier1"] == env.config.budget - obs["budget_left"] <= cost
    obs, _, _, _ = read(env, fields_of("tier1", env.config, count=1), start=0, end=3)
    assert env.time == 3 and obs["result"]["price"] == 0  # recorded now, so the read returns at once and free
    bounded = make("advanced_real_time", horizon=5)
    obs, _, done, _ = read(bounded, fields_of("tier1", bounded.config, count=1), start=0, end=50)
    assert bounded.time == 5 and obs["result"]["cycles"] == [0, 1, 2, 3, 4] and done


@pytest.mark.parametrize("row", ["base", "priced"])
def test_without_a_clock_an_order_closes_in_its_call_and_no_reading_follows(make, row):
    env = make(row)
    submit_truth(env)
    component = next(c for c, level in zip(COMPONENTS, env.initial) if level != HEALTHY[c])
    obs, cost, _, _ = env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})
    order = obs["result"]["work_order"]
    assert order["status"] == "closed" and order["closed_t"] == env.time == 0
    assert obs["result"]["price"] == cost == env.config.order_price(component) + env.config.prices["visit"]
    assert [e["event"] for e in env.public_events()][-2:] == ["work_order_created", "work_order_closed"]
    assert env.recorded_cycles() == prehistory(env) and env.hidden[COMPONENTS.index(component)] == HEALTHY[component]


def test_a_timed_read_the_budget_cannot_pay_is_refused_before_any_cycle_plays(make):
    env = make("advanced_real_time", budget=5)
    moved = (env.time, env.recorded_cycles(), env._rng.getstate(), env.clock.downtime_total)
    group = "tier3"
    obs, cost, done, info = read(env, fields_of(group, env.config), group=group, start=0, end=3)
    assert info["refused"] == "budget" and done and cost == 0 and env.log[-1]["reason"] == "budget"
    assert (env.time, env.recorded_cycles(), env._rng.getstate(), env.clock.downtime_total) == moved


def test_a_timed_read_the_budget_can_pay_is_charged_what_its_dry_run_priced(make):
    env = make("advanced_real_time")
    group = "tier1"
    obs, cost, _, _ = read(env, fields_of(group, env.config), group=group, start=0, end=4)
    result = obs["result"]
    assert result["cycles"] == [0, 1, 2, 3] == result["paid"] and env.time == 4
    assert result["price"] == 4 * env.config.prices["read"][group] == env.prices_paid


def test_an_aob_read_is_free_logged_as_a_read_and_counts_a_step(make):
    env = make("priced")
    obs, cost, _, _ = env.step({"type": "aob_read", "server": "wo", "tool": "list_workorders"})
    entry = env.log[-1]
    assert (entry["event"], entry["class"], entry["cost"], entry["server"], entry["tool"]) == ("aob_read", "read", 0, "wo", "list_workorders")
    assert cost == 0 and obs["steps_left"] == env.config.max_steps - 1
    obs, _, _, info = env.step({"type": "aob_read", "server": "wo"})
    assert info["refused"] and env.log[-1]["class"] == "read"


def test_read_sensors_and_malformed_reads_are_refused_without_a_charge(make):
    env = make("priced")
    before = env.recorded_cycles()
    assert env.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 2})[3]["refused"] == UNRECORDED_REASON
    tier1 = fields_of("tier1", env.config, count=1)
    refusals = [read(env, tier1, tool="sensor_stats"), read(env, tier1, group="tier9"),
                read(env, fields_of("tier3", env.config)[-1:]), read(env, tier1, offset=-1), read(env, tier1, start="0")]
    assert all(r[3].get("refused") and r[1] == 0 for r in refusals) and all(e["class"] == "read" for e in events(env.log, "refused"))
    assert env.recorded_cycles() == before and env.prices_paid == 0


def test_a_read_the_budget_cannot_pay_ends_the_episode(make):
    env = make("priced", budget=5)
    obs, cost, done, info = read(env, fields_of("tier1", env.config, count=1), group="tier3", start=-3, end=0)
    assert info["refused"] == "budget" and done and cost == 0 and env.log[-1]["reason"] == "budget"


def test_the_fields_of_a_group_are_the_names_a_reading_serves(make):
    env = make("base")
    index = env._recorded[-1][0]
    for group, sensors in env.config.sensor_groups.items():
        assert env._group_fields(group) == set(served_values(env.cycles[index], sensors))
    assert set(env.record_values(-1)) == set(served_values(env.cycles[index], SENSORS))


def test_a_replay_of_the_stored_actions_gives_the_same_record_and_log(make):
    tier1 = fields_of("tier1", ROWS_CONFIG := EpisodeConfig(**ROWS["real_time"]), count=2)
    script = [({"type": "aob_read", "server": "iot", "tool": "sites"}, 3),
              ({"type": "read_recorded", "tool": "history", "sensor_group": "tier1", "fields": tier1, "start": -2, "end": 10,
                "offset": 0, "limit": 100}, 2),
              ({"type": "read_recorded", "tool": "latest_reading", "sensor_group": "tier1", "fields": tier1}, 1)]
    runs = []
    for _ in range(2):
        env = make("real_time")
        for action, think in script:
            env.step(action, think_cycles=think)
        runs.append((env.log, env.recorded_cycles(), [env.record_values(t) for t in env.recorded_cycles()]))
    assert runs[0] == runs[1] and ROWS_CONFIG.clock_mode == "generation_time"


def test_a_read_that_serves_nothing_names_no_group_for_the_alarms(make):
    """A history window past the record costs nothing and serves nothing, so it may not turn the alarms of its group
    free: only a read that released a reading names its group, the rule of read_counts in the verifier."""
    env = make("real_time", start={"combination": [3, 90, 0, 130], "entry_offset": 0})
    tier3 = fields_of("tier3", env.config)
    obs, cost, _, _ = read(env, tier3, group="tier3", start=10 ** 6, end=10 ** 6 + 5)
    assert cost == 0 and obs["result"]["cycles"] == [] and env.log[-1]["served"] == []
    advance(env, 5)
    assert summary(env, 0)[0]["result"]["alarms"] == []
    obs, cost, _, _ = read(env, tier3, group="tier3", start=0, end=5)
    assert cost > 0 and [e for e in env.log if e["event"] == "read"][-1]["served"]  # the read's alarms follow it in the log
    assert summary(env, 0)[0]["result"]["alarms"]


def test_a_field_or_an_action_type_that_is_no_string_is_refused_and_counted(make):
    """A list where a field or an action type is expected is refused, logged and counted like any malformed action, and
    raises nothing: the fields by their refusal, the action type as an unknown action."""
    env = make("base")
    obs, cost, done, info = env.step({"type": "read_recorded", "tool": "history", "sensor_group": "tier1", "fields": [["x"]],
                                      "start": None, "end": None, "offset": 0, "limit": 1})
    assert info["refused"].startswith("fields must be a non-empty list") and env.steps == 1 and env.log[-1]["event"] == "refused"
    obs, cost, done, info = env.step({"type": ["read_recorded"]})
    assert info == {"refused": "unknown action type"} and env.steps == 2 and env.log[-1]["event"] == "refused"
    assert not done


def test_the_oracle_policy_costs_the_same_on_both_engines_on_every_preset(world):
    """The normalized cost of an episode of the AssetOpsBench rows divides by the cost of the oracle policy's episode, which
    the drivers play on HydraulicRigEnv. That is its cost on the engine of those rows as well: for every preset of the
    drivers, composed as they compose it, and the first seeds of its start set, the two costs are equal and the two logs
    equal entry for entry, since the oracle policy reads nothing and the engine of the rows changes no charge of a
    diagnosis, a work order, a wait or a finish. The presets are read from their folder, so a new preset is covered without
    an edit."""
    import json
    from pathlib import Path

    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
    from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import combination_seeds

    cycles, blocks = world
    presets = sorted(preset_files(CONFIGS).values())
    assert presets
    for path in presets:
        config = compose(json.loads(path.read_text(encoding="utf-8")))
        for seed in combination_seeds(HydraulicRigEnv(cycles, blocks, config), 1)[:3]:
            rig, rows = HydraulicRigEnv(cycles, blocks, config), RecordedRigEnv(cycles, blocks, config)
            assert oracle_cost(rig, seed) == oracle_cost(rows, seed), (path.stem, seed)
            logs = []
            for env in (HydraulicRigEnv(cycles, blocks, config), RecordedRigEnv(cycles, blocks, config)):
                env.reset(seed)
                logs.append(run_episode(env, OraclePolicy(env), seed))
            assert logs[0] == logs[1], (path.stem, seed)


def _order_on_a_faulty_component(env):
    submit_truth(env)
    component = next(c for c, level in zip(COMPONENTS, env.initial) if level != HEALTHY[c])
    return env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})


def test_the_dry_run_of_a_timed_read_leaves_the_engine_as_it_was(make):
    """A timed priced read that the budget refuses plays its cycles on a copy alone: afterwards the log holds the entries it
    held, as the same objects, and the clock, the work orders, the served windows and the record equal a deep copy taken
    before the call. A copy that shared the log, the orders or the windows with the engine fails, since the cycles of the
    refused read execute the open order and advance the window."""
    env = make("advanced_real_time")
    _order_on_a_faulty_component(env)
    read(env, fields_of("tier1", env.config, count=1), start=env.time, end=env.time + 1)
    group = "tier3"
    # more cycles than the budget pays even when the repair of the open order stops the rig for some of them
    cycles = env.budget_left() // env.config.prices["read"][group] + 1 + max(env.config.durations["repair_downtime"].values())
    assert env.clock.open_orders() and env.time + cycles < env.config.horizon
    entries = list(env.log)
    before = copy.deepcopy(env, memo={id(env.cycles): env.cycles, id(env.blocks): env.blocks})
    _obs, _cost, done, info = read(env, fields_of(group, env.config), group=group, start=env.time, end=env.time + cycles, limit=cycles)
    assert info["refused"] == "budget" and done
    assert all(entry is held for entry, held in zip(env.log, entries)) and [e["event"] for e in env.log[len(entries):]] == ["refused", "end"]
    clock = ("time", "played", "orders", "history", "levels", "downtime", "stopped_cycles")
    assert {name: getattr(env.clock, name) for name in clock} == {name: getattr(before.clock, name) for name in clock}
    assert (env.source.window, env.source.served, env._recorded, env.recorded_cycles()) == \
        (before.source.window, before.source.served, before._recorded, before.recorded_cycles())


def test_the_record_is_in_cycle_order_without_a_sort(make):
    """After an episode with timed reads, a repair whose downtime stops the rig and waits, the record lists its cycles in
    cycle order without a sort, and every window of a history read is the cut that the filter of the whole record gives."""
    env = make("advanced_real_time")
    read(env, fields_of("tier1", env.config, count=1), start=0, end=3)
    _order_on_a_faulty_component(env)
    advance(env, 8)
    read(env, fields_of("tier2", env.config, count=1), group="tier2", start=env.time - 2, end=env.time + 3)
    cycles = sorted(env._recorded)
    assert events(env.log, "work_order_closed") and env.clock.stopped_cycles and env.recorded_cycles() == cycles
    bounds = [None, cycles[0] - 1, cycles[0], -1, 0, 1, *sorted(env.clock.stopped_cycles), cycles[-1], cycles[-1] + 1, 10_000]
    for start in bounds:
        for end in bounds:
            assert env._recorded_in(start, end) == [c for c in cycles if (start is None or c >= start) and (end is None or c < end)]


def test_in_the_instant_clock_mode_every_latest_reading_is_charged_for_the_cycle_it_serves(make):
    """A read is priced for the cycles it serves: in the instant clock mode the call takes its cycle before the page is read,
    so every latest_reading serves a new cycle and pays the price of that cycle, the second as the first."""
    env = make("priced", clock_mode="instant")
    field = fields_of("tier1", env.config, count=1)
    for _ in range(2):
        result = read(env, field, tool="latest_reading")[0]["result"]
        assert result["paid"] == result["cycles"] == [env.time - 1]
        assert result["price"] == env.config.read_price("tier1", 1) > 0
