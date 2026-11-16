"""HydraulicRigEnv (CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py): determinism and replay, the envelope, the information barrier, the
served recordings, every refusal, the ways an episode ends, the change summary, the failure modes and the clock modes.
No test states a value computed from the rig data; data-dependent assertions are relations."""
import json
import random
import re

import pytest

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader.profile_data import processes_block
from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data, write_recordings  # noqa: F401
from CausalDiagnosis.datasets.hyd_1.tests import knowledge_tables as knowledge
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, percentile
from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT, segment_positions
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import (BAND_QUANTILES, BLOCK_CAP, DIAGNOSIS_COMPONENTS_REASON, DIAGNOSIS_LEVEL_REASON, EVIDENCE_IDS_REASON,
                                    HEALTHY_COMBO, HIDDEN_FIELDS, ORDER_BEFORE_WRITE_REASON, SEPARATING_TEST_REASON, WRITE_LOCK_REASON,
                                    HydraulicRigEnv, mean_field, served_values)
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.datasets.hyd_1.former_field_names import HYD1_FIELDS

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, faulty_components, finish, keys_of, order, read, repaired, submit_truth, summary, truth, wait

ENVELOPE_KEYS = {"time", "budget_left", "steps_left", "open_work_orders", "episode_over", "notifications", "result"}
PROFILE_KEYS = {"asset_id", "asset", "components", "levels", "healthy", "sensor_groups", "statistics", "prices", "durations", "downtime_rate",
                "budget", "horizon", "max_steps", "processes", "clock_mode", "seconds_per_cycle", "think_seconds_per_cycle", "clock", "tools",
                "normal_ranges"}
HAND_START = (3, 90, 0, 130)  # cooler and valve faulty: a start every test can anchor on


def test_a_fixed_start_without_a_recording_is_refused_when_the_environment_is_built(world):
    """A fixed start whose combination the data holds no block of is refused by name when the environment is built, not at
    the first reset."""
    cycles, blocks = world
    held = {combination: block for combination, block in blocks.items() if combination != HAND_START}
    config = EpisodeConfig(start={"combination": list(HAND_START), "entry_offset": 0})
    with pytest.raises(ValueError, match=r"the start combination \[3, 90, 0, 130\] has no recording in the data of hyd_1"):
        HydraulicRigEnv(cycles, held, config)
    HydraulicRigEnv(cycles, blocks, config).reset(0)


def test_a_fixed_start_at_an_entry_offset_the_data_does_not_serve_is_refused_when_the_environment_is_built(world):
    """A fixed start plays at the entry offset it names, the last one the source serves included, and an offset the source
    does not serve for the combination is refused by name, with the count of the offsets it serves, rather than played at
    another offset."""
    cycles, blocks = world
    last = HydraulicRigEnv(cycles, blocks, EpisodeConfig(start={"combination": list(HAND_START), "entry_offset": BLOCK_CAP - 1}))
    last.reset(0)
    assert last.entry_offset == BLOCK_CAP - 1
    beyond = EpisodeConfig(start={"combination": list(HAND_START), "entry_offset": BLOCK_CAP})
    with pytest.raises(ValueError, match=rf"the start entry offset {BLOCK_CAP} is not among the {BLOCK_CAP} entry offsets that the data of hyd_1 serves"):
        HydraulicRigEnv(cycles, blocks, beyond)


def anchored(make_env, combination=HAND_START, **overrides):
    """A hand start whose live segment opens with the first recording of its block."""
    return make_env(start={"combination": list(combination), "entry_offset": SEGMENT}, **overrides)


def scripted_log(env, seed):
    """A fixed action sequence that touches every tool and every end: a refusal, reads at two groups, the profile, a
    summary, the failure modes, the diagnosis, one order per faulty component, time until they close, a finish."""
    env.reset(seed)
    env.step({"type": "teleport"})
    read(env, "tier1", 2)
    read(env, "tier3", 1)
    env.step({"type": "get_asset_profile"})
    summary(env, 0)
    env.step({"type": "get_failure_modes", "component": ""})
    submit_truth(env, ["obs/1/" + mean_field("TS1")])
    for c in faulty_components(env.initial):
        order(env, c)
    while env.clock.open_orders() and not env.done:
        advance(env, 1)
    if not env.done:
        finish(env)
    return env.log


# ----- determinism and the envelope -----
def test_a_seed_and_an_action_sequence_reproduce_the_log_byte_for_byte(make_env):
    env = make_env()
    for seed in range(12):
        random.seed(1)
        first = json.dumps(scripted_log(env, seed))
        random.seed(2)  # the environment must not draw from the global generator
        random.random()
        assert json.dumps(scripted_log(env, seed)) == first
        log = json.loads(first)
        assert log == scripted_log(env, seed)  # the log is JSON native, so a saved log equals the live one


def test_reset_draws_the_hidden_state_from_the_seed_only(make_env):
    env = make_env(schedule={"mode": "seeded", "components": 1, "first_drop": [5, 40], "interval": 10})
    draws = set()
    for seed in range(30):
        env.reset(seed)
        first = (env.initial, env.window_start, env.entry_offset, json.dumps(env.clock.schedule))
        env.reset(seed)
        assert (env.initial, env.window_start, env.entry_offset, json.dumps(env.clock.schedule)) == first
        draws.add(first)
        assert env.config.within_fault_rule(env.initial) and env.initial in env.candidates
    assert len(draws) > 1


def test_the_first_envelope_carries_the_profile_and_every_envelope_has_the_pinned_keys(make_env):
    env = make_env()
    obs = env.reset(0)
    assert set(obs) == ENVELOPE_KEYS and obs["time"] == 0 and obs["episode_over"] is False and obs["open_work_orders"] == []
    assert obs["budget_left"] == env.config.budget and obs["steps_left"] == env.config.max_steps
    profile = obs["result"]
    assert set(profile) == PROFILE_KEYS and profile == env.public_profile()
    # the parameters of the rules and never the fault rule of the starts
    assert profile["processes"] == processes_block(env.config) and "min_faulty" not in json.dumps(profile)
    assert profile["levels"] == {c: list(LEVELS[c]) for c in COMPONENTS} and profile["healthy"] == dict(HEALTHY)
    assert profile["sensor_groups"] == {f"tier{t}": [HYD1_FIELDS[s] for s in sensors] for t, sensors in TIER_SENSORS.items()}
    assert profile["statistics"] == ["mean", "std"] and profile["asset"]["asset_id"] == "hyd_1"
    assert [row["name"] for row in profile["tools"]] == list(contract.load_contract()["tools"])
    assert set(profile["normal_ranges"]) == {mean_field(s) for s in SENSORS}
    obs, cost, done, info = read(env, "tier2", 1)
    assert set(obs) == ENVELOPE_KEYS and obs["time"] == 1 and info == {}
    assert cost == env.config.prices["read"]["tier2"] + env.clock.downtime_total  # a step costs its price and the downtime it let accrue
    refused, cost, done, info = env.step({"type": "teleport"})
    assert set(refused) == (ENVELOPE_KEYS - {"result"}) | {"refused"} and cost == 0 and info == {"refused": "unknown action type"}


def test_the_profile_lists_the_diagnostics_tools_only_when_the_server_is_configured(make_env):
    with_server = make_env().reset(0)["result"]["tools"]
    without = make_env(diagnostics_server=False).reset(0)["result"]["tools"]
    assert {row["server"] for row in with_server} == {contract.ACTION_SERVER, contract.DIAGNOSTICS_SERVER}
    assert {row["server"] for row in without} == {contract.ACTION_SERVER}
    assert all(row["class"] in ("read", "write") and isinstance(row["scored"], bool) for row in with_server)


def test_anchored_episodes_carry_the_day_and_start_at_the_named_recording(make_env, world):
    cycles, blocks = world
    env = anchored(make_env, anchors={"date0": "2022-07-01"})
    obs = env.reset(4)
    assert obs["day"] == "2022-07-01" and env.initial == HAND_START and env.window_start == 0 and env.entry_offset == SEGMENT
    obs, *_ = read(env, "tier1", 3)
    assert obs["day"] == "2022-07-04"
    served = events(env.log, "read")[0]["served"]
    assert [o["cycle"] for o in served] == blocks[HAND_START][:3] and [o["t"] for o in served] == [0, 1, 2]
    position = 4
    env = make_env(start={"combination": list(HAND_START), "entry_offset": (position + SEGMENT) % BLOCK_CAP})
    env.reset(0)
    read(env, "tier1", 1)
    assert events(env.log, "read")[0]["served"][0]["cycle"] == blocks[HAND_START][position]


# ----- the information barrier -----
def test_no_envelope_public_event_or_profile_holds_a_hidden_field(make_env):
    env = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [2, 2], "interval": 3},
                   failure={"patience": 2, "strict": False, "penalty": 50})
    envelopes = [env.reset(7)]
    for action in ({"type": "teleport"}, {"type": "read_sensors", "sensor_group": "tier3", "cycles": 2}, {"type": "summarize_changes_since", "cycle": 0},
                   {"type": "get_failure_modes", "component": "pump"}, {"type": "submit_diagnosis", "diagnosis": truth(env), "evidence_ids": []},
                   {"type": "create_work_order", "component": faulty_components(env.initial)[0], "prediction_id": "", "evidence_ids": []},
                   {"type": "wait_for_notification", "timeout": 8}, {"type": "get_current_time"}, {"type": "wait_for_notification", "timeout": 8},
                   {"type": "summarize_changes_since", "cycle": 2}, {"type": "get_asset_profile"},
                   {"type": "finish", "summary": "done"}):
        obs, *_ = env.step(action)
        envelopes.append(obs)
    hidden = set(HIDDEN_FIELDS)
    for obs in envelopes:
        assert not keys_of(obs) & hidden, obs
    for entry in env.log:
        assert not keys_of(env.public_event(entry)) & hidden, entry["event"]
    assert not keys_of(env.public_profile()) & hidden
    # the schedule stays hidden: a drop has no public view, and a failure names no component
    assert events(env.log, "scheduled_drop") and all(env.public_event(e) is None for e in events(env.log, "scheduled_drop"))
    assert "scheduled_drop" not in {e["event"] for e in env.public_events()}
    assert all("component" not in env.public_event(e) and "component" in e for e in events(env.log, "failure"))
    assert "played" in env.log[-1] and "played" not in env.public_event(env.log[-1])
    # the log itself keeps the hidden fields for the verifier
    assert {"seed", "hidden", "schedule", "config"} <= set(env.log[0])
    assert all({"cycle", "combination"} <= set(o) for e in events(env.log, "read") for o in e["served"])
    assert all("unnecessary" in e for e in events(env.log, "work_order_created"))
    assert all("combination_after" in e for e in events(env.log, "work_order_closed")) and "final" in env.log[-1]
    text = json.dumps(envelopes) + json.dumps([env.public_event(e) for e in env.log])
    assert not any(label in text for label in ("cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"))


def test_the_log_keeps_copies_so_a_policy_cannot_rewrite_its_diagnosis_afterwards(make_env):
    env = make_env()
    env.reset(6)
    wrong = {c: LEVELS[c][0] for c in COMPONENTS}
    obs, *_ = env.step({"type": "submit_diagnosis", "diagnosis": wrong, "evidence_ids": []})
    stored = json.dumps(events(env.log, "submit_diagnosis")[0]["diagnosis"])
    wrong["cooler"] = env.initial[0]  # the policy edits the dict it handed in
    obs["result"]["diagnosis"]["valve"] = env.initial[1]  # and the answer it was shown
    assert json.dumps(events(env.log, "submit_diagnosis")[0]["diagnosis"]) == stored


# ----- reads -----
def test_reads_serve_the_recording_of_the_current_combination_cycle_by_cycle(make_env, world):
    cycles, blocks = world
    by_index = {c.index: c for c in cycles}
    env = make_env()
    for seed in range(10):
        env.reset(seed)
        read(env, "tier1", SEGMENT + 2)
        entry = events(env.log, "read")[0]
        served = entry["served"]
        assert entry["t_start"] == 0 and entry["t_end"] == SEGMENT + 2 and entry["class"] == "read" and entry["stopped"] == 0
        assert [o["t"] for o in served] == list(range(SEGMENT + 2)) and [o["n"] for o in served] == list(range(1, SEGMENT + 3))
        assert all(o["combination"] == list(env.initial) for o in served)
        window = [o["cycle"] for o in served[:SEGMENT]]
        # the live segment of the seeded entry offset, in its order, then the wrap inside it
        assert window == [blocks[env.initial][p] for p in segment_positions(env.entry_offset + SEGMENT)]
        assert [o["cycle"] for o in served[SEGMENT:]] == window[:2] and [o["repeated"] for o in served] == [False] * SEGMENT + [True] * 2
        for o in served:
            assert o["values"] == served_values(by_index[o["cycle"]], TIER_SENSORS[1])
            assert set(o["values"]) == {f"{HYD1_FIELDS[s]}_{stat}" for s in TIER_SENSORS[1] for stat in ("mean", "std")}



def test_the_repeat_flag_stays_in_the_hidden_log_and_leaves_every_public_view(make_env):
    """A served cycle played again after its window wrapped keeps its repeat flag in the log, for the verifier and the
    walkthrough, but no answer, public event or released file carries it, since after an unchanged close the flag would
    tell the agent that a repair changed nothing."""
    env = make_env()
    env.reset(0)
    envelope, *_ = read(env, "tier1", SEGMENT + 2)
    logged = events(env.log, "read")[0]["served"]
    assert [o["repeated"] for o in logged] == [False] * SEGMENT + [True] * 2
    assert all("repeated" not in o for o in envelope["result"]["served"])
    assert all("repeated" not in o for e in env.public_events() if e["event"] == "read" for o in e["served"])
    assert "repeated" not in keys_of(released_from_env(env))

def test_the_entry_offset_varies_with_the_seed_and_a_new_combination_opens_its_live_segment(make_env, world):
    cycles, blocks = world
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 1})
    starts = set()
    for seed in range(40):
        env.reset(seed)
        assert env.window_start == 0 and 0 <= env.entry_offset < BLOCK_CAP
        starts.add((env.initial, env.window_start, env.entry_offset))
    assert len({entry for _combination, _start, entry in starts}) > 1
    env = anchored(make_env, durations={**EpisodeConfig().durations, "lead_time": {c: 0 for c in COMPONENTS}, "repair_downtime": {c: 0 for c in COMPONENTS}})
    env.reset(0)
    submit_truth(env)
    order(env, "cooler")
    read(env, "tier1", 2)
    served = events(env.log, "read")[0]["served"]
    live = [blocks[repaired(HAND_START, "cooler")][p] for p in segment_positions(env.entry_offset + SEGMENT)]
    assert all(o["combination"] == list(repaired(HAND_START, "cooler")) for o in served) and [o["cycle"] for o in served] == live[:2]


def test_a_later_window_shares_the_entry_offset_and_draws_nothing(make_env, world):
    """A draw per window would tell the windows of one episode apart and move every later draw of its generator."""
    _cycles, blocks = world
    env = make_env()
    env.reset(0)
    rng = random.Random(1)
    before = rng.getstate()
    env.source.open(next(c for c in blocks if c != env.initial), 3, rng)
    assert rng.getstate() == before and env.source.window["offset"] == env.entry_offset and env.source.window["t"] == 3


def test_every_combination_has_one_start_state_per_entry_offset_whatever_the_length_of_its_block(make_env, world):
    source = make_env().source
    assert {tuple(source.starts(c)) for c in world[1]} == {tuple((0, entry) for entry in range(BLOCK_CAP))}


def test_a_combination_with_too_few_recordings_for_every_entry_offset_is_refused(world):
    """The refusal names the number of the short blocks and the first of them, as the porting page says."""
    cycles, blocks = world
    short = next(iter(blocks))
    with pytest.raises(ValueError, match=re.escape(f"at least {BLOCK_CAP} stable recordings, and 1 hold fewer, first {short}")):
        HydraulicRigEnv(cycles, {**blocks, short: blocks[short][:BLOCK_CAP - 1]}, EpisodeConfig())


def test_normal_ranges_are_the_band_quantiles_of_the_healthy_block_on_served_values(make_env, world):
    cycles, blocks = world
    by_index = {c.index: c for c in cycles}
    env = make_env()
    served = [served_values(by_index[i], SENSORS) for i in blocks[HEALTHY_COMBO]]
    for s in SENSORS:
        values = [v[mean_field(s)] for v in served]
        band = env.normal_ranges[mean_field(s)]
        assert band == {"low": percentile(values, BAND_QUANTILES[0]), "high": percentile(values, BAND_QUANTILES[1])}
        assert band["low"] <= band["high"]
    assert env.reset(0)["result"]["normal_ranges"] == env.normal_ranges


def test_no_healthy_recording_is_outside_a_band_while_every_single_fault_recording_is_on_every_tier(make_env, world):
    """The normal range of a channel mean spans its smallest to its largest served value over the stable healthy
    recordings, so no healthy recording marks a channel or raises an alarm on any tier, while every served recording of every
    single fault level leaves a channel of every tier outside its normal range, so a fault alarms in its first played cycle whatever
    group was read, which one read of each group on every single fault start shows. A rig repaired to health raises no alarm
    after the close."""
    assert BAND_QUANTILES == (0.0, 1.0)
    cycles, blocks = world
    by_index = {c.index: c for c in cycles}
    env = make_env()

    def outside(index, tier):
        values = served_values(by_index[index], TIER_SENSORS[tier])
        return [f for f in (mean_field(s) for s in TIER_SENSORS[tier])
                if not env.normal_ranges[f]["low"] <= values[f] <= env.normal_ranges[f]["high"]]

    singles = [tuple(c) for c in env.candidates if len(faulty_components(c)) == 1]
    assert singles
    for tier in TIER_SENSORS:
        assert not [i for i in blocks[HEALTHY_COMBO] if outside(i, tier)], tier
        assert all(outside(i, tier) for combination in singles for i in blocks[combination][:BLOCK_CAP]), tier
    for combination in singles:
        for group in TIER_SENSORS:
            faulty_rig = anchored(make_env, combination)
            faulty_rig.reset(0)
            read(faulty_rig, f"tier{group}", 1)
            assert [e["t_start"] for e in events(faulty_rig.log, "alarm")] == [0] * len(events(faulty_rig.log, "alarm")) != [], (combination, group)
    repaired_rig = anchored(make_env, (3, 100, 0, 130))
    repaired_rig.reset(0)
    for group in ("tier1", "tier2", "tier3"):
        read(repaired_rig, group, 1)
    submit_truth(repaired_rig)
    order(repaired_rig, "cooler")
    advance(repaired_rig, 20)
    closed = [o.closed_t for o in repaired_rig.clock.orders if o.component == "cooler" and o.closed_t is not None]
    assert closed and not [e for e in events(repaired_rig.log, "alarm") if e["t_end"] >= closed[0]]


# ----- refusals -----
def test_malformed_and_out_of_order_actions_are_refused_without_charge_or_state_change(make_env):
    env = make_env()
    env.reset(5)
    right = truth(env)
    bad = [
        ({"type": "teleport"}, "unknown action type"), ({}, "unknown action type"), ({"type": ["read_sensors"]}, "unknown action type"),
        ({"type": "read_sensors", "sensor_group": "tier4", "cycles": 1}, "unknown sensor group"),
        ({"type": "read_sensors", "sensor_group": 1, "cycles": 1}, "unknown sensor group"),
        ({"type": "read_sensors", "sensor_group": "tier1", "cycles": 0}, "cycles must be a positive integer"),
        ({"type": "read_sensors", "sensor_group": "tier1", "cycles": "two"}, "cycles must be a positive integer"),
        ({"type": "read_sensors", "sensor_group": "tier1", "cycles": True}, "cycles must be a positive integer"),
        ({"type": "read_sensors", "sensor_group": "tier1"}, "cycles must be a positive integer"),
        ({"type": "wait_for_notification", "timeout": 0}, "timeout must be a positive integer, or empty for the cycles left before the horizon"),
        ({"type": "wait_for_notification", "timeout": 2.5}, "timeout must be a positive integer, or empty for the cycles left before the horizon"),
        ({"type": "advance_time", "cycles": 2}, "unknown action type"),  # removed, since no tool may set the time
        ({"type": "summarize_changes_since", "cycle": 1}, "cycle must be an integer between 0 and the current time"),
        ({"type": "summarize_changes_since", "cycle": -1}, "cycle must be an integer between 0 and the current time"),
        ({"type": "summarize_changes_since", "cycle": "0.5"}, "cycle must be an integer between 0 and the current time"),
        ({"type": "get_failure_modes", "component": "turbine"}, "unknown component"),
        ({"type": "get_failure_modes", "component": None}, "unknown component"),
        ({"type": "submit_diagnosis", "diagnosis": {"cooler": 100}, "evidence_ids": []}, DIAGNOSIS_COMPONENTS_REASON),
        ({"type": "submit_diagnosis", "diagnosis": {c: 999 for c in COMPONENTS}, "evidence_ids": []}, DIAGNOSIS_LEVEL_REASON),
        ({"type": "submit_diagnosis", "diagnosis": {**right, "pump": True}, "evidence_ids": []}, DIAGNOSIS_LEVEL_REASON),
        ({"type": "submit_diagnosis", "diagnosis": {**right, "cooler": right["cooler"] + 0.5}, "evidence_ids": []}, DIAGNOSIS_LEVEL_REASON),
        ({"type": "submit_diagnosis", "diagnosis": {**right, "valve": "high"}, "evidence_ids": []}, DIAGNOSIS_LEVEL_REASON),
        ({"type": "submit_diagnosis", "diagnosis": right, "evidence_ids": "obs/1/x"}, EVIDENCE_IDS_REASON),
        ({"type": "submit_diagnosis", "diagnosis": right}, EVIDENCE_IDS_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier9", "cycles": 1}, "evidence_ids": []},
         SEPARATING_TEST_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": 0}, "evidence_ids": []},
         SEPARATING_TEST_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": "tier1", "evidence_ids": []},
         SEPARATING_TEST_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": ["tier1"], "cycles": 1}, "evidence_ids": []},
         SEPARATING_TEST_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": {"a": 1}, "cycles": 1}, "evidence_ids": []},
         SEPARATING_TEST_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": 1}, "evidence_ids": [1]},
         EVIDENCE_IDS_REASON),
        ({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []}, ORDER_BEFORE_WRITE_REASON),
        ({"type": "finish", "summary": None}, "summary must be a string"),
        ({"type": "finish"}, "summary must be a string"),
    ]
    before = (env.time, env.cost, env.hidden, env.clock._window["played"], len(env.clock.orders), env.diagnosed, env.declared)
    for action, reason in bad:
        obs, cost, done, info = env.step(action)
        assert info == {"refused": reason} and obs["refused"] == reason and cost == 0 and not done, action
        assert (env.time, env.cost, env.hidden, env.clock._window["played"], len(env.clock.orders), env.diagnosed, env.declared) == before
        logged = env.log[-1]
        assert logged["event"] == "refused" and logged["action"] == action and logged["reason"] == reason and logged["cost"] == 0
        assert logged["step"] == env.steps and logged["t_start"] == logged["t_end"] == env.time and logged["class"] in ("read", "write")
        assert env.public_event(logged)["action"] == action
    assert env.steps == len(bad) and len(events(env.log, "refused")) == len(bad)
    submit_truth(env)
    after = [
        ({"type": "submit_diagnosis", "diagnosis": right, "evidence_ids": []}, WRITE_LOCK_REASON),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": 1}, "evidence_ids": []}, WRITE_LOCK_REASON),
        ({"type": "create_work_order", "component": "turbine", "prediction_id": "", "evidence_ids": []}, "unknown component"),
        ({"type": "create_work_order", "component": "pump", "prediction_id": None, "evidence_ids": []}, "prediction_id must be a string, empty when there is no prediction"),
        ({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": "x"}, EVIDENCE_IDS_REASON),
    ]
    for action, reason in after:
        obs, cost, done, info = env.step(action)
        assert info == {"refused": reason} and cost == 0 and not done, action
    component = faulty_components(env.initial)[0]
    obs, cost, *_ = order(env, component)
    assert cost == env.config.order_price(component) + env.config.prices["visit"] and obs["result"]["work_order"]["status"] == "open"
    obs, cost, done, info = order(env, component)
    assert info == {"refused": f"a work order on the {component} is open"} and cost == 0


def test_a_declaration_locks_the_diagnosis_and_leaves_reads_and_orders_allowed(make_env):
    env = make_env()
    env.reset(9)
    test = {"sensor_group": "tier3", "cycles": 2}
    obs, cost, done, info = env.step({"type": "declare_insufficient_evidence", "separating_test": test, "evidence_ids": ["alarm/0/x"]})
    assert obs["result"] == {"separating_test": test, "evidence_ids": ["alarm/0/x"]} and cost == 0 and env.declared and not env.diagnosed
    entry = events(env.log, "declare_insufficient_evidence")[0]
    assert entry["separating_test"] == test and entry["class"] == "write"
    obs, cost, done, info = submit_truth(env)
    assert info == {"refused": WRITE_LOCK_REASON}
    assert not read(env, "tier1", 1)[3].get("refused")
    component = faulty_components(env.initial)[0]
    assert not order(env, component)[3].get("refused")


# ----- the ways an episode ends -----
def test_an_action_the_budget_cannot_pay_is_refused_and_ends_the_episode(make_env):
    env = make_env(budget=5)
    env.reset(1)
    obs, cost, done, info = read(env, "tier3", 1)
    assert done and cost == 0 and obs["refused"] == "budget" and obs["episode_over"]
    assert [e["event"] for e in env.log[-2:]] == ["refused", "end"] and env.log[-1]["reason"] == "budget" and env.log[-2]["reason"] == "budget"
    assert info["reason"] == "budget" and env.cost == 0
    pump_only = (100, 100, 1, 130)
    env = anchored(make_env, pump_only, budget=EpisodeConfig().order_price("pump") - 1)
    env.reset(0)
    submit_truth(env)
    obs, cost, done, info = order(env, "pump")
    assert done and info["reason"] == "budget" and env.hidden == list(pump_only) and not env.clock.orders


def test_the_horizon_ends_the_episode_and_the_clock_never_passes_it(make_env):
    env = make_env()
    env.reset(2)
    obs, cost, done, info = advance(env, 1000)
    assert done and env.time == env.config.horizon == obs["time"] and info["reason"] == "horizon"
    assert env.log[-2]["event"] == "wait" and env.log[-2]["t_end"] == env.config.horizon and env.log[-1]["time"] == env.config.horizon
    env = make_env(horizon=5)
    env.reset(2)
    obs, cost, done, info = read(env, "tier1", 8)
    assert done and info["reason"] == "horizon" and len(obs["result"]["served"]) == 5
    # the cycles that returned a reading are charged, the five before the horizon, while the entry keeps the cycles asked for
    assert events(env.log, "read")[0]["cost"] == 5 * env.config.prices["read"]["tier1"] and events(env.log, "read")[0]["cycles"] == 8
    assert [o["t"] for o in obs["result"]["served"]] == [0, 1, 2, 3, 4]


def test_the_step_cap_counts_refusals_and_ends_an_episode_that_never_diagnoses(make_env):
    env = make_env(max_steps=3)
    env.reset(11)
    assert not env.step({"type": "finish", "summary": 3})[2]
    assert not env.step({"type": "teleport"})[2]
    obs, cost, done, info = read(env, "tier1", 1)
    assert done and info["reason"] == "max_steps" and obs["steps_left"] == 0
    assert events(env.log, "read")[0]["cost"] == env.config.prices["read"]["tier1"]  # the capped step is carried out and charged
    assert [e["event"] for e in env.log[1:]] == ["refused", "refused", "read", "end"]


def test_finish_ends_the_episode_after_the_clock_has_run_to_the_last_close(make_env):
    """On a row with a clock finish settles the open work orders, so the cooler ordered just before the
    finish closes and the end entry holds the rig after that close."""
    env = anchored(make_env)
    env.reset(0)
    submit_truth(env)
    order(env, "cooler")
    close_t = env.clock.orders[0].close_t
    obs, cost, done, info = finish(env, "cooler ordered")
    assert done and info["reason"] == "finish" and obs["result"] == {"summary": "cooler ordered"} and obs["episode_over"]
    assert obs["open_work_orders"] == [] and env.time == close_t > 0 and env.log[-1]["final"] == list(repaired(HAND_START, "cooler"))
    assert [n["id"] for n in obs["notifications"] if n["event"] == "work_order_closed"] == ["wo-1"]
    assert events(env.log, "finish")[0]["summary"] == "cooler ordered" and events(env.log, "finish")[0]["class"] == "write"


def test_the_end_entry_carries_the_episode_cost_and_the_downtime(make_env):
    env = anchored(make_env)
    scripted_log(env, 0)
    end = env.log[-1]
    prices = sum(e["cost"] for e in env.log[1:-1] if e["step"] is not None)
    assert end["event"] == "end" and end["reason"] == "finish" and end["final"] == list(HEALTHY_COMBO)
    assert end["cost"] == env.cost == prices + end["downtime_total"]
    assert end["downtime_total"] == sum(d["charge"] for d in end["downtime_by_cycle"]) > 0
    assert all(d["charge"] > 0 for d in end["downtime_by_cycle"]) and end["downtime_by_cycle"] == env.clock.downtime
    assert [p["t"] for p in end["played"]] == list(range(env.time)) and all(env.cycle_at(p["t"]) == p["cycle"] for p in end["played"])
    assert all(tuple(p["combination"]) == env.combination_at(p["t"]) for p in end["played"]) and env.cycle_at(env.time) is None


# ----- the change summary and the failure modes -----
def test_the_summary_reports_alarms_of_the_paid_groups_from_the_played_recording_and_the_closed_orders(make_env, world):
    cycles, blocks = world
    by_index = {c.index: c for c in cycles}
    env = anchored(make_env)
    env.reset(0)
    obs, *_ = summary(env, 0)
    assert obs["result"] == {"since": 0, "time": 0, "cycles_elapsed": 0, "alarms": [], "work_orders_closed": []}
    advance(env, 4)
    assert summary(env, 1)[0]["result"]["alarms"] == []  # no group was paid for yet
    read(env, "tier1", 1)
    obs, *_ = summary(env, 1)
    expected = {}
    for t, index in env.clock.played:
        if t < 1:
            continue
        values = served_values(by_index[index], SENSORS)
        for field in (mean_field(s) for s in TIER_SENSORS[1]):
            band = env.normal_ranges[field]
            if field not in expected and not band["low"] <= values[field] <= band["high"]:
                expected[field] = t
    assert obs["result"]["alarms"] == [{"channel": f, "first_t": t} for f, t in sorted(expected.items(), key=lambda kv: (kv[1], kv[0]))]
    assert obs["result"]["cycles_elapsed"] == 4 and expected  # a faulty cooler leaves its tier1 channels outside the normal range
    submit_truth(env)
    order(env, "cooler")
    advance(env, 10)
    obs, *_ = summary(env, 0)
    closed = obs["result"]["work_orders_closed"]
    assert [o["id"] for o in closed] == ["wo-1"] and closed[0]["status"] == "closed" and closed[0]["source"] == "agent"
    assert summary(env, closed[0]["closed_t"] + 1)[0]["result"]["work_orders_closed"] == []
    entry = events(env.log, "summary")[-1]
    assert entry["since"] == closed[0]["closed_t"] + 1 and entry["class"] == "read" and entry["cost"] == 0


def test_the_truth_model_releases_exactly_the_failure_mode_identifiers_the_tool_serves(make_env):
    """The provenance check of the verifier reads the truth model, so the identifiers it releases after a get_failure_modes
    call are the identifiers the tool served, for every component and for one."""
    from CausalDiagnosis.datasets.dataset_loader.ground_truth import LevelsTruth

    env = make_env()
    env.reset(0)
    served = env.step({"type": "get_failure_modes", "component": ""})[0]["result"]["failure_modes"]
    truth_model = LevelsTruth(env.profile)
    assert truth_model.fmea_ids("") == {m["evidence_id"] for m in served}
    assert truth_model.fmea_ids("pump") == {m["evidence_id"] for m in served if m["component"] == "pump"}
    assert len(served) < len(env.profile.failure_modes)  # the profile lists states the tool never serves


def test_failure_modes_serve_the_fault_codes_of_the_knowledge_base(make_env):
    env = make_env()
    env.reset(0)
    obs, cost, *_ = env.step({"type": "get_failure_modes", "component": ""})
    assert cost == 0 and obs["result"]["component"] is None
    modes = obs["result"]["failure_modes"]
    assert [m["code"] for m in modes] == [e["code"] for e in knowledge.FAULT_CODES]
    assert all(m["evidence_id"] == f"fmea/{knowledge.FMEA_SOURCE}/{m['code']}" for m in modes)
    pump = env.step({"type": "get_failure_modes", "component": "pump"})[0]["result"]
    assert pump["component"] == "pump" and {m["component"] for m in pump["failure_modes"]} == {"pump"}
    assert pump["failure_modes"][0]["channels"] == [HYD1_FIELDS[s] for s in SENSORS if "pump" in knowledge.INDICATES.get(s, ())]
    assert events(env.log, "failure_modes")[-1]["component"] == "pump"


# ----- clock modes -----
def test_instant_mode_charges_one_cycle_per_step_whatever_the_tool(make_env):
    env = anchored(make_env, clock_mode="instant")
    env.reset(0)
    obs, *_ = read(env, "tier1", 3)
    # the step's cycle plays and serves one recording; the two further observations come from the window at the new time
    assert obs["time"] == 1 and [o["t"] for o in obs["result"]["served"]] == [0, 1, 1] and len({o["n"] for o in obs["result"]["served"]}) == 3
    assert env.step({"type": "teleport"})[0]["time"] == 2  # a refused step takes the agent's cycle too
    assert wait(env, 10)[0]["time"] == 3  # a wait takes its one cycle like every other call
    assert env.step({"type": "get_current_time"})[0]["time"] == 3  # the one call that never moves the clock
    assert env.step({"type": "get_asset_profile"})[0]["time"] == 4
    assert submit_truth(env)[0]["time"] == 5
    cycles = [o["cycle"] for e in events(env.log, "read") for o in e["served"]]
    assert len(set(cycles)) == 3  # the three observations are three recorded cycles of the window


def test_generation_time_mode_charges_the_thinking_cycles_before_the_action_and_other_modes_ignore_them(make_env):
    declared = anchored(make_env)
    declared.reset(0)
    obs, *_ = declared.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 2}, think_cycles=5)
    assert obs["time"] == 2 and events(declared.log, "read")[0]["think_cycles"] == 0
    timed = anchored(make_env, clock_mode="generation_time")
    timed.reset(0)
    obs, cost, done, info = timed.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 2}, think_cycles=5)
    entry = events(timed.log, "read")[0]
    assert obs["time"] == 7 and entry["t_start"] == 5 and entry["t_end"] == 7 and entry["think_cycles"] == 5
    assert [o["t"] for o in obs["result"]["served"]] == [5, 6]
    with pytest.raises(TypeError):
        timed.step({"type": "wait_for_notification", "timeout": 1}, think_cycles=-1)
    obs, cost, done, info = timed.step({"type": "finish", "summary": "late"}, think_cycles=timed.config.horizon)
    assert done and info["reason"] == "horizon" and obs["refused"] == "the episode ended before the action ran" and timed.time == timed.config.horizon


def test_zero_read_duration_serves_cycles_from_the_window_without_moving_the_clock(make_env):
    env = anchored(make_env, durations={**EpisodeConfig().durations, "read_per_cycle": 0})
    env.reset(0)
    obs, *_ = read(env, "tier2", 3)
    assert obs["time"] == 0 and [o["t"] for o in obs["result"]["served"]] == [0, 0, 0]
    obs, *_ = read(env, "tier2", 1)
    cycles = [o["cycle"] for e in events(env.log, "read") for o in e["served"]]
    assert len(set(cycles)) == 4 and obs["time"] == 0


def test_a_read_that_takes_no_time_is_refused_above_the_horizon(make_env):
    """Where the clock does not cover the cycles of a read, a read of zero duration per cycle or any read of the instant mode,
    they come at once, so one call never serves more observations than the horizon holds cycles: a read of one cycle more
    than the horizon is refused and counted with no price paid, a read of the horizon is served. The cycle that the instant mode
    gives every action still charges the downtime of the hidden condition to the ledger."""
    zero = {**EpisodeConfig().durations, "read_per_cycle": 0}
    for overrides in ({"durations": zero}, {"clock_mode": "instant"}):
        env = make_env(**overrides)
        env.reset(0)
        horizon = env.config.horizon
        obs, cost, done, info = read(env, "tier1", horizon + 1)
        assert info == {"refused": "cycles must not exceed the horizon"} and env.prices_paid == 0 and not done and env.steps == 1, overrides
        assert env.log[-1]["event"] == "refused" and env.log[-1]["reason"] == "cycles must not exceed the horizon"
        obs, *_ = read(env, "tier1", horizon)
        assert "refused" not in obs and len(obs["result"]["served"]) == horizon, overrides


def test_a_read_whose_served_cycles_the_budget_can_pay_is_accepted(make_env):
    """The budget decides on the price of the cycles that return a reading: a read of eight cycles on a horizon of five costs
    five times the price, so a budget of that price pays it, and the budget left falls by it. The downtime of the hidden
    condition goes to the ledger and not to the price."""
    price = EpisodeConfig().prices["read"]["tier1"]
    env = make_env(horizon=5, budget=5 * price)
    env.reset(2)
    obs, _charges, done, info = read(env, "tier1", 8)
    entry = next(e for e in reversed(env.log) if e["event"] == "read")
    assert "refused" not in obs and entry["cost"] == 5 * price and entry["cycles"] == 8 and len(obs["result"]["served"]) == 5
    assert info.get("reason") == "horizon" and env.prices_paid == 5 * price and obs["budget_left"] == 0


def test_an_integer_argument_is_read_by_its_value_whatever_type_writes_it(make_env):
    """Every integer argument of a tool is read by its value (the cycles of a read and of a declaration, the timeout of a
    wait, the cycle of a change summary, the level of every component): its decimal string, its string with a zero fraction
    and its float give the same log entry and the same answer as the integer, and True, "1e2", null and a blank string are
    refused with the sentence of the argument, null being the empty timeout of a wait."""
    def played(action, seed=5):
        env = make_env()
        env.reset(seed)
        if action["type"] == "summarize_changes_since":
            read(env, "tier1", 1)
        obs, cost, done, info = env.step(action)
        entry = {k: v for k, v in env.log[-1].items() if k != "action"}
        return obs, entry

    def with_value(kind, value):
        if kind == "read":
            return {"type": "read_sensors", "sensor_group": "tier1", "cycles": value}
        if kind == "wait":
            return {"type": "wait_for_notification", "timeout": value}
        if kind == "summary":
            return {"type": "summarize_changes_since", "cycle": value}
        return {"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": value}, "evidence_ids": []}

    for kind, n, reason in (("read", 2, "cycles must be a positive integer"),
                            ("wait", 2, "timeout must be a positive integer, or empty for the cycles left before the horizon"),
                            ("summary", 1, "cycle must be an integer between 0 and the current time"),
                            ("declare", 2, SEPARATING_TEST_REASON)):
        plain = played(with_value(kind, n))
        assert plain[1]["event"] != "refused", kind
        for written in (str(n), f" {n}.0 ", float(n)):
            assert played(with_value(kind, written)) == plain, (kind, written)
        for wrong in (True, "1e2", "  ", "[2]"):
            obs, entry = played(with_value(kind, wrong))
            assert entry["event"] == "refused" and entry["reason"] == reason, (kind, wrong)
        obs, entry = played(with_value(kind, None))
        assert (entry["event"] == "refused") == (kind != "wait"), kind
    env = make_env()
    env.reset(5)
    right = truth(env)
    obs, *_ = env.step({"type": "submit_diagnosis", "diagnosis": {c: f"{level}.0" for c, level in right.items()}, "evidence_ids": []})
    assert obs["result"]["diagnosis"] == right and events(env.log, "submit_diagnosis")[0]["diagnosis"] == right


def test_a_level_written_as_a_string_is_scored_as_the_number(make_env):
    """The true levels sent as strings are accepted, stored as integers and scored right."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import verify

    env = make_env()
    env.reset(5)
    right = truth(env)
    read(env, "tier1", 1)
    env.step({"type": "submit_diagnosis", "diagnosis": {c: str(level) for c, level in right.items()}, "evidence_ids": []})
    finish(env)
    stored = events(env.log, "submit_diagnosis")[0]["diagnosis"]
    assert stored == right and all(isinstance(v, int) for v in stored.values())
    verdict = verify(env.log, released_from_env(env), env.oracle(), None)
    assert verdict["verified"] and verdict["diagnosis_all_correct"]


def _healthy_copy(path, out, profile):
    """The data file at path with every recording of a faulty combination given the features of the healthy recording at the
    same place of its block, under the same index, so that no recording of out leaves the normal range."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    keys = [profile.records["label_keys"][c] for c in profile.components]
    combination = [tuple(row["labels"][k] for k in keys) for row in rows]
    healthy = [row for row, combo in zip(rows, combination) if combo == profile.healthy_state()]
    place = {}
    for row, combo in zip(rows, combination):
        place[combo] = place.get(combo, -1) + 1
        if combo != profile.healthy_state():
            row["features"] = healthy[place[combo]]["features"]
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return out


def _alarms_of_a_read(profile, path, seed):
    """The alarm entries after a read of every channel and a wait, from the seed, over the recordings of the data file."""
    cycles = records.load_cycles(path, profile)
    env = HydraulicRigEnv(cycles, records.build_blocks(cycles), profile.default_config())
    env.reset(seed)
    widest = max(env.config.sensor_groups, key=lambda g: len(env.config.sensor_groups[g]))
    env.step({"type": "read_sensors", "sensor_group": widest, "cycles": 2})
    env.step({"type": "wait_for_notification", "timeout": 3})
    return events(env.log, "alarm")


def test_the_excursions_of_a_recording_belong_to_its_source(tiny, tmp_path):
    """Two environments over two worlds of the tiny asset whose recordings differ under the same indices alarm on their own
    values: in the second world every recording of a faulty combination carries the values of a healthy one, so nothing
    leaves the normal range there, while the first world alarms, before the second is played and after it. A store of the
    excursions of a recording kept by its index alone fails."""
    first = write_recordings(tmp_path / "first.jsonl", tiny)
    second = _healthy_copy(first, tmp_path / "second.jsonl", tiny)
    alarmed = _alarms_of_a_read(tiny, first, 0)
    assert alarmed and _alarms_of_a_read(tiny, second, 0) == [] and _alarms_of_a_read(tiny, first, 0) == alarmed
