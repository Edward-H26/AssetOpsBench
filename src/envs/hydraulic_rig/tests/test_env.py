"""HydraulicRigEnv (src/envs/hydraulic_rig/env.py): determinism and replay, the envelope, the information barrier, the
served recordings, every refusal, the ways an episode ends, the change summary, the failure modes and the clock modes.
No test states a value computed from the rig data; data-dependent assertions are relations."""
import json
import random

import pytest

from envs.hydraulic_rig import knowledge
from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, percentile
from envs.hydraulic_rig.env import (BAND_QUANTILES, BLOCK_CAP, HEALTHY_COMBO, HIDDEN_FIELDS, ORDER_BEFORE_WRITE_REASON, WRITE_LOCK_REASON,
                                    HydraulicRigEnv, mean_field, served_values)
from servers import contract
from servers.hydraulic_rig.names import HYD1_FIELDS

from .conftest import advance, events, faulty_components, finish, keys_of, order, read, repaired, submit_truth, summary, truth

ENVELOPE_KEYS = {"time", "budget_left", "steps_left", "open_work_orders", "episode_over", "result"}
PROFILE_KEYS = {"asset_id", "asset", "components", "levels", "healthy", "sensor_groups", "statistics", "prices", "durations", "downtime_rate",
                "budget", "horizon", "max_steps", "fault_rule", "clock_mode", "seconds_per_cycle", "tools", "healthy_bands"}
HAND_START = (3, 90, 0, 130)  # cooler and valve faulty: a start every test can anchor on


def anchored(make_env, combination=HAND_START, **overrides):
    return make_env(start={"combination": list(combination), "block_offset": 0}, **overrides)


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
    env = make_env(schedule={"mode": "seeded", "components": 1, "first_drop": [5, 40], "interval": 10},
                   external_events={"mode": "seeded", "probability": 0.5, "window": [5, 50]})
    draws = set()
    for seed in range(30):
        env.reset(seed)
        first = (env.initial, env.window_start, env.entry_offset, json.dumps(env.clock.schedule), json.dumps(env.clock.external))
        env.reset(seed)
        assert (env.initial, env.window_start, env.entry_offset, json.dumps(env.clock.schedule), json.dumps(env.clock.external)) == first
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
    assert profile["levels"] == {c: list(LEVELS[c]) for c in COMPONENTS} and profile["healthy"] == dict(HEALTHY)
    assert profile["sensor_groups"] == {f"tier{t}": [HYD1_FIELDS[s] for s in sensors] for t, sensors in TIER_SENSORS.items()}
    assert profile["statistics"] == ["mean", "std"] and profile["asset"]["asset_id"] == "hyd_1"
    assert [row["name"] for row in profile["tools"]] == list(contract.load_contract()["tools"])
    assert set(profile["healthy_bands"]) == {mean_field(s) for s in SENSORS}
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
    assert obs["day"] == "2022-07-01" and env.initial == HAND_START and env.window_start == 0 and env.entry_offset == 0
    obs, *_ = read(env, "tier1", 3)
    assert obs["day"] == "2022-07-04"
    served = events(env.log, "read")[0]["served"]
    assert [o["cycle"] for o in served] == blocks[HAND_START][:3] and [o["t"] for o in served] == [0, 1, 2]
    offset = 4
    env = make_env(start={"combination": list(HAND_START), "block_offset": offset})
    env.reset(0)
    read(env, "tier1", 1)
    assert events(env.log, "read")[0]["served"][0]["cycle"] == blocks[HAND_START][offset]


# ----- the information barrier -----
def test_no_envelope_public_event_or_profile_holds_a_hidden_field(make_env):
    env = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [2, 2], "interval": 3},
                   external_events={"mode": "seeded", "probability": 1.0, "window": [3, 3]}, failure={"patience": 2, "strict": False, "penalty": 50})
    envelopes = [env.reset(7)]
    for action in ({"type": "teleport"}, {"type": "read_sensors", "sensor_group": "tier3", "cycles": 2}, {"type": "summarize_changes_since", "cycle": 0},
                   {"type": "get_failure_modes", "component": "pump"}, {"type": "submit_diagnosis", "diagnosis": truth(env), "evidence_ids": []},
                   {"type": "create_work_order", "component": faulty_components(env.initial)[0], "prediction_id": "", "evidence_ids": []},
                   {"type": "advance_time", "cycles": 8}, {"type": "summarize_changes_since", "cycle": 2}, {"type": "get_asset_profile"},
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
    assert {"seed", "hidden", "schedule", "external_events", "config"} <= set(env.log[0])
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
        read(env, "tier1", BLOCK_CAP + 2)
        entry = events(env.log, "read")[0]
        served = entry["served"]
        assert entry["t_start"] == 0 and entry["t_end"] == BLOCK_CAP + 2 and entry["class"] == "read" and entry["stopped"] == 0
        assert [o["t"] for o in served] == list(range(BLOCK_CAP + 2)) and [o["n"] for o in served] == list(range(1, BLOCK_CAP + 3))
        assert all(o["combination"] == list(env.initial) and o["cycle"] in blocks[env.initial] for o in served)
        window = [o["cycle"] for o in served[:BLOCK_CAP]]
        assert sorted(window) == list(range(min(window), min(window) + BLOCK_CAP))  # a window of consecutive recorded cycles
        breaks = [(a, b) for a, b in zip(window, window[1:]) if b != a + 1]
        assert breaks in ([], [(max(window), min(window))])  # entered at a seeded offset, wrapping once inside the window
        assert [o["cycle"] for o in served[BLOCK_CAP:]] == window[:2] and [o["repeated"] for o in served] == [False] * BLOCK_CAP + [True] * 2
        for o in served:
            assert o["values"] == served_values(by_index[o["cycle"]], TIER_SENSORS[1])
            assert set(o["values"]) == {f"{HYD1_FIELDS[s]}_{stat}" for s in TIER_SENSORS[1] for stat in ("mean", "std")}


def test_the_window_start_and_offset_vary_with_the_seed_and_a_new_combination_opens_a_new_window(make_env, world):
    cycles, blocks = world
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 1})
    starts = set()
    for seed in range(40):
        env.reset(seed)
        assert 0 <= env.window_start <= len(blocks[env.initial]) - BLOCK_CAP and 0 <= env.entry_offset < BLOCK_CAP
        starts.add((env.initial, env.window_start, env.entry_offset))
    assert len(starts) > 1
    env = anchored(make_env, durations={**EpisodeConfig().durations, "lead_time": {c: 0 for c in COMPONENTS}, "repair_downtime": {c: 0 for c in COMPONENTS}})
    env.reset(0)
    submit_truth(env)
    order(env, "cooler")
    read(env, "tier1", 2)
    served = events(env.log, "read")[0]["served"]
    assert all(o["combination"] == list(repaired(HAND_START, "cooler")) and o["cycle"] in blocks[repaired(HAND_START, "cooler")] for o in served)


def test_healthy_bands_are_the_band_quantiles_of_the_healthy_block_on_served_values(make_env, world):
    cycles, blocks = world
    by_index = {c.index: c for c in cycles}
    env = make_env()
    served = [served_values(by_index[i], SENSORS) for i in blocks[HEALTHY_COMBO]]
    for s in SENSORS:
        values = [v[mean_field(s)] for v in served]
        band = env.healthy_bands[mean_field(s)]
        assert band == {"low": percentile(values, BAND_QUANTILES[0]), "high": percentile(values, BAND_QUANTILES[1])}
        assert band["low"] <= band["high"]
    assert env.reset(0)["result"]["healthy_bands"] == env.healthy_bands


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
        ({"type": "read_sensors", "sensor_group": "tier1", "cycles": "2"}, "cycles must be a positive integer"),
        ({"type": "read_sensors", "sensor_group": "tier1", "cycles": True}, "cycles must be a positive integer"),
        ({"type": "read_sensors", "sensor_group": "tier1"}, "cycles must be a positive integer"),
        ({"type": "advance_time", "cycles": 0}, "cycles must be a positive integer"),
        ({"type": "advance_time", "cycles": 2.0}, "cycles must be a positive integer"),
        ({"type": "summarize_changes_since", "cycle": 1}, "cycle must be an integer between 0 and the current time"),
        ({"type": "summarize_changes_since", "cycle": -1}, "cycle must be an integer between 0 and the current time"),
        ({"type": "summarize_changes_since", "cycle": "0"}, "cycle must be an integer between 0 and the current time"),
        ({"type": "get_failure_modes", "component": "turbine"}, "unknown component"),
        ({"type": "get_failure_modes", "component": None}, "unknown component"),
        ({"type": "submit_diagnosis", "diagnosis": {"cooler": 100}, "evidence_ids": []}, "diagnosis must give one level per component"),
        ({"type": "submit_diagnosis", "diagnosis": {c: 999 for c in COMPONENTS}, "evidence_ids": []}, "diagnosis level not in the component's levels"),
        ({"type": "submit_diagnosis", "diagnosis": {**right, "pump": True}, "evidence_ids": []}, "diagnosis level not in the component's levels"),
        ({"type": "submit_diagnosis", "diagnosis": {**right, "cooler": float(right["cooler"])}, "evidence_ids": []}, "diagnosis level not in the component's levels"),
        ({"type": "submit_diagnosis", "diagnosis": {**right, "valve": str(right["valve"])}, "evidence_ids": []}, "diagnosis level not in the component's levels"),
        ({"type": "submit_diagnosis", "diagnosis": right, "evidence_ids": "obs/1/x"}, "evidence_ids must be a list of strings"),
        ({"type": "submit_diagnosis", "diagnosis": right}, "evidence_ids must be a list of strings"),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier9", "cycles": 1}, "evidence_ids": []},
         "separating_test must name a sensor group and a positive number of cycles"),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": 0}, "evidence_ids": []},
         "separating_test must name a sensor group and a positive number of cycles"),
        ({"type": "declare_insufficient_evidence", "separating_test": "tier1", "evidence_ids": []},
         "separating_test must name a sensor group and a positive number of cycles"),
        ({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": 1}, "evidence_ids": [1]},
         "evidence_ids must be a list of strings"),
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
        ({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": "x"}, "evidence_ids must be a list of strings"),
    ]
    for action, reason in after:
        obs, cost, done, info = env.step(action)
        assert info == {"refused": reason} and cost == 0 and not done, action
    component = faulty_components(env.initial)[0]
    obs, cost, *_ = order(env, component)
    assert cost == env.config.order_price(component) and obs["result"]["work_order"]["status"] == "open"
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
    assert env.log[-2]["event"] == "advance" and env.log[-2]["t_end"] == env.config.horizon and env.log[-1]["time"] == env.config.horizon
    env = make_env(horizon=5)
    env.reset(2)
    obs, cost, done, info = read(env, "tier1", 8)
    assert done and info["reason"] == "horizon" and len(obs["result"]["served"]) == 5
    assert events(env.log, "read")[0]["cost"] == 8 * env.config.prices["read"]["tier1"]  # the declared price, whatever the horizon cut
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


def test_finish_ends_the_episode_and_leaves_open_orders_open(make_env):
    env = anchored(make_env)
    env.reset(0)
    submit_truth(env)
    order(env, "cooler")
    obs, cost, done, info = finish(env, "cooler ordered")
    assert done and info["reason"] == "finish" and obs["result"] == {"summary": "cooler ordered"} and obs["episode_over"]
    assert [o["id"] for o in obs["open_work_orders"]] == ["wo-1"] and env.log[-1]["final"] == list(HAND_START)
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
def test_the_summary_reports_alarms_from_the_played_recording_and_the_closed_orders(make_env, world):
    cycles, blocks = world
    by_index = {c.index: c for c in cycles}
    env = anchored(make_env)
    env.reset(0)
    obs, *_ = summary(env, 0)
    assert obs["result"] == {"since": 0, "time": 0, "cycles_elapsed": 0, "alarms": [], "work_orders_closed": []}
    advance(env, 4)
    obs, *_ = summary(env, 1)
    expected = {}
    for t, index in env.clock.played:
        if t < 1:
            continue
        values = served_values(by_index[index], SENSORS)
        for field, band in env.healthy_bands.items():
            if field not in expected and not band["low"] <= values[field] <= band["high"]:
                expected[field] = t
    assert obs["result"]["alarms"] == [{"channel": f, "first_t": t} for f, t in sorted(expected.items(), key=lambda kv: (kv[1], kv[0]))]
    assert obs["result"]["cycles_elapsed"] == 3 and expected  # a faulty cooler leaves its indicating channels outside the healthy band
    submit_truth(env)
    order(env, "cooler")
    advance(env, 10)
    obs, *_ = summary(env, 0)
    closed = obs["result"]["work_orders_closed"]
    assert [o["id"] for o in closed] == ["wo-1"] and closed[0]["status"] == "closed" and closed[0]["source"] == "agent"
    assert summary(env, closed[0]["closed_t"] + 1)[0]["result"]["work_orders_closed"] == []
    entry = events(env.log, "summary")[-1]
    assert entry["since"] == closed[0]["closed_t"] + 1 and entry["class"] == "read" and entry["cost"] == 0


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
    assert advance(env, 10)[0]["time"] == 3
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
        timed.step({"type": "advance_time", "cycles": 1}, think_cycles=-1)
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
