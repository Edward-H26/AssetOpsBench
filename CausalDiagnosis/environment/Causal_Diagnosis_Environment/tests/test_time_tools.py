"""The two time tools (time_tools.py, notifications.py, env.py): get_current_time reads the clock and never moves it,
wait_for_notification runs it until the next notification or its timeout, every answer lists the public events since the
previous call exactly once, hidden events never notify, an alarm notifies only for a sensor group the agent paid for, the
contract declares the clock, and the off values of two rule cards and an episode without a budget behave as declared. No
test states a value computed from the rig data."""
from CausalDiagnosis.utils.paths import CONFIGS
import copy
import json
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import time_tools
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.episode_clock import CYCLE_ORDER
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import CLOCK_MODES, EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HIDDEN_EVENTS, HIDDEN_FIELDS, NOTIFYING_EVENTS, HydraulicRigEnv, mean_field
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import TOKEN_MODE
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, RandomPolicy
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import events, keys_of, order, read, submit_truth, summary, wait

HAND_START = (3, 90, 0, 130)  # the cooler at its worst level, the valve one level below healthy
MESSAGE_EVENTS = set(NOTIFYING_EVENTS) | {"wait_timeout", "end"}
ZERO_DURATIONS = {"read_per_cycle": 0, "advance_per_cycle": 1, "lead_time": dict.fromkeys(COMPONENTS, 0),
                  "repair_downtime": dict.fromkeys(COMPONENTS, 0)}


def anchored(make_env, combination=HAND_START, **overrides):
    return make_env(start={"combination": list(combination), "entry_offset": 0}, **overrides)


def test_a_wait_that_reaches_its_timeout_runs_the_clock_as_the_removed_advance_did(make_env):
    timed = {"schedule": {"mode": "seeded", "components": 1, "first_drop": [2, 2], "interval": 3}}
    waited, stepped = anchored(make_env, **timed), anchored(make_env, **timed)
    waited.reset(0)
    stepped.reset(0)
    obs, cost, done, _info = wait(waited, 5)
    stepped._advance(5)  # the clock primitive behind a wait, called directly for the same five cycles
    assert obs["result"] == {"time": 5, "waited": 5, "ended_by": "timeout"} and waited.time == stepped.time == 5 and not done
    assert waited.log[:-1] == stepped.log and events(waited.log, "scheduled_drop")  # the hidden drop ran and did not end the wait
    assert waited.clock.played == stepped.clock.played and waited.clock.downtime == stepped.clock.downtime and cost == stepped.cost
    entry = waited.log[-1]
    assert (entry["event"], entry["t_start"], entry["t_end"], entry["timeout"], entry["ended_by"]) == ("wait", 0, 5, 5, "timeout")
    assert [n["event"] for n in obs["notifications"]] == ["wait_timeout"] and obs["notifications"][0]["message"].endswith("after 5 cycles")


def test_a_wait_stops_at_a_work_order_close_and_at_the_horizon(make_env):
    env = anchored(make_env, durations={**EpisodeConfig().durations, "repair_downtime": {c: 0 for c in COMPONENTS}})
    env.reset(0)
    submit_truth(env)
    created = order(env, "cooler")[0]["result"]["work_order"]
    obs, *_ = wait(env, 50)
    closed = events(env.log, "work_order_closed")[0]
    assert obs["result"]["ended_by"] == "notification" and obs["time"] == closed["t_end"] + 1 == created["expected_close_t"] + 1
    assert [(n["event"], n["id"], n["t"]) for n in obs["notifications"]] == [("work_order_closed", created["id"], closed["t_end"])]
    # nothing is left to happen, so the default timeout waits up to the horizon, where the episode ends
    obs, _cost, done, info = wait(env)
    assert done and info["reason"] == "horizon" and env.time == env.config.horizon and obs["result"]["ended_by"] == "timeout"
    assert [n["event"] for n in obs["notifications"]] == ["wait_timeout", "end"] and obs["notifications"][-1]["message_type"] == "ENVIRONMENT_STOP"


def test_get_current_time_takes_no_cycle_in_any_clock_mode(make_env):
    for mode in CLOCK_MODES:
        env = anchored(make_env, clock_mode=mode)
        env.reset(0)
        read(env, "tier1", 2)
        before = env.time
        obs, cost, _done, _info = env.step({"type": "get_current_time"})
        entry = env.log[-1]
        assert obs["time"] == env.time == before and cost == 0, mode
        assert (entry["event"], entry["t_start"], entry["t_end"], entry["class"]) == ("current_time", before, before, "read")
        token = {"seconds_per_token": env.config.seconds_per_token} if mode == TOKEN_MODE else {}
        assert obs["result"] == {"time": before, "horizon": env.config.horizon, "seconds_per_cycle": env.config.seconds_per_cycle,
                                 "think_seconds_per_cycle": env.config.think_seconds_per_cycle, **token, "budget_left": obs["budget_left"]}


def test_every_public_event_notifies_once_and_no_hidden_event_ever_does(make_env):
    configs = [{"schedule": {"mode": "seeded", "components": 1, "first_drop": [2, 10], "interval": 3},
                "failure": {"patience": 3, "strict": False, "penalty": 50}},
               {"hazard": {"mode": "stochastic", "base_rate": 0.1, "depth_factor": 1.0, "collateral": 0.2}}]
    hidden_seen = set()
    for overrides in configs:
        for seed in range(6):
            for make in (lambda env: OraclePolicy(env), lambda env: RandomPolicy(seed)):
                env = make_env(**overrides)
                obs = env.reset(seed)
                policy = make(env)
                policy.start(obs)
                delivered = []
                while not env.done:
                    obs, *_ = env.step(policy.act(obs))
                    delivered += obs["notifications"]
                public = [(e["event"], e["t_end"]) for e in env.log if e["step"] is None and e["event"] in NOTIFYING_EVENTS]
                assert [(n["event"], n["t"]) for n in delivered if n["event"] in NOTIFYING_EVENTS] == public
                assert {n["event"] for n in delivered} <= MESSAGE_EVENTS and [n["event"] for n in delivered].count("end") == 1
                assert not keys_of(delivered) & set(HIDDEN_FIELDS) and all("component" not in n for n in delivered if n["event"] == "failure")
                hidden_seen |= {e["event"] for e in env.log} & set(HIDDEN_EVENTS)
    assert {"scheduled_drop", "hazard_drop"} <= hidden_seen  # the hidden processes ran, and none of their events notified


def test_a_notification_from_the_thinking_time_ends_a_wait_at_once(make_env):
    env = anchored(make_env, clock_mode="generation_time", failure={"patience": 3, "strict": False, "penalty": 50})
    env.reset(0)
    obs, *_ = env.step({"type": "wait_for_notification", "timeout": 10}, think_cycles=5)  # the cooler fails while the agent thinks
    assert obs["result"] == {"time": 5, "waited": 0, "ended_by": "notification"}
    assert [n["event"] for n in obs["notifications"]] == ["failure"]


def test_an_alarm_notifies_once_and_only_for_the_sensor_groups_the_agent_paid_for(make_env):
    """Alarms notify under the rule of the change summary: a channel of a group that a charged read named,
    once, at the first cycle it stood outside its normal range, so that no notification tells of a group nobody paid for."""
    env = anchored(make_env)
    env.reset(0)
    obs, *_ = wait(env, 4)
    assert "alarm" not in [n["event"] for n in obs["notifications"]]  # the rig runs faulty, but no group is paid for yet
    obs, *_ = read(env, "tier1")
    alarms = [(n["channel"], n["t"]) for n in obs["notifications"] if n["event"] == "alarm"]
    tier1 = {mean_field(s) for s in env.config.sensor_groups["tier1"]}
    assert alarms and {channel for channel, _ in alarms} <= tier1
    obs, *_ = wait(env, 4)
    assert "alarm" not in [n["event"] for n in obs["notifications"]]  # a channel alarms once
    assert alarms == [(a["channel"], a["first_t"]) for a in summary(env, 0)[0]["result"]["alarms"]]
    other = anchored(make_env)  # the same start read through tier3 alarms channels outside tier1, which the paid rule kept out
    other.reset(0)
    wait(other, 4)
    obs, *_ = read(other, "tier3")
    assert {n["channel"] for n in obs["notifications"] if n["event"] == "alarm"} - tier1


def test_a_wait_ends_at_the_first_alarm_of_a_paid_group(make_env):
    env = anchored(make_env, durations=ZERO_DURATIONS)
    env.reset(0)
    obs, *_ = read(env, "tier1")
    assert env.time == 0 and "alarm" not in [n["event"] for n in obs["notifications"]]  # the read takes no time, no cycle has run
    obs, *_ = wait(env)
    assert (obs["result"]["ended_by"], env.time) == ("notification", 1) and {n["event"] for n in obs["notifications"]} == {"alarm"}


def test_the_contract_and_the_profile_declare_the_clock(make_env):
    clock = contract.clock()
    assert clock["cycle_order"] == list(CYCLE_ORDER) and clock["notifying_events"] == list(NOTIFYING_EVENTS)
    assert set(clock["modes"]) == set(CLOCK_MODES) - {TOKEN_MODE} and "seconds_per_cycle" in clock
    env = make_env()
    profile = env.reset(0)["result"]
    rows = {r["name"]: r for r in profile["tools"]}
    # the profile's clock is the contract's, with this episode's mode, seconds_per_cycle and declared durations as values
    block = profile["clock"]
    assert {k: v for k, v in block.items() if k not in ("mode", "durations")} == {**clock, "seconds_per_cycle": env.config.seconds_per_cycle,
                                                                                 "think_seconds_per_cycle": env.config.think_seconds_per_cycle}
    assert profile["think_seconds_per_cycle"] == env.config.think_seconds_per_cycle
    assert block["mode"] == env.config.clock_mode and profile["seconds_per_cycle"] == env.config.seconds_per_cycle
    assert block["durations"] == {name: rows[name]["duration_value"] for name in contract.tools_of(contract.ACTION_SERVER)}
    assert "advance_time" not in rows and (rows["get_current_time"]["price_value"], rows["get_current_time"]["duration_value"]) == (0, 0)
    assert (rows["wait_for_notification"]["price_value"], rows["wait_for_notification"]["duration_value"]) == (0, 1)


def test_the_clock_and_the_time_tool_say_that_the_seconds_of_thinking_add_up():
    """The thinking rate in the texts an agent reads: the generation_time mode of the contract's clock carries the seconds from
    call to call at think_seconds_per_cycle instead of rounding a call up, and the description of get_current_time says so."""
    mode = contract.clock()["modes"]["generation_time"]
    assert "carried" in mode and "think_seconds_per_cycle" in mode and "rounded up" not in mode
    assert "add up" in time_tools.GET_CURRENT_TIME and "think_seconds_per_cycle" in time_tools.GET_CURRENT_TIME
    # both thinking modes of the clock charge thinking, and the description names both
    assert all(f"the {mode} clock mode" in time_tools.GET_CURRENT_TIME for mode in layers.real_time.THINKING_MODES)


def test_the_clock_states_the_counted_call_the_handover_and_the_credits():
    """The clock of the texts an agent reads is the clock as built: in the generation_time mode the thinking time of a counted
    call runs from the previous counted call, or from the handover for the first, less the credited starts of the tool servers,
    services of the counted calls and failed attempts of a model request; a counted call is a call that reaches the environment
    or the step of a turn that calls no tool of the rig, which the token mode counts the same way and which takes its cycle in
    the instant mode."""
    modes, tokens = contract.clock()["modes"], contract.token_clock()["mode"]
    timed = modes["generation_time"]
    assert all(words in timed for words in ("counted call", "handover", "first start of each tool server", "service of each counted call",
                                            "failed attempt of a model request", "turn that calls no tool of the rig")), timed
    assert "wall clock seconds since the previous call" not in timed
    assert "since the previous counted call" in tokens and "turn that calls no tool of the rig" in tokens, tokens
    assert "turn that calls no tool of the rig" in modes["instant"]


def test_the_off_values_switch_the_emergency_mode_and_the_failure_rule_off(make_env):
    env = anchored(make_env, emergency={"lead": 1, "surcharge": 2.0, "hold": 0}, failure={"patience": 0, "strict": True, "penalty": 50})
    env.reset(0)
    submit_truth(env)
    obs, cost, done, _info = env.step({"type": "create_work_order", "component": "cooler", "prediction_id": "", "evidence_ids": [], "mode": "emergency"})
    assert obs["refused"] == "the emergency mode is off in this episode" and cost == 0 and not done
    assert "emergency" not in {r["name"]: r for r in env.public_profile()["tools"]}["create_work_order"]["price_value"]
    obs, _cost, done, info = wait(env)  # the cooler stays at its worst level up to the horizon, and nothing fails
    assert done and info["reason"] == "horizon" and not events(env.log, "failure")


def test_without_a_budget_no_answer_shows_a_budget_and_no_price_ends_the_episode(make_env):
    env = anchored(make_env, budget=None, horizon=80)
    obs = env.reset(0)
    assert "budget_left" not in obs
    for _ in range(40):  # a price total far above the default budget
        obs, _cost, done, _info = read(env, "tier3", 1)
    assert "budget_left" not in obs and not done
    assert "budget_left" not in env.step({"type": "get_current_time"})[0]["result"]


def test_an_alarm_from_the_thinking_time_ends_a_wait_at_once(make_env):
    """An alarm that arises in a cycle of the agent's thinking is pending when the action runs, as a failure or a close of
    those cycles is (time_tools.py), so a wait right after it ends at once by the notification, and the
    alarm entry stands before the entry of the wait. The test takes the first seed whose thinking cycles, played on a copy
    of the episode, bring a channel of the paid group out of its normal range and raise no other notification, and it asserts that
    such a seed exists."""
    think = 4
    timed = {"clock_mode": "generation_time", "schedule": {"mode": "seeded", "components": 1, "first_drop": [1, 1], "interval": 50}}
    for seed in range(60):
        env = make_env(**timed)
        env.reset(seed)
        group = max(env.config.sensor_groups, key=lambda g: len(env.config.sensor_groups[g]))
        read(env, group, 1)
        mark, start = len(env.log), env.time
        twin = copy.deepcopy(env, memo={id(env.cycles): env.cycles, id(env.blocks): env.blocks})
        twin._advance(think)
        others = [e for e in twin.log[mark:] if e["step"] is None and e["event"] in NOTIFYING_EVENTS]
        if not others and [a for a in twin._alarms(0) if a["channel"] not in twin._alarmed]:
            break
    else:
        pytest.fail("no seed of the first 60 raises an alarm in the thinking cycles alone")
    obs, *_ = env.step({"type": "wait_for_notification", "timeout": 10}, think_cycles=think)
    assert obs["result"] == {"time": start + think, "waited": 0, "ended_by": "notification"}
    alarm = next(i for i, e in enumerate(env.log) if i >= mark and e["event"] == "alarm")
    waited = next(i for i, e in enumerate(env.log) if e["event"] == "wait")
    assert alarm < waited and start <= env.log[alarm]["t_start"] < start + think
    assert "alarm" in [n["event"] for n in obs["notifications"]]


# ----- the alarms checked on the cycles played since the last check give the log of the check over the whole history -----

TIMED_PRESET = CONFIGS / "development" / "timed.json"
ALARM_SEEDS = 40  # the seeds searched for the episodes whose alarms the tests need


def _timed_config():
    return layers.compose(json.loads(TIMED_PRESET.read_text(encoding="utf-8")))


def _alarms_over_history(source, played, since, channels):
    """The alarms as the engine computed them before it kept the excursions of a recording, the test's own copy: the served
    values of every played cycle formatted again and set against the normal ranges, the first cycle of each field, sorted
    by cycle and field."""
    first = {}
    for t, index in played:
        if t < since or index is None or not channels:
            continue
        values = evidence.served_values(source.cycles[index], channels, source.profile)
        for field in (evidence.band_field(s, source.profile) for s in channels):
            band = source.normal_ranges[field]
            if field not in first and not band["low"] <= values[field] <= band["high"]:
                first[field] = t
    return [{"channel": field, "first_t": t} for field, t in sorted(first.items(), key=lambda kv: (kv[1], kv[0]))]


class _HistoryAlarmsEnv(HydraulicRigEnv):
    """The engine with the earlier alarm check: every check over the whole history, under the paid channels that the reads
    of the log name."""

    def _alarms(self, since):
        groups = {e["sensor_group"] for e in self.log if e["event"] == "read" and e["served"]}
        paid = tuple(s for s in self.profile.channels if any(s in self.config.sensor_groups[g] for g in groups))
        return _alarms_over_history(self.source, self.clock.played, since, paid)

    def _record_alarms(self):
        for alarm in self._alarms(0):
            if alarm["channel"] not in self._alarmed:
                self._alarmed.add(alarm["channel"])
                self.log.append({"event": "alarm", "step": None, "t_start": alarm["first_t"], "t_end": alarm["first_t"], "class": "system",
                                 "cost": 0, "channel": alarm["channel"]})


def _narrow_and_wide(config):
    groups = sorted(config.sensor_groups, key=lambda g: (len(config.sensor_groups[g]), g))
    return groups[0], groups[-1]


def _played_on_both(world, seed, script):
    """The engine after every (action, thinking cycles) of script(env) from the seed on the timed preset, with its log
    asserted equal to that of the engine with the earlier alarm check after every step."""
    cycles, blocks = world
    config = _timed_config()
    env, earlier = HydraulicRigEnv(cycles, blocks, config), _HistoryAlarmsEnv(cycles, blocks, config)
    env.reset(seed)
    earlier.reset(seed)
    for action, think in script(env):
        if env.done:
            break
        env.step(copy.deepcopy(action), think_cycles=think)
        earlier.step(copy.deepcopy(action), think_cycles=think)
        assert env.log == earlier.log, (seed, action)
    return env


def _repair_script(env):
    """Reads of the narrow group, two waits, the true diagnosis, an order on a faulty component, a wait past its close, a
    read of the wide group, a wait, the change summary and the finish, some of them after thinking cycles."""
    narrow, wide = _narrow_and_wide(env.config)
    start = dict(zip(env.config.components, env.initial))
    faulty = [c for c in env.config.components if start[c] != env.config.healthy[c]]
    script = [({"type": "read_sensors", "sensor_group": narrow, "cycles": 1}, 0), ({"type": "wait_for_notification", "timeout": 8}, 2),
              ({"type": "wait_for_notification", "timeout": 8}, 0), ({"type": "submit_diagnosis", "diagnosis": start, "evidence_ids": []}, 1)]
    script += [({"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}, 0) for c in faulty[:1]]
    return script + [({"type": "wait_for_notification", "timeout": 15}, 0), ({"type": "read_sensors", "sensor_group": wide, "cycles": 2}, 3),
                     ({"type": "wait_for_notification", "timeout": 10}, 0), ({"type": "summarize_changes_since", "cycle": 0}, 0),
                     ({"type": "finish", "summary": "done"}, 0)]


def _late_script(env):
    """A read of the narrow group, a wait, then a first read of the wide group and one more wait."""
    narrow, wide = _narrow_and_wide(env.config)
    return [({"type": "read_sensors", "sensor_group": narrow, "cycles": 1}, 0), ({"type": "wait_for_notification", "timeout": 20}, 0),
            ({"type": "read_sensors", "sensor_group": wide, "cycles": 1}, 0), ({"type": "wait_for_notification", "timeout": 5}, 0)]


def _late_alarms(env):
    """The alarms of the channels that only the wide group pays, from a cycle before the first read of that group."""
    narrow, wide = _narrow_and_wide(env.config)
    late = next(e for e in env.log if e["event"] == "read" and e["sensor_group"] == wide)
    fields = {evidence.band_field(s, env.profile) for s in env.config.sensor_groups[wide] if s not in env.config.sensor_groups[narrow]}
    return [a for a in events(env.log, "alarm") if a["channel"] in fields and a["t_start"] < late["t_start"]]


def test_the_alarms_of_a_long_wait_equal_the_alarms_checked_over_the_whole_history(world):
    """After every step of a scripted episode on the timed preset, with waits, thinking cycles, a repair and a group paid
    later, the log equals that of the engine whose every check reads the whole history. Among the seeds searched are an
    alarm before the close of the order, one after it and one of the group paid later from a cycle before its read, so each
    case in which a check of the new cycles alone could differ occurs."""
    seen = set()
    for seed in range(ALARM_SEEDS):
        env = _played_on_both(world, seed, _repair_script)
        closes = [e["t_start"] for e in events(env.log, "work_order_closed")]
        for alarm in events(env.log, "alarm"):
            if closes:
                seen.add("before the close" if alarm["t_start"] < closes[0] else "after the close")
        if _late_alarms(env):
            seen.add("paid later")
    assert seen == {"before the close", "after the close", "paid later"}


def test_a_group_paid_late_raises_the_alarms_of_the_cycles_before_it(world):
    """A channel of a group first read at cycle t that stood outside its normal range before t is alarmed with that earlier cycle,
    as the check of the whole history alarms it, so new paid channels have every played cycle checked."""
    for seed in range(ALARM_SEEDS):
        env = _played_on_both(world, seed, _late_script)
        if _late_alarms(env):
            break
    else:
        pytest.fail(f"no seed below {ALARM_SEEDS} shows a channel of the wide group outside its normal range before the group was read")
