"""The action server (CausalDiagnosis.mcp_servers.Environment_Actions_MCP): field names, state and replay, the envelope, the released evidence, the
briefing, the thinking time of the generation_time mode, the contract match, the policies through the tools, and the MCP
server over stdio when the mcp package is installed (mcp 1.26.0, pinned in the fork's uv.lock). No test states a value
computed from the rig data."""
from CausalDiagnosis.utils.paths import CONFIGS, ROOT, SRC, preset_files, preset_path
import asyncio
import dataclasses
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.dataset_loader.profile_data import processes_block
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import wall_clock
from CausalDiagnosis.environment.reference_policies.tier_classifiers import TierClassifiers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HIDDEN_FIELDS, HydraulicRigEnv, served_values
from CausalDiagnosis.environment.reference_policies.greedy_policy import GreedyPolicy, ServedClassifiers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose, layer_4_repair_rules as repair_rules
from CausalDiagnosis.environment.reference_policies.policies import DoNothingPolicy, run_episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import (ALARM_ONCE, BRIEFING_VERSION, DECLARATION_RULE, DECLARE_STEP, EVIDENCE_FORMS, HOLD_SCORING,
                                            SCORING_SENTENCE, TIME_SENTENCE, WALL_END_SENTENCE, asset_text, briefing, idle_turn_sentence,
                                            outlasts_its_call, repair_step)
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import PROCESS_WORDS, PROFILE_PARTS, build_server, evidence_list
from CausalDiagnosis.datasets.hyd_1.former_field_names import HYD1_FIELDS, public_name, tier_fields
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.policy_through_tools import play, tool_call

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

HYD1_FILE = SRC / "couchdb" / "scenarios_data" / "shared" / "iot" / "hydraulic_pump_1.json"
TOOLS = set(contract.tools_of(contract.ACTION_SERVER))
SERVED_STATISTICS = tuple(get_profile("hyd_1").statistics["served"])  # what a reading of the rig returns per channel
STATE_KEYS = {"format", "seed", "config", "facts_path", "actions", "think_cycles", "think_seconds", "think_seconds_carried", "tool_calls",
              "last_call_wall_time", "episode_started_at", "called_at", "think_seconds_monotonic", "last_call_monotonic_time",
              "think_seconds_wall", "handover_wall", "handover_mono", "charged_until_mono", "transcript_cursor", "failures_cursor",
              "credited_segments", "started_servers", "open_service", "refused_processes", "credited_seconds", "uncredited_seconds",
              "failed_model_seconds", "failed_model_attempts"}
RULE_KEYS = {"components", "levels", "healthy", "sensor_groups", "statistics", "prices", "durations", "downtime_rate", "budget", "horizon",
             "max_steps", "processes", "clock_mode", "clock", "tools", "normal_ranges"}
HAND_START = (3, 90, 0, 130)
needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp package (pinned in the fork's uv.lock)")
MALFORMED = [  # (tool, arguments, the action env.step receives): wrong types arrive as sent, missing arguments as None
    ("read_sensors", {"sensor_group": "tier1", "cycles": "two"}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": "two"}),
    ("read_sensors", {"sensor_group": "tier1", "cycles": "[2]"}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": "[2]"}),
    ("read_sensors", {"sensor_group": "tier1"}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": None}),
    ("wait_for_notification", {"timeout": 2.5}, {"type": "wait_for_notification", "timeout": 2.5}),
    ("submit_diagnosis", {"cooler": 100, "valve": 100, "pump": 0, "evidence_ids": ""},
     {"type": "submit_diagnosis", "diagnosis": {"cooler": 100, "valve": 100, "pump": 0, "accumulator": None}, "evidence_ids": []}),
    ("create_work_order", {"component": 5, "prediction_id": "", "evidence_ids": ""},
     {"type": "create_work_order", "component": 5, "prediction_id": "", "evidence_ids": []}),
]


def _new_episode(tmp_path, seed=3, at_start=False, **overrides):
    """A new episode handed over now, or with at_start at the readings of its start, so that a call stamped on both clocks
    from those readings is charged exactly its stamped seconds."""
    path = tmp_path / "episode.json"
    episode.start(path, seed, EpisodeConfig(**overrides), default_facts_path())
    state = episode.load_state(path)
    episode.hand_over(path, **({"wall": state["episode_started_at"], "mono": state["last_call_monotonic_time"]} if at_start else {}))
    return path


def _anchored(tmp_path, **overrides):
    return _new_episode(tmp_path, start={"combination": list(HAND_START), "entry_offset": 0}, **overrides)


def _call(path, tool, arguments, action, now=None, monotonic=None):
    return json.loads(episode.call(path, tool, arguments, action, now=now, monotonic=monotonic))


def _truth(path):
    return dict(zip(COMPONENTS, episode.replay(episode.load_state(path)).initial))


# the processes block of the rules holds the parameters of the rules under the name schedule, which the hidden
# draw carries too, so the scan skips it as it skips the tool parameter tables
def _keys(obj, skip=("action", "parameters", "processes")):
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(k)
            if k not in skip:
                found |= _keys(v, skip)
    elif isinstance(obj, list):
        for v in obj:
            found |= _keys(v, skip)
    return found


def _send(path):
    def send(action):
        tool, arguments = tool_call(action)
        return episode.call(path, tool, arguments, action)
    return send


# ----- names -----
def test_every_served_field_maps_to_a_distinct_hyd1_name():
    cycles, _ = episode.load_world(str(default_facts_path()))
    values = served_values(cycles[0], SENSORS)
    assert len(values) == len(SENSORS) * len(SERVED_STATISTICS)
    assert set(values) == {f"{HYD1_FIELDS[s]}_{stat}" for s in SENSORS for stat in SERVED_STATISTICS}
    assert public_name("MotorPower_EPS1_mean") == "P1_MotorPower_W_100Hz_mean" and public_name("Pressure_PS1_std") == "PS1_Pressure_bar_100Hz_std"
    assert set(HYD1_FIELDS) == set(SENSORS) and all(tier_fields(t) == [HYD1_FIELDS[s] for s in TIER_SENSORS[t]] for t in TIER_SENSORS)


@pytest.mark.skipif(not HYD1_FILE.exists(), reason="AssetOpsBench's hyd_1 file is not checked out")
def test_hyd1_field_names_are_the_asset_s_own_columns():
    columns = set(json.loads(HYD1_FILE.read_text())[0])
    assert set(HYD1_FIELDS.values()) <= columns


# ----- state, replay and the envelope -----
def test_the_state_file_holds_what_replay_needs_and_nothing_else(tmp_path):
    state = episode.load_state(_new_episode(tmp_path))
    assert set(state) == STATE_KEYS and state["format"] == 4 and state["think_seconds_carried"] == 0.0
    assert state["actions"] == state["think_cycles"] == state["think_seconds"] == state["tool_calls"] == []
    assert EpisodeConfig.from_json(state["config"]) == EpisodeConfig() and isinstance(state["last_call_wall_time"], float)
    assert state["episode_started_at"] == state["last_call_wall_time"] and isinstance(state["last_call_monotonic_time"], float)
    assert state["called_at"] == state["think_seconds_monotonic"] == []
    with pytest.raises(ValueError, match="unknown episode state format"):
        episode.save_state(tmp_path / "old.json", {**state, "format": 1})
        episode.load_state(tmp_path / "old.json")


def test_replay_builds_the_environment_class_it_is_given(tmp_path):
    class Engine(HydraulicRigEnv):  # a subclass in the place of the AssetOpsBench rows' engine
        pass

    state = episode.load_state(_new_episode(tmp_path))
    assert type(episode.replay(state)) is HydraulicRigEnv and type(episode.replay(state, env_class=Engine)) is Engine


def test_start_takes_the_engine_class_and_the_briefing_of_another_row(tmp_path):
    class Engine(HydraulicRigEnv):  # a subclass in the place of the AssetOpsBench rows' engine
        def public_profile(self):
            return {**super().public_profile(), "engine": "subclass"}

    def compose(profile, config):
        return f"Find the fault of {config.asset_id} on the {profile.get('engine', 'rig')} engine."

    path = tmp_path / "episode.json"
    episode.start(path, 3, EpisodeConfig(), default_facts_path(), env_class=Engine, compose=compose)
    episode.hand_over(path)
    assert json.loads(episode.released_path(path).read_text())["task_text"] == "Find the fault of hyd_1 on the subclass engine."


def test_every_call_reaches_env_step_and_replay_reproduces_the_log(tmp_path):
    path = _new_episode(tmp_path)
    truth = _truth(path)
    faulty = [c for c in COMPONENTS if truth[c] != HEALTHY[c]]
    actions = [
        ("read_sensors", {"sensor_group": "tier1", "cycles": 2}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 2}),
        ("read_sensors", {"sensor_group": "tier4", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier4", "cycles": 1}),
        ("create_work_order", {"component": faulty[0], "prediction_id": "", "evidence_ids": ""},
         {"type": "create_work_order", "component": faulty[0], "prediction_id": "", "evidence_ids": []}),
        ("submit_diagnosis", {**truth, "pump": 7, "evidence_ids": "obs/1/x"}, {"type": "submit_diagnosis", "diagnosis": dict(truth, pump=7), "evidence_ids": ["obs/1/x"]}),
        ("submit_diagnosis", {**truth, "evidence_ids": "obs/1/x, obs/2/y"}, {"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": ["obs/1/x", "obs/2/y"]}),
        ("summarize_changes_since", {"cycle": 0}, {"type": "summarize_changes_since", "cycle": 0}),
        ("get_failure_modes", {"component": ""}, {"type": "get_failure_modes", "component": ""}),
    ] + [("create_work_order", {"component": c, "prediction_id": "", "evidence_ids": ""},
          {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}) for c in faulty]
    wait = ("wait_for_notification", {"timeout": 20}, {"type": "wait_for_notification", "timeout": 20})
    tail = [("get_current_time", {}, {"type": "get_current_time"}), ("get_asset_profile", {}, {"type": "get_asset_profile"}),
            ("finish", {"summary": "repaired"}, {"type": "finish", "summary": "repaired"})]
    direct = HydraulicRigEnv(*episode.load_world(str(default_facts_path())), EpisodeConfig())
    direct.reset(3)

    def both(tool, arguments, action):
        answer = _call(path, tool, arguments, action)
        envelope, *_ = direct.step(action)
        assert answer == json.loads(json.dumps(envelope))
    for call in actions:
        both(*call)
    while direct.clock.open_orders():  # every notification ends a wait, so the agent waits again until the orders closed
        both(*wait)
        actions.append(wait)
    for call in tail:
        both(*call)
    actions += tail
    state = episode.load_state(path)
    assert len(state["actions"]) == len(actions) == len(state["tool_calls"]) == len(state["think_cycles"]) and set(state["think_cycles"]) == {0}
    assert json.dumps(episode.replay(state).log) == json.dumps(direct.log)
    assert direct.done and direct.log[-1]["reason"] == "finish" and direct.log[-1]["final"] == [HEALTHY[c] for c in COMPONENTS]


def test_every_read_buys_cycles_no_earlier_read_served_and_the_release_lists_each_once(tmp_path):
    """The read rule of the rig server: read_sensors serves cycles that no earlier read served,
    the cycles the rig plays for it or, without a clock, the next recordings of the current cycle, so no read is a re-read
    and a tier2 read after a tier1 read at the same cycle pays the tier2 price per cycle. released.json lists every served
    observation once, under its own n and with the fields of its own group."""
    zero = {"read_per_cycle": 0, "advance_per_cycle": 1, "lead_time": dict.fromkeys(COMPONENTS, 0), "repair_downtime": dict.fromkeys(COMPONENTS, 0)}
    path = _anchored(tmp_path, durations=zero)
    first = _call(path, "read_sensors", {"sensor_group": "tier1", "cycles": 2}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 2})
    second = _call(path, "read_sensors", {"sensor_group": "tier2", "cycles": 2}, {"type": "read_sensors", "sensor_group": "tier2", "cycles": 2})
    assert first["time"] == second["time"] == 0
    assert first["budget_left"] - second["budget_left"] == 2 * EpisodeConfig().prices["read"]["tier2"]
    observations = json.loads(episode.released_path(path).read_text())["observations"]
    assert [(o["n"], o["t"], o["sensor_group"]) for o in observations] == [(1, 0, "tier1"), (2, 0, "tier1"), (3, 0, "tier2"), (4, 0, "tier2")]
    fields = {tier: {f"{f}_{s}" for f in tier_fields(tier) for s in SERVED_STATISTICS} for tier in (1, 2)}
    assert [set(o["values"]) for o in observations] == [fields[1], fields[1], fields[2], fields[2]]


def test_refusals_come_back_as_envelopes_cost_nothing_and_count_as_steps(tmp_path):
    path = _new_episode(tmp_path)
    truth = _truth(path)
    budget, steps = EpisodeConfig().budget, EpisodeConfig().max_steps
    for tool, arguments, action, reason in [
        ("read_sensors", {"sensor_group": "tier4", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier4", "cycles": 1}, "unknown sensor group"),
        ("create_work_order", {"component": "pump", "prediction_id": "", "evidence_ids": ""},
         {"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []}, "a diagnosis or a declaration must precede a work order"),
        ("finish", {"summary": None}, {"type": "finish", "summary": None}, "summary must be a string"),
    ]:
        response = _call(path, tool, arguments, action)
        assert response["refused"] == reason and response["budget_left"] == budget and not response["episode_over"] and "result" not in response
    _call(path, "submit_diagnosis", {**truth, "evidence_ids": ""}, {"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": []})
    response = _call(path, "create_work_order", {"component": "motor", "prediction_id": "", "evidence_ids": ""},
                     {"type": "create_work_order", "component": "motor", "prediction_id": "", "evidence_ids": []})
    assert response["refused"] == "unknown component" and response["budget_left"] == budget and response["steps_left"] == steps - 5


def test_calls_after_the_end_are_refused_and_recorded_only_in_tool_calls(tmp_path):
    path = _new_episode(tmp_path)
    end = _call(path, "finish", {"summary": "nothing"}, {"type": "finish", "summary": "nothing"})
    assert end["episode_over"] and end["result"] == {"summary": "nothing"}
    after = _call(path, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1})
    assert after["episode_over"] and after["refused"] == episode.EPISODE_OVER and "result" not in after
    assert set(after) == {"refused", "time", "budget_left", "steps_left", "open_work_orders", "episode_over", "notifications"}
    state = episode.load_state(path)
    assert len(state["actions"]) == 1 and len(state["tool_calls"]) == 2
    # atomic writes leave no temporary files: the state with its lock, the released evidence, and the transcript of the servers with its lock
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.json", "episode.json.lock", "released.json", "transcript.jsonl", "transcript.jsonl.lock"]


# ----- the released evidence -----
def test_released_evidence_holds_only_what_the_agent_saw(tmp_path):
    path = _anchored(tmp_path, anchors={"date0": "2022-07-01"}, question="What is wrong with the unit?")
    released = json.loads(episode.released_path(path).read_text())
    assert set(released) == {"format", "asset_id", "task_text", "guidance_level", "rules", "status", "events", "observations"}
    assert released["format"] == 3 and released["asset_id"] == "hyd_1" and released["guidance_level"] == 1 and released["events"] == []
    assert set(released["rules"]) == RULE_KEYS and released["rules"]["sensor_groups"]["tier1"] == tier_fields(1)
    # the parameters of the rules and never the fault rule of the starts
    config = json.loads(path.read_text())["config"]
    assert released["rules"]["processes"] == processes_block(config) and "min_faulty" not in json.dumps(released)
    # the clock block and the tool table with the values of this configuration
    clock, config = released["rules"]["clock"], EpisodeConfig()
    assert {k: v for k, v in clock.items() if k not in ("mode", "durations")} == {**contract.clock(), "seconds_per_cycle": config.seconds_per_cycle,
                                                                                    "think_seconds_per_cycle": config.think_seconds_per_cycle}
    assert clock["mode"] == config.clock_mode and clock["durations"]["read_sensors"] == config.durations["read_per_cycle"]
    tools = {row["name"]: row for row in released["rules"]["tools"]}
    assert tools["read_sensors"]["price_value"] == EpisodeConfig().prices["read"] and tools["wait_for_notification"]["duration_value"] == 1
    assert released["status"] == {"time": 0, "day": "2022-07-01", "budget_left": EpisodeConfig().budget, "steps_left": EpisodeConfig().max_steps,
                                  "open_work_orders": [], "episode_over": False}
    assert "What is wrong with the unit?" in released["task_text"]
    truth = _truth(path)
    _call(path, "read_sensors", {"sensor_group": "tier2", "cycles": 2}, {"type": "read_sensors", "sensor_group": "tier2", "cycles": 2})
    _call(path, "read_sensors", {"sensor_group": "tier4", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier4", "cycles": 1})
    _call(path, "submit_diagnosis", {**truth, "evidence_ids": "obs/1/x"}, {"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": ["obs/1/x"]})
    _call(path, "create_work_order", {"component": "cooler", "prediction_id": "", "evidence_ids": ""},
          {"type": "create_work_order", "component": "cooler", "prediction_id": "", "evidence_ids": []})
    closed = _call(path, "wait_for_notification", {"timeout": 6}, {"type": "wait_for_notification", "timeout": 6})
    assert closed["result"]["ended_by"] == "notification" and [n["event"] for n in closed["notifications"]] == ["work_order_executed", "work_order_closed"]
    _call(path, "wait_for_notification", {"timeout": 8 - closed["time"]}, {"type": "wait_for_notification", "timeout": 8 - closed["time"]})
    _call(path, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1})
    released = json.loads(episode.released_path(path).read_text())
    env = episode.replay(episode.load_state(path))
    assert not _keys(released) & set(HIDDEN_FIELDS)
    assert released["events"] == json.loads(json.dumps(env.public_events()))
    names = [e["event"] for e in released["events"]]
    assert [n for n in names if n != "alarm"] == ["read", "refused", "submit_diagnosis", "work_order_created", "work_order_executed",
                                                  "work_order_closed", "wait", "wait", "read"]
    # the alarms of the group the first read paid for follow that read
    alarms = [e["channel"] for e in released["events"] if e["event"] == "alarm"]
    assert names[1:1 + len(alarms)] == ["alarm"] * len(alarms) and set(alarms) <= {f"{f}_mean" for f in tier_fields(2)}
    assert all({"t_start", "t_end", "class"} <= set(e) for e in released["events"])
    assert [(o["n"], o["t"], o["sensor_group"]) for o in released["observations"]] == [(1, 0, "tier2"), (2, 1, "tier2"), (3, 8, "tier1")]
    assert set(released["observations"][0]["values"]) == {f"{f}_{s}" for f in tier_fields(2) for s in SERVED_STATISTICS}
    assert released["status"]["time"] == 9 and released["status"]["day"] == "2022-07-10" and released["status"]["open_work_orders"] == []
    assert released["task_text"] == briefing(env.public_profile(), env.config)
    text = episode.released_path(path).read_text()
    assert not any(label in text for label in ("cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "seed"))


# ----- the generation_time mode -----
def test_generation_time_mode_stores_the_thinking_cycles_and_replays_them(tmp_path):
    """The seconds charged between calls add up from call to call and every full think_seconds_per_cycle of them is one cycle, the
    rest carried in the state file to the next call, so 75 seconds at 30 seconds of thinking per cycle charge
    two cycles and carry 15, and one more second charges none and carries 16. The task is handed over at the readings of the
    start and every call is stamped on both clocks, so a call is charged its monotonic seconds since the entry before it. Replay
    passes the stored cycles to step again, and the stored seconds give the stored cycles and the carried seconds back through
    the same function."""
    seconds = 30
    path = _anchored(tmp_path, at_start=True, clock_mode="generation_time", think_seconds_per_cycle=seconds)
    start = episode.load_state(path)
    t0, m0 = start["last_call_wall_time"], start["last_call_monotonic_time"]
    truth = _truth(path)
    calls = [  # the seconds after the start at which each call enters
        ("read_sensors", {"sensor_group": "tier1", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1}, 2.5 * seconds),
        ("submit_diagnosis", {**truth, "evidence_ids": ""}, {"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": []}, 2.5 * seconds + 1),
        ("wait_for_notification", {"timeout": 2}, {"type": "wait_for_notification", "timeout": 2}, 2.5 * seconds + 1),
    ]
    answers = [_call(path, tool, arguments, action, now=t0 + after, monotonic=m0 + after) for tool, arguments, action, after in calls]
    state = episode.load_state(path)
    assert state["think_cycles"] == [2, 0, 0] and state["think_seconds"] == [75.0, 1.0, 0.0] and state["think_seconds_carried"] == 16.0
    assert state["last_call_wall_time"] == t0 + calls[-1][3]
    assert [a["time"] for a in answers] == [3, 3, 5]
    direct = HydraulicRigEnv(*episode.load_world(str(default_facts_path())), EpisodeConfig.from_json(state["config"]))
    direct.reset(state["seed"])
    for (_tool, _arguments, action, _after), think in zip(calls, state["think_cycles"]):
        direct.step(action, think_cycles=think)
    assert json.dumps(episode.replay(state).log) == json.dumps(direct.log)
    assert [e["think_cycles"] for e in direct.log if e["step"] is not None] == [2, 0, 0]
    other = dict(state, think_cycles=[0, 0, 0])
    assert episode.replay(other).time != direct.time  # the stored thinking cycles are part of the episode
    carried, charged = 0.0, []
    for spent in state["think_seconds"]:
        cycles, _charged_seconds, carried = wall_clock.think_cycles(direct.config, carried, spent)
        charged.append(cycles)
    assert charged == state["think_cycles"] and carried == state["think_seconds_carried"]


def test_the_thinking_seconds_add_up_and_only_the_generation_time_mode_charges_them():
    """think_cycles is the one conversion of thinking seconds into cycles: a call's seconds join the seconds carried from the calls before
    it, every full think_seconds_per_cycle is one cycle and the rest is carried, seconds_per_cycle, the recorded cycle, plays no
    part, and the other clock modes charge no thinking time."""
    config = EpisodeConfig(clock_mode="generation_time", think_seconds_per_cycle=30, seconds_per_cycle=7)
    assert wall_clock.think_cycles(config, 0.0, 0.5) == (0, 0.5, 0.5)  # a short call charges no cycle and carries its seconds
    assert wall_clock.think_cycles(config, 20.0, 9.5) == (0, 9.5, 29.5) and wall_clock.think_cycles(config, 20.0, 10.0) == (1, 10.0, 0.0)
    assert wall_clock.think_cycles(config, 29.5, 45.5) == (2, 45.5, 15.0)  # one call may charge several cycles
    assert wall_clock.think_cycles(EpisodeConfig(), 4.0, 1000.0) == (0, 0.0, 4.0)  # the declared mode charges nothing and keeps the carry

def test_every_call_keeps_its_wall_time_and_the_monotonic_gap_beside_the_wall_gap(tmp_path):
    """The monotonic gap of a call is charged and the wall gap stands beside it, so that their difference, a sleep of the
    machine or a step of the wall clock, which the charge never holds, can be shown. Here the wall clock moves 75 s and then 1 s
    while the monotonic clock moves 15 s and then 1 s, as across a sleep of 60 s, and the seconds charged are those of the
    monotonic clock alone. The declared mode charges nothing and records zero, a wall clock set back gives a wall gap of zero,
    and a start without a monotonic reading records None."""
    path = _anchored(tmp_path, at_start=True, clock_mode="generation_time", think_seconds_per_cycle=30)
    start = episode.load_state(path)
    t0, m0 = start["last_call_wall_time"], start["last_call_monotonic_time"]
    read = {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1}
    for now, monotonic in ((t0 + 75, m0 + 15), (t0 + 76, m0 + 16), (t0 + 70, m0 + 17)):
        episode.call(path, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, read, now=now, monotonic=monotonic)
    state = episode.load_state(path)
    assert state["think_seconds"] == state["think_seconds_monotonic"] == [15.0, 1.0, 1.0] and state["think_seconds_wall"] == [75.0, 1.0, 0.0]
    assert state["think_cycles"] == [0, 0, 0] and state["called_at"] == [t0 + 75, t0 + 76, t0 + 70] and state["episode_started_at"] == t0
    assert state["last_call_monotonic_time"] == m0 + 17
    (tmp_path / "declared").mkdir()
    declared = _new_episode(tmp_path / "declared")
    episode.call(declared, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, read, now=time.time() + 50)
    assert episode.load_state(declared)["think_seconds_monotonic"] == [0.0] and episode.load_state(declared)["think_seconds"] == [0.0]
    unread = tmp_path / "unread" / "episode.json"  # as a replay starts an episode whose start line names no monotonic reading
    unread.parent.mkdir()
    episode.save_state(unread, {**start, "last_call_monotonic_time": None})
    episode.call(unread, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, read, now=t0 + 5)
    assert episode.load_state(unread)["think_seconds_monotonic"] == [None] and episode.load_state(unread)["called_at"] == [t0 + 5]


# ----- the briefing -----
def test_the_briefing_grows_with_the_guidance_level_and_names_every_legal_value_at_level_two(tmp_path):
    env = episode.replay(episode.load_state(_anchored(tmp_path, anchors={"date0": "2022-07-01"})))
    profile = env.public_profile()
    assert BRIEFING_VERSION == "v2.20"
    one = briefing(profile, EpisodeConfig(guidance_level=1, anchors={"date0": "2022-07-01"}))
    two = briefing(profile, EpisodeConfig(guidance_level=2))
    three = briefing(profile, EpisodeConfig(guidance_level=3))
    assert one.startswith("Asset hyd_1") and "On 2022-07-01" in one and f"budget of {profile['budget']}" in one and f"{profile['horizon']} cycles" in one
    assert "read_sensors(" not in one and "Procedure" not in one
    assert DECLARATION_RULE not in one and DECLARATION_RULE in two and DECLARATION_RULE in three  # the rule beside the tool's line
    assert "cite the evidence identifiers behind every diagnosis, declaration and work order" in three
    assert all(text.count(SCORING_SENTENCE) == 1 for text in (one, two, three))  # the scoring rule at every guidance level
    assert all(text.count(TIME_SENTENCE) == 1 for text in (one, two, three))  # the two time tools and the notifications too
    plain_one = briefing(profile, EpisodeConfig(guidance_level=1))
    assert two.startswith(plain_one + "\n\n")  # level two opens with the whole level one text of its configuration
    # an anchored start adds its date alone, and the opening sentence goes on in lower case after it
    assert one != plain_one and one.replace("On 2022-07-01, the performance", "The performance", 1) == plain_one
    assert all(c in two and all(str(v) in two for v in LEVELS[c]) for c in COMPONENTS)
    assert all(f in two for t in TIER_SENSORS for f in tier_fields(t)) and all(f"{name}(" in two for name in TOOLS)
    assert EVIDENCE_FORMS in two
    assert f"repair charge of {profile['prices']['repair_charge']}" in two and f"Budget: {profile['budget']}." in two
    assert "precede" in two or "refused before a diagnosis or a declaration" in two
    assert "diagnostics server offers" in two and "Procedure" not in two
    assert three.startswith(two) and "Procedure: 1. read sensors before diagnosing" in three and "rank_root_causes" in three
    without = briefing(profile, EpisodeConfig(guidance_level=3, diagnostics_server=False))
    assert "diagnostics server" not in without and "rank_root_causes" not in without and "pred/<n>" not in without
    assert "prediction_id is empty in this episode" in without
    asked = briefing(profile, EpisodeConfig(question="Why is the pressure low?"))
    assert "Why is the pressure low?" in asked and "Which condition level" not in asked
    instant = briefing(profile, EpisodeConfig(guidance_level=2, clock_mode="instant"))
    timed = briefing(profile, EpisodeConfig(guidance_level=2, clock_mode="generation_time"))
    strict = briefing(profile, EpisodeConfig(guidance_level=2, failure={"patience": 10, "strict": True, "penalty": 50}))
    assert "advances the clock by one cycle" in instant and "response time" in timed and "ends the episode" in strict


def test_the_briefing_counts_cycles_in_their_number_and_closes_the_end_conditions_with_or():
    """A count of cycles takes the singular only for one cycle, and the end conditions read as one list closed by "or",
    with the budget stop only where the episode has a budget."""
    world = episode.load_world(str(default_facts_path()))

    def text(**fields):
        config = EpisodeConfig(guidance_level=2, **fields)
        return briefing(HydraulicRigEnv(*world, config).reset(0)["result"], config)

    durations = EpisodeConfig().durations
    free = text(durations={**durations, "read_per_cycle": 0}, budget=None)
    timed = text(durations={**durations, "read_per_cycle": 1})
    assert "(read, 0 cycles per cycle read," in free and "(read, 1 cycle per cycle read," in timed
    assert "The episode ends on finish, at the horizon, at the step cap, or after 1800 seconds of wall time; a component" in free
    assert "The episode ends on finish, at the horizon, at the step cap, after 1800 seconds of wall time, or when an action would exceed the budget" in timed
    assert " 1 cycles" not in free + timed and "cycle(s)" not in free + timed


def test_the_briefing_tells_the_step_of_a_turn_without_a_call_and_the_scored_end_at_the_wall_limit():
    """The briefing tells what the environment does: levels 2 and 3 state once, beside the step cap, that a turn that calls no
    tool of the rig takes one step and the end such turns bring, in the words of the system prompt of the CausalDiagnosis agent,
    opened where the diagnostics server is registered by the rule that a call of that server is no call of the rig and takes no
    step, and after the ends of an episode that an end at the wall limit while every call is answered is scored; level 1 leaves
    the rules to get_asset_profile and states neither. In the instant mode the clock sentence adds that such a step takes its
    cycle too."""
    world = episode.load_world(str(default_facts_path()))
    idle = idle_turn_sentence("rig")
    assert idle == ("A turn that calls no tool of the rig still takes one step, and as many requests in a row without a call as the step "
                    "cap allows end the episode.")
    assert idle_turn_sentence("rig", diagnostics_server=True) == (
        "A call of the diagnostics server is no call of the rig and takes no step, but a turn that calls no tool of the rig still takes "
        "one step, and as many requests in a row without a call as the step cap allows end the episode.")
    for fields in ({}, {"budget": None}, {"clock_mode": "generation_time"}, {"clock_mode": "instant"}, {"diagnostics_server": False}):
        served = fields.get("diagnostics_server", True)
        sentence = idle_turn_sentence("rig", served)
        texts = {}
        for level in (1, 2, 3):
            config = EpisodeConfig(guidance_level=level, **fields)
            texts[level] = briefing(HydraulicRigEnv(*world, config).reset(0)["result"], config)
        assert [texts[level].count(sentence) for level in (1, 2, 3)] == [0, 1, 1], fields
        assert [texts[level].count(WALL_END_SENTENCE) for level in (1, 2, 3)] == [0, 1, 1], fields
        two = texts[2]
        assert two.index("Step cap: ") < two.index(sentence) < two.index("The episode ends ") < two.index(WALL_END_SENTENCE), fields
        assert ("is no call of the rig and takes no step" in two) == served, fields
        instant = "and so does the step of a turn that calls no tool of the rig" in two
        assert instant == (fields.get("clock_mode") == "instant"), fields


def test_the_procedure_declares_when_the_evidence_cannot_separate_the_candidates_and_orders_as_one_visit():
    """Level 3 states, after the reads and before the work orders, that the agent declares insufficient evidence when the evidence
    cannot separate the remaining candidates, in the words of the entry of declare_insufficient_evidence at level 2, with and
    without the diagnostics server, and keeps its repair step: the manual, the orders one after another as one visit, the wait."""
    world = episode.load_world(str(default_facts_path()))
    assert repair_step(EpisodeConfig()) == ("look up the repair manual of every diagnosed fault, create one work order per diagnosed fault, one after "
                             "another, with no other call between them, so that one visit repairs every component after the components "
                             "it depends on, and wait for the notifications of their closes")
    assert "the evidence cannot separate the remaining candidates" in DECLARE_STEP and "declare_insufficient_evidence" in DECLARE_STEP
    for server in (True, False):
        config = EpisodeConfig(guidance_level=3, diagnostics_server=server)
        text = briefing(HydraulicRigEnv(*world, config).reset(0)["result"], config)
        assert "records that the evidence cannot separate the remaining candidates and names the reading that would" in text
        procedure = text[text.index("Procedure: "):]
        assert procedure.count(DECLARE_STEP) == 1 and procedure.count(repair_step(config)) == 1, server
        assert procedure.index("read sensors before diagnosing") < procedure.index(DECLARE_STEP) < procedure.index(repair_step(config)), server
        assert not server or procedure.index("take the reading it names before you diagnose") < procedure.index(DECLARE_STEP)


def test_the_order_texts_state_a_visit_and_its_wait_only_where_an_order_outlasts_its_call():
    """Every sentence a model reads is true on the row where it is served: on every preset of the drivers the work order entry
    of level 2 and the repair step of level 3 state one visit, its wait and its arrival only where an order outlasts its call
    (a clock or the repair card), and the dependencies with their card alone; the engine closes an order inside its call on the
    other rows, and the contract states the visit by row in the words of the visit schedule."""
    world = episode.load_world(str(default_facts_path()))
    for path in sorted(preset_files(PRESETS).values()):
        config = dataclasses.replace(compose(json.loads(path.read_text(encoding="utf-8"))), guidance_level=3)
        env = HydraulicRigEnv(*world, config)
        text = briefing(env.reset(0)["result"], config)
        procedure, later = text[text.index("Procedure: "):], outlasts_its_call(config)
        assert "create one work order per diagnosed fault, one after another, with no other call between them" in procedure, path.stem
        assert ("one visit repairs" in procedure) == later and ("wait for the notifications of their closes" in procedure) == later, path.stem
        assert ("after the components it depends on" in procedure) == ("dependencies" in repair_rules.cards(config)), path.stem
        assert ("closes inside its call" in text) == (not later) and ("form one visit" in text.split("Procedure: ")[0]) == later, path.stem
        assert "latest creation cycle plus lead time" not in text, path.stem
        env.step({"type": "submit_diagnosis", "diagnosis": dict(config.healthy), "evidence_ids": []})
        component = config.components[0]
        env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})
        assert env.clock.order_open_on(component) == later, path.stem  # the engine: an order outlasts its call exactly there
    base = compose(json.loads((CONFIGS / "layer_presets" / "base.json").read_text(encoding="utf-8")))
    timed = compose(json.loads((CONFIGS / "layer_presets" / "real_time.json").read_text(encoding="utf-8")))
    assert repair_step(base) == "look up the repair manual of every diagnosed fault, create one work order per diagnosed fault, one after another, with no other call between them"
    assert repair_step(timed).endswith("with no other call between them, so that one visit repairs them, and wait for the notifications of their closes")
    assert repair_step(EpisodeConfig()).endswith("so that one visit repairs every component after the components it depends on, and wait for the notifications of their closes")
    rule = next(r for r in contract.public_table([contract.ACTION_SERVER], "hyd_1") if r["name"] == "create_work_order")["duration"]
    assert rule.startswith("on a row without a clock an order closes inside its call while the repair card is off; otherwise the standard orders")
    assert "the largest creation cycle plus lead time among the orders not yet begun" in rule and "a repair that has begun keeps its cycles" in rule


def test_the_briefing_reads_the_same_whatever_the_fault_rule_of_the_starts():
    """The fault rule of the starts reaches no agent, so no briefing level changes with it."""
    world = episode.load_world(str(default_facts_path()))
    for level in (1, 2, 3):
        texts = set()
        for low, high in ((1, 2), (1, 1), (2, 2), (0, 0), (0, 2)):
            config = EpisodeConfig(guidance_level=level, fault_rule={"min_faulty": low, "max_faulty": high})
            texts.add(briefing(HydraulicRigEnv(*world, config).reset(0)["result"], config))
        assert len(texts) == 1, level


PRESETS = CONFIGS
SEEDS_READ = range(6)


def _preset_profiles():
    """(name, configuration, the public profile after each of the first seeds, the hidden starts of those seeds) of every
    preset of the drivers, composed as the drivers compose it."""
    world = episode.load_world(str(default_facts_path()))
    for path in sorted(preset_files(PRESETS).values()):
        config = compose(json.loads(path.read_text(encoding="utf-8")))
        env = HydraulicRigEnv(*world, config)
        profiles, starts = [], set()
        for seed in SEEDS_READ:
            profiles.append(env.reset(seed)["result"])
            starts.add(tuple(env.initial))
        assert len(starts) > 1 or len(env.candidates) == 1, path.stem
        yield path.stem, config, profiles


def test_the_opening_and_the_question_keep_their_wording_on_every_preset():
    """Every briefing opens with the sentence on the performance monitor and asks its one question, whatever the preset,
    the guidance level and the hidden start, so no text tells one start from another, a start without a fault included.
    An anchored preset puts its date before the sentence, which then goes on in lower case."""
    for name, preset, profiles in _preset_profiles():
        unit = asset_text(profiles[0])["unit_noun"]
        for level in (1, 2, 3):
            config = dataclasses.replace(preset, guidance_level=level)
            texts = {briefing(profile, config) for profile in profiles}
            assert len(texts) == 1, (name, level)
            text = texts.pop()
            opening = f"The performance monitor flagged the {unit}."
            if config.anchors is not None:
                opening = f"On {config.date_of(0)}, the performance monitor flagged the {unit}."
            assert text.count(opening) == 1 and text.count("erformance monitor") == 1, (name, level)
            if config.question is None:
                assert text.count("Which condition level is each of its components (") == 1, (name, level)
                assert text.count(f"and which work orders restore the {unit}, within ") == 1, (name, level)
            assert text.count(f"create the work orders that restore the {unit}, and finish with a summary.") == 1, (name, level)


def test_the_alarm_rule_stands_once_where_the_notifications_are_listed():
    """Level 2 lists the notifications that every answer carries, and there the briefing says once that an alarm is raised
    once per channel in an episode, the rule of the environment's alarm log, and names what shows a later excursion."""
    assert "once per channel" in ALARM_ONCE and "summarize_changes_since" in ALARM_ONCE
    for name, preset, profiles in _preset_profiles():
        counts = [briefing(profiles[0], dataclasses.replace(preset, guidance_level=level)).count(ALARM_ONCE) for level in (1, 2, 3)]
        assert counts == [0, 1, 1], name


def test_the_scoring_of_a_hold_stands_once_where_the_emergency_mode_is_on_and_in_the_description_of_finish(tmp_path):
    """A component that an emergency order still holds at the end is scored at its level before that order. Level 2 says so
    once, with the emergency clause, on every preset whose emergency card is on and on no other, and the description of
    finish, which every episode lists, ends with the same sentence."""
    seen = set()
    for name, preset, profiles in _preset_profiles():
        counts = [briefing(profiles[0], dataclasses.replace(preset, guidance_level=level)).count(HOLD_SCORING) for level in (1, 2, 3)]
        assert counts == ([0, 1, 1] if preset.emergency_on() else [0, 0, 0]), name
        seen.add(preset.emergency_on())
    assert seen == {True, False}  # the presets hold both cases, so both branches are read
    descriptions = {t.name: t.description for t in build_server(_new_episode(tmp_path))._tool_manager.list_tools()}
    assert descriptions["finish"].count(HOLD_SCORING) == 1 and descriptions["finish"].endswith(HOLD_SCORING)
    assert not [name for name, text in descriptions.items() if name != "finish" and HOLD_SCORING in text]


# ----- the contract -----
def test_advertised_tools_and_hints_match_the_contract(tmp_path):
    tools = build_server(_new_episode(tmp_path))._tool_manager.list_tools()
    hints = {t.name: t.annotations.readOnlyHint for t in tools}
    parameters = {t.name: {k: v["type"] for k, v in t.parameters["properties"].items()} for t in tools}
    assert contract.check_advertised(contract.ACTION_SERVER, hints, parameters) == []
    assert set(hints) == TOOLS and all(t.parameters.get("required", []) == list(t.parameters["properties"]) for t in tools)
    assert hints["read_sensors"] is True and hints["get_current_time"] is True and hints["wait_for_notification"] is True
    assert hints["create_work_order"] is False and hints["finish"] is False and "advance_time" not in hints
    assert contract.check_advertised(contract.ACTION_SERVER, {**hints, "finish": True}) == ["finish advertises readOnlyHint True but is a write in the contract"]
    assert contract.check_advertised(contract.ACTION_SERVER, {k: v for k, v in hints.items() if k != "finish"}) == ["finish is in the contract but not advertised"]


def test_the_contract_names_both_servers_and_the_proof_of_concept_aliases():
    tools = contract.load_contract()["tools"]
    assert {e["server"] for e in tools.values()} == {contract.ACTION_SERVER, contract.DIAGNOSTICS_SERVER}
    assert set(contract.tools_of(contract.DIAGNOSTICS_SERVER)) == {"get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome"}
    assert all(e["class"] == "read" and not e["scored"] and e["duration"]["cycles_per_unit"] == 0 for e in contract.tools_of(contract.DIAGNOSTICS_SERVER).values())
    assert {name for name, e in tools.items() if e["scored"]} == {"submit_diagnosis", "declare_insufficient_evidence", "create_work_order", "finish"}
    assert {name for name, e in tools.items() if e["class"] == "write"} == {name for name, e in tools.items() if e["scored"]}
    aliases = {a: name for name, e in tools.items() for a in e["aliases"]}
    assert aliases == {"inspect": "read_sensors", "diagnose": "submit_diagnosis", "replace": "create_work_order"}
    assert [row["name"] for row in contract.public_table([contract.ACTION_SERVER])] == list(contract.tools_of(contract.ACTION_SERVER))
    assert len(TOOLS) == 12 and {name for name in TOOLS if contract.tool_class(name) == "read"} >= {"get_fix_history", "lookup_fix", "get_current_time"}


# ----- policies through the tools -----
def test_every_action_has_one_tool_call_and_evidence_lists_round_trip():
    diagnosis = dict(HEALTHY)
    assert tool_call({"type": "get_asset_profile"}) == ("get_asset_profile", {})
    assert tool_call({"type": "read_sensors", "sensor_group": "tier2", "cycles": 3}) == ("read_sensors", {"sensor_group": "tier2", "cycles": 3})
    assert tool_call({"type": "get_current_time"}) == ("get_current_time", {})
    assert tool_call({"type": "wait_for_notification", "timeout": 4}) == ("wait_for_notification", {"timeout": 4})
    assert tool_call({"type": "wait_for_notification"}) == ("wait_for_notification", {"timeout": None})
    assert tool_call({"type": "summarize_changes_since", "cycle": 2}) == ("summarize_changes_since", {"cycle": 2})
    assert tool_call({"type": "get_failure_modes", "component": ""}) == ("get_failure_modes", {"component": ""})
    assert tool_call({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": ["obs/1/a", "wo/wo-1"]}) == (
        "submit_diagnosis", {**diagnosis, "evidence_ids": "obs/1/a,wo/wo-1"})
    assert tool_call({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier3", "cycles": 2}, "evidence_ids": []}) == (
        "declare_insufficient_evidence", {"sensor_group": "tier3", "cycles": 2, "evidence_ids": ""})
    assert tool_call({"type": "create_work_order", "component": "pump", "prediction_id": "pred/1", "evidence_ids": ["obs/2/b"]}) == (
        "create_work_order", {"component": "pump", "prediction_id": "pred/1", "evidence_ids": "obs/2/b"})
    assert tool_call({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": [], "mode": "emergency"}) == (
        "create_work_order", {"component": "pump", "prediction_id": "", "evidence_ids": "", "mode": "emergency"})
    assert tool_call({"type": "get_fix_history"}) == ("get_fix_history", {}) and tool_call({"type": "lookup_fix", "component": "valve"}) == (
        "lookup_fix", {"component": "valve"})
    assert tool_call({"type": "finish", "summary": "done"}) == ("finish", {"summary": "done"})
    for removed in ({"type": "wait"}, {"type": "advance_time", "cycles": 4}):
        with pytest.raises(ValueError):
            tool_call(removed)
    assert evidence_list("obs/1/a, wo/wo-1,,") == ["obs/1/a", "wo/wo-1"] and evidence_list("") == [] and evidence_list(None) is None


def test_do_nothing_and_greedy_through_the_tools_take_their_in_process_actions(tmp_path):
    cycles, blocks = episode.load_world(str(default_facts_path()))
    path = _new_episode(tmp_path / "nothing", seed=0) if (tmp_path / "nothing").mkdir() is None else None
    play(DoNothingPolicy(), episode.opening(episode.load_state(path)), _send(path))
    env = episode.replay(episode.load_state(path))
    assert env.done and env.log[-1]["reason"] == "horizon"
    assert json.dumps(env.log) == json.dumps(run_episode(HydraulicRigEnv(cycles, blocks, EpisodeConfig()), DoNothingPolicy(), 0))
    classifiers = ServedClassifiers(TierClassifiers(cycles, blocks), cycles)
    for seed in range(3):
        in_process = run_episode(HydraulicRigEnv(cycles, blocks, EpisodeConfig()), GreedyPolicy(classifiers), seed)
        folder = tmp_path / f"greedy_{seed}"
        folder.mkdir()
        path = _new_episode(folder, seed=seed)
        play(GreedyPolicy(classifiers), episode.opening(episode.load_state(path)), _send(path))
        assert json.dumps(episode.replay(episode.load_state(path)).log) == json.dumps(in_process)


# ----- the server over stdio -----
def isolated_launcher(path, src):
    """The in-tree launcher with the entries of src, joined by os.pathsep as in PYTHONPATH (the repository root and the fork's
    src), put on the import path first, since a server process starts under python -I (the thinking clock refuses the counted
    calls of a process that is not isolated) and -I ignores PYTHONPATH."""
    text = (Path(__file__).resolve().parents[1] / "launcher.py").read_text(encoding="utf-8")
    path.write_text(f"import sys\nsys.path[:0] = {src.split(os.pathsep)!r}\n" + text, encoding="utf-8")


@needs_mcp
def test_server_offers_twelve_flat_tools_with_hints_and_answers_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path = _anchored(tmp_path)
    truth = _truth(path)
    src = os.pathsep.join([str(ROOT), str(SRC)])
    isolated_launcher(tmp_path / "rig_server.py", src)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    legal = [
        ("get_asset_profile", {}), ("read_sensors", {"sensor_group": "tier1", "cycles": 2}), ("wait_for_notification", {"timeout": 1}),
        ("get_current_time", {}), ("summarize_changes_since", {"cycle": 0}), ("get_failure_modes", {"component": "cooler"}),
        ("submit_diagnosis", {**truth, "evidence_ids": "obs/1/" + tier_fields(1)[0] + "_mean,alarm/0/x"}),
        ("declare_insufficient_evidence", {"sensor_group": "tier3", "cycles": 1, "evidence_ids": ""}),
        ("create_work_order", {"component": "cooler", "prediction_id": "", "evidence_ids": "obs/2/" + tier_fields(1)[0] + "_std"}),
        ("finish", {"summary": "cooler ordered"}),
    ]

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=["-I", str(tmp_path / "rig_server.py")], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                answers = [await session.call_tool(tool, arguments) for tool, arguments in legal[:-1]]
                bad = [await session.call_tool(tool, arguments) for tool, arguments, _ in MALFORMED]
                answers.append(await session.call_tool(*legal[-1]))
                return tools, answers, bad

    tools, answers, bad = asyncio.run(session_calls())
    assert set(tools) == TOOLS
    hints = {name: t.annotations.readOnlyHint for name, t in tools.items()}
    parameters = {name: {k: v["type"] for k, v in t.inputSchema["properties"].items()} for name, t in tools.items()}
    assert contract.check_advertised(contract.ACTION_SERVER, hints, parameters) == []
    assert all(schema.inputSchema.get("required", []) == list(schema.inputSchema["properties"]) for schema in tools.values())
    assert not any("default" in p for schema in tools.values() for p in schema.inputSchema["properties"].values())
    assert tools["finish"].inputSchema["properties"]["summary"]["type"] == "string" and len(tools["finish"].inputSchema["properties"]) == 1
    assert not any(a.isError for a in answers + bad)
    decoded = [json.loads(a.content[0].text) for a in answers]
    assert "normal_ranges" in decoded[0]["result"] and len(decoded[1]["result"]["served"]) == 2
    assert decoded[2]["result"] == {"time": 3, "waited": 1, "ended_by": "timeout"} and decoded[2]["notifications"][0]["event"] == "wait_timeout"
    assert decoded[3]["time"] == decoded[3]["result"]["time"] == 3 and decoded[3]["result"]["horizon"] == EpisodeConfig().horizon
    assert decoded[4]["result"]["cycles_elapsed"] == 3 and decoded[5]["result"]["component"] == "cooler"
    assert decoded[6]["result"]["evidence_ids"] == ["obs/1/" + tier_fields(1)[0] + "_mean", "alarm/0/x"]
    assert decoded[7]["refused"] == "a diagnosis or a declaration was already submitted"
    assert decoded[8]["result"]["work_order"]["id"] == "wo-1" and decoded[8]["open_work_orders"][0]["component"] == "cooler"
    assert decoded[9]["episode_over"] and decoded[9]["result"] == {"summary": "cooler ordered"} and "refused" not in decoded[9]
    refusals = [json.loads(b.content[0].text) for b in bad]
    assert all("refused" in r for r in refusals)
    state = episode.load_state(path)
    assert state["actions"][len(legal) - 1:-1] == [action for _, _, action in MALFORMED]
    assert refusals[-1]["steps_left"] == EpisodeConfig().max_steps - (len(legal) - 1) - len(MALFORMED)


@needs_mcp
def test_get_fix_history_lookup_fix_and_an_emergency_order_round_trip_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path = _anchored(tmp_path)
    truth = _truth(path)
    src = os.pathsep.join([str(ROOT), str(SRC)])
    isolated_launcher(tmp_path / "rig_server.py", src)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    calls = [
        ("get_fix_history", {}), ("lookup_fix", {"component": "pump"}), ("lookup_fix", {"component": "turbine"}),
        ("submit_diagnosis", {**truth, "evidence_ids": ""}),
        ("create_work_order", {"component": "cooler", "prediction_id": "", "evidence_ids": "", "mode": "emergency"}),
        ("create_work_order", {"component": "valve", "prediction_id": "", "evidence_ids": ""}),
    ]

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=["-I", str(tmp_path / "rig_server.py")], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [await session.call_tool(tool, arguments) for tool, arguments in calls]

    answers = [json.loads(a.content[0].text) for a in asyncio.run(session_calls())]
    direct = episode.replay(episode.load_state(path))
    assert answers[0]["result"] == {"history": direct.fix_history} and answers[1]["result"] == direct.fix_manual("pump")
    assert answers[2]["refused"] == "unknown component" and "result" not in answers[2]
    assert answers[4]["result"]["work_order"]["mode"] == "emergency" and answers[5]["result"]["work_order"]["mode"] == "standard"
    assert answers[4]["result"]["work_order"]["visit"] != answers[5]["result"]["work_order"]["visit"]  # an emergency order is its own visit
    actions = episode.load_state(path)["actions"]
    assert actions[:2] == [{"type": "get_fix_history"}, {"type": "lookup_fix", "component": "pump"}]
    # the mode reaches the environment only when the client sends it, so an order without it is the action it always was
    assert actions[4]["mode"] == "emergency" and "mode" not in actions[5]
    released = json.loads(episode.released_path(path).read_text())
    assert [e["event"] for e in released["events"][:2]] == ["fix_history", "fix_lookup"] and released["events"][0]["history"] == direct.fix_history


# ----- integer arguments written as numbers or strings -----
FILLERS = {"sensor_group": "tier1", "evidence_ids": "", "prediction_id": "", "component": "cooler", "mode": "standard", "summary": "done"}
INTEGERS = {"cycles": 2, "timeout": 2, "cycle": 0}  # a legal value of each integer parameter at the start of an episode


def _integer_parameters(server):
    """{tool: [its parameters of type integer]} of the tools that a server lists."""
    return {t.name: [k for k, v in t.parameters["properties"].items() if v.get("type") == "integer"] for t in server._tool_manager.list_tools()}


def test_every_integer_parameter_that_a_tool_declares_reads_a_quoted_number(tmp_path):
    """The parameters of type integer come from the tool definitions that the server lists, the levels of the diagnosis
    among them, and each tool is called once with the integer and once with its decimal string: the two calls give the same
    log entry. A tool that gains an integer parameter without the one reader of the environment fails here unchanged."""
    (tmp_path / "probe").mkdir()
    declared = {tool: names for tool, names in _integer_parameters(build_server(_new_episode(tmp_path / "probe"))).items() if names}
    assert {"read_sensors", "wait_for_notification", "summarize_changes_since", "submit_diagnosis", "declare_insufficient_evidence"} <= set(declared)
    for tool, names in declared.items():
        entries = []
        for written in (lambda value: value, str):
            folder = tmp_path / f"{tool}_{'string' if written is str else 'number'}"
            folder.mkdir()
            path = _new_episode(folder)
            truth = _truth(path)
            server = build_server(path)
            properties = next(t for t in server._tool_manager.list_tools() if t.name == tool).parameters["properties"]
            arguments = {name: written(truth.get(name, INTEGERS.get(name, 1))) if name in names else FILLERS.get(name, "") for name in properties}
            asyncio.run(server.call_tool(tool, arguments))
            entries.append(episode.replay(episode.load_state(path)).log[-1])
        assert entries[0]["event"] != "refused" and entries[0] == entries[1], (tool, entries)


def test_a_quoted_number_stays_as_sent_in_the_state_file_and_is_read_as_the_number(tmp_path):
    """Through the tools, the true levels sent as strings are accepted: the action and the tool call of the state file hold
    the strings as sent, the log entry holds the integers, and a replay of the state file gives the same log."""
    path = _new_episode(tmp_path)
    truth = _truth(path)
    quoted = {c: str(level) for c, level in truth.items()}
    server = build_server(path)
    reply = json.loads(asyncio.run(server.call_tool("submit_diagnosis", {**quoted, "evidence_ids": ""}))[0].text)
    assert "refused" not in reply and reply["result"]["diagnosis"] == truth
    state = episode.load_state(path)
    assert state["actions"][-1]["diagnosis"] == quoted and state["tool_calls"][-1]["arguments"] == {**quoted, "evidence_ids": ""}
    env = episode.replay(state)
    assert env.log[-1]["event"] == "submit_diagnosis" and env.log[-1]["diagnosis"] == truth
    assert episode.replay(episode.load_state(path)).log == env.log


# ----- the start without a fault -----
@pytest.mark.parametrize("orders", (0, 1))
def test_all_healthy_and_no_work_order_is_a_valid_and_scored_answer_through_the_tools(tmp_path, orders):
    """On the control preset, through the tools of the action server, the diagnosis that names every component at its healthy
    level and the finish are accepted with no refusal, and the verdict passes with no false alarm and no unnecessary order.
    A work order before the finish is executed, counted as unnecessary, and fails the pass."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

    config = compose(json.loads((PRESETS / "healthy_control.json").read_text(encoding="utf-8")))
    path = tmp_path / "episode.json"
    episode.start(path, 3, config, default_facts_path())
    episode.hand_over(path)
    healthy = {c: config.healthy[c] for c in config.components}
    # a reading before the diagnosis, so that the class names the order
    answers = [_call(path, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1}),
               _call(path, "submit_diagnosis", {**healthy, "evidence_ids": ""},
                     {"type": "submit_diagnosis", "diagnosis": healthy, "evidence_ids": []})]
    for component in config.components[:orders]:
        answers.append(_call(path, "create_work_order", {"component": component, "prediction_id": "", "evidence_ids": ""},
                             {"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []}))
    answers.append(_call(path, "finish", {"summary": "every component is healthy"}, {"type": "finish", "summary": "every component is healthy"}))
    assert not [a for a in answers if "refused" in a]
    env = episode.replay(episode.load_state(path))
    cost = oracle_cost(HydraulicRigEnv(*episode.load_world(str(default_facts_path())), config), 3)
    verdict = verify(env.log, released_from_env(env), env.oracle(), cost)
    assert verdict["verified"] and verdict["false_alarms"] == 0 and verdict["unnecessary"] == orders
    assert verdict["pass"] == (orders == 0)
    if orders:
        assert events_named(env.log, "work_order_closed") and verdict["failure_class"] == "unnecessary_replacement"


def events_named(log, name):
    return [e for e in log if e["event"] == name]


def test_the_description_of_get_asset_profile_names_every_part_of_the_profile_on_every_preset(tmp_path):
    """The description of the profile tool is built from the parts of the profile, so it names what the tool returns: on every
    preset the keys of the public profile are those of PROFILE_PARTS, each in one part, the fields of its processes block are
    those of PROCESS_WORDS, and the description that the server lists holds the words of every part and of every process."""
    keys = [key for part, _words in PROFILE_PARTS for key in part]
    assert len(keys) == len(set(keys))
    for name, _config, profiles in _preset_profiles():
        assert all(set(profile) == set(keys) and set(profile["processes"]) == set(PROCESS_WORDS) for profile in profiles), name
    tools = build_server(_new_episode(tmp_path))._tool_manager.list_tools()
    description = next(t.description for t in tools if t.name == "get_asset_profile")
    assert all(words in description for _keys, words in PROFILE_PARTS)
    assert all(word in description for word in PROCESS_WORDS.values())


def test_the_rig_briefing_of_an_asset_that_states_its_sentences_shows_each_once_after_the_date(tmp_path):
    """The briefing of the rig's own row takes the question, the opening, the scoring sentence, the meaning of the levels
    and the sentence on the readings that an asset file states, each once, and on an anchored start the opening follows the
    date with its first letter in lower case."""
    from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, recordings as records
    from CausalDiagnosis.datasets.dataset_loader.evidence_source import RecordingSource
    from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import register_profile, unregister_profile
    from CausalDiagnosis.datasets.tests.conftest import TINY_FILE, write_recordings

    stated = {"question": "Which level does each of the {components} of the bench stand at, within {limits}?",
              "opening": "The bench monitor flagged a deviation.", "scoring": "The diagnosis is scored at the cycle it is submitted.",
              "level_meaning": "A larger belt level is a tighter belt.", "readings": "A reading is one summary of a recorded cycle of the bench."}
    kind = kinds.register_kind(kinds.EvidenceKind("tiny_stated", record_keys=("label_keys", "stable_flag_stable"), check=records.record_problems,
                                                  loader=records.load_recordings, source=RecordingSource))
    data = json.loads(TINY_FILE.read_text(encoding="utf-8"))
    data["evidence_kind"] = kind.name
    data["text"].update(stated)
    profile = register_profile(AssetProfile.from_json(data))
    try:
        cycles = records.load_cycles(write_recordings(tmp_path / "stated.jsonl", profile), profile)
        config = dataclasses.replace(profile.default_config(), anchors={"date0": "2026-03-02"}, guidance_level=2)
        env = HydraulicRigEnv(cycles, records.build_blocks(cycles), config)
        text = briefing(env.reset(0)["result"], config)
        assert text.count("On 2026-03-02, the bench monitor flagged a deviation.") == 1
        for key in ("scoring", "level_meaning", "readings"):
            assert text.count(stated[key]) == 1, key
        assert text.count("Which level does each of the fan, belt, bearing of the bench stand at, within a budget of ") == 1
        assert stated["opening"] not in text and "Every reading is a recorded cycle" not in text
    finally:
        unregister_profile(profile.asset_id)
        kinds.unregister_kind(kind.name)


def test_two_concurrent_calls_of_one_episode_keep_both_actions(tmp_path, monkeypatch):
    """The action server holds the lock of the state file from the load of the state to its save, as the metering layer
    does, so two calls that arrive together run one after the other: while the replay of one is slow the other waits, and
    the state file keeps both actions and both tool calls."""
    import threading

    path = _new_episode(tmp_path)
    replay = episode.replay

    def slow(state, *args, **kwargs):
        env = replay(state, *args, **kwargs)
        time.sleep(0.3)
        return env

    monkeypatch.setattr(episode, "replay", slow)
    calls = [threading.Thread(target=_call, args=(path, "get_current_time", {}, {"type": "get_current_time"})) for _ in range(2)]
    for call in calls:
        call.start()
    for call in calls:
        call.join()
    state = episode.load_state(path)
    assert len(state["actions"]) == len(state["tool_calls"]) == 2


@pytest.mark.parametrize("preset", sorted(p.stem for p in preset_files(PRESETS).values()))
def test_the_tool_texts_follow_the_cards_and_the_manual_names_the_indicating_channels(tmp_path, preset):
    """The texts of lookup_fix and create_work_order name the emergency mode, the visit charge and the dependencies only
    where the episode switches their card on, create_work_order advertises its mode only with the emergency mode, the time
    a free tool states follows the clock, and the manual answers with the indicating channels of the component in a field
    of its own, the channels its effect names."""
    config = compose(json.loads(preset_path(preset).read_text(encoding="utf-8")))
    path = tmp_path / "episode.json"
    episode.start(path, 3, config, default_facts_path())
    episode.hand_over(path)
    cards = repair_rules.cards(config)
    tools = {t.name: t for t in asyncio.run(build_server(path).list_tools())}
    order = tools["create_work_order"]
    assert ("emergency" in tools["lookup_fix"].description) == ("emergency" in order.description) == ("emergency" in cards)
    assert ("mode" in order.inputSchema["properties"]) == ("emergency" in cards)
    assert ("visit charge" in order.description) == ("visits" in cards) and ("depends on" in order.description) == ("dependencies" in cards)
    assert ("takes one cycle" in tools["get_fix_history"].description) == (config.clock_mode == "instant")
    env = episode.replay(episode.load_state(path))  # the engine behind the tool, which every clock mode answers in process
    for component in config.components:
        manual = env.step({"type": "lookup_fix", "component": component})[0]["result"]
        assert manual["indicating_channels"] and all(field in manual["effect"] for field in manual["indicating_channels"])
        assert ("emergency" in manual) == ("emergency" in cards)
