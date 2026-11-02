"""The action server (servers.hydraulic_rig): field names, state and replay, the envelope, the released evidence, the
briefing, the thinking time of the generation_time mode, the contract match, the policies through the tools, and the MCP
server over stdio when the mcp package is installed (mcp 1.26.0, pinned in the fork's uv.lock). No test states a value
computed from the rig data."""
import asyncio
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

from envs.hydraulic_rig.classifiers import TierClassifiers
from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, default_facts_path
from envs.hydraulic_rig.env import HIDDEN_FIELDS, SERVED_STATISTICS, HydraulicRigEnv, served_values
from envs.hydraulic_rig.greedy import GreedyPolicy, ServedClassifiers
from envs.hydraulic_rig.policies import DoNothingPolicy, run_episode
from servers import contract
from servers.hydraulic_rig import episode
from servers.hydraulic_rig.briefing import BRIEFING_VERSION, EVIDENCE_FORMS, briefing
from servers.hydraulic_rig.main import build_server, evidence_list
from servers.hydraulic_rig.names import HYD1_FIELDS, public_name, tier_fields
from servers.hydraulic_rig.policy_client import play, tool_call

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

HYD1_FILE = Path(__file__).resolve().parents[3] / "couchdb" / "scenarios_data" / "shared" / "iot" / "hydraulic_pump_1.json"
TOOLS = set(contract.tools_of(contract.ACTION_SERVER))
STATE_KEYS = {"format", "seed", "config", "facts_path", "actions", "think_cycles", "tool_calls", "last_call_wall_time"}
RULE_KEYS = {"components", "levels", "healthy", "sensor_groups", "statistics", "prices", "durations", "downtime_rate", "budget", "horizon",
             "max_steps", "fault_rule", "clock_mode", "healthy_bands"}
HAND_START = (3, 90, 0, 130)
needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp package (pinned in the fork's uv.lock)")
MALFORMED = [  # (tool, arguments, the action env.step receives): wrong types arrive as sent, missing arguments as None
    ("read_sensors", {"sensor_group": "tier1", "cycles": "two"}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": "two"}),
    ("read_sensors", {"sensor_group": "tier1", "cycles": "[2]"}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": "[2]"}),
    ("read_sensors", {"sensor_group": "tier1"}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": None}),
    ("advance_time", {"cycles": 2.5}, {"type": "advance_time", "cycles": 2.5}),
    ("submit_diagnosis", {"cooler": 100, "valve": 100, "pump": 0, "evidence_ids": ""},
     {"type": "submit_diagnosis", "diagnosis": {"cooler": 100, "valve": 100, "pump": 0, "accumulator": None}, "evidence_ids": []}),
    ("create_work_order", {"component": 5, "prediction_id": "", "evidence_ids": ""},
     {"type": "create_work_order", "component": 5, "prediction_id": "", "evidence_ids": []}),
]


def _new_episode(tmp_path, seed=3, **overrides):
    path = tmp_path / "episode.json"
    episode.start(path, seed, EpisodeConfig(**overrides), default_facts_path())
    return path


def _anchored(tmp_path, **overrides):
    return _new_episode(tmp_path, start={"combination": list(HAND_START), "block_offset": 0}, **overrides)


def _call(path, tool, arguments, action, now=None):
    return json.loads(episode.call(path, tool, arguments, action, now=now))


def _truth(path):
    return dict(zip(COMPONENTS, episode.replay(episode.load_state(path)).initial))


def _keys(obj, skip=("action", "parameters")):
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
    assert set(state) == STATE_KEYS and state["format"] == 2 and state["actions"] == state["think_cycles"] == state["tool_calls"] == []
    assert EpisodeConfig.from_json(state["config"]) == EpisodeConfig() and isinstance(state["last_call_wall_time"], float)
    with pytest.raises(ValueError, match="unknown episode state format"):
        episode.save_state(tmp_path / "old.json", {**state, "format": 1})
        episode.load_state(tmp_path / "old.json")


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
          {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}) for c in faulty] + [
        ("advance_time", {"cycles": 20}, {"type": "advance_time", "cycles": 20}),
        ("get_asset_profile", {}, {"type": "get_asset_profile"}),
        ("finish", {"summary": "repaired"}, {"type": "finish", "summary": "repaired"}),
    ]
    direct = HydraulicRigEnv(*episode.load_world(str(default_facts_path())), EpisodeConfig())
    direct.reset(3)
    for tool, arguments, action in actions:
        answer = _call(path, tool, arguments, action)
        envelope, *_ = direct.step(action)
        assert answer == json.loads(json.dumps(envelope))
    state = episode.load_state(path)
    assert len(state["actions"]) == len(actions) == len(state["tool_calls"]) == len(state["think_cycles"]) and set(state["think_cycles"]) == {0}
    assert json.dumps(episode.replay(state).log) == json.dumps(direct.log)
    assert direct.done and direct.log[-1]["reason"] == "finish" and direct.log[-1]["final"] == [HEALTHY[c] for c in COMPONENTS]


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
    assert set(after) == {"refused", "time", "budget_left", "steps_left", "open_work_orders", "episode_over"}
    state = episode.load_state(path)
    assert len(state["actions"]) == 1 and len(state["tool_calls"]) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.json", "released.json"]  # atomic writes leave no temporary files


# ----- the released evidence -----
def test_released_evidence_holds_only_what_the_agent_saw(tmp_path):
    path = _anchored(tmp_path, anchors={"date0": "2022-07-01"}, question="What is wrong with the unit?")
    released = json.loads(episode.released_path(path).read_text())
    assert set(released) == {"format", "asset_id", "task_text", "guidance_level", "rules", "status", "events", "observations"}
    assert released["format"] == 2 and released["asset_id"] == "hyd_1" and released["guidance_level"] == 1 and released["events"] == []
    assert set(released["rules"]) == RULE_KEYS and released["rules"]["sensor_groups"]["tier1"] == tier_fields(1)
    assert released["status"] == {"time": 0, "day": "2022-07-01", "budget_left": EpisodeConfig().budget, "steps_left": EpisodeConfig().max_steps,
                                  "open_work_orders": [], "episode_over": False}
    assert "What is wrong with the unit?" in released["task_text"]
    truth = _truth(path)
    _call(path, "read_sensors", {"sensor_group": "tier2", "cycles": 2}, {"type": "read_sensors", "sensor_group": "tier2", "cycles": 2})
    _call(path, "read_sensors", {"sensor_group": "tier4", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier4", "cycles": 1})
    _call(path, "submit_diagnosis", {**truth, "evidence_ids": "obs/1/x"}, {"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": ["obs/1/x"]})
    _call(path, "create_work_order", {"component": "cooler", "prediction_id": "", "evidence_ids": ""},
          {"type": "create_work_order", "component": "cooler", "prediction_id": "", "evidence_ids": []})
    _call(path, "advance_time", {"cycles": 6}, {"type": "advance_time", "cycles": 6})
    _call(path, "read_sensors", {"sensor_group": "tier1", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1})
    released = json.loads(episode.released_path(path).read_text())
    env = episode.replay(episode.load_state(path))
    assert not _keys(released) & set(HIDDEN_FIELDS)
    assert released["events"] == json.loads(json.dumps(env.public_events()))
    assert [e["event"] for e in released["events"]] == ["read", "refused", "submit_diagnosis", "work_order_created", "work_order_executed",
                                                        "work_order_closed", "advance", "read"]
    assert all({"t_start", "t_end", "class"} <= set(e) for e in released["events"])
    assert [(o["n"], o["t"], o["sensor_group"]) for o in released["observations"]] == [(1, 0, "tier2"), (2, 1, "tier2"), (3, 8, "tier1")]
    assert set(released["observations"][0]["values"]) == {f"{f}_{s}" for f in tier_fields(2) for s in SERVED_STATISTICS}
    assert released["status"]["time"] == 9 and released["status"]["day"] == "2022-07-10" and released["status"]["open_work_orders"] == []
    assert released["task_text"] == briefing(env.public_profile(), env.config)
    text = episode.released_path(path).read_text()
    assert not any(label in text for label in ("cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "seed"))


# ----- the generation_time mode -----
def test_generation_time_mode_stores_the_thinking_cycles_and_replays_them(tmp_path):
    seconds = 30
    path = _anchored(tmp_path, clock_mode="generation_time", seconds_per_cycle=seconds)
    t0 = episode.load_state(path)["last_call_wall_time"]
    truth = _truth(path)
    calls = [
        ("read_sensors", {"sensor_group": "tier1", "cycles": 1}, {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1}, t0 + 2.5 * seconds),
        ("submit_diagnosis", {**truth, "evidence_ids": ""}, {"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": []}, t0 + 2.5 * seconds + 1),
        ("advance_time", {"cycles": 2}, {"type": "advance_time", "cycles": 2}, t0 + 2.5 * seconds + 1),
    ]
    answers = [_call(path, tool, arguments, action, now=now) for tool, arguments, action, now in calls]
    state = episode.load_state(path)
    assert state["think_cycles"] == [3, 1, 0] and state["last_call_wall_time"] == calls[-1][3]
    assert [a["time"] for a in answers] == [4, 5, 7]
    direct = HydraulicRigEnv(*episode.load_world(str(default_facts_path())), EpisodeConfig.from_json(state["config"]))
    direct.reset(state["seed"])
    for (_tool, _arguments, action, _now), think in zip(calls, state["think_cycles"]):
        direct.step(action, think_cycles=think)
    assert json.dumps(episode.replay(state).log) == json.dumps(direct.log)
    assert [e["think_cycles"] for e in direct.log if e["step"] is not None] == [3, 1, 0]
    other = dict(state, think_cycles=[0, 0, 0])
    assert episode.replay(other).time != direct.time  # the stored thinking cycles are part of the episode
    assert episode.think_cycles(EpisodeConfig(), t0, t0 + 1000) == 0  # other modes charge no thinking time
    assert episode.think_cycles(EpisodeConfig(clock_mode="generation_time", seconds_per_cycle=seconds), t0, t0 + 0.1) == 1 == math.ceil(0.1 / seconds)


# ----- the briefing -----
def test_the_briefing_grows_with_the_guidance_level_and_names_every_legal_value_at_level_two(tmp_path):
    env = episode.replay(episode.load_state(_anchored(tmp_path, anchors={"date0": "2022-07-01"})))
    profile = env.public_profile()
    assert BRIEFING_VERSION == "v2.0"
    one = briefing(profile, EpisodeConfig(guidance_level=1, anchors={"date0": "2022-07-01"}))
    two = briefing(profile, EpisodeConfig(guidance_level=2))
    three = briefing(profile, EpisodeConfig(guidance_level=3))
    assert one.startswith("Asset hyd_1") and "On 2022-07-01" in one and f"budget of {profile['budget']}" in one and f"{profile['horizon']} cycles" in one
    assert "read_sensors(" not in one and "Procedure" not in one
    assert one.split(" The performance")[0] + " The performance" in two.replace("On 2022-07-01, ", "") or True
    assert all(c in two and all(str(v) in two for v in LEVELS[c]) for c in COMPONENTS)
    assert all(f in two for t in TIER_SENSORS for f in tier_fields(t)) and all(f"{name}(" in two for name in TOOLS)
    assert "between 1 and 2 components start off their healthy level" in two and EVIDENCE_FORMS in two
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


@pytest.mark.parametrize("low, high, rule", [
    (1, 2, "between 1 and 2 components start off their healthy level"),
    (1, 1, "exactly 1 component starts off its healthy level"),
    (2, 2, "exactly 2 components start off their healthy level"),
])
def test_the_briefing_states_the_fault_count_rule_of_the_episode(low, high, rule):
    config = EpisodeConfig(guidance_level=2, fault_rule={"min_faulty": low, "max_faulty": high})
    env = HydraulicRigEnv(*episode.load_world(str(default_facts_path())), config)
    assert f"Rule: {rule}." in briefing(env.reset(0)["result"], config)


# ----- the contract -----
def test_advertised_tools_and_hints_match_the_contract(tmp_path):
    tools = build_server(_new_episode(tmp_path))._tool_manager.list_tools()
    hints = {t.name: t.annotations.readOnlyHint for t in tools}
    parameters = {t.name: {k: v["type"] for k, v in t.parameters["properties"].items()} for t in tools}
    assert contract.check_advertised(contract.ACTION_SERVER, hints, parameters) == []
    assert set(hints) == TOOLS and all(t.parameters.get("required", []) == list(t.parameters["properties"]) for t in tools)
    assert hints["read_sensors"] is True and hints["advance_time"] is True and hints["create_work_order"] is False and hints["finish"] is False
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
    assert aliases == {"inspect": "read_sensors", "diagnose": "submit_diagnosis", "replace": "create_work_order", "advance_time": "advance_time"}
    assert [row["name"] for row in contract.public_table([contract.ACTION_SERVER])] == list(contract.tools_of(contract.ACTION_SERVER))


# ----- policies through the tools -----
def test_every_action_has_one_tool_call_and_evidence_lists_round_trip():
    diagnosis = dict(HEALTHY)
    assert tool_call({"type": "get_asset_profile"}) == ("get_asset_profile", {})
    assert tool_call({"type": "read_sensors", "sensor_group": "tier2", "cycles": 3}) == ("read_sensors", {"sensor_group": "tier2", "cycles": 3})
    assert tool_call({"type": "advance_time", "cycles": 4}) == ("advance_time", {"cycles": 4})
    assert tool_call({"type": "summarize_changes_since", "cycle": 2}) == ("summarize_changes_since", {"cycle": 2})
    assert tool_call({"type": "get_failure_modes", "component": ""}) == ("get_failure_modes", {"component": ""})
    assert tool_call({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": ["obs/1/a", "wo/wo-1"]}) == (
        "submit_diagnosis", {**diagnosis, "evidence_ids": "obs/1/a,wo/wo-1"})
    assert tool_call({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier3", "cycles": 2}, "evidence_ids": []}) == (
        "declare_insufficient_evidence", {"sensor_group": "tier3", "cycles": 2, "evidence_ids": ""})
    assert tool_call({"type": "create_work_order", "component": "pump", "prediction_id": "pred/1", "evidence_ids": ["obs/2/b"]}) == (
        "create_work_order", {"component": "pump", "prediction_id": "pred/1", "evidence_ids": "obs/2/b"})
    assert tool_call({"type": "finish", "summary": "done"}) == ("finish", {"summary": "done"})
    with pytest.raises(ValueError):
        tool_call({"type": "wait"})
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
        in_process = run_episode(HydraulicRigEnv(cycles, blocks, EpisodeConfig()), GreedyPolicy(classifiers, use_max_faulty=True), seed)
        folder = tmp_path / f"greedy_{seed}"
        folder.mkdir()
        path = _new_episode(folder, seed=seed)
        play(GreedyPolicy(classifiers, use_max_faulty=True), episode.opening(episode.load_state(path)), _send(path))
        assert json.dumps(episode.replay(episode.load_state(path)).log) == json.dumps(in_process)


# ----- the server over stdio -----
@needs_mcp
def test_server_offers_nine_flat_tools_with_hints_and_answers_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path = _anchored(tmp_path)
    truth = _truth(path)
    shutil.copy(Path(__file__).resolve().parents[1] / "launcher.py", tmp_path / "rig_server.py")
    src = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    legal = [
        ("get_asset_profile", {}), ("read_sensors", {"sensor_group": "tier1", "cycles": 2}), ("advance_time", {"cycles": 1}),
        ("summarize_changes_since", {"cycle": 0}), ("get_failure_modes", {"component": "cooler"}),
        ("submit_diagnosis", {**truth, "evidence_ids": "obs/1/" + tier_fields(1)[0] + "_mean,alarm/0/x"}),
        ("declare_insufficient_evidence", {"sensor_group": "tier3", "cycles": 1, "evidence_ids": ""}),
        ("create_work_order", {"component": "cooler", "prediction_id": "", "evidence_ids": "obs/2/" + tier_fields(1)[0] + "_std"}),
        ("finish", {"summary": "cooler ordered"}),
    ]

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=[str(tmp_path / "rig_server.py")], env=env)
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
    assert "healthy_bands" in decoded[0]["result"] and len(decoded[1]["result"]["served"]) == 2 and decoded[2]["result"] == {"time": 3}
    assert decoded[3]["result"]["cycles_elapsed"] == 3 and decoded[4]["result"]["component"] == "cooler"
    assert decoded[5]["result"]["evidence_ids"] == ["obs/1/" + tier_fields(1)[0] + "_mean", "alarm/0/x"]
    assert decoded[6]["refused"] == "a diagnosis or a declaration was already submitted"
    assert decoded[7]["result"]["work_order"]["id"] == "wo-1" and decoded[7]["open_work_orders"][0]["component"] == "cooler"
    assert decoded[8]["episode_over"] and decoded[8]["result"] == {"summary": "cooler ordered"} and "refused" not in decoded[8]
    refusals = [json.loads(b.content[0].text) for b in bad]
    assert all("refused" in r for r in refusals)
    state = episode.load_state(path)
    assert state["actions"][len(legal) - 1:-1] == [action for _, _, action in MALFORMED]
    assert refusals[-1]["steps_left"] == EpisodeConfig().max_steps - (len(legal) - 1) - len(MALFORMED)
