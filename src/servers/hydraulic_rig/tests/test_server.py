"""The tool interface of tiny world (servers.hydraulic_rig): field names, state and replay, responses, the briefing, and the
MCP server over stdio when the mcp package is installed (mcp 1.26.0, pinned in the fork's uv.lock)."""
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from servers.hydraulic_rig import episode
from servers.hydraulic_rig.briefing import briefing
from servers.hydraulic_rig.names import HYD1_FIELDS, public_name, tier_fields
from servers.hydraulic_rig.policy_client import PublicClassifiers, play, tool_call
from envs.hydraulic_rig.classifiers import TierClassifiers
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, default_facts_path, tier_features
from envs.hydraulic_rig.env import DEFAULT_PRICES, TinyHydraulicEnv
from envs.hydraulic_rig.greedy import GreedyPolicy
from envs.hydraulic_rig.policies import run_episode

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

HYD1_FILE = Path(__file__).resolve().parents[3] / "couchdb" / "scenarios_data" / "shared" / "iot" / "hydraulic_pump_1.json"
STATISTICS = ("mean", "std", "min", "max")
TOOLS = {"measure", "submit", "submit_insufficient_evidence", "replace", "finish"}
needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp package (pinned in the fork's uv.lock)")
MALFORMED = [  # (tool, arguments, the action env.step receives): wrong types arrive as sent, missing arguments as None
    ("measure", {"tier": "two"}, {"type": "measure", "tier": "two"}),
    ("measure", {"tier": "[2]"}, {"type": "measure", "tier": "[2]"}),
    ("measure", {}, {"type": "measure", "tier": None}),
    ("submit", {"cooler": 100, "valve": 100, "pump": 0}, {"type": "submit", "diagnosis": {"cooler": 100, "valve": 100, "pump": 0, "accumulator": None}}),
    ("replace", {"component": 5}, {"type": "replace", "component": 5}),
]


def _new_episode(tmp_path, seed=3, **overrides):
    kwargs = episode.default_env_kwargs()
    kwargs.update(overrides)
    path = tmp_path / "episode.json"
    episode.save_state(path, episode.new_state(seed, kwargs, default_facts_path()))
    return path


def _call(path, tool, arguments, action):
    return json.loads(episode.call(path, tool, arguments, action))


def _truth(path):
    return dict(zip(COMPONENTS, episode.replay(episode.load_state(path)).initial))


def test_every_served_feature_maps_to_a_distinct_hyd1_name():
    cycles, _ = episode.load_world(str(default_facts_path()))
    names = list(tier_features(cycles[0], 3))
    assert len({public_name(n) for n in names}) == len(names) == 68
    assert public_name("MotorPower_EPS1_mean") == "P1_MotorPower_W_100Hz_mean"
    assert public_name("Pressure_PS1_std") == "PS1_Pressure_bar_100Hz_std"
    assert set(HYD1_FIELDS) == set(SENSORS)
    assert all(tier_fields(t) == [HYD1_FIELDS[s] for s in TIER_SENSORS[t]] for t in TIER_SENSORS)


@pytest.mark.skipif(not HYD1_FILE.exists(), reason="AssetOpsBench's hyd_1 file is not checked out")
def test_hyd1_field_names_are_the_asset_s_own_columns():
    columns = set(json.loads(HYD1_FILE.read_text())[0])
    assert set(HYD1_FIELDS.values()) <= columns


def test_environment_parameters_survive_the_json_round_trip():
    kwargs = episode.default_env_kwargs()
    kwargs["allowed"] = {(20, 100, 0, 130), (100, 90, 0, 130)}
    assert episode.decode_params(json.loads(json.dumps(episode.encode_params(kwargs)))) == kwargs


def test_every_call_reaches_env_step_and_replay_reproduces_the_log(tmp_path):
    path = _new_episode(tmp_path)
    truth = _truth(path)
    faulty = [c for c in COMPONENTS if truth[c] != HEALTHY[c]]
    actions = [
        ("measure", {"tier": 1}, {"type": "measure", "tier": 1}),
        ("measure", {"tier": 4}, {"type": "measure", "tier": 4}),
        ("replace", {"component": faulty[0]}, {"type": "replace", "component": faulty[0]}),
        ("submit", dict(truth, pump=7), {"type": "submit", "diagnosis": dict(truth, pump=7)}),
        ("submit", truth, {"type": "submit", "diagnosis": truth}),
        ("measure", {"tier": 3}, {"type": "measure", "tier": 3}),
    ] + [("replace", {"component": c}, {"type": "replace", "component": c}) for c in faulty]
    direct = TinyHydraulicEnv(*episode.load_world(str(default_facts_path())), **episode.default_env_kwargs())
    direct.reset(3)
    for tool, arguments, action in actions:
        _call(path, tool, arguments, action)
        direct.step(action)
    state = episode.load_state(path)
    assert len(state["actions"]) == len(actions) == len(state["tool_calls"])
    assert episode.replay(state).log == direct.log
    assert direct.done and direct.log[-1]["reason"] == "recovered"


def test_refusals_come_back_as_text_cost_nothing_and_count_as_steps(tmp_path):
    path = _new_episode(tmp_path)
    truth = _truth(path)
    for tool, arguments, action, reason in [
        ("measure", {"tier": 4}, {"type": "measure", "tier": 4}, "unknown tier"),
        ("replace", {"component": "pump"}, {"type": "replace", "component": "pump"}, "diagnosis must be submitted before the first repair"),
        ("finish", {}, {"type": "finish"}, "finish is allowed after the diagnosis"),
    ]:
        response = _call(path, tool, arguments, action)
        assert response["event"] == {"event": "refused", "action": action, "reason": reason}
        assert response["budget_left"] == 150 and response["cost"] == 0 and not response["episode_over"]
    _call(path, "submit", truth, {"type": "submit", "diagnosis": truth})
    response = _call(path, "replace", {"component": "motor"}, {"type": "replace", "component": "motor"})
    assert response["event"]["reason"] == "unknown component" and response["budget_left"] == 150
    assert response["steps_left"] == 200 - 5


def test_invalid_submissions_are_refused_and_the_phase_stays_open(tmp_path):
    path = _new_episode(tmp_path)
    truth = _truth(path)
    bad = dict(truth, accumulator=50)
    response = _call(path, "submit", bad, {"type": "submit", "diagnosis": bad})
    assert response["event"]["reason"] == "diagnosis level not in the component's levels" and response["phase"] == "investigate"
    response = _call(path, "submit_insufficient_evidence", {"next_action": ""}, {"type": "submit", "insufficient_evidence": True, "next_action": ""})
    assert response["event"]["reason"] == "insufficient_evidence needs next_action"
    response = _call(path, "submit_insufficient_evidence", {"next_action": "measure(3)"}, {"type": "submit", "insufficient_evidence": True, "next_action": "measure(3)"})
    assert response["event"] == {"event": "submit", "diagnosis": None, "insufficient_evidence": True, "next_action": "measure(3)"}
    assert response["phase"] == "repair"


def test_calls_after_the_end_change_nothing_and_are_recorded(tmp_path):
    path = _new_episode(tmp_path)
    truth = _truth(path)
    _call(path, "submit", truth, {"type": "submit", "diagnosis": truth})
    end = _call(path, "finish", {}, {"type": "finish"})
    assert end["event"] is None and end["episode_over"] and end["cost"] == 0
    after = _call(path, "measure", {"tier": 1}, {"type": "measure", "tier": 1})
    assert after["episode_over"] and "note" in after
    assert set(end) == {"event", "cost", "phase", "budget_left", "repairs_left", "steps_left", "episode_over"}
    state = episode.load_state(path)
    assert len(state["actions"]) == 2 and len(state["tool_calls"]) == 3


def test_measurements_use_hyd1_names_and_carry_no_cycle_numbers(tmp_path):
    path = _new_episode(tmp_path)
    for tier in TIER_SENSORS:
        response = _call(path, "measure", {"tier": tier}, {"type": "measure", "tier": tier})
        event = response["event"]
        assert response["cost"] == event["cost"] == DEFAULT_PRICES["measure"][tier]
        assert set(event["result"]) == {f"{f}_{s}" for f in tier_fields(tier) for s in STATISTICS}
        assert set(event) == {"event", "tier", "cost", "result", "repeated"}


def test_briefing_states_every_legal_value_and_the_public_rules(tmp_path):
    obs = episode.replay(episode.load_state(_new_episode(tmp_path))).observation()
    text = briefing(obs)
    assert all(c in text and all(str(v) in text for v in LEVELS[c]) for c in COMPONENTS)
    assert all(f in text for t in TIER_SENSORS for f in tier_fields(t))
    assert all(name in text for name in TOOLS)
    assert f"between {obs['min_faulty']} and {obs['max_faulty']} components" in text and f"Budget: {obs['budget_left']}." in text
    assert f"after {obs['steps_left']} steps" in text and f"replace up to {obs['repairs_left']} components" in text


def test_the_state_file_holds_what_replay_needs_and_nothing_else(tmp_path):
    path = _new_episode(tmp_path)
    assert set(episode.load_state(path)) == {"format", "seed", "env_params", "facts_path", "actions", "tool_calls"}


def test_state_writes_leave_no_temporary_files(tmp_path):
    path = _new_episode(tmp_path)
    _call(path, "measure", {"tier": 2}, {"type": "measure", "tier": 2})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.json", "released.json"]


def test_every_action_has_one_tool_call():
    diagnosis = dict(HEALTHY)
    assert tool_call({"type": "measure", "tier": 2}) == ("measure", {"tier": 2})
    assert tool_call({"type": "submit", "diagnosis": diagnosis}) == ("submit", diagnosis)
    assert tool_call({"type": "submit", "insufficient_evidence": True, "next_action": "measure(3)"}) == (
        "submit_insufficient_evidence", {"next_action": "measure(3)"})
    assert tool_call({"type": "replace", "component": "pump"}) == ("replace", {"component": "pump"})
    assert tool_call({"type": "finish"}) == ("finish", {})
    with pytest.raises(ValueError):
        tool_call({"type": "wait"})


def test_greedy_policy_on_tool_answers_takes_its_in_process_actions(tmp_path):
    cycles, blocks = episode.load_world(str(default_facts_path()))
    classifiers = TierClassifiers(cycles, blocks)
    public = PublicClassifiers(classifiers)
    for use_max_faulty in (False, True):
        for seed in range(5):
            env = TinyHydraulicEnv(cycles, blocks, **episode.default_env_kwargs())
            in_process = run_episode(env, GreedyPolicy(classifiers, use_max_faulty=use_max_faulty), seed)
            ep_dir = tmp_path / f"{use_max_faulty}_{seed}"
            ep_dir.mkdir()
            path = _new_episode(ep_dir, seed=seed)

            def send(action, path=path):
                tool, arguments = tool_call(action)
                return episode.call(path, tool, arguments, action)

            play(GreedyPolicy(public, use_max_faulty=use_max_faulty), episode.replay(episode.load_state(path)).observation(), send)
            assert episode.replay(episode.load_state(path)).log == in_process


@needs_mcp
def test_server_offers_five_flat_tools_and_answers_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path = _new_episode(tmp_path)
    shutil.copy(Path(__file__).resolve().parents[1] / "launcher.py", tmp_path / "rig_server.py")
    src = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=[str(tmp_path / "rig_server.py")], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t.inputSchema for t in (await session.list_tools()).tools}
                ok = await session.call_tool("measure", {"tier": 1})
                bad = [await session.call_tool(tool, arguments) for tool, arguments, _ in MALFORMED]
                return tools, ok, bad

    tools, ok, bad = asyncio.run(session_calls())
    assert set(tools) == TOOLS
    assert sorted(tools["submit"]["required"]) == sorted(COMPONENTS)
    assert all(tools["submit"]["properties"][c]["type"] == "integer" for c in COMPONENTS)
    assert tools["finish"]["properties"] == {} and tools["measure"]["properties"]["tier"]["type"] == "integer"
    assert tools["replace"]["properties"]["component"]["type"] == tools["submit_insufficient_evidence"]["properties"]["next_action"]["type"] == "string"
    assert all(schema.get("required", []) == list(schema["properties"]) for schema in tools.values())
    assert not any("default" in p for schema in tools.values() for p in schema["properties"].values())
    assert not ok.isError and json.loads(ok.content[0].text)["event"]["event"] == "measure"
    answers = [json.loads(b.content[0].text) for b in bad]
    assert not any(b.isError for b in bad) and all(a["event"]["event"] == "refused" for a in answers)
    assert [a["event"]["action"] for a in answers] == [action for _, _, action in MALFORMED]
    assert episode.load_state(path)["actions"][1:] == [action for _, _, action in MALFORMED]
    assert answers[-1]["steps_left"] == 200 - 1 - len(MALFORMED) and answers[-1]["budget_left"] == 150 - 1


@pytest.mark.parametrize("low, high, rule", [
    (1, 2, "between 1 and 2 components start off their healthy level"),
    (1, 1, "exactly 1 component starts off its healthy level"),
    (2, 2, "exactly 2 components start off their healthy level"),
])
def test_briefing_states_the_fault_count_rule_of_the_episode(low, high, rule):
    cycles, blocks = episode.load_world(str(default_facts_path()))
    env = TinyHydraulicEnv(cycles, blocks, **{**episode.default_env_kwargs(), "min_faulty": low, "max_faulty": high})
    assert f"condition levels, and {rule}. Find the level" in briefing(env.reset(0))


def _clocked(tmp_path, **changes):
    import copy
    from envs.hydraulic_rig.env import DEFAULT_CLOCK
    clock = copy.deepcopy(DEFAULT_CLOCK)
    clock.update(changes)
    return _new_episode(tmp_path, clock=clock)


def test_a_clocked_episode_reports_the_clock_and_hides_the_schedule(tmp_path):
    path = _clocked(tmp_path, degradation={"events": 1, "min_cycle": 3, "max_cycle": 3})
    first = _call(path, "measure", {"tier": 3}, {"type": "measure", "tier": 3})
    assert first["clock"] == 5 and first["event"]["clock"] == 0
    waited = _call(path, "advance_time", {"cycles": 4}, {"type": "advance_time", "cycles": 4})
    assert waited["event"] == {"event": "advance_time", "cycles": 4, "clock": 5} and waited["clock"] == 9
    released = json.loads(episode.released_path(path).read_text())
    assert released["clock_rules"]["degradation"] is True and released["status"]["clock"] == 9
    assert "schedule" not in json.dumps(released) and "degrade" not in json.dumps(released["events"])
    env = episode.replay(episode.load_state(path))
    assert env.clock == 9 and any(e["event"] == "degrade" for e in env.log)


def test_the_briefing_explains_the_clock_only_when_there_is_one(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "timed").mkdir()
    plain = episode.replay(episode.load_state(_new_episode(tmp_path / "plain"))).observation()
    timed = episode.replay(episode.load_state(_clocked(tmp_path / "timed", fault_cost_per_cycle=2))).observation()
    assert "advance_time" not in briefing(plain)
    text = briefing(timed)
    assert "advance_time(cycles)" in text and "work order" in text and "costs 2" in text and "wear further" in text


def test_advance_time_has_one_tool_call():
    assert tool_call({"type": "advance_time", "cycles": 7}) == ("advance_time", {"cycles": 7})


@needs_mcp
def test_server_offers_advance_time_only_with_a_clock(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    src = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}

    async def tool_names(folder):
        shutil.copy(Path(__file__).resolve().parents[1] / "launcher.py", folder / "rig_server.py")
        params = StdioServerParameters(command=sys.executable, args=[str(folder / "rig_server.py")], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {t.name for t in (await session.list_tools()).tools}

    (tmp_path / "plain").mkdir()
    (tmp_path / "timed").mkdir()
    _new_episode(tmp_path / "plain")
    _clocked(tmp_path / "timed")
    assert asyncio.run(tool_names(tmp_path / "plain")) == TOOLS
    assert asyncio.run(tool_names(tmp_path / "timed")) == TOOLS | {"advance_time"}
