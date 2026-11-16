"""The no_call entry of the AssetOpsBench rows: Meter.no_call steps the engine through the meter
as the driver's idle hook binds it. It counts one step of the step cap, logs the event no_call apart from the refusals, passes no
cycles of the harness, records its call, and the engine ends the episode at the step cap as on the rig."""
import json

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import NO_CALL_ACTION, HydraulicRigEnv
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness


async def test_a_no_call_takes_one_step_logs_no_call_and_no_refusal(world):
    meter, engine, _store, sessions = harness.memory_engine_episode(world, "priced")
    steps, time = engine.steps, engine.time
    assert await meter.no_call() is False
    assert engine.steps == steps + 1 and engine.time == time  # a step of no duration on a row without a clock
    assert engine.log[-1]["event"] == NO_CALL_ACTION and not any(e["event"] == "refused" for e in engine.log)
    assert sessions.actions == [({"type": NO_CALL_ACTION}, 0)]  # the harness hands over no cycles
    call = sessions.tool_calls[-1]
    assert (call["server"], call["tool"], call["price"], call["cycles"]) == (TASK_SERVER, NO_CALL_ACTION, 0, 0) and "refused" not in call
    assert call["steps_left"] == engine.config.max_steps - engine.steps


async def test_no_calls_end_the_episode_at_the_step_cap_as_on_the_rig(world):
    meter, engine, _store, sessions = harness.memory_engine_episode(world, "priced", max_steps=2)
    assert await meter.no_call() is False
    assert await meter.no_call() is True
    assert engine.done and engine.log[-1]["reason"] == "max_steps"
    rig = HydraulicRigEnv(*world, EpisodeConfig(max_steps=2))  # the rig's own environment on the same recordings
    rig.reset(7)
    rig.step({"type": NO_CALL_ACTION})
    _envelope, _cost, done, info = rig.step({"type": NO_CALL_ACTION})
    assert done and info["reason"] == engine.log[-1]["reason"]
    # a no_call after the end is answered as a refusal and recorded, as a call after the end is on the rig (episode.call)
    assert await meter.no_call() is True
    assert sessions.tool_calls[-1]["refused"] == episode.EPISODE_OVER and engine.steps == 2


async def test_a_no_call_through_the_state_file_is_recorded_as_a_stepped_call_of_the_task_server(tmp_path):
    state_path = tmp_path / "state.json"
    meter = harness.state_file_episode(state_path, "priced", MemoryServer())
    assert await meter.no_call() is False
    state = episode.load_state(state_path)
    assert [a["type"] for a in state["actions"]] == [NO_CALL_ACTION]
    assert [(c["server"], c["tool"]) for c in state["tool_calls"]] == [(TASK_SERVER, NO_CALL_ACTION)]
    lines = transcript.read(tmp_path)
    kinds = [(line["kind"], line.get("tool")) for line in lines[1:] if line["kind"] in ("enter", "call", "exit")]
    assert kinds == [("enter", NO_CALL_ACTION), ("call", NO_CALL_ACTION), ("exit", None)]
    # an exit line names its entry and the outcome of the call, not the tool
    entered, left = (next(line for line in lines if line["kind"] == kind) for kind in ("enter", "exit"))
    assert (left["server"], left["enter_n"], left["outcome"]) == (TASK_SERVER, entered["n"], "answered")
    call = next(line for line in lines if line["kind"] == "call")
    assert call["server"] == TASK_SERVER and call["stepped"] is True
    released = json.loads(episode.released_path(state_path).read_text(encoding="utf-8"))
    assert released["status"]["steps_left"] == released["rules"]["max_steps"] - 1
