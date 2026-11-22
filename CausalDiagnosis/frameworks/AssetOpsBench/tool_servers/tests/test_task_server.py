"""The task server of the AssetOpsBench rows: the task tools of the rig's action
server with their signatures, run through the metering layer over the engine of the rows on the presets and the memory
store. The profile is the layer's view of its row, a diagnosis cites a row by its timestamp, orders and finish end the
episode, a refusal is an error that counts a step, the clock rows add the two time tools with the cycle's datetime, and
the calls go through the state file like every other."""
import json

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import DIAGNOSIS_LEVEL_REASON
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
from CausalDiagnosis.mcp_servers.utils import profile_levels
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import DEFINITIONS_ONLY
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import DECLARATION_RULE
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server as build_rig_server

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import NAMES, answer, body, receipt
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.test_aob_tool_contract import json_type

CE = "CE_CoolingEfficiency_percent_1Hz"
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}


def truth(engine):
    return dict(zip(COMPONENTS, engine.initial))


@pytest.mark.parametrize("row", layer_contract.ROWS)
async def test_every_row_and_the_action_server_state_one_rule_for_a_declaration(world, row):
    meter, *_ = harness.memory_engine_episode(world, row)
    listed = {t.name: t.description for t in await meter.listed_tools(TASK_SERVER)}
    action_server = {t.name: t.description for t in await build_rig_server(DEFINITIONS_ONLY).list_tools()}
    assert DECLARATION_RULE in listed["declare_insufficient_evidence"] and listed["declare_insufficient_evidence"] == action_server["declare_insufficient_evidence"]


@pytest.mark.parametrize("row", ["base", "real_time"])
async def test_the_task_server_lists_the_rig_definitions_of_the_tools_of_its_row(world, row):
    meter, engine, *_ = harness.memory_engine_episode(world, row)
    listed = await meter.listed_tools(TASK_SERVER)
    entries = layer_contract.entries(TASK_SERVER)
    assert [t.name for t in listed] == list(layer_contract.tools_of(TASK_SERVER, row))
    for tool in listed:
        assert {p: json_type(s) for p, s in tool.inputSchema.get("properties", {}).items()} == entries[tool.name]["parameters"]
        assert tool.annotations.readOnlyHint == (entries[tool.name]["class"] == "read")
    if row == "base":
        # a tool the row does not list is refused by the engine and counted as a step, as the briefing says of refused calls
        steps = engine.steps
        reply = await meter.call(TASK_SERVER, "get_current_time", {})
        assert reply.is_error and f"the server {TASK_SERVER} has no tool get_current_time in this episode" in reply.text
        assert "base row" not in reply.text
        assert engine.steps == steps + 1


async def test_the_repair_rules_row_serves_the_manual_on_the_task_server(world):
    meter, engine, store, sessions = harness.memory_engine_episode(world, "repair_rules")
    assert meter.config.row.name == "repair_rules"
    assert "lookup_fix" in [t.name for t in await meter.listed_tools(TASK_SERVER)]
    manual = body(await meter.call(TASK_SERVER, "lookup_fix", {"component": "pump"}))
    assert manual["component"] == "pump" and manual["dependencies"] == ["valve", "accumulator"] and "emergency" in manual
    entry = sessions.tool_calls[-1]
    assert (entry["server"], entry["tool"], entry["price"]) == (TASK_SERVER, "lookup_fix", 0)


@pytest.mark.parametrize("row", ["base", "priced"])
async def test_the_profile_is_the_layer_view_of_its_row(world, row):
    meter, engine, store, sessions = harness.memory_engine_episode(world, row)
    reply = await meter.call(TASK_SERVER, "get_asset_profile", {"response_format": "raw"})
    profile = answer(reply, row)
    assert "fault_rule" not in profile and profile["asset"]["site"] == "MAIN"
    assert profile["asset"]["children"] == {c: f"hyd_1_{c}" for c in COMPONENTS}
    fields = {f for group in profile["sensor_groups"].values() for f in group}
    assert set(profile["normal_ranges"]) == fields  # under the record field, which carries the channel's mean
    assert profile["tools"] == layer_contract.public_table(engine.config)
    assert not any(t["server"] == contract.ACTION_SERVER for t in profile["tools"])
    entry = sessions.tool_calls[-1]
    assert (entry["server"], entry["tool"], entry["price"], entry["cycles"]) == (TASK_SERVER, "get_asset_profile", 0, 0)
    if row == "priced":
        assert receipt(reply) == {"price": 0, "budget_left": engine.config.budget, "steps_left": engine.config.max_steps - 1}


@pytest.mark.parametrize("row", ["base", "priced"])
async def test_a_diagnosis_cites_a_row_by_its_timestamp_and_orders_then_finish_end_the_episode(world, row):
    meter, engine, store, sessions = harness.memory_engine_episode(world, row)
    faulty = [c for c, level in truth(engine).items() if level != HEALTHY[c]]
    with bound(store, NAMES):
        rows = answer(await meter.call("iot", "history", {**ASSET, "sensors": [CE]}), row)["observations"]
        cited = f"obs/{rows[-1]['timestamp']}/{CE}"
        diagnosis = await meter.call(TASK_SERVER, "submit_diagnosis", {**truth(engine), "evidence_ids": cited})
        for component in faulty:
            await meter.call("wo", "generate_work_order", {"description": f"repair the {component}", "asset_num": f"hyd_1_{component}",
                                                           "site_id": "MAIN"})
        finish = await meter.call(TASK_SERVER, "finish", {"summary": "diagnosed and repaired"})
        orders = store.all_docs(NAMES.workorder)
    served = next(o for e in engine.public_events() if e["event"] == "read" for o in e["served"] if o["t"] == -1)
    expected = {"diagnosis": truth(engine), "evidence_ids": [f"obs/{served['n']}/{CE}_mean"]}
    assert answer(diagnosis, row) == expected and next(e for e in engine.log if e["event"] == "submit_diagnosis")["evidence_ids"] == expected["evidence_ids"]
    assert [o["status"] for o in orders] == ["COMP"] * len(faulty)
    assert engine.done and engine.log[-1]["reason"] == "finish" and answer(finish, row) == {"summary": "diagnosed and repaired"}
    assert tuple(engine.hidden) == tuple(HEALTHY[c] for c in COMPONENTS)
    assert [c["server"] for c in sessions.tool_calls] == ["iot", TASK_SERVER, *["wo"] * len(faulty), TASK_SERVER]
    assert sessions.tool_calls[1]["summary"] == expected  # a write logs the engine's result


@pytest.mark.parametrize("row", ["base", "real_time"])
async def test_a_declaration_that_cites_a_row_by_its_timestamp_cites_a_reading_for_the_verifier(world, row):
    """The form in which the agents of the five rows read and cite a value reaches the verifier as a reading released before
    the declaration, and a declaration that cites nothing does not."""
    verdicts = {}
    for cites in (True, False):
        meter, engine, store, _sessions = harness.memory_engine_episode(world, row)
        with bound(store, NAMES):
            rows_read = answer(await meter.call("iot", "history", {**ASSET, "sensors": [CE]}), row)["observations"]
            cited = f"obs/{rows_read[-1]['timestamp']}/{CE}" if cites else ""
            await meter.call(TASK_SERVER, "declare_insufficient_evidence", {"sensor_group": "tier1", "cycles": 1, "evidence_ids": cited})
            await meter.call(TASK_SERVER, "finish", {"summary": "insufficient evidence"})
        verdicts[cites] = verify(engine.log, released_from_env(engine), engine.oracle(), 1.0, grade={"separable": False, "e18": {"separable": False}})
    assert verdicts[True]["checks"]["provenance"]["ok"] and verdicts[True]["abstention_cites_reading"] and verdicts[True]["abstention_correct"]
    assert not verdicts[False]["abstention_cites_reading"] and not verdicts[False]["success"]
    assert verdicts[False]["failure_class"] == "abstention_without_reading"


async def test_a_refused_task_call_is_an_error_with_the_engine_reason_that_counts_a_step(world):
    meter, engine, store, sessions = harness.memory_engine_episode(world, "priced")
    reply = await meter.call(TASK_SERVER, "submit_diagnosis", {**{c: -5 for c in COMPONENTS}, "evidence_ids": ""})
    reason = DIAGNOSIS_LEVEL_REASON
    assert reply.is_error and body(reply) == {"error": reason} and sessions.tool_calls[-1]["refused"] == reason
    assert receipt(reply)["steps_left"] == engine.config.max_steps - 1 == engine.config.max_steps - engine.steps


async def test_the_clock_rows_add_the_time_tools_with_the_cycle_datetime_and_record_the_cycles_they_play(world):
    meter, engine, store, sessions = harness.memory_engine_episode(world, "real_time", thinks=[2, 0])
    now = await meter.call(TASK_SERVER, "get_current_time", {})
    result = body(now)
    assert result["time"] == receipt(now)["cycle"] == 2 and result["current_datetime"] == fixture.TIMELINE.timestamp(2)
    assert (result["horizon"], result["seconds_per_cycle"], result["budget_left"]) == (engine.config.horizon, engine.config.seconds_per_cycle,
                                                                                      engine.config.budget)
    waited = await meter.call(TASK_SERVER, "wait_for_notification", {"timeout": 3})
    assert body(waited) == {"time": 5, "waited": 3, "ended_by": "timeout"}
    assert any(n.get("event") == "wait_timeout" for n in receipt(waited)["notifications"])
    records = store.all_docs(NAMES.iot)
    assert len(records) == len(engine.recorded_cycles()) and engine.recorded_cycles()[-1] == 4


async def test_task_calls_go_through_the_state_file_and_the_next_process_replays_them(world, tmp_path):
    state_path, store = tmp_path / "state.json", MemoryServer()
    meter = harness.state_file_episode(state_path, "priced", store)
    started = json.loads(episode.released_path(state_path).read_text(encoding="utf-8"))
    engine = rows.replay(episode.load_state(state_path))
    assert started["task_text"] == briefing(engine.public_profile(), engine.config)  # the question of the rows, not the rig's
    assert started["rules"]["tools"] == layer_contract.public_table(engine.config)
    diagnosis = truth(engine)
    first = await meter.call(TASK_SERVER, "submit_diagnosis", {**diagnosis, "evidence_ids": ""})
    again = await meter.call(TASK_SERVER, "submit_diagnosis", {**diagnosis, "evidence_ids": ""})
    assert not first.is_error and again.is_error  # the second call replays the first diagnosis, so the write lock holds
    state = episode.load_state(state_path)
    assert [a["type"] for a in state["actions"]] == ["submit_diagnosis"] * 2
    assert [(c["server"], c["tool"]) for c in state["tool_calls"]] == [(TASK_SERVER, "submit_diagnosis")] * 2
    released = json.loads(episode.released_path(state_path).read_text(encoding="utf-8"))
    assert released["status"]["steps_left"] == released["rules"]["max_steps"] - 2 and released["task_text"] == started["task_text"]


@pytest.mark.parametrize("row", ["base", "real_time"])
async def test_the_task_server_reads_a_quoted_number_on_every_row_as_assetopsbench_tools_do(world, row):
    """Behind the metering layer the task tools read an integer written as a string by its value, as the validation of
    AssetOpsBench's own tools reads it: the history of the iot server takes the limit "3" as 3, the diagnosis with the levels
    as strings is accepted and logged with integers, and on a row with a clock the wait with the timeout "3" waits three
    cycles."""
    meter, engine, store, sessions = harness.memory_engine_episode(world, row)
    with bound(store, NAMES):
        if engine.config.has_clock():  # before any read, so that no alarm of a paid group ends the wait early
            start = engine.time
            waited = await meter.call(TASK_SERVER, "wait_for_notification", {"timeout": "3"})
            assert not waited.is_error and body(waited)["waited"] == 3 and engine.time == start + 3
        history = answer(await meter.call("iot", "history", {**ASSET, "sensors": [CE], "limit": "3"}), row)
        assert len(history["observations"]) == 3
        reply = await meter.call(TASK_SERVER, "submit_diagnosis", {**{c: str(level) for c, level in truth(engine).items()}, "evidence_ids": ""})
        assert not reply.is_error and answer(reply, row)["diagnosis"] == truth(engine)
        assert next(e for e in engine.log if e["event"] == "submit_diagnosis")["diagnosis"] == truth(engine)


@pytest.mark.parametrize("orders", (0, 1))
async def test_all_healthy_and_no_work_order_is_a_valid_and_scored_answer_through_the_tools(world, orders):
    """On the base row with the fault rule of the control, behind the metering layer, the diagnosis that names every component at
    its healthy level and the finish are accepted, and the verdict passes with no false alarm and no unnecessary order. A work
    order of the wo server before the finish is executed, counted as unnecessary, and fails the pass."""
    meter, engine, store, sessions = harness.memory_engine_episode(world, "base", fault_rule={"min_faulty": 0, "max_faulty": 0})
    healthy = {c: engine.config.healthy[c] for c in engine.config.components}
    assert tuple(engine.initial) == tuple(healthy.values())
    with bound(store, NAMES):
        replies = [await meter.call(TASK_SERVER, "submit_diagnosis", {**healthy, "evidence_ids": ""})]
        for component in engine.config.components[:orders]:
            replies.append(await meter.call("wo", "generate_work_order", {"description": f"repair the {component}",
                                                                           "asset_num": f"hyd_1_{component}", "site_id": "MAIN"}))
        replies.append(await meter.call(TASK_SERVER, "finish", {"summary": "every component is healthy"}))
    assert not [r for r in replies if r.is_error] and engine.done
    verdict = verify(engine.log, released_from_env(engine), engine.oracle(), 1.0)
    assert verdict["verified"] and verdict["false_alarms"] == 0 and verdict["unnecessary"] == orders
    assert verdict["pass"] == (orders == 0)

def _raw_of(shorter):
    """The raw profile rebuilt from a shorter one by the rule of profile_levels.py: the asset with the profile's asset_id, and
    every long table back to one entry per field or per tool."""
    rows = lambda table: [{column: value for column, value in zip(table["columns"], row) if value is not None} for row in table["rows"]]
    view = dict(shorter)
    view["asset"] = {"asset_id": shorter["asset_id"], **shorter["asset"]}
    view["normal_ranges"] = {entry.pop("field"): entry for entry in rows(shorter["normal_ranges"])}
    view["tools"] = rows(shorter["tools"])
    return view


@pytest.mark.parametrize("row", ["base", "priced"])
async def test_the_default_profile_keeps_every_rule_and_value_of_the_raw_profile_in_one_line(world, row):
    """concise, the default, and detailed hold every key and value of raw, since the sub-agents read the rules from the profile
    alone: only the asset named twice and the long tables of the normal ranges and the tools are written once, as one line of
    JSON before the receipt; headline is one line; a level of another name is an error answer of a counted step. The control:
    a level that drops a rule, such as the clock or a tool, or rounds a normal range fails."""
    meter, engine, *_ = harness.memory_engine_episode(world, row)
    raw = answer(await meter.call(TASK_SERVER, "get_asset_profile", {"response_format": "raw"}), row)
    for level in ("concise", "detailed", ""):
        reply = await meter.call(TASK_SERVER, "get_asset_profile", {"response_format": level} if level else {})
        text = reply.text.split("\n")[0]
        assert not reply.is_error and json.loads(text) == answer(reply, row) and _raw_of(answer(reply, row)) == raw == profile_levels.raw_of(answer(reply, row))
        assert "asset_id" not in answer(reply, row)["asset"] and answer(reply, row)["normal_ranges"]["columns"] == ["field", "low", "high"]
    headline = answer(await meter.call(TASK_SERVER, "get_asset_profile", {"response_format": "headline"}), row)
    assert set(headline) == {"headline"} and "\n" not in headline["headline"] and raw["asset_id"] in headline["headline"]
    steps = engine.steps
    wrong = await meter.call(TASK_SERVER, "get_asset_profile", {"response_format": "compact"})
    assert wrong.is_error and "response_format must be one of headline, concise, detailed, raw" in wrong.text and engine.steps == steps + 1


async def test_the_profile_lists_the_level_that_the_layer_adds_to_the_rig_signature(world):
    """get_asset_profile of the task server takes response_format, which its description names in one line, while the rig's
    action server keeps its own signature. The control: the rig's definition served unchanged lacks the parameter."""
    meter, *_ = harness.memory_engine_episode(world, "base")
    listed = {t.name: t for t in await meter.listed_tools(TASK_SERVER)}["get_asset_profile"]
    rig = {t.name: t for t in await build_rig_server(DEFINITIONS_ONLY).list_tools()}["get_asset_profile"]
    assert listed.inputSchema["properties"]["response_format"] == {"type": "string", "default": "concise"}
    assert "response_format" not in (rig.inputSchema.get("properties") or {}) and listed.description == rig.description + profile_levels.LEVELS_LINE
