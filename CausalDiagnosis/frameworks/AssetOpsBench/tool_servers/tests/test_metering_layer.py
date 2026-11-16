"""The metering layer over the memory store and the stub engine."""
import fcntl
import json
from types import SimpleNamespace

import pytest

import servers.iot.main as iot_server
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig, default_sensor_groups
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, default_facts_path
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound, database_names
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER, tools_of
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import Meter, StateFileSessions, meter_config, rewrite_evidence
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import Producer
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data  # noqa: F401

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import engine_stub
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import body, direct, receipt

NAMES = database_names("emeter0000000")
PRICES = harness.PRICES
CE = "CE_CoolingEfficiency_percent_1Hz"
PS1 = "PS1_Pressure_bar_100Hz"
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}


def make(row, thinks=(), **stub):
    store = MemoryServer()
    meter, engine, producer, sessions = harness.episode(store, NAMES, row, thinks, **stub)
    return meter, engine, store, sessions


@pytest.fixture
def base():
    meter, engine, store, sessions = make("base")
    with bound(store, NAMES):
        yield meter, engine, store, sessions


# ----- the base row -----

async def test_the_base_row_shows_aob_text_byte_for_byte_and_charges_nothing(base):
    meter, engine, store, sessions = base
    arguments = {**ASSET, "sensors": [CE]}
    reply = await meter.call("iot", "history", arguments)
    assert reply.text == await direct(iot_server, "history", arguments)
    entry = sessions.tool_calls[-1]
    assert (entry["price"], entry["summary"]["rows"], entry["summary"]["cycle_span"], entry["cycles"]) == (0, 10, [-10, -1], 0)
    assert set(entry) == {"server", "tool", "arguments", "summary", "price", "time", "cycles", "budget_left", "steps_left",
                          "notifications", "response"}
    read = [e for e in engine.public_events() if e["event"] == "read"][-1]
    assert (read["sensor_group"], len(read["served"]), list(read["served"][0]["values"])) == ("tier1", 10, [f"{CE}_mean"])
    assert engine.time == 0


async def test_free_reads_count_a_step_and_take_the_cycle_clock(base):
    meter, engine, store, sessions = base
    await meter.call("iot", "sites")
    calendar = json.loads((await meter.call("wo", "get_schedule_calendar", {"site_id": "MAIN"})).text)
    assert calendar["date_from"] == "2026-01-01"  # the cycle's day, never the wall clock's
    assert [a["type"] for a, _ in sessions.actions] == ["aob_read", "aob_read"] and engine.steps == 2


async def test_latest_reading_ages_by_the_cycle_clock(base):
    meter, engine, store, sessions = base
    latest = json.loads((await meter.call("iot", "latest_reading", {**ASSET, "sensor": CE})).text)
    assert (latest["timestamp"], latest["age_seconds"]) == (fixture.timestamp(-1), 60.0)


async def test_hidden_tools_are_neither_listed_nor_callable(base):
    meter = base[0]
    assert "sensor_stats" not in [t.name for t in await meter.listed_tools("iot")]
    assert [t.name for t in await meter.listed_tools("fmsr")] == ["get_failure_modes"]
    reply = await meter.call("iot", "sensor_stats", ASSET)
    assert reply.is_error and "the server iot has no tool sensor_stats in this episode" in reply.text


# ----- prices from the priced row on -----

async def test_every_newly_paid_cycle_is_charged_at_the_cheapest_covering_group():
    meter, engine, store, sessions = make("priced", budget=300)
    with bound(store, NAMES):
        tier1 = await meter.call("iot", "history", {**ASSET, "sensors": [CE], "limit": 4})
        tier3 = await meter.call("iot", "history", {**ASSET, "sensors": [PS1, f"{CE}_std"], "limit": 2})
        everything = await meter.call("iot", "latest_reading", ASSET)
    assert [receipt(r)["price"] for r in (tier1, tier3, everything)] == [4, 20, 10]
    assert receipt(everything) == {"price": 10, "budget_left": 266, "steps_left": 57}
    assert [a["sensor_group"] for a, _ in sessions.actions] == ["tier1", "tier3", "tier3"]


async def test_a_cycle_read_again_under_a_paid_group_is_free():
    meter, engine, store, sessions = make("priced", budget=300)
    with bound(store, NAMES):
        first = await meter.call("iot", "history", {**ASSET, "sensors": [CE]})
        again = await meter.call("iot", "history", {**ASSET, "sensors": [CE]})
        wider = await meter.call("iot", "history", {**ASSET, "sensors": [PS1]})
        narrower = await meter.call("iot", "history", {**ASSET, "sensors": [f"{CE}_std"]})
    assert [receipt(r)["price"] for r in (first, again, wider, narrower)] == [10, 0, 100, 0]
    assert (sessions.tool_calls[1]["summary"]["rows_read_again"], sessions.tool_calls[1]["summary"]["new_observations"]) == (10, [])
    # a field seen for the first time under a paid group is free yet released, so the server counts what the agent saw
    released = [o for e in engine.public_events() if e["event"] == "read" for o in e["served"]]
    assert len(released) == 30 and set(released[-1]["values"]) == {f"{CE}_std"}


async def test_a_read_the_budget_cannot_pay_is_refused_and_the_episode_ends():
    meter, engine, store, sessions = make("priced", budget=5)
    with bound(store, NAMES):
        refused = await meter.call("iot", "history", {**ASSET, "sensors": [CE]})
        after = await meter.call("iot", "sites")
    assert refused.is_error and body(refused) == {"error": "budget"} and engine.done
    assert body(after) == {"error": episode.EPISODE_OVER} and engine.steps == 1
    assert sessions.tool_calls[0]["refused"] == "budget"


async def test_argument_errors_and_other_assets_are_free_reads(base):
    meter, engine, store, sessions = base
    unknown = await meter.call("iot", "history", {**ASSET, "sensors": ["nothing"]})
    child = await meter.call("iot", "history", {"site_name": "MAIN", "asset_id": "hyd_1_cooler"})
    assert "unknown sensors" in json.loads(unknown.text)["error"]
    assert json.loads(child.text)["returned"] == 0
    assert [a["type"] for a, _ in sessions.actions] == ["aob_read", "aob_read"]


# ----- the clock rows -----

async def test_the_thinking_time_records_cycles_before_the_read():
    meter, engine, store, sessions = make("real_time", thinks=[3, 0], budget=300)
    with bound(store, NAMES):
        reply = await meter.call("iot", "history", {**ASSET, "sensors": [CE]})
        latest = await meter.call("iot", "latest_reading", {**ASSET, "sensor": CE})
    assert (body(reply)["returned"], receipt(reply)["cycle"], receipt(reply)["price"]) == (13, 3, 13)
    assert (body(latest)["timestamp"], body(latest)["age_seconds"]) == (fixture.timestamp(2), 60.0)


async def test_row_three_plays_the_cycles_a_window_reaches_and_row_two_does_not():
    window = {**ASSET, "sensors": [CE], "start": fixture.timestamp(-2), "end": fixture.timestamp(5)}
    played = []
    for row, duration in (("advanced_real_time", 1), ("real_time", 0)):
        meter, engine, store, sessions = make(row, budget=300, read_duration=duration)
        with bound(store, NAMES):
            reply = await meter.call("iot", "history", window)
        played.append(([o["timestamp"] for o in body(reply)["observations"]], receipt(reply)["cycle"], receipt(reply)["price"]))
    assert played[0] == ([fixture.timestamp(c) for c in range(-2, 5)], 5, 7)
    assert played[1] == ([fixture.timestamp(-2), fixture.timestamp(-1)], 0, 2)


async def test_a_window_never_serves_a_cycle_the_rig_did_not_record():
    meter, engine, store, sessions = make("advanced_real_time", budget=300, read_duration=1, repair_cycles=2)
    engine.step({"type": "submit_diagnosis", "diagnosis": {}})
    with bound(store, NAMES):
        await meter.call("wo", "generate_work_order", {"description": "replace the cooler", "asset_num": "hyd_1_cooler", "site_id": "MAIN"})
        reply = await meter.call("iot", "history", {**ASSET, "sensors": [CE], "start": fixture.timestamp(0), "end": fixture.timestamp(6)})
    # the order stops the rig at cycles 0 and 1 while in repair, so cycles 2 to 5 are recorded and served
    assert [o["timestamp"] for o in body(reply)["observations"]] == [fixture.timestamp(c) for c in range(2, 6)]


# ----- work orders -----

async def test_an_order_before_a_diagnosis_is_refused_and_writes_nothing(base):
    meter, engine, store, sessions = base
    reply = await meter.call("wo", "generate_work_order", {"description": "replace", "asset_num": "hyd_1_cooler", "site_id": "MAIN"})
    assert json.loads(reply.text) == {"error": engine_stub.ORDER_BEFORE_WRITE_REASON}  # the base row shows no receipt
    assert store.all_docs(NAMES.workorder) == []


async def test_an_order_reaches_the_engine_with_rewritten_evidence_and_the_engine_id(base):
    meter, engine, store, sessions = base
    await meter.call("iot", "history", {**ASSET, "sensors": [CE]})
    engine.step({"type": "submit_diagnosis", "diagnosis": {}})
    cited = f"obs/{fixture.timestamp(-3)}/{CE}"
    reply = await meter.call("wo", "generate_work_order", {"description": "replace the cooler", "asset_num": "hyd_1_cooler", "site_id": "MAIN",
                                                           "wonum": "agent-pick", "aob_source": json.dumps({"evidence_ids": [cited], "prediction_id": "p1"})})
    action = sessions.actions[-1][0]
    assert (action["component"], action["mode"], action["evidence_ids"], action["prediction_id"]) == ("cooler", "standard", [f"obs/8/{CE}_mean"], "p1")
    created = json.loads(reply.text)["work_order"]
    assert (created["wonum"], created["reportdate"], created["assetnum"]) == ("wo-1", "2026-01-01T00:00:00+00:00", "hyd_1_cooler")
    assert [d["_id"] for d in store.all_docs(NAMES.workorder)] == ["wo:MAIN:wo-1"]


async def test_an_order_the_database_would_refuse_writes_nothing_and_costs_nothing(base):
    meter, engine, store, sessions = base
    engine.step({"type": "submit_diagnosis", "diagnosis": {}})
    reply = await meter.call("wo", "generate_work_order", {"description": "x", "asset_num": "hyd_1_cooler", "site_id": "MAIN",
                                                           "aob_source": {"agent": "causal"}})
    assert reply.is_error and "HTTP 403" in reply.text
    assert sessions.actions[-1][0]["type"] == "aob_read" and store.all_docs(NAMES.workorder) == []


async def test_without_a_clock_an_order_closes_inside_the_order_call(base):
    meter, engine, store, sessions = base
    engine.step({"type": "submit_diagnosis", "diagnosis": {}})
    await meter.call("wo", "generate_work_order", {"description": "replace the pump", "asset_num": "hyd_1_pump", "site_id": "MAIN"})
    assert [d["status"] for d in store.all_docs(NAMES.workorder)] == ["COMP"]  # the order closes inside its call on a row without a clock, the agent need not wait
    assert json.loads((await meter.call("wo", "list_workorders", {"site_id": "MAIN", "status": "OPEN"})).text)["total"] == 0


def test_unmapped_identifiers_pass_unchanged():
    config = make("base")[0].config
    observations = [{"n": 1, "t": -1, "values": {f"{CE}_mean": 1.0}}]
    assert rewrite_evidence([f"obs/{fixture.timestamp(-1)}/{CE}", f"obs/{fixture.timestamp(-2)}/{CE}", "wo/wo-1", "obs/1/x"],
                            observations, config) == [f"obs/1/{CE}_mean", f"obs/{fixture.timestamp(-2)}/{CE}", "wo/wo-1", "obs/1/x"]


# ----- the state file -----

async def test_state_file_sessions_lock_replay_and_save(tmp_path):
    state_path = tmp_path / "state.json"
    # the stub's configuration beside every other field of a new state of the action server, the call logs and the thinking
    # clock that its step keeps (episode.new_state), and the task handed over before the first call, as the driver hands it over
    fields = episode.new_state(0, EpisodeConfig(), default_facts_path(), 0.0, 0.0)
    state_path.write_text(json.dumps({**{key: value for key, value in fields.items() if key not in ("seed", "config", "facts_path")},
                                      "stub_config": {}}))
    episode.hand_over(state_path)
    held = []

    def replay(state):
        with open(tmp_path / "state.json.lock", "a") as other:  # a second process would wait here
            try:
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(True)
        return engine_stub.replay(state)

    store = MemoryServer()
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, "hyd_1", fixture.SITE)
    producer.reset(engine_stub.replay(json.loads(state_path.read_text())))
    config = SimpleNamespace(asset_id="hyd_1", components=COMPONENTS, sensor_groups=default_sensor_groups(), prices={"read": {g: 0 for g in PRICES}})
    sessions = StateFileSessions(state_path, replay=replay, write_released=lambda path, engine: None, wall_clock=lambda: 100.0)
    meter = Meter(meter_config("base", config, fixture.TIMELINE, task_tools=tools_of(TASK_SERVER, "base")), sessions, producer)
    with bound(store, NAMES):
        await meter.call("iot", "sites")
        await meter.call("iot", "history", {**ASSET, "sensors": [CE]})
    state = json.loads(state_path.read_text())
    assert held == [True, True]
    assert [a["type"] for a in state["actions"]] == ["aob_read", "read_recorded"] and state["think_cycles"] == [0, 0]
    assert state["think_seconds"] == [0.0, 0.0]  # the stub's rows without a clock charge no thinking time
    assert [c["tool"] for c in state["tool_calls"]] == ["sites", "history"] and state["last_call_wall_time"] == 100.0


def test_the_meter_takes_the_site_the_class_keys_and_the_statistics_from_the_asset_file(tiny):
    """The layer reads the site, the fmsr class key of every component and the served statistics from the asset file of the
    episode, and a site written in another case names the same site."""
    config = meter_config("base", tiny.default_config(), fixture.TIMELINE, task_tools=())
    assert (config.site, config.classes, config.statistics) == ("TEST", {"fan": "fan", "belt": "belt", "bearing": "bearing"}, ("mean", "std"))
    assert config.component_of("tiny_1_belt", "test") == config.component_of("tiny_1_belt", "TEST") == "belt"
    assert config.component_of("tiny_1_belt", "MAIN") is None
    assert config.engine_field("T1_Temperature_C_1Hz") == "T1_Temperature_C_1Hz_mean" and config.engine_field("T1_Temperature_C_1Hz_std") == "T1_Temperature_C_1Hz_std"
