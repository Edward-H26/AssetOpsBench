"""The metering layer over the engine of the AssetOpsBench rows on the presets of the six rows
and the memory store: AssetOpsBench returns exactly the cycles the engine served, with the engine's values, a re-read is
free, an order closes inside its call where no clock runs, time passes only as each row lets it, no live
cycle replays a recording of the prehistory, a return to the start combination continues its live window,
finish on a clock row settles the open orders through the record, and the work order history
carries the fix history card, the past orders a wo read shows logged as the fix history the agent read."""
import json
from datetime import datetime
from functools import partial

import pytest

import servers.iot.main as iot_server
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import BLOCK_CAP, HydraulicRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import HISTORY_CYCLES, RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import ROWS, TASK_SERVER, history_wonum
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import Meter, StateFileSessions, meter_config
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import Producer
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from servers.wo import workorders

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import ProfileDocuments
from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data, tiny_recordings  # noqa: F401
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import conformance
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import NAMES, body, direct, receipt
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import memory_engine_episode as make

CE = "CE_CoolingEfficiency_percent_1Hz"
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}
HISTORY_SEED = 2  # a start whose fix history on the repair rules preset holds past orders on two components
HORIZON_SEED = 4  # a start that runs to the horizon on every clock row, where the strict failure rule ends most starts of the deterioration row early
RETURN_SEED = 3  # a start on the repair rules preset with one degraded component, whose emergency repair expires back to its start


def cycles_of(reply):
    return [round(fixture.TIMELINE.cycle_of(datetime.fromisoformat(o["timestamp"]))) for o in body(reply)["observations"]]


def window(first, end):
    return {**ASSET, "sensors": [CE], "start": fixture.timestamp(first), "end": fixture.timestamp(end)}


async def test_the_base_row_serves_the_prehistory_with_the_engines_values(world):
    meter, engine, store, sessions = make(world, "base")
    arguments = {**ASSET, "sensors": [CE]}
    with bound(store, NAMES):
        reply = await meter.call("iot", "history", arguments)
        assert reply.text == await direct(iot_server, "history", arguments)  # no receipt in the base row
    rows = json.loads(reply.text)["observations"]
    prehistory = engine.recorded_cycles()
    assert [round(fixture.TIMELINE.cycle_of(datetime.fromisoformat(r["timestamp"]))) for r in rows] == prehistory == list(range(-len(rows), 0))
    assert [r[CE] for r in rows] == [engine.record_values(c)[f"{CE}_mean"] for c in prehistory]
    entry = sessions.tool_calls[-1]
    assert (entry["price"], entry["summary"]["newly_paid"], entry["cycles"]) == (0, len(prehistory), 0)


async def test_a_priced_read_charges_its_newly_paid_cycles_and_a_re_read_is_free(world):
    meter, engine, store, sessions = make(world, "priced")
    with bound(store, NAMES):
        first = await meter.call("iot", "history", window(-3, 0))
        again = await meter.call("iot", "history", window(-3, 0))
    price = 3 * engine.config.prices["read"][meter.config.group_of([CE])]
    assert cycles_of(first) == cycles_of(again) == [-3, -2, -1]
    assert (receipt(first)["price"], receipt(again)["price"]) == (price, 0)
    assert receipt(again)["budget_left"] == engine.config.budget - price
    assert sessions.tool_calls[-1]["summary"]["rows_read_again"] == 3


@pytest.mark.parametrize("row", ["base", "priced"])
async def test_without_a_clock_an_order_closes_inside_its_call_and_no_reading_follows(world, row):
    meter, engine, store, sessions = make(world, row)
    component = next(c for c, level in zip(COMPONENTS, engine.initial) if level != HEALTHY[c])
    sessions.step({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENTS, engine.initial)), "evidence_ids": []})
    with bound(store, NAMES):
        reply = await meter.call("wo", "generate_work_order", {"description": "repair", "asset_num": f"hyd_1_{component}", "site_id": "MAIN"})
        orders = store.all_docs(NAMES.workorder)
    assert [(o["wonum"], o["status"]) for o in orders] == [("wo-1", "COMP")]
    assert engine.time == 0 and all(t < 0 for t in engine.recorded_cycles())
    if row == "priced":
        assert receipt(reply)["price"] == engine.config.order_price(component) + engine.config.prices["visit"]


async def test_on_the_real_time_clock_thinking_records_cycles_and_a_window_stays_in_the_present(world):
    meter, engine, store, sessions = make(world, "real_time", thinks=[3, 0])
    with bound(store, NAMES):
        await meter.call("iot", "sites", {})
        reply = await meter.call("iot", "history", window(0, 10))
        records = store.all_docs(NAMES.iot)
    assert engine.time == 3 and cycles_of(reply) == [0, 1, 2] and len(records) == len(engine.recorded_cycles())
    assert receipt(reply)["cycle"] == 3


async def test_on_the_advanced_real_time_clock_a_window_plays_the_cycles_it_reaches(world):
    meter, engine, store, sessions = make(world, "advanced_real_time")
    with bound(store, NAMES):
        reply = await meter.call("iot", "history", window(0, 3))
    assert engine.time == 3 and cycles_of(reply) == [0, 1, 2] and receipt(reply)["cycle"] == 3
    assert receipt(reply)["price"] == 3 * engine.config.prices["read"][meter.config.group_of([CE])]


@pytest.mark.parametrize("row", ROWS)
def test_the_prehistory_and_the_live_cycles_never_share_a_recording(world, row):
    cycles, blocks = world
    config = harness.preset_config(row)
    for seed in range(40):
        env, rig = RecordedRigEnv(cycles, blocks, config), HydraulicRigEnv(cycles, blocks, config)
        env.reset(seed)
        rig.reset(seed)
        history, live = [env._recorded[t][0] for t in env.recorded_cycles()], env.clock._window["cycles"]
        # the same start and the same live segment as the rig's, the prehistory from the other segment of the block
        assert (env.initial, env.window_start, env.entry_offset) == (rig.initial, rig.window_start, rig.entry_offset)
        assert live == rig.clock._window["cycles"]
        assert len(set(history)) == HISTORY_CYCLES and sorted(history + live) == sorted(blocks[env.initial][:BLOCK_CAP])
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(HORIZON_SEED)
    history = {env._recorded[t][0] for t in env.recorded_cycles()}
    env.step({"type": "aob_read", "server": "iot", "tool": "sites"}, think_cycles=config.horizon)
    played = [env._recorded[t] for t in env.recorded_cycles() if t >= 0]
    assert len(played) == env.time == (config.horizon if config.has_clock() else 0)
    assert not history & {index for index, _ in played}


def test_a_return_to_the_start_combination_continues_its_live_window_and_replays_no_prehistory_recording(world):
    """When the emergency repair of the degraded component expires, the rig is back at its start combination,
    and the start window continues its live recordings where the rig left them, the repeats marked once they wrap, so
    no recording of the prehistory comes back on the timeline."""
    cycles, blocks = world
    config = harness.preset_config("repair_rules")
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(RETURN_SEED)
    start, window = tuple(env.initial), env.clock._window
    history = {env._recorded[t][0] for t in env.recorded_cycles()}
    live = [window["cycles"][(window["offset"] + k) % len(window["cycles"])] for k in range(len(window["cycles"]))]
    component = next(c for c, level in zip(COMPONENTS, start) if level != HEALTHY[c])
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENTS, start)), "evidence_ids": []})
    env.step({"type": "create_work_order", "component": component, "mode": "emergency", "prediction_id": "", "evidence_ids": []})
    env.step({"type": "aob_read", "server": "iot", "tool": "sites"}, think_cycles=config.horizon)
    assert [combination == start for _t, combination in env.clock.history] == [True, False, True]
    at_start = [env._recorded[t] for t in env.recorded_cycles() if t >= 0 and env.clock.combination_at(t) == start]
    assert len(at_start) > len(live) + 1  # the rig played past a wrap of the live window after its return
    assert not history & {index for index, _ in at_start}
    assert at_start == [(live[k % len(live)], k >= len(live)) for k in range(len(at_start))]


def test_finish_on_a_clock_row_settles_the_open_order_and_records_the_cycles_of_the_span(world):
    """The finish settle through the engine of the rows: finish runs the clock to the close of the open order, and every
    cycle of that span joins the record except the cycles the repair stops the rig."""
    cycles, blocks = world
    env = RecordedRigEnv(cycles, blocks, harness.preset_config("advanced_real_time"))
    env.reset(HORIZON_SEED)
    component = next(c for c, level in zip(COMPONENTS, env.initial) if level != HEALTHY[c])
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENTS, env.initial)), "evidence_ids": []})
    envelope, *_ = env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})
    close = envelope["result"]["work_order"]["expected_close_t"]
    envelope, _cost, done, _info = env.step({"type": "finish", "summary": "ordered"})
    assert done and envelope["open_work_orders"] == [] and env.time == close > 0
    assert [t for t, _ in env.clock.played] == list(range(close)) and any(index is None for _, index in env.clock.played)
    assert [t for t in env.recorded_cycles() if t >= 0] == [t for t, index in env.clock.played if index is not None]


@pytest.mark.parametrize("row", ["base", "repair_rules"])
async def test_the_work_order_history_lists_exactly_the_fix_history_of_the_engine(world, row):
    """wo's list_workorders carries the fix history card, so the producer writes every past order of the engine's
    history closed at its closing cycle, and the read logs the orders it shows as the rig's get_fix_history does. The base
    row has no history card, so its database holds no order and the read stays a free read."""
    meter, engine, store, sessions = make(world, row, seed=HISTORY_SEED)
    with bound(store, NAMES):
        reply = await meter.call("wo", "list_workorders", {"site_id": "MAIN", "page_size": 0})
    orders = harness.answer(reply, row)["work_orders"]
    stamp = {e["id"]: workorders._iso(fixture.TIMELINE.utc(e["closed_t"])) for e in engine.fix_history}
    expected = [(history_wonum(e["id"]), f"hyd_1_{e['component']}", "CM", "COMP", stamp[e["id"]], stamp[e["id"]]) for e in engine.fix_history]
    assert sorted((o["wonum"], o["assetnum"], o["worktype"], o["status"], o["reportdate"], o["actfinish"]) for o in orders) == sorted(expected)
    assert (len(engine.fix_history) >= 2) is (row == "repair_rules")
    last = engine.log[-1]
    assert (last["event"], last.get("history")) == (("fix_history", engine.fix_history) if row == "repair_rules" else ("aob_read", None))


async def test_a_filtered_work_order_read_releases_only_the_past_orders_it_shows_and_their_wonum_cites_the_engine_order(world):
    """A read that shows one component's past orders releases those alone, and a diagnosis citing wo/<wonum> of a past
    order reaches the engine as its hist/<n>, the identifier the verifier accepts once the history was released."""
    meter, engine, store, sessions = make(world, "repair_rules", seed=HISTORY_SEED)
    first = engine.fix_history[0]
    with bound(store, NAMES):
        await meter.call("wo", "list_workorders", {"site_id": "MAIN", "asset_num": f"hyd_1_{first['component']}"})
        released = engine.log[-1]
        await meter.call(TASK_SERVER, "submit_diagnosis", {**dict(zip(COMPONENTS, engine.initial)),
                                                           "evidence_ids": [f"wo/{history_wonum(first['id'])}", "wo/wo-1"]})
    assert (released["event"], released["history"]) == ("fix_history", [first])
    assert engine.log[-1]["event"] == "submit_diagnosis" and engine.log[-1]["evidence_ids"] == [first["id"], "wo/wo-1"]


async def test_the_conformance_script_runs_on_the_engine_of_the_rows_and_repeats_itself(world):
    on_engine = partial(harness.engine_episode, row="real_time", world=world, budget=1000)
    first, second = await conformance.run(MemoryServer(), on_engine), await conformance.run(MemoryServer(), on_engine)
    assert conformance.compare(first, second) == []
    errors = [text for text, is_error in first["replies"] if is_error]
    assert len(errors) == 1 and "HTTP 403" in errors[0]  # the order the work order database refuses
    assert [d["status"] for d in first["documents"]["workorder"]] == ["COMP", "COMP"]


async def test_the_state_file_replays_the_engine_of_the_rows_between_calls(world, tmp_path):
    state_path, store = tmp_path / "state.json", MemoryServer()
    meter = harness.state_file_episode(state_path, "priced", store)
    config = harness.preset_config("priced")
    with bound(store, NAMES):
        first = await meter.call("iot", "history", window(-3, 0))
        again = await meter.call("iot", "history", window(-3, 0))
    # the second call replays the first from the state file, so its cycles are paid and the re-read is free
    price = 3 * config.prices["read"][meter.config.group_of([CE])]
    assert (receipt(first)["price"], receipt(again)["price"]) == (price, 0)
    state = episode.load_state(state_path)
    assert [a["type"] for a in state["actions"]] == ["read_recorded", "read_recorded"] and len(state["tool_calls"]) == 2
    released = json.loads(episode.released_path(state_path).read_text())
    assert [o["t"] for o in released["observations"]] == [-3, -2, -1]


async def test_the_rig_server_and_the_metering_layer_charge_the_same_cycles_for_the_same_call_seconds(world, tmp_path):
    """The thinking time is one function on both surfaces: the same seconds between calls give the rig's
    action server and the metering layer of the AssetOpsBench rows the same cycles, the same seconds charged and the same
    seconds carried in the state file, and the same clock. The seconds add up from call to call, so six calls at 10 seconds of
    thinking per cycle that take 4, 7.5, 0, 13, 25 and 9.5 seconds charge 0, 1, 0, 1, 2 and 1 cycles and carry 9 seconds."""
    config = harness.preset_config("advanced_real_time", think_seconds_per_cycle=10)
    gaps = [4.0, 7.5, 0.0, 13.0, 25.0, 9.5]
    walls = [1000.0 + sum(gaps[:i + 1]) for i in range(len(gaps))]
    rig_path, aob_path = tmp_path / "rig" / "state.json", tmp_path / "aob" / "state.json"
    for path, start in ((rig_path, episode.start), (aob_path, rows.start)):
        path.parent.mkdir()
        start(path, 7, config, default_facts_path())
        episode.hand_over(path, wall=1000.0, mono=1000.0)
    rig_clock = [json.loads(episode.call(rig_path, "get_current_time", {}, {"type": "get_current_time"}, now=wall, monotonic=wall))["time"]
                 for wall in walls]
    store, clock, ticks = MemoryServer(), iter(walls), iter(walls)
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, config.asset_id, fixture.SITE)
    producer.reset(rows.replay(episode.load_state(aob_path)))
    sessions = StateFileSessions(aob_path, wall_clock=lambda: next(clock), monotonic_clock=lambda: next(ticks))
    meter = Meter(meter_config(layer_contract.row_of(config), config, fixture.TIMELINE), sessions, producer)
    with bound(store, NAMES):
        aob_clock = [receipt(await meter.call(TASK_SERVER, "get_current_time"))["cycle"] for _ in walls]
    rig, aob = episode.load_state(rig_path), episode.load_state(aob_path)
    assert rig["think_cycles"] == aob["think_cycles"] == [0, 1, 0, 1, 2, 1]
    assert rig["think_seconds"] == aob["think_seconds"] == gaps and rig["think_seconds_carried"] == aob["think_seconds_carried"] == 9.0
    assert rig_clock == aob_clock == [0, 1, 1, 2, 4, 5]


async def test_both_surfaces_keep_the_same_monotonic_gap_beside_the_wall_gap(world, tmp_path):
    """The two clocks of a call on both surfaces: the same wall and monotonic readings give the rig's action server and the metering layer of
    the AssetOpsBench rows the same wall time of every call and the same gaps, here a monotonic clock that moves 3 s less than the
    wall clock in the second gap, as across a sleep of the machine, while the charge follows the monotonic seconds of the chess
    clock and the wall gaps are kept beside it."""
    config = harness.preset_config("advanced_real_time", think_seconds_per_cycle=10)
    walls, monotonics = [1004.0, 1015.0, 1016.0], [504.0, 512.0, 513.0]
    rig_path, aob_path = tmp_path / "rig" / "state.json", tmp_path / "aob" / "state.json"
    for path, start in ((rig_path, episode.start), (aob_path, rows.start)):
        path.parent.mkdir()
        start(path, 7, config, default_facts_path())
        episode.save_state(path, {**episode.load_state(path), "last_call_wall_time": 1000.0, "last_call_monotonic_time": 500.0})
        episode.hand_over(path, wall=1000.0, mono=500.0)
    for wall, monotonic in zip(walls, monotonics):
        episode.call(rig_path, "get_current_time", {}, {"type": "get_current_time"}, now=wall, monotonic=monotonic)
    store, clock, ticks = MemoryServer(), iter(walls), iter(monotonics)
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, config.asset_id, fixture.SITE)
    producer.reset(rows.replay(episode.load_state(aob_path)))
    sessions = StateFileSessions(aob_path, wall_clock=lambda: next(clock), monotonic_clock=lambda: next(ticks))
    meter = Meter(meter_config(layer_contract.row_of(config), config, fixture.TIMELINE), sessions, producer)
    with bound(store, NAMES):
        for _ in walls:
            await meter.call(TASK_SERVER, "get_current_time")
    rig, aob = episode.load_state(rig_path), episode.load_state(aob_path)
    assert rig["think_seconds"] == aob["think_seconds"] == [4.0, 8.0, 1.0]
    assert rig["think_seconds_monotonic"] == aob["think_seconds_monotonic"] == [4.0, 8.0, 1.0]
    assert rig["think_seconds_wall"] == aob["think_seconds_wall"] == [4.0, 11.0, 1.0]
    assert rig["called_at"] == aob["called_at"] == walls and rig["last_call_monotonic_time"] == aob["last_call_monotonic_time"] == 513.0


async def test_the_tiny_asset_runs_through_the_producer_and_the_meter(tiny, tiny_recordings, tmp_path, monkeypatch):
    """A tiny episode on the priced row behind the metering layer and the memory store: the producer writes the tiny
    registry, failure modes and telemetry, a history serves the cycles recorded before the episode, get_failure_modes of a
    class is logged on its component, the task server takes the diagnosis with the three tiny parameters, an order on
    tiny_1_belt at the site of the asset file is the engine's order, and finish ends the episode."""
    config = layers.compose({"asset_id": tiny.asset_id, "layers": ["priced"]})
    path = tmp_path / "state.json"
    # the state names a data file outside the fork by its basename, and a replay finds it through the variable of the asset
    monkeypatch.setenv(tiny.records["data_env"], str(tiny_recordings))
    started = rows.start(path, 1, config, tiny_recordings)
    episode.hand_over(path)
    store, site = MemoryServer(), tiny.text["fallback"]["site"]
    producer = Producer(store, NAMES, ProfileDocuments(tiny, fixture.TIMELINE), fixture.TIMELINE, tiny.asset_id, site)
    producer.reset(rows.replay(started))
    meter = Meter(meter_config(layer_contract.row_of(config), config, fixture.TIMELINE), StateFileSessions(path), producer)
    start = rows.replay(started).initial
    with bound(store, NAMES):
        assert sorted(d["assetnum"] for d in store.all_docs(NAMES.asset)) == ["tiny_1", "tiny_1_bearing", "tiny_1_belt", "tiny_1_fan"]
        assert {d["siteid"] for d in store.all_docs(NAMES.asset)} == {site}
        assert {d["asset_class"] for d in store.all_docs(NAMES.failure_mode)} == {"fan", "belt", "bearing", "test bench"}
        telemetry = store.all_docs(NAMES.iot)
        assert telemetry and all("T1_Temperature_C_1Hz" in r and "T1_Temperature_C_1Hz_std" in r for r in telemetry)
        history = await meter.call("iot", "history", {"site_name": site, "asset_id": tiny.asset_id, "sensors": ["T1_Temperature_C_1Hz"]})
        await meter.call("fmsr", "get_failure_modes", {"asset_class": "belt"})
        listed = {t.name: t for t in await meter.listed_tools(TASK_SERVER)}
        await meter.call(TASK_SERVER, "submit_diagnosis", {**dict(zip(tiny.components, start)), "evidence_ids": ""})
        order = await meter.call("wo", "generate_work_order", {"description": "repair", "asset_num": "tiny_1_belt", "site_id": site})
        await meter.call(TASK_SERVER, "finish", {"summary": "The belt was repaired."})
    assert len(body(history)["observations"]) == len([c for c in rows.replay(started).recorded_cycles() if c < 0])
    assert list(listed["submit_diagnosis"].inputSchema["properties"]) == [*tiny.components, "evidence_ids"]
    engine = rows.replay(episode.load_state(path))
    events = {e["event"]: e for e in engine.log}
    assert events["failure_modes"]["component"] == "belt" and events["work_order_created"]["component"] == "belt"
    assert not order.is_error and receipt(order)["price"] > 0 and engine.done
