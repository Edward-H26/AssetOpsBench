"""The producer on the memory store."""
import json

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import database_names
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import COLLECTIONS, EXPORT_NAME, Producer
from servers.wo import workorders
from servers.wo.couch import CouchError

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.engine_stub import StubConfig, StubEngine

NAMES = database_names("eproducer0000")


def start(**stub):
    engine = StubEngine(StubConfig(**stub), fixture.engine_values)
    store = MemoryServer()
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, "hyd_1", fixture.SITE)
    producer.reset(engine)
    return engine, store, producer


def workorder(store, wonum):
    return next(d for d in store.all_docs(NAMES.workorder) if d["wonum"] == wonum)


async def order(engine, store, producer, component="cooler"):
    """An order as the metering layer makes one: the engine's action, wo's document under the engine's id, a sync."""
    engine.diagnosed = True
    envelope, *_ = engine.step({"type": "create_work_order", "component": component, "mode": "standard", "prediction_id": "", "evidence_ids": []})
    created = envelope["result"]["work_order"]
    await workorders.create_workorder(store.couch_client(NAMES.workorder), description=f"replace the {component}",
                                      asset_num=f"hyd_1_{component}", site_id="MAIN", wonum=created["id"],
                                      now=fixture.TIMELINE.utc(created["created_t"]))
    await producer.sync(engine)
    return created["id"]


def test_reset_writes_what_the_loader_writes_and_no_stock_hydraulic_data():
    engine, store, producer = start()
    reference = MemoryServer()
    fixture.load(reference, NAMES)
    assert all(store.all_docs(n) == reference.all_docs(n) for n in NAMES.all())
    records = store.all_docs(NAMES.iot)
    assert [r["timestamp"] for r in records] == [fixture.timestamp(c) for c in range(-10, 0)]
    assert all("cycle" not in r for r in records) and store.all_docs(NAMES.workorder) == []
    assert [d["_id"] for d in store.all_docs(NAMES.failure_mode)].count("fm:pump") == 1


async def test_the_work_order_database_keeps_its_validation_and_indexes():
    engine, store, producer = start()
    with pytest.raises(CouchError, match="HTTP 403"):
        await store.couch_client(NAMES.workorder).next_wonum("MAIN")


async def test_each_recorded_cycle_is_written_once_and_never_before_it_is_recorded():
    engine, store, producer = start(clock_mode="generation_time")
    engine.step({"type": "aob_read"}, think_cycles=3)
    await producer.sync(engine)
    await producer.sync(engine)
    assert [r["timestamp"] for r in store.all_docs(NAMES.iot)][-4:] == [fixture.timestamp(c) for c in (-1, 0, 1, 2)]
    assert len(store.all_docs(NAMES.iot)) == 13 and engine.time == 3


async def test_without_a_clock_an_order_closes_in_its_step_at_the_cycle_time():
    engine, store, producer = start()
    doc = workorder(store, await order(engine, store, producer))
    stamp = "2026-01-01T00:00:00+00:00"
    assert (doc["status"], doc["schedstart"], doc["schedfinish"], doc["actstart"], doc["actfinish"]) == ("COMP", stamp, stamp, stamp, stamp)
    assert len(store.all_docs(NAMES.iot)) == 10  # no reading follows a repair in the base and priced rows


async def test_under_a_clock_the_statuses_follow_the_engine_cycle_by_cycle():
    engine, store, producer = start(clock_mode="generation_time", repair_cycles=2)
    wonum = await order(engine, store, producer)
    assert (workorder(store, wonum)["status"], workorder(store, wonum)["schedfinish"]) == ("APPR", "2026-01-01T00:02:00+00:00")
    engine.step({"type": "aob_read"}, think_cycles=1)
    await producer.sync(engine)
    assert (workorder(store, wonum)["status"], workorder(store, wonum)["actstart"]) == ("INPRG", "2026-01-01T00:00:00+00:00")
    engine.step({"type": "aob_read"}, think_cycles=2)
    await producer.sync(engine)
    assert (workorder(store, wonum)["status"], workorder(store, wonum)["actfinish"]) == ("COMP", "2026-01-01T00:02:00+00:00")
    # the rig was stopped while the order was in repair, so cycles 0 and 1 have no record
    assert [r["timestamp"] for r in store.all_docs(NAMES.iot)][-1] == fixture.timestamp(2)


async def test_an_agent_order_waits_for_the_document_the_layer_writes():
    engine, store, producer = start()
    engine.diagnosed = True
    engine.step({"type": "create_work_order", "component": "valve", "mode": "standard", "prediction_id": "", "evidence_ids": []})
    await producer.sync(engine)
    assert store.all_docs(NAMES.workorder) == []
    await workorders.create_workorder(store.couch_client(NAMES.workorder), description="replace the valve", asset_num="hyd_1_valve",
                                      site_id="MAIN", wonum="wo-1", now=fixture.TIMELINE.utc(0))
    await producer.sync(engine)
    assert workorder(store, "wo-1")["status"] == "COMP"


async def test_export_is_byte_identical_across_two_runs_and_drop_removes_the_set(tmp_path):
    exports = []
    for run in ("a", "b"):
        engine, store, producer = start(clock_mode="generation_time", repair_cycles=1)
        await order(engine, store, producer, "pump")
        engine.step({"type": "aob_read"}, think_cycles=4)
        await producer.sync(engine)
        (tmp_path / run).mkdir()
        exports.append(producer.export(tmp_path / run).read_bytes())
        producer.drop()
        assert store.database_names() == []
    assert exports[0] == exports[1]
    data = json.loads(exports[0])
    assert list(data) == list(COLLECTIONS) and all("_rev" not in d for docs in data.values() for d in docs)
    assert (tmp_path / "a" / EXPORT_NAME).exists()


async def test_a_producer_of_a_new_process_resumes_on_a_loaded_store(tmp_path):
    """One process per call, as Plan-Execute runs the servers: the store goes through a file and the producer of the
    second process resumes from the engine replayed to the same point, so the documents equal those of one process."""
    actions = [({"type": "aob_read"}, 2), ({"type": "aob_read"}, 0), ({"type": "aob_read"}, 3)]
    engine, single, producer = start(clock_mode="generation_time", repair_cycles=2)
    wonum = await order(engine, single, producer)
    for action, think in actions:
        engine.step(action, think_cycles=think)
        await producer.sync(engine)

    engine, store, producer = start(clock_mode="generation_time", repair_cycles=2)
    assert await order(engine, store, producer) == wonum
    for action, think in actions:
        store.dump(tmp_path / "store.json")
        store = MemoryServer.load(tmp_path / "store.json")
        producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, "hyd_1", fixture.SITE)
        producer.resume(engine)
        engine.step(action, think_cycles=think)
        await producer.sync(engine)
    assert all(store.all_docs(n) == single.all_docs(n) for n in NAMES.all()) and workorder(store, wonum)["status"] == "COMP"


async def test_a_call_repeated_after_a_stop_between_the_two_writes_answers_and_writes_each_record_once():
    """A call that stopped after its records reached the store and before the state file of the episode was written leaves the
    next call to replay the state from before it and to write the same records again: a producer of a new process resumes on
    the engine replayed to that state and repeats the call. The store holds those records with equal bodies, so the sync
    answers and the store holds each record once. A record stored with another body under the identifier of a cycle that the
    engine records next contradicts the engine, and the sync refuses it."""
    engine, store, producer = start(clock_mode="generation_time")
    engine.step({"type": "aob_read"}, think_cycles=3)
    await producer.sync(engine)
    written = store.all_docs(NAMES.iot)
    again = StubEngine(StubConfig(clock_mode="generation_time"), fixture.engine_values)
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, "hyd_1", fixture.SITE)
    producer.resume(again)
    again.step({"type": "aob_read"}, think_cycles=3)
    await producer.sync(again)
    assert store.all_docs(NAMES.iot) == written
    ahead = StubEngine(StubConfig(clock_mode="generation_time"), fixture.engine_values)
    ahead.step({"type": "aob_read"}, think_cycles=4)
    record = producer._record(ahead, 3)
    field = next(k for k, v in record.items() if not k.startswith("_") and isinstance(v, float))
    store.bulk_insert(NAMES.iot, [{**record, field: record[field] + 1.0}])
    again.step({"type": "aob_read"}, think_cycles=1)
    with pytest.raises(ValueError, match="with another body than the record of the engine"):
        await producer.sync(again)
