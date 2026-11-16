"""The memory store under AssetOpsBench's own iot, fmsr and wo tool code."""
import json
from datetime import datetime, timezone

import pytest
from couchdb3.exceptions import BadRequestError, ConflictError, NotFoundError
from mcp.server.fastmcp.exceptions import ToolError

import servers.fmsr.main as fmsr_server
import servers.iot.main as iot_server
import servers.wo.main as wo_server
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound, database_names, episode_prefix
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from servers.db_errors import DATA_UNAVAILABLE
from servers.iot.telemetry_helper import _iter_records
from servers.wo import workorders
from servers.wo.couch import CouchConflict, CouchError

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture

NAMES = database_names("etest000000000")
T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def server():
    store = MemoryServer()
    fixture.load(store, NAMES)
    with bound(store, NAMES):
        yield store


async def call(mcp, tool, arguments=None):
    contents, _ = await mcp.call_tool(tool, arguments or {})
    return json.loads(contents[0].text)


async def create(server, wonum, asset="hyd_1_cooler", now=T0, **fields):
    return await workorders.create_workorder(server.couch_client(NAMES.workorder), description=f"repair {asset}", asset_num=asset,
                                             site_id="MAIN", wonum=wonum, now=now, **fields)


# ----- Mango rules -----

def test_a_missing_field_matches_only_exists_false():
    store = MemoryServer()
    store.create_database("d")
    store.bulk_insert("d", [{"_id": "a", "x": 1}, {"_id": "b"}])
    db = store.database("d")
    assert [d["_id"] for d in db.find({"x": {"$ne": 2}})["docs"]] == ["a"]
    assert [d["_id"] for d in db.find({"x": {"$exists": False}})["docs"]] == ["b"]
    assert [d["_id"] for d in db.find({"x": {"$exists": True, "$ne": None}})["docs"]] == ["a"]


def test_operators_and_the_order_across_types():
    store = MemoryServer()
    store.create_database("d")
    store.bulk_insert("d", [{"_id": "a", "v": 1, "l": [{"code": "x"}]}, {"_id": "b", "v": "1", "l": []}, {"_id": "c", "v": True}])
    db = store.database("d")
    assert [d["_id"] for d in db.find({"v": 1})["docs"]] == ["a"]
    assert [d["_id"] for d in db.find({"v": {"$gte": 1}})["docs"]] == ["a", "b"]
    assert [d["_id"] for d in db.find({"v": {"$in": [True, "1"]}})["docs"]] == ["b", "c"]
    assert [d["_id"] for d in db.find({"l": {"$elemMatch": {"code": "x"}}})["docs"]] == ["a"]
    with pytest.raises(NotImplementedError):
        db.find({"$or": [{"v": 1}]})


# ----- order -----

async def test_unsorted_finds_follow_ids_unless_an_index_is_usable(server):
    for wonum in ("wo-2", "wo-1", "wo-3"):
        await create(server, wonum)
    client = server.couch_client(NAMES.workorder)
    await workorders.approve_workorder(client, "wo-3", "MAIN", now=T0)
    by_id = await client.find({"type": "workorder", "siteid": "MAIN"})
    assert [d["wonum"] for d in by_id] == ["wo-1", "wo-2", "wo-3"]
    # type, siteid and status are all required, so CouchDB uses the [type, siteid, status] index: APPR before WAPPR
    by_status = await client.find({"type": "workorder", "siteid": "MAIN", "status": {"$in": ["APPR", "WAPPR"]}})
    assert [d["wonum"] for d in by_status] == ["wo-3", "wo-1", "wo-2"]


def test_a_sort_needs_an_index(server):
    with pytest.raises(BadRequestError):
        server.database(NAMES.asset).find({"assetnum": "hyd_1"}, sort=[{"assetnum": "asc"}])


async def test_a_sort_without_index_fails_in_wo_as_couchdb_answers(server):
    with pytest.raises(CouchError, match="HTTP 400"):
        await server.couch_client(NAMES.workorder).find({"type": "workorder"}, sort=[{"reportdate": "desc"}])


def test_bookmarks_page_through_every_record_in_time_order(server):
    docs = list(_iter_records(server.database(NAMES.iot), {"asset_id": "hyd_1"}, page_size=3))
    assert [d["timestamp"] for d in docs] == [fixture.timestamp(c) for c in range(-10, 0)]


# ----- revisions and validation -----

async def test_an_update_needs_the_current_revision(server):
    client = server.couch_client(NAMES.workorder)
    await create(server, "wo-1")
    doc = await client.get("wo:MAIN:wo-1")
    await client.put(dict(doc, status="APPR"))
    with pytest.raises(CouchConflict):
        await client.put(dict(doc, status="COMP"))
    db = server.database(NAMES.failure_mode)
    stale = db.get("fm:cooler")
    db.save(stale)
    with pytest.raises(ConflictError):
        db.save(stale)


async def test_the_work_order_validation_refuses_the_wonum_counter(server):
    with pytest.raises(CouchError, match="HTTP 403"):
        await server.couch_client(NAMES.workorder).next_wonum("MAIN")


@pytest.mark.parametrize("fields", [{"status": "DONE"}, {"worktype": "XX"}, {"wopriority": 9}, {"aob_source": {"agent": "rig"}}])
async def test_the_work_order_validation_refuses_what_couchdb_refuses(server, fields):
    doc = {"_id": "wo:MAIN:wo-9", "type": "workorder", "wonum": "wo-9", **fields}
    with pytest.raises(CouchError, match="HTTP 403"):
        await server.couch_client(NAMES.workorder).put(doc)


async def test_a_dumped_store_loads_with_its_revisions_indexes_and_validation(server, tmp_path):
    await create(server, "wo-1")
    await create(server, "wo-2", asset="hyd_1_valve")
    server.dump(tmp_path / "store.json")
    loaded = MemoryServer.load(tmp_path / "store.json")
    assert all(loaded.all_docs(n) == server.all_docs(n) for n in NAMES.all())
    with bound(loaded, NAMES):
        listed = await call(wo_server.mcp, "list_workorders", {"site_id": "MAIN"})  # a sorted find needs the loaded index
        assert [w["wonum"] for w in listed["work_orders"]] == ["wo-1", "wo-2"]
        client = loaded.couch_client(NAMES.workorder)
        doc = await client.get("wo:MAIN:wo-1")
        await client.put(dict(doc, status="APPR"))
        with pytest.raises(CouchConflict):
            await client.put(dict(doc, status="COMP"))
        with pytest.raises(CouchError, match="HTTP 403"):
            await client.next_wonum("MAIN")


def test_a_changed_design_document_is_refused():
    store = MemoryServer()
    store.create_database("w")
    with pytest.raises(NotImplementedError):
        store.install_design("w", {"_id": "_design/workorders", "validate_doc_update": "function (newDoc) {}"})


# ----- a missing database -----

async def test_a_missing_database_reads_as_unavailable():
    store = MemoryServer()
    with bound(store, NAMES):
        assert (await call(iot_server.mcp, "sites"))["error"] == DATA_UNAVAILABLE
        assert (await call(fmsr_server.mcp, "get_failure_modes", {"asset_class": "cooler"}))["error"] == DATA_UNAVAILABLE
        with pytest.raises(ToolError, match="the data source does not exist"):  # wo lets DatabaseUnavailable through, as on CouchDB
            await call(wo_server.mcp, "list_workorders", {"site_id": "MAIN"})
    with pytest.raises(NotFoundError, match="Database does not exist"):
        store.database("nothing").get("x", check=True)


# ----- AssetOpsBench's tools over the hyd_1 fixture -----

async def test_the_registry_tools(server):
    assert (await call(iot_server.mcp, "sites"))["sites"] == ["MAIN"]
    ids = (await call(iot_server.mcp, "asset_ids", {"site_name": "MAIN"}))["assets"]
    assert ids == ["hyd_1", "hyd_1_accumulator", "hyd_1_cooler", "hyd_1_pump", "hyd_1_valve"]
    detail = await call(iot_server.mcp, "asset_detail", {"site_name": "MAIN", "asset_id": "hyd_1"})
    assert (detail["assettype"], detail["n_installed_sensors"]) == ("HYDRAULIC_PUMP", 17)
    assert (await call(iot_server.mcp, "installed_sensors", {"site_name": "MAIN", "asset_id": "hyd_1"}))["sensors"] == fixture.FIELDS
    measured = (await call(iot_server.mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "hyd_1"}))["sensors"]
    assert measured == sorted(fixture.FIELDS + [f"{f}_std" for f in fixture.FIELDS])
    found = await call(iot_server.mcp, "find_assets_by_sensors", {"site_name": "MAIN", "sensors": ["PS1_Pressure_bar_100Hz"]})
    assert [a["asset_id"] for a in found["assets"]] == ["hyd_1"]
    pumps = await call(iot_server.mcp, "assets", {"site_name": "MAIN", "assettype": "PUMP"})
    assert [a["asset_id"] for a in pumps["assets"]] == ["hyd_1_pump"]


async def test_the_telemetry_tools(server):
    args = {"site_name": "MAIN", "asset_id": "hyd_1"}
    extent = await call(iot_server.mcp, "stream_extent", args)
    assert (extent["total_records"], extent["start_time"], extent["end_time"]) == (10, fixture.timestamp(-10), fixture.timestamp(-1))
    window = {**args, "start": fixture.timestamp(-6), "end": fixture.timestamp(-1), "sensors": ["CE_CoolingEfficiency_percent_1Hz"], "limit": 3}
    first = await call(iot_server.mcp, "history", window)
    second = await call(iot_server.mcp, "history", {**window, "cursor": first["next_cursor"]})
    rows = first["observations"] + second["observations"]
    assert [r["timestamp"] for r in rows] == [fixture.timestamp(c) for c in range(-6, -1)]
    assert list(rows[0]) == ["asset_id", "timestamp", "CE_CoolingEfficiency_percent_1Hz"]
    latest = await call(iot_server.mcp, "latest_reading", args)
    assert latest["timestamp"] == fixture.timestamp(-1) and list(latest["values"])[:2] == ["PS1_Pressure_bar_100Hz", "PS1_Pressure_bar_100Hz_std"]
    coverage = await call(iot_server.mcp, "sensor_coverage", args)
    assert coverage["docs_scanned"] == 10 and {s["non_null_count"] for s in coverage["sensors"]} == {10}


async def test_failure_modes_by_fmsr_class_key(server):
    cooler = await call(fmsr_server.mcp, "get_failure_modes", {"asset_class": "COOLER"})
    assert (cooler["asset_class"], cooler["failure_modes"][0], cooler["exhaustive"]) == ("cooler", "cooler_efficiency_critical", True)
    rig = await call(fmsr_server.mcp, "get_failure_modes", {"asset_class": "HYDRAULIC_PUMP"})
    assert (rig["asset_class"], rig["failure_modes"]) == ("hydraulic pump", ["transient", "steady"])
    pump = await call(fmsr_server.mcp, "get_failure_modes", {"asset_class": "pump"})
    assert pump["failure_modes"][0] == "pump_internal_leakage_severe"  # the rig's pump, never the stock sample


async def test_the_work_order_tools(server):
    client = server.couch_client(NAMES.workorder)
    created = await create(server, "wo-1", aob_source={"prediction_id": "p1", "evidence_ids": ["obs/1/x"]})
    assert created["data"]["reportdate"] == "2026-01-01T00:00:00+00:00"
    await create(server, "wo-2", asset="hyd_1_valve")
    listed = await call(wo_server.mcp, "list_workorders", {"site_id": "MAIN"})
    assert [w["wonum"] for w in listed["work_orders"]] == ["wo-1", "wo-2"]  # a tie in reportdate keeps the id order
    assert (await call(wo_server.mcp, "get_workorder", {"wonum": "wo-1", "site_id": "MAIN"}))["work_order"]["assetnum"] == "hyd_1_cooler"
    assert (await call(wo_server.mcp, "get_workorder_tasks", {"wonum": "wo-1", "site_id": "MAIN"}))["total"] == 0
    assert (await call(wo_server.mcp, "get_workorder_costs", {"wonum": "wo-1", "site_id": "MAIN"}))["total_cost"] == 0
    assert (await call(wo_server.mcp, "get_my_assigned_workorders", {"labor_code": "T1"}))["total"] == 0
    codes = await call(wo_server.mcp, "get_failure_codes")
    assert codes["total"] == 10 and codes["failure_codes"][0]["code"] == "FC001"
    assert (await call(wo_server.mcp, "get_failure_codes", {"code": "fc004"}))["failure_codes"][0]["description"] == "fluid leak observed"
    await workorders.close_workorder(client, "wo-1", "MAIN", now=T0)
    kpis = await workorders.get_workorder_kpis(client, "MAIN", 3, now=T0)
    assert (kpis["data"]["completed"], kpis["data"]["backlog"]) == (1, 1)


async def test_two_runs_give_the_same_documents_and_texts():
    outputs = []
    for _ in range(2):
        store = MemoryServer()
        fixture.load(store, NAMES)
        with bound(store, NAMES):
            await create(store, "wo-1")
            texts = [await call(iot_server.mcp, "latest_reading", {"site_name": "MAIN", "asset_id": "hyd_1"}),
                     await call(wo_server.mcp, "list_workorders", {"site_id": "MAIN"})]
        outputs.append((texts, [store.all_docs(n) for n in NAMES.all()]))
    first, second = outputs
    assert first[1] == second[1]
    assert first[0][1] == second[0][1]
    assert {k: v for k, v in first[0][0].items() if k != "age_seconds"} == {k: v for k, v in second[0][0].items() if k != "age_seconds"}


def test_episode_prefixes_are_valid_couchdb_names(tmp_path):
    prefix = episode_prefix(tmp_path / "episode_0001")
    assert prefix[0] == "e" and len(prefix) == 13
    assert all(n.startswith(prefix + "_") for n in database_names(prefix).all())
