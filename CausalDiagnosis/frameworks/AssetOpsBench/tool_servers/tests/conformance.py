"""The conformance sequence of the two backends: one recorded script of calls through the metering layer over an episode of the real
time clock row, run on the memory store and on CouchDB, must give the same texts, the same tool_calls entries and the
same documents. The episode runs on the stub engine, or on the engine of the AssetOpsBench rows when the caller builds it. The script calls every AssetOpsBench tool of the rows, pages history
with a cursor, asks for an unknown sensor and an unknown failure mode class, creates two orders in one cycle so that
their reportdate ties, has one order refused by the work order database, filters orders through each Mango index, and
lets the clock run until the repairs close.
"""
import json

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound, database_names
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import COLLECTIONS

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness

NAMES = database_names("econformance0")
SITE = {"site_name": "MAIN"}
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}
CE, PS1 = "CE_CoolingEfficiency_percent_1Hz", "PS1_Pressure_bar_100Hz"
NEXT_PAGE = "the previous history call with its next_cursor"
DIAGNOSIS = "the engine records a diagnosis, as the task server would"
ORDER = {"wonum": "wo-1", "site_id": "MAIN"}

# (server, tool, arguments, thinking cycles before the call)
SCRIPT = [
    ("iot", "sites", {}, 0), ("iot", "asset_ids", SITE, 0), ("iot", "assets", SITE, 0), ("iot", "assets", {**SITE, "assettype": "PUMP"}, 0),
    ("iot", "asset_detail", ASSET, 0), ("iot", "installed_sensors", ASSET, 0), ("iot", "measured_sensors", ASSET, 0),
    ("iot", "find_assets_by_sensors", {**SITE, "sensors": ["cooling"], "substring": True, "match": "any"}, 0),
    ("iot", "find_assets_by_sensors", {**SITE, "sensors": [PS1], "source": "installed"}, 0),
    ("iot", "stream_extent", ASSET, 1), ("iot", "stream_extent", {**ASSET, "sensor": CE, "start": fixture.timestamp(-5)}, 0),
    ("iot", "sensor_coverage", ASSET, 0),
    ("iot", "history", {**ASSET, "sensors": [CE, PS1], "limit": 4}, 0), ("iot", "history", NEXT_PAGE, 0),
    ("iot", "history", {**ASSET, "start": fixture.timestamp(-3), "end": fixture.timestamp(0)}, 2),
    ("iot", "latest_reading", {**ASSET, "sensor": CE}, 0), ("iot", "latest_reading", ASSET, 0),
    ("iot", "history", {**ASSET, "sensors": ["nothing"]}, 0),
    ("fmsr", "get_failure_modes", {"asset_class": "COOLER"}, 0), ("fmsr", "get_failure_modes", {"asset_class": "HYDRAULIC_PUMP"}, 0),
    ("fmsr", "get_failure_modes", {"asset_class": "compressor"}, 0),
    (None, DIAGNOSIS, {"type": "submit_diagnosis", "diagnosis": {c: HEALTHY[c] for c in COMPONENTS}, "evidence_ids": []}, 0),
    ("wo", "generate_work_order", {"description": "replace the cooler", "asset_num": "hyd_1_cooler", "site_id": "MAIN", "priority": 2}, 0),
    ("wo", "generate_work_order", {"description": "service the valve", "asset_num": "hyd_1_valve", "site_id": "MAIN"}, 0),
    ("wo", "generate_work_order", {"description": "pump", "asset_num": "hyd_1_pump", "site_id": "MAIN", "aob_source": {"agent": "rig"}}, 0),
    ("wo", "list_workorders", {"site_id": "MAIN"}, 0), ("wo", "list_workorders", {"site_id": "MAIN", "status": "OPEN"}, 0),
    ("wo", "list_workorders", {"site_id": "MAIN", "asset_num": "hyd_1_cooler", "date_from": "2026-01-01", "date_to": "2026-01-02"}, 0),
    ("wo", "get_workorder", ORDER, 0), ("wo", "get_workorder_tasks", ORDER, 0), ("wo", "get_workorder_costs", ORDER, 0),
    ("wo", "get_workorder_actuals_vs_planned", ORDER, 0), ("wo", "get_workorder_kpis", {"site_id": "MAIN"}, 0),
    ("wo", "get_schedule_calendar", {"site_id": "MAIN"}, 0), ("wo", "get_my_assigned_workorders", {"labor_code": "T1"}, 0),
    ("wo", "get_failure_codes", {}, 0), ("wo", "get_failure_codes", {"code": "fc004"}, 0),
    ("iot", "history", {**ASSET, "sensors": [CE]}, 4), ("wo", "list_workorders", {"site_id": "MAIN", "status": "OPEN"}, 0),
    ("wo", "list_workorders", {"site_id": "MAIN"}, 0), ("wo", "get_workorder_kpis", {"site_id": "MAIN"}, 0),
]


def stub_episode(server, names, thinks):
    return harness.episode(server, names, "real_time", thinks, budget=1000, repair_cycles=2)


async def run(server, episode=stub_episode):
    """The script on one store over the episode that `episode(server, names, thinks=...)` builds; returns the texts, the
    tool_calls entries and the documents, and drops the databases."""
    thinks = [think for *_, think in SCRIPT]
    meter, engine, producer, sessions = episode(server, NAMES, thinks=thinks)
    replies, previous = [], None
    with bound(server, NAMES):
        for server_name, tool, arguments, _ in SCRIPT:
            if server_name is None:
                sessions.step(arguments)
                continue
            if arguments == NEXT_PAGE:
                arguments = {**previous, "cursor": json.loads(replies[-1][0].rsplit("\n", 1)[0])["next_cursor"]}
            reply = await meter.call(server_name, tool, arguments)
            replies.append((reply.text, reply.is_error))
            previous = arguments
        documents = {c: server.all_docs(n) for c, n in zip(COLLECTIONS, NAMES.all())}
    producer.drop()
    return {"replies": replies, "tool_calls": sessions.tool_calls, "documents": documents}


def compare(first, second):
    """Every difference between two runs, by call and by collection, or an empty list."""
    differences = []
    calls = [(server, tool) for server, tool, *_ in SCRIPT if server is not None]
    for i, (a, b) in enumerate(zip(first["replies"], second["replies"])):
        if a != b:
            differences.append({"call": i, "tool": calls[i], "first": a, "second": b})
    for i, (a, b) in enumerate(zip(first["tool_calls"], second["tool_calls"])):
        if a != b:
            differences.append({"tool_calls entry": i, "first": a, "second": b})
    for collection in COLLECTIONS:
        a, b = first["documents"][collection], second["documents"][collection]
        if a != b:
            ids = [x.get("_id") for x, y in zip(a, b) if x != y] or [len(a), len(b)]
            differences.append({"collection": collection, "first differing ids or lengths": ids[:5]})
    if len(first["replies"]) != len(second["replies"]):
        differences.append({"calls": [len(first["replies"]), len(second["replies"])]})
    return differences
