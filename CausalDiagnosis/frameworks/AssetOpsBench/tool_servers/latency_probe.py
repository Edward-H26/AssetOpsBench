"""Latency probe of the metering layer: what one metered call costs before any agent runs.

A Plan-Execute call starts a Python process, imports AssetOpsBench's servers, replays the episode and calls one tool,
so the probe times each part on the laptop with the memory store: a bare interpreter start, the start with the three
server imports, every tool of the base row over a hyd_1 store of recorded cycles, and the replay of the rig engine
for a growing number of stored actions. A cluster job repeats the tool timings on its CouchDB with --backend couchdb,
which reads COUCHDB_URL, COUCHDB_USERNAME and COUCHDB_PASSWORD.

    python -m CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.latency_probe [--backend memory] [--cycles N] [--repeats 20]
"""
import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import HISTORY_CYCLES
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound, database_names
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from servers.wo import workorders

from CausalDiagnosis.utils.paths import SRC
NAMES = database_names("eprobe0000000")
SITE = {"site_name": "MAIN"}
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}
WORK_ORDER = {"wonum": "wo-1", "site_id": "MAIN"}


def base_row_calls(cycles):
    """Every AssetOpsBench tool of the base row with arguments an agent would send, sensor_stats left out."""
    window = {**ASSET, "start": fixture.timestamp(cycles.start), "end": fixture.timestamp(cycles.stop - 1),
              "sensors": fixture.FIELDS[:6], "limit": 1000}
    return [("iot", "sites", {}), ("iot", "asset_ids", SITE), ("iot", "assets", SITE), ("iot", "asset_detail", ASSET),
            ("iot", "installed_sensors", ASSET), ("iot", "measured_sensors", ASSET),
            ("iot", "find_assets_by_sensors", {**SITE, "sensors": fixture.FIELDS[:1]}), ("iot", "stream_extent", ASSET),
            ("iot", "history", window), ("iot", "latest_reading", ASSET), ("iot", "sensor_coverage", ASSET),
            ("fmsr", "get_failure_modes", {"asset_class": "cooler"}),
            ("wo", "list_workorders", {"site_id": "MAIN"}), ("wo", "get_workorder", WORK_ORDER),
            ("wo", "get_workorder_tasks", WORK_ORDER), ("wo", "get_workorder_costs", WORK_ORDER),
            ("wo", "get_workorder_actuals_vs_planned", WORK_ORDER), ("wo", "get_workorder_kpis", {"site_id": "MAIN"}),
            ("wo", "get_schedule_calendar", {"site_id": "MAIN"}), ("wo", "get_my_assigned_workorders", {"labor_code": "T1"}),
            ("wo", "get_failure_codes", {})]


def median_ms(seconds):
    return round(1000 * statistics.median(seconds), 2)


def time_starts(repeats):
    """Median wall time of a bare interpreter and of one that imports the three servers, each a fresh process."""
    results = {}
    for label, code in (("python start", "pass"),
                        ("python start + iot, fmsr, wo imports", "import servers.iot.main, servers.fmsr.main, servers.wo.main")):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            subprocess.run([sys.executable, "-c", code], cwd=SRC, check=True, capture_output=True)
            samples.append(time.perf_counter() - start)
        results[label] = median_ms(samples)
    return results


def make_store(backend):
    if backend == "memory":
        return MemoryServer()
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.couchdb_backend import CouchServer
    return CouchServer(os.environ["COUCHDB_URL"], os.environ.get("COUCHDB_USERNAME", "admin"), os.environ.get("COUCHDB_PASSWORD", "password"))


async def time_tools(cycles, repeats, store):
    import servers.fmsr.main as fmsr_server
    import servers.iot.main as iot_server
    import servers.wo.main as wo_server
    servers = {"iot": iot_server.mcp, "fmsr": fmsr_server.mcp, "wo": wo_server.mcp}
    fixture.load(store, NAMES, cycles)
    results = {}
    with bound(store, NAMES):
        created = time.perf_counter()
        await workorders.create_workorder(store.couch_client(NAMES.workorder), description="repair the cooler", asset_num="hyd_1_cooler",
                                          site_id="MAIN", wonum="wo-1", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        results["wo create_workorder"] = round(1000 * (time.perf_counter() - created), 2)
        for server, tool, arguments in base_row_calls(cycles):
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                await servers[server].call_tool(tool, arguments)
                samples.append(time.perf_counter() - start)
            results[f"{server} {tool}"] = median_ms(samples)
        full, _ = await servers["iot"].call_tool("history", {**ASSET, "limit": 1000})
        results["characters of a history without a sensor list"] = len(full[0].text)
    for name in NAMES.all():
        store.drop_database(name)
    if hasattr(store, "aclose"):
        await store.aclose()
    return results


def time_replays(actions_counts, repeats):
    """reset and replay of the rig engine as every call of the action server runs it today, the data file loaded once
    per process as load_world caches it. Seed 0 of the default preset accepts 42 one cycle reads before its budget ends
    the episode, so the counts stay below that."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import environment_factory as world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
    facts = str(world.default_facts_path(EpisodeConfig()))
    start = time.perf_counter()
    episode.load_world(facts)
    results = {"load the data file": round(1000 * (time.perf_counter() - start), 2)}
    for count in actions_counts:
        state = episode.new_state(0, EpisodeConfig(), facts, time.time())
        state["actions"] = [{"type": "read_sensors", "sensor_group": "tier1", "cycles": 1}] * count
        state["think_cycles"] = [0] * count
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            episode.replay(state)
            samples.append(time.perf_counter() - start)
        results[f"replay of {count} stored actions"] = median_ms(samples)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("memory", "couchdb"), default="memory")
    parser.add_argument("--cycles", type=int, default=HISTORY_CYCLES + 60,
                        help=f"recorded cycles in the store, the prehistory of {HISTORY_CYCLES} plus a horizon of 60")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    cycles = range(-HISTORY_CYCLES, args.cycles - HISTORY_CYCLES)
    print(f"{args.backend} backend with {len(cycles)} recorded cycles, median of {args.repeats} calls, milliseconds")
    rows = {**time_starts(max(5, args.repeats // 4)), **asyncio.run(time_tools(cycles, args.repeats, make_store(args.backend))),
            **time_replays((0, 10, 20, 40), max(3, args.repeats // 4))}
    width = max(len(k) for k in rows)
    for key, value in rows.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
