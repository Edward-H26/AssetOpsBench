"""The stdio server of a server of the AssetOpsBench rows behind the meter (CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server), as the driver's
launcher runs it, over the engine of the rows and the memory store: the databases created from the state file before the
first call, the tool list, a call through the meter that writes the state file, the released evidence and the store
file, a second process that submits the diagnosis through the task server, a third that reads the store file and creates
a work order, the export and the drop, and the server over stdio from a launcher as the agent runners start it."""
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, default_facts_path
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import database_names, episode_prefix
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER, row_of, tools_of
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import TOOLS
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import STORE_NAME, EpisodeServer, EpisodeStore
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness

from CausalDiagnosis.utils.paths import ROOT, SRC
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}
CE = "CE_CoolingEfficiency_percent_1Hz"
SETTINGS = {"kind": "memory"}
LAUNCHER = f'''import sys
from pathlib import Path
sys.path[:0] = {[str(ROOT), str(SRC)]!r}
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import main
main(Path(__file__).resolve().with_name("episode.json"), server="iot", store={SETTINGS!r})
'''


def stored(folder, collection):
    """The documents of one collection as the memory store's file beside the state file holds them."""
    name = getattr(database_names(episode_prefix(folder)), collection)
    return json.loads((folder / STORE_NAME).read_text(encoding="utf-8"))[name]["docs"]


def start_episode(folder, row="base"):
    """An episode of a row's preset with its databases created, as the driver starts one."""
    config = harness.preset_config(row)
    state_path = folder / "episode.json"
    rows.start(state_path, 7, config, default_facts_path())
    episode.hand_over(state_path)
    EpisodeStore(state_path, SETTINGS).create()
    return state_path


async def test_the_server_serves_assetopsbench_tools_through_the_meter_and_keeps_the_store_between_processes(world, tmp_path):
    assert row_of(harness.preset_config("base")) == "base"
    state_path = start_episode(tmp_path)
    prehistory = rows.replay(episode.load_state(state_path)).recorded_cycles()
    assert len(stored(tmp_path, "iot")) == len(prehistory) > 0 and stored(tmp_path, "workorder") == []
    assert {d["assetnum"] for d in stored(tmp_path, "asset")} > {"hyd_1", "hyd_1_cooler"}

    iot = EpisodeServer(state_path, "iot", SETTINGS)
    assert {t.name for t in await iot.list_tools()} == set(TOOLS["iot"])
    result = await iot.call("history", {**ASSET, "sensors": [CE], "limit": 3})
    assert not result.isError, result.content[0].text
    records = json.loads(result.content[0].text)["observations"]
    assert len(records) == 3 and all(CE in record for record in records)
    state = episode.load_state(state_path)
    assert [a["type"] for a in state["actions"]] == ["read_recorded"] and [c["tool"] for c in state["tool_calls"]] == ["history"]
    assert state["tool_calls"][0]["summary"]["rows"] == 3 and state["tool_calls"][0]["price"] == 0
    released = json.loads(episode.released_path(state_path).read_text(encoding="utf-8"))
    assert [o["t"] for o in released["observations"]] == prehistory[:3]

    # a second process, the task server, records the diagnosis of the hidden start, which the third process replays
    engine = rows.replay(state)
    task = EpisodeServer(state_path, TASK_SERVER, SETTINGS)
    assert {t.name for t in await task.list_tools()} == set(tools_of(TASK_SERVER, "base"))
    result = await task.call("submit_diagnosis", {**dict(zip(COMPONENTS, engine.initial)), "evidence_ids": ""})
    assert not result.isError, result.content[0].text
    component = next(c for c, level in zip(COMPONENTS, engine.initial) if level != HEALTHY[c])
    wo = EpisodeServer(state_path, "wo", SETTINGS)
    result = await wo.call("generate_work_order", {"description": "repair", "asset_num": f"hyd_1_{component}", "site_id": "MAIN"})
    assert not result.isError, result.content[0].text
    assert [(d["wonum"], d["status"]) for d in stored(tmp_path, "workorder")] == [("wo-1", "COMP")]  # closed in its call without a clock
    state = episode.load_state(state_path)
    assert [c["tool"] for c in state["tool_calls"]] == ["history", "submit_diagnosis", "generate_work_order"]
    assert state["tool_calls"][2]["summary"]["work_order"]["id"] == "wo-1"

    store = EpisodeStore(state_path, SETTINGS)
    exported = json.loads(store.export(tmp_path).read_text(encoding="utf-8"))
    assert exported["workorder"][0]["status"] == "COMP" and len(exported["iot"]) == len(prehistory) and "_rev" not in exported["iot"][0]
    store.drop()
    assert not (tmp_path / STORE_NAME).exists() and (tmp_path / "aob_documents.json").exists()


async def test_the_meter_takes_the_two_clocks_of_its_caller_and_keeps_its_defaults_without_them(world, tmp_path):
    """A call made by the driver in its own process is stamped with the clocks it gives, as the harness finish of an
    episode is stamped with the time of the last model call, and a meter without them reads the clocks of the system."""
    state_path = start_episode(tmp_path)
    meter = EpisodeStore(state_path, SETTINGS).meter(wall_clock=lambda: 1_000_000_000.0, monotonic_clock=lambda: 42.0)
    await meter.call("iot", "sites", {})
    state = episode.load_state(state_path)
    assert (state["last_call_wall_time"], state["last_call_monotonic_time"]) == (1_000_000_000.0, 42.0)
    await EpisodeStore(state_path, SETTINGS).meter().call("iot", "sites", {})
    assert episode.load_state(state_path)["last_call_wall_time"] > 1_000_000_000.0


async def test_a_launcher_serves_the_tools_over_stdio(world, tmp_path):
    state_path = start_episode(tmp_path)
    launcher = tmp_path / "iot_server.py"
    launcher.write_text(LAUNCHER, encoding="utf-8")
    async with stdio_client(StdioServerParameters(command=sys.executable, args=["-I", str(launcher)])) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {t.name for t in listed.tools} == set(TOOLS["iot"])
            result = await session.call_tool("sites", {})
            assert not result.isError and "MAIN" in result.content[0].text
            refused = await session.call_tool("history", {**ASSET, "sensors": ["nothing"]})
            assert refused.isError or "unknown sensors" in refused.content[0].text
    state = episode.load_state(state_path)
    assert [c["tool"] for c in state["tool_calls"]] == ["sites", "history"] and [a["type"] for a in state["actions"]] == ["aob_read", "aob_read"]
