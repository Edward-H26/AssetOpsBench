"""The transcript of an episode on the AssetOpsBench rows (servers/transcript.py, CausalDiagnosis/mcp_servers/utils/transcript_hooks.py): the metering
layer writes the line of every call it answers inside the lock of the state file, with the hash of the whole text the agent
reads, and the wrapper of the server writes the tool listings and the calls that fail before the layer answers. Both surfaces
serve a call under the clock of the action server's episode module, which a replay sets. No test states a value computed from
the rig data."""
import json

import mcp.types as types
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import HEALTHY, default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore, build_server
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import sent_text

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness

SETTINGS = {"kind": "memory"}
ASSET = {"site_name": "MAIN", "asset_id": "hyd_1"}
CE = "CE_CoolingEfficiency_percent_1Hz"
START_WALL = 100.0
# a reading, an order before any diagnosis (refused, and a step), a diagnosis, the finish, a call after the end (no step)
SEQUENCE = [
    ("iot", "history", {**ASSET, "sensors": [CE], "limit": 2}),
    ("wo", "generate_work_order", {"description": "replace", "asset_num": "hyd_1_cooler", "site_id": "MAIN"}),
    (TASK_SERVER, "submit_diagnosis", {**HEALTHY, "evidence_ids": ""}),
    (TASK_SERVER, "finish", {"summary": "done"}),
    ("iot", "sites", {}),
]


def start(folder, row):
    """An episode of a row's preset with its databases created, as the driver starts one, under a recorded clock."""
    path = folder / "episode.json"
    with episode.recorded_clock(START_WALL, 5.0):
        rows.start(path, 7, harness.preset_config(row), default_facts_path())
        episode.hand_over(path)
    EpisodeStore(path, SETTINGS).create()
    return path


async def ask(app, tool, arguments, wall):
    """One tool call through the handler that the MCP package calls for a client, under a recorded clock."""
    request = types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name=tool, arguments=arguments))
    with episode.recorded_clock(wall, wall - 95.0):
        return (await app.request_handlers[types.CallToolRequest](request)).root


async def play(path, calls=SEQUENCE):
    """The calls through one server process per server, one second apart: (the result of every call, the hash of the
    released evidence after every call)."""
    apps = {server: build_server(path, server, SETTINGS) for server in dict.fromkeys(server for server, _tool, _arguments in calls)}
    answers, released = [], []
    for k, (server, tool, arguments) in enumerate(calls, start=1):
        answers.append(await ask(apps[server], tool, arguments, START_WALL + k))
        released.append(transcript.file_sha256(episode.released_path(path)))
    return answers, released


@pytest.mark.parametrize("row", ["base", "priced"])
async def test_the_transcript_holds_the_hash_of_the_text_the_agent_read_with_and_without_a_receipt(world, tmp_path, row):
    path = start(tmp_path, row)
    answers, released = await play(path)
    texts = [sent_text(answer) for answer in answers]
    begun, handover, *lines = transcript.read(tmp_path)
    assert (begun["kind"], begun["wall"]) == ("start", START_WALL) and begun["released_sha256"] is not None
    assert (handover["kind"], handover["wall"], handover["mono"]) == ("handover", START_WALL, 5.0)
    # the metering layer enters and exits every call around its line (episode.enter, episode.leave)
    assert [line["kind"] for line in lines] == ["enter", "call", "exit"] * len(SEQUENCE)
    lines = [line for line in lines if line["kind"] == "call"]
    assert [(line["kind"], line["server"], line["tool"], line["arguments"]) for line in lines] == [
        ("call", server, tool, arguments) for server, tool, arguments in SEQUENCE]
    # the hash of every line is the hash of the whole text the client received
    assert [line["answer_sha256"] for line in lines] == [transcript.text_sha256(text) for text in texts]
    stored = episode.load_state(path)["tool_calls"]
    for text, entry in zip(texts, stored):
        if row == "base":  # AssetOpsBench's text alone: the state file keeps a receipt line that the agent never read
            assert not text.endswith(entry["response"])
        else:  # AssetOpsBench's text, a line feed and the receipt line, where the state file keeps the receipt line alone
            assert text.rsplit("\n", 1)[1] == entry["response"] and set(json.loads(entry["response"])) >= {"price", "budget_left", "steps_left"}
    assert [(entry["server"], entry["tool"]) for entry in stored] == [(line["server"], line["tool"]) for line in lines]
    # the order before a diagnosis is refused and takes a step, the call after the end is refused and takes none
    assert [answer.isError for answer in answers] == [line["is_error"] for line in lines] == [False, True, False, False, True]
    assert "refused" in stored[1] and stored[4]["refused"] == episode.EPISODE_OVER
    assert [line["stepped"] for line in lines] == [True, True, True, True, False]
    assert [line["wall"] for line in lines] == [START_WALL + k for k in range(1, 6)]
    assert [line["released_sha256"] for line in lines] == released
    assert not any("answer" in line for line in lines)  # the text of a call that changes the episode is not kept


async def test_a_listing_and_a_tool_that_is_no_tool_of_the_server_are_recorded_and_the_refused_call_takes_a_step(world, tmp_path):
    """A task tool asked of the iot server is a tool that the row does not list on that server: the engine refuses it, names the
    server, and counts it as a step, as the briefing says of refused calls."""
    path = start(tmp_path, "base")
    app = build_server(path, "iot", SETTINGS)
    listed = (await app.request_handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list"))).root.tools
    before = path.read_bytes()
    failed = await ask(app, "submit_diagnosis", {**HEALTHY, "evidence_ids": ""}, START_WALL + 1)  # a task tool, asked of the iot server
    assert path.read_bytes() != before and "the server iot has no tool submit_diagnosis in this episode" in sent_text(failed)
    _begun, handover, listing, entered, call, left = transcript.read(tmp_path)
    assert (handover["kind"], entered["kind"], left["kind"]) == ("handover", "enter", "exit")
    assert entered["n"] == call["enter_n"] == left["enter_n"]
    assert (listing["kind"], listing["server"], listing["listing_sha256"]) == ("list", "iot", transcript.listing_sha256(listed))
    assert failed.isError and (call["kind"], call["server"], call["tool"], call["is_error"], call["stepped"]) == ("call", "iot", "submit_diagnosis", True, True)
    assert call["answer_sha256"] == transcript.text_sha256(sent_text(failed)) and call["arguments"] == {**HEALTHY, "evidence_ids": ""}
    assert call["wall"] == START_WALL + 1 and "answer" not in call  # a stepped call carries its wall time


async def test_a_recorded_clock_gives_both_surfaces_the_recorded_wall_time(world, tmp_path):
    """Under episode.recorded_clock a call of the rig's action server and a call through the state file sessions of the
    metering layer, built with no clock of their own as the launchers build them, both charge and store the recorded reading."""
    rig, aob = tmp_path / "rig", tmp_path / "aob"
    rig.mkdir(), aob.mkdir()
    with episode.recorded_clock(START_WALL, 5.0):
        episode.start(rig / "episode.json", 7, harness.preset_config("static"), default_facts_path())
        episode.hand_over(rig / "episode.json")
    aob_path = start(aob, "base")
    with episode.recorded_clock(START_WALL + 30.0, 35.0):
        episode.call(rig / "episode.json", "get_asset_profile", {}, {"type": "get_asset_profile"})
    await ask(build_server(aob_path, TASK_SERVER, SETTINGS), "get_asset_profile", {}, START_WALL + 30.0)
    for folder in (rig, aob):
        state = episode.load_state(folder / "episode.json")
        assert state["episode_started_at"] == START_WALL and state["called_at"] == [START_WALL + 30.0] == [state["last_call_wall_time"]]
        # the start, the handover, the entry and the line of the call carry the reading of the wall clock, and the exit none
        assert [(line["kind"], line.get("wall")) for line in transcript.read(folder)] == [
            ("start", START_WALL), ("handover", START_WALL), ("enter", START_WALL + 30.0), ("call", START_WALL + 30.0), ("exit", None)]
    assert [line["server"] for line in transcript.read(rig)[2:]] == [contract.ACTION_SERVER] * 3
    assert [line["server"] for line in transcript.read(aob)[2:]] == [TASK_SERVER] * 3
