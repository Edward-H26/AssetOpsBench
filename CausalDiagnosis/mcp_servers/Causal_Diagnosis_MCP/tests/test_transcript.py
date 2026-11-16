"""The transcript of an episode at the diagnostics server (servers/transcript.py, CausalDiagnosis/mcp_servers/utils/transcript_hooks.py): every listing
and every answer enters it, with the hash and the text of the answer and the hash of the released evidence that the call read.
The server only reads the released evidence, so no other file of a server keeps what it answered. No test states a value
computed from the rig data."""
import asyncio
import json
from pathlib import Path

import pytest

mcp_types = pytest.importorskip("mcp.types")  # the MCP package is pinned in the fork's uv.lock

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode  # noqa: E402
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import lowlevel_of, sent_text  # noqa: E402

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

START = (100, 80, 0, 90)


def ask(app, tool, arguments):
    """One tool call through the handler that the MCP package calls for a client. Returns the result the client would get."""
    request = mcp_types.CallToolRequest(method="tools/call", params=mcp_types.CallToolRequestParams(name=tool, arguments=arguments))
    return asyncio.run(lowlevel_of(app).request_handlers[mcp_types.CallToolRequest](request)).root


def test_every_answer_of_the_server_enters_the_transcript_with_the_evidence_it_read(tmp_path):
    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), START, seed=0)
    episode.read("tier1")
    app = build_server(episode.released_path, left_out=START)
    listed = asyncio.run(lowlevel_of(app).request_handlers[mcp_types.ListToolsRequest](mcp_types.ListToolsRequest(method="tools/list"))).root.tools
    first = transcript.file_sha256(episode.released_path)
    calls = [("rank_root_causes", {"top_k": 2}), ("estimate_intervention_effect", {"action": "replace:valve"})]
    answers = [ask(app, tool, arguments) for tool, arguments in calls]
    episode.read("tier1")  # the evidence moves on between two calls of the server
    second = transcript.file_sha256(episode.released_path)
    later = [("rank_root_causes", {"top_k": 0}), ("verify_outcome", {"prediction_id": "pred/1"}), ("no_such_tool", {})]
    answers += [ask(app, tool, arguments) for tool, arguments in later]
    listing, *lines = transcript.read(tmp_path)
    assert first != second and (listing["kind"], listing["server"]) == ("list", contract.DIAGNOSTICS_SERVER)
    assert listing["listing_sha256"] == transcript.listing_sha256(listed)
    assert [(line["kind"], line["server"], line["tool"], line["arguments"]) for line in lines] == [
        ("call", contract.DIAGNOSTICS_SERVER, tool, arguments) for tool, arguments in calls + later]
    # every line keeps the text the client received and its hash, and the server takes no step of the environment
    assert [line["answer"] for line in lines] == [sent_text(answer) for answer in answers]
    assert [line["answer_sha256"] for line in lines] == [transcript.text_sha256(sent_text(answer)) for answer in answers]
    assert not any(line["stepped"] for line in lines) and all("wall" not in line for line in lines)
    assert json.loads(lines[0]["answer"])["candidates"] and json.loads(lines[1]["answer"])["prediction_id"] == "pred/1"
    # each call names the released evidence it read; a call that answers before it reads the evidence, or that fails, names none
    assert [line.get("released_sha256") for line in lines] == [first, first, None, second, None]
    assert "error" in json.loads(lines[2]["answer"]) and "released_sha256" not in lines[2]
    assert [line["is_error"] for line in lines] == [False, False, False, False, True] and answers[4].isError


def test_a_server_that_serves_no_episode_writes_no_transcript(tmp_path):
    app = build_server(None)
    assert "error" in json.loads(sent_text(ask(app, "rank_root_causes", {"top_k": 1})))
    assert list(tmp_path.iterdir()) == [] and not (Path.cwd() / transcript.NAME).exists()
