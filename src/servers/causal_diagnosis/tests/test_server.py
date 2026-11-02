"""The MCP server over stdio when the mcp package is installed: four flat tools, every one declared read only, their
parameters as the tool contract pins them, and answers through a launcher copied beside the released evidence."""
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from servers.causal_diagnosis.main import TOOLS, build_server
from servers.causal_diagnosis.tests.fixture_episode import FixtureEpisode

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp package (pinned in the fork's uv.lock)")
START = (100, 80, 0, 90)


def advertised(server):
    tools = asyncio.run(server.list_tools())
    hints = {t.name: bool(t.annotations and t.annotations.readOnlyHint) for t in tools}
    parameters = {t.name: {name: schema["type"] for name, schema in t.inputSchema["properties"].items()} for t in tools}
    schemas = {t.name: t.inputSchema for t in tools}
    return hints, parameters, schemas


def test_the_four_tools_are_read_only_and_required(tmp_path):
    hints, parameters, schemas = advertised(build_server(None))
    assert {n: h for n, h in advertised(build_server(None, ranking_model="discriminant_urn"))[0].items()} == hints
    assert set(hints) == set(TOOLS) and all(hints.values())
    assert parameters == {"get_episode_evidence": {"sensor_group": "string"}, "rank_root_causes": {"top_k": "integer"},
                          "estimate_intervention_effect": {"action": "string"}, "verify_outcome": {"prediction_id": "string"}}
    assert all(schema.get("required", []) == list(schema["properties"]) for schema in schemas.values())
    assert not any("default" in p for schema in schemas.values() for p in schema["properties"].values())


def test_advertised_tools_match_the_tool_contract():
    if importlib.util.find_spec("servers.contract") is None:
        pytest.skip("servers.contract is not written yet (env-core); the contract test runs once it lands")
    from servers import contract

    hints, parameters, _schemas = advertised(build_server(None))
    assert contract.check_advertised(contract.DIAGNOSTICS_SERVER, hints, parameters) == []
    assert all(entry["class"] == "read" and not entry["scored"] for entry in contract.tools_of(contract.DIAGNOSTICS_SERVER).values())


@needs_mcp
def test_server_answers_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), START, seed=0)
    episode.read("tier1")
    shutil.copy(Path(__file__).resolve().parents[1] / "launcher.py", tmp_path / "causal_server.py")
    src = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=[str(tmp_path / "causal_server.py")], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = (await session.list_tools()).tools
                ranking = await session.call_tool("rank_root_causes", {"top_k": 2})
                effect = await session.call_tool("estimate_intervention_effect", {"action": "replace:valve"})
                bad = await session.call_tool("estimate_intervention_effect", {"action": "replace:motor"})
                return listed, ranking, effect, bad

    listed, ranking, effect, bad = asyncio.run(session_calls())
    assert {t.name for t in listed} == set(TOOLS) and all(t.annotations.readOnlyHint for t in listed)
    ranked = json.loads(ranking.content[0].text)
    assert not ranking.isError and len(ranked["candidates"]) == 2 and ranked["ranking_model"] == "joint_block"
    assert not effect.isError and json.loads(effect.content[0].text)["prediction_id"] == "pred/1"
    assert not bad.isError and "error" in json.loads(bad.content[0].text)
    assert (tmp_path / "predictions.json").exists()

    async def ablation_call():
        params = StdioServerParameters(command=sys.executable, env=env,
                                       args=[str(tmp_path / "causal_server.py"), "ranking_model=discriminant_urn", "abstention_rule=fixed_cut", "abstention_cut=0.9"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("rank_root_causes", {"top_k": 1})

    ablation = json.loads(asyncio.run(ablation_call()).content[0].text)
    assert ablation["ranking_model"] == "discriminant_urn" and "abstention_rule fixed_cut" in " ".join(ablation["assumptions"])
    assert "cut 0.9" in " ".join(ablation["assumptions"])
