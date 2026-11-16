"""The MCP server over stdio when the mcp package is installed: four flat tools, every one declared read only, their
parameters as the tool contract pins them (required, except the count of candidates and the view option of the branches),
and answers through a launcher copied beside the released evidence, which names the block the fits of the episode leave out."""
from CausalDiagnosis.utils.paths import ROOT, SRC
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import DEFAULT_TOP_K
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import combo_key
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import TOOLS, build_server
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

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
    assert parameters == {"get_episode_evidence": {"sensor_group": "string"}, "rank_root_causes": {"top_k": "integer", "branch_detail": "string"},
                          "estimate_intervention_effect": {"action": "string"}, "verify_outcome": {"prediction_id": "string"}}
    # the two optional parameters are the count of candidates, the first DEFAULT_TOP_K unless the caller names more, and the view
    # option of the reply, so a caller that knows neither gets the first candidates in the compact reply
    optional = {"rank_root_causes": {"top_k": DEFAULT_TOP_K, "branch_detail": "compact"}}
    for name, schema in schemas.items():
        assert schema.get("required", []) == [p for p in schema["properties"] if p not in optional.get(name, {})]
        assert {p: s["default"] for p, s in schema["properties"].items() if "default" in s} == optional.get(name, {})


def test_the_texts_of_the_tools_take_the_words_of_the_asset_and_name_no_server_of_the_rig():
    """The texts a model reads come from the asset, and the write tool of the row is named by its role, since every
    row of the environment table serves it through its own servers."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import tool_words

    words = tool_words()
    server = build_server(None)
    described = {t.name: t.description for t in asyncio.run(server.list_tools())}
    assert set(described) == set(TOOLS) and words["noun"] in server.instructions
    assert f"such as {words['channel']}" in described["get_episode_evidence"] and words["field"] in described["get_episode_evidence"]
    assert f"replace:{words['component']}" in described["estimate_intervention_effect"]
    assert all("hydraulic_rig" not in text and "create_work_order" not in text for text in [*described.values(), server.instructions])


def test_advertised_tools_match_the_tool_contract():
    if importlib.util.find_spec("CausalDiagnosis.environment.Causal_Diagnosis_Environment.tool_contract") is None:
        pytest.skip("CausalDiagnosis.environment.Causal_Diagnosis_Environment.tool_contract is not importable")
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract

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
    src = os.pathsep.join([str(ROOT), str(SRC)])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=[str(tmp_path / "causal_server.py"), f"left_out={combo_key(START)}"], env=env)
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
                                       args=[str(tmp_path / "causal_server.py"), "ranking_model=discriminant_urn", "abstention_rule=fixed_cut", "abstention_cut=0.9",
                                             f"left_out={combo_key(START)}"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("rank_root_causes", {"top_k": 1, "branch_detail": "summary"})

    ablation = json.loads(asyncio.run(ablation_call()).content[0].text)
    assert ablation["ranking_model"] == "discriminant_urn" and "abstention_rule fixed_cut" in " ".join(ablation["assumptions"])
    assert "cut 0.9" in " ".join(ablation["assumptions"])


def test_the_launcher_casts_the_server_fields_of_a_threshold_card(monkeypatch):
    """A decision threshold card reaches the server as key=value arguments of the launcher, which casts the margin to a
    number and the cap and the reserved plans to integers, beside the abstention cut, and passes the rest as given."""
    import runpy
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import main as server_main
    seen = {}
    monkeypatch.setattr(server_main, "main", lambda released, **options: seen.update(options))
    launcher = Path(server_main.__file__).with_name("launcher.py")
    monkeypatch.setattr(sys, "argv", [str(launcher), "e10_margin=0.5", "batch_cap=2", "reserve_plans=2", "abstention_cut=0.9",
                                      "ranking_model=joint"])
    runpy.run_path(str(launcher), run_name="__main__")
    assert seen == {"e10_margin": 0.5, "batch_cap": 2, "reserve_plans": 2, "abstention_cut": 0.9, "ranking_model": "joint"}
    assert isinstance(seen["e10_margin"], float) and isinstance(seen["batch_cap"], int) and isinstance(seen["reserve_plans"], int)


def test_the_launcher_casts_the_left_out_block_and_the_server_fits_without_it(monkeypatch):
    """The driver names at launch the combination the episode started in, the launcher casts it to the levels, and
    the server builds the fits of the episode without that block and passes it nowhere else."""
    import runpy
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import main as server_main
    seen = {}
    monkeypatch.setattr(server_main, "main", lambda released, **options: seen.update(options))
    launcher = Path(server_main.__file__).with_name("launcher.py")
    monkeypatch.setattr(sys, "argv", [str(launcher), "left_out=3,100,0,130"])
    runpy.run_path(str(launcher), run_name="__main__")
    assert seen == {"left_out": (3, 100, 0, 130)}
    built, tools = {}, server_main.CausalTools
    monkeypatch.setattr(server_main, "CausalTools", lambda models, released, **options: built.update(models=models, **options) or tools(models, released, **options))
    server_main.build_server(None, left_out=(3, 100, 0, 130))
    assert built["models"].effect.mu_table.shape == (len(built["models"].space.combinations), len(built["models"].space.fields))
    assert "left_out" not in built and (3, 100, 0, 130) not in built["models"]._pool and len(built["models"]._pool) == len(built["models"].space.combinations) - 1
    server_main.build_server(None)
    assert built["models"] is None


def test_the_instructions_do_not_say_that_a_call_takes_no_time():
    """A call of this server moves no clock of its own. Where the clock of an episode counts the seconds between two calls of
    the action servers, the seconds of a call to this server lie between two such calls and are counted with them, so the
    instructions do not say that a call takes no time."""
    text = build_server(None).instructions
    assert "no simulated time" not in text and "takes no time" not in text
    assert "move no clock themselves" in text and "the seconds of a call to these tools are counted with them" in text
