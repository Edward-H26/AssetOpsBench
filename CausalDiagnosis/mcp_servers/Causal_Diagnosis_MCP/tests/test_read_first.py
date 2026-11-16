"""The flag of the ranking under its one name: the key the reply carries, the property the tool returns and the step of both
briefings at guidance level 3. The old key is spelled from two parts, so that a search of the tree for the old key finds no
line of this file."""
import asyncio
import re
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_briefing
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import Decision
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import E10Rule, RootCauses
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode, episode_models
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import briefing as rig_briefing

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

OLD = "insufficient" + "_evidence"
OLD_STEP = "declare " + "insufficient evidence with the separating test the ranking names when it reports the evidence as insufficient"
NEW_STEP = "while the ranking says read first, take the reading it names before you diagnose"
START = (100, 80, 0, 90)  # a start with two faulty components, the valve and the accumulator


@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


def test_the_reply_carries_the_flag_as_read_first(world, tmp_path):
    """On the fixture row, whose rules hold prices, before any reading the rule reads the record first: the reply in every
    form carries read_first true and no key under the old name, the block e10_rule of the long form names its own flag
    read_first, the result model and the decision of E10 name the field read_first, and the message of the long form says read
    first."""
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0)
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    for detail in ("compact", "summary", "full"):
        ranking = tools.rank_root_causes(3, detail)
        reply = ranking.model_dump()
        assert reply["read_first"] is True and OLD not in reply
        if detail != "compact":
            assert set(reply["e10_rule"]) == set(E10Rule.model_fields) and "read_first" in reply["e10_rule"] and OLD not in reply["e10_rule"]
            assert "read first: " in ranking.message
    assert "read_first" in RootCauses.model_fields and OLD not in RootCauses.model_fields
    assert "read_first" in Decision._fields and OLD not in Decision._fields


def test_the_ranking_tool_returns_read_first():
    """The output schema of rank_root_causes, as the server lists its tools, names the property read_first and no property
    under the old name, and the description of the tool states the decision under the new name."""
    tools = {tool.name: tool for tool in asyncio.run(build_server(None).list_tools())}
    rank = tools["rank_root_causes"]
    properties = rank.outputSchema["$defs"]["RootCauses"]["properties"]
    assert "read_first" in properties and OLD not in properties
    assert "the decision whether to read first, read_first: " in rank.description and OLD not in rank.description


def test_both_briefings_carry_the_new_step_at_guidance_level_3(world):
    """The briefing of the rig and the briefing of the AssetOpsBench rows at guidance level 3, with the diagnostics server: the
    procedure holds the step on read first, the step on the declaration stands in neither, and no step begins with declare."""
    cycles, blocks = world
    config = EpisodeConfig(guidance_level=3)
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(7)
    profile = env.public_profile()
    for module in (rig_briefing, aob_briefing):
        text = module.briefing(profile, config)
        assert NEW_STEP in text and OLD_STEP not in text
        assert not re.search(r"(?:Procedure: |; )\d+\. declare", text)
