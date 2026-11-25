"""Every preset serves the same environment tools with or without CausalDiagnosis MCP advice.
The profile lists the actual tools; only the separate diagnostic server changes."""
import re
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layers_on, layer_2_real_time as real_time, layer_4_repair_rules as repair_rules
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import AOB_SERVERS, COVERS, TASK_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import DEFINITIONS_ONLY
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config

from CausalDiagnosis.utils.paths import ROOT
GRIDS = sorted((ROOT / "CausalDiagnosis" / "cluster_jobs" / "grids").glob("grid_*.txt"))
PRESETS = sorted({m for g in GRIDS for m in re.findall(r"--config (\S+)", g.read_text(encoding="utf-8"))})
DIAGNOSTICS_TOOLS = set(contract.tools_of(contract.DIAGNOSTICS_SERVER))


def expected_task_tools(config):
    """The environment's tools depend only on the enabled layers."""
    tools = {"get_asset_profile", "summarize_changes_since", "submit_diagnosis", "declare_insufficient_evidence", "finish"}
    if config.time_tools:
        tools |= {"get_current_time", "wait_for_notification"}
    if repair_rules.NAME in layers_on(config) and real_time.has_clock(config.clock_mode, config.durations):
        tools.add("lookup_fix")
    return tools


def test_the_grid_files_name_presets():
    assert len(GRIDS) == 5 and len(PRESETS) == 19


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("ours", (False, True))
def test_each_level_serves_its_tools(preset, ours):
    config = preset_config(preset, diagnostics_server=ours)
    assert set(layer_contract.tools_for(TASK_SERVER, config)) == expected_task_tools(config)
    served = {name for server in (*AOB_SERVERS, TASK_SERVER) for name in layer_contract.tools_for(server, config)}
    table = {row["name"] for row in layer_contract.public_table(config)}
    assert table == served | (DIAGNOSTICS_TOOLS if ours else set())
    assert not set(COVERS) & set(layer_contract.tools_for(TASK_SERVER, config))  # one entry per action


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("ours", (False, True))
def test_the_profile_of_the_rows_lists_what_the_row_serves(world, preset, ours):
    config = preset_config(preset, diagnostics_server=ours)
    cycles, blocks = world
    engine = RecordedRigEnv(cycles, blocks, config)
    engine.reset(7)
    names = [row["name"] for row in engine.public_profile()["tools"]]
    assert names == [row["name"] for row in layer_contract.public_table(config)]


def test_the_framework_actions_cover_the_same_physical_operations():
    assert set(COVERS) == {"read_sensors", "get_failure_modes", "create_work_order", "get_fix_history"}
    assert layer_contract.ACTION_SERVER_NAME == TASK_SERVER


async def test_the_rig_action_server_lists_all_twelve_actions():
    names = [t.name for t in await build_server(DEFINITIONS_ONLY, record=False).list_tools()]
    assert names == list(contract.tools_of(contract.ACTION_SERVER))


@pytest.mark.parametrize("preset", PRESETS)
def test_diagnostic_advice_changes_only_the_diagnostic_server(preset):
    """The ablation preserves environment tools, their public prices and durations on every grid preset."""
    off = preset_config(preset, diagnostics_server=False)
    on = preset_config(preset, diagnostics_server=True)
    for server in (*AOB_SERVERS, TASK_SERVER):
        assert layer_contract.tools_for(server, off) == layer_contract.tools_for(server, on)
    before = layer_contract.public_table(off)
    after = layer_contract.public_table(on)
    assert before == [row for row in after if row["server"] != contract.DIAGNOSTICS_SERVER]
    assert {row["name"] for row in after if row["server"] == contract.DIAGNOSTICS_SERVER} == DIAGNOSTICS_TOOLS
