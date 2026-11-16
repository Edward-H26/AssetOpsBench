"""The token clock on the metering layer of the AssetOpsBench rows: a call reads the total of
the episode's ledger of generated tokens as the action server does and charges the same cycles for the same tokens, an
episode without a ledger is refused by name, and the briefing of the rows names seconds_per_token in that mode alone. No
test states a value computed from the rig data."""
import dataclasses

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import TOKEN_MODE
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound, database_names
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import Meter, StateFileSessions, meter_config
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import Producer
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config

NAMES = database_names("etoken0000000")
RATE, THINK = 0.0625, 10
TOKENS = (160, 240, 400)
ROW = "real_time"


def token_rows_config(**changes):
    return preset_config(ROW, clock_mode=TOKEN_MODE, seconds_per_token=RATE, think_seconds_per_cycle=THINK, **changes)


def rows_episode(state_path, store, config, ledger=True):
    if ledger:
        token_clock.open_ledger(state_path.parent)
    started = rows.start(state_path, 7, config, default_facts_path())
    episode.hand_over(state_path)
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, config.asset_id, fixture.SITE)
    producer.reset(rows.replay(started))
    sessions = StateFileSessions(state_path, wall_clock=lambda: started["last_call_wall_time"])
    return Meter(meter_config(layer_contract.row_of(config), config, fixture.TIMELINE), sessions, producer)


async def test_the_metering_layer_charges_the_token_clock_as_the_action_server_does(tmp_path, world):
    store = MemoryServer()
    state_path = tmp_path / "rows" / "state.json"
    state_path.parent.mkdir()
    meter = rows_episode(state_path, store, token_rows_config())
    with bound(store, NAMES):
        for tokens in TOKENS:
            token_clock.add(state_path.parent, tokens)
            await meter.call(TASK_SERVER, "get_current_time", {})
    state = episode.load_state(state_path)
    assert state["think_tokens"] == list(TOKENS) and state["generated_tokens_read"] == sum(TOKENS)
    assert state["think_cycles"] == [1, 1, 3] and state["think_seconds"] == [10.0, 15.0, 25.0]


async def test_a_token_episode_of_the_rows_without_a_ledger_is_refused_by_name(tmp_path, world):
    store = MemoryServer()
    state_path = tmp_path / "state.json"
    meter = rows_episode(state_path, store, token_rows_config(), ledger=False)
    with bound(store, NAMES), pytest.raises(ValueError, match="charges the tokens that the harness of the agent writes"):
        await meter.call(TASK_SERVER, "get_current_time", {})


def test_the_briefing_of_the_rows_names_seconds_per_token_in_the_token_mode_alone(world):
    cycles, blocks = world
    for mode, stated in ((TOKEN_MODE, True), ("generation_time", False)):
        config = dataclasses.replace(token_rows_config(), clock_mode=mode, guidance_level=2)
        env = RecordedRigEnv(cycles, blocks, config)
        env.reset(7)
        text = briefing(env.public_profile(), config)
        assert ("seconds_per_cycle, think_seconds_per_cycle and seconds_per_token" in text) is stated, mode
        assert ("tokens you generate between two calls" in text) is stated, mode
