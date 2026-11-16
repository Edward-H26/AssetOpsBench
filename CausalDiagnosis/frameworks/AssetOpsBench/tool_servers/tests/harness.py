"""One episode behind the metering layer on a given store, over the stub engine or over the engine of the AssetOpsBench
rows on a row's preset, and the helpers that read a reply, shared by the meter tests and the conformance run."""
from CausalDiagnosis.utils.paths import CONFIGS, preset_path
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig, default_sensor_groups
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, default_facts_path
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import database_names
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import Meter, StateFileSessions, meter_config
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import Producer
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as rig_episode

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.engine_stub import StubConfig, StubEngine

PRICES = {"tier1": 1, "tier2": 3, "tier3": 10}
PRESETS = CONFIGS
NAMES = database_names("eengine000000")  # the databases of the episodes over the engine of the rows


class MemorySessions:
    """The engine held in memory instead of the state file, the thinking cycles of successive calls given in advance."""

    def __init__(self, engine, thinks=()):
        self.engine = engine
        self.thinks = list(thinks)
        self.actions = []
        self.tool_calls = []
        self.answers = []  # (server, tool, reply) of every call the meter answered

    def enter(self, server, tool):
        return {"wall": 0.0, "mono": 0.0, "thread": 0.0, "n": None}

    def leave(self, server, entered, outcome):
        pass

    @contextmanager
    def __call__(self, entered):
        yield self

    def step(self, action):
        think = self.thinks.pop(0) if self.thinks else 0
        self.actions.append((action, think))
        return self.engine.step(action, think_cycles=think)

    def log_call(self, entry):
        self.tool_calls.append(entry)

    def answered(self, server, tool, arguments, reply):
        self.answers.append((server, tool, reply))


def episode(server, names, row, thinks=(), **stub):
    """A stub engine of the row, its producer reset on the store, and the meter over both."""
    priced = row != "base"
    stub.setdefault("clock_mode", "generation_time" if row in ("real_time", "advanced_real_time") else "none")
    stub.setdefault("read_prices", PRICES if priced else {g: 0 for g in PRICES})
    engine = StubEngine(StubConfig(**stub), fixture.engine_values)
    producer = Producer(server, names, fixture.DOCUMENTS, fixture.TIMELINE, "hyd_1", fixture.SITE)
    producer.reset(engine)
    config = SimpleNamespace(asset_id="hyd_1", components=COMPONENTS, sensor_groups=default_sensor_groups(), prices={"read": stub["read_prices"]})
    sessions = MemorySessions(engine, thinks)
    meter = Meter(meter_config(row, config, fixture.TIMELINE, task_tools=layer_contract.tools_of(TASK_SERVER, row)), sessions, producer)
    return meter, engine, producer, sessions


def preset_config(row, **overrides):
    """The configuration of a row of the environment table, its preset composed from its big layers as the
    drivers compose it, with the given fields replaced."""
    config = layers.compose(json.loads(preset_path(row).read_text(encoding="utf-8")))
    return EpisodeConfig.from_json({**config.to_json(), **overrides})


def engine_episode(server, names, row, world, thinks=(), seed=7, **overrides):
    """The engine of the AssetOpsBench rows on the row's preset, its producer reset on the store, and the meter over both,
    on the row the configuration plays once the overrides apply."""
    cycles, blocks = world
    config = preset_config(row, **overrides)
    engine = RecordedRigEnv(cycles, blocks, config)
    engine.reset(seed)
    producer = Producer(server, names, fixture.DOCUMENTS, fixture.TIMELINE, config.asset_id, fixture.SITE)
    producer.reset(engine)
    sessions = MemorySessions(engine, thinks)
    return Meter(meter_config(layer_contract.row_of(config), config, fixture.TIMELINE), sessions, producer), engine, producer, sessions


def memory_engine_episode(world, row, thinks=(), **overrides):
    """engine_episode on a fresh memory store under the harness's database names: the meter, the engine, the store and
    the sessions."""
    store = MemoryServer()
    meter, engine, _producer, sessions = engine_episode(store, NAMES, row, world, thinks, **overrides)
    return meter, engine, store, sessions


def state_file_episode(state_path, row, store, seed=7):
    """An episode of a row's preset started as the drivers start the AssetOpsBench rows and replayed on every call as
    the launchers run it, with no thinking time between calls: its producer reset on the store and the meter over the
    state file sessions."""
    config = preset_config(row)
    started = rows.start(state_path, seed, config, default_facts_path())
    rig_episode.hand_over(state_path)
    producer = Producer(store, NAMES, fixture.DOCUMENTS, fixture.TIMELINE, config.asset_id, fixture.SITE)
    producer.reset(rows.replay(started))
    sessions = StateFileSessions(state_path, wall_clock=lambda: started["last_call_wall_time"])
    return Meter(meter_config(layer_contract.row_of(config), config, fixture.TIMELINE), sessions, producer)


def receipt(reply):
    return json.loads(reply.text.rsplit("\n", 1)[1])


def body(reply):
    return json.loads(reply.text.rsplit("\n", 1)[0])


def answer(reply, row):
    """The JSON answer of a reply, without the receipt line that ends it from the priced row on."""
    return json.loads(reply.text) if row == "base" else body(reply)


async def direct(server, tool, arguments):
    contents, _ = await server.mcp.call_tool(tool, arguments)
    return contents[0].text
