"""The stdio server of one server of the AssetOpsBench rows behind the metering layer, which the driver's launcher of an
episode starts.

A launcher names one of iot, fmsr, wo and the task server and the store of the run. The process rebuilds what it needs from the episode
folder, because Plan-Execute starts a process per call and Agent-As-Tool one per server: the store, which is the memory
store read from a file beside the state file or the CouchDB of the job, the binding of AssetOpsBench's modules to the
episode's databases, the producer over the documents of the asset profile, and the meter over the state file. The tools
the process lists are AssetOpsBench's own, or the task tools of the rig's action server, as the meter lists them, and
every call runs through the meter, so the text the agent reads is the text the layer logged. The driver uses the same construction before the first call, to create
the databases, and after the last, to export the documents and drop the databases.
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import ProfileDocuments
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import bind_episode_databases as binding, aob_tool_contract as layer_contract, episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import Meter, StateFileSessions, meter_config
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import Producer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.cycle_timestamps import Timeline
from CausalDiagnosis.mcp_servers.utils import ready_line as ready
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.mcp_servers.utils.service_loop import ServiceLoop
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import transcribed

EPOCH = datetime(2026, 1, 1)  # the timestamp of cycle 0 in every episode, so that two runs of one seed write the same documents
STORE_NAME = "aob_store.json"  # the memory store of an episode between two server processes, beside its state file


class EpisodeStore:
    """One episode's databases on the store of the run: the names derived from the episode folder, the producer over
    the documents of the asset, and the meter of one server over the state file, replayed as the engine of the rows
    with the rows' released writer. A memory store lives in one process, so between two processes it lives in a file
    beside the state file, loaded after the state file lock is taken and dumped before the lock is released; CouchDB
    keeps itself."""

    def __init__(self, state_path, settings, replay=rows.replay, write_released=rows.write_released):
        self.state_path = Path(state_path)
        self.config = EpisodeConfig.from_json(episode.load_state(self.state_path)["config"])
        self.names = binding.database_names(binding.episode_prefix(self.state_path.parent))
        self.store_path = self.state_path.with_name(STORE_NAME) if settings["kind"] == "memory" else None
        self.store = self._open(settings)
        self.timeline = Timeline(EPOCH, self.config.seconds_per_cycle)
        profile = get_profile(self.config.asset_id)
        self.field_channels = {profile.field_name(channel): channel for channel in profile.channels}
        self.producer = Producer(self.store, self.names, ProfileDocuments(profile, self.timeline), self.timeline, self.config.asset_id,
                                 profile.text["fallback"]["site"])
        self._replay, self._write_released = replay, write_released

    @staticmethod
    def _open(settings):
        if settings["kind"] == "memory":
            return MemoryServer()
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.couchdb_backend import CouchServer
        return CouchServer(settings["url"], settings["username"], settings["password"])

    def create(self):
        """The databases of the episode before its first call, holding everything the agent may read at the start."""
        self.producer.reset(self._replay(episode.load_state(self.state_path)))
        if self.store_path:
            self.store.dump(self.store_path)

    def export(self, folder):
        if self.store_path:
            self._load()
        return self.producer.export(folder)

    def drop(self):
        self.producer.drop()
        if self.store_path and self.store_path.exists():
            self.store_path.unlink()

    def meter(self, **clocks):
        """The meter of one server process, or of a call that the driver makes in its own process, the finish that ends an
        episode its runner left open. The keywords wall_clock and monotonic_clock go to StateFileSessions, whose defaults
        stand without them, so a call can be stamped with the time of the last model call of the episode."""
        binding.bind(self.store, self.names)
        sessions = StateFileSessions(self.state_path, replay=self._replay, write_released=self._write_released, **clocks)
        config = meter_config(layer_contract.row_of(self.config), self.config, self.timeline, self.field_channels)
        return Meter(config, _EpisodeSessions(sessions, self), self.producer)

    def _load(self):
        """The memory store as the file holds it, which another server process may have written since this one last
        read it, with the producer and AssetOpsBench's modules pointed at the loaded store."""
        self.store = self.producer.server = MemoryServer.load(self.store_path)
        binding.bind(self.store, self.names)

    def begin_call(self, engine):
        if self.store_path:
            self._load()
        self.producer.resume(engine)

    def end_call(self):
        if self.store_path:
            self.store.dump(self.store_path)


class _EpisodeSessions:
    """The state file sessions of the meter with what a fresh process lacks, inside the state file lock: the memory
    store read from its file before the call and written after it, and the producer told what the replayed engine has
    already published, so that its next sync writes the difference alone."""

    def __init__(self, sessions, episode_store):
        self._sessions, self._store = sessions, episode_store

    def enter(self, server, tool):
        return self._sessions.enter(server, tool)

    def leave(self, server, entered, outcome):
        self._sessions.leave(server, entered, outcome)

    @contextmanager
    def __call__(self, entered):
        with self._sessions(entered) as session:
            self._store.begin_call(session.engine)
            yield session
            self._store.end_call()


class EpisodeServer:
    """One AssetOpsBench server of an episode as the launcher serves it: the tool list and the calls through the meter. With
    a service loop every call of the meter runs on its one long-lived thread, whose thread time credits the service, and the
    store's clients, the meter's lock and the clock patch of the meter all live on that loop; without one, as in a caller's
    own process, the calls run on the caller's loop."""

    def __init__(self, state_path, server, settings, service=None, **sessions):
        self.server = server
        self.meter = EpisodeStore(state_path, settings, **sessions).meter()
        self._service = service

    async def list_tools(self):
        """AssetOpsBench's tool definitions as the meter lists them, without the output schema FastMCP adds to them, since
        the layer serves AssetOpsBench's text with its receipt line and a client refuses text under an output schema."""
        return [tool.model_copy(update={"outputSchema": None}) for tool in await self.meter.listed_tools(self.server)]

    async def call(self, name, arguments):
        served = self.meter.call(self.server, name, arguments or {})
        reply = await (self._service.run(served) if self._service is not None else served)
        return types.CallToolResult(content=[types.TextContent(type="text", text=reply.text)], isError=reply.is_error)


def build_server(state_path, server, store, service=None):
    """The low level MCP server of one AssetOpsBench server behind the meter. The tools and their schemas are
    AssetOpsBench's own and the layer checks the arguments as AssetOpsBench does, so the server validates nothing itself.
    service is the service loop of a server process (ServiceLoop), None in a caller's own process."""
    served = EpisodeServer(state_path, server, store, service=service)
    app = Server(server)
    app.list_tools()(served.list_tools)
    app.call_tool(validate_input=False)(served.call)
    return transcribed(app, Path(state_path).parent, server)


def main(state_path, server, store, started=None):
    """The server process: the service loop is started before the ready line, so no request can reach the meter before it,
    and the ready line goes out immediately before the stdio loop (CausalDiagnosis/mcp_servers/utils/ready_line.py). started is the launcher's stamp."""
    import anyio

    service = ServiceLoop()
    app = build_server(state_path, server, store, service=service)
    ready.announce(Path(state_path).parent, server, started)

    async def serve():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    anyio.run(serve)
