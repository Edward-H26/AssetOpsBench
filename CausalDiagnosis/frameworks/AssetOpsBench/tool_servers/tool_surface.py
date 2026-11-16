"""Everything the core asks of the servers of AssetOpsBench when a run plays them (--servers aob): the engine that replays an
episode of these rows and its start, the store of the episode's documents, the server that takes the finish of the harness, the
version of the briefing, the tools a server lists, the options of a server's launch, the store a run names, and the surface on
which the CausalDiagnosis agent plays them. AOB is the entry of the declaration (frameworks/AssetOpsBench/__init__.py, SURFACES["aob"]); the
literal part of that entry (servers, state file, launch module, stores) is read by the core without importing this module.
"""
import os

COUCHDB_URL_MISSING = ("--store couchdb needs COUCHDB_URL (CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/start_couchdb.sh "
                       "exports it on the cluster)")
COUCHDB_PASSWORD_MISSING = ("--store couchdb needs COUCHDB_PASSWORD with COUCHDB_URL "
                            "(CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/start_couchdb.sh exports both on the cluster)")


class AobToolSurface:
    """The tool surface of the AssetOpsBench rows: AssetOpsBench's iot, fmsr and wo servers and the task server, all behind the
    metering layer of this folder."""

    harness_server = "task"  # the server whose finish the harness calls on an episode its runner left open

    @property
    def env_class(self):
        """The engine an episode of these rows replays on: the engine of the asset over the recorded history."""
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
        return RecordedRigEnv

    @property
    def briefing_version(self):
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import BRIEFING_VERSION
        return BRIEFING_VERSION

    def start(self, state_path, seed, config, facts_path):
        """The state file of a new episode of these rows, with the engine of the rows and their briefing."""
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as rows
        return rows.start(state_path, seed, config, facts_path)

    def store(self, state_path, settings):
        """The databases of an episode on the store a run names: created before the first call, exported and dropped after the
        verifier read the episode, and the meter the harness finish goes through. The module loads AssetOpsBench's servers."""
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
        return EpisodeStore(state_path, settings)

    def store_settings(self, kind):
        """The store of the servers for the launchers and run.json: the memory store, or the CouchDB of a cluster job that the
        variables COUCHDB_URL, COUCHDB_USERNAME and COUCHDB_PASSWORD name."""
        if kind == "memory":
            return {"kind": "memory"}
        url = os.environ.get("COUCHDB_URL")
        if not url:
            raise SystemExit(COUCHDB_URL_MISSING)
        password = os.environ.get("COUCHDB_PASSWORD")
        if not password:  # CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/start_couchdb.sh draws a password per job, so a fallback would fail inside the first episode
            raise SystemExit(COUCHDB_PASSWORD_MISSING)
        return {"kind": "couchdb", "url": url, "username": os.environ.get("COUCHDB_USERNAME", "admin"), "password": password}

    def contract_tools(self, server, row=None, config=None):
        """The tool names a server lists: those a configuration's big layers list when config is given, else those of the given
        row or of every row."""
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
        if config is not None:
            return frozenset(layer_contract.tools_for(server, config))
        return frozenset(layer_contract.tools_of(server, row) if row else layer_contract.tools_of(server))

    def launch_options(self, server, store):
        """The options of a server's launcher beyond the core's: its name and the store, since its process gets no environment
        of its own."""
        return {"server": server, "store": dict(store)}

    @property
    def agent_surface(self):
        """The class of the surface on which the CausalDiagnosis agent plays these servers."""
        from CausalDiagnosis.frameworks.AssetOpsBench.agents.causal_diagnosis_agent_surface import AobSurface
        return AobSurface

    def records_reads(self, row):
        """Whether a row of a view's tool table reads a recorded cycle: a tool of these servers that the contract of the rows binds
        to the engine action read_recorded."""
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
        return (row.get("server") in layer_contract.AOB_SERVERS
                and layer_contract.entries(row["server"]).get(row.get("name"), {}).get("action") == "read_recorded")

    def first_read(self, env, group):
        """The first read of the full sensor group on these rows: the latest reading of every served field of the group."""
        fields = sorted(f"{env.profile.field_name(sensor)}_{statistic}" for sensor in env.config.sensor_groups[group]
                        for statistic in env.profile.statistics["served"])
        return {"type": "read_recorded", "tool": "latest_reading", "sensor_group": group, "fields": fields}

    def tools_of_action(self, action):
        """The tools of these servers that the contract of the rows binds to an engine action."""
        from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
        return frozenset(name for server in layer_contract.AOB_SERVERS for name, entry in layer_contract.entries(server).items()
                         if entry.get("action") == action)


AOB = AobToolSurface()
