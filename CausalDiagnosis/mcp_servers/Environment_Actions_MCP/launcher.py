"""Per-episode entry point of the hydraulic rig action server.

A copy of this file beside an episode's state file binds one server process to that episode, as the server tests use it;
the benchmark drivers write an equivalent launcher that also puts the fork's src on the import path
(CausalDiagnosis/experiments/episode_runners/episode_launchers.py, CausalDiagnosis/experiments/episode_runners/run_configuration.py, CausalDiagnosis/experiments/episode_runners/run_folder_layout.py, CausalDiagnosis/experiments/episode_runners/servers_of_a_run.py, CausalDiagnosis/utils/csv_rows.py and CausalDiagnosis/utils/json_files.py). A server process receives no arguments and only a few environment
variables (CausalDiagnosis/docs/environment-actions-mcp.md), so the copy finds its state file by its own location and
imports the server from the installed package (src/ is on sys.path in the fork's environment).
"""
from pathlib import Path

from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import main

if __name__ == "__main__":  # the server runs as a script; an import of this module writes nothing
    main(Path(__file__).resolve().with_name("episode.json"))
