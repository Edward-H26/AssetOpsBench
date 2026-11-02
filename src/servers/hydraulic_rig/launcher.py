"""Per-episode entry point of the hydraulic rig action server.

A copy of this file beside an episode's state file binds one server process to that episode, as the server tests use it;
the benchmark drivers write an equivalent launcher that also puts the fork's src on the import path
(benchmarks/hydraulic_rig/common.py). A server process receives no arguments and only a few environment
variables (docs/hydraulic_rig/action-server.md), so the copy finds its state file by its own location and
imports the server from the installed package (src/ is on sys.path in the fork's environment).
"""
from pathlib import Path

from servers.hydraulic_rig.main import main

state_path = Path(__file__).resolve().with_name("episode.json")
main(state_path)
