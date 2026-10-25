# A copy beside an episode's released answers binds one server process to that episode, as the environment-action
# server's launcher does beside its state file; the benchmark drivers write an equivalent launcher that also puts the
# fork's src on the import path (benchmarks/hydraulic_rig/common.py). In this folder the server reads released.json,
# reads and writes predictions.json, and touches nothing else.
from pathlib import Path

from servers.causal_diagnosis.main import main

main(Path(__file__).resolve().with_name("released.json"))
