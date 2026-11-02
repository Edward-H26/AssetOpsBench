# A copy beside an episode's released evidence binds one server process to that episode, as the action server's launcher
# does beside its state file; the benchmark drivers write an equivalent launcher that also puts the fork's src on the
# import path (benchmarks/hydraulic_rig/common.py). In this folder the server reads released.json, reads and writes
# predictions.json, and touches nothing else. Optional key=value arguments select the ablation variants of main:
# ranking_model=joint_block|joint|discriminant_urn (default joint_block), abstention_rule=decision|fixed_cut,
# abstention_cut=<probability>.
import sys
from pathlib import Path

from servers.causal_diagnosis.main import main

options = dict(argument.split("=", 1) for argument in sys.argv[1:])
if "abstention_cut" in options:
    options["abstention_cut"] = float(options["abstention_cut"])
main(Path(__file__).resolve().with_name("released.json"), **options)
