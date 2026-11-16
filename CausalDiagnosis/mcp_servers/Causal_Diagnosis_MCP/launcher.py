# A copy beside an episode's released evidence binds one server process to that episode, as the action server's launcher
# does beside its state file; the benchmark drivers write an equivalent launcher that also puts the fork's src on the
# import path (CausalDiagnosis/experiments/episode_runners/episode_launchers.py, CausalDiagnosis/experiments/episode_runners/run_configuration.py, CausalDiagnosis/experiments/episode_runners/run_folder_layout.py, CausalDiagnosis/experiments/episode_runners/servers_of_a_run.py, CausalDiagnosis/utils/csv_rows.py and CausalDiagnosis/utils/json_files.py). In this folder the server reads released.json, reads and writes
# predictions.json, appends its tool listings and its answers to transcript.jsonl, the transcript of the episode, and touches
# nothing else. Optional key=value arguments select the ablation variants of main:
# ranking_model=joint_block|joint|discriminant_urn (default joint_block), abstention_rule=decision|fixed_cut,
# abstention_cut=<probability>, the server fields of a decision threshold card, e10_margin=<number>, batch_cap=<integer>
# and reserve_plans=<integer>, and left_out=<levels joined by commas>, the combination the episode started in, whose block every
# fit leaves out.
import sys
from pathlib import Path

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import parse_combo
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import main

options = dict(argument.split("=", 1) for argument in sys.argv[1:])
if "abstention_cut" in options:
    options["abstention_cut"] = float(options["abstention_cut"])
for key, kind in (("e10_margin", float), ("batch_cap", int), ("reserve_plans", int), ("left_out", parse_combo)):
    if key in options:
        options[key] = kind(options[key])
if __name__ == "__main__":  # the server runs as a script; an import of this module writes nothing
    main(Path(__file__).resolve().with_name("released.json"), **options)
