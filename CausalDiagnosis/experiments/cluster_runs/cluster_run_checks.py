"""The checks of the cluster run order (CausalDiagnosis/cluster_jobs/README.md), each read from the run folders of a job and printed as lines for its
log. No check writes into a run folder and none enters a table.

    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks latency RUNS   the latency of the model calls, per run and pooled
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks round RUNS     the six rows of a development round against the registered limits, exit 1 on a miss
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks pilot RUNS     the thinking rate and the token rate of the pilot, the patience rule of
                                                                the failure card and the projected duration of a registered run
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks gate RUNS      the gate read off the last round, exit 1 on a miss
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks cap [ASSET]    the --start-cap of a round and of a pilot on the asset, the rig by default
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks suspect RUNS   the runs of a suspended or requeued job, exit 1 when one is named
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks overlap RUNS   every run alive during an episode on the wall clock, exit 1 on any
    python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks nodes RUNS...  the node of every run from its ledger, with a count per node

RUNS is a folder that holds run folders, such as runs/grid_<job id>. The round, the pilot and the gate read finished runs of the
development starts alone, the registered share of them whose count cap prints, and never a run of the test set. Every number they
compare with comes from the registered file, and a number that is still open stops the check with its name. The pilot reads every
step of its patience rule under one folder, the runs of a step sharing the patience of the failure card that their configuration
records.
"""
import sys
from pathlib import Path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from . import run_health_checks, development_checks


CHECKS = {"latency": run_health_checks.latency, "round": development_checks.round_check, "pilot": development_checks.pilot, "gate": development_checks.gate, "suspect": run_health_checks.suspect, "overlap": run_health_checks.overlap}


def main(argv):
    if len(argv) >= 2 and argv[0] == "nodes":
        return run_health_checks.nodes([Path(root) for root in argv[1:]])
    if argv[:1] == ["cap"] and len(argv) <= 2:
        print(development_checks.start_cap(argv[1] if len(argv) == 2 else DEFAULT_ASSET))
        return 0
    if len(argv) != 2 or argv[0] not in CHECKS:
        raise SystemExit(__doc__)
    return CHECKS[argv[0]](Path(argv[1]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
