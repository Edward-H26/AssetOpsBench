#!/bin/bash
set -euo pipefail
source "/insomnia001/depts/edu/users/yc4953/projects/assetopsbench-causaldiagnosis/runtime/clones/9fc86eabcd7c/CausalDiagnosis/cluster_jobs/env.sh"
fork_python
cd "$FORK"
export ROOT RUNS_IN_FLIGHT=1 HYD_ENDPOINT_FILE="$ENDPOINT_FILE"
ROUND="/insomnia001/depts/edu/users/yc4953/projects/assetopsbench-causaldiagnosis/runtime/clones/9fc86eabcd7c/CausalDiagnosis/debug/maintenance_loop/rounds/v4_24_20261120"
for n in 1 2 3 4 5 6 7 8; do
 python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables --config "$ROUND/configs/scn$n.json" --check
done
python "$ROUND/source/run_round.py" --round "$ROUND" --base-url "$BASE_URL"
echo ROUND_ARCHIVED_RELEASING_GPU
