#!/bin/bash
set -u
export HPC_DIR="/insomnia001/depts/edu/users/yc4953/projects/assetopsbench-causaldiagnosis/runtime/clones/9fc86eabcd7c/CausalDiagnosis/cluster_jobs" HPML_V4_ROUND=1 RUN_KIND=development SERVE_REACH=nodes
export RUN_AFTER_SERVE="/insomnia001/depts/edu/users/yc4953/projects/assetopsbench-causaldiagnosis/preparation/v4_round24/run_round.sh"
bash "$HPC_DIR/2_serve_model/serve_model.sbatch"
status=$?
echo "V4_ROUND_GPU_JOB_EXIT $status"
exit "$status"
