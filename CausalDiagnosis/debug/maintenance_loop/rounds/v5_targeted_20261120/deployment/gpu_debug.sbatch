#!/bin/bash
set -u
export HPC_DIR="/insomnia001/depts/edu/users/yc4953/projects/assetopsbench-causaldiagnosis/runtime/clones/9fc86eabcd7c/CausalDiagnosis/cluster_jobs" HPML_V5_DEBUG=1 RUN_KIND=development SERVE_REACH=nodes
export RUN_AFTER_SERVE="/insomnia001/depts/edu/users/yc4953/projects/assetopsbench-causaldiagnosis/preparation/v5_targeted/run_targeted.sh"
bash "$HPC_DIR/2_serve_model/serve_model.sbatch"
status=$?
echo "V5_DEBUG_GPU_JOB_EXIT $status"
exit "$status"
