#!/bin/bash
# Check C7 of the development steps, run as the command of cpu_job.sbatch: the generated tables stay as they are when runs
# play. It prints the SHA-256 of every table, plays one scripted run per preset of each named grid file under the start set
# named for it, into the smoke folder, prints the hashes again and ends with C7_TABLES_KEPT, or with C7_TABLES_CHANGED and
# the table and status 1. The presets come from grid_job_lists.py, and the command file of the step names the pairs.
#   bash $H/submit.sh cpu_job.sbatch -- bash CausalDiagnosis/cluster_jobs/1_checks/check_tables.sh GRID_FILE=START_SET ...
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"

table_hashes() {
  local table
  for table in $TABLES; do echo "$(sha256_of "$FORK/$table") $table"; done
}

check_tables() {
  [ $# -gt 0 ] || { echo "usage: check_tables.sh GRID_FILE=START_SET ..." >&2; return 2; }
  local pair grid starts entry config mode before after out=$ROOT/runs/smoke/check_tables_${SLURM_JOB_ID:-manual} status=0 table
  before=$(table_hashes)
  echo "$before"
  for pair in "$@"; do
    grid=${pair%%=*}; starts=${pair#*=}
    [ "$grid" != "$pair" ] && [ -n "$starts" ] || { echo "$pair is no GRID_FILE=START_SET" >&2; return 2; }
    for entry in $(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists presets "$GRIDS/$grid"); do
      config=${entry%%:*}; mode=""
      [ "$entry" = "$config" ] || mode="--clock-mode ${entry#*:}"
      # shellcheck disable=SC2086
      python -m CausalDiagnosis.experiments.episode_runners.run_episodes --config "$config" $mode --agent scripted --seeds 1 --starts "$starts" \
        --out "$out" > /dev/null || { echo "the scripted run of $config on $grid failed" >&2; status=1; }
    done
  done
  after=$(table_hashes)
  echo "$after"
  for table in $TABLES; do
    if [ "$(grep " $table\$" <<< "$before")" != "$(grep " $table\$" <<< "$after")" ]; then echo "C7_TABLES_CHANGED $table"; status=1; fi
  done
  [ "$status" != 0 ] || echo "C7_TABLES_KEPT"
  return "$status"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then check_tables "$@"; exit $?; fi
