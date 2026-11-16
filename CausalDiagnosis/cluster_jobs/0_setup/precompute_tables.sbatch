#!/bin/bash
#SBATCH -J cd-tables
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
# The tables that every run reads, computed once on a CPU node before the first run of a clone and again after a commit
# that changes a rule of the environment (CausalDiagnosis/cluster_jobs/README.md): the models of the CausalDiagnosis MCP, the separability labels and
# the grades of every rules key that a line of the grid files of the folder grids plays, through precompute_grid_tables.py. No model
# and no card take part, the computation is one process under a file lock, and a second job computes nothing again but a
# table whose code or asset changed since it was written. It is the one process that writes the tables of a clone, and it
# ends with the check that every other job runs before its first run. GRID_FILE, GRID_STARTS and GRID_EXTRA name a launch
# with its own start set or run options, such as the pilot with its start cap; without them it takes every grid file. The
# job prints the SHA-256 of every table after it.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
fork_python; check_data
cd "$FORK" || exit 2
job_header "tables of the grid files of $GRIDS"
grid_tables compute || exit 4
grid_tables check || exit 4
# shellcheck disable=SC2086
sha256sum $TABLES
echo "CD_TABLES_DONE"
