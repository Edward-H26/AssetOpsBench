#!/bin/bash
#SBATCH -J cd-cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
# One job for any command of the tree that needs no card: the rescoring, the replay certificates, the offline table of the
# MCP, the label of the digests and check_tables.sh. The command runs in the folder of the tree with
# the environment of its lock, after the data check, with RUNS of the kind of the tree, and the job ends with its status.
# The folders of env.sh and DATA_FILE, the data file of record of the tree, are exported, so that a command given as
# bash -c '...' can name them. With CPU_COUCHDB=1 the job starts CouchDB first, for a command that replays runs on that
# store, and with CPU_TABLES=1 it first checks the tables of the grid files of the folder grids as the tables job leaves them, and computes none, so
# that a replay hashes the tables the runs recorded. With CPU_CARD_OF_RECORD=1 the job first checks that the card of its node
# is the card of record, which env.sh must name, and stops with GPU_NOT_OF_RECORD and status 2 on another card, for a command on
# a node with a card whose result binds the nodes of record, the label of record of the digests.
#   bash $H/submit.sh cpu_job.sbatch [options of sbatch] -- COMMAND [ARGUMENTS]
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
COUCH_PID=
CPU_LA=

cpu_job() {
  [ $# -gt 0 ] || { echo "usage: bash CausalDiagnosis/cluster_jobs/submit.sh cpu_job.sbatch [options of sbatch] -- COMMAND [ARGUMENTS]" >&2; return 2; }
  job_header "command $*"
  fork_python; check_data
  cd "$FORK" || return 2
  runs_root
  if [ "${CPU_CARD_OF_RECORD:-}" = 1 ]; then check_card required; fi
  DATA_FILE=$(data_file)
  export ROOT CLONES ENVS DATA LOGS ENDPOINTS SUITES PLATFORM DIGESTS DATA_FILE
  mkdir -p "$DIGESTS"                                 # the label of the digests is written there by a command of this job
  if [ "${CPU_COUCHDB:-}" = 1 ]; then
    CPU_LA=$(node_local_dir)/cpu_job_$$
    mkdir -p "$CPU_LA"
    start_couchdb "$CPU_LA"
  fi
  if [ "${CPU_TABLES:-}" = 1 ]; then grid_tables check || return 4; fi
  "$@"
  local status=$?
  echo "CD_CPU_JOB_DONE $status"
  return "$status"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  trap 'if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; fi; [ -z "$CPU_LA" ] || rm -rf "$CPU_LA"' EXIT
  cpu_job "$@"
  exit $?
fi
