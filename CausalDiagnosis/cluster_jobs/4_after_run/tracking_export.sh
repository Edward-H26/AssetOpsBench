#!/bin/bash
# Sends finished run folders, the grid folder of a job or the checkpoint folder of a training to Weights and Biases, in the
# login shell after the job, where the person's key stays: no compute node holds it, and no job sends anything while it plays.
# It reads from env.sh WANDB_BASE_URL, the address of the service's API on which the account lives, WANDB_ENTITY, the team or
# user on the service, WANDB_PROJECT, the project of the runs of record and the only one that is ever opened, and
# WANDB_PROJECT_DEVELOPMENT, the project of every other run. It runs the command with the Python of the lock of this tree and
# starts the client through the uv of UV_BIN with the one version of the client that
# CausalDiagnosis/performance/tracking_client.py pins, the cache of uv under ROOT, since the home filesystem has a small quota.
# --login runs the login of that client once for the declared address, which asks for the key and keeps it in the file .netrc
# of the home folder, or in the file that NETRC names, and WANDB_API_KEY in the environment serves as well; both stop while that
# file is open to group or others. --dry-run prints the whole payload and sends nothing, the first thing to run
# (CausalDiagnosis/performance/README.md).
#   bash CausalDiagnosis/cluster_jobs/4_after_run/tracking_export.sh --login
#   bash CausalDiagnosis/cluster_jobs/4_after_run/tracking_export.sh --dry-run FOLDER ...
#   bash CausalDiagnosis/cluster_jobs/4_after_run/tracking_export.sh FOLDER ...
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh

main() {
  [ -f "$HERE/env.sh" ] || { echo "no env.sh in $HERE: copy env.example.sh to env.sh and set ROOT" >&2; exit 2; }
  HPC_DIR=$HERE
  source "$HERE/env.sh"
  [ -n "${CD_FUNCTIONS:-}" ] || { echo "env.sh in $HERE predates functions.sh: copy env.example.sh to env.sh once more and set your values in it" >&2; exit 2; }
  fork_python
  export WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_PROJECT_DEVELOPMENT UV_BIN UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/tools/uv-cache}
  # -P keeps the folder the person stands in off the import path, where a folder of CausalDiagnosis/ would hide a package
  python -P -m CausalDiagnosis performance tracking-export "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
