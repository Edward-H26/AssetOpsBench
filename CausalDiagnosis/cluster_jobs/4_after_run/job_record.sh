#!/bin/bash
# Writes the accounting record of the job of every named run folder, grid_<job id> or gepa_online_<job id> and the folders
# of their earlier attempts, into its job_record.txt, in the login shell after the job, so that the record also holds what
# happened after the job's own last write, a suspension or the end at the time limit among them. It copies the logs of the
# job from LOGS into the folder job_logs of the run folder, every file <name>-<job id>.<ext>: the Slurm log, and for a
# serving job the logs of its servers and the file of its GPU sampler, so that the logs travel with the runs.
#   bash CausalDiagnosis/cluster_jobs/4_after_run/job_record.sh FOLDER ...
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh

copy_job_logs() {
  # the logs of the job $2 into $1/job_logs, or a line that names the job when LOGS holds none
  local logs=() f
  for f in "$LOGS"/*-"$2".*; do [ -f "$f" ] && logs+=("$f"); done
  [ ${#logs[@]} -gt 0 ] || { echo "no log of job $2 under $LOGS" >&2; return 2; }
  mkdir -p "$1/job_logs" && cp -p "${logs[@]}" "$1/job_logs/"
}

record_folders() {
  local folder name job status=0
  for folder in "$@"; do
    name=$(basename "$folder")
    case "$name" in
      grid_*) job=${name#grid_} ;;
      gepa_online_*) job=${name#gepa_online_} ;;
      *) job= ;;
    esac
    job=${job%%_*}
    case "$job" in
      ''|*[!0-9]*) echo "$folder is no folder grid_<job id> or gepa_online_<job id>" >&2; status=2; continue ;;
    esac
    [ -d "$folder" ] || { echo "no folder $folder" >&2; status=2; continue; }
    write_job_record "$folder" "$job"
    copy_job_logs "$folder" "$job" || status=2
  done
  return "$status"
}

main() {
  [ -f "$HERE/env.sh" ] || { echo "no env.sh in $HERE: copy env.example.sh to env.sh and set ROOT" >&2; exit 2; }
  [ $# -gt 0 ] || { echo "usage: bash CausalDiagnosis/cluster_jobs/4_after_run/job_record.sh FOLDER ..." >&2; exit 2; }
  HPC_DIR=$HERE
  source "$HERE/env.sh"
  [ -n "${CD_FUNCTIONS:-}" ] || { echo "env.sh in $HERE predates functions.sh: copy env.example.sh to env.sh once more and set your values in it" >&2; exit 2; }
  record_folders "$@"
  exit $?
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
