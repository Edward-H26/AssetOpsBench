#!/bin/bash
# The one setup command of a clone: it submits the chain of seven jobs that a first run needs, through submit.sh, and reads
# their state back.
#   bash $H/0_setup/setup.sh            submits the chain, prints one SUBMITTED line per job, then SETUP_CHAIN <job ids>
#   bash $H/0_setup/setup.sh status     one line per job: its CD_*_DONE line, PENDING or RUNNING, or the last line of a failed log;
#                               SETUP_OK once all seven DONE lines exist
# The chain: the image and the environment in parallel; then the agent model, the judge model and the data of record after the
# environment, whose uv the two downloads fetch with, the tables after the data, and the digest label of the clone after the
# tables (each afterok on the one before). Every job with a dependency is submitted with --kill-on-invalid-dep=yes, so a
# failed job takes the jobs after it out of the queue instead of leaving them pending for ever with the reason
# DependencyNeverSatisfied; status names each of them FAILED, and a second setup.sh submits again every job whose DONE line
# this clone has not printed, and skips the others. The jobs of a clone are recorded in ROOT/setup/<tree id>.jobs, one line
# per job (step, job id, log), since the logs of every clone share LOGS.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh

STEPS=(image environment agent_model judge_model data tables digests)   # each after the step it waits for

step_script() {
  case "$1" in
    image) echo pull_image.sbatch ;;
    agent_model|judge_model) echo download_model.sbatch ;;
    environment) echo setup_env.sbatch ;;
    data) echo fetch_data.sbatch ;;
    tables) echo precompute_tables.sbatch ;;
    digests) echo cpu_job.sbatch ;;
  esac
}

step_done_line() {
  # The line whose presence in a log marks the step done.
  case "$1" in
    image) echo CD_IMAGE_DONE ;;
    agent_model|judge_model) echo CD_DOWNLOAD_DONE ;;
    environment) echo CD_SETUP_DONE ;;
    data) echo CD_DATA_DONE ;;
    tables) echo CD_TABLES_DONE ;;
    digests) echo CD_DIGESTS_DONE ;;
  esac
}

step_after() {
  # The step whose job a step waits for, or nothing.
  case "$1" in
    agent_model|judge_model|data) echo environment ;;
    tables) echo data ;;
    digests) echo tables ;;
  esac
}

judge_hub_id() {
  # The judge's Hugging Face id from the tracked serving file, read with the standard library alone, since the environment of
  # the fork that run_ledger/ needs is built by this very chain.
  python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["judge"]["hub_id"])' \
    "$FORK/CausalDiagnosis/experiments/table_definitions/serving_of_record.json"
}

recorded() {
  # recorded STEP FIELD: the job id (field 2) or the log (field 3) of the last job of STEP in the record of this clone.
  [ -f "$RECORD" ] || return 0
  awk -v step="$1" -v field="$2" '$1 == step { value = $field } END { if (value != "") print value }' "$RECORD"
}

step_is_done() {
  local log
  log=$(recorded "$1" 3)
  [ -n "$log" ] && [ -f "$log" ] && grep -q "^$(step_done_line "$1")" "$log"
}

queue_state() {
  # The state of the job $1 in the queue (PENDING, RUNNING and so on), or nothing once it has left the queue.
  [ -n "$1" ] || return 0
  squeue -h -j "$1" -o %T 2>/dev/null | head -n 1
}

submit_step() {
  # Submits STEP with its dependency, records it and prints the SUBMITTED line of submit.sh; the job id lands in JOB.
  local step=$1 dependency=$2 line options=()
  # the scheduler's own option ends a job whose dependency can never be met, so no stranded job blocks a second setup.sh
  [ -z "$dependency" ] || options+=("--dependency=afterok:$dependency" --kill-on-invalid-dep=yes)
  case "$step" in
    agent_model) options+=(-J cd-download-agent) ;;
    judge_model) options+=(-J cd-download-judge "--export=ALL,MODEL=$JUDGE_HUB_ID") ;;
    digests) options+=(-J cd-digests) ;;
  esac
  if [ "$step" = digests ]; then
    line=$(bash "$HERE/submit.sh" cpu_job.sbatch "${options[@]}" -- bash -c \
      "python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts \"\$DATA_FILE\" --label clone --write \"\$DIGESTS/$ID.json\" && echo CD_DIGESTS_DONE")
  else
    line=$(bash "$HERE/submit.sh" "$(step_script "$step")" ${options[@]+"${options[@]}"})
  fi || { echo "the submission of the step $step failed: $line" >&2; return 3; }
  echo "$line"
  JOB=$(awk '$1 == "SUBMITTED" { print $3 }' <<< "$line")
  [ -n "$JOB" ] || { echo "submit.sh printed no SUBMITTED line for the step $step" >&2; return 3; }
  echo "$step $JOB $(awk '$1 == "SUBMITTED" { print $4 }' <<< "$line")" >> "$RECORD"
}

setup_chain() {
  local step job state after ids=() step_jobs=()
  for step in "${STEPS[@]}"; do
    job=$(recorded "$step" 2)
    state=$(queue_state "$job")
    if [ -n "$state" ]; then
      echo "the job $job of the step $step is still in the queue ($state): read bash $HERE/0_setup/setup.sh status" >&2
      return 4
    fi
  done
  JUDGE_HUB_ID=$(judge_hub_id) || { echo "no judge hub_id in the serving file of $FORK" >&2; return 2; }
  for step in "${STEPS[@]}"; do
    if step_is_done "$step"; then
      echo "SKIPPED $step $(recorded "$step" 2), its DONE line exists"
      continue
    fi
    after=$(step_after "$step")
    job=""
    if [ -n "$after" ]; then job=${step_jobs[$(step_index "$after")]:-}; fi
    submit_step "$step" "$job" || return 3
    step_jobs[$(step_index "$step")]=$JOB
    ids+=("$JOB")
  done
  echo "SETUP_CHAIN ${ids[*]:-}" | tee -a "$ROOT/setup/$ID.ids"
}

step_index() {
  local i
  for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && { echo "$i"; return; }; done
}

setup_status() {
  local step job log state done_count=0
  [ -f "$RECORD" ] || { echo "no setup chain of this clone: run bash $HERE/0_setup/setup.sh" >&2; return 2; }
  for step in "${STEPS[@]}"; do
    job=$(recorded "$step" 2)
    log=$(recorded "$step" 3)
    if step_is_done "$step"; then
      echo "$step $job $(grep -m 1 "^$(step_done_line "$step")" "$log")"
      done_count=$((done_count + 1))
    elif [ -z "$job" ]; then
      echo "$step never submitted"
    elif state=$(queue_state "$job") && [ -n "$state" ]; then
      echo "$step $job $state"
    elif [ -f "$log" ]; then
      echo "$step $job FAILED $(tail -n 1 "$log")"
    else
      echo "$step $job FAILED no log at $log"
    fi
  done
  if [ "$done_count" = "${#STEPS[@]}" ]; then echo "SETUP_OK"; return 0; fi
  return 1
}

main() {
  [ -f "$HERE/env.sh" ] || { echo "no env.sh in $HERE: copy env.example.sh to env.sh and set ROOT" >&2; exit 2; }
  HPC_DIR=$HERE
  source "$HERE/env.sh"
  ID=$(tree_id) || exit 2
  mkdir -p "$ROOT/setup"
  RECORD=$ROOT/setup/$ID.jobs
  case "${1:-}" in
    '') setup_chain ;;
    status) setup_status ;;
    *) echo "usage: bash CausalDiagnosis/cluster_jobs/0_setup/setup.sh [status]" >&2; exit 2 ;;
  esac
  exit $?
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
