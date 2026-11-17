#!/bin/bash
# The judge after a grid, the RUN_AFTER_SERVE script of a serving job that serves the judge alone (MODEL and JUDGE_MODEL both
# name the judge folder, so that the server takes JUDGE_ALONE_SGLANG_EXTRA). JUDGE_FOLDERS names the grid folders one by one.
# Each pass of judge_finished_runs.py judges the episodes of those folders that hold no judge record, with JUDGE_WORKERS processes (1
# when unset) under JUDGE_PROTOCOL (paper when unset), and the passes repeat until one judges no episode or JUDGE_PASSES
# passes (20 when unset) have run. One more pass judges again the records that hold a failed call. The static line of a grid
# is judged by the same calls, in the static mode of the judge. A second submission of the same folders goes on where the
# first stopped, since only an episode without a record is judged. With JUDGE_SMOKE=<grid folder> the job judges copies of
# the first episode of at most two runs of that folder under the smoke folder instead, and changes no run folder. With
# JUDGE_RECORD_ONLY=1 the job keeps the endpoint record of the judge in PLATFORM, which the platform record needs, and judges
# nothing. Before it judges the grid folders, the job counts the replies of the judge that hold thinking under the request of
# the judge (episode_judge.py --thinking, JUDGE_THINKING) and stops with status 2 on a count above 0; the smoke checks the same count.
# Every judging reads the endpoint file of the judge through HYD_JUDGE_ENDPOINT_FILE, so that each record names the serving
# identity of the judge.
#   RUN_AFTER_SERVE=$H/3_run/run_judge.sh MODEL=$J JUDGE_MODEL=$J bash $H/submit.sh serve_model.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
JUDGE_WORKERS=${JUDGE_WORKERS:-1}
JUDGE_PROTOCOL=${JUDGE_PROTOCOL:-paper}
JUDGE_PASSES=${JUDGE_PASSES:-20}

# the calls of the judge records of the episode folders named, and the calls that failed, for the smoke of the judge; what is left
# to judge in the grid folders is counted by judge_finished_runs.py --count and --names (judge_count), the selection of the pass itself
RECORDS='import json, sys
from pathlib import Path
records = [Path(e) / "hidden" / "judge.json" for e in sys.argv[1:]]
calls = [call for path in records if path.is_file() for call in json.loads(path.read_text(encoding="utf-8")).get("calls") or []]
print(len(calls), sum(1 for call in calls if call.get("error") is not None))'

judge_facts() {
  # The endpoint of the judge, its model list, the token pool of its server and the Python of the node, and the endpoint
  # file kept in PLATFORM for the platform record of the registered runs.
  local tree
  echo "### the endpoint file $ENDPOINT_FILE"; cat "$ENDPOINT_FILE"
  echo "### the models of the endpoint"; key_curl "$BASE_URL/models"; echo
  echo "### the token pool of the server"
  grep -m 3 -E "max_total_num_tokens|context_len|max_total_tokens" "$LOGS/sglang-${SLURM_JOB_ID:-manual}.log" \
    || echo "no line of the token pool in $LOGS/sglang-${SLURM_JOB_ID:-manual}.log"
  echo "python3 of the node: $NODE_PYTHON"
  tree=$(tree_id) || return 2
  mkdir -p "$PLATFORM"
  cp "$ENDPOINT_FILE" "$PLATFORM/judge_endpoint_$tree.txt"
}

judge_thinking() {
  # The replies of the judge that hold thinking under the request of the judge, as episode_judge.py --thinking counts them: the chat
  # template decides whether the switch of episode_judge.JUDGE_REQUEST turns thinking off, so every judge job checks it on its judge,
  # and only a count of 0 passes.
  local thinking
  thinking=$(python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge --thinking --model "$SERVED_NAME" --base-url "$BASE_URL")
  echo "${thinking:-JUDGE_THINKING unknown}"
  [ "$thinking" = "JUDGE_THINKING 0" ]
}

judge_smoke() {
  # Copies of the first episode of at most two runs of the grid folder $1, judged under the smoke folder.
  local smoke=$ROOT/runs/smoke/judge_${SLURM_JOB_ID:-manual} run episode copies=() start seconds counts
  for run in "$1"/*/; do
    [ -f "${run}run.json" ] || continue
    episode=
    for candidate in "$run"*/; do [ -f "${candidate}released.json" ] && { episode=${candidate%/}; break; }; done
    [ -n "$episode" ] || continue
    mkdir -p "$smoke/$(basename "$run")"
    cp "${run}run.json" "$smoke/$(basename "$run")/"
    cp -R "$episode" "$smoke/$(basename "$run")/"
    copies+=("$smoke/$(basename "$run")/$(basename "$episode")")
    [ ${#copies[@]} -lt 2 ] || break
  done
  [ ${#copies[@]} -gt 0 ] || { echo "JUDGE_SMOKE no episode under $1" >&2; return 1; }
  start=$(date +%s)
  python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge --model "$SERVED_NAME" --base-url "$BASE_URL" --protocol "$JUDGE_PROTOCOL" "${copies[@]}"
  seconds=$(( $(date +%s) - start ))
  read -r calls failed <<< "$(python -c "$RECORDS" "${copies[@]}")"
  echo "JUDGE_SMOKE episodes ${#copies[@]} calls ${calls:-0} failed ${failed:-1} seconds $seconds"
  judge_thinking && [ "${failed:-1}" = 0 ]
}

judge_pass() {
  # One pass of judge_finished_runs.py over the folders with the options of the job: prints the number of episodes it judged and ends
  # with the status of judge_finished_runs.py, so that a pass that fails fails the job.
  local out status
  out=$(mktemp)
  # shellcheck disable=SC2086
  python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.judge_finished_runs $JUDGE_FOLDERS --judge-model "$SERVED_NAME" --judge-endpoint "$BASE_URL" \
    --protocol "$JUDGE_PROTOCOL" --workers "$JUDGE_WORKERS" "$@" > "$out"
  status=$?
  cat "$out" >> "$JUDGE_LOG"
  sed -n 's/.*: judged \([0-9][0-9]*\) episode.*/\1/p' "$out" | awk '{ n += $1 } END { print n + 0 }'
  rm -f "$out"
  return "$status"
}

judge_count() {
  # what a pass with the options of the job leaves in the folders, judge_finished_runs.py --count (or --names with that option): the episodes
  # left, judged and with a failed call, the calls and the failed ones, the episodes no pass can judge and those of refused runs
  # shellcheck disable=SC2086
  python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.judge_finished_runs $JUDGE_FOLDERS --judge-model "$SERVED_NAME" --judge-endpoint "$BASE_URL" "$@"
}

judge_folders() {
  local pass judged start seconds left pace refused
  read -r left _ <<< "$(judge_count --count)"
  echo "JUDGE_PLAN episodes $left without a judge record in $JUDGE_FOLDERS"
  for pass in $(seq 1 "$JUDGE_PASSES"); do
    start=$(date +%s)
    judged=$(judge_pass) || { echo "JUDGE_PASS $pass: judge_finished_runs.py failed, see $JUDGE_LOG" >&2; return 1; }
    seconds=$(( $(date +%s) - start ))
    if [ "$pass" = 1 ]; then
      read -r left _ <<< "$(judge_count --count)"
      pace=0
      [ "$judged" = 0 ] || pace=$(( seconds / judged ))
      echo "JUDGE_PACE seconds per judged episode $pace, episodes left $left"
    fi
    echo "JUDGE_PASS $pass judged $judged"
    [ "$judged" != 0 ] || break
  done
  judge_pass --rejudge-failed > /dev/null || { echo "the pass of the failed calls: judge_finished_runs.py failed, see $JUDGE_LOG" >&2; return 1; }
  # an episode that a pass can judge and that is still without a record counts against the job as a failed call does, since a
  # table scores an unjudged episode false on every criterion; an episode that no pass can judge and the episodes of a run that
  # the pass refuses are counted and named apart
  read -r left judged failed _ _ not_judged refused <<< "$(judge_count --count)"
  echo "JUDGE_RECORDS judged $judged, with a failed call $failed"
  echo "JUDGE_LEFT $left"
  echo "JUDGE_FAILED_CALLS $failed"
  echo "JUDGE_NOT_JUDGED $not_judged"
  echo "JUDGE_REFUSED $refused"
  judge_count --names
  [ "$failed" = 0 ] && [ "$left" = 0 ]
}

main() {
  NODE_PYTHON=$(python3 --version 2>&1)
  fork_python
  cd "$FORK" || exit 2
  JUDGE_LOG=$LOGS/judge-${SLURM_JOB_ID:-manual}.log
  judge_facts || exit 2
  if [ "${JUDGE_RECORD_ONLY:-}" = 1 ]; then
    echo "JUDGE_RECORD_ONLY the judge endpoint record of this tree is kept in $PLATFORM, and nothing is judged"
    echo "CD_JUDGE_DONE"
    exit 0
  fi
  export HYD_JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE
  # the judge against the serving file of record: a judge job of record stops on a difference, any other job names it and judges on,
  # and judge_finished_runs.py refuses a run of record with a judge that differs either way
  local serving
  if ! serving=$(python -m CausalDiagnosis.experiments.run_verification.run_ledger serving --check "$ENDPOINT_FILE" --role judge --mode alone 2>&1); then
    if [ "${RUN_KIND:-}" = record ]; then
      printf '%s\n' "$serving" >&2
      echo "SERVING_DIFFERS the judge is not the one of the serving file of record, so a judge job of record judges nothing" >&2; exit 2
    fi
    echo "SERVING_DIFFERS judge, a job of kind ${RUN_KIND:-unnamed} judges on: $(printf '%s' "$serving" | tr '\n' ' ')"
  fi
  local status=0
  if [ -n "${JUDGE_SMOKE:-}" ]; then
    judge_smoke "$JUDGE_SMOKE" || status=1
  else
    [ -n "${JUDGE_FOLDERS:-}" ] || { echo "JUDGE_FOLDERS is empty: name the grid folders one by one" >&2; exit 2; }
    judge_thinking || { echo "the judge answers with thinking under the request of the judge, so the job judges no episode" >&2; exit 2; }
    judge_folders || status=1
  fi
  echo "CD_JUDGE_DONE"
  exit "$status"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main; fi
