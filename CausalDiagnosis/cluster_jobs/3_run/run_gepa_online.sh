#!/bin/bash
# RUN_AFTER_SERVE script of serve_model.sbatch for the online loops of GEPA of the training experiment (CausalDiagnosis/cluster_jobs/README.md): one
# CouchDB on the node's local disk for the job, then gepa_prompt_optimization/ for every agent of
# GEPA_AGENTS, fold of GEPA_FOLDS (every registered fold by default) and card of GEPA_CARDS on the preset GEPA_CONFIG, the
# agent without training or, with GEPA_VARIANT=ft, on the filtered adapter of the fold, which needs a server started with
# SERVE_ADAPTERS=1, with the judge as the reflection model, served by a job of its own with SERVE_REACH=nodes and never beside
# the agent model, and then the export of the final block of every finished loop to $HYD_TRAINING/blocks. REFLECTION_JOB names
# that job: its endpoint file is read by the job id, and the serving job of the loops takes the key of that job. On the base weights the
# cards are the verifier's and the score-only control by default, on the adapter the verifier's alone.
# The budget of the stream and the proposals of a round are the registered numbers of configs/training/<asset_id>/rft.json, which
# GEPA_EPISODES and GEPA_CHILDREN override only for a trial. GEPA_SPLIT=development plays the loops of a trial of the training
# path on the development split, with HYD_TRAINING and RUNS of the trial, and unset the registered split. GRID_PARALLEL names
# the loops in flight on the card, every loop at once when unset, and a registered job names it. On a preset whose clock charges
# the wall seconds of thinking, the preset of training among them, the job keeps one loop alive and hands it GRID_PARALLEL=1,
# since the driver registers a loop on such a preset only at the registered runs in flight (design.RUNS_IN_FLIGHT), so those
# loops play one at a time, or in jobs of their own that name them with GEPA_AGENTS, GEPA_FOLDS and GEPA_CARDS.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
H=$HPC_DIR
[ -n "$JUDGE_MODEL" ] || { echo "JUDGE_MODEL is empty: the reflection model is the judge of its own serving job (CausalDiagnosis/cluster_jobs/README.md)" >&2; exit 2; }
[ -n "${REFLECTION_JOB:-}" ] || { echo "name the job of the reflection model, REFLECTION_JOB=<job id>" >&2; exit 2; }
fork_python; check_data
runs_root
if [ "$RUNS_KIND" = record ] && [ -z "${GRID_PARALLEL:-}" ]; then
  echo "a registered job names its loops in flight, GRID_PARALLEL=<n>, the registered number of the preset of the loops" >&2; exit 2
fi
LA=$(node_local_dir)/gepa_online_$$
mkdir -p "$LA"
COUCH_PID=
cleanup() {
  if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; wait "$COUCH_PID" 2>/dev/null; fi
  rm -rf "$LA"
}
trap cleanup EXIT
reflection_url() {
  # The URL of the endpoint file of the job $1 once its model list answers, while that job is queued or running.
  local file=$ENDPOINTS/$1.txt url
  while :; do
    if [ -s "$file" ]; then
      url=$(head -n 1 "$file")
      key_curl "$url/models" >/dev/null 2>&1 && { echo "$url"; return 0; }
    fi
    [ -n "$(squeue -h -j "$1" -o %T 2>/dev/null)" ] || { echo "the job $1 of the reflection model is neither queued nor running" >&2; return 1; }
    sleep 10
  done
}
require_module couchdb3
start_couchdb "$LA"
[ -n "${BASE_URL:-}" ] || BASE_URL=$(endpoint_url "${ENDPOINT_FILE:-$ENDPOINTS/$SERVED_NAME.txt}") || exit 3
JUDGE_BASE_URL=$(reflection_url "$REFLECTION_JOB") || exit 3
OUT=$RUNS/gepa_online_${SLURM_JOB_ID:-manual}
# a requeued job keeps its id and its folder: every loop starts with --resume, which takes up its unfinished loop there and plays a
# finished one never again
mkdir -p "$OUT"
cd "$FORK" || exit 2
VARIANT=${GEPA_VARIANT:-as_released}
if [ "$VARIANT" = as_released ]; then CARDS=${GEPA_CARDS:-verifier score_only}; else CARDS=${GEPA_CARDS:-verifier}; fi
# the folds come from the registered split unless GEPA_FOLDS names some, so that no fold list is typed here
FOLDS=${GEPA_FOLDS:-$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists folds)} || exit 2
[ -n "$FOLDS" ] || { echo "no fold to loop over: grid_job_lists.py folds printed none and GEPA_FOLDS is empty" >&2; exit 2; }
# one loop alive on a preset whose clock charges the wall seconds of thinking, handed GRID_PARALLEL=1 (design.RUNS_IN_FLIGHT),
# the clock read by grid_job_lists.py from a grid file of one line, so that the shell composes no configuration, and the settings
# line names the number the job was given beside the loops in flight
echo "--config ${GEPA_CONFIG:-deterioration}" > "$LA/preset.txt"
WALL=$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists clocked "$LA/preset.txt") || exit 2
JOB_PARALLEL=${GRID_PARALLEL:-all}
[ -z "$WALL" ] || export GRID_PARALLEL=1
echo "### $(hostname) $(date) online loops of GEPA on $SERVED_NAME at $BASE_URL, variant $VARIANT, reflection $JUDGE_SERVED_NAME at $JUDGE_BASE_URL, loops in flight ${GRID_PARALLEL:-all}${WALL:+ on the wall clock, where the job named $JOB_PARALLEL}, into $OUT"
# GRID_PARALLEL loops in flight against the one CouchDB, since every episode names its databases after a hash of its folder,
# every loop at once when it is unset
PIDS=
for AGENT in ${GEPA_AGENTS:-plan_execute agent_as_tool causal}; do
  for FOLD in $FOLDS; do
    for CARD in $CARDS; do
      while [ -n "${GRID_PARALLEL:-}" ] && [ "$(jobs -rp | wc -l)" -ge "$GRID_PARALLEL" ]; do sleep 1; done
      python -m CausalDiagnosis.training.gepa_prompt_optimization --model "$SERVED_NAME" --base-url "$BASE_URL" --reflection-model "$JUDGE_SERVED_NAME" \
        --reflection-base-url "$JUDGE_BASE_URL" --agent "$AGENT" --training-variant "$VARIANT" --training-fold "$FOLD" --card "$CARD" \
        --config "${GEPA_CONFIG:-deterioration}" --servers aob --store couchdb ${GEPA_EPISODES:+--max-episodes $GEPA_EPISODES} \
        ${GEPA_CHILDREN:+--children $GEPA_CHILDREN} ${GEPA_SPLIT:+--training-split $GEPA_SPLIT} --out "$OUT" --resume \
        > "$OUT/${AGENT}_f${FOLD}_$CARD.log" 2>&1 &
      PIDS="$PIDS $!"
    done
  done
done
# a loop that failed, a reflection endpoint that stopped answering among them, sets the exit status, and its folder waits for --resume
STATUS=0
for PID in $PIDS; do
  wait "$PID" || STATUS=5
done
# the job record that the check of held jobs reads, as a grid job writes it
write_job_record "$OUT"
# the final block of every finished loop, which the gepa, gepa_score_only and ft_gepa variants play
for RUN in "$OUT"/*_gepa_online_*/; do
  [ -d "$RUN" ] || continue
  python -m CausalDiagnosis.training.gepa_prompt_optimization export "$RUN" || STATUS=4
done
echo "CD_GEPA_ONLINE_DONE $OUT"
exit $STATUS
