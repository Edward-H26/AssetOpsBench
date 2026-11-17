#!/bin/bash
# RUN_AFTER_SERVE script of serve_model.sbatch for one agent on one row or a few grid lines (CausalDiagnosis/cluster_jobs/README.md): one CouchDB on
# the node's local disk for the job, then the lines of GRID_FILE through run_grid.sbatch against the served model. The judge
# scores the runs after the grid (CausalDiagnosis/cluster_jobs/README.md), and the reference agents play as a CPU job of their own unless
# GRID_REFERENCE=1. GRID_FILE names a file of the folder grids (local_*.txt is ignored by git), or GRID_PART a part and GRID_ROW
# one development row of grid_job_lists.py, whose grid file grid_job_lists.py of this tree prints into the job's local folder.
# GRID_STARTS (sampled, the test set, development, the development starts, seeds, which reads GRID_SEEDS, or training, the
# sampling runs of the training experiment), GRID_FRACTION (the share of the test set, DEFAULT_FRACTION of start_state_seeds.py when unset) or
# GRID_REPLICATES (whole replicates named by hand, one run of a line each),
# GRID_PARALLEL (1) and GRID_REFERENCE (0) are optional. A requeued job plays the pairs of lines and replicates that no
# finished run of its earlier attempts covers, and with GRID_PENDING=1 a job plays the pairs that no finished run of any grid
# folder of RUNS covers, as pending_grid_lines.py counts a finished run, so that no pair is played twice into one RUNS root. A
# registered job plays only on a node of NODES_OF_RECORD and the card of record when env.sh names them, every job with a
# served model opens with the three checks of opening_checks, a job whose card also serves the judge plays no line with the
# wall clock, and the script ends with the status of the runner.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
H=$HPC_DIR
runs_root
case "${GRID_FILE:+file}${GRID_PART:+part}${GRID_ROW:+row}" in
  file|part|row) ;;
  *) echo "name the lines with one of GRID_FILE (a grid file of $GRIDS, for example local_one.txt), GRID_PART and GRID_ROW" >&2; exit 2 ;;
esac
[ -z "${GRID_FILE:-}" ] || [ -f "$GRIDS/$GRID_FILE" ] || { echo "no grid file at $GRIDS/$GRID_FILE" >&2; exit 2; }
# a registered run plays only on a node of NODES_OF_RECORD and on the card of record, as the endpoint file of its serving job
# names the card, when env.sh names them
if [ "$RUNS_KIND" = record ] && [ -n "$NODES_OF_RECORD" ]; then
  case " $NODES_OF_RECORD " in
    *" $(hostname -s) "*) ;;
    *) echo "$(hostname -s) is no node of record of NODES_OF_RECORD in env.sh, so it plays no registered run" >&2; exit 2 ;;
  esac
fi
if [ "$RUNS_KIND" = record ] && [ -n "$GPU_OF_RECORD$GPU_MIB_OF_RECORD" ]; then
  require_gpu_of_record
  CARD=$(endpoint_card "${ENDPOINT_FILE:-}")
  card_of_record "${CARD%|*}" "${CARD#*|}" \
    || { echo "the endpoint file ${ENDPOINT_FILE:-} names the card ${CARD:-none}, not the card of record $GPU_OF_RECORD $GPU_MIB_OF_RECORD" >&2; exit 2; }
fi
fork_python; check_data
LA=$(node_local_dir)/run_lines_$$
mkdir -p "$LA"
COUCH_PID=
cleanup() {
  if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; fi
  rm -rf "$LA"
}
trap cleanup EXIT
if [ -z "${GRID_FILE:-}" ]; then
  GRID_FILE=$LA/grid_${GRID_PART:-$GRID_ROW}.txt
  python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists ${GRID_PART:+part "$GRID_PART"} ${GRID_ROW:+row "$GRID_ROW"} > "$GRID_FILE" \
    || { echo "grid_job_lists.py printed no grid file for ${GRID_PART:+part $GRID_PART}${GRID_ROW:+row $GRID_ROW}" >&2; exit 2; }
fi
case "$GRID_FILE" in /*) GRID_PATH=$GRID_FILE ;; *) GRID_PATH=$GRIDS/$GRID_FILE ;; esac
PLAY=$GRID_FILE
# the grid folders whose finished runs the job leaves out: a requeued job keeps its id, so the folder of the interrupted
# attempt is moved aside and the job plays the pairs that no finished run of an earlier attempt covers, and GRID_PENDING=1
# reads every grid folder of RUNS, for a part whose lines another part has played or a part submitted again
COVERED=()
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
  d=$RUNS/grid_$SLURM_JOB_ID
  [ -d "$d" ] && mv "$d" "${d}_attempt$SLURM_RESTART_COUNT" && echo "requeued job: moved $d aside as ${d}_attempt$SLURM_RESTART_COUNT"
  for a in "$d"_attempt*; do [ -d "$a" ] && COVERED+=("$a"); done
fi
if [ "${GRID_PENDING:-0}" = 1 ]; then
  COVERED=()
  for a in "$RUNS"/grid_*; do [ -d "$a" ] && COVERED+=("$a"); done
fi
if [ ${#COVERED[@]} -gt 0 ]; then
  PLAY=$LA/pending_lines.txt
  REPLICATED=()
  [ "${GRID_STARTS:-sampled}" != sampled ] || REPLICATED=(${GRID_FRACTION:+--fraction "$GRID_FRACTION"} ${GRID_REPLICATES:+--replicates $GRID_REPLICATES})
  # the lines name no model: run_grid.sbatch adds the served model after each line, so a finished run records SERVED_NAME; it adds
  # its seeds and GRID_EXTRA after each line as well, so the tool plans every line with the episodes that the job plays
  # shellcheck disable=SC2086
  python -m CausalDiagnosis.experiments.cluster_runs.pending_grid_lines --grid "$GRID_PATH" --runs "${COVERED[@]}" \
    --starts "${GRID_STARTS:-sampled}" ${REPLICATED[@]+"${REPLICATED[@]}"} ${SERVED_NAME:+--model "$SERVED_NAME"} \
    --seeds "${GRID_SEEDS:-3}" --first-seed "${GRID_FIRST_SEED:-0}" ${GRID_EXTRA:-} > "$PLAY" \
    || { echo "pending_grid_lines.py failed" >&2; exit 2; }
  echo "$(grep -c . "$PLAY") pending pairs of $GRID_FILE in $PLAY, after the finished runs of ${#COVERED[@]} grid folders"
fi
# a grid file without a line, or a requeued job with nothing pending, plays nothing
case "$PLAY" in /*) PLAY_PATH=$PLAY ;; *) PLAY_PATH=$GRIDS/$PLAY ;; esac
[ -r "$PLAY_PATH" ] || { echo "no file of lines at $PLAY_PATH" >&2; exit 2; }
# a line that is no comment and not blank is pending; one awk reads the file, since under pipefail a reader that closes a
# pipe early would make a long file read as one with nothing pending
if ! awk '!/^#/ && /[^[:space:]]/ { found = 1; exit } END { exit !found }' "$PLAY_PATH"; then
  echo "nothing pending in $PLAY"
  echo "CD_RUN_LINES_DONE $RUNS/grid_${SLURM_JOB_ID:-manual}"
  exit 0
fi
require_module couchdb3
# start_couchdb exports COUCHDB_URL, COUCHDB_USERNAME and COUCHDB_PASSWORD, which every line with --store couchdb reads
start_couchdb "$LA"
# A card that also serves the judge plays no line with the wall clock, since the judge's requests would share the card with an
# episode whose wall seconds count.
if [ -n "${CO_SERVE_JUDGE:-}" ] && [ -n "$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists clocked "$PLAY_PATH")" ]; then
  echo "WALL_CLOCK_WITH_JUDGE $PLAY holds lines with the wall clock, and this job serves the judge on its card: submit them without CO_SERVE_JUDGE" >&2
  exit 2
fi
opening_checks() {
  # The three checks that open every job with a served model, each a line of the log, before its first line plays:
  # 1. one scripted episode and the baseline policies on development starts, their digests (invariant_digests.py, the families
  #    scripted and logs) against the label of this clone that setup.sh wrote on the cluster; a difference or no label stops;
  # 2. the endpoint file of the agent model against the tracked serving file of record (run_ledger/ serving --check); a
  #    difference stops a registered job and is printed for a development job;
  # 3. one timed call of the served model, its seconds and the cycles it would charge at the thinking rate of every asset
  #    profile, printed only.
  local label status
  label=$DIGESTS/$(tree_id).json
  if [ ! -s "$label" ]; then
    echo "OPENING_DIGESTS_MISSING no digest label of this clone at $label: run bash \$H/0_setup/setup.sh" >&2; return 2
  fi
  ( cd "$FORK" && python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts "$(data_file)" --label clone --check "$label" \
      --families scripted,logs )
  status=$?
  [ "$status" = 0 ] || { echo "OPENING_DIGESTS_DIFFER status $status: this node computes other digests than the setup job of the clone" >&2; return 2; }
  echo "OPENING_DIGESTS_OK"
  ( cd "$FORK" && python -m CausalDiagnosis.experiments.run_verification.run_ledger serving --check "$ENDPOINT_FILE" --role agent )
  status=$?
  if [ "$status" != 0 ] && [ "$RUNS_KIND" = record ]; then
    echo "SERVING_DIFFERS the served model is not the one of the serving file of record, so the job plays no registered line" >&2; return 2
  fi
  python - "$(endpoint_url "$ENDPOINT_FILE")" "$SERVED_NAME" "$FORK" <<'PY'
import glob, json, os, sys, time, urllib.request
base, model, fork = sys.argv[1:4]
body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Reply with the word ready."}]}).encode()
request = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={
    "Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})})
started = time.monotonic()
try:
    urllib.request.urlopen(request, timeout=600).read()
    outcome = "answered"
except Exception as exc:  # noqa: BLE001 - the opening call prints what it met and stops nothing
    outcome = f"failed {type(exc).__name__}"
seconds = time.monotonic() - started
rates = []
for path in sorted(glob.glob(os.path.join(fork, "CausalDiagnosis", "datasets", "*", "asset.json"))):
    clock = json.load(open(path)).get("clock") or {}
    if clock.get("think_seconds_per_cycle"):
        rates.append(f"{os.path.basename(os.path.dirname(path))} {seconds / clock['think_seconds_per_cycle']:.3f}")
print(f"OPENING_CALL {outcome} seconds {seconds:.3f} cycles " + ("; ".join(rates) or "none"))
PY
}
# a job with a served model, whose serving job exports ENDPOINT_FILE, opens with the checks
if [ -n "${ENDPOINT_FILE:-}" ]; then opening_checks || exit 2; fi
echo "### $(hostname) $(date) lines of $GRID_FILE on $SERVED_NAME into $RUNS/grid_${SLURM_JOB_ID:-manual}"
# the runner plays in the background, so that the warning signal of the time limit reaches it at once
HPC_DIR=$H GRID_FILE=$PLAY GRID_STARTS=${GRID_STARTS:-sampled} GRID_FRACTION=${GRID_FRACTION:-} GRID_REPLICATES=${GRID_REPLICATES:-} \
  GRID_PARALLEL=${GRID_PARALLEL:-1} GRID_REFERENCE=${GRID_REFERENCE:-0} bash "$H/3_run/run_grid.sbatch" &
GRID_PID=$!
FORWARDED=
trap 'FORWARDED=1; kill -TERM "$GRID_PID" 2>/dev/null' TERM
wait "$GRID_PID"
STATUS=$?
if [ -n "$FORWARDED" ]; then wait "$GRID_PID" 2>/dev/null; AGAIN=$?; [ "$AGAIN" = 127 ] || STATUS=$AGAIN; fi
echo "RUN_GRID_EXIT $STATUS"
[ "$STATUS" != 0 ] || echo "CD_RUN_LINES_DONE $RUNS/grid_${SLURM_JOB_ID:-manual}"
exit "$STATUS"
