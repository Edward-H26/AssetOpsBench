#!/bin/bash
#SBATCH -J cd-grid
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --signal=B:TERM@180
# The grid runner: one run of run_episodes/ per line of the grid file, and per replicate of the test set, on the served model with
# the agent the line names (the Plan-Execute agent of AssetOpsBench when it names none), GRID_SEEDS seeds from GRID_FIRST_SEED, judged by
# JUDGE_MODEL when it is set, which the run order leaves unset, since the judge scores the runs after the grid with judge_finished_runs.py. The
# reference agents, the reference policies of experiment_tables.py and the scripted agent of the wiring check, run once per preset that the grid
# names and per replicate without a model. It runs inside a serving job through run_lines.sh, which starts the CouchDB that every line
# with --store couchdb needs. With GRID_REFERENCE=only it plays the reference agents alone, which need no served model, so it is
# submitted as a CPU job, which starts CouchDB itself when GRID_EXTRA names that store. A grid line that names a reference agent
# plays without the model too, so the pending reference pairs that pending_grid_lines.py prints play as a CPU job with GRID_REFERENCE=0.
# The runs go to the folder grid_<job id> under RUNS, the folder of the kind of the tree (env.sh, runs_root). At the end the job prints
# one CALL_LATENCY line per run folder, the count, the total, the median, the p90 and the longest seconds of its model calls, keeps the
# record of its own job beside the run folders and names the runs of a suspended or requeued job (CausalDiagnosis/cluster_jobs/README.md), and a job that the
# time limit ends does the same before it stops. A unit or a reference run whose driver fails fails the job, and one that stopped
# is named apart.
# A job plays its runs in two phases (play_scheduled): first every run on a configuration that charges no wall seconds of
# thinking, GRID_PARALLEL at once, then, after the last of them has ended, every run that charges wall seconds, one at a time,
# so that no other episode of the job shares the card with it; no filler runs. in_flight.tsv in the grid folder records
# every start and end of a run, which check_grid_complete.py reads. After a model run stops with status 75 (harness errors) the job
# probes the endpoint once and ends with SERVICE_DOWN and status 6 when the server does not answer. A failed overlap check at
# the end, a run alive in an episode on the wall clock or an unclean flush, ends the job with status 7.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
job_header "grid ${GRID_FILE:-grid_agents.txt}, runs in flight ${GRID_PARALLEL:-1}"
require_model; fork_python; check_data
[ -n "${RUNS:-}" ] || runs_root
# GRID_FILE names the grid to run, a file of the folder grids, grid_agents.txt by default, the rows of the result tables and the
# supporting ablations of the paper, or the absolute path of the pending lines that run_lines.sh writes for a requeued job.
case "${GRID_FILE:-grid_agents.txt}" in /*) GRID_PATH=$GRID_FILE ;; *) GRID_PATH=$GRIDS/${GRID_FILE:-grid_agents.txt} ;; esac
mapfile -t LINES < <(grep -v '^#' "$GRID_PATH" | sed '/^[[:space:]]*$/d')
# The run ledgers (CausalDiagnosis/experiments/run_verification/run_ledger/) copy the grid file, the grid line and the endpoint files of the
# serving jobs, whose lines name the weights commit and the engine, and mark a file whose URL is not the run's endpoint.
HYD_ENDPOINT_FILE=$(readlink -f "${ENDPOINT_FILE:-$ENDPOINTS/$SERVED_NAME.txt}" 2>/dev/null)
HYD_JUDGE_ENDPOINT_FILE=
[ -z "${JUDGE_SERVED_NAME:-}" ] || HYD_JUDGE_ENDPOINT_FILE=$(readlink -f "$ENDPOINTS/$JUDGE_SERVED_NAME.txt" 2>/dev/null)
export HYD_GRID_FILE=$GRID_PATH HYD_ENDPOINT_FILE HYD_JUDGE_ENDPOINT_FILE
OUT=$RUNS/grid_${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
cd "$FORK"
# GRID_STARTS=sampled, the default, plays the test set of the paper, every start combination the config allows at five start
# states of its block, one run of 47 episodes per replicate that GRID_FRACTION or GRID_REPLICATES names (run_episodes/
# --replicate, a cut look of replicate 1 below one replicate), paired across lines, and GRID_SEEDS is not read. GRID_STARTS=development plays the development starts,
# GRID_STARTS=seeds plays GRID_SEEDS plain seeds (three by default) from GRID_FIRST_SEED, one random start each,
# GRID_STARTS=combinations plays every combination under its GRID_SEEDS smallest seeds, one run per line, and
# GRID_STARTS=training plays the training start states of a sampling line of the training experiment, one run per line.
# Which starts a run may play is the driver's to decide.
SEEDS="--seeds ${GRID_SEEDS:-3} --first-seed ${GRID_FIRST_SEED:-0} --starts ${GRID_STARTS:-sampled}"
# the replicates of the test set and the option of a cut look come from GRID_FRACTION or GRID_REPLICATES by the one rule of
# pending_grid_lines.py (grid_job_lists.py share), so a job and the job that continues it count the same pairs
FRACTION_OPTION=
if [ "${GRID_STARTS:-sampled}" = sampled ]; then
  SHARE=$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists share "${GRID_FRACTION:-}" "${GRID_REPLICATES:-}") || exit 2
  REPLICATES=$(sed -n 1p <<< "$SHARE"); FRACTION_OPTION=$(sed -n 2p <<< "$SHARE")
else REPLICATES=none; fi
replicate_option() { [ "$1" = none ] || echo "--replicate $1${FRACTION_OPTION:+ $FRACTION_OPTION}"; }
# The agents of the wiring check, read by grid_job_lists.py: one run per preset proves the wiring, so on the test set such an agent
# plays the first replicate alone, as pending_grid_lines.py counts its pairs, and on another start set its one run
WIRING_AGENTS=$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists wiring)
wiring_skipped() {
  # whether the agent $1 on the replicate $2 is a run of the wiring check after the first replicate of the test set
  case " $WIRING_AGENTS " in *" $1 "*) [ "$2" != none ] && [ "$2" != 1 ] ;; *) return 1 ;; esac
}
line_agent() {
  # the agent that the grid line $1 names after --agent, or nothing
  local words k
  read -r -a words <<< "$1"
  for k in "${!words[@]}"; do
    if [ "${words[$k]}" = --agent ]; then echo "${words[$((k + 1))]:-}"; return; fi
  done
}
# The runs of the grid, one per line and replicate in replicate order, so that a job cut short at the time limit holds the
# first replicate of every line, the test set with one episode per start, before any second one, and pending_grid_lines.py lists
# the rest. A line
# that names its --replicate, as pending_grid_lines.py prints the pairs a grid still lacks, plays that replicate alone.
UNITS=()
for r in $REPLICATES; do
  for i in "${!LINES[@]}"; do
    case " ${LINES[$i]} " in
      *" --replicate "*) [ "$r" = "${REPLICATES%% *}" ] && UNITS+=("$i none") ;;
      *) wiring_skipped "$(line_agent "${LINES[$i]}")" "$r" || UNITS+=("$i $r") ;;
    esac
  done
done
# GRID_REFERENCE_CONFIGS lists the presets and clock modes the reference agents play, one token per preset with an
# optional clock mode after a colon (real_time:instant); the default is every preset that the lines of the grid file name,
# read by grid_job_lists.py, so the reference runs follow the grid. The agents are the reference policies of experiment_tables.py and the
# scripted agent of the wiring check.
REFERENCE_CONFIGS=${GRID_REFERENCE_CONFIGS:-$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists presets "$GRID_PATH")}
REFERENCE_AGENTS=$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists policies)
[ -n "$REFERENCE_CONFIGS" ] && [ -n "$REFERENCE_AGENTS" ] || { echo "grid_job_lists.py named no preset or no reference agent for $GRID_PATH" >&2; exit 4; }

reference_line() {
  # whether a grid line names a reference agent, which plays without the model and the judge
  local agent
  for agent in $REFERENCE_AGENTS; do
    case " $1 " in *" --agent $agent "*) return 0 ;; esac
  done
  return 1
}

model_lines() {
  # whether a line of the grid file needs the served model: any line but one of a reference agent; a grid file without a
  # line needs none, and the expansion holds under set -u on a bash older than 4.4
  local line
  for line in ${LINES[@]+"${LINES[@]}"}; do
    reference_line "$line" || return 0
  done
  return 1
}

# the reference agents alone wait for no endpoint, since no model plays, and so does a grid file of reference lines alone,
# the pending reference pairs of a CPU job
if [ "${GRID_REFERENCE:-1}" = only ] || ! model_lines; then
  BASE_URL=; JUDGE=
else
  [ -n "${BASE_URL:-}" ] || BASE_URL=$(endpoint_url "${ENDPOINT_FILE:-$ENDPOINTS/$SERVED_NAME.txt}") || exit 3
  JUDGE=$(judge_args) || exit 3
fi

tables_ready() {
  # The tables every run of this job reads, checked before any line starts and never computed here: the models of the
  # diagnostics server, the label table and the grade table of every rules key of the lines and the reference presets of this
  # grid, on its start set with GRID_EXTRA and on the development starts, which the fillers play. The tables job
  # (precompute_tables.sbatch) is the one process that writes them on a clone, so that no table changes under a run in
  # flight; a table that is missing or of other code stops the job before its first line with the line that computes it.
  local entry configs=()
  for entry in $REFERENCE_CONFIGS; do configs+=(--config "$entry"); done
  # shellcheck disable=SC2086
  if python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables --check --grid "$GRID_PATH" ${configs[@]+"${configs[@]}"} --starts "${GRID_STARTS:-sampled}" ${GRID_EXTRA:-} \
      && python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables --check --starts development --grid "$GRID_PATH" ${configs[@]+"${configs[@]}"} ${GRID_EXTRA:-}; then
    return 0
  fi
  echo "TABLES_MISSING the tables of $GRID_PATH: run the tables job first, bash \$H/submit.sh precompute_tables.sbatch with GRID_FILE naming its grid file, GRID_STARTS=${GRID_STARTS:-sampled}${GRID_EXTRA:+ and GRID_EXTRA=$GRID_EXTRA}" >&2
  return 4
}

# The places of the processes alive: the pid, the key, the kind (run) and the stem of the log of each, kept apart by
# index
SLOT_PIDS=(); SLOT_KEYS=(); SLOT_KINDS=(); SLOT_NAMES=()

in_flight() {
  # one line of OUT/in_flight.tsv, as check_grid_complete.py reads it: the epoch seconds, start or end ($1), the loop of the job that
  # started the process (LOOP, reference or units), run ($2), the stem of its log ($3), its pid ($4), and 1 when the
  # configuration of its key $5 charges the wall seconds of thinking, else 0
  local wall=0
  if "$WALL_UNIT" "$5"; then wall=1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$1" "$LOOP" "$2" "$3" "$4" "$wall" >> "$OUT/in_flight.tsv"
}

start_slot() {
  # starts the key $1 as a run ($2) in the background and records its place and its start: at its position $3 in the
  # schedule, its output in OUT/<the stem that NAME_UNIT gives>.log. The start function puts the run in the background
  # itself, after its line in the log of the job, so that no process holds the output of the job once the job ends.
  local name
  name=$("$NAME_UNIT" "$1")
  "$START_UNIT" "$1" "$3" "$OUT/$name.log"
  SLOT_PIDS+=("$!"); SLOT_KEYS+=("$1"); SLOT_KINDS+=("$2"); SLOT_NAMES+=("$name")
  in_flight start "$2" "$name" "$!" "$1"
}

reap_slot() {
  # waits until the process of a place has exited, records its end, frees its place and sets REAPED_KEY, REAPED_KIND,
  # REAPED_NAME and REAPED_STATUS; it polls the jobs of the shell in place of wait -n, so it runs on any bash
  local k running
  while :; do
    running=" $(jobs -rp | tr '\n' ' ') "
    for k in "${!SLOT_PIDS[@]}"; do
      case "$running" in *" ${SLOT_PIDS[$k]} "*) continue ;; esac
      wait "${SLOT_PIDS[$k]}" 2>/dev/null
      REAPED_STATUS=$?
      REAPED_KEY=${SLOT_KEYS[$k]}; REAPED_KIND=${SLOT_KINDS[$k]}; REAPED_NAME=${SLOT_NAMES[$k]}
      in_flight end "$REAPED_KIND" "$REAPED_NAME" "${SLOT_PIDS[$k]}" "$REAPED_KEY"
      unset "SLOT_PIDS[$k]" "SLOT_KEYS[$k]" "SLOT_KINDS[$k]" "SLOT_NAMES[$k]"
      return 0
    done
    sleep "${SLOT_POLL:-1}"
  done
}

service_answers() {
  # whether the served model still answers: one GET of its model list through failures.endpoint_health, read after a model run
  # stopped with status 75, the class of a run whose last calls ended in harness errors
  python - "$BASE_URL" <<'PY'
import os, sys
sys.path.insert(0, "benchmarks/hydraulic_rig")
from failures import endpoint_health
health = endpoint_health(sys.argv[1], os.environ.get("API_KEY", ""), 30)
print(f"ENDPOINT_HEALTH ok {health['ok']} status {health['status']} seconds {health['seconds']} error {health['error']}")
sys.exit(0 if health["ok"] else 1)
PY
}

service_down() {
  # A model run that stopped with status 75 while its endpoint no longer answers ends the job: every run alive gets TERM, the
  # job keeps its record and its latency, suspect and overlap lines, prints SERVICE_DOWN and ends with status 6, so that a dead server
  # holds the card no longer than the run that met it.
  local k
  echo "SERVICE_DOWN $BASE_URL after $REAPED_NAME stopped with status 75"
  for k in "${!SLOT_PIDS[@]}"; do kill -TERM "${SLOT_PIDS[$k]}" 2>/dev/null; done
  for k in "${!SLOT_PIDS[@]}"; do
    wait "${SLOT_PIDS[$k]}" 2>/dev/null
    in_flight end "${SLOT_KINDS[$k]}" "${SLOT_NAMES[$k]}" "${SLOT_PIDS[$k]}" "${SLOT_KEYS[$k]}"
  done
  job_checks "$OUT"
  gpu_snapshot END
  exit 6
}

play_phase() {
  # Plays the positions $2... of SCHEDULE in their order with $1 runs alive. After a model run of the units that stopped with
  # status 75 the endpoint is probed once, and a server that does not answer ends the job (service_down).
  local limit=$1 next=0 positions
  shift
  positions=("$@")
  while [ "$next" -lt "${#positions[@]}" ] || [ "${#SLOT_PIDS[@]}" -gt 0 ]; do
    while [ "$next" -lt "${#positions[@]}" ] && [ "${#SLOT_PIDS[@]}" -lt "$limit" ]; do
      [ "$limit" != 1 ] || [ "$WALL_PHASE" != 1 ] || echo "WALL_EPISODE $("$NAME_UNIT" "${SCHEDULE[${positions[$next]}]}")" >&2
      start_slot "${SCHEDULE[${positions[$next]}]}" run "${positions[$next]}"
      next=$((next + 1))
    done
    reap_slot
    if [ "$REAPED_STATUS" = 75 ] && [ "$LOOP" = units ] && [ -n "${BASE_URL:-}" ] && ! service_answers; then service_down; fi
  done
}

play_scheduled() {
  # Plays the keys of SCHEDULE, through the functions that START_UNIT, NAME_UNIT and WALL_UNIT name, in two phases, so that
  # the wall seconds of thinking stay clean: first every key whose configuration charges none, with GRID_PARALLEL runs alive (one by
  # default); then, once the last of them has ended, every key that charges wall seconds, one run alive, which the phase hands
  # its runs as GRID_PARALLEL=1, the runs in flight that the start check of a preset with a clock reads (design.RUNS_IN_FLIGHT),
  # so that no other episode of the job shares the card with an episode whose wall seconds count. No filler runs.
  local position free=() wall=()
  for position in "${!SCHEDULE[@]}"; do
    if "$WALL_UNIT" "${SCHEDULE[$position]}"; then wall+=("$position"); else free+=("$position"); fi
  done
  echo "PHASE_FREE start ${#free[@]} runs, ${GRID_PARALLEL:-1} in flight" >&2
  WALL_PHASE=0 play_phase "${GRID_PARALLEL:-1}" ${free[@]+"${free[@]}"}
  echo "PHASE_FREE end" >&2
  echo "PHASE_WALL start ${#wall[@]} runs, 1 in flight" >&2
  WALL_PHASE=1 GRID_PARALLEL=1 play_phase 1 ${wall[@]+"${wall[@]}"}
  echo "PHASE_WALL end" >&2
}

gpu_snapshot() {
  # the cards of the job as nvidia-smi sees them at its start and at its end ($1), one GPU_<$1> line per card
  command -v nvidia-smi >/dev/null 2>&1 && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || return 0
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed "s/^/GPU_$1 /"
}

clocked_lines() {
  # the lines of the grid file $1 whose configuration charges the wall seconds of thinking, as grid_job_lists.py composes them
  python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists clocked "$1"
}

reference_name() {
  # the stem of the log of the reference key $1, "<preset[:clock mode]> <agent> <replicate>"
  local entry agent r
  read -r entry agent r <<< "$1"
  echo "reference_${entry/:/_}_${agent}_replicate_$r"
}

start_reference() {
  # starts the reference run of the key $1 in the background, its output in the log $3 beside the run folders and its exit status
  # in the file beside the log, $3.status, as run_unit writes the status of a unit
  local entry agent r config mode=""
  read -r entry agent r <<< "$1"
  config=${entry%%:*}
  [ "$entry" = "$config" ] || mode="--clock-mode ${entry#*:}"
  # shellcheck disable=SC2086,SC2046
  ( HYD_GRID_LINE=reference python -m CausalDiagnosis.experiments.episode_runners.run_episodes --config $config $mode --agent $agent $SEEDS $(replicate_option "$r") \
      --out "$OUT" ${GRID_EXTRA:-}
    echo "$?" > "$3.status" ) > "$3" 2>&1 &
}

wall_reference() {
  # whether the preset of the reference key $1 charges the wall seconds of thinking
  case " $WALL_ENTRIES " in *" ${1%% *} "*) return 0 ;; *) return 1 ;; esac
}

reference_rows() {
  # The reference runs, every preset of REFERENCE_CONFIGS with every reference agent on every replicate, with GRID_PARALLEL
  # runs alive in the phase without the wall clock (play_scheduled). On a preset whose clock charges the wall seconds between two tool calls as
  # thinking time (the clock mode generation_time) the seconds of the harness are charged too, so no more runs fly than the
  # job has cores; the clock of generated tokens charges a reference run no thinking time. Which presets charge wall seconds
  # grid_job_lists.py reads from a grid file of one line per preset, so the shell composes no configuration. An agent of the
  # wiring check plays the first replicate alone (wiring_skipped).
  local entry agent r lines line
  SCHEDULE=()
  lines=$(mktemp) || return 1
  for entry in $REFERENCE_CONFIGS; do
    if [ "$entry" = "${entry%%:*}" ]; then echo "--config $entry --agent ${REFERENCE_AGENTS%% *}"; else echo "--config ${entry%%:*} --clock-mode ${entry#*:} --agent ${REFERENCE_AGENTS%% *}"; fi
    for agent in $REFERENCE_AGENTS; do
      for r in $REPLICATES; do
        wiring_skipped "$agent" "$r" || SCHEDULE+=("$entry $agent $r")
      done
    done
  done > "$lines"
  WALL_ENTRIES=
  while read -r line; do
    set -- $line
    if [ "$3" = --clock-mode ]; then WALL_ENTRIES="$WALL_ENTRIES $2:$4"; else WALL_ENTRIES="$WALL_ENTRIES $2"; fi
  done < <(clocked_lines "$lines")
  rm -f "$lines"
  REFERENCE_KEYS=(${SCHEDULE[@]+"${SCHEDULE[@]}"})
  LOOP=reference; START_UNIT=start_reference; NAME_UNIT=reference_name; WALL_UNIT=wall_reference
  play_scheduled
}

reference_statuses() {
  # The status of every reference run, read as unit_statuses reads a unit: 0 is a finished run, 75 a run that stopped and left
  # its folder, any other value or none a failed run, which fails the job with status 5.
  local key status failed=0
  for key in ${REFERENCE_KEYS[@]+"${REFERENCE_KEYS[@]}"}; do
    status=$(cat "$OUT/$(reference_name "$key").log.status" 2>/dev/null)
    case "${status:-none}" in
      0) ;;
      75) echo "REFERENCE_STOPPED $key" ;;
      *) echo "REFERENCE_FAILED $key ${status:-none}"; failed=1 ;;
    esac
  done
  [ "$failed" = 0 ] || return 5
}

run_unit() {
  # One run of a grid line, whose exit status lands in the file beside its log, line_<i>_replicate_<r>.log.status. A line of a
  # reference agent plays without the model and the judge, as reference_rows plays it.
  local i=${1%% *} r=${1#* } status
  echo "### $(hostname) $(date) grid line $i, replicate $r: ${LINES[$i]}"
  if reference_line "${LINES[$i]}"; then
    # shellcheck disable=SC2086,SC2046
    HYD_GRID_LINE=$i python -m CausalDiagnosis.experiments.episode_runners.run_episodes ${LINES[$i]} $SEEDS $(replicate_option "$r") --out "$OUT" ${GRID_EXTRA:-}
    status=$?
    echo "$status" > "$OUT/line_${i}_replicate_${r}.log.status"
    return "$status"
  fi
  # a line that names no agent runs the Plan-Execute agent of AssetOpsBench, the default of the first grid
  local agent=""; case " ${LINES[$i]} " in *" --agent "*) ;; *) agent="--agent plan_execute";; esac
  # shellcheck disable=SC2086,SC2046
  HYD_GRID_LINE=$i python -m CausalDiagnosis.experiments.episode_runners.run_episodes ${LINES[$i]} $agent --model "$SERVED_NAME" --base-url "$BASE_URL" \
    $JUDGE $SEEDS $(replicate_option "$r") --out "$OUT" ${GRID_EXTRA:-}
  status=$?
  echo "$status" > "$OUT/line_${i}_replicate_${r}.log.status"
  return "$status"
}

line_name() {
  # the stem of the log of the unit $1, "<line index> <replicate>"
  echo "line_${1%% *}_replicate_${1#* }"
}

start_line() {
  # starts the run of the unit $1 at the position $2 of the schedule in the background, its output in the log $3 beside the
  # run folders, after its line in the log of the job
  echo "### $(date) start grid run $2, line ${1%% *}, replicate ${1#* }: ${LINES[${1%% *}]}"
  run_unit "$1" > "$3" 2>&1 &
}

wall_line() {
  # whether the line of the unit $1 charges the wall seconds of thinking
  grep -qxF -- "${LINES[${1%% *}]}" <<< "$WALL_LINES"
}

play_units() {
  # GRID_PARALLEL runs alive at once (one by default) in the phase without the wall clock (play_scheduled): the served model batches their
  # requests and every run writes its own run folder; each run's output goes to line_<i>_replicate_<r>.log beside the run
  # folders.
  WALL_LINES=$(clocked_lines "$GRID_PATH")
  SCHEDULE=(${UNITS[@]+"${UNITS[@]}"})
  LOOP=units; START_UNIT=start_line; NAME_UNIT=line_name; WALL_UNIT=wall_line
  play_scheduled
}

unit_statuses() {
  # The status of every unit: 0 is a finished run, 75 a run that stopped and left its folder, any other value or none a
  # failed run, which fails the job with status 5.
  local u i r status failed=0
  for u in "${!UNITS[@]}"; do
    i=${UNITS[$u]%% *}; r=${UNITS[$u]#* }
    status=$(cat "$OUT/line_${i}_replicate_${r}.log.status" 2>/dev/null)
    case "${status:-none}" in
      0) ;;
      75) echo "UNIT_STOPPED $i $r" ;;
      *) echo "UNIT_FAILED $i $r ${status:-none}"; failed=1 ;;
    esac
  done
  [ "$failed" = 0 ] || return 5
}

job_checks() {
  # The generation time clock charges the wall seconds between the agent's tool calls as thinking time, so the latency of
  # every model call moves the clock and so does any hold of the job. cluster_run_checks.py prints the latency of the model calls
  # per run folder and names the runs of a requeued or suspended job. A job that Slurm suspends sees its wall clock and its
  # monotonic clock advance alike, so the suspension is read from the accounting record of the job, which write_job_record
  # of env.sh keeps beside the run folders as job_record.txt. An episode on the wall clock plays alone on a clean cache, so the
  # overlap check names every run alive during such an episode and every such episode of an agent with a model endpoint whose
  # flush of the model server's cache is not recorded with 200 and no cached token in its first request. Such an episode voids
  # the wall-clock runs that the check names, so a failed overlap check returns 7, where a suspect run leaves the status alone.
  python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks latency "$1"
  write_job_record "$1"
  python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks suspect "$1" || echo "SUSPECT_EXIT $?"
  python -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks overlap "$1" || { echo "OVERLAP_EXIT $?"; return 7; }
}

COUCH_PID=
couchdb_if_needed() {
  # A job of its own whose runs name the store couchdb, the CPU job of the reference rows on the AssetOpsBench servers,
  # starts CouchDB on node local disk. Under run_lines.sh COUCHDB_URL is set already, and nothing starts.
  case " ${GRID_EXTRA:-} " in *" --store couchdb "*) ;; *) return 0 ;; esac
  [ -z "${COUCHDB_URL:-}" ] || return 0
  COUCH_LA=$(node_local_dir)/grid_couchdb_$$
  mkdir -p "$COUCH_LA"
  start_couchdb "$COUCH_LA"
}

play_grid() {
  # GRID_REFERENCE=0 skips the reference agents, for a rerun of model lines into a folder that already holds them, and
  # GRID_REFERENCE=only plays them alone; a failed reference run fails the job as a failed unit does
  local status=0
  [ "${GRID_REFERENCE:-1}" = 0 ] || reference_rows
  if [ "${GRID_REFERENCE:-1}" != only ]; then
    play_units
  fi
  # a failed overlap check fails the job with status 7, which the status 5 of a failed run replaces
  job_checks "$OUT" || status=$?
  if [ "${GRID_REFERENCE:-1}" != 0 ]; then reference_statuses || status=$?; fi
  if [ "${GRID_REFERENCE:-1}" != only ]; then unit_statuses || status=$?; fi
  return "$status"
}

trap 'if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; fi; [ -z "${COUCH_LA:-}" ] || rm -rf "$COUCH_LA"' EXIT
couchdb_if_needed
tables_ready || exit 4
# a job that the time limit ends keeps its record and its latency, suspect and overlap lines
trap 'job_checks "$OUT"; exit 143' TERM
write_job_record "$OUT"
gpu_snapshot START
play_grid; GRID_STATUS=$?
gpu_snapshot END
[ "$GRID_STATUS" = 0 ] || exit "$GRID_STATUS"
echo "CD_GRID_DONE $OUT"
