#!/bin/bash
#SBATCH -J cd-aob
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
# The AssetOpsBench reproduction of the paper's granite row (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md in the fork). The paper-era
# Agent-As-Tool and Plan-Execute play the runnable scenarios against the served model, CouchDB runs under Apptainer on node
# local disk with the data of layer B (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/fetch_public_servers_and_data.py), and the run is then scored (judge, efficiency) and
# each port it played is checked against the paper. AOB_MODE=pilot plays the ten pilot scenarios once with both ports
# (20 episodes), AOB_MODE=full the 120 runnable scenarios AOB_REPEATS times (once by default, the paper's count) with both
# ports (240 episodes), and AOB_MODE=ids the scenarios of AOB_IDS AOB_REPEATS times, as for a targeted rerun. AOB_AGENTS,
# when set, names the ports to play (both by default). With agent_as_tool_paper among them acceptance_check.py checks the run
# against the paper after scoring it and writes results/acceptance.md (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md), and with
# plan_execute_paper among them it checks that port and writes results/acceptance_plan_execute_paper.md. The log carries
# ACCEPTANCE_EXIT and PE_ACCEPTANCE_EXIT with the exit code of the check: 0 for a pass, 1 for a miss and 3 when the runs
# cannot decide, which holds for the pilot, for a list of ids and for more than one run per scenario. The model endpoint
# is the one of the serving job, BASE_URL, which the job exported, else the endpoint file of SERVED_NAME. The judge
# scores every episode under both protocols (five calls at temperature 0.3 and one call at temperature 0) when
# JUDGE_MODEL is set, at JUDGE_BASE_URL or at the endpoint file of JUDGE_SERVED_NAME.
# AOB_RECOVERED_ROWS names an optional private rows file outside the clone, and AOB_MODE=recovered plays the runnable
# scenarios and those it unlocks.
# Run folders go to $RUNS/aob_repro/<mode>_<job id>, with RUNS the folder of the kind of the tree. Inside a new serving job,
# which exports BASE_URL and, with CO_SERVE_JUDGE, JUDGE_BASE_URL,
#   A=${H%/cluster_jobs}/frameworks/AssetOpsBench/cluster_jobs
#   E="CO_SERVE_JUDGE=1 JUDGE_MODEL=<judge folder> AOB_MODE=pilot RUN_AFTER_SERVE=$A/run_reproduction.sh"
#   env $E bash $H/submit.sh serve_model.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
H=$HPC_DIR
fork_python
cd "$FORK" || exit 2
runs_root

MODE=${AOB_MODE:-pilot}
case "$MODE" in
  pilot) SET_ARGS=(--set pilot --repeats 1) ;;
  full) SET_ARGS=(--set runnable --repeats "${AOB_REPEATS:-1}") ;;
  recovered) SET_ARGS=(--set recovered --repeats "${AOB_REPEATS:-1}") ;;
  ids) [ -n "${AOB_IDS:-}" ] || { echo "AOB_MODE=ids needs the scenario ids in AOB_IDS" >&2; exit 2; }
       SET_ARGS=(--ids $AOB_IDS --repeats "${AOB_REPEATS:-1}") ;;
  *) echo "AOB_MODE must be pilot, full, recovered or ids, not $MODE" >&2; exit 2 ;;
esac
AGENTS=${AOB_AGENTS:-agent_as_tool_paper plan_execute_paper}
SET_ARGS+=(--agents $AGENTS)
OUT=${AOB_OUT:-$RUNS/aob_repro}/${MODE}_${SLURM_JOB_ID:-manual}
# a requeued job keeps its id, so the folder of the interrupted attempt is moved aside, as the grid scripts do
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && [ -d "$OUT" ]; then mv "$OUT" "${OUT}_attempt$SLURM_RESTART_COUNT"; fi
mkdir -p "$OUT"

# A folder of its own under the job's node local folder, which a serving job running this script also uses for SGLang.
LA=$(node_local_dir)/aob_$$
mkdir -p "$LA"
COUCH_PID=
cleanup() {
  if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; wait "$COUCH_PID" 2>/dev/null; fi
  rm -rf "$LA"
}
trap cleanup EXIT
export UV_CACHE_DIR=$LA/uv-cache TMPDIR=$LA/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
job_header "mode $MODE out $OUT"

# 1. The packages that the servers of main-0.x import, which the environment of the lock holds: couchdb3, litellm for the
# FMSR generate tools, and torch, transformers and granite-tsfm for the TSFM tools.
for module in couchdb3 litellm torch transformers tsfm_public; do require_module "$module"; done

# 2. The scenario file of the paper's 141 scenarios, downloaded once from its pinned Hugging Face commit, then layer B, built
# once from IBM's public branches into a folder outside the clone. fetch_public_servers_and_data.py checks the SHA-256 and the driver reads its file.
SCEN=$DATA/aob_all_utterance_d925adb.jsonl
SCEN_URL=$(python -c "from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import SCENARIO_URL; print(SCENARIO_URL)")
if [ ! -s "$SCEN" ]; then
  curl -sSfL -o "$SCEN.part" "$SCEN_URL" && mv "$SCEN.part" "$SCEN" || { echo "could not download $SCEN_URL" >&2; exit 2; }
fi
LAYER=${AOB_LAYER_B:-$DATA/aob_layer_b${AOB_RECOVERED_ROWS:+_recovered}}
python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.fetch_public_servers_and_data --scenarios "$SCEN" --out "$LAYER" || { echo "could not build layer B in $LAYER" >&2; exit 2; }

# 3. CouchDB 3.5 under Apptainer on node local disk, at a free port and under an Erlang node name of its own, so that it
# shares the node with any other CouchDB (start_couchdb.sh).
start_couchdb "$LA"
export TSFM_WORKDIR=$LA/tsfm_work
mkdir -p "$TSFM_WORKDIR"

# 4. The model endpoint and the judge.
[ -n "${BASE_URL:-}" ] || BASE_URL=$(endpoint_url "${ENDPOINT_FILE:-$ENDPOINTS/$SERVED_NAME.txt}") || exit 3
# the run ledger copies the endpoint file of the serving job, whose lines name the weights commit, the engine and the card, and
# marks a file whose URL is not the run's endpoint
export HYD_ENDPOINT_FILE=$(readlink -f "${ENDPOINT_FILE:-$ENDPOINTS/$SERVED_NAME.txt}" 2>/dev/null)
key_curl "$BASE_URL/models"; echo
# the judge scores the run when JUDGE_BASE_URL names it, the judge that this job serves with CO_SERVE_JUDGE=1 or a remote
# route, and no job waits for a judge served elsewhere
JUDGE_ARGS=()
if [ -n "$JUDGE_MODEL" ] && [ -n "${JUDGE_BASE_URL:-}" ]; then
  JUDGE_ARGS=(--judge-model "$JUDGE_SERVED_NAME" --judge-base-url "$JUDGE_BASE_URL" --workers "${AOB_JUDGE_WORKERS:-4}")
else
  echo "no judge in the run, so the run is scored without the judge; score_runs.py can judge the folder later" >&2
fi

# 5. The episodes, then the scores and the check of each port the run played.
python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_scenarios --scenarios "$LAYER/scenarios.jsonl" --layer-b "$LAYER" --out "$OUT" "${SET_ARGS[@]}" --model "$SERVED_NAME" \
  --base-url "$BASE_URL" 2>&1 | tee "$OUT/driver.log"
DRIVER=${PIPESTATUS[0]}; echo "DRIVER_EXIT $DRIVER"
# the first step that failed gives the status of the job, after every later step has run on what was played
STATUS=0
failed() { [ "$1" = 0 ] || [ "$STATUS" != 0 ] || STATUS=$1; }
failed "$DRIVER"
RUN_DIR=$(sed -n 's/^AOB_REPRO_RUN //p' "$OUT/driver.log" | tail -1)
[ -n "$RUN_DIR" ] || { echo "the driver ended without a run folder, see $OUT/driver.log" >&2; exit 3; }
python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.score_runs "$RUN_DIR" ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}; s=$?; echo "SCORE_EXIT $s"; failed "$s"
# the exit code of acceptance_check.py is its verdict, 0 a pass, 1 a miss and 3 runs that cannot decide, and not a failure of the job;
# a check that wrote no report failed
case " $AGENTS " in
  *" agent_as_tool_paper "*)
    python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check "$RUN_DIR" --out "$OUT/results"; echo "ACCEPTANCE_EXIT $?"
    [ -s "$OUT/results/acceptance.md" ] || failed 3 ;;
esac
case " $AGENTS " in
  *" plan_execute_paper "*)
    python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check "$RUN_DIR" --port plan_execute_paper --out "$OUT/results"; echo "PE_ACCEPTANCE_EXIT $?"
    [ -s "$OUT/results/acceptance_plan_execute_paper.md" ] || failed 3 ;;
esac
echo "CD_AOB_REPRO_DONE $OUT"
exit "$STATUS"
