# The functions of every cluster job of this tree and the folders they read, a tracked file that env.sh sources, so that
# env.sh, the copy of env.example.sh that each person keeps, holds values alone. HPC_DIR is the folder of this file, and every
# script reaches a sibling through it, never through $0, which sbatch points at a copy in its spool. FORK is the root of the
# tree, the nearest folder above HPC_DIR that holds uv.lock, which a clone and an upload both carry, so that no script counts
# the folders between them.
HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HPC_DIR

fork_root() {
  # The nearest folder above HPC_DIR that holds uv.lock, the root of the tree, or a stop when no folder does.
  local d=$HPC_DIR
  while [ ! -f "$d/uv.lock" ]; do
    [ "$d" != / ] || { echo "no uv.lock in $HPC_DIR or above it, so the root of the tree is unknown" >&2; return 2; }
    d=$(dirname "$d")
  done
  echo "$d"
}

FORK=$(fork_root) || exit 2                          # the tree of these scripts, so that a job runs the code of its own tree
GRIDS=$HPC_DIR/grids                                 # the grid files, the folder that GRIDS of CausalDiagnosis/utils/paths.py names
PYTHON_VERSION=${PYTHON_VERSION:-$(head -n 1 "$FORK/.python-version" 2>/dev/null)}   # the Python of the lock
CD_FUNCTIONS=1                                       # env.sh sourced this file, which an env.sh older than this file does not
# Every fetch from outside the cluster follows one rule, that of CausalDiagnosis/utils/download_retry.py, whose defaults a test
# compares with these. A compute node fetches over a link that breaks off and stalls, so an attempt waits at most FETCH_TIMEOUT
# seconds for its next bytes wherever its tool takes a timeout, the timeout under which the training environment passed on a
# compute node where the 30 seconds of uv's default timed out three times in a row, and a failed attempt is made again after
# FETCH_PAUSE seconds, with FETCH_RETRIES retries in a row that bring nothing before the fetch fails, a count that starts again
# whenever an attempt brought bytes, as huggingface_hub takes up a download that broke off. CACHE keeps on scratch what the
# tools fetched, the packages of uv and the layers of an image not yet in place, so that a job that a broken link or its time
# limit ended leaves them to the next.
# A value set in env.sh or on the submit line wins over these defaults.
CACHE=${CACHE:-$ROOT/cache}
FETCH_TIMEOUT=${FETCH_TIMEOUT:-300}
FETCH_RETRIES=${FETCH_RETRIES:-5}
FETCH_PAUSE=${FETCH_PAUSE:-1}
export FETCH_TIMEOUT FETCH_RETRIES FETCH_PAUSE           # the Python of a job fetches under the same rule
mkdir -p "$LOGS" "$ENDPOINTS" "$DATA" 2>/dev/null || true

sha256_of() {
  # The SHA-256 of a file, or nothing when the file cannot be read.
  sha256sum "$1" 2>/dev/null | cut -d' ' -f1
}

fetched_size() {
  # The kilobytes under the path $1, a file or a folder, or 0 when it does not exist.
  local kb
  kb=$(du -sk "$1" 2>/dev/null | cut -f1)
  echo "${kb:-0}"
}

retry_fetch() {
  # retry_fetch PATH COMMAND [ARGUMENTS]: a fetch from outside the cluster, made again under the rule of the fetches above while
  # it fails. PATH is where the command leaves what it fetched, its cache, partial file or folder, so an attempt that left PATH
  # larger brought bytes. A retry prints FETCH_RETRY with the name of the command alone, since its arguments may hold a key,
  # and a fetch that the rule gives up prints FETCH_FAILED and returns the status of its last attempt.
  local path=$1 left=$FETCH_RETRIES before status
  shift
  while :; do
    before=$(fetched_size "$path")
    "$@" && return 0
    status=$?
    [ "$(fetched_size "$path")" -le "$before" ] || left=$FETCH_RETRIES
    if [ "$left" -le 0 ]; then
      echo "FETCH_FAILED $1 with status $status after $FETCH_RETRIES retries that brought nothing" >&2
      return "$status"
    fi
    left=$((left - 1))
    echo "FETCH_RETRY $((FETCH_RETRIES - left)) of $FETCH_RETRIES in $FETCH_PAUSE s, $1 ended with status $status" >&2
    sleep "$FETCH_PAUSE"
  done
}

pull_image_once() {
  # pull_image_once SIF SOURCE LOCAL: the image SOURCE, such as docker://couchdb:3.5, into the file SIF, and nothing when SIF
  # holds an image. The pull writes a partial file of its job, renamed to SIF once the pull ended, so that no job takes half an
  # image, unpacks on the node local folder LOCAL, and keeps the layers in a cache of the image under CACHE, so that a pull that
  # a broken link or the time limit ended goes on from the layers it finished. The cache goes once SIF is in place.
  local sif=$1 origin=$2 local_dir=$3 cache partial
  [ ! -s "$sif" ] || return 0
  cache=$CACHE/apptainer/$(basename "$sif")
  partial=$sif.partial.${SLURM_JOB_ID:-$$}
  mkdir -p "$(dirname "$sif")" "$cache" "$local_dir/apptainer"
  APPTAINER_CACHEDIR=$cache APPTAINER_TMPDIR=$local_dir/apptainer retry_fetch "$cache" apptainer_pull "$partial" "$origin" \
    || { rm -f "$partial"; return 1; }
  mv "$partial" "$sif" && rm -rf "$cache"
}

apptainer_pull() {
  # One attempt of a pull of the image $2 into the file $1, which an attempt before it may have left.
  rm -f "$1"
  apptainer pull "$1" "$2"
}

node_local_dir() {
  # A scratch folder on node local disk for one job, removed by the caller's trap.
  local d=/local/$USER-cd-${SLURM_JOB_ID:-$$}
  if ! mkdir -p "$d" 2>/dev/null; then
    d=/tmp/$USER-cd-${SLURM_JOB_ID:-$$}
    echo "no node local disk under /local, so the job works in $d" >&2
    mkdir -p "$d"
  fi
  echo "$d"
}

require_model() {
  [ -n "$MODEL" ] || { echo "MODEL is empty: submit with MODEL=<Hugging Face id or model folder>" >&2; exit 2; }
}

require_gpu_of_record() {
  # A job that compares its card with the card of record needs both settings.
  local name
  for name in GPU_OF_RECORD GPU_MIB_OF_RECORD; do
    [ -n "${!name}" ] || { echo "$name is empty in env.sh: the card of record needs GPU_OF_RECORD and GPU_MIB_OF_RECORD" >&2; exit 2; }
  done
}

card_record() {
  # "name|MiB|uuid|driver version" of the first card that the job sees
  nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | awk -F', ' '{ print $1 "|" $2 "|" $3 "|" $4 }'
}

cuda_version() {
  # the CUDA Version of the header of nvidia-smi
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n 1
}

card_of_record() {
  # whether the card named in $1, with the memory in $2 counted in MiB, is the card of record
  [ -n "$GPU_OF_RECORD" ] && [ "$1" = "$GPU_OF_RECORD" ] && [ "$2" = "$GPU_MIB_OF_RECORD" ]
}

check_card() {
  # The card of the job into GPU_NAME, GPU_MIB, GPU_UUID, DRIVER_VERSION and CUDA_VERSION, checked against the card of record
  # when the caller passes required or when env.sh names either setting,
  # so that every job with a card plays on one card model once the card of record is set: those jobs need both settings, and on
  # another card the job prints GPU_NOT_OF_RECORD with its host and card and ends with status 2 before its work starts.
  IFS='|' read -r GPU_NAME GPU_MIB GPU_UUID DRIVER_VERSION <<< "$(card_record)"
  CUDA_VERSION=$(cuda_version)
  : "${GPU_NAME:=unknown}" "${GPU_MIB:=unknown}" "${GPU_UUID:=unknown}" "${DRIVER_VERSION:=unknown}" "${CUDA_VERSION:=unknown}"
  if [ "${1:-}" = required ] || [ -n "$GPU_OF_RECORD$GPU_MIB_OF_RECORD" ]; then
    require_gpu_of_record
    card_of_record "$GPU_NAME" "$GPU_MIB" || { echo "GPU_NOT_OF_RECORD $(hostname) $GPU_NAME $GPU_MIB"; exit 2; }
  fi
}

card_line() {
  # the card of check_card as one line of key and value pairs, every space of the name written as an underscore, the form of
  # the card line of an endpoint file, which run_ledger/ reads
  echo "gpu_name ${GPU_NAME// /_} gpu_mib $GPU_MIB gpu_uuid $GPU_UUID driver_version $DRIVER_VERSION cuda_version $CUDA_VERSION"
}

endpoint_card() {
  # "name|MiB" of the card that the card line of the endpoint file $1 names, the spaces of the name restored, or nothing
  local card
  card=$(sed -n 's/^gpu_name \([^ ]*\) gpu_mib \([^ ]*\) .*/\1 \2/p' "$1" 2>/dev/null | head -n 1)
  [ -z "$card" ] || echo "$(tr _ ' ' <<< "${card% *}")|${card##* }"
}

model_revision() {
  # the weights commit that the tracked serving file of record pins for the Hugging Face id $1, or nothing; read with the
  # standard library alone, since download_model.sbatch runs before the environment of the lock exists
  python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); print(next((b["weights_commit"] or "" for b in (d["agent"], d["judge"]) if b["hub_id"] == sys.argv[2]), ""))' \
    "$FORK/CausalDiagnosis/experiments/table_definitions/serving_of_record.json" "$1" 2>/dev/null
}

model_dir() {
  # The folder of a model: MODEL itself when it is an existing folder, else the download folder, or the copy under SHARED
  # when the download folder holds no weights and that copy is readable. A serving job that starts while
  # download_model.sbatch is still writing the download folder waits for the HUB_COMMIT.txt that the download writes last.
  # A folder that does not exist is returned at once, so the caller stops without holding its card.
  local m=$1 d
  if [[ "$m" == */* && -d "$m" ]]; then echo "$m"; return; fi
  d=$MODELS/${m##*/}
  if [ ! -f "$d/config.json" ] && [ -n "$SHARED" ] && [ -r "$SHARED/models/${m##*/}/config.json" ]; then echo "$SHARED/models/${m##*/}"; return; fi
  if [ "${SLURM_JOB_NAME:-}" = cd-serve ] && [ -d "$d" ] && [ ! -f "$d/config.json" ]; then
    for _ in $(seq 1 180); do
      [ -f "$d/HUB_COMMIT.txt" ] && break
      echo "waiting for the download of $m into $d" >&2; sleep 10
    done
  fi
  echo "$d"
}

weights_record() {
  # The line weights_commit of an endpoint file for the model folder $1: the first line of HUB_COMMIT.txt, else the commit
  # of the download metadata, else the SHA-256 of config.json, so the line never reads none.
  local commit
  commit=$(head -n 1 "$1/HUB_COMMIT.txt" 2>/dev/null)
  [ -n "$commit" ] || commit=$(cat "$1"/.cache/huggingface/download/*.metadata 2>/dev/null | grep -m1 -E '^[0-9a-f]{40}$')
  [ -n "$commit" ] || commit=config:$(sha256_of "$1/config.json")
  echo "weights_commit $commit"
  # the SHA-256 of the list of the folder's file hashes that download_model.sbatch writes, so the record names the files themselves
  [ ! -f "$1/FILES.sha256" ] || echo "weights_files_sha256 $(sha256_of "$1/FILES.sha256")"
}

grid_tables() {
  # The tables of every rules key that a launch plays, through precompute_grid_tables.py in the tree: the lines of the grid files
  # and the presets they name, which the reference runs of a grid job play, on the start set of the launch and on the
  # development starts, which the fillers play, with the run options of the launch. A launch is named as for its serving job,
  # by GRID_FILE (a file of the folder grids, or a path), GRID_STARTS and GRID_EXTRA; without GRID_FILE every grid file of the
  # folder grids is taken. With $1 compute the call writes every missing table, which the tables job alone does on a clone, so
  # it runs with CD_TABLES_READ_ONLY=0; with $1 check it writes nothing and returns 3 when a table is missing or of other code.
  local mode=${1:-compute} f folder grids=() starts=() check=() read_only=0
  folder=$(tree_location GRIDS) || return 4
  if [ -n "${GRID_FILE:-}" ]; then
    case "$GRID_FILE" in /*) grids=(--grid "$GRID_FILE") ;; *) grids=(--grid "$folder/$GRID_FILE") ;; esac
  else
    for f in "$folder"/grid_*.txt; do grids+=(--grid "$f"); done
  fi
  if [ "$mode" = check ]; then check=(--check); read_only=1; fi
  [ -z "${GRID_STARTS:-}" ] || starts=(--starts "$GRID_STARTS")
  # shellcheck disable=SC2086
  ( cd "$FORK" && export CD_TABLES_READ_ONLY=$read_only \
      && python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables ${check[@]+"${check[@]}"} --grid-presets "${grids[@]}" ${starts[@]+"${starts[@]}"} ${GRID_EXTRA:-} \
      && python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables ${check[@]+"${check[@]}"} --grid-presets --starts development "${grids[@]}" ${GRID_EXTRA:-} )
}

check_data() {
  # The data of record of this clone, which fetch_data.sbatch built into its data folder, against the pins of the tracked
  # CHECKSUMS.txt, through the check of the fetch command. It needs the Python of fork_python.
  ( cd "$FORK" && python -m CausalDiagnosis.datasets.hyd_1.download_and_build --check ) >&2 \
    || { echo "the data of record of $FORK is missing or differs from its pins: run fetch_data.sbatch" >&2; exit 2; }
}

data_file() {
  # The data file of record of this tree, as the loader of its default configuration resolves it. It needs the Python of
  # fork_python.
  python -c 'from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig; from CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory import default_facts_path; print(default_facts_path(EpisodeConfig()))'
}

lock_id() {
  # The first 12 digits of the SHA-256 of the lock file of the tree, which names its environment.
  local sum
  sum=$(sha256_of "$FORK/uv.lock")
  [ -n "$sum" ] || { echo "no uv.lock in $FORK" >&2; exit 2; }
  echo "${sum:0:12}"
}

env_synced() {
  # Whether the environment $1 holds the mark that setup_env.sbatch writes after a sync that finished, the SHA-256 of the lock of
  # this tree; a sync that stopped halfway leaves bin/python and no mark.
  [ "$(cat "$1/.cd_synced" 2>/dev/null)" = "$(sha256_of "$FORK/uv.lock")" ]
}

fork_python() {
  # The environment of the lock of this tree leads PATH, since the runner starts every tool server as python, and the code
  # is imported from the root of this tree and its src, whichever tree built the environment: the package CausalDiagnosis
  # from FORK/CausalDiagnosis and the tool servers of upstream from FORK/src/servers. The probe runs at the root, never in
  # CausalDiagnosis/, whose folders agents, datasets and evaluation would hide installed packages of those names.
  local env where
  env=$ENVS/$(lock_id)
  [ -x "$env/bin/python" ] || { echo "no environment at $env for the lock of $FORK: run setup_env.sbatch" >&2; exit 2; }
  env_synced "$env" || { echo "the environment at $env holds no mark of a finished sync of this lock: run setup_env.sbatch" >&2; exit 2; }
  export PATH=$env/bin:$PATH PYTHONPATH=$FORK:$FORK/src
  # every job reads the generated tables and writes neither them nor byte code into the clone that other jobs run; the
  # tables job alone writes the tables (grid_tables compute), so that no table changes under a run in flight
  export PYTHONDONTWRITEBYTECODE=1 CD_TABLES_READ_ONLY=1
  where=$(cd "$FORK" && python -c 'import os, CausalDiagnosis, servers; print(os.path.dirname(os.path.abspath(CausalDiagnosis.__file__)), os.path.dirname(os.path.abspath(servers.__file__)))' 2>/dev/null)
  [ "$where" = "$FORK/CausalDiagnosis $FORK/src/servers" ] \
    || { echo "the packages CausalDiagnosis and servers import from ${where:-nowhere}, not from $FORK/CausalDiagnosis and $FORK/src/servers" >&2; exit 2; }
}

require_module() {
  python -c "import $1" 2>/dev/null || { echo "the environment of this lock lacks $1, build it with setup_env.sbatch" >&2; exit 2; }
}

tree_id() {
  # The first 12 digits of the commit of the tree, or of the SHA-256 of the manifest of an upload.
  local id
  if [ -f "$FORK/UPLOAD_MANIFEST.txt" ] && [ ! -e "$FORK/.git" ]; then id=$(sha256_of "$FORK/UPLOAD_MANIFEST.txt")
  else id=$(git -C "$FORK" rev-parse HEAD 2>/dev/null); fi
  [ -n "$id" ] || { echo "the tree $FORK has no commit and no upload manifest" >&2; exit 2; }
  echo "${id:0:12}"
}

tree_location() {
  # The location that CausalDiagnosis/utils/paths.py of the tree names $1, such as GRIDS, asked with the Python of the lock of
  # the tree at its root, so that the scripts and the Python code read one place.
  local python
  python=$ENVS/$(lock_id)/bin/python
  [ -x "$python" ] || { echo "no environment at ${python%/bin/python} for the lock of $FORK: run setup_env.sbatch" >&2; return 2; }
  ( cd "$FORK" && PYTHONPATH="$FORK:$FORK/src" "$python" -m CausalDiagnosis.utils.paths "$1" )
}

record_check() {
  # The verdict of run_ledger/ record-check on this tree, with the environment of its lock, its lines into the log: RECORD
  # <commit> for a clone of a commit on origin with nothing changed and the data and tables of this tree, else DEVELOPMENT
  # with one REASON line per reason, and DEVELOPMENT when the check cannot run. True on RECORD.
  local out
  out=$( (fork_python && cd "$FORK" && python -m CausalDiagnosis.experiments.run_verification.run_ledger record-check) 2>&1 ) \
    || out=$(printf 'DEVELOPMENT\nREASON the record check did not run: %s' "${out:-no output}")
  printf '%s\n' "$out" >&2
  [ "$(head -n 1 <<< "$out" | cut -d' ' -f1)" = RECORD ]
}

runs_root() {
  # RUNS by the kind of the job, and RUNS_KIND that kind. A job is of record when record_check finds the tree a run of
  # record and the job plays the start set of a table (GRID_STARTS neither development, training nor seeds), else
  # development; RUN_KIND=record asks for a job of record and stops on DEVELOPMENT, RUN_KIND=development and
  # RUN_KIND=smoke name the other kinds. A folder of record holds runs of record alone, and the driver's start check
  # stamps development any run of it that fails a rule of its own line.
  local kind=${RUN_KIND:-} id
  case "$kind" in
    smoke) RUNS_KIND=smoke ;;
    development) RUNS_KIND=development ;;
    record|'')
      case "${GRID_STARTS:-sampled}" in
        development|training|seeds)
          [ "$kind" != record ] || { echo "RUN_KIND=record plays the start set of a table, not GRID_STARTS=$GRID_STARTS" >&2; exit 2; }
          RUNS_KIND=development ;;
        *)
          if record_check; then RUNS_KIND=record
          elif [ "$kind" = record ]; then echo "RUN_KIND=record needs a run of record, and the record check of $FORK says DEVELOPMENT" >&2; exit 2
          else RUNS_KIND=development; fi ;;
      esac ;;
    *) echo "RUN_KIND must be development, record or smoke, not $kind" >&2; exit 2 ;;
  esac
  if [ "$RUNS_KIND" = smoke ]; then RUNS=$ROOT/runs/smoke; else id=$(tree_id) || exit 2; RUNS=$ROOT/runs/$RUNS_KIND/$id; fi
  export RUNS RUNS_KIND
}

job_header() {
  # The first line of every job: the host, the time, the tree and whether it differs from its commit, then the caller's words.
  local state
  if [ -e "$FORK/.git" ]; then
    state=$(git -C "$FORK" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    if [ "$state" = 0 ]; then state="equal to its commit"; else state="$state paths differ from its commit"; fi
  elif [ -f "$FORK/UPLOAD_MANIFEST.txt" ]; then state="an upload"
  else state="no commit"; fi
  echo "### $(hostname) $(date '+%Y-%m-%d %H:%M:%S %Z') fork $FORK cluster_jobs $HPC_DIR tree $(tree_id 2>/dev/null || echo unknown) ($state)${*:+ $*}"
}

write_job_record() {
  # The accounting record of the job $2 (this job by default) into $1/job_record.txt, with every start of a requeued job
  # (-D), the allocation alone (-X) and the suspended time in the field Suspended, which cluster_run_checks.py reads. Outside a job
  # and without a job id it does nothing.
  local job=${2:-${SLURM_JOB_ID:-}}
  [ -n "$job" ] || return 0
  sacct -j "$job" -D -X -n -P -o JobID,State,Start,End,Elapsed,Suspended,NodeList > "$1/job_record.txt" \
    || echo "sacct gave no record of job $job, so read its Suspended field in the login shell" >&2
  echo "JOB_RECORD restarts ${SLURM_RESTART_COUNT:-0}: $(tr '\n' ' ' < "$1/job_record.txt")"
}

key_file() {
  echo "$ENDPOINTS/$1.key"
}

key_curl() {
  # curl with the key of API_KEY in a configuration file of mode 600 on node local disk, so that the key stands in no
  # argument of a process.
  local conf
  [ -n "${API_KEY:-}" ] || { curl -sf "$@"; return; }
  conf=$(node_local_dir)/curl_key.conf
  [ -s "$conf" ] || ( umask 077; printf 'header = "Authorization: Bearer %s"\n' "$API_KEY" > "$conf" )
  curl -sf -K "$conf" "$@"
}

endpoint_url() {
  # Prints the base URL a serving job wrote once its /v1/models answers; $1 is the endpoint file, $2 the tries of ten seconds.
  local file=$1 tries=${2:-180} url
  for _ in $(seq 1 "$tries"); do
    if [ -s "$file" ]; then
      url=$(head -1 "$file")
      key_curl "$url/models" >/dev/null 2>&1 && { echo "endpoint $url answers" >&2; echo "$url"; return 0; }
    fi
    sleep 10
  done
  echo "no answering endpoint in $file after $tries tries; is serve_model.sbatch running?" >&2
  return 1
}

judge_args() {
  # The judge options of run_episodes/ when JUDGE_BASE_URL names the judge, one that this job serves or a remote route, or
  # nothing: no grid job waits for a judge, since run_judge.sh scores the runs of a grid after the grid.
  if [ -z "$JUDGE_MODEL" ] || [ -z "$JUDGE_BASE_URL" ]; then
    echo "no judge in the run: run_judge.sh scores the runs after the grid" >&2
    return 0
  fi
  echo "--judge local --judge-model $JUDGE_SERVED_NAME --judge-base-url $JUDGE_BASE_URL --judge-protocol paper"
}

# The services of the frameworks of this tree. The job_services.sh of every frameworks/<name>/cluster_jobs defines the
# functions of the services that its rows need and adds its steps to three lists, so that no script of this folder names a
# framework: DATA_FETCHES holds the commands that fetch_data.sbatch runs after the data of record, IMAGE_PULLS the functions
# that pull_image.sbatch calls with a node local folder to pull the images that the jobs of the framework start, and
# PLATFORM_RECORDS the functions that platform_record.sh calls with a node local folder to record the images that the runs use.
DATA_FETCHES=()
IMAGE_PULLS=()
PLATFORM_RECORDS=()
for services in "$FORK"/CausalDiagnosis/frameworks/*/cluster_jobs/job_services.sh; do
  if [ -f "$services" ]; then source "$services"; fi
done
unset services
