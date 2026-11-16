#!/bin/bash
#SBATCH -J cd-serve
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH -t 12:00:00
#SBATCH --signal=B:TERM@180
# Serves MODEL with SGLang v0.5.19 from the Apptainer image on one card, as an OpenAI compatible endpoint behind the key of
# the job. The server listens on one address: the loopback address, so that the script of the job is its one client, or,
# with SERVE_REACH=nodes, the address of its own node, for a server that a job on another node reads. The key is drawn for
# the job, or with REFLECTION_JOB=<job id> taken from that job, so that one key serves the jobs that read both endpoints.
# It lies in ENDPOINTS/<job id>.key with mode 600 while the job runs and reaches the server through its environment,
# never as an argument. Once the server answers, the job writes ENDPOINTS/<job id>.txt, linked under the served name and as
# latest.txt: the URL first, then the job, the weights commit, the engine and the chat template, which the run ledgers copy.
# MODEL may be a Hugging Face id (downloaded by download_model.sbatch) or the absolute path of a model folder, whose
# basename becomes the served name. The image is unpacked once per job into a sandbox on node local disk.
# With RUN_AFTER_SERVE=<script> the job runs that script on the same node after the health check and ends when the script
# ends, with its status, or with status 3 and SERVICE_EXITED when a model server of the job exits first.
# Without a script it serves until it is cancelled or until the file ENDPOINTS/<job id>.stop
# appears. A job with neither a script nor SERVE_REACH=nodes would serve nobody and is refused. The judge is served by a
# job of its own with MODEL and JUDGE_MODEL both naming the judge, which gives it the flags of JUDGE_ALONE_SGLANG_EXTRA, or,
# with CO_SERVE_JUDGE=1 and JUDGE_MODEL set, by this job on the same card before the agent model (env.sh explains the
# memory split). With SERVE_ADAPTERS=1 it serves the LoRA adapters of the training experiment beside the agent model.
#   RUN_AFTER_SERVE=$H/3_run/run_lines.sh GRID_FILE=local_one.txt bash $H/submit.sh serve_model.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
if [ -z "${RUN_AFTER_SERVE:-}" ] && [ "${SERVE_REACH:-}" != nodes ]; then
  echo "neither RUN_AFTER_SERVE nor SERVE_REACH=nodes is set, so the server would serve nobody" >&2
  exit 2
fi
require_model
module load apptainer/1.4.5
[ -s "$SIF" ] || { echo "SGLang image not found at $SIF; run pull_image.sbatch or set SHARED in env.sh" >&2; exit 2; }
MODEL_DIR=$(model_dir "$MODEL")
[ -d "$MODEL_DIR" ] || { echo "model folder $MODEL_DIR missing; run download_model.sbatch first" >&2; exit 2; }
fork_python
FREE_FACTOR=${FREE_FACTOR:-3}                         # the free space of node local disk that the sandbox needs, in sizes of the image
SCRIPT_GRACE=${SCRIPT_GRACE:-150}                     # the seconds the script has after the warning signal before the server stops

L=$(node_local_dir)
ENDPOINT_FILE=$ENDPOINTS/$SLURM_JOB_ID.txt
JUDGE_ENDPOINT_FILE=$ENDPOINTS/$SLURM_JOB_ID-judge.txt
KEY_FILE=$(key_file "$SLURM_JOB_ID")
STOP_FILE=$ENDPOINTS/$SLURM_JOB_ID.stop
SGL_PID=
JUDGE_PID=
SCRIPT_PID=
SAMPLER_PID=
WARNED=
SERVICE_EXITED=
TEMPLATE_KIND=model
stop_servers() {
  local pid
  for pid in $SGL_PID $JUDGE_PID; do kill "$pid" 2>/dev/null; done
  for pid in $SGL_PID $JUDGE_PID; do wait "$pid" 2>/dev/null; done
  SGL_PID=
  JUDGE_PID=
  if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=; fi
}
cleanup() {
  local link target
  stop_servers
  for link in "$ENDPOINTS/latest.txt" "$ENDPOINTS/$SERVED_NAME.txt" "$ENDPOINTS/$JUDGE_SERVED_NAME.txt"; do
    target=$(readlink "$link" 2>/dev/null)
    if [ "$target" = "$ENDPOINT_FILE" ] || [ "$target" = "$JUDGE_ENDPOINT_FILE" ]; then rm -f "$link"; fi
  done
  rm -f "$ENDPOINT_FILE" "$JUDGE_ENDPOINT_FILE" "$KEY_FILE" "$STOP_FILE"
  rm -rf "$L"
}
trap cleanup EXIT
export APPTAINER_TMPDIR=$L/apptainer APPTAINER_CACHEDIR=$L/apptainer-cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$L/tmp"
export APPTAINERENV_HF_HOME=$L/hf APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TMPDIR=$L/tmp

job_header "model $MODEL served as $SERVED_NAME"; df -h "$L" | tail -1
if ! GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1) \
   || grep -q "Unknown Error\|Unable to determine" <<< "$GPU_INFO"; then
  echo "$GPU_INFO"; echo "GPU_UNHEALTHY on $(hostname): resubmit with --exclude=$(hostname)"; exit 3
fi
echo "$GPU_INFO"
# The card of the job, recorded in its endpoint file: a registered job, and any job once env.sh names the card of record,
# plays on that card alone and stops before the sandbox, the server and any endpoint file on another card (check_card)
runs_root
check_card
SAMPLER_FILE=$LOGS/gpu-$SLURM_JOB_ID.csv
start_sampler() {
  # nvidia-smi samples the card every 5 seconds into SAMPLER_FILE for the life of the job, with the fields of the wanted list
  # that the driver of the node lists, so no field list is typed for one driver; a sampler that cannot run prints
  # GPU_SAMPLER_FAILED and the server goes on without it
  local help f fields=()
  help=$(nvidia-smi --help-query-gpu 2>/dev/null)
  for f in timestamp utilization.gpu utilization.memory memory.used power.draw clocks.sm clocks.mem temperature.gpu; do
    if grep -q "\"$f\"" <<< "$help"; then fields+=("$f"); fi
  done
  for f in clocks_event_reasons.active clocks_throttle_reasons.active; do
    if grep -q "\"$f\"" <<< "$help"; then fields+=("$f"); break; fi
  done
  if [ ${#fields[@]} -eq 0 ]; then echo "GPU_SAMPLER_FAILED: nvidia-smi --help-query-gpu lists none of the fields"; return 0; fi
  nvidia-smi --query-gpu="$(IFS=,; echo "${fields[*]}")" --format=csv -l 5 > "$SAMPLER_FILE" 2>&1 &
  SAMPLER_PID=$!
  sleep 2
  if ! kill -0 "$SAMPLER_PID" 2>/dev/null; then
    echo "GPU_SAMPLER_FAILED: $(tail -n 1 "$SAMPLER_FILE" 2>/dev/null)"
    SAMPLER_PID=
  fi
}
start_sampler

# The SHA-256 of the image file, for the run ledgers, found in the background while the sandbox is built and the models load
( sha256_of "$SIF" > "$L/image_sha256.txt" ) &
SHA_PID=$!
# One sandbox per job on node local disk, so that no later step unpacks the image again, after a check of the free space
NEED_KB=$(( $(du -k "$SIF" | cut -f1) * FREE_FACTOR ))
FREE_KB=$(df -Pk "$L" | awk 'NR == 2 { print $4 }')
[ "${FREE_KB:-0}" -ge "$NEED_KB" ] || { echo "only ${FREE_KB:-0} KB free at $L, and the sandbox needs $FREE_FACTOR times the image, $NEED_KB KB" >&2; exit 3; }
SANDBOX=$L/sglang_sandbox
apptainer build --sandbox "$SANDBOX" "$SIF" || { echo "the sandbox of $SIF could not be built in $L" >&2; exit 3; }
echo "sandbox $(du -sh "$SANDBOX" | cut -f1) at $SANDBOX"
SGLANG_VERSION=$(apptainer exec "$SANDBOX" python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null | tail -1)

reflection_key() {
  # The key of the job $1, read from its key file once that job has written it, while it is queued or running.
  local file
  file=$(key_file "$1")
  for _ in $(seq 1 "$JUDGE_WAIT_TRIES"); do
    if [ -s "$file" ]; then cat "$file"; return 0; fi
    [ -n "$(squeue -h -j "$1" -o %T 2>/dev/null)" ] || break
    sleep 10
  done
  echo "no key file of the job $1 at $file" >&2
  return 1
}

# The key of the job and the address of its servers. Both servers of a co-serving job take them.
if [ -n "${REFLECTION_JOB:-}" ]; then
  API_KEY=$(reflection_key "$REFLECTION_JOB") || exit 2
else
  API_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
fi
[ -n "$API_KEY" ] || { echo "no key for the job" >&2; exit 3; }
( umask 077; printf '%s\n' "$API_KEY" > "$KEY_FILE" )
export API_KEY
if [ "${SERVE_REACH:-}" = nodes ]; then SERVE_HOST=$(hostname -i | awk '{ print $1 }'); else SERVE_HOST=127.0.0.1; fi
[ -n "$SERVE_HOST" ] || { echo "no address of this node from hostname -i" >&2; exit 3; }
export SERVE_HOST

free_port() { python -c 'import os, socket; s = socket.socket(); s.bind((os.environ["SERVE_HOST"], 0)); print(s.getsockname()[1])'; }

# The server module starts through a one line wrapper that appends the key option from the environment variable
# CD_ENDPOINT_KEY, which Apptainer passes into the container, so that the key stands in no argument of a process.
SERVER_WRAPPER='import os, runpy, sys; sys.argv = ["sglang.launch_server", *sys.argv[1:], "--api-key", os.environ.pop("CD_ENDPOINT_KEY")]; runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)'

launch_server() {
  # Starts SGLang in the background on the model folder $1 as the served name $2 at port $3 with the flags $4, logging
  # to $5, on the address SERVE_HOST behind the key of the job; the caller reads the pid from $!, which is why this runs in
  # the job's own shell and not in a substitution. The models another user of the cluster shares sit under SHARED, and their S may not exist, so
  # each folder is bound only when it is there.
  local binds=(--bind "$L") d
  for d in "$S" "$SHARED"; do [ -n "$d" ] && [ -d "$d" ] && binds+=(--bind "$d"); done
  # shellcheck disable=SC2086
  CD_ENDPOINT_KEY=$API_KEY apptainer exec --nv "${binds[@]}" "$SANDBOX" python3 -c "$SERVER_WRAPPER" \
    --model-path "$1" --served-model-name "$2" --host "$SERVE_HOST" --port "$3" \
    --context-length "$CONTEXT_LENGTH" $4 > "$5" 2>&1 &
}

server_up() {
  # Waits up to $3 tries of ten seconds for the server at port $1 with pid $2 to pass its health check, then lists its models.
  for _ in $(seq 1 "$3"); do
    key_curl "http://$SERVE_HOST:$1/health" >/dev/null && break
    kill -0 "$2" 2>/dev/null || return 1
    sleep 10
  done
  key_curl "http://$SERVE_HOST:$1/v1/models"
}

key_not_in_ps() {
  # The arguments of every process of this user on the node, searched for the key by a process that reads the key from its
  # environment, so that the search shows no key either.
  if CD_ENDPOINT_KEY=$API_KEY python -c 'import os, subprocess, sys
key = os.environ["CD_ENDPOINT_KEY"]
listing = subprocess.run(["ps", "-ww", "-u", str(os.getuid()), "-o", "args="], capture_output=True, text=True)
sys.exit(2 if listing.returncode else 3 if key in listing.stdout else 0)'; then
    echo "KEY_NOT_IN_PS ok"
  else
    echo "KEY_NOT_IN_PS failed: the key stands in the arguments of a process, or the processes could not be listed" >&2
    return 3
  fi
}

TEMPLATE_HASH='import hashlib, json, sys
template = json.load(open(sys.argv[1], encoding="utf-8")).get("chat_template")
text = template if isinstance(template, str) else json.dumps(template, sort_keys=True)
print(hashlib.sha256(text.encode("utf-8")).hexdigest())'

template_record() {
  # The last line of the endpoint file $1 of the model folder $2: the template the server was given, the rewritten one or
  # the model's own, with its SHA-256. A template that cannot be read leaves the hash out, so a ledger names it missing.
  local sum
  if [ "$1" = "$ENDPOINT_FILE" ] && [ "$TEMPLATE_KIND" = tool_response ]; then
    echo "chat_template tool_response template_sha256 $(sha256_of "$L/agent_chat_template.jinja")"
    return
  fi
  sum=$(python -c "$TEMPLATE_HASH" "$2/tokenizer_config.json" 2>/dev/null)
  echo "chat_template model${sum:+ template_sha256 $sum}"
}

engine_record() {
  # The engine lines of an endpoint file for the server at port $1, started on the model folder $2 as $3 with the flags $4
  # for the model $5: engine_record.py reads the server's /server_info, keeps its version, dtype, quantization and
  # kv_cache_dtype alone, and writes the launch flags in public form. The status is engine_record.py's, 3 when a field is
  # missing, and nothing of the answer but those four fields reaches a file or the log.
  local commit
  commit=$(weights_record "$2" | head -n 1 | cut -d' ' -f2)
  # shellcheck disable=SC2086
  key_curl "http://$SERVE_HOST:$1/server_info" | python -m CausalDiagnosis.experiments.cluster_runs.engine_record --model-id "$5" \
    --revision "$commit" --fork "$FORK" -- --model-path "$2" --served-model-name "$3" --host "$SERVE_HOST" --port "$1" \
    --context-length "$CONTEXT_LENGTH" $4
}

write_endpoint() {
  # Writes the base URL $3 of the model $4 served as $2 to the file $1 and links the file under the served name, which
  # is how the scripts of the job find a model's endpoint. The lines after the URL are key and value pairs that the run
  # ledgers copy: the job, the weights commit of the model folder $5, the engine, the chat template and the LoRA adapters $6
  # that the server holds beside the model, which run.json records as the serving setup of a run, and the engine lines $7
  # of engine_record. No line holds the key.
  local template
  template=$(template_record "$1" "$5")
  wait "$SHA_PID" 2>/dev/null
  {
    printf '%s\n%s\n%s\n' "$3" "job $SLURM_JOB_ID node $(hostname) model $4 served_name $2" "started $(date -Iseconds)"
    echo "$(card_line) gpu_sampler $SAMPLER_FILE"
    weights_record "$5"
    echo "sglang $SGLANG_VERSION"
    printf '%s\n' "$7"
    echo "lora_adapters $6"
    echo "image $SIF"
    echo "image_sha256 $(cat "$L/image_sha256.txt" 2>/dev/null)"
    echo "$template"
  } > "$1"
  ln -sfn "$1" "$ENDPOINTS/$2.txt"
  echo "endpoint $3 written to $1"
}

if [ -n "$CO_SERVE_JUDGE" ]; then
  [ -n "$JUDGE_MODEL" ] || { echo "CO_SERVE_JUDGE is set but JUDGE_MODEL is empty" >&2; exit 2; }
  JUDGE_DIR=$(model_dir "$JUDGE_MODEL")
  [ -d "$JUDGE_DIR" ] || { echo "judge model folder $JUDGE_DIR missing" >&2; exit 2; }
  JUDGE_PORT=$(free_port)
  launch_server "$JUDGE_DIR" "$JUDGE_SERVED_NAME" "$JUDGE_PORT" "$JUDGE_SGLANG_EXTRA" "$LOGS/sglang-judge-$SLURM_JOB_ID.log"
  JUDGE_PID=$!
  if server_up "$JUDGE_PORT" "$JUDGE_PID" "$JUDGE_UP_TRIES"; then
    echo
    JUDGE_BASE_URL=http://$SERVE_HOST:$JUDGE_PORT/v1
    export JUDGE_BASE_URL
    JUDGE_ENGINE=$(engine_record "$JUDGE_PORT" "$JUDGE_DIR" "$JUDGE_SERVED_NAME" "$JUDGE_SGLANG_EXTRA" "$JUDGE_MODEL") || {
      echo "the engine record of the judge $JUDGE_MODEL failed, so no endpoint file of it is written" >&2; exit 3; }
    printf '%s\n' "$JUDGE_ENGINE" | sed -n 's/^dtype /ENGINE_PRECISION judge dtype /p'
    write_endpoint "$JUDGE_ENDPOINT_FILE" "$JUDGE_SERVED_NAME" "$JUDGE_BASE_URL" "$JUDGE_MODEL" "$JUDGE_DIR" 0 "$JUDGE_ENGINE"
  else
    echo "judge $JUDGE_MODEL not answering; see $LOGS/sglang-judge-$SLURM_JOB_ID.log; the run goes on without the judge" >&2
    kill "$JUDGE_PID" 2>/dev/null; wait "$JUDGE_PID" 2>/dev/null; JUDGE_PID=; sleep 10
    JUDGE_MODEL=; export JUDGE_MODEL   # so that the run scripts do not wait for a judge endpoint
  fi
fi

# With SERVE_ADAPTERS=1 the agent model is served with every LoRA adapter under $HYD_TRAINING/adapters, each under the name of
# its folder, which a request selects with the model <served name>:<adapter> (SGLang v0.5.19), so that the variants of the
# training experiment play against one server and differ in the request alone. The base weights take a slot of the batch too.
LORA_ADAPTERS=0
if [ -n "${SERVE_ADAPTERS:-}" ]; then
  ADAPTERS=()
  for d in "$HYD_TRAINING"/adapters/*/; do
    [ -s "${d}adapter_config.json" ] && ADAPTERS+=("$(basename "$d")=${d%/}")
  done
  [ ${#ADAPTERS[@]} -gt 0 ] || { echo "SERVE_ADAPTERS is set and no adapter lies under $HYD_TRAINING/adapters" >&2; exit 2; }
  SGLANG_EXTRA="$SGLANG_EXTRA --enable-lora --max-loras-per-batch $(( ${#ADAPTERS[@]} + 1 )) --lora-paths ${ADAPTERS[*]}"
  LORA_ADAPTERS=${#ADAPTERS[@]}
  echo "LoRA adapters served: ${ADAPTERS[*]}"
fi
# granite reads a tool result under the role its own template prints, and SGLang's OpenAI endpoint passes no role but
# tool, so with MODEL_ADAPTER the agent model is served with its template changed to print a tool response
# (CausalDiagnosis/agents/utils/model_adapters.py). A model whose template no entry recognizes keeps its own, and a rewrite
# that fails for any other reason stops the job.
AGENT_FLAGS=$SGLANG_EXTRA
# a job whose one model is the judge, a judge job or the reflection model of the prompt optimizer, serves it with the flags
# of a judge served alone, the reasoning parser among them, so its JSON never comes mixed with its thinking
# and the agent model with the dtype of AGENT_DTYPE, so that the precision of record is the one asked for and not SGLang's auto
if [ -n "${JUDGE_MODEL:-}" ] && [ "$SERVED_NAME" = "$JUDGE_SERVED_NAME" ]; then AGENT_FLAGS=$JUDGE_ALONE_SGLANG_EXTRA
else
  [ -z "${AGENT_DTYPE:-}" ] || AGENT_FLAGS="--dtype $AGENT_DTYPE $AGENT_FLAGS"
  # the agent model reports the cached tokens of every request, which the cache flush before a wall-clock episode reads;
  # the serving file of record declares the flag in both launch modes, before the chat template
  AGENT_FLAGS="$AGENT_FLAGS --enable-cache-report"
fi
TEMPLATE_KIND=model
if [ -n "${MODEL_ADAPTER-1}" ]; then
  python -m CausalDiagnosis.agents.utils.model_adapters "$MODEL_DIR/tokenizer_config.json" "$L/agent_chat_template.jinja"
  case $? in
    0) AGENT_FLAGS="$AGENT_FLAGS --chat-template $L/agent_chat_template.jinja"; TEMPLATE_KIND=tool_response ;;
    3) echo "the chat template of $MODEL keeps its own form, since it lacks the expression of a tool result" ;;
    *) echo "the chat template of $MODEL could not be rewritten" >&2; exit 3 ;;
  esac
fi
PORT=${SGLANG_PORT:-$(free_port)}
launch_server "$MODEL_DIR" "$SERVED_NAME" "$PORT" "$AGENT_FLAGS" "$LOGS/sglang-$SLURM_JOB_ID.log"
SGL_PID=$!
server_up "$PORT" "$SGL_PID" 180 || { echo "SGLang exited or not answering after 30 minutes; see $LOGS/sglang-$SLURM_JOB_ID.log" >&2; exit 3; }
echo
key_not_in_ps || exit 3
BASE_URL=http://$SERVE_HOST:$PORT/v1
ENGINE=$(engine_record "$PORT" "$MODEL_DIR" "$SERVED_NAME" "$AGENT_FLAGS" "$MODEL") || {
  echo "the engine record of $MODEL failed, so no endpoint file is written" >&2; exit 3; }
# the one line of the log that says what precision the image settled on, read before any run is played on it
printf '%s\n' "$ENGINE" | sed -n 's/^dtype /ENGINE_PRECISION dtype /p'
write_endpoint "$ENDPOINT_FILE" "$SERVED_NAME" "$BASE_URL" "$MODEL" "$MODEL_DIR" "$LORA_ADAPTERS" "$ENGINE"
ln -sfn "$ENDPOINT_FILE" "$ENDPOINTS/latest.txt"
tail -n +2 "$ENDPOINT_FILE"

warn_script() {
  # The warning signal of the time limit: the script gets it and SCRIPT_GRACE seconds to end, then the servers stop.
  WARNED=1
  echo "the time limit is near: the script gets the signal and $SCRIPT_GRACE seconds, then the server stops"
  kill -TERM "$SCRIPT_PID" 2>/dev/null
  local waited=0
  while kill -0 "$SCRIPT_PID" 2>/dev/null && [ "$waited" -lt "$SCRIPT_GRACE" ]; do sleep 1; waited=$((waited + 1)); done
  stop_servers
}

watch_services() {
  # While the script runs: a server of this job that exits, the agent's or the co-served judge's, ends the job at once with
  # SERVICE_EXITED <role> <status>, so that no episode waits out its wall limit on a dead service. The script and its children
  # get TERM through their process group, then SCRIPT_GRACE seconds, then KILL. The sleep runs in the background so that the
  # warning signal of the time limit is taken at once; warn_script empties the server pids, so its own stop is no exit here.
  local role pid status waited
  while kill -0 "$SCRIPT_PID" 2>/dev/null; do
    for role in agent judge; do
      if [ "$role" = agent ]; then pid=$SGL_PID; else pid=$JUDGE_PID; fi
      { [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; } || continue
      wait "$pid" 2>/dev/null
      status=$?
      echo "SERVICE_EXITED $role $status"
      if [ "$role" = agent ]; then SGL_PID=; else JUDGE_PID=; fi
      kill -TERM -- -"$SCRIPT_PID" 2>/dev/null
      waited=0
      while kill -0 "$SCRIPT_PID" 2>/dev/null && [ "$waited" -lt "$SCRIPT_GRACE" ]; do sleep 1; waited=$((waited + 1)); done
      kill -KILL -- -"$SCRIPT_PID" 2>/dev/null
      SERVICE_EXITED=$role
      return
    done
    sleep 5 &
    wait $!
  done
}

if [ -n "${RUN_AFTER_SERVE:-}" ]; then
  # The script runs in the background and the job waits for it, so that the trap of the warning signal fires at once.
  export BASE_URL ENDPOINT_FILE
  # set -m gives the script a process group of its own, which watch_services ends as a whole
  set -m
  bash "$RUN_AFTER_SERVE" &
  SCRIPT_PID=$!
  set +m
  trap warn_script TERM
  watch_services
  if [ -n "$SERVICE_EXITED" ]; then stop_servers; exit 3; fi
  wait "$SCRIPT_PID"
  STATUS=$?
  if [ -n "$WARNED" ]; then wait "$SCRIPT_PID" 2>/dev/null; AGAIN=$?; [ "$AGAIN" = 127 ] || STATUS=$AGAIN; fi
  echo "RUN_AFTER_SERVE finished with $STATUS"
  stop_servers
  echo "CD_SERVE_DONE"
  exit "$STATUS"
fi
trap 'WARNED=1' TERM
echo "serving at $BASE_URL until the job is cancelled or until $STOP_FILE appears"
while [ -z "$WARNED" ] && [ ! -e "$STOP_FILE" ] && kill -0 "$SGL_PID" 2>/dev/null; do sleep 10; done
if [ -e "$STOP_FILE" ]; then echo "the stop file appeared"; elif [ -n "$WARNED" ]; then echo "the time limit is near"; else echo "the server ended on its own"; fi
stop_servers
echo "CD_SERVE_DONE"
