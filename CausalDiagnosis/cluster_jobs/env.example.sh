# The values of the cluster jobs of this tree, the template of env.sh. Copy it to env.sh beside it, which git ignores, and
# set ROOT there. Every job script sources env.sh through HPC_DIR, which submit.sh exports to the job, and a variable
# given on the submit line overrides a default here. What differs between two people stands in env.sh alone and has no
# value here: ROOT, the account, the partitions, the nodes to leave out, the nodes of record and a remote route with its key.
# S, HOME_FS and SHARED hold the values of this cluster. Everything large lies under ROOT on the scratch filesystem, and
# everything temporary on node local disk inside a job. The functions of every job stand in functions.sh beside this file,
# which the last line sources, so that env.sh holds values alone; an env.sh made before functions.sh existed lacks that
# line, so every person makes env.sh again from this template once.
S=${S:-/insomnia001/depts/edu/users/$USER}            # your scratch folder on this cluster, for a copy of your own of the SGLang image and the models
HOME_FS=${HOME_FS:-/insomnia001/home/$USER}           # your home filesystem on this cluster, which holds nothing of a run
ROOT=${ROOT:-}                                        # a folder of your own on the scratch filesystem, readable by you alone
[ -n "$ROOT" ] || { echo "set ROOT in env.sh to a folder of your own on the scratch filesystem" >&2; exit 2; }
case "$ROOT/" in
  "$HOME_FS"/*) echo "ROOT $ROOT lies in the home filesystem: set ROOT in env.sh to a folder of your own on the scratch filesystem" >&2; exit 2 ;;
esac
[ -d "$ROOT" ] || mkdir -p -m 700 "$ROOT" || { echo "could not make ROOT at $ROOT" >&2; exit 2; }
CLONES=${CLONES:-$ROOT/clones}                        # one clone per pushed commit, named by its first 12 digits, never updated
ENVS=${ENVS:-$ROOT/envs}                              # one Python environment per lock file, named by the first 12 digits of its SHA-256
DATA=${DATA:-$ROOT/data}                              # the archive of the data of record, fetched once by fetch_data.sbatch
# IMAGES, the folder of the CouchDB image of the AssetOpsBench rows, has its default in the job_services.sh of that framework
# CACHE and the rule of every fetch from outside the cluster, FETCH_TIMEOUT, FETCH_RETRIES and FETCH_PAUSE, have their defaults in functions.sh
LOGS=${LOGS:-$ROOT/logs}                              # the log of every job and of every server
ENDPOINTS=${ENDPOINTS:-$ROOT/endpoints}               # the endpoint file of every served model and the key file of every serving job
SUITES=${SUITES:-$ROOT/suites}                        # the reports of run_suite.sbatch
PLATFORM=${PLATFORM:-$ROOT/platform}                  # the optional platform records of platform_record.sh
DIGESTS=${DIGESTS:-$ROOT/digests}                     # the labels of the digest families
export HYD_TRAINING=${HYD_TRAINING:-$ROOT/training}   # the training files: examples, adapters, checkpoints and blocks
TRAIN_VENV=${TRAIN_VENV:-$ROOT/train_venv}            # the training environment of setup_train_env.sbatch
FORK_URL=${FORK_URL:-https://github.com/Edward-H26/AssetOpsBench.git}
SHARED=${SHARED:-}   # the group readable folder of this cluster with the SGLang image and the models, whose value comes with the message that names the commit; empty: copies of your own under S
MODELS=${MODELS:-$S/models}                           # model weights downloaded by download_model.sbatch, one folder per model
MODEL=${MODEL:-ibm-granite/granite-3.3-8b-instruct}   # the Hugging Face id of the agent model, or the absolute path of a model folder
SERVED_NAME=${SERVED_NAME:-${MODEL##*/}}              # the model name the endpoint advertises and the drivers send
# The Hub commit of every model that a job downloads stands in the tracked serving file of record,
# CausalDiagnosis/experiments/table_definitions/serving_of_record.json, which model_revision reads, so env.sh pins no model.
SIF_NAME=sglang-v0.5.19-cu129.sif                     # the SGLang v0.5.19 image
if [ -z "${SIF:-}" ]; then                            # a SIF given on the submit line is taken as it is
  SIF=$S/images/$SIF_NAME                             # your own copy, written by pull_image.sbatch
  if [ ! -s "$SIF" ] && [ -n "$SHARED" ] && [ -r "$SHARED/images/$SIF_NAME" ]; then SIF=$SHARED/images/$SIF_NAME; fi
fi
SGLANG_PORT=${SGLANG_PORT:-}                          # empty: the serving job picks a free port
SGLANG_EXTRA=${SGLANG_EXTRA:-}                        # extra launch_server flags of the agent model
AGENT_DTYPE=${AGENT_DTYPE-bfloat16}                   # the --dtype of the agent model (its checkpoint is BF16), empty: SGLang's auto
MODEL_ADAPTER=${MODEL_ADAPTER-1}          # 1: the agent model is served with its chat template changed so that a tool result reads as a tool response, empty: its own template
CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}              # the full context of the agent model and of the judge, since the judge reads the whole trajectory
UV_VERSION=${UV_VERSION:-0.11.7}                      # the uv that wrote uv.lock
UV_BIN=${UV_BIN:-$ROOT/tools/uv-$UV_VERSION/uv}       # where setup_env.sbatch installs that uv
API_KEY=${API_KEY:-}                                  # the one key the drivers read: a serving job sets the key of its server, a remote route needs its key here
[ -z "$API_KEY" ] || export API_KEY
ACCOUNT=${ACCOUNT:-}                                  # the account of every job, set in env.sh
PARTITION_CPU=${PARTITION_CPU:-}                      # the partition of a job without a card, set in env.sh
PARTITION_GPU=${PARTITION_GPU:-}                      # the partition of a job with a card, one without preemption, set in env.sh
PARTITION_GPU_RESUMABLE=${PARTITION_GPU_RESUMABLE:-}  # the partitions of a job with a card that resumes by itself after a requeue, set in env.sh
EXCLUDE_NODES=${EXCLUDE_NODES:-}                      # the nodes no job may take, a comma separated list
NODES_OF_RECORD=${NODES_OF_RECORD:-}                  # optional: the nodes whose digest check passed (check_nodes.sh); once set, a job of record takes only these
GPU_OF_RECORD=${GPU_OF_RECORD:-}                      # optional: the card that every job with a card checks once set, its name exactly as nvidia-smi prints it
GPU_MIB_OF_RECORD=${GPU_MIB_OF_RECORD:-}              # the memory.total of that card in MiB, as nvidia-smi prints it without the unit
TABLES=${TABLES:-"CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json CausalDiagnosis/datasets/hyd_1/data/labels.json"}   # the generated tables, as paths of the tree
# The judge must be another model than the agent, since the fork's evaluator refuses self judging. JUDGE_MODEL names the
# judge by its Hugging Face id or its folder, served by a job of its own (run_judge.sh after the grid, the reflection model of
# the loops of GEPA) or, with CO_SERVE_JUDGE=1, by the serving job of the agent model on the same card before the agent
# model, or JUDGE_BASE_URL names a remote route with API_KEY. A run judges its episodes in the run only when JUDGE_BASE_URL
# names the judge, which a co-served judge sets, and run_judge.sh scores a grid after the grid.
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.8-27B-FP8}
JUDGE_SERVED_NAME=${JUDGE_SERVED_NAME:-${JUDGE_MODEL##*/}}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-}
JUDGE_WAIT_TRIES=${JUDGE_WAIT_TRIES:-1080}            # tries of ten seconds to wait for the judge endpoint, since a second card can queue long
# With CO_SERVE_JUDGE=1 one card carries both models: the judge first with the static memory share below, then the agent
# with a share that counts the judge's footprint, since SGLang sizes its cache from the memory other processes already hold.
# A judge that does not answer is dropped and the run goes on without it. The reasoning parser keeps the judge's JSON apart
# from its thinking. A judge served alone, as the judge job or the reflection model of the prompt optimizer, takes the
# parser and the graph batch without the share of two models on one card.
CO_SERVE_JUDGE=${CO_SERVE_JUDGE:-}
JUDGE_SGLANG_EXTRA=${JUDGE_SGLANG_EXTRA:-"--mem-fraction-static 0.45 --cuda-graph-max-bs 16 --reasoning-parser qwen3"}
JUDGE_ALONE_SGLANG_EXTRA=${JUDGE_ALONE_SGLANG_EXTRA:-"--cuda-graph-max-bs 16 --reasoning-parser qwen3"}
JUDGE_UP_TRIES=${JUDGE_UP_TRIES:-180}                 # tries of ten seconds for the co-served judge to answer its health check
if [ -n "$CO_SERVE_JUDGE" ]; then SGLANG_EXTRA=${SGLANG_EXTRA:-"--mem-fraction-static 0.85 --cuda-graph-max-bs 16"}; fi
SERVE_ADAPTERS=${SERVE_ADAPTERS:-}                    # 1: the serving job loads every adapter under HYD_TRAINING beside the agent model


# HPC_DIR, FORK, GRIDS and the functions of every job, from the tracked file beside this one
source "$(dirname "${BASH_SOURCE[0]}")/functions.sh"
