#!/bin/bash
#SBATCH -J cd-train-setup
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
# Builds the training environment of the LoRA adapters of the training experiment (CausalDiagnosis/cluster_jobs/README.md) on a CPU node, a virtual
# environment of its own at TRAIN_VENV (ROOT/train_venv by default) with the versions that requirements-train.txt pins, torch
# from the CUDA 12.6 index of PyTorch first. The environment of the lock stays as it is. The job needs the uv that
# setup_env.sbatch installs, keeps its Python under ROOT/tools, and writes the frozen list of the environment beside the logs.
# Both installs run under retry_fetch of functions.sh with the timeout of the fetches, and uv keeps its cache under CACHE on
# scratch, so the wheels that one attempt fetched serve the next attempt and the next job.
# The limit of two hours: on a compute node of the cluster the build took 18 minutes under the timeout of the fetches, after two
# builds under uv's default timeout of 30 seconds had stopped on a timeout of the package index, while a node fetched at 1 to
# 2 MB/s and the login node at several hundred. A retry, of an attempt inside the job or of the job, resumes from the wheels in
# the cache under ROOT, so it fetches only what is missing.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
export UV_CACHE_DIR=$CACHE/uv UV_HTTP_TIMEOUT=$FETCH_TIMEOUT TMPDIR=$L/tmp UV_PYTHON_INSTALL_DIR=$ROOT/tools/python UV_PYTHON_INSTALL_BIN=0
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
[ -x "$UV_BIN" ] || { echo "no uv at $UV_BIN; run setup_env.sbatch first" >&2; exit 2; }
export PATH=$(dirname "$UV_BIN"):$PATH
job_header "training environment at $TRAIN_VENV"; df -h "$(dirname "$TRAIN_VENV")" | tail -1
[ -x "$TRAIN_VENV/bin/python" ] || uv venv --python "$PYTHON_VERSION" "$TRAIN_VENV"
TORCH=$(grep -E '^torch==' "$HPC_DIR/0_setup/requirements-train.txt")
retry_fetch "$UV_CACHE_DIR" uv pip install --python "$TRAIN_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu126 "$TORCH" || exit 3
retry_fetch "$UV_CACHE_DIR" uv pip install --python "$TRAIN_VENV/bin/python" -r "$HPC_DIR/0_setup/requirements-train.txt" || exit 3
"$TRAIN_VENV/bin/python" -c "import torch, transformers, trl, peft, datasets; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__, 'datasets', datasets.__version__)" || exit 3
uv pip freeze --python "$TRAIN_VENV/bin/python" > "$LOGS/train-venv-freeze-${SLURM_JOB_ID:-manual}.txt"
echo "CD_TRAIN_SETUP_DONE"
