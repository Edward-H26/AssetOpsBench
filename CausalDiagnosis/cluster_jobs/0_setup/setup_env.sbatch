#!/bin/bash
#SBATCH -J cd-setup
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
# Builds the Python environment of the lock of this tree on a CPU node, once per lock: uv sync --frozen into ENVS/<lock>, the
# form of the laptop's environment, with the uv that wrote the lock and its Python under ROOT/tools, so that the home folder
# gains no file. Every clone with the same lock uses that environment, and fork_python of env.sh imports the code of the
# clone that runs. An environment whose sync finished, which the mark .cd_synced with the SHA-256 of the lock states, is checked
# and not built again, and any other is synced again. The job records the installed list in the
# environment and prints the SHA-256 of the lock and of that list. It checks the environment alone, since the data of record
# is built after it, by fetch_data.sbatch. run_suite.sbatch is the acceptance of an environment.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"

L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
export UV_CACHE_DIR=$L/uv-cache TMPDIR=$L/tmp UV_PYTHON_INSTALL_DIR=$ROOT/tools/python UV_PYTHON_INSTALL_BIN=0
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
job_header "environment of the lock $(lock_id)"; df -h "$ROOT" "$L" | cat

if [ ! -x "$UV_BIN" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$(dirname "$UV_BIN")" sh
fi
[ -x "$UV_BIN" ] || { echo "no uv $UV_VERSION at $UV_BIN" >&2; exit 3; }
export PATH=$(dirname "$UV_BIN"):$PATH
uv --version

cd "$FORK" || exit 2
LOCK=$(lock_id) || exit 2
ENV=$ENVS/$LOCK
if env_synced "$ENV"; then
  echo "the environment of this lock exists at $ENV, so it is checked and not built again"
else
  uv python install "$PYTHON_VERSION" || exit 3
  UV_PROJECT_ENVIRONMENT=$ENV UV_LINK_MODE=copy uv sync --frozen --python "$PYTHON_VERSION" || exit 3
  sha256_of uv.lock > "$ENV/.cd_synced"
fi
fork_python
python --version
uv pip freeze --python "$ENV/bin/python" | grep -v '^-e ' > "$ENV/installed.txt" || exit 3
sha256sum uv.lock "$ENV/installed.txt"
echo "CD_SETUP_DONE"
