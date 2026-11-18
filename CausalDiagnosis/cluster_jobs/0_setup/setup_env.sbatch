#!/bin/bash
#SBATCH -J cd-setup
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
# Builds the Python environment of the lock of this tree on a CPU node, once per lock: uv sync --frozen into ENVS/<lock>, the
# form of the laptop's environment, with the uv that wrote the lock and its Python under ROOT/tools, so that the home folder
# gains no file. Every clone with the same lock uses that environment, and fork_python of env.sh imports the code of the
# clone that runs. An environment whose sync finished, which the mark .cd_synced with the SHA-256 of the lock states, is checked
# and not built again, and any other is synced again. Every fetch of the job, the installer of uv, the Python and the packages
# of the lock, runs under retry_fetch of functions.sh, uv and the fetch of the installer script under the timeout of the
# fetches, and uv keeps its cache under CACHE on scratch, so the packages that one attempt fetched serve the next attempt and the next job. The job records the installed
# list in the environment and prints the SHA-256 of the lock and of that list. It checks the environment alone, since the data
# of record is built after it, by fetch_data.sbatch. run_suite.sbatch is the acceptance of an environment.
# The limit of two hours: on compute nodes of the cluster a cold sync of the lock took 35 minutes for the 4304 MiB of its large
# wheels, about 2 MiB/s, while a node fetched at 1 to 2 MB/s and the login node at several hundred, and a cold sync whose
# largest wheels started again after a stall ran into these two hours. A retry, of an attempt inside the job or of the job,
# resumes from the packages in the cache under ROOT, so it fetches only what is missing.
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"

L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
export UV_CACHE_DIR=$CACHE/uv UV_HTTP_TIMEOUT=$FETCH_TIMEOUT TMPDIR=$L/tmp UV_PYTHON_INSTALL_DIR=$ROOT/tools/python UV_PYTHON_INSTALL_BIN=0
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
job_header "environment of the lock $(lock_id)"; df -h "$ROOT" "$L" | cat

install_uv() {
  # the installer of the uv that wrote the lock, fetched into a file and then run, which puts uv into the folder of UV_BIN
  curl -LsSf --speed-limit 1 --speed-time "$FETCH_TIMEOUT" -o "$L/uv-installer.sh" "https://astral.sh/uv/$UV_VERSION/install.sh" \
    && env UV_UNMANAGED_INSTALL="$(dirname "$UV_BIN")" sh "$L/uv-installer.sh"
}
if [ ! -x "$UV_BIN" ]; then retry_fetch "$(dirname "$UV_BIN")" install_uv; fi
[ -x "$UV_BIN" ] || { echo "no uv $UV_VERSION at $UV_BIN" >&2; exit 3; }
export PATH=$(dirname "$UV_BIN"):$PATH
uv --version

cd "$FORK" || exit 2
LOCK=$(lock_id) || exit 2
ENV=$ENVS/$LOCK
if env_synced "$ENV"; then
  echo "the environment of this lock exists at $ENV, so it is checked and not built again"
else
  retry_fetch "$UV_PYTHON_INSTALL_DIR" uv python install "$PYTHON_VERSION" || exit 3
  UV_PROJECT_ENVIRONMENT=$ENV UV_LINK_MODE=copy retry_fetch "$UV_CACHE_DIR" uv sync --frozen --python "$PYTHON_VERSION" || exit 3
  sha256_of uv.lock > "$ENV/.cd_synced"
fi
fork_python
python --version
uv pip freeze --python "$ENV/bin/python" | grep -v '^-e ' > "$ENV/installed.txt" || exit 3
sha256sum uv.lock "$ENV/installed.txt"
echo "CD_SETUP_DONE"
