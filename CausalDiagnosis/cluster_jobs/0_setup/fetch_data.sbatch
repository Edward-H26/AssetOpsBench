#!/bin/bash
#SBATCH -J cd-data
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
# The data of record on a CPU node: the one fetch and build command of the data of record, which keeps the archive from the
# official page in DATA, downloaded once, and builds the files of record into the ignored data folder of this clone, then the
# fetch of the scenario file of the anchored preset at its pinned revision into the same folder, where the runs of that
# preset read it, then the check of the files of record against the pins of the tracked CHECKSUMS.txt. Both fetches follow the
# retry policy of every fetch (CausalDiagnosis/utils/download_retry.py, with the settings of functions.sh), and an archive whose
# download broke off stays in DATA as a partial file, from which the next attempt or the next job goes on.
#   bash $H/submit.sh fetch_data.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"

fetch_and_build() {
  # The one fetch and build command of the data of record: the archive is downloaded into DATA when it is not there and
  # checked like a fresh one, and the files are built into the data folder of the clone; its last line is DATA_OK.
  python -m CausalDiagnosis.datasets.hyd_1.download_and_build --download --dest "$DATA"
}

fetch_data() {
  job_header "data of record into $DATA"
  fork_python
  cd "$FORK" || return 2
  mkdir -p "$DATA"
  fetch_and_build || return 3
  # the data of the frameworks of the tree, the commands that their job_services.sh lists
  local fetch
  for fetch in ${DATA_FETCHES[@]+"${DATA_FETCHES[@]}"}; do $fetch || return 3; done
  check_data
  echo "CD_DATA_DONE"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then fetch_data; exit $?; fi
