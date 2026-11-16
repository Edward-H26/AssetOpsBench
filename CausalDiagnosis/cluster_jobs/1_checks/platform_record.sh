#!/bin/bash
# An optional platform record, off the path of CausalDiagnosis/cluster_jobs/README.md, the RUN_AFTER_SERVE script of a serving job of the agent model on a node of
# record: run_ledger/ platform writes what a run would record there, the serving identity of the endpoint file of the job
# and of the judge endpoint that run_judge.sh kept in PLATFORM, and the SHA-256 of the CouchDB image, into PLATFORM/<tree>.json.
# The record names the card of the agent and of the judge,
# so it needs the card of record in env.sh and stops with GPU_NOT_OF_RECORD and status 2, before it writes
# anything, when either endpoint file names another card. It names the judge as well, so the record
# stops with status 2 without the judge endpoint record that the judge job of the registered order keeps. run_ledger/ platform
# checks each endpoint record against the models of env.sh and writes nothing on a difference: HYD_MODEL and HYD_JUDGE_MODEL
# against the served names, which the run and judge records carry, HYD_HUB_ID and HYD_JUDGE_HUB_ID against the Hugging Face
# ids, and HYD_REVISION and HYD_JUDGE_REVISION against the commits that the serving file of record pins.
#   RUN_AFTER_SERVE=$H/1_checks/platform_record.sh bash $H/submit.sh serve_model.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"

platform_record() {
  local tree out la endpoint card
  fork_python
  cd "$FORK" || return 2
  tree=$(tree_id) || return 2
  out=$PLATFORM/$tree.json
  mkdir -p "$PLATFORM"
  require_gpu_of_record
  if [ ! -s "$PLATFORM/judge_endpoint_$tree.txt" ]; then
    echo "no judge endpoint record of this tree in $PLATFORM, so the record would name no judge: run the judge job first" >&2
    return 2
  fi
  for endpoint in "$ENDPOINT_FILE" "$PLATFORM/judge_endpoint_$tree.txt"; do
    card=$(endpoint_card "$endpoint")
    card_of_record "${card%|*}" "${card#*|}" || { echo "GPU_NOT_OF_RECORD $endpoint names ${card:-no card}"; return 2; }
  done
  # the images of the services of the frameworks, which the platform record holds (the job_services.sh of each framework)
  la=$(node_local_dir)/platform_$$
  mkdir -p "$la"
  for record in ${PLATFORM_RECORDS[@]+"${PLATFORM_RECORDS[@]}"}; do "$record" "$la"; done
  rm -rf "$la"
  HYD_ENDPOINT_FILE=$ENDPOINT_FILE HYD_JUDGE_ENDPOINT_FILE=$PLATFORM/judge_endpoint_$tree.txt \
    HYD_MODEL=$SERVED_NAME HYD_HUB_ID=$MODEL HYD_REVISION=$(model_revision "$MODEL") \
    HYD_JUDGE_MODEL=$JUDGE_SERVED_NAME HYD_JUDGE_HUB_ID=$JUDGE_MODEL HYD_JUDGE_REVISION=$(model_revision "$JUDGE_MODEL") \
    python -m CausalDiagnosis.experiments.run_verification.run_ledger platform --out "$out" || return 3
  echo "CD_PLATFORM_DONE $out"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then platform_record; exit $?; fi
