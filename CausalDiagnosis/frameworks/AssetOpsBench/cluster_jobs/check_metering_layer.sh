#!/bin/bash
#SBATCH -J cd-aob-base
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
# The conformance job of the AssetOpsBench rows: one CouchDB for the job, started as run_reproduction.sh starts it but without
# loading AssetOpsBench's default data, then the conformance test, which runs one recorded script of calls through the
# metering layer on that CouchDB and on the memory store and requires the same texts, tool_calls entries and documents, then
# the latency probe on CouchDB. No model and no card take part. The report goes to RUNS/metering_layer/conformance_<job id>, with
# RUNS the smoke folder unless RUN_KIND names another kind. The code is the tree of FORK, unless AOB_BASE_CODE names another
# tree, such as an upload of upload_fork.sh that no other job runs from, and the Python is the environment of the lock. The
# test reads the data of record of that tree, so the job checks it first, as every job does, and stops on missing data before
# CouchDB starts.
#   bash $H/submit.sh check_metering_layer.sh
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
CODE=${AOB_BASE_CODE:-$FORK}
fork_python
[ "$CODE" = "$FORK" ] || export PYTHONPATH=$CODE:$CODE/src
FORK=$CODE check_data
cd "$CODE" || exit 2
RUN_KIND=${RUN_KIND:-smoke} runs_root

OUT=${METERING_LAYER_OUT:-$RUNS/metering_layer}/conformance_${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
LA=$(node_local_dir)/aob_base_$$
mkdir -p "$LA"
COUCH_PID=
cleanup() {
  if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; fi
  rm -rf "$LA"
}
trap cleanup EXIT
export UV_CACHE_DIR=$LA/uv-cache TMPDIR=$LA/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
job_header "code $CODE out $OUT"
# an upload's manifest hash names the exact tree this report tested, and a clone records its git describe, marked dirty
# when the tree differs from the commit, as run_scenarios/ records it in run.json
if [ -f "$CODE/UPLOAD_MANIFEST.txt" ]; then sha256sum "$CODE/UPLOAD_MANIFEST.txt" | tee "$OUT/upload_manifest.sha256"
else git -C "$CODE" describe --always --tags --dirty | tee "$OUT/git_describe.txt"; fi

require_module couchdb3
start_couchdb "$LA"

AOB_BASE_REPORT=$OUT/conformance.json python -m pytest CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/tests/test_conformance.py -q -p no:cacheprovider -rs \
  2>&1 | tee "$OUT/conformance.log"
STATUS=${PIPESTATUS[0]}
python -m CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.latency_probe --backend couchdb 2>&1 | tee "$OUT/latency_couchdb.txt"
cp "$LA/couchdb/couchdb.log" "$OUT/couchdb.log" 2>/dev/null
echo "CONFORMANCE_EXIT $STATUS"
exit "$STATUS"
