#!/bin/bash
# Copies the run folders from the cluster to the laptop and proves the copy. The cluster first hashes every file of its runs
# folder with sha256sum, the laptop fetches only the files whose hash its copy lacks, as a tar stream over ssh since scp is
# unsupported on Insomnia, and CausalDiagnosis/experiments/run_verification/check_synced_runs.py hashes the copy again. A file of a finished run that
# differs from the cluster's hash stops the script with no ledger touched, a run still in progress is only reported, and
# every finished run with a ledger gets a sync event that names the files the sync replaced. Nothing is published or
# aggregated into the fork, since no number from a run enters it before the test set runs, and no
# file is deleted on either side. Run on the laptop after `ssh insomnia` has a live master connection. Usage:
#   bash CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh <ROOT>/runs/<kind>/<tree> [local folder]
# The remote folder is required, since the laptop does not know the UNI, and the local default is CausalDiagnosis/cluster_jobs/runs_synced/, which
# git ignores. SSH and HOST name the ssh command and the host (a local stand in for a dry run).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh
REMOTE=${1:?usage: bash CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh <ROOT>/runs/<kind>/<tree> [local folder]}
LOCAL=${2:-$HERE/runs_synced}
FORK=${FORK:-$HERE}
while [ ! -f "$FORK/uv.lock" ] && [ "$FORK" != / ]; do FORK=$(dirname "$FORK"); done   # the root of the tree, the folder of uv.lock
SSH=${SSH:-ssh}
HOST=${HOST:-insomnia}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
CHECK=(env PYTHONPATH="$FORK:$FORK/src" "$FORK/.venv/bin/python" -m CausalDiagnosis.experiments.run_verification.check_synced_runs)
mkdir -p "$LOCAL"
$SSH -o BatchMode=yes "$HOST" true || { echo "no ssh master for $HOST, log in once with: ssh $HOST" >&2; exit 2; }
$SSH "$HOST" "cd '$REMOTE' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum" > "$STAGE/remote.sha256" \
  || { echo "hashing $HOST:$REMOTE failed" >&2; exit 3; }
"${CHECK[@]}" plan --manifest "$STAGE/remote.sha256" --local "$LOCAL" --plan "$STAGE/plan.json" > "$STAGE/wanted.txt" || exit 3
if [ -s "$STAGE/wanted.txt" ]; then
  $SSH "$HOST" "cd '$REMOTE' && tar czf - -T -" < "$STAGE/wanted.txt" | tar xzf - -C "$LOCAL" \
    || { echo "the transfer from $HOST:$REMOTE failed" >&2; exit 3; }
fi
echo "fetched $(wc -l < "$STAGE/wanted.txt" | tr -d ' ') of $(wc -l < "$STAGE/remote.sha256" | tr -d ' ') files from $HOST:$REMOTE into $LOCAL"
"${CHECK[@]}" check --manifest "$STAGE/remote.sha256" --local "$LOCAL" --plan "$STAGE/plan.json" --source "$HOST:$REMOTE"
STATUS=$?
if [ "$STATUS" = 0 ]; then
  echo "SYNC_OK"
else
  echo "SYNC_MISMATCH: no ledger was touched, so the sync can simply run again" >&2
fi
exit "$STATUS"
