#!/bin/bash
# Uploads the fork's working tree from the laptop to a folder on the cluster together with UPLOAD_MANIFEST.txt, one sha256sum
# line per file it ships, so that every run ledger there records the manifest's hash and the laptop can name the commit the
# upload matches. The file list and the manifest come from CausalDiagnosis/experiments/run_verification/upload_manifest.py, which leaves out
# the environment, the repository, caches, the run folders, the walkthrough pages and every generated table, which the cluster
# computes before the first line of a grid, and tar ships exactly that list.
# The cluster then checks every file against the manifest, a copy of the manifest stays on the laptop under CausalDiagnosis/cluster_jobs/uploads/
# named by its SHA-256, and files on the cluster that the manifest does not name (left by an earlier upload) are listed,
# never deleted. The upload carries no env.sh, which the manifest leaves out, so the env.sh of the first clone is copied into
# the folder CausalDiagnosis/cluster_jobs of the upload before its first job. Run on the laptop after `ssh insomnia` has a live
# master connection, into a folder no running job uses:
#   bash CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh <ROOT>/uploads/<tree>
# then, for the commit an upload matches:
#   .venv/bin/python -m CausalDiagnosis.experiments.run_verification.upload_manifest match CausalDiagnosis/cluster_jobs/uploads/<manifest sha256>.txt
# SSH and HOST name the ssh command and the host (a local stand in for a dry run).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh
REMOTE=${1:?usage: bash upload_fork.sh <remote fork folder>}
FORK=${FORK:-$HERE}
while [ ! -f "$FORK/uv.lock" ] && [ "$FORK" != / ]; do FORK=$(dirname "$FORK"); done   # the root of the tree, the folder of uv.lock
SSH=${SSH:-ssh}
HOST=${HOST:-insomnia}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
$SSH -o BatchMode=yes "$HOST" true || { echo "no ssh master for $HOST, log in once with: ssh $HOST" >&2; exit 2; }
cd "$FORK" || exit 2
.venv/bin/python -m CausalDiagnosis.experiments.run_verification.upload_manifest write --out "$STAGE/UPLOAD_MANIFEST.txt" || exit 2
if command -v sha256sum >/dev/null 2>&1; then SUM=$(sha256sum "$STAGE/UPLOAD_MANIFEST.txt" | cut -d' ' -f1)
else SUM=$(shasum -a 256 "$STAGE/UPLOAD_MANIFEST.txt" | cut -d' ' -f1); fi
# the path column of the manifest, after the 64 hex digits and two spaces
cut -c67- "$STAGE/UPLOAD_MANIFEST.txt" > "$STAGE/files.txt"
# COPYFILE_DISABLE and --no-xattrs keep the macOS metadata out of the stream, so the cluster gets no ._ files
COPYFILE_DISABLE=1 tar czf - --no-xattrs -C "$FORK" -T "$STAGE/files.txt" \
  | $SSH "$HOST" "mkdir -p '$REMOTE' && tar xzf - -C '$REMOTE'" || { echo "the upload to $HOST:$REMOTE failed" >&2; exit 3; }
$SSH "$HOST" "cat > '$REMOTE/UPLOAD_MANIFEST.txt'" < "$STAGE/UPLOAD_MANIFEST.txt" || { echo "the manifest upload failed" >&2; exit 3; }
REMOTE_SUM=$($SSH "$HOST" "cd '$REMOTE' && sha256sum -c --quiet UPLOAD_MANIFEST.txt >&2 && sha256sum UPLOAD_MANIFEST.txt" | cut -d' ' -f1)
if [ "$REMOTE_SUM" != "$SUM" ]; then
  echo "UPLOAD_MISMATCH: the copy on $HOST:$REMOTE does not match the manifest $SUM (the files above failed the check)" >&2
  exit 4
fi
mkdir -p "$HERE/uploads" && cp "$STAGE/UPLOAD_MANIFEST.txt" "$HERE/uploads/$SUM.txt"
echo "UPLOAD_OK $(wc -l < "$STAGE/files.txt" | tr -d ' ') files to $HOST:$REMOTE, manifest $SUM, copy in $HERE/uploads/$SUM.txt"
$SSH "$HOST" "cd '$REMOTE' && find . -type f -not -path './.venv/*' -not -path '*/__pycache__/*' -not -path './CausalDiagnosis/results/runs/*' -not -name UPLOAD_MANIFEST.txt" \
  | sed 's|^\./||' | LC_ALL=C sort > "$STAGE/remote.txt"
LC_ALL=C sort "$STAGE/files.txt" | LC_ALL=C comm -13 - "$STAGE/remote.txt" > "$STAGE/stale.txt"
if [ -s "$STAGE/stale.txt" ]; then
  echo "files on $HOST:$REMOTE that this upload does not name, left by an earlier upload and not deleted:"
  cat "$STAGE/stale.txt"
fi
