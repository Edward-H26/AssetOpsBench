#!/bin/bash
# Makes the clone of one pushed commit under CLONES, in a folder named by the first 12 digits of the commit, and copies env.sh
# of this tree into it. A clone is never updated in place: a new commit is a new clone.
#   bash CausalDiagnosis/cluster_jobs/0_setup/clone_commit.sh COMMIT
# It prints the line H=<clone>/CausalDiagnosis/cluster_jobs, the folder of the scripts of the new clone.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh

clone_commit() {
  local commit=${1:-} target
  case "$commit" in ''|*[!0-9a-f]*) echo "COMMIT must be the full hash of 40 hex digits, not '$commit'" >&2; return 2 ;; esac
  [ "${#commit}" -eq 40 ] || { echo "COMMIT must be the full hash of 40 hex digits, not '$commit'" >&2; return 2; }
  target=$CLONES/${commit:0:12}
  [ ! -e "$target" ] || { echo "$target exists, and a clone is never updated in place" >&2; return 4; }
  mkdir -p "$CLONES"
  git clone --quiet "$FORK_URL" "$target" || { echo "git clone of $FORK_URL into $target failed" >&2; return 3; }
  git -C "$target" checkout --quiet --detach "$commit" || { echo "the commit $commit could not be checked out in $target" >&2; return 3; }
  [ "$(git -C "$target" rev-parse HEAD)" = "$commit" ] || { echo "HEAD of $target is not $commit" >&2; return 3; }
  [ -z "$(git -C "$target" status --porcelain)" ] || { echo "$target differs from the commit $commit" >&2; return 3; }
  cp "$HERE/env.sh" "$target/CausalDiagnosis/cluster_jobs/env.sh" || { echo "could not copy env.sh into $target/CausalDiagnosis/cluster_jobs" >&2; return 3; }
  echo "H=$target/CausalDiagnosis/cluster_jobs"
}

main() {
  [ -f "$HERE/env.sh" ] || { echo "no env.sh in $HERE: copy env.example.sh to env.sh and set ROOT" >&2; exit 2; }
  HPC_DIR=$HERE
  source "$HERE/env.sh"
  [ -n "${CD_FUNCTIONS:-}" ] || { echo "env.sh in $HERE predates functions.sh: copy env.example.sh to env.sh once more and set your values in it" >&2; exit 2; }
  clone_commit "$@"
  exit $?
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
