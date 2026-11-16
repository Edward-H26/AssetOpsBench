#!/bin/bash
#SBATCH -J cd-suite
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
# The suite of a tree as a CPU job: the acceptance of an environment and the suite of record on the cluster. The
# tree, SUITE_TREE or this tree when unset, is copied without its repository and without an environment to node local disk,
# with the data of record that fetch_data.sbatch built into it, and tested there with the environment of its lock, under
# env -i with a fixed list of variables, so that no variable of the
# scheduler or of a login reaches the code under test and the tree stays as it was. A failed test is run again alone into a
# second log. The job writes the JUnit file and a table of class, name and outcome under SUITES/<tree>_<job id>. With
# SUITE_COUCHDB=1 it starts CouchDB, so that the test that compares the memory store with CouchDB runs, and with
# SUITE_DIGESTS=1 the digest tests run. SUITE_COUNT names the number of tests that the same roots collect on the Mac for the
# commit, and the job stops with SUITE_COUNT_MISMATCH and status 4 before its first test when the count here differs.
#   bash $H/submit.sh run_suite.sbatch --export=ALL,SUITE_COUNT=<count on the Mac>
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
ROOTS="src CausalDiagnosis benchmarks"
COUCH_PID=
SUITE_LA=

# the outcome of every test of a JUnit file as a table of class, name and outcome, the counts, and the node ids of the
# failed tests, for the reruns
OUTCOMES='import sys, xml.etree.ElementTree as ET
from pathlib import Path
junit, table, failed, tree = sys.argv[1:5]
rows, ids = [], []
for case in ET.parse(junit).getroot().iter("testcase"):
    kinds = [child.tag for child in case if child.tag in ("skipped", "failure", "error")]
    kind = kinds[0] if kinds else "passed"
    classname, name = case.get("classname") or "", case.get("name") or ""
    rows.append(f"{classname}\t{name}\t{kind}")
    if kind in ("failure", "error"):
        parts = classname.split(".")
        for cut in range(len(parts), 0, -1):
            path = Path(*parts[:cut]).with_suffix(".py")
            if (Path(tree) / path).is_file():
                ids.append("::".join([path.as_posix(), *parts[cut:], name]))
                break
Path(table).write_text("".join(row + "\n" for row in rows), encoding="utf-8")
Path(failed).write_text("".join(i + "\n" for i in ids), encoding="utf-8")
count = [sum(1 for row in rows if row.endswith("\t" + kind)) for kind in ("passed", "skipped", "failure", "error")]
print("SUITE_COUNTS passed %d skipped %d failed %d errors %d" % tuple(count))'

clean_run() {
  # clean_run COMMAND ...: the command in the copy of the tree with the fixed list of variables
  ( cd "$SUITE_LA/tree" && env -i "${SUITE_ENV[@]}" "$@" )
}

run_suite() {
  local tree=${SUITE_TREE:-$FORK} id out lock status collected
  [ -n "${SUITE_COUNT:-}" ] || { echo "name SUITE_COUNT=<n>, the count that python -m pytest --collect-only -q $ROOTS prints on the Mac for this commit" >&2; return 2; }
  id=$(FORK=$tree tree_id) || return 2
  lock=$(sha256_of "$tree/uv.lock")
  [ -x "$ENVS/${lock:0:12}/bin/python" ] || { echo "no environment of the lock of $tree under $ENVS: run setup_env.sbatch" >&2; return 2; }
  out=$SUITES/${id}_${SLURM_JOB_ID:-manual}
  mkdir -p "$out" "$SUITE_LA/tree" "$SUITE_LA/tmp"
  tar -C "$tree" --exclude=./.git --exclude=./.venv -cf - . | tar -C "$SUITE_LA/tree" -xf - || { echo "the copy of $tree failed" >&2; return 2; }
  SUITE_ENV=(HOME="$HOME" PATH="$ENVS/${lock:0:12}/bin:/usr/bin:/bin" PYTHONPATH="$SUITE_LA/tree:$SUITE_LA/tree/src" PYTHONUTF8=1
             PYTHONDONTWRITEBYTECODE=1 TMPDIR="$SUITE_LA/tmp")
  if [ "${SUITE_COUCHDB:-}" = 1 ]; then
    start_couchdb "$SUITE_LA"
    SUITE_ENV+=(COUCHDB_URL="$COUCHDB_URL" COUCHDB_USERNAME="$COUCHDB_USERNAME" COUCHDB_PASSWORD="$COUCHDB_PASSWORD")
  fi
  [ "${SUITE_DIGESTS:-}" != 1 ] || SUITE_ENV+=(ASSET_DIGESTS=1)
  echo "### suite of $tree, tree $id, into $out"
  # shellcheck disable=SC2086
  collected=$(clean_run python -m pytest $ROOTS --collect-only -q -p no:cacheprovider 2> /dev/null | sed -n 's/^\([0-9][0-9]*\) tests\{0,1\} collected.*/\1/p' | tail -n 1)
  echo "SUITE_COLLECTED ${collected:-none}, SUITE_COUNT $SUITE_COUNT"
  [ "$collected" = "$SUITE_COUNT" ] || { echo "SUITE_COUNT_MISMATCH the suite collects ${collected:-no} tests here and $SUITE_COUNT on the Mac" >&2; return 4; }
  # shellcheck disable=SC2086
  clean_run python -m pytest $ROOTS -q -p no:cacheprovider -rs --junitxml="$out/suite.xml" > "$out/suite.log" 2>&1
  status=$?
  tail -n 3 "$out/suite.log"
  [ -s "$out/suite.xml" ] || { echo "no JUnit file, see $out/suite.log" >&2; return 3; }
  "$ENVS/${lock:0:12}/bin/python" -c "$OUTCOMES" "$out/suite.xml" "$out/suite.tsv" "$out/failed.txt" "$SUITE_LA/tree"
  while IFS= read -r test_id; do
    clean_run python -m pytest "$test_id" -q -p no:cacheprovider < /dev/null >> "$out/rerun.log" 2>&1
    echo "RERUN $? $test_id"
  done < "$out/failed.txt"
  echo "CD_SUITE_DONE $out"
  return "$status"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  trap 'if [ -n "$COUCH_PID" ]; then kill "$COUCH_PID" 2>/dev/null; fi; [ -z "$SUITE_LA" ] || rm -rf "$SUITE_LA"' EXIT
  job_header "suite of ${SUITE_TREE:-$FORK}"
  SUITE_LA=$(node_local_dir)
  run_suite
  exit $?
fi
