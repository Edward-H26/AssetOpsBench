#!/bin/bash
# The label of record of the digests and the check of the nodes of record. The label of record is the label current of the
# tracked digest file, written in the clone of this folder by one CPU job on a node with the card of record, which the job
# checks first (CPU_CARD_OF_RECORD=1 of cpu_job.sbatch), so that the label comes from the card of every registered run. The
# check computes the digests again on every node of the partition with that card, one CPU job pinned to each node, and
# compares them with the label of record, so that a node that computes another digest takes no registered run. The job of
# each node first prints the name and the memory of its card, and only a node on the card of record (GPU_OF_RECORD and
# GPU_MIB_OF_RECORD of env.sh) joins the nodes of record.
#   bash CausalDiagnosis/cluster_jobs/1_checks/check_nodes.sh label    the label of record, one CPU job on a node with the card
#   bash CausalDiagnosis/cluster_jobs/1_checks/check_nodes.sh submit   one job per node that is up, recorded in DIGESTS/node_checks.tsv
#   bash CausalDiagnosis/cluster_jobs/1_checks/check_nodes.sh read     one line per node, then the line NODES_OF_RECORD="..." for env.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$HERE/submit.sh" ] && [ "$HERE" != / ]; do HERE=$(dirname "$HERE"); done   # the folder of submit.sh and env.sh
LABELS=CausalDiagnosis/experiments/run_verification/invariant_digests.json   # the tracked digest file, relative to the tree
LABEL=current                                        # the label of record in it, which the golden tests compare with

card_type() {
  sed -n 's/^#SBATCH .*--gres=gpu:\([A-Za-z0-9_]*\):[0-9].*/\1/p' "$HERE/2_serve_model/serve_model.sbatch" | head -n 1
}

card_nodes() {
  # "<node> <state>" of every node of PARTITION_GPU with the card of the serving jobs, from sinfo
  sinfo -h -N -p "$PARTITION_GPU" -o '%N %G %T' 2>/dev/null \
    | awk -v card="$(card_type)" '$2 ~ ("(^|,)gpu:" card "([:(,]|$)") { print $1, $3 }' | sort -u
}

write_label() {
  bash "$HERE/submit.sh" cpu_job.sbatch -p "$PARTITION_GPU" --gres=gpu:"$(card_type)":1 -J cd-label --export=ALL,CPU_CARD_OF_RECORD=1 -- bash -c \
    "python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts \"\$DATA_FILE\" --label $LABEL --write $LABELS --commit \"\$(git rev-parse HEAD)\""
}

submit_checks() {
  local node state out job
  [ -s "$FORK/$LABELS" ] || { echo "no digest file at $FORK/$LABELS: run check_nodes.sh label first" >&2; return 2; }
  mkdir -p "$DIGESTS"
  : > "$DIGESTS/node_checks.tsv"
  while read -r node state; do
    case "$state" in
      down*|drain*|fail*|maint*|unk*|no_respond*|power*) echo "NODE_SKIPPED $node $state"; continue ;;
    esac
    out=$(bash "$HERE/submit.sh" cpu_job.sbatch -p "$PARTITION_GPU" --gres=gpu:"$(card_type)":1 -w "$node" -J cd-node -- bash -c \
      "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1 | sed 's/^/GPU_OF_NODE /'; python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts \"\$DATA_FILE\" --label $LABEL --check $LABELS") || return 3
    echo "$out"
    job=$(sed -n 's/^SUBMITTED [^ ]* \([0-9][0-9]*\) .*/\1/p' <<< "$out")
    printf '%s\t%s\n' "$node" "$job" >> "$DIGESTS/node_checks.tsv"
  done < <(card_nodes)
}

read_checks() {
  local node job log status summary passed="" checked=" " state card name mib
  [ -s "$DIGESTS/node_checks.tsv" ] || { echo "no checks submitted: run check_nodes.sh submit" >&2; return 2; }
  while IFS=$'\t' read -r node job; do
    log=$LOGS/cd-node-$job.out
    status=$(sed -n 's/^CD_CPU_JOB_DONE \([0-9][0-9]*\).*/\1/p' "$log" 2>/dev/null | tail -n 1)
    summary=$(grep -E '^[0-9]+ same, [0-9]+ changed' "$log" 2>/dev/null | tail -n 1)
    [ -n "$status" ] || continue
    card=$(sed -n 's/^GPU_OF_NODE //p' "$log" 2>/dev/null | tail -n 1)
    name=${card%, *}; mib=${card##*, }
    echo "NODE $node $status $summary"
    echo "NODE_GPU $node ${name:-unknown} ${mib:-unknown}"
    checked="$checked$node "
    if [ "$status" = 0 ] && [ -n "$summary" ]; then
      if card_of_record "$name" "$mib"; then passed=${passed:+$passed }$node; else echo "NODE_OTHER_GPU $node ${name:-unknown} ${mib:-unknown}"; fi
    fi
  done < "$DIGESTS/node_checks.tsv"
  if [ -z "$GPU_OF_RECORD" ] || [ -z "$GPU_MIB_OF_RECORD" ]; then
    echo "GPU_OF_RECORD and GPU_MIB_OF_RECORD must both be set in env.sh before the nodes of record: copy the card of a NODE_GPU line" >&2
    return 2
  fi
  echo "NODES_OF_RECORD=\"$passed\""
  while read -r node state; do
    case "$checked" in *" $node "*) ;; *) echo "NODE_UNCHECKED $node" ;; esac
  done < <(card_nodes)
}

main() {
  [ -f "$HERE/env.sh" ] || { echo "no env.sh in $HERE: copy env.example.sh to env.sh and set ROOT" >&2; exit 2; }
  HPC_DIR=$HERE
  source "$HERE/env.sh"
  [ -n "${CD_FUNCTIONS:-}" ] || { echo "env.sh in $HERE predates functions.sh: copy env.example.sh to env.sh once more and set your values in it" >&2; exit 2; }
  case "${1:-}" in
    label) write_label ;;
    submit) submit_checks ;;
    read) read_checks ;;
    *) echo "usage: bash CausalDiagnosis/cluster_jobs/1_checks/check_nodes.sh label, then submit when its job has ended, then read" >&2; exit 2 ;;
  esac
  exit $?
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
