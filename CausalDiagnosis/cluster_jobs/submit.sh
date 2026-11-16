#!/bin/bash
# Submits one job script of this folder or of one of its numbered step folders, named by its file name, with the settings
# of env.sh beside it: the account, the partition that the card of
# the script asks for, the log under LOGS, the nodes to leave out, and HPC_DIR in the environment of the job, so that a tree
# at any path runs its own code and no script names an account, a partition or a node.
#   bash CausalDiagnosis/cluster_jobs/submit.sh SCRIPT [options of sbatch] [-- arguments of the script]
# A variable set before the command reaches the job, for example MODEL=<folder> bash CausalDiagnosis/cluster_jobs/submit.sh serve_model.sbatch.
# A job with a card takes only a node of NODES_OF_RECORD when env.sh names them, unless RUN_KIND names development or smoke.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

has_directive() {
  # has_directive SCRIPT PATTERN: whether a #SBATCH line of the script matches the extended pattern
  grep -Eq "^#SBATCH .*$2" "$1"
}

script_path() {
  # The job script $1: the one file of that name at the top of this folder, in one of its numbered step folders or in the
  # folder cluster_jobs of a framework.
  local found=() candidate
  for candidate in "$HERE/$1" "$HERE"/[0-9]_*/"$1" "$FORK"/CausalDiagnosis/frameworks/*/cluster_jobs/"$1"; do
    [ -f "$candidate" ] && found+=("$candidate")
  done
  case ${#found[@]} in
    1) echo "${found[0]}" ;;
    0) echo "no script $1 in $HERE or its step folders" >&2; return 2 ;;
    *) echo "more than one script $1 in $HERE: ${found[*]}" >&2; return 2 ;;
  esac
}

job_name() {
  # The job name that the log file carries: the caller's -J, else the script's directive, else the name of the script.
  local script=$1 name="" previous=""
  shift
  for option in "$@"; do
    case "$previous" in -J|--job-name) name=$option ;; esac
    case "$option" in -J?*) name=${option#-J} ;; --job-name=*) name=${option#--job-name=} ;; esac
    previous=$option
  done
  [ -n "$name" ] || name=$(sed -n -e 's/^#SBATCH -J \([^ ]*\).*/\1/p' -e 's/^#SBATCH --job-name=\([^ ]*\).*/\1/p' "$script" | head -n 1)
  echo "${name:-$(basename "$script")}"
}

card_nodes() {
  # The nodes of the partitions $1 that hold the card type $2, or any card when $2 is empty, one per line, from sinfo.
  sinfo -h -N -p "$1" -o '%N %G' 2>/dev/null \
    | awk -v card="$2" '{ if (card == "" ? $2 ~ /(^|,)gpu/ : $2 ~ ("(^|,)gpu:" card "([:(,]|$)")) print $1 }' | sort -u
}

exclude_list() {
  # EXCLUDE_NODES, and once env.sh names NODES_OF_RECORD, for a job with a card every node of its partitions with that card
  # outside them, unless RUN_KIND names development or smoke.
  # A list of nodes to take is never passed, since a list of two names asks the scheduler for both.
  local script=$1 partition=$2 exclude=$EXCLUDE_NODES card node
  if has_directive "$script" '--gres=gpu' && [ -n "$NODES_OF_RECORD" ] && { [ -z "${RUN_KIND:-}" ] || [ "$RUN_KIND" = record ]; }; then
    card=$(sed -n 's/^#SBATCH .*--gres=gpu:\([A-Za-z0-9_]*\):[0-9].*/\1/p' "$script" | head -n 1)
    for node in $(card_nodes "$partition" "$card"); do
      case " $NODES_OF_RECORD " in *" $node "*) continue ;; esac
      case ",$exclude," in *",$node,"*) continue ;; esac
      exclude=${exclude:+$exclude,}$node
    done
  fi
  echo "$exclude"
}

main() {
  [ -f "$HERE/env.sh" ] || { echo "no env.sh in $HERE: copy env.example.sh to env.sh and set ROOT" >&2; exit 2; }
  HPC_DIR=$HERE
  source "$HERE/env.sh"
  [ -n "${CD_FUNCTIONS:-}" ] || { echo "env.sh in $HERE predates functions.sh: copy env.example.sh to env.sh once more and set your values in it" >&2; exit 2; }
  export HPC_DIR
  local script=${1:-} options=() arguments=() dashes="" setting partition name exclude out status job path="" dir
  [ -n "$script" ] || { echo "usage: bash CausalDiagnosis/cluster_jobs/submit.sh SCRIPT [options of sbatch] [-- arguments of the script]" >&2; exit 2; }
  shift
  case "$script" in
    */*) dir=$(cd "$(dirname "$script")" 2>/dev/null && pwd)
         case "$dir/" in "$HERE"/|"$HERE"/[0-9]_*/|"$FORK"/CausalDiagnosis/frameworks/*/cluster_jobs/) ;;
           *) echo "$script is no script of $HERE or of the folder cluster_jobs of a framework" >&2; exit 2 ;;
         esac
         script=$(basename "$script"); path=$dir/$script ;;
  esac
  case "$script" in *.sbatch|*.sh) ;; *) echo "$script is no .sbatch or .sh script" >&2; exit 2 ;; esac
  [ -n "$path" ] || path=$(script_path "$script") || exit 2
  [ -f "$path" ] || { echo "no script $script in $HERE" >&2; exit 2; }
  while [ $# -gt 0 ]; do
    if [ -z "$dashes" ] && [ "$1" = -- ]; then dashes=1
    elif [ -n "$dashes" ]; then arguments+=("$1")
    else options+=("$1"); fi
    shift
  done
  if has_directive "$path" '--gres=gpu'; then
    if has_directive "$path" '--requeue'; then setting=PARTITION_GPU_RESUMABLE; else setting=PARTITION_GPU; fi
  else
    setting=PARTITION_CPU
  fi
  partition=${!setting}
  for name in ACCOUNT "$setting"; do
    [ -n "${!name}" ] || { echo "$name is empty: set it in env.sh, the settings of this cluster" >&2; exit 2; }
  done
  exclude=$(exclude_list "$path" "$partition") || exit 2
  local line=(sbatch --parsable -A "$ACCOUNT" -p "$partition" -o "$LOGS/%x-%j.out")
  [ -z "$exclude" ] || line+=("--exclude=$exclude")
  out=$("${line[@]}" ${options[@]+"${options[@]}"} "$path" ${arguments[@]+"${arguments[@]}"})
  status=$?
  [ "$status" = 0 ] || exit "$status"
  job=${out%%;*}
  echo "SUBMITTED $script $job $LOGS/$(job_name "$path" ${options[@]+"${options[@]}"})-$job.out"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
