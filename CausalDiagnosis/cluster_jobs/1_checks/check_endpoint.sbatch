#!/bin/bash
#SBATCH -J cd-endpoint
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 02:00:00
# The check of a served model from another node: while the serving job SERVE_JOB, submitted with SERVE_REACH=nodes, runs on
# its node, this job asks that job's endpoint for the model list from its own node, once with the key of that job and once
# without, and then writes the stop file of that job, so that the serving job ends with the check. With the key the list must
# answer, and without it the server must refuse with 401 or 403. The job fails when either does not hold, and when the
# endpoint file never appeared, since then nothing was checked.
#   SERVE_JOB=<job id> bash $H/submit.sh check_endpoint.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"

check_endpoint() {
  local serve=$1 file url server code status=0 tries
  file=$ENDPOINTS/$serve.txt
  for tries in $(seq 1 "${ENDPOINT_TRIES:-660}"); do
    [ -s "$file" ] && break
    [ -n "$(squeue -h -j "$serve" -o %T 2>/dev/null)" ] || break
    sleep 10
  done
  if [ ! -s "$file" ]; then
    echo "HTTP_BETWEEN_NODES not run: the endpoint file $file did not appear"
    return 1
  fi
  url=$(head -n 1 "$file")
  server=$(sed -n 's/^job [0-9]* node \([^ ]*\) .*/\1/p' "$file" | head -n 1)
  API_KEY=$(cat "$(key_file "$serve")" 2>/dev/null)
  code=$(key_curl -m 20 -o /dev/null -w '%{http_code}' "$url/models")
  echo "from $(hostname) to $url on node ${server:-unknown} with the key: HTTP $code"
  if [ "$code" != 200 ]; then echo "HTTP_BETWEEN_NODES failed with code $code"; status=1
  elif [ "$(hostname -s)" = "${server%%.*}" ]; then echo "HTTP_BETWEEN_NODES same node: this job landed on the serving node"
  else echo "HTTP_BETWEEN_NODES ok"; fi
  code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$url/models")
  echo "without the key: HTTP $code"
  case "$code" in
    401|403) echo "ENDPOINT_CLOSED ok" ;;
    200) echo "ENDPOINT_OPEN the server answers without the key"; status=1 ;;
    *) echo "ENDPOINT_CLOSED unknown: HTTP $code without the key"; status=1 ;;
  esac
  return "$status"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  [ -n "${SERVE_JOB:-}" ] || { echo "SERVE_JOB is empty: name the serving job with SERVE_REACH=nodes" >&2; exit 2; }
  job_header "check of the endpoint of job $SERVE_JOB"
  check_endpoint "$SERVE_JOB"
  STATUS=$?
  touch "$ENDPOINTS/$SERVE_JOB.stop"
  echo "CD_ENDPOINT_DONE"
  exit "$STATUS"
fi
