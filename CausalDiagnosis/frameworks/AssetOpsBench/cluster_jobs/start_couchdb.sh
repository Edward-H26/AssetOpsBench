# Functions of the jobs that run CouchDB, sourced after env.sh: start_couchdb starts CouchDB 3.5 under Apptainer on node local
# disk at a free port, with its Erlang distribution at another free port on the loopback address alone, and under an Erlang
# node name of its own, so that any number of jobs on one node each start one, exactly as run_reproduction.sh needs it, and
# exports COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD and COUCHDB_IMAGE_SHA256, which the platform record of a run holds.
# The image is the shared one when it is readable, else a copy of one's own under IMAGES, pulled once by the image job of the
# setup, which calls couchdb_image through IMAGE_PULLS, and the job unpacks it once. The caller kills $COUCH_PID in its exit
# trap. No database is loaded or reset here: each episode's producer creates its databases itself.

couchdb_image() {
  # The CouchDB image of the job, the shared one when it is readable, else a copy of one's own under IMAGES pulled once by
  # pull_image_once of functions.sh, and its SHA-256 in COUCHDB_IMAGE_SHA256. $1 is the job's node local folder.
  module load apptainer/1.4.5
  if [ -z "${COUCH_IMAGE:-}" ]; then
    COUCH_IMAGE=$IMAGES/couchdb-3.5.sif
    if [ -n "$SHARED" ] && [ -r "$SHARED/images/couchdb-3.5.sif" ]; then COUCH_IMAGE=$SHARED/images/couchdb-3.5.sif; fi
  fi
  export APPTAINER_TMPDIR=$1/apptainer APPTAINER_CACHEDIR=$1/apptainer-cache
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$COUCH_IMAGE")"
  pull_image_once "$COUCH_IMAGE" docker://couchdb:3.5 "$1" || { echo "could not pull couchdb:3.5 into $COUCH_IMAGE" >&2; exit 3; }
  COUCHDB_IMAGE_SHA256=$(sha256_of "$COUCH_IMAGE")
  export COUCHDB_IMAGE_SHA256
  echo "couchdb_image $COUCH_IMAGE"
  echo "couchdb_image_sha256 $COUCHDB_IMAGE_SHA256"
}

start_couchdb() {
  # $1 is the job's node local folder
  local la=$1 couch port dist_port pw cookie old_umask sandbox
  couchdb_image "$la"
  sandbox=$la/couchdb_sandbox
  apptainer build --sandbox "$sandbox" "$COUCH_IMAGE" > /dev/null || { echo "the sandbox of $COUCH_IMAGE could not be built in $la" >&2; exit 3; }
  couch=$la/couchdb
  mkdir -p "$couch/data"
  # Two free ports of the loopback address, held at once so that they differ: HTTP, and the Erlang distribution, which the
  # vm.args of the image fixes at one port that a second CouchDB on the node cannot take
  read -r port dist_port <<< "$(python -c 'import socket; s, t = socket.socket(), socket.socket(); s.bind(("127.0.0.1", 0)); t.bind(("127.0.0.1", 0)); print(s.getsockname()[1], t.getsockname()[1])')"
  # A password and an Erlang cookie of this job alone, since any user of a shared node reaches the port and could change the
  # data under a run. Both files stay private to the job's user, and curl reads the password on stdin to keep it out of ps.
  pw=$(python -c 'import secrets; print(secrets.token_hex(16))')
  cookie=$(python -c 'import secrets; print(secrets.token_hex(16))')
  old_umask=$(umask); umask 077
  cat > "$couch/aob.ini" <<INI
[couchdb]
single_node = true
database_dir = $couch/data
view_index_dir = $couch/data

[chttpd]
port = $port
bind_address = 127.0.0.1

[admins]
admin = $pw

[log]
writer = stderr
level = warning
INI
  apptainer exec "$sandbox" cat /opt/couchdb/etc/vm.args \
    | sed -e "s/^-s\{0,1\}name .*/-name aob${SLURM_JOB_ID:-$$}_$$@127.0.0.1/" -e "s/^-setcookie .*/-setcookie $cookie/" \
          -e "/^-kernel inet_dist_listen_min /d" -e "/^-kernel inet_dist_listen_max /d" -e "/^-kernel inet_dist_use_interface /d" > "$couch/vm.args"
  grep -q '^-name ' "$couch/vm.args" || echo "-name aob${SLURM_JOB_ID:-$$}_$$@127.0.0.1" >> "$couch/vm.args"
  grep -q '^-setcookie ' "$couch/vm.args" || echo "-setcookie $cookie" >> "$couch/vm.args"
  # the distribution listens at the port of this start and on the loopback address alone, since no other host reaches it
  printf '%s\n' "-kernel inet_dist_listen_min $dist_port" "-kernel inet_dist_listen_max $dist_port" \
    "-kernel inet_dist_use_interface {127,0,0,1}" >> "$couch/vm.args"
  umask "$old_umask"
  # COUCHDB_ARGS_FILE and COUCHDB_INI_FILES are read by the image's bin/couchdb, the APPTAINERENV_ prefix sets them inside only
  export APPTAINERENV_COUCHDB_ARGS_FILE=$couch/vm.args
  export APPTAINERENV_COUCHDB_INI_FILES="/opt/couchdb/etc/default.ini $couch/aob.ini"
  apptainer exec --bind "$la" "$sandbox" /opt/couchdb/bin/couchdb > "$couch/couchdb.log" 2>&1 &
  COUCH_PID=$!
  # the server is no job of the caller's shell, so that a wait of the caller for its own runs never waits for CouchDB
  disown "$COUCH_PID"
  for _ in $(seq 1 90); do
    curl -sf -K - "http://127.0.0.1:$port/" <<< "user = admin:$pw" >/dev/null && break
    kill -0 "$COUCH_PID" 2>/dev/null || break
    sleep 5
  done
  curl -sf -K - "http://127.0.0.1:$port/" <<< "user = admin:$pw" || { echo "CouchDB did not answer, its log ends with" >&2; tail -40 "$couch/couchdb.log" >&2; exit 3; }
  echo
  export COUCHDB_URL=http://127.0.0.1:$port COUCHDB_USERNAME=admin COUCHDB_PASSWORD=$pw
}
