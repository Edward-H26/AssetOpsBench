#!/usr/bin/env bash
# Run one script of this folder in the runtime that requirements-smoke.txt pins, without syncing the project, from the
# fork root:  bash benchmarks/hydraulic_rig/run.sh run_episodes.py --scenario S1 --agent scripted --per-record 1
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
script="$1"
shift
exec uv run --no-project --python 3.12 --with-requirements "$here/requirements-smoke.txt" python "$here/$script" "$@"
