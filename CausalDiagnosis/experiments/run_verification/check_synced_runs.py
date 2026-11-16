"""The laptop side of a sync of run folders from the cluster (CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh, CausalDiagnosis/docs/experiments.md).

The cluster hashes every file under its runs folder first, in the format of sha256sum. plan lists the files whose hash
the local copy lacks, which are the only ones the sync fetches, and keeps their local hash before the copy. check then
hashes the local copy against the cluster's: a file of a finished run that differs stops the sync with no record
written, a file of a run still in progress that changed during the transfer is reported only, and so is a local file
of a run that the cluster no longer holds. Every finished run with a ledger then gets one event that names the source,
the number of files copied and the files the sync replaced, with their old and new hashes.

usage, on the laptop:
  python -m CausalDiagnosis.experiments.run_verification.check_synced_runs plan --manifest REMOTE.sha256 --local DIR --plan PLAN.json
  python -m CausalDiagnosis.experiments.run_verification.check_synced_runs check --manifest REMOTE.sha256 --local DIR --plan PLAN.json --source HOST:PATH
"""
import argparse
import json
import sys
from pathlib import Path

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger


def remote_hashes(manifest):
    """{path relative to the runs folder: SHA-256} of the cluster's sha256sum lines, whose paths start with ./"""
    return {path.removeprefix("./"): digest for path, digest in ledger.read_manifest(manifest).items()}


def plan(manifest, local):
    """{path: local SHA-256 or None} of every file whose local copy is missing or differs from the cluster's."""
    local = Path(local)
    wanted = {}
    for path, digest in remote_hashes(manifest).items():
        current = ledger.sha256_of(local / path)
        if current != digest:
            wanted[path] = current
    return wanted


def run_of(path, runs):
    """The run folder (relative) that holds a path, or None for a file outside every run folder (a grid log)."""
    parts = path.split("/")
    for depth in range(len(parts) - 1, 0, -1):
        folder = "/".join(parts[:depth])
        if folder in runs:
            return folder
    return None


def finished(run_dir):
    """A run counts as finished once its ledger says so, or, without a ledger, once run.json holds the end time. The rig
    driver writes run.json at the end only, while the AssetOpsBench reproduction (CausalDiagnosis/frameworks/AssetOpsBench/reproduction) writes it before
    the first episode, rewrites it after each one and sets ended last, so a run.json caught while it is rewritten is a run
    in progress."""
    if (run_dir / ledger.LEDGER).exists():
        return ledger.load(run_dir).get("status") == "finished"
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(meta.get("ended"))


def check(manifest, local, planned, source):
    """Compares the local copy with the cluster's hashes and appends the sync events. Returns (problems, warnings, summary):
    a problem is a file of a finished run that differs, which leaves every ledger untouched."""
    local = Path(local)
    remote = remote_hashes(manifest)
    runs = {p.parent.relative_to(local).as_posix() for name in ("run.json", ledger.LEDGER) for p in local.rglob(name)}
    differs = {path for path, digest in remote.items() if ledger.sha256_of(local / path) != digest}
    problems, warnings = [], []
    for path in sorted(differs):
        run = run_of(path, runs)
        if run is not None and finished(local / run):
            problems.append(f"{path}: the copy differs from the cluster's hash in a finished run")
        else:
            warnings.append(f"{path}: changed during the transfer ({'a run in progress' if run else 'outside a run folder'})")
    on_cluster = {run_of(path, runs) for path in remote} - {None}
    for run in sorted(on_cluster):
        for path in sorted(p.relative_to(local).as_posix() for p in (local / run).rglob("*") if p.is_file()):
            # the events of the laptop stand beside the ledger, which stays the cluster's byte for byte
            if path not in remote and Path(path).name != ledger.LOCAL_EVENTS:
                warnings.append(f"{path}: only in the local copy, the cluster's run folder lacks it")
    closed = sorted(run for run in runs if finished(local / run))
    with_ledger = [run for run in closed if (local / run / ledger.LEDGER).exists()]
    summary = {"files": len(remote), "copied": len(planned), "runs": len(runs), "events": 0, "runs_in_progress": len(runs) - len(closed),
               "runs_without_ledger": len(closed) - len(with_ledger)}
    if problems:
        return problems, warnings, summary
    for run in with_ledger:
        run_dir = local / run
        copied = {path[len(run) + 1:]: old for path, old in planned.items() if run_of(path, runs) == run}
        if not copied:  # the sync left this run as it was, so it records nothing
            continue
        # a first copy replaces nothing, so the event lists only the files that existed here with another hash
        replaced = {path: (old, remote[f"{run}/{path}"]) for path, old in copied.items() if old is not None and path != ledger.LEDGER}
        # the event goes to local_events.json beside the ledger, so the ledger stays equal to the cluster's and the next sync fetches
        # it no more, and every later event of the laptop, an archive or a judge pass, lands there too (ledger.append_event)
        if not (run_dir / ledger.LOCAL_EVENTS).exists():
            (run_dir / ledger.LOCAL_EVENTS).write_text("[]\n", encoding="utf-8")
        # the source names a host and a folder of the cluster, which no published file holds (public_record)
        ledger.append_event(run_dir, "sync_runs.sh", f"copied {len(copied)} file{'' if len(copied) == 1 else 's'} from the cluster, every "
                            "file of the run checked against the cluster's hashes", replaced)
        summary["events"] += 1
    return problems, warnings, summary


def main(argv):
    ap = argparse.ArgumentParser(description="the laptop side of a sync of run folders from the cluster")
    ap.add_argument("command", choices=("plan", "check"))
    ap.add_argument("--manifest", type=Path, required=True, help="the cluster's sha256sum lines of its runs folder")
    ap.add_argument("--local", type=Path, required=True, help="the local copy of the runs folder")
    ap.add_argument("--plan", type=Path, required=True, help="the files to fetch with their local hash before the copy")
    ap.add_argument("--source", default="the cluster", help="host and folder the copy comes from, for the sync events")
    a = ap.parse_args(argv)
    if a.command == "plan":
        wanted = plan(a.manifest, a.local)
        a.plan.write_text(json.dumps(wanted, indent=1, sort_keys=True), encoding="utf-8")
        sys.stdout.write("".join(f"{path}\n" for path in sorted(wanted)))
        return 0
    problems, warnings, summary = check(a.manifest, a.local, json.loads(a.plan.read_text(encoding="utf-8")), a.source)
    for line in warnings:
        print(f"warning {line}")
    for line in problems:
        print(f"MISMATCH {line}", file=sys.stderr)
    print(f"{summary['files']} files on the cluster, {summary['copied']} copied, {summary['runs']} run folders, "
          f"{summary['events']} sync events, {summary['runs_in_progress']} runs in progress, "
          f"{summary['runs_without_ledger']} finished runs without a ledger")
    return 4 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
