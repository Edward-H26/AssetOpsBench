"""The manifest of an upload of the fork to the cluster, and the commit whose files it matches (CausalDiagnosis/docs/experiments.md).

The cluster's copy of the fork is an upload of the working tree without .git, so a run there cannot name its commit.
CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh writes UPLOAD_MANIFEST.txt into the upload, one sha256sum line per file it ships, every run
ledger keeps the manifest's hash, and on the laptop match names the commit whose files equal the manifest's, or the
nearest commit with the files that differ. Files that .gitignore covers (the data, the fitted models, the result tables)
belong to no commit, so match lists them apart.

usage, from the fork root:
  python -m CausalDiagnosis.experiments.run_verification.upload_manifest write [--root DIR] [--out FILE]
  python -m CausalDiagnosis.experiments.run_verification.upload_manifest match MANIFEST [--commits N]
"""
import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.experiments.table_definitions.experiment_tables import GENERATED_TABLES
from CausalDiagnosis.experiments.run_verification.run_ledger.run_data import ROOT
from CausalDiagnosis.experiments.run_verification.run_ledger import UPLOAD_MANIFEST, read_manifest, sha256_of

# what an upload leaves out: the environment, the repository, caches, the screenshots of the show-your-work plugin, the
# run folders with their published subsets, the run folders that CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh copies back to a laptop and the copies
# of earlier manifests that CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh keeps, the walkthrough pages, which quote readings and stay on the laptop,
# and the folder files of macOS. An upload ships no generated table: the fitted models, the grade table and the label table, with
# their lock files, are computed on the cluster before the first line of a grid (precompute_grid_tables.py), and run.json holds
# their hashes as the job computed them. No file of the data folder of an asset travels but its CHECKSUMS.txt: the
# cluster fetches and builds its own data file and checks it against those pins.
SKIPPED_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".show-your-work"}
SKIPPED_PATHS = {"CausalDiagnosis/results/runs", "CausalDiagnosis/results/published_runs", "CausalDiagnosis/docs/walkthroughs",
                 "CausalDiagnosis/cluster_jobs/runs_synced", "CausalDiagnosis/cluster_jobs/uploads"}
# nor does an upload ship the files that git ignores and that stay on the machine that wrote them: the key file .env, the
# settings file CausalDiagnosis/cluster_jobs/env.sh, which holds the key of a remote judge and of which the cluster keeps its own, and the local grid
# files CausalDiagnosis/cluster_jobs/grids/local_*.txt, and the calibration reports CausalDiagnosis/results/calibration_*.md
SKIPPED_NAMES = {".DS_Store", UPLOAD_MANIFEST, ".env"}
LOCAL_FILES = ("CausalDiagnosis/cluster_jobs/env.sh", "CausalDiagnosis/cluster_jobs/grids/local_*.txt",
               "CausalDiagnosis/results/calibration_*.md")
DATA_FILES = re.compile(r"^CausalDiagnosis/datasets/(?:[^/]+/)*data/(?!CHECKSUMS\.txt$)")


def skipped(path):
    """Whether an upload leaves out a path relative to the fork root."""
    parts = path.split("/")
    return (any(part in SKIPPED_DIRS for part in parts[:-1]) or parts[-1] in SKIPPED_NAMES
            or any(fnmatch.fnmatchcase(path, rule) for rule in LOCAL_FILES)
            or any(path.startswith(f"{folder}/") for folder in SKIPPED_PATHS)
            or path in GENERATED_TABLES or path.endswith(".json.lock") or bool(DATA_FILES.match(path)))


def shipped(root):
    """The files below root that an upload ships, as paths relative to root in sorted order."""
    root = Path(root)
    paths = []
    for folder, dirs, files in os.walk(root):
        relative = Path(folder).relative_to(root)
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS and (relative / d).as_posix() not in SKIPPED_PATHS]
        paths += [(relative / name).as_posix() for name in files if not skipped((relative / name).as_posix())]
    return sorted(paths)


def manifest_text(root):
    root = Path(root)
    return "".join(f"{sha256_of(root / path)}  {path}\n" for path in shipped(root))


def git(root, *args, stdin=None):
    return subprocess.run(["git", *args], cwd=root, input=stdin, capture_output=True, check=True).stdout


def commit_files(root, commit):
    """{path: blob id} of the files of a commit."""
    files = {}
    for entry in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            _mode, kind, blob = meta.decode().split()
            if kind == "blob":
                files[path.decode()] = blob
    return files


def blob_hashes(root, blobs):
    """{blob id: SHA-256 of its content}, read through one git cat-file --batch call."""
    out, hashes, at = git(root, "cat-file", "--batch", stdin="".join(f"{b}\n" for b in blobs).encode()), {}, 0
    while at < len(out):
        header_end = out.index(b"\n", at)
        blob, _kind, size = out[at:header_end].decode().split()
        start = header_end + 1
        hashes[blob] = hashlib.sha256(out[start:start + int(size)]).hexdigest()
        at = start + int(size) + 1
    return hashes


def ignored(root, paths):
    """The paths that .gitignore covers, which no commit holds."""
    if not paths:
        return set()
    # check-ignore exits with 1 when it ignores none of the paths, which is an answer, not a failure
    done = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=root, capture_output=True,
                          input="".join(f"{p}\n" for p in paths).encode())
    if done.returncode not in (0, 1):
        raise subprocess.CalledProcessError(done.returncode, done.args, done.stdout, done.stderr)
    return set(done.stdout.decode().splitlines())


def differences(manifest, files, hashes, unversioned):
    """The files in which a commit and an upload differ: changed, only in the upload (and versioned), only in the commit."""
    return {"changed": sorted(p for p in files if p in manifest and hashes[files[p]] != manifest[p]),
            "only_in_upload": sorted(p for p in manifest if p not in files and p not in unversioned),
            "only_in_commit": sorted(p for p in files if p not in manifest and not skipped(p))}


def match(manifest_path, root=ROOT, commits=50):
    """The newest commits of the fork against a manifest: (exact matches, the nearest commit and its differences, the
    unversioned files of the upload)."""
    manifest = read_manifest(manifest_path)
    candidates = git(root, "rev-list", "--all", f"--max-count={commits}").decode().split()
    hashes, scored = {}, []
    tracked_somewhere = set()
    for commit in candidates:
        files = commit_files(root, commit)
        tracked_somewhere |= set(files)
        new = {blob for blob in files.values() if blob not in hashes}
        if new:
            hashes.update(blob_hashes(root, sorted(new)))
        scored.append((commit, files))
    # a file no scanned commit holds and .gitignore covers is data or a generated file, never a code difference
    unversioned = ignored(root, sorted(p for p in manifest if p not in tracked_somewhere))
    results = [(commit, differences(manifest, files, hashes, unversioned)) for commit, files in scored]
    exact = [commit for commit, diff in results if not any(diff.values())]
    nearest = min(results, key=lambda item: sum(len(v) for v in item[1].values())) if results else None
    return exact, nearest, sorted(unversioned)


def describe(root, commit):
    return git(root, "log", "-1", "--format=%h %ad %s", "--date=short", commit).decode().strip()


def main(argv):
    ap = argparse.ArgumentParser(description="the manifest of an upload of the fork and the commit it matches")
    sub = ap.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write", help="print or write the manifest of the files an upload of the fork ships")
    write.add_argument("--root", type=Path, default=ROOT, help="the tree to list (default: this fork)")
    write.add_argument("--out", type=Path, default=None, help="the manifest file (default: standard output)")
    find = sub.add_parser("match", help="name the commit whose files equal a manifest's")
    find.add_argument("manifest", type=Path)
    find.add_argument("--commits", type=int, default=50, help="how many of the newest commits to compare")
    a = ap.parse_args(argv)
    if a.command == "write":
        text = manifest_text(a.root)
        if a.out:
            a.out.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    exact, nearest, unversioned = match(a.manifest, commits=a.commits)
    print(f"manifest {a.manifest} sha256 {sha256_of(a.manifest)}, {len(read_manifest(a.manifest))} files, "
          f"{len(unversioned)} of them unversioned (covered by .gitignore)")
    if exact:
        for commit in exact:
            print(f"exact match: {describe(ROOT, commit)}")
        return 0
    if nearest is None:
        print("no commit to compare")
        return 1
    commit, diff = nearest
    print(f"no commit of the newest {a.commits} matches, the nearest is {describe(ROOT, commit)}, which differs in "
          f"{sum(len(v) for v in diff.values())} files")
    for kind, paths in diff.items():
        for path in paths:
            print(f"  {kind.replace('_', ' ')}: {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
