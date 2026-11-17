from CausalDiagnosis.utils.paths import PACKAGE
import subprocess
import sys
from pathlib import Path
from CausalDiagnosis.experiments.run_verification.public_record import public_value
from CausalDiagnosis.experiments.run_verification.run_identity_file_lists import AGENT_LIST, AGENT_LISTS, CODE_LISTS, CODE_ROOTS, excluded, lists_of, role_of
from . import hashing, serving_endpoints, run_data


UPLOAD_MANIFEST = "UPLOAD_MANIFEST.txt"  # written into the fork's copy on the cluster by CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh, absent in a clone


def git_state(paths, root=run_data.ROOT):
    """(commit, the given paths that differ from it) of the fork, or (None, None) when the fork is not a repository of its
    own, as on the cluster, whose copy is an upload without .git inside a folder that may lie in another repository."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    try:
        if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != Path(root).resolve():
            return None, None
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--", *paths) if paths else ""
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, sorted(line[3:] for line in status.splitlines())


def recorded_code(meta):
    """{path: SHA-256} of the code run.json recorded: the lists of code.files of the rig driver, or the SHA-256 prefixes of
    code.sha256 that a run before the lists recorded, or those under fork of the AssetOpsBench reproduction."""
    code = meta.get("code") or {}
    if code.get("files"):
        return {path: digest for listed in code["files"].values() for path, digest in listed.items()}
    return code.get("sha256") or (meta.get("fork") or {}).get("code_sha256") or {}


def tree_files(root=run_data.ROOT):
    """{list name: {path: full SHA-256}} of every file of a tree under design.CODE_ROOTS that an upload ships and that no rule of
    design.EXCLUDED takes out, each file under every list of design.lists_of. A file of no list stands under the empty name, so
    that a check can name it."""
    from CausalDiagnosis.experiments.run_verification.upload_manifest import skipped  # here, since upload_manifest imports this module

    root = Path(root)
    lists = {}
    for top in CODE_ROOTS:
        base = root / top.rstrip("/")
        paths = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if skipped(relative) or excluded(relative):
                continue
            for name in lists_of(relative) or ("",):
                lists.setdefault(name, {})[relative] = hashing.sha256_of(path)
    return {name: dict(sorted(files.items())) for name, files in sorted(lists.items())}


def identity_lists(agent):
    """The lists whose files enter the identity of a run of an agent, those of role identity for it."""
    names = (*CODE_LISTS, "upstream", *(f"{AGENT_LIST}{name}" for name in AGENT_LISTS))
    return [name for name in names if role_of(name, agent) == "identity"]


def identity_of(files, agent):
    """The identity hash of a run of an agent: the SHA-256 over the sorted (list, path, hash) triples of its lists of role identity."""
    return hashing.canonical_sha256(sorted([name, path, digest] for name in identity_lists(agent) for path, digest in (files.get(name) or {}).items()))


def code_record(agent, root=run_data.ROOT):
    """The value of run.json code: the commit and every path under the roots that differs from it, untracked files included,
    both None without a repository of its own, the full SHA-256 of every file per list, the identity of the agent, and the
    hash of the upload manifest with the listed files that differ from it, for a tree without a commit."""
    files = tree_files(root)
    commit, dirty = git_state([top.rstrip("/") for top in CODE_ROOTS], root)
    manifest = Path(root) / UPLOAD_MANIFEST
    uploaded = hashing.read_manifest(manifest) if manifest.is_file() else None
    flat = {path: digest for name, listed in files.items() if name for path, digest in listed.items()}
    return {"commit": commit, "dirty": dirty, "files": files, "identity": identity_of(files, agent),
            "upload_manifest_sha256": hashing.sha256_of(manifest),
            "differs_from_upload": None if uploaded is None else sorted(p for p, h in flat.items() if uploaded.get(p) != h)}


def git_lines(root, *args):
    """The output lines of a git command in root, None when git fails."""
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


def tree_record_reasons(root=run_data.ROOT):
    """Rules 1, 2 and 4 of the start check of a run of record (CausalDiagnosis/docs/experiments.md): the tree is a repository at its
    own root, nothing under the code roots differs from its commit and no file lies in no list, and the commit stands on a branch
    of origin, so that others can check it out. A tree that CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh copied has no repository and gives development runs."""
    commit, dirty = git_state([top.rstrip("/") for top in CODE_ROOTS], root)
    if commit is None:
        return ["the tree is no git repository at its own root, as an upload of CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh is not"]
    reasons = [f"{len(dirty)} paths differ from the commit {commit[:12]}, first {dirty[0]}"] if dirty else []
    if unlisted := tree_files(root).get(""):
        reasons.append(f"{len(unlisted)} files lie in no list of design.py, first {next(iter(unlisted))}")
    if not git_lines(root, "branch", "-r", "--contains", "HEAD"):
        reasons.append(f"the commit {commit[:12]} stands on no branch of origin, so no one else can check it out")
    return reasons


def untracked_loaded(paths, root=run_data.ROOT):
    """Rule 3: the paths among paths that git does not track at HEAD, as an ignored file that a process imports."""
    tracked = set(git_lines(root, "ls-files", "--", *paths) or []) if paths else set()
    return [path for path in paths if path not in tracked]


def data_reasons(facts_path, asset_id=None):
    """Rule 5 for the data: the data file of the run is the release file of the asset and equals its pin in CHECKSUMS.txt."""
    from CausalDiagnosis.datasets.dataset_loader import recordings as records
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET

    profile = get_profile(asset_id or DEFAULT_ASSET)
    release = records.release_path(profile)
    if facts_path is not None and Path(facts_path).resolve() != release.resolve():
        return [f"the data file {public_value(str(facts_path))} is not the release file {release.name} that CHECKSUMS.txt pins"]
    try:
        records.check_data(profile)
    except records.DataFileError as e:
        return [str(e)]
    return []


def tables_reasons(asset_id=None):
    """Rule 5 for the generated tables: the check of precompute_grid_tables.py finds every table of the asset written by the code of
    this tree. It runs in its own process, which reads the tables and writes none."""
    command = [sys.executable, str(PACKAGE / "experiments" / "cluster_runs" / "precompute_grid_tables.py"), "--check", *(["--asset", asset_id] if asset_id else [])]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode == 0:
        return []
    missing = [line for line in done.stdout.splitlines() if line.startswith("TABLES_MISSING")]
    return missing or [f"the tables check of precompute_tables.py ended with status {done.returncode}"]


def open_reasons():
    """Rule 6: no open value of the registered analysis file, and a serving file of record that is complete."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # here: the analysis list enters a process that asks for it

    reasons = [f"the registered analysis file holds the open value {name}" for name in registered.missing()]
    try:
        serving_endpoints.serving_of_record()
    except serving_endpoints.OpenServingFile as e:
        reasons.append(str(e))
    return reasons


def record_reasons(facts_path=None, asset_id=None, root=run_data.ROOT, tables=True):
    """The reasons of rules 1, 2, 4, 5 and 6, which need no run line and no endpoint; [] for a tree that can play runs of record."""
    return [*tree_record_reasons(root), *data_reasons(facts_path, asset_id), *(tables_reasons(asset_id) if tables else []), *open_reasons()]


def code_files(meta):
    """The commit and the paths that differ from it, or on the cluster the upload manifest's hash and the paths whose file
    no longer equals the manifest's (a file copied over the upload), and the full SHA-256 of every file whose prefix
    run.json keeps, so the ledger hashes exactly the files the run recorded."""
    paths = sorted(recorded_code(meta))
    commit, dirty = git_state(paths, run_data.ROOT)
    files = {path: hashing.sha256_of(run_data.ROOT / path) for path in paths}
    manifest = run_data.ROOT / UPLOAD_MANIFEST
    uploaded = hashing.read_manifest(manifest) if manifest.is_file() else None
    return {"commit": commit, "dirty": dirty, "upload_manifest_sha256": hashing.sha256_of(manifest),
            "differs_from_upload": None if uploaded is None else sorted(p for p, h in files.items() if uploaded.get(p) != h),
            "files": files}
