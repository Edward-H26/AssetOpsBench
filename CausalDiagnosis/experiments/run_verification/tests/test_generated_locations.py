"""Every generated location that CausalDiagnosis/utils/paths.py names stays out of the repository and out of the
code that a run hashes: git ignores it, the file lists of a run leave it out, and an upload to the cluster does not ship it.
The data folder of an asset is the one exception to the lists: an episode reads its files, so the list episode holds them
as it holds them in the base, and only its CHECKSUMS.txt travels with an upload. Every name that no ledger records, a lock
file among them, stays out of a run folder committed to the repository, at any depth, while every file that its ledger records
stays in, so the tracked copy of a real run folder of either tool surface passes its ledger check. Only run folders are
committed under results/cluster_runs/, never a grid folder with its job logs, whose server log the rule of the root that
leaves out every .log file would drop. The rule that leaves out the lock files catches no file that the tree tracks."""
import shutil
import subprocess

import pytest

from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.episode_runners.episode_launchers import LAUNCH_PRIVATE
from CausalDiagnosis.experiments.episode_runners.tests.test_agent_files import needs_a_run
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import upload_manifest
from CausalDiagnosis.experiments.run_verification import run_identity_file_lists
from CausalDiagnosis.experiments.run_verification import verify_results_folder
from CausalDiagnosis.experiments.run_verification.public_record import PRIVATE_FILE
from CausalDiagnosis.utils.atomic_writes_and_locks import LOCK_SUFFIX, READ_ONLY_TABLES, locked
from CausalDiagnosis.utils.paths import GENERATED, ROOT

IGNORE_FILES = ("CausalDiagnosis/.gitignore", "CausalDiagnosis/cluster_jobs/.gitignore")
COMMITTED_RUNS = "CausalDiagnosis/results/cluster_runs/"  # where the run folders synced from the cluster are committed
ASSET_DATA = "CausalDiagnosis/datasets/**/data/*"  # the data an episode reads, which the list episode holds
# files that the servers of an episode write under a lock, and a made-up name, since the rule goes by the suffix of the lock alone
LOCKED = ("episode.json", "predictions.json", "transcript.jsonl", "generated_tokens.jsonl", "made_up.txt")


def sample(location):
    """A file that a program would write at a generated location: a folder gets a file inside it, a pattern its first match."""
    path = location.replace("**", "hyd_1").replace("calibration_*", "calibration_2026-01-01").replace("/*", "/cycles.jsonl")
    return path + "a_run/run.json" if path.endswith("/") else path


@pytest.fixture(scope="module")
def ignore_root(tmp_path_factory):
    """A git repository that holds only the ignore files of the fork, so that git check-ignore reads them alone."""
    root = tmp_path_factory.mktemp("ignore")
    for name in IGNORE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


@pytest.mark.parametrize("location", GENERATED)
def test_a_generated_location_is_ignored_unlisted_and_not_uploaded(location, ignore_root):
    path = sample(location)
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ignore_root)
    assert ignored.returncode == 0, f"{path} is not ignored by git"
    if location == ASSET_DATA:
        assert run_identity_file_lists.lists_of(path) == ("episode",), f"{path} is data an episode reads, outside the list episode"
    else:
        assert run_identity_file_lists.lists_of(path) == (), f"{path} lies in a file list of a run"
        assert not run_identity_file_lists.in_roots(path) or run_identity_file_lists.excluded(path), path
    assert upload_manifest.skipped(path), f"an upload ships {path}"


@pytest.fixture(scope="module")
def fork_ignore_root(tmp_path_factory):
    """A git repository that holds the ignore file of the root beside those of the fork, every rule that git applies in the fork."""
    root = tmp_path_factory.mktemp("fork_ignore")
    for name in (".gitignore", *IGNORE_FILES):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def test_every_name_that_no_ledger_records_is_ignored_in_a_committed_run_folder_and_its_ledger_is_not(fork_ignore_root):
    """ledger.IGNORED_NAMES holds the names that lie outside the published set of a run folder, the private file of the run
    (public_record.PRIVATE_FILE) and of the launchers of an episode (episode_launchers.LAUNCH_PRIVATE) among them, and no ledger
    records a lock file: git ignores each in a committed run folder, in an episode folder below it and in a folder of interrupted
    episodes, and keeps the ledger."""
    def is_ignored(path):
        return subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=fork_ignore_root).returncode == 0

    for place in ("a_run/", "a_run/seed000000/", f"a_run/{ledger.INTERRUPTED}/seed000001/"):
        for name in [*sorted(ledger.IGNORED_NAMES), *(name + LOCK_SUFFIX for name in LOCKED)]:
            assert is_ignored(COMMITTED_RUNS + place + name), f"{COMMITTED_RUNS}{place}{name} is not ignored by git"
        assert not is_ignored(COMMITTED_RUNS + place + ledger.LEDGER), f"{COMMITTED_RUNS}{place}{ledger.LEDGER} is ignored by git"


def test_the_rule_of_the_lock_files_catches_no_file_that_the_tree_tracks(fork_ignore_root):
    """The rule that leaves out the lock files stands in the ignore file of the folder CausalDiagnosis and catches no file that the
    tree tracks, so the lock file of the package manager at the root of the repository stays tracked."""
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode != 0:
        pytest.skip("the tree is no git repository, as an upload to the cluster is not")
    tracked = listed.stdout.splitlines()
    assert "uv.lock" in tracked
    found = subprocess.run(["git", "check-ignore", "--no-index", "-v", "--stdin"], cwd=fork_ignore_root, input="\n".join(tracked) + "\n",
                           capture_output=True, text=True)
    caught = [line.split("\t", 1) for line in found.stdout.splitlines()]
    rules = {meta.split(":", 2)[2] for meta, _path in caught}
    by_lock_rule = sorted(path for meta, path in caught if meta.split(":", 2)[2] == "*" + LOCK_SUFFIX)
    assert by_lock_rule == [] and "uv.lock" not in {path for _meta, path in caught}, (by_lock_rule[:5], sorted(rules))


def test_the_ledger_records_no_lock_file_that_the_locking_function_leaves(tmp_path):
    """Every lock file that atomic_writes_and_locks.locked leaves beside a file it locks, whatever the name of that file, stays out
    of the files a ledger records and of the files its check reads, while the locked files stay in, and the check leaves a lock
    file out of a ledger that recorded one."""
    run_dir = tmp_path / "a_run"
    for name in LOCKED:
        with locked(run_dir / "seed000000" / name):
            (run_dir / "seed000000" / name).write_text("{}\n", encoding="utf-8")
    locks = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*" + LOCK_SUFFIX))
    assert locks == sorted(f"seed000000/{name}{LOCK_SUFFIX}" for name in LOCKED) and all(map(ledger.ignored, locks))
    files = {f"seed000000/{name}" for name in LOCKED}
    assert set(ledger.file_hashes(run_dir)) == files and set(verify_results_folder.current_files(run_dir)) == files
    recorded = {name: {"sha256": ledger.sha256_of(run_dir / "seed000000" / name)} for name in (*LOCKED, *(name + LOCK_SUFFIX for name in LOCKED))}
    record = {"episodes": {"seed000000": {"files": recorded}}, "run_files": {}, "events": []}
    assert set(verify_results_folder.expected_files(record)) == files


@needs_a_run
def test_the_tracked_copy_of_a_committed_run_folder_of_either_surface_holds_its_whole_ledger_set(tmp_path, monkeypatch):
    """Only run folders are committed under results/cluster_runs/, each whole as its sync left it, never a grid folder with its
    job logs. A scripted run on the action server and one on AssetOpsBench's servers, committed there in a repository with the
    ignore files of the fork, track exactly the files that their ledger records and the ledger itself, while the private files of
    the run and of its launchers and the lock files that the run left stay untracked, and the tracked copy, checked out alone,
    passes verify_results_folder.check_ledger."""
    monkeypatch.setenv(READ_ONLY_TABLES, "0")  # the driver sets it for its process, so the teardown removes it again
    repo = tmp_path / "fork"
    for name in (".gitignore", *IGNORE_FILES):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, repo / name)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for surface, options in (("rig", ["--config", "static"]), ("aob", ["--config", "base", "--servers", "aob", "--store", "memory"])):
        run_episodes.main([*options, "--agent", "scripted", "--seeds", "1", "--out", str(tmp_path / surface)])
        (run_dir,) = [p for p in (tmp_path / surface).iterdir() if p.is_dir()]
        shutil.copytree(run_dir, repo / COMMITTED_RUNS / run_dir.name)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    tracked = subprocess.run(["git", "ls-files", "--", COMMITTED_RUNS], cwd=repo, capture_output=True, text=True, check=True).stdout.splitlines()
    copy = tmp_path / "tracked"
    subprocess.run(["git", "checkout-index", "-a", f"--prefix={copy}/"], cwd=repo, check=True)
    for run_dir in sorted((repo / COMMITTED_RUNS).iterdir()):
        expected = {*verify_results_folder.expected_files(ledger.load(run_dir)), ledger.LEDGER}
        kept = {path[len(COMMITTED_RUNS):].split("/", 1)[1] for path in tracked if path[len(COMMITTED_RUNS):].split("/", 1)[0] == run_dir.name}
        assert kept == expected, (run_dir.name, sorted(kept ^ expected)[:5])
        assert verify_results_folder.check_ledger(copy / COMMITTED_RUNS / run_dir.name) == ("checked", []), run_dir.name
        left_out = {p.name for p in run_dir.rglob("*") if p.is_file() and ledger.ignored(p.relative_to(run_dir))}
        assert {PRIVATE_FILE, LAUNCH_PRIVATE} <= left_out and any(name.endswith(LOCK_SUFFIX) for name in left_out), (run_dir.name, sorted(left_out))
