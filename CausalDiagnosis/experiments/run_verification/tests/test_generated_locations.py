"""Every generated location that CausalDiagnosis/utils/paths.py names stays out of the repository and out of the
code that a run hashes: git ignores it, the file lists of a run leave it out, and an upload to the cluster does not ship it.
The data folder of an asset is the one exception to the lists: an episode reads its files, so the list episode holds them
as it holds them in the base, and only its CHECKSUMS.txt travels with an upload. Every name that no ledger records stays out
of a run folder committed to the repository, at any depth, while the files of its published set stay in."""
import shutil
import subprocess

import pytest

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import upload_manifest
from CausalDiagnosis.experiments.run_verification import run_identity_file_lists
from CausalDiagnosis.utils.paths import GENERATED, ROOT

IGNORE_FILES = ("CausalDiagnosis/.gitignore", "CausalDiagnosis/cluster_jobs/.gitignore")
COMMITTED_RUNS = "CausalDiagnosis/results/cluster_runs/"  # where the run folders synced from the cluster are committed
ASSET_DATA = "CausalDiagnosis/datasets/**/data/*"  # the data an episode reads, which the list episode holds


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
    (public_record.PRIVATE_FILE) and of the launchers of an episode (episode_launchers.LAUNCH_PRIVATE) among them: git ignores
    each in a committed run folder, in an episode folder below it and in a folder of interrupted episodes, and keeps the ledger."""
    def is_ignored(path):
        return subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=fork_ignore_root).returncode == 0

    for place in ("a_run/", "a_run/seed000000/", f"a_run/{ledger.INTERRUPTED}/seed000001/"):
        for name in sorted(ledger.IGNORED_NAMES):
            assert is_ignored(COMMITTED_RUNS + place + name), f"{COMMITTED_RUNS}{place}{name} is not ignored by git"
        assert not is_ignored(COMMITTED_RUNS + place + ledger.LEDGER), f"{COMMITTED_RUNS}{place}{ledger.LEDGER} is ignored by git"
