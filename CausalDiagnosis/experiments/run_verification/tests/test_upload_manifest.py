"""The upload manifest lists what an upload ships and nothing it leaves out, in the format sha256sum checks. The
matcher names the commit of a clean tree exactly and, for a changed tree, the nearest commit with the files that differ,
keeping the files .gitignore covers apart. The ledger of a run on an upload records the manifest's hash and the code
files changed after the upload. Everything happens in a small repository under pytest's tmp_path."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import upload_manifest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _commit(root, message):
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _tree(root, files):
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")


def _repo(tmp_path):
    """A fork with two code files, ignored data, a run folder, byte code and a folder file of macOS, one commit."""
    root = tmp_path / "fork"
    _tree(root, {".gitignore": "data/\nbenchmarks/hydraulic_rig/runs/\n__pycache__/\n.DS_Store\n", "src/a.py": "a = 1\n", "src/b.py": "b = 2\n",
                 "data/facts.jsonl": "{}\n", "CausalDiagnosis/results/runs/r1/run.json": "{}\n", "src/__pycache__/a.cpython-312.pyc": "x",
                 ".DS_Store": "x"})
    _git(root, "init", "-q")
    return root, _commit(root, "one")


def _manifest(root, out):
    out.write_text(upload_manifest.manifest_text(root), encoding="utf-8")
    return out


def test_the_manifest_lists_every_file_an_upload_ships_and_sha256sum_checks_it(tmp_path):
    root, _ = _repo(tmp_path)
    assert upload_manifest.shipped(root) == [".gitignore", "data/facts.jsonl", "src/a.py", "src/b.py"]
    manifest = _manifest(root, tmp_path / "manifest.txt")
    assert manifest.read_text(encoding="utf-8").splitlines()[2] == f"{_sha(root / 'src/a.py')}  src/a.py"
    assert ledger.read_manifest(manifest) == {p: _sha(root / p) for p in upload_manifest.shipped(root)}
    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum not installed")
    checked = subprocess.run(["sha256sum", "-c", "--quiet", str(manifest)], cwd=root, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_clean_tree_matches_its_commit_and_a_changed_tree_names_the_nearest_commit_and_its_differences(tmp_path):
    root, first = _repo(tmp_path)
    exact, _, unversioned = upload_manifest.match(_manifest(root, tmp_path / "clean.txt"), root=root)
    assert exact == [first] and unversioned == ["data/facts.jsonl"]
    (root / "src/a.py").unlink()
    _tree(root, {"src/b.py": "b = 3\n", "src/c.py": "c = 4\n"})
    exact, (nearest, diff), _ = upload_manifest.match(_manifest(root, tmp_path / "changed.txt"), root=root)
    assert exact == [] and nearest == first
    assert diff == {"changed": ["src/b.py"], "only_in_upload": ["src/c.py"], "only_in_commit": ["src/a.py"]}
    second = _commit(root, "two")
    exact, _, _ = upload_manifest.match(_manifest(root, tmp_path / "second.txt"), root=root)
    assert exact == [second]


def test_the_ledger_of_a_run_on_an_upload_records_the_manifest_and_the_files_changed_after_it(tmp_path, monkeypatch):
    upload = tmp_path / "upload"
    _tree(upload, {"src/a.py": "a = 1\n", "src/b.py": "b = 2\n"})
    manifest = _manifest(upload, upload / ledger.UPLOAD_MANIFEST)
    (upload / "src/b.py").write_text("b = 5\n", encoding="utf-8")  # one file copied over the upload afterwards
    monkeypatch.setattr(ledger.run_data, "ROOT", upload)
    code = ledger.code_files({"code": {"sha256": {"src/a.py": "0" * 12, "src/b.py": "0" * 12}}})
    assert code["commit"] is None and code["dirty"] is None
    assert code["upload_manifest_sha256"] == _sha(manifest) and code["differs_from_upload"] == ["src/b.py"]
    assert code["files"] == {"src/a.py": _sha(upload / "src/a.py"), "src/b.py": _sha(upload / "src/b.py")}


def test_an_upload_ships_the_hpc_scripts_and_neither_the_synced_run_folders_nor_the_kept_manifests(tmp_path):
    """CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh copies run folders from the cluster into CausalDiagnosis/cluster_jobs/runs_synced and CausalDiagnosis/cluster_jobs/0_setup/upload_fork.sh keeps a copy of
    every manifest it wrote under CausalDiagnosis/cluster_jobs/uploads, so an upload of the fork ships the scripts beside them and neither folder."""
    root = tmp_path / "fork"
    _tree(root, {"CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh": "#!/bin/bash\n", "CausalDiagnosis/cluster_jobs/runs_synced/run_a/run.json": "{}\n", "CausalDiagnosis/cluster_jobs/uploads/0123.txt": "x\n"})
    assert upload_manifest.shipped(root) == ["CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh"]


def test_an_upload_ships_no_generated_table(tmp_path):
    """The fitted models, the grade table and the label table with their lock files are computed on the cluster before the first
    line of a grid, and each run records their hashes as its clone computed them, so no upload ships them."""
    root = tmp_path / "fork"
    tables = sorted(upload_manifest.GENERATED_TABLES)
    _tree(root, {**{path: "{}\n" for path in tables}, "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/grades.json.lock": "", "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/fit_models.py": "x = 1\n"})
    assert upload_manifest.shipped(root) == ["CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/fit_models.py"]
    assert len(tables) == 3 and all(upload_manifest.skipped(path) for path in tables)


def test_an_upload_ships_the_checksums_of_a_data_folder_and_none_of_its_files(tmp_path):
    """The cluster fetches and builds the data file of every asset in its own clone and checks it against the pins of
    CHECKSUMS.txt, so an upload ships the pins of a data folder and no file built from the recordings."""
    root = tmp_path / "fork"
    _tree(root, {"CausalDiagnosis/datasets/x/data/CHECKSUMS.txt": "pins\n", "CausalDiagnosis/datasets/x/data.py": "x = 1\n",
                 "CausalDiagnosis/datasets/x/data/cycles.jsonl": "{}\n", "CausalDiagnosis/datasets/x/data/labels.json": "{}\n"})
    assert upload_manifest.shipped(root) == ["CausalDiagnosis/datasets/x/data.py", "CausalDiagnosis/datasets/x/data/CHECKSUMS.txt"]


def test_an_upload_ships_no_key_file_settings_file_local_grid_file_or_lock_file(tmp_path):
    """The key file .env, the settings file CausalDiagnosis/cluster_jobs/env.sh, which holds the key of a remote judge and of which the cluster keeps its own,
    the local grid files CausalDiagnosis/cluster_jobs/grids/local_*.txt and the lock files of the generated tables stay on the machine that wrote them, so an upload
    ships none of them, while the template of the settings, the grid files of the fork and uv.lock go up."""
    root = tmp_path / "fork"
    _tree(root, {".env": "API_KEY=x\n", "CausalDiagnosis/cluster_jobs/env.sh": "API_KEY=x\n", "CausalDiagnosis/cluster_jobs/grids/local_lines.txt": "--agent causal\n", "CausalDiagnosis/cluster_jobs/env.example.sh": "#!/bin/bash\n",
                 "CausalDiagnosis/cluster_jobs/grids/grid_agents.txt": "--agent causal\n", "CausalDiagnosis/datasets/hyd_1/data/labels.json.lock": "", "uv.lock": "version = 1\n"})
    assert upload_manifest.shipped(root) == ["CausalDiagnosis/cluster_jobs/env.example.sh", "CausalDiagnosis/cluster_jobs/grids/grid_agents.txt", "uv.lock"]


def test_the_settings_file_names_the_generated_tables_of_the_design():
    """The generated tables are listed once, in design.GENERATED_TABLES, which the code lists leave out and no upload ships; the
    settings file of the cluster, sourced before any Python of the tree is chosen, names the same tables in TABLES."""
    import re

    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.experiments.run_verification import run_identity_file_lists

    text = (CLUSTER_JOBS / "env.example.sh").read_text(encoding="utf-8")
    assert re.search(r'^TABLES=\$\{TABLES:-"([^"]*)"\}', text, re.M).group(1).split() == list(design.GENERATED_TABLES)
    assert upload_manifest.GENERATED_TABLES is design.GENERATED_TABLES
    assert all(run_identity_file_lists.excluded(path) and upload_manifest.skipped(path) for path in design.GENERATED_TABLES)
