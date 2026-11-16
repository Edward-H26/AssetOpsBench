"""Every repository path that the two READMEs of the hydraulic rig, the page of the cluster folder and the pages under
CausalDiagnosis/docs/ (the fork's sections of the server page among them) name exists, so a reader who follows them finds the file."""
import re
from pathlib import Path

from CausalDiagnosis.utils.paths import ROOT
READMES = ("CausalDiagnosis/experiments/README.md", "CausalDiagnosis/environment/README.md", "CausalDiagnosis/cluster_jobs/README.md")
PATH = re.compile(r"(?<![\w./-])(?:CausalDiagnosis|src|benchmarks|docs|hpc)/[A-Za-z0-9_.\-/]*[A-Za-z0-9_/]")
DOC_NAME = re.compile(r"`((?:hydraulic-rig|causal-diagnosis)-[a-z\-]+\.md)`")  # documents named without docs/
# generated, so a fresh clone may lack them: the run folders of the drivers with their published subsets and the
# walkthrough pages made from them, all git-ignored since they hold sensor readings of single cycles, the result
# tables that tables_from_runs/ and leaderboard.py write, the data folder that download_and_build.py fills (all but CHECKSUMS.txt,
# which the repository tracks), the models.json that fit_models.py writes and the grades.json
# that diagnosability_grade.py writes beside it, and the two folders of the cluster folder that sync_runs.sh and upload_fork.sh fill on a laptop
GENERATED = ("CausalDiagnosis/results/runs/", "CausalDiagnosis/results/", "CausalDiagnosis/docs/walkthroughs/",
             "CausalDiagnosis/datasets/hyd_1/data/", "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json", "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json",
             "CausalDiagnosis/cluster_jobs/runs_synced/", "CausalDiagnosis/cluster_jobs/uploads/")


def generated(path):
    """Whether a path lies under GENERATED, a folder named with or without its closing slash."""
    return f"{path.rstrip('/')}/".startswith(GENERATED)


def named_paths(text):
    paths = {m.group(0) for m in PATH.finditer(text)} | {f"docs/{m.group(1)}" for m in DOC_NAME.finditer(text)}
    return sorted(p for p in paths if not generated(p))


def documents():
    """(name, text) of every document checked; the pages under CausalDiagnosis/docs/ are found when the test runs."""
    docs = [(name, (ROOT / name).read_text(encoding="utf-8")) for name in READMES]
    pages = sorted(p for p in (ROOT / "CausalDiagnosis" / "docs").rglob("*.md") if not generated(str(p.relative_to(ROOT))))
    return docs + [(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8")) for p in pages]


def test_named_paths_exist():
    missing = [(name, path) for name, text in documents() for path in named_paths(text) if not (ROOT / path).exists()]
    assert missing == []
