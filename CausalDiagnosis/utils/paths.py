"""The locations of the repository, the one place that knows where things are. Every module of CausalDiagnosis reaches
the configs, the datasets, the results, the docs and the cluster jobs through these names, so that a move of a folder
changes this file alone. ROOT is the repository root, which holds CausalDiagnosis/ and the upstream src/."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
SRC = ROOT / "src"
CONFIGS = PACKAGE / "configs"
DATASETS = PACKAGE / "datasets"
RESULTS = PACKAGE / "results"
DOCS = PACKAGE / "docs"
CLUSTER_JOBS = PACKAGE / "cluster_jobs"
GRIDS = CLUSTER_JOBS / "grids"  # the grid files of the cluster jobs
# The generated locations: what the programs write and git ignores. test_generated_locations.py checks each
# against CausalDiagnosis/.gitignore, the identity lists and the upload manifest.
RUNS = RESULTS / "runs"  # the run folders of every driver
PUBLISHED_RUNS = RESULTS / "published_runs"  # the published subsets of run folders
WALKTHROUGHS = DOCS / "walkthroughs"  # pages that quote sensor readings of single cycles
SYNCED_RUNS = CLUSTER_JOBS / "runs_synced"  # run folders copied back from the cluster
UPLOADS = CLUSTER_JOBS / "uploads"  # the packed copies of the fork sent to the cluster
FITTED_TABLES = PACKAGE / "mcp_servers" / "Causal_Diagnosis_MCP" / "offline"  # models.json, grades.json and their locks
GENERATED = (  # relative to ROOT
    "CausalDiagnosis/results/runs/", "CausalDiagnosis/results/published_runs/", "CausalDiagnosis/docs/walkthroughs/",
    "CausalDiagnosis/cluster_jobs/runs_synced/", "CausalDiagnosis/cluster_jobs/uploads/",
    "CausalDiagnosis/results/calibration_*.md", "CausalDiagnosis/datasets/**/data/*",
    "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json",
    "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json",
    "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json.lock",
)
# configs/ holds the presets in one folder per kind; prices/ holds the price tables and training/ the fold and filter files
# of each asset, which are no presets
NOT_PRESET_FOLDERS = ("prices", "training")


def preset_files(root=None):
    """{name: path} of every preset under root (CONFIGS by default), whichever folder of a kind holds it, and under CONFIGS
    also the presets a dataset brings in datasets/<asset_id>/presets/ and those a framework brings, the JSON files of each folder
    that its declaration names under CONFIGS (CausalDiagnosis/utils/frameworks.py). The one lookup of a preset by name: a name
    that two files share stops the program."""
    root = CONFIGS if root is None else Path(root)
    found = {}
    paths = [path for path in sorted(root.rglob("*.json")) if path.relative_to(root).parts[0] not in NOT_PRESET_FOLDERS]
    if root == CONFIGS:
        paths += dataset_presets() + framework_presets()
    for path in paths:
        if path.stem in found:
            raise SystemExit(f"two presets share the name {path.stem}: {found[path.stem]} and {path}")
        found[path.stem] = path
    return found


def dataset_presets():
    """The preset files that the datasets bring: the JSON files directly inside datasets/<asset_id>/presets/ of every asset
    folder, sorted by asset."""
    return sorted(DATASETS.glob("*/presets/*.json"))


def framework_presets():
    """The preset files of every framework: the JSON files directly inside each folder its declaration names under CONFIGS,
    sorted by framework and folder. The registry reads the declaration as data, so no framework module is imported."""
    from CausalDiagnosis.utils import frameworks  # the registry reads this module's PACKAGE

    return [path for name, folder in frameworks.names("CONFIGS") for path in sorted((frameworks.framework_dir(name) / folder).glob("*.json"))]


def preset_path(name, root=None):
    """The file of the preset of that name under root (CONFIGS by default)."""
    found = preset_files(root)
    if name not in found:
        raise SystemExit(f"unknown config preset {name!r}: one of {sorted(found)}")
    return found[name]


if __name__ == "__main__":
    # the cluster scripts ask for a location by its name (python -m CausalDiagnosis.utils.paths GRIDS), so that a shell
    # script and the Python code read one place
    import sys

    location = globals().get(sys.argv[1]) if len(sys.argv) == 2 else None
    if not isinstance(location, Path):
        raise SystemExit("usage: python -m CausalDiagnosis.utils.paths <the name of a location, such as GRIDS or RESULTS>")
    print(location)
