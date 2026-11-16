"""Shared pieces of the AssetOpsBench reproduction (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md): the import paths, the pinned scenario
file, the runnable subset by family, the family and task kind of a scenario id, and the names of the run and episode files.

The scenario rows are read at run time from the Hugging Face file the README names, and nothing of the dataset is stored here.
"""
from CausalDiagnosis.utils.paths import SRC
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
from CausalDiagnosis.utils.paths import ROOT  # noqa: E402  run_scenarios reads ROOT from here


for _path in (str(SRC),):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# The paper's 141 scenarios are the config "scenarios" of ibm-research/AssetOpsBench at the commit below, the only one that
# holds exactly 141 rows. Later commits rewrote the characteristic form of id 112 and added rows, so the file is pinned by
# its SHA-256, and a run on any other file says so in run.json.
SCENARIO_REVISION = "d925adba1f1f05b54e44503ad01299c999af37ef"
SCENARIO_URL = ("https://huggingface.co/datasets/ibm-research/AssetOpsBench/resolve/"
                f"{SCENARIO_REVISION}/data/scenarios/all_utterance.jsonl")
SCENARIO_SHA256 = "ed87bdbd38c150c28130b5ebbfb60fc9a8c974a05514db919eed2a92690b5bd0"
# That file with the forms of 621 and 622, which it lacks, from a later commit (fetch_public_servers_and_data.py). The driver reads this one.
LAYER_SCENARIO_SHA256 = "955a091f7fc0c85c75d6ce0fe1835e481378d9de9a5787c22102ea7be28dd279"


def sha256_of(path):
    from CausalDiagnosis.datasets.dataset_loader.recordings import file_sha256  # the one SHA-256 of a file of the tree

    return file_sha256(path)


def load_scenarios(path, ids, pinned=True):
    """{id: row} of the scenario file for the given ids. A pinned load refuses any file but the one fetch_public_servers_and_data.py builds, so that
    the texts and forms of a run are the paper's. Every requested id must be present."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"scenario file {path} not found. Download it with: curl -L -o {path} {SCENARIO_URL}")
    digest = sha256_of(path)
    if pinned and digest != LAYER_SCENARIO_SHA256:
        raise SystemExit(f"scenario file {path} has SHA-256 {digest}, not the {LAYER_SCENARIO_SHA256} that fetch_public_servers_and_data.py builds "
                         f"from commit {SCENARIO_REVISION[:7]}. Pass that file or --allow-unpinned")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["id"])] = row
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"scenario file {path} lacks ids {missing}")
    return {i: rows[i] for i in ids}
