"""Shared fixtures of the asset layer tests (CausalDiagnosis/docs/adding-a-dataset.md): the raw hyd_1 asset file, the tiny synthetic
asset of three components in tiny_asset.json, a seeded Gaussian generator of recordings for any recorded levels profile,
the made-up port of the folder ports with its records, the stored digests of the hyd_1 configuration, and the lines that
point a child process at the session's copies of the generated tables. No test here needs the data file except those
marked needs_data."""
import importlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.records import default_facts_path
from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # CausalDiagnosis/datasets/tests -> the fork root
HYD_1_FILE = ASSETS_DIR / "hyd_1" / "asset.json"
TINY_FILE = HERE / "tiny_asset.json"
PORTS = HERE / "ports"  # the folder of the made-up ports, two levels below the assets folder, which only mount_port mounts
PORT_ID = "demo_1"
DIGESTS_FILE = ROOT / "CausalDiagnosis" / "experiments" / "run_verification" / "invariant_digests.json"
BASELINE = "baseline"  # the stored digests that the hyd_1 asset file reproduces

HAS_DATA = Path(default_facts_path()).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")


@pytest.fixture
def hyd_1_data():
    """A fresh copy of the hyd_1 asset file, free to mutate."""
    return json.loads(HYD_1_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def tiny_data():
    return json.loads(TINY_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def tiny(tiny_data):
    """The validated tiny profile, registered under its asset id for the length of the test."""
    profile = register_profile(AssetProfile.from_json(tiny_data, str(TINY_FILE)))
    yield profile
    unregister_profile(profile.asset_id)


def write_recordings(path, profile, seed=0, stable=10):
    """A data file of the recorded levels kind for any profile: for every combination in profile order, stable cycles
    followed by one unstable cycle, so every stable block is contiguous. The mean of a channel moves by the degradation
    rank of each component the channel indicates, with seeded Gaussian noise, so the components are identifiable."""
    rng = random.Random(seed)
    indicates = profile.knowledge["indicates"]
    unstable = 1 - profile.records["stable_flag_stable"]
    index = 0
    with open(path, "w", encoding="utf-8") as f:
        for combo in profile.combinations():
            rank = {c: profile.degradation[c].index(level) for c, level in zip(profile.components, combo)}
            labels = {profile.records["label_keys"][c]: level for c, level in zip(profile.components, combo)}
            for flag in [profile.records["stable_flag_stable"]] * stable + [unstable]:
                features = []
                for position, (channel, spec) in enumerate(profile.channels.items()):
                    centre = 10.0 * (position + 1) + sum(rank[c] for c in indicates.get(channel, ()))
                    values = {"mean": centre + rng.gauss(0, 0.1), "std": abs(rng.gauss(0.5, 0.1)),
                              "min": centre - 1 + rng.gauss(0, 0.1), "max": centre + 1 + rng.gauss(0, 0.1)}
                    features += [{"name": f"{spec['quantity']}_{channel}_{s}", "value": values[s]} for s in profile.statistics["recorded"]]
                f.write(json.dumps({"cycle_index": index, "labels": {**labels, "stable_flag": flag}, "features": features}) + "\n")
                index += 1
    return path


@pytest.fixture
def tiny_recordings(tmp_path, tiny):
    return write_recordings(tmp_path / "tiny_cycles.jsonl", tiny)


@contextmanager
def mount_port(folder=PORTS):
    """The asset folders under folder found by the engine for the length of a with block, as a port installed beside the
    tree would be: folder joins the search path of the package CausalDiagnosis.datasets, and afterwards the path entry, the evidence
    kinds that the packages registered, the cached profiles and the modules of the packages leave again. No file of the
    tree is written."""
    import CausalDiagnosis.datasets
    from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import _bundled

    ids = [path.parent.name for path in Path(folder).glob("*/asset.json")]
    registered = set(kinds.names())
    CausalDiagnosis.datasets.__path__.append(str(folder))
    try:
        yield Path(folder)
    finally:
        CausalDiagnosis.datasets.__path__.remove(str(folder))
        for name in set(kinds.names()) - registered:
            kinds.unregister_kind(name)
        _bundled.cache_clear()
        for module in [m for m in sys.modules if any(m == f"CausalDiagnosis.datasets.{i}" or m.startswith(f"CausalDiagnosis.datasets.{i}.") for i in ids)]:
            del sys.modules[module]


@pytest.fixture
def mounted_port():
    """The made-up port demo_1, mounted from its folder alone for the length of the test, with its package imported, so that
    the evidence kind it registers is served to a test that validates its asset file before any profile of it is read."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports

    with mount_port() as folder:
        ports.package(PORT_ID)
        yield folder / PORT_ID


def write_port_records(path, levels=(0, 1), per_level=7):
    """A data file of the made-up port: per_level records of the press at each level, one value for each of its two
    channels, the force rising and the stroke falling with the level."""
    with open(path, "w", encoding="utf-8") as f:
        index = 0
        for level in levels:
            for k in range(per_level):
                f.write(json.dumps({"index": index, "press_level": level, "F1": 100.0 + 30 * level + 0.5 * k, "F2": 50.0 - 3 * level + 0.1 * k}) + "\n")
                index += 1
    return path


@pytest.fixture
def port_records(tmp_path, monkeypatch, mounted_port):
    """The data file of the mounted port in the temporary folder, named by the variable of its asset file."""
    path = write_port_records(tmp_path / "demo_records.jsonl")
    monkeypatch.setenv("DEMO_1_RECORDS_PATH", str(path))
    return path


@pytest.fixture(scope="session")
def baseline_digests():
    """The digests of BASELINE, stored once per facts file. The configuration digests are the same under every key."""
    baselines = json.loads(DIGESTS_FILE.read_text(encoding="utf-8"))["baselines"]
    return next(by_label[BASELINE]["digests"] for by_label in baselines.values() if BASELINE in by_label)


def session_tables():
    """The lines that point a child Python process at the copy of every generated table that this process reads
    (CausalDiagnosis.datasets.dataset_loader.dataset_packages.generated_tables), the copies of the session that the root conftest.py makes, so that a child which
    computes a table writes its copy and not the file of the tree."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports

    lines = ["import importlib", "from pathlib import Path"]
    for module, attribute in ports.generated_tables():
        path = getattr(importlib.import_module(module), attribute)
        lines.append(f"setattr(importlib.import_module({module!r}), {attribute!r}, Path({str(path)!r}))")
    return lines
