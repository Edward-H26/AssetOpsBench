"""The tables and the record loaders of hyd_1 under the names that the rig modules gave them, for the tests.

The tables and the record loaders belong to the asset hyd_1 in CausalDiagnosis/datasets/hyd_1 (CausalDiagnosis/docs/adding-a-dataset.md).
The engine, the servers, the agent and the drivers read them from the asset profile, and this module serves them under
their former names to the tests of hyd_1. The environment reads the release file, which
CausalDiagnosis/datasets/hyd_1/download_and_build.py builds from the CC BY 4.0 UCI raw data with the labels and the per-cycle features
(CausalDiagnosis/docs/environment.md).
"""
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.hyd_1 import records as hyd_1
from CausalDiagnosis.datasets.dataset_loader.recordings import Cycle, build_blocks, percentile  # noqa: F401  (served under their former names)

_PROFILE = hyd_1.tables()
COMPONENTS = _PROFILE.components
LABEL_KEYS = dict(_PROFILE.records["label_keys"])
LEVELS = dict(_PROFILE.levels)
HEALTHY = dict(_PROFILE.healthy)
SENSORS = tuple(_PROFILE.channels)
TIER_SENSORS = {int(group.removeprefix("tier")): tuple(spec["channels"]) for group, spec in _PROFILE.observation_kinds.items()}
SUMMARY_SUFFIXES = records.summary_suffixes(_PROFILE)
STABLE_FLAG_STABLE = _PROFILE.records["stable_flag_stable"]  # UCI convention: 0 = conditions were stable, 1 = static conditions might not have been reached
DATA_DIR = Path(__file__).resolve().parent / "data"  # what CausalDiagnosis/datasets/hyd_1/download_and_build.py downloads and builds


def load_cycles(facts_path):
    return records.load_cycles(facts_path, _PROFILE)


def group_features(cycle, sensors, suffixes=SUMMARY_SUFFIXES):
    """The recorded summaries of the given sensors whose names end with one of the suffixes, from the cycle's features."""
    return records.group_features(cycle, sensors, _PROFILE, suffixes)


def tier_features(cycle, tier):
    """The mean, std, min and max of every sensor in the tier, taken from the cycle's recorded features."""
    return group_features(cycle, TIER_SENSORS[tier])


def default_facts_path():
    """The file that HYD_FACTS_PATH names when it is set and not empty, else the release file that
    CausalDiagnosis/datasets/hyd_1/download_and_build.py builds into DATA_DIR."""
    return hyd_1.default_facts_path()
