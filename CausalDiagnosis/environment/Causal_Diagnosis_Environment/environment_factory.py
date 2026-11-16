"""Environments built from their episode configuration: the asset that config.asset_id names supplies the data file and
the recordings that the environment serves (CausalDiagnosis/docs/adding-a-dataset.md)."""
from functools import lru_cache

from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, recordings as records
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv


def default_facts_path(config):
    """The data file of the configured asset: the file that the environment variable of its profile names when the
    variable is set and not empty, else its release file."""
    return records.default_facts_path(get_profile(config.asset_id))


@lru_cache(maxsize=2)
def load_world(asset_id, facts_path):
    """The records and the blocks of one data file of an asset, read by the loader of its evidence kind once per process,
    since the file is large and read only and every environment of the process shares them. For the recorded levels kind
    these are the recorded cycles and their stable blocks."""
    profile = get_profile(asset_id)
    return kinds.kind_of(profile.evidence_kind).part("loader")(facts_path, profile)


def make_env(config, facts_path=None):
    """The environment of an episode configuration over a data file of its asset, the asset's default file when facts_path
    is None."""
    path = str(facts_path or default_facts_path(config))
    return HydraulicRigEnv(*load_world(config.asset_id, path), config)
