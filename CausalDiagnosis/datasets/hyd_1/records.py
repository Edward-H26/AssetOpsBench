"""The recordings of the hydraulic test rig hyd_1: the loaders of the recorded levels kind bound to the hyd_1 profile, with
the signatures that CausalDiagnosis.datasets.hyd_1.hyd_1_tables gives them (CausalDiagnosis/docs/adding-a-dataset.md). The data file is the release file
that CausalDiagnosis/datasets/hyd_1/download_and_build.py builds from the CC BY 4.0 UCI archive, or the file that HYD_FACTS_PATH names
when a developer sets it."""
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

ASSET_ID = "hyd_1"


def profile():
    return get_profile(ASSET_ID)


def tables():
    """The profile for the rig modules that read its tables while the engine is imported, validated without the check of
    the engine configuration, which needs the engine (CausalDiagnosis.datasets.dataset_loader.asset_profile.validate)."""
    return get_profile(ASSET_ID, engine=False)


def load_cycles(facts_path):
    return records.load_cycles(facts_path, profile())


def build_blocks(cycles):
    return records.build_blocks(cycles)


def release_path():
    """The release file that CausalDiagnosis/datasets/hyd_1/download_and_build.py builds into the data folder of the rig."""
    return records.release_path(profile())


def default_facts_path():
    """The file that HYD_FACTS_PATH names when it is set and not empty, else the release file."""
    return records.default_facts_path(profile())
