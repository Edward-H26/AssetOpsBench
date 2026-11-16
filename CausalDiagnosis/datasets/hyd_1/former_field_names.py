"""Field names of the served values: the channel names as AssetOpsBench's hyd_1 asset spells them, with the statistic
appended, and values at six significant digits (CausalDiagnosis/docs/environment.md, section Data).

The facts file names a summary quantity first, as in Pressure_PS1_mean; hyd_1 names the channel first with its unit and
sampling rate, as in PS1_Pressure_bar_100Hz, and calls the motor power P1 rather than EPS1. hyd_1 holds per-cycle means
only, so the statistic is appended as a suffix: Pressure_PS1_std becomes PS1_Pressure_bar_100Hz_std. The environment
serves every value under these names (CausalDiagnosis.environment.Causal_Diagnosis_Environment.env.served_values), so the tool answers and the log agree.
The names are the channel fields of the asset file CausalDiagnosis/datasets/hyd_1/asset.json, which the engine reads from the
asset profile, and this module serves them under their former names to the tests of hyd_1.
"""
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.hyd_1 import records as hyd_1
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import TIER_SENSORS

_PROFILE = hyd_1.tables()
ASSET_ID = hyd_1.ASSET_ID  # the asset of AssetOpsBench whose records the rig's cycles are served as
HYD1_FIELDS = {channel: spec["field"] for channel, spec in _PROFILE.channels.items()}


def public_name(feature_name):
    """Pressure_PS1_std -> PS1_Pressure_bar_100Hz_std."""
    return records.public_name(feature_name, _PROFILE)


def public_fields(sensors):
    """The hyd_1 names of the given channels."""
    return [HYD1_FIELDS[s] for s in sensors]


def tier_fields(tier):
    """The hyd_1 names of the channels of a measurement tier."""
    return public_fields(TIER_SENSORS[tier])
