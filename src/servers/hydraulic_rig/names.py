"""Field names of the served values: the channel names as AssetOpsBench's hyd_1 asset spells them, with the statistic
appended, and values at six significant digits (docs/hydraulic_rig/environment.md, section Data).

The facts file names a summary quantity first, as in Pressure_PS1_mean; hyd_1 names the channel first with its unit and
sampling rate, as in PS1_Pressure_bar_100Hz, and calls the motor power P1 rather than EPS1. hyd_1 holds per-cycle means
only, so the statistic is appended as a suffix: Pressure_PS1_std becomes PS1_Pressure_bar_100Hz_std. The environment
serves every value under these names (envs.hydraulic_rig.env.served_values), so the tool answers and the log agree.
"""
from envs.hydraulic_rig.data import TIER_SENSORS

ASSET_ID = "hyd_1"  # the asset of AssetOpsBench whose records the rig's cycles are served as
HYD1_FIELDS = {
    "PS1": "PS1_Pressure_bar_100Hz",
    "PS2": "PS2_Pressure_bar_100Hz",
    "PS3": "PS3_Pressure_bar_100Hz",
    "PS4": "PS4_Pressure_bar_100Hz",
    "PS5": "PS5_Pressure_bar_100Hz",
    "PS6": "PS6_Pressure_bar_100Hz",
    "EPS1": "P1_MotorPower_W_100Hz",
    "FS1": "FS1_VolumeFlow_l_per_min_10Hz",
    "FS2": "FS2_VolumeFlow_l_per_min_10Hz",
    "TS1": "TS1_Temperature_C_1Hz",
    "TS2": "TS2_Temperature_C_1Hz",
    "TS3": "TS3_Temperature_C_1Hz",
    "TS4": "TS4_Temperature_C_1Hz",
    "VS1": "VS1_Vibration_mm_per_s_1Hz",
    "CE": "CE_CoolingEfficiency_percent_1Hz",
    "CP": "CP_CoolingPower_kW_1Hz",
    "SE": "SE_EfficiencyFactor_percent_1Hz",
}


def public_name(feature_name):
    """Pressure_PS1_std -> PS1_Pressure_bar_100Hz_std."""
    _quantity, channel, statistic = feature_name.rsplit("_", 2)
    return f"{HYD1_FIELDS[channel]}_{statistic}"


def public_result(result):
    """Recorded summaries as a tool answer carries them: public names, six significant digits, sorted by name."""
    return {public_name(k): float(f"{v:.6g}") for k, v in sorted(result.items())}


def public_fields(sensors):
    """The hyd_1 names of the given channels."""
    return [HYD1_FIELDS[s] for s in sensors]


def tier_fields(tier):
    """The hyd_1 names of the channels of a measurement tier."""
    return public_fields(TIER_SENSORS[tier])
