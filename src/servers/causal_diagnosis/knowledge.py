"""Static knowledge the causal-diagnosis server serves: the channel-to-component associations, the failure-mode codes of the
hydraulic rig, and the asset description (docs/hydraulic_rig/causal-diagnosis-server.md).

INDICATES: a channel indicates a component when it, or the virtual sensor served as CE, CP or SE, is among the component's
most correlated features in Helwig et al. 2015, Table II (p. 3), with SE taken as the table's system efficiency; the
temperatures follow the cooler (pp. 3 to 4). PS4 to PS6, FS2 and VS1 indicate no component. An association is diagnostic
evidence, not a causal edge.

FAILURE_CODES: the 16 distinct failure-mode entries of the IndustryAssetEQA hydraulic facts file, extracted once from its
records without their cycles (component, the condition level that carries the code, code, severity and
recommended action); ten of them are faults, four say a component is healthy, two describe the operating regime.
"""
import json
from pathlib import Path

from envs.hydraulic_rig.data import SENSORS, TIER_SENSORS
from servers.hydraulic_rig.names import ASSET_ID, HYD1_FIELDS

ASSOCIATION_SOURCE = "helwig2015condition"  # Helwig, Pignanelli and Schuetze 2015, Table II, pp. 3 to 4
FMEA_SOURCE = "industryasseteqa_hydraulic_facts"
STATISTICS = ("mean", "std", "min", "max")

INDICATES = {
    "CE": ("cooler",), "CP": ("cooler",), "TS1": ("cooler",), "TS2": ("cooler",), "TS3": ("cooler",), "TS4": ("cooler",),
    "PS1": ("valve", "accumulator"), "PS2": ("valve", "accumulator"), "PS3": ("valve", "pump", "accumulator"),
    "FS1": ("valve", "pump", "accumulator"), "EPS1": ("valve", "accumulator"), "SE": ("valve", "pump", "accumulator"),
}
# Helwig et al. 2015, Table II: the correlation of each component's condition with its most correlated features, returned
# as an assumption because they belong to interval features of the fixed working cycle rather than to the served summaries.
HELWIG_CORRELATIONS = {
    "cooler": {"CE": 0.98, "CP": 0.98, "TS1": 0.95, "TS2": 0.95, "TS3": 0.95, "TS4": 0.95},
    "valve": {"PS1": 0.69, "PS2": 0.67, "PS3": 0.60, "FS1": 0.59, "EPS1": 0.57, "SE": 0.37},
    "pump": {"FS1": 0.72, "SE": 0.65, "PS3": 0.45},
    "accumulator": {"EPS1": 0.69, "PS1": 0.60, "PS2": 0.58, "PS3": 0.52, "FS1": 0.46, "SE": 0.37},
}

FAILURE_CODES = (
    {"component": "cooler", "level": 3, "code": "cooler_efficiency_critical", "severity": "critical", "action": "inspect_cooler;clean_or_replace_heat_exchanger"},
    {"component": "cooler", "level": 20, "code": "cooler_efficiency_reduced", "severity": "degraded", "action": "schedule_maintenance;check_flow_and_fouling"},
    {"component": "cooler", "level": 100, "code": "cooler_ok", "severity": "healthy", "action": "none"},
    {"component": "valve", "level": 73, "code": "valve_switching_critical", "severity": "critical", "action": "inspect_valve_actuator;replace_valve_or_control"},
    {"component": "valve", "level": 80, "code": "valve_switching_severe_lag", "severity": "degraded", "action": "calibrate_actuator;check_drive"},
    {"component": "valve", "level": 90, "code": "valve_switching_small_lag", "severity": "degraded", "action": "monitor"},
    {"component": "valve", "level": 100, "code": "valve_ok", "severity": "healthy", "action": "none"},
    {"component": "pump", "level": 2, "code": "pump_internal_leakage_severe", "severity": "critical", "action": "replace_pump_or_repair_seal"},
    {"component": "pump", "level": 1, "code": "pump_internal_leakage_weak", "severity": "degraded", "action": "monitor;plan_repair"},
    {"component": "pump", "level": 0, "code": "pump_ok", "severity": "healthy", "action": "none"},
    {"component": "accumulator", "level": 90, "code": "accumulator_pressure_low_critical", "severity": "critical", "action": "inspect_accumulator;service_or_replace"},
    {"component": "accumulator", "level": 100, "code": "accumulator_pressure_severely_reduced", "severity": "degraded", "action": "service_accumulator"},
    {"component": "accumulator", "level": 115, "code": "accumulator_pressure_reduced", "severity": "degraded", "action": "monitor"},
    {"component": "accumulator", "level": 130, "code": "accumulator_ok", "severity": "healthy", "action": "none"},
    {"component": "system_regime", "level": None, "code": "transient", "severity": "info", "action": "wait_until_stable"},
    {"component": "system_regime", "level": None, "code": "steady", "severity": "info", "action": "none"},
)
CODES = {e["code"]: e for e in FAILURE_CODES}
CODE_OF_LEVEL = {(e["component"], e["level"]): e["code"] for e in FAILURE_CODES if e["level"] is not None}
FAULT_CODES = tuple(e for e in FAILURE_CODES if e["severity"] in ("critical", "degraded"))
PREFIX_TO_SENSOR = {HYD1_FIELDS[s].split("_", 1)[0]: s for s in SENSORS}  # P1 -> EPS1
ASSET_PROFILE = Path(__file__).resolve().parents[2] / "couchdb" / "scenarios_data" / "shared" / "iot" / "asset_profile_sample.json"


def asset_profile(asset_id):
    """The registry entry of hyd_1 from the fork's sample profiles, or its known facts when the file is absent. The server
    describes no other asset, since the components, levels and channels it serves are the rig's."""
    if asset_id != ASSET_ID:
        return None
    try:
        for entry in json.loads(ASSET_PROFILE.read_text(encoding="utf-8")):
            if entry.get("assetnum") == asset_id:
                return {"asset_id": asset_id, "asset_type": entry.get("assettype"), "description": entry.get("description"),
                        "site": entry.get("siteid")}
    except (OSError, ValueError):
        pass
    return {"asset_id": asset_id, "asset_type": "HYDRAULIC_PUMP", "description": "Hydraulic pump condition-monitoring rig", "site": "MAIN"}


def sensor_of(name):
    """PS1, P1, PS1_Pressure_bar_100Hz or PS1_Pressure_bar_100Hz_mean -> the sensor's short name (P1 is EPS1), or None."""
    if name in SENSORS:
        return name
    return PREFIX_TO_SENSOR.get(name.split("_", 1)[0])


def unit_and_rate(sensor):
    """FS1 -> ('l_per_min', '10Hz'), from the asset's field name FS1_VolumeFlow_l_per_min_10Hz."""
    parts = HYD1_FIELDS[sensor].split("_")
    return "_".join(parts[2:-1]), parts[-1]


def statistic_of(field):
    """PS1_Pressure_bar_100Hz_mean -> mean."""
    return field.rsplit("_", 1)[1]


def group_of(sensor):
    """The cheapest measurement tier that serves the sensor, as tier1, tier2 or tier3."""
    return f"tier{min(t for t, sensors in TIER_SENSORS.items() if sensor in sensors)}"


def fault_codes_of(component):
    return [e for e in FAULT_CODES if e["component"] == component]


def level_code(component, level):
    return CODE_OF_LEVEL.get((component, level))
