"""Explicit development environment semantics, independent of asset and agent choice.

Legacy configurations omit the mode and retain their original bytes and behavior.
"""
import copy

MODES = ("static", "dynamic")
STATIC_OFF = ("time_tools", "read_per_cycle", "lead_time", "repair_downtime",
              "downtime_rate", "deterioration", "collateral", "failure", "emergency")


def normalize(data, mode):
    from . import layers
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

    if mode not in MODES:
        raise ValueError(f"environment_mode must be one of {MODES}, not {mode!r}")
    out = copy.deepcopy(data)
    profile = get_profile(out["asset_id"])
    settings = ({name: "off" for name in STATIC_OFF} | {"clock_mode": "declared"}
                if mode == "static" else
                {"clock_mode": "generation_time", "time_tools": "on", "read_per_cycle": "on",
                 "lead_time": "on", "repair_downtime": "on"})
    for name, setting in settings.items():
        for path, value in layers.SWITCHES[name].SWITCHES[name].values(setting, profile).items():
            layers.set_path(out, path, value)
    out["environment_mode"] = mode
    out.pop("seconds_per_token", None)
    return out


def apply(config, mode):
    return type(config).from_json(normalize(config.to_json(), mode))


def problems(config):
    mode = config.environment_mode
    if mode is None:
        return []
    if mode not in MODES:
        return [f"environment_mode must be one of {MODES}"]
    expected = normalize(config.to_json(), mode)
    # Reject inconsistent raw serialized configurations rather than silently repairing a replay.
    actual = config.to_json()
    return [f"environment_mode {mode} conflicts with {key}"
            for key, value in expected.items() if actual.get(key) != value]
