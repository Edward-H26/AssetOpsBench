"""The scaffold of a new asset folder: a made up asset of two components gets an empty package beside its asset file,
reports exactly the open fields left and validates once they are filled, an existing asset is never overwritten, hyd_1 stays
as it is, and an invalid identifier or the name of a module or a folder of the asset layer is refused. The drafts go under
pytest's tmp_path, outside the repository."""
import hashlib
import json

import pytest

from CausalDiagnosis.datasets import new_dataset_scaffold as scaffold
from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR
from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate

# fan healthy at 0, belt healthy at 100, each listed from healthy to worst
DEGRADATION = {"fan": [0, 1], "belt": [100, 75, 50]}
FILLED = {
    "observation_kinds": {"basic": {"channels": ["T1"]}, "full": {"channels": ["T1", "V1"]}},
    "channels": {"T1": {"field": "T1_Temperature_C_1Hz", "quantity": "Temperature"},
                 "V1": {"field": "V1_Vibration_mm_per_s_10Hz", "quantity": "Vibration"}},
    "statistics.recorded": ["mean", "std"], "statistics.served": ["mean"],
    "defaults.prices.read": {"basic": 2, "full": 6},
    "defaults.prices.parts.fan": 10, "defaults.prices.parts.belt": 5, "defaults.prices.repair_charge": 5,
    "defaults.prices.fix_kind.fan": "replace", "defaults.prices.fix_kind.belt": "replace", "defaults.prices.misdiagnosis": 4,
    "defaults.durations.read_per_cycle": 1, "defaults.durations.advance_per_cycle": 1,
    "defaults.durations.lead_time.fan": 2, "defaults.durations.lead_time.belt": 1,
    "defaults.durations.repair_downtime.fan": 1, "defaults.durations.repair_downtime.belt": 1,
    "defaults.limits.budget": 100, "defaults.limits.horizon": 30, "defaults.limits.max_steps": 30,
    "defaults.clock.clock_mode": "declared", "defaults.clock.seconds_per_cycle": 60, "defaults.clock.think_seconds_per_cycle": 10,
    "rules": [{"kind": "failure", "parameters": {"patience": 5, "strict": False, "penalty": 10}}],
    "sampler.fault_rule.min_faulty": 1, "sampler.fault_rule.max_faulty": 1,
    **{f"failure_modes[{i}].{key}": value for i, (code, severity) in enumerate(
        [("fan_ok", "healthy"), ("fan_stalled", "critical"), ("belt_ok", "healthy"), ("belt_worn", "degraded"), ("belt_slipping", "critical")])
       for key, value in (("code", code), ("severity", severity), ("action", "none" if severity == "healthy" else "replace"))},
    "knowledge.fmea_source": "made_up", "knowledge.association_source": "made_up",
    "knowledge.indicates": {"T1": ["fan"], "V1": ["belt"]}, "knowledge.correlations": {"fan": {"T1": 0.9}, "belt": {"V1": 0.8}},
    "records.label_keys.fan": "fan_state", "records.label_keys.belt": "belt_tension", "records.stable_flag_stable": 0,
    "records.data_env": "FAN_RIG_FACTS_PATH", "records.release_dir": "CausalDiagnosis/datasets/fan_rig/data", "records.release_file": "cycles.jsonl",
    "records.page": "https://example.org/fan_rig", "records.fetch": "python CausalDiagnosis/datasets/fan_rig/download_and_build.py",
    "text.noun": "fan test rig", "text.short_noun": "rig", "text.episode_noun": "fan rig", "text.unit_noun": "unit",
    "text.whole_asset_patterns[0]": "rig", "text.registry_file": None, "text.example_channel": "T1",
    "text.fallback.asset_type": "FAN_RIG", "text.fallback.description": "Made up fan rig of the scaffold test", "text.fallback.site": "TEST",
    "scoring": "program",
}


def filled(value):
    """The profile with every open field replaced by its value in FILLED."""
    if isinstance(value, dict):
        return {k: filled(v) for k, v in value.items()}
    if isinstance(value, list):
        return [filled(v) for v in value]
    if isinstance(value, str) and value.startswith(scaffold.OPEN):
        return FILLED[value[len(scaffold.OPEN):]]
    return value


def digest(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_a_made_up_two_component_asset_lists_exactly_its_open_fields_and_validates_once_they_are_filled(tmp_path):
    path = scaffold.scaffold("fan_rig", DEGRADATION, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "fan_rig" / "asset.json"
    assert sorted(p.name for p in path.parent.iterdir()) == ["__init__.py", "asset.json"]
    assert (path.parent / "__init__.py").read_text(encoding="utf-8") == ""
    assert data["levels"] == {"fan": [0, 1], "belt": [50, 75, 100]} and data["healthy"] == {"fan": 0, "belt": 100}
    assert data["degradation"] == DEGRADATION
    left = scaffold.open_fields(data)
    assert sorted(left) == sorted(FILLED) and len(left) == len(set(left))
    lines = scaffold.report(data)
    assert lines[0] == f"{len(left)} open fields to fill:" and [line.strip() for line in lines[1:len(left) + 1]] == left
    assert validate(data) and lines[len(left) + 1].startswith("validate reports ")
    complete = filled(data)
    assert scaffold.open_fields(complete) == [] and validate(complete) == []
    assert scaffold.report(complete) == ["0 open fields to fill:", "validate: valid"]


def test_an_existing_asset_is_never_overwritten_and_hyd_1_stays_as_it_is(tmp_path):
    path = scaffold.scaffold("fan_rig", DEGRADATION, tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        scaffold.scaffold("fan_rig", {"fan": [0, 1]}, tmp_path)
    assert path.read_bytes() == before
    hyd_1 = digest(ASSETS_DIR / "hyd_1")
    with pytest.raises(SystemExit):
        scaffold.main(["hyd_1", "--component", "pump=0,1,2"])
    assert digest(ASSETS_DIR / "hyd_1") == hyd_1


@pytest.mark.parametrize("asset_id", ["fan-rig", "../fan_rig", "", "fan rig", "dataset_loader", "tests"])
def test_an_invalid_identifier_is_refused_and_writes_nothing(tmp_path, asset_id):
    with pytest.raises(SystemExit):
        scaffold.scaffold(asset_id, DEGRADATION, tmp_path)
    assert list(tmp_path.iterdir()) == []
