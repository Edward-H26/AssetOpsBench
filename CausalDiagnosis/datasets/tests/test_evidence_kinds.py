"""The registry of evidence kinds (CausalDiagnosis/datasets/dataset_loader/evidence_kinds.py): the two built in kinds, the registration of a further kind, the
validation that reads the registry and checks a registered kind as it checks recorded levels, the rule cards of recorded
levels on every kind that plays episodes, the served statistics of the asset file, and the deterioration cards refused for
a kind that replays no repair."""
import copy
import dataclasses

import pytest

from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence, evidence_kinds as kinds, recordings as records
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose

from CausalDiagnosis.datasets.tests.conftest import write_recordings
from CausalDiagnosis.datasets.tests.test_asset_profile import SCRIPTED

MADE_UP = "made_up_readings"
DETERIORATION_REFUSAL = ("an asset of the demo_history kind holds no recording after a change that no work order made, so the "
                         "deterioration cards must be off")


@pytest.fixture
def made_up_kind():
    """A kind registered for the length of a test, with the source and the loader of recorded levels, one record key of
    its own and a check of its own."""
    kind = kinds.register_kind(kinds.EvidenceKind(MADE_UP, record_keys=("made_up_key",), check=lambda data: ["the made-up check ran"],
                                                  loader=records.load_recordings, source=evidence.RecordingSource, loop=evidence.loop))
    yield kind
    kinds.unregister_kind(MADE_UP)


def test_the_built_in_kinds_are_recorded_levels_and_scripted_cases():
    assert kinds.names() == ("recorded_levels", "scripted_cases")
    recorded, scripted = kinds.kind_of("recorded_levels"), kinds.kind_of("scripted_cases")
    assert recorded.plays_episodes and recorded.replays_repairs and recorded.record_keys == ("label_keys", "stable_flag_stable")
    live, before = recorded.part("loop")(get_profile("hyd_1"))
    assert live == before == evidence.SEGMENT
    assert not scripted.plays_episodes and not scripted.replays_repairs and scripted.loop is None


def test_a_registered_kind_is_served_until_removed_and_a_taken_name_is_refused():
    kind = kinds.EvidenceKind(MADE_UP, loader=records.load_recordings, source=evidence.RecordingSource)
    assert kinds.register_kind(kind) is kind
    try:
        assert kinds.kind_of(MADE_UP) is kind and kinds.names() == ("recorded_levels", "scripted_cases", MADE_UP)
        for taken in (kind, dataclasses.replace(kind, name="recorded_levels"), dataclasses.replace(kind, name="scripted_cases")):
            with pytest.raises(ValueError, match="is already declared"):
                kinds.register_kind(taken)
    finally:
        kinds.unregister_kind(MADE_UP)
    assert kinds.kind_of(MADE_UP) is None and kinds.names() == ("recorded_levels", "scripted_cases")
    with pytest.raises(ValueError, match="the built in evidence kind 'recorded_levels' cannot be removed"):
        kinds.unregister_kind("recorded_levels")


def test_a_kind_that_plays_episodes_needs_a_loader_and_a_source():
    for given, missing in (({}, "a loader and a source"), ({"loader": records.load_recordings}, "a source"),
                           ({"source": evidence.RecordingSource}, "a loader")):
        with pytest.raises(ValueError, match=f"plays episodes, so it needs {missing}$"):
            kinds.register_kind(kinds.EvidenceKind(MADE_UP, **given))
    assert kinds.kind_of(MADE_UP) is None
    kinds.register_kind(kinds.EvidenceKind(MADE_UP, plays_episodes=False, replays_repairs=False))
    kinds.unregister_kind(MADE_UP)


def test_validation_reads_the_registry_and_checks_a_registered_kind_as_it_checks_recorded_levels(made_up_kind, tiny_data):
    unknown = dict(tiny_data, evidence_kind="unknown_kind")
    assert validate(unknown) == ["evidence_kind must be one of ['recorded_levels', 'scripted_cases', 'made_up_readings']"]
    broken = copy.deepcopy(tiny_data)
    broken["healthy"]["belt"] = 60
    assert validate(broken) == ["the healthy level of belt must be one of its levels"]
    assert validate(dict(broken, evidence_kind=MADE_UP)) == [
        "the healthy level of belt must be one of its levels",
        "records must hold ['made_up_key', 'data_env', 'release_dir', 'release_file'] and may add ['page', 'fetch']",
        "the made-up check ran",
    ]


def test_a_card_for_recorded_levels_applies_to_every_kind_that_plays_episodes(tiny_data):
    kind = kinds.register_kind(kinds.EvidenceKind(MADE_UP, record_keys=("label_keys", "stable_flag_stable"), check=records.record_problems,
                                                  loader=records.load_recordings, source=evidence.RecordingSource, loop=evidence.loop))
    try:
        assert any(card["kind"] == "fix_history" for card in tiny_data["rules"])
        assert validate(dict(tiny_data, evidence_kind=kind.name)) == []
    finally:
        kinds.unregister_kind(MADE_UP)
    scripted = copy.deepcopy(SCRIPTED)
    scripted["rules"].append({"kind": "fix_history", "parameters": {"healthy_probability": 0.5, "healthy_ago": [2, 20],
                                                                     "degraded_probability": 0.5, "degraded_ago": [2, 20]}})
    assert validate(scripted) == ["rules[0] (fix_history): the kind applies to ['recorded_levels'], not to scripted_cases"]


def _tiny_episode(tiny_data, served, tmp_path):
    data = copy.deepcopy(tiny_data)
    data["statistics"]["served"] = served
    profile = register_profile(AssetProfile.from_json(data))
    cycles = records.load_cycles(write_recordings(tmp_path / f"{'_'.join(served)}.jsonl", profile), profile)
    return HydraulicRigEnv(cycles, records.build_blocks(cycles), profile.default_config())


def test_a_reading_serves_the_statistics_of_the_asset_file_in_their_order(tiny_data, tmp_path):
    try:
        for served in (["mean"], ["std", "mean"]):
            env = _tiny_episode(tiny_data, served, tmp_path)
            profile = env.reset(0)["result"]
            assert profile["statistics"] == served
            envelope, _cost, _done, _info = env.step({"type": "read_sensors", "sensor_group": "full", "cycles": 1})
            values = envelope["result"]["served"][0]["values"]
            assert set(values) == {f"{env.profile.field_name(c)}_{s}" for c in ("T1", "V1", "I1") for s in served}
            assert set(env.normal_ranges) == {f"{env.profile.field_name(c)}_{served[0]}" for c in env.profile.channels}
    finally:
        unregister_profile("tiny_1")


def test_a_kind_that_replays_no_repair_is_refused_the_deterioration_cards(mounted_port):
    config = get_profile("demo_1").default_config()
    drawing = ({"schedule": {"mode": "seeded", "components": 1, "first_drop": [2, 5], "interval": 3}},
               {"hazard": {"mode": "stochastic", "base_rate": 0.1, "depth_factor": 1.0, "collateral": 0.0}})
    for change in drawing:
        with pytest.raises(ValueError, match=DETERIORATION_REFUSAL):
            dataclasses.replace(config, **change)
    with pytest.raises(ValueError, match=DETERIORATION_REFUSAL):
        compose({"asset_id": "demo_1", "layers": ["deterioration"], "switches": {"deterioration": {"mode": "hazard", "base_rate": 0.1}}})
    composed = compose({"asset_id": "demo_1", "layers": ["priced", "real_time", "advanced_real_time", "repair_rules"]})
    assert composed.asset_id == "demo_1" and composed.has_clock()
