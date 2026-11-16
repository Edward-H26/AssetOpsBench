"""The generated_tokens mode of the real-time clock and the patience scale of the failure card: the rate a configuration takes from the asset file, the refusal of the mode where an asset states no rate, the
texts of a token episode, the words of every preset of another mode, and the scaled patience with its refusals. No test
states a value computed from the rig data."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from CausalDiagnosis.utils.paths import preset_files  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import CLOCK_MODES, TOKEN_MODE
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "CausalDiagnosis" / "configs"
ASSET_FILE = ROOT / "CausalDiagnosis" / "datasets" / "hyd_1" / "asset.json"
FULL = {"layers": ["priced", "real_time", "advanced_real_time", "repair_rules", "deterioration"]}
TOKEN_WORDS = ("seconds_per_token", "tokens you generate", TOKEN_MODE)
REFUSED = "the clock mode generated_tokens needs seconds_per_token in the clock of the asset file"


def token_config(**changes):
    return dataclasses.replace(EpisodeConfig(), clock_mode=TOKEN_MODE, **changes)


def other_mode_presets():
    """The presets of the rig whose configuration plays another clock mode than generated_tokens."""
    out = []
    for path in sorted(preset_files(CONFIGS).values()):
        config = layers.compose(json.loads(path.read_text(encoding="utf-8")))
        if config.asset_id == "hyd_1" and config.clock_mode != TOKEN_MODE:
            out.append(pytest.param(config, id=path.stem))
    return out


def test_a_token_configuration_takes_the_rate_of_the_asset_file_and_every_other_mode_holds_none():
    rate = get_profile("hyd_1", engine=False).defaults["clock"]["seconds_per_token"]
    assert token_config().seconds_per_token == rate and token_config(seconds_per_token=0.0625).seconds_per_token == 0.0625
    for mode in set(CLOCK_MODES) - {TOKEN_MODE}:
        other = dataclasses.replace(token_config(), clock_mode=mode)
        assert other.seconds_per_token is None and "seconds_per_token" not in other.to_json()
    data = token_config().to_json()
    assert data["seconds_per_token"] == rate and EpisodeConfig.from_json(json.loads(json.dumps(data))) == token_config()
    keys = list(data)
    assert keys.index("seconds_per_token") == keys.index("think_seconds_per_cycle") + 1


def test_a_token_configuration_without_seconds_per_token_is_refused_by_name():
    bundled = get_profile("hyd_1")
    clock = {key: value for key, value in bundled.defaults["clock"].items() if key != "seconds_per_token"}
    twin = register_profile(dataclasses.replace(bundled, asset_id="hyd_without_token_rate", defaults={**bundled.defaults, "clock": clock}))
    try:
        assert twin.config_fields()["seconds_per_token"] is None
        with pytest.raises(ValueError, match=REFUSED):
            dataclasses.replace(twin.default_config(), clock_mode=TOKEN_MODE)
        assert dataclasses.replace(twin.default_config(), clock_mode="generation_time").seconds_per_token is None
        for rate in (0, -1.0, float("inf")):
            with pytest.raises(ValueError, match=REFUSED):
                dataclasses.replace(twin.default_config(), clock_mode=TOKEN_MODE, seconds_per_token=rate)
    finally:
        unregister_profile(twin.asset_id)


def test_the_clock_table_of_an_asset_file_may_hold_the_rate_and_nothing_else_beside_its_keys():
    data = json.loads(ASSET_FILE.read_text(encoding="utf-8"))
    assert "seconds_per_token" in data["defaults"]["clock"] and validate(data) == []
    without = json.loads(json.dumps(data))
    del without["defaults"]["clock"]["seconds_per_token"]
    assert validate(without) == []
    misspelled = json.loads(json.dumps(data))
    misspelled["defaults"]["clock"]["seconds_per_tokens"] = 1.0
    assert any(problem.startswith("defaults.clock must hold exactly") and "may hold ['seconds_per_token']" in problem
               for problem in validate(misspelled))


def test_the_profile_clock_block_and_time_tool_of_a_token_episode_state_the_mode_and_the_rate(make_env):
    env = make_env(clock_mode=TOKEN_MODE)
    profile = env.reset(0)["result"]
    rate = env.config.seconds_per_token
    assert profile["clock"]["seconds_per_token"] == rate and "seconds_per_token" not in profile
    assert profile["clock"]["modes"] == {**contract.clock()["modes"], TOKEN_MODE: contract.token_clock()["mode"]}
    assert list(profile) == list(make_env().reset(0)["result"])  # the keys of the profile, which get_asset_profile names, stay
    obs, cost, _done, _info = env.step({"type": "get_current_time"})
    assert obs["result"]["seconds_per_token"] == rate and cost == 0


@pytest.mark.parametrize("config", other_mode_presets())
def test_every_preset_of_another_clock_mode_states_no_word_of_the_token_clock(config, world):
    assert config.seconds_per_token is None and "seconds_per_token" not in config.to_json()
    profile = HydraulicRigEnv(*world, config).reset(0)["result"]
    text = json.dumps(profile)
    assert not any(word in text for word in TOKEN_WORDS), [word for word in TOKEN_WORDS if word in text]


def test_the_failure_switch_scales_the_asset_patience_with_ceil_and_refuses_a_patience_below_one_or_at_the_horizon():
    fields = get_profile("hyd_1").config_fields()
    patience, horizon = fields["failure"]["patience"], fields["horizon"]
    full = layers.compose(FULL).failure
    for scale in (0.5, 2, 0.3, 1):
        scaled = layers.compose({**FULL, "switches": {"failure": {"scale": scale}}}).failure
        assert scaled == {**full, "patience": math.ceil(round(scale * patience, 9))}, scale
    assert layers.compose({**FULL, "switches": {"failure": {"scale": 0.3}}}).failure["patience"] == math.ceil(round(0.3 * patience, 9))
    with pytest.raises(ValueError, match=f"reaches the horizon {horizon} never fires"):
        layers.compose({**FULL, "switches": {"failure": {"scale": horizon / patience}}})
    for bad in ({"scale": 0}, {"scale": -1}, {"scale": True}, {"scale": "2"}, {"scale": float("inf")}, {"scale": 2, "strict": False}):
        with pytest.raises(ValueError, match="the small switch failure takes the mapping"):
            layers.compose({**FULL, "switches": {"failure": bad}})
    assert layers.compose({**FULL, "switches": {"failure": {"strict": False}}}).failure == {**full, "strict": False}
    with pytest.raises(ValueError, match="or a mapping of entries of its card"):
        layers.compose({**FULL, "switches": {"failure": {"bogus": 1}}})
