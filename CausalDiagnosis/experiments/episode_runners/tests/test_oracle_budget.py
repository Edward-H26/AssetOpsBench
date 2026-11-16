"""The oracle policy (OraclePolicy) finishes within the budget on every start the paper reports, so that
cost_oracle is defined on every reported episode: the combinations set (the smallest seed of each start combination) and
the evaluation pool (the 8 smallest seeds of each combination, from which master seed 42 draws the evaluation set), under
the static and the timed presets, and its logs on the combinations set stay byte identical now that it submits the current
combination for the scoring at the write time."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import combination_seeds, sampled_combination_seeds  # noqa: E402


class StartOracle(OraclePolicy):
    """The oracle policy as it was before scoring at the write time: it submits the start combination."""

    def act(self, obs):
        action = super().act(obs)
        if action["type"] == "submit_diagnosis":
            action = dict(action, diagnosis=dict(zip(self.env.config.components, self.env.initial)))
        return action


@pytest.mark.parametrize("preset", ["static", "timed", "stochastic"])
def test_the_oracle_s_logs_stay_byte_identical_now_that_it_submits_the_current_combination(preset):
    _facts, cycles, blocks = load_world()
    config = load_config("static")[1]
    if preset == "timed":
        config = load_config("timed")[1]
    elif preset == "stochastic":
        config = EpisodeConfig.from_json({**config.to_json(), "hazard": {"mode": "stochastic", "base_rate": 0.01, "depth_factor": 1.0, "collateral": 0.0}})
    env = HydraulicRigEnv(cycles, blocks, config)
    for seed in combination_seeds(env, 1):
        assert json.dumps(run_episode(env, OraclePolicy(env), seed)) == json.dumps(run_episode(env, StartOracle(env), seed)), seed


@pytest.mark.parametrize("preset", ["static", "timed"])
def test_the_oracle_finishes_within_the_budget_on_the_development_set_and_the_evaluation_pool(preset):
    _facts, cycles, blocks = load_world()
    config = load_config(preset)[1]
    env = HydraulicRigEnv(cycles, blocks, config)
    pool = combination_seeds(env, 8)
    assert set(combination_seeds(env, 1)) <= set(pool) and set(sampled_combination_seeds(env, master_seed=42, pool=8)) <= set(pool)
    for seed in pool:
        log = run_episode(env, OraclePolicy(env), seed)
        end = log[-1]
        assert end["reason"] == "finish" and end["final"] == [HEALTHY[c] for c in COMPONENTS] and end["cost"] <= config.budget, (seed, end["cost"])
        assert not any(e["event"] in ("refused", "order_ineffective") for e in log), seed
