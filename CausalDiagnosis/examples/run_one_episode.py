"""One rollout of the oracle policy and one of the do nothing policy through the Gymnasium wrapper: the oracle and
baseline check of OpenEnv's RFC 008, which asks that the oracle earn more than doing nothing, since otherwise the
reward does not measure the task. The script prints which policy earned the higher reward and no value: a number
from a run is a result, and results come from the drivers under CausalDiagnosis/ only.

    PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.datasets.hyd_1.download_and_build --download
    PATH=$PWD/.venv/bin:$PATH python CausalDiagnosis/examples/run_one_episode.py --seed 0
"""
import argparse
import sys
from pathlib import Path

# the fork is not installed as a package; a script started by its path puts the fork root, which holds CausalDiagnosis/, on the path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import HydraulicRigGym  # noqa: E402
from CausalDiagnosis.environment.reference_policies.policies import DoNothingPolicy, OraclePolicy  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory import default_facts_path  # noqa: E402


def rollout(gym, policy, seed):
    """Play one episode to its end and return the terminal reward. The policy reads the environment's own answer
    dict, which the gym keeps beside the text it returns, so an in-process policy and a text policy see the same
    episode."""
    gym.reset(seed)
    while True:
        _text, reward, terminated, truncated, _info = gym.step(policy.act(gym.observation))
        if terminated or truncated:
            return reward


def main():
    parser = argparse.ArgumentParser(description="Oracle and do nothing rollouts through the hydraulic rig gym.")
    parser.add_argument("--seed", type=int, default=0, help="seed of both episodes")
    args = parser.parse_args()
    config = EpisodeConfig()
    gym = HydraulicRigGym(config, default_facts_path(config))
    oracle = rollout(gym, OraclePolicy(gym.env), args.seed)
    do_nothing = rollout(gym, DoNothingPolicy(), args.seed)
    if oracle is None or do_nothing is None:
        print("The verifier could not run on at least one episode, so no reward was produced.")
    elif oracle > do_nothing:
        print("The oracle policy earned the higher reward and the do nothing policy sits below it, as RFC 008 asks.")
    elif oracle == do_nothing:
        print("Both policies earned the same reward, so the reward does not separate them.")
    else:
        print("The do nothing policy earned the higher reward, so the reward is wrong.")


if __name__ == "__main__":
    main()
