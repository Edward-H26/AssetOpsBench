"""One rollout of the oracle policy and one of the do nothing policy through the Gymnasium wrapper: the oracle and
baseline check of OpenEnv's RFC 008, which asks that the oracle earn more than doing nothing, since otherwise the
reward does not measure the task. The script prints which policy earned the higher reward and no value: a number
from a run is a result, and results come from the drivers under benchmarks/hydraulic_rig/ only.

    export HYD_FACTS_PATH=/path/to/hyd_facts.jsonl
    PATH=$PWD/.venv/bin:$PATH python examples/hydraulic_rig/rollout.py --seed 0
"""
import argparse
import sys
from pathlib import Path

# the fork is not installed as a package; the tests and the drivers put src on the path the same way
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.hydraulic_rig.config import EpisodeConfig  # noqa: E402
from envs.hydraulic_rig.data import default_facts_path  # noqa: E402
from envs.hydraulic_rig.gym_env import HydraulicRigGym  # noqa: E402
from envs.hydraulic_rig.policies import DoNothingPolicy, OraclePolicy  # noqa: E402


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
    gym = HydraulicRigGym(EpisodeConfig(), default_facts_path())
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
