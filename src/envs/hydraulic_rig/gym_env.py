"""Gymnasium wrapper of the hydraulic rig environment: text observations, dict or JSON text actions, and the
verifier's reward on the terminal step (docs/hydraulic_rig/environment.md, section Gym and OpenEnv).

The wrapper exists for policy learning. An episode is a function of the seed, the configuration and the action
sequence, it contains no language model call, and it is scored by verifier.py from the log and the hidden state,
never from the answer text. Observations are text so that a language model policy reads what an agent of the
action server reads: reset returns the briefing followed by the asset profile as JSON, and every step returns the
answer envelope of the action as JSON. The reward is 0 until the terminal step and then the verifier's reward; the
cost of each step is reported in info and never subtracted from the reward, so that the outcome term stays the
signal (design v2, section 7). A verifier that could not run gives no reward: the tuple then carries 0.0 with
truncated set and info["reward"] is None, so that a trainer can drop the episode instead of scoring it zero.

gymnasium is optional and imported lazily. With it installed the class is a gymnasium.Env with Text spaces; without
it the class is a duck typed stand in with the same reset and step methods, so the fork adds no dependency.
"""
import json
import string
from functools import lru_cache

from servers.hydraulic_rig.briefing import briefing

from .data import build_blocks, load_cycles
from .env import HydraulicRigEnv
from .policies import OraclePolicy
from .verifier import released_from_env, reward as terminal_reward, verify

try:
    import gymnasium
    from gymnasium.spaces import Text

    GYMNASIUM_AVAILABLE = True
    _EnvBase = gymnasium.Env
except ImportError:
    GYMNASIUM_AVAILABLE = False

    class _EnvBase:
        """Stand in for gymnasium.Env when gymnasium is not installed."""

        metadata = {"render_modes": []}


TEXT_MAX_LENGTH = 2**20  # a declared bound for the Text spaces, which describe the interface and are never sampled
TRUNCATION_REASONS = ("horizon", "max_steps")  # ends by a limit of the clock or of the step cap, not by the state of the rig


@lru_cache(maxsize=2)
def load_world(facts_path):
    """The recorded cycles and the stable blocks of one facts file, loaded once: the file is large and read only,
    and every wrapper instance and its oracle copy share it."""
    cycles = load_cycles(facts_path)
    return cycles, build_blocks(cycles)


def parse_action(action):
    """The environment's action dict for what a policy emitted. A dict passes as is and a JSON string of an object
    is decoded; anything else becomes an action of type malformed, which the environment refuses, logs and counts
    like any other bad action, so a policy that emits unparsable text loses a step and never raises into the
    trainer."""
    if isinstance(action, dict):
        return action
    if isinstance(action, str):
        try:
            decoded = json.loads(action)
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            return decoded
    return {"type": "malformed", "text": action if isinstance(action, str) else repr(action)}


def end_reason(log):
    """The reason of the end event, the last entry the environment writes."""
    return next(e["reason"] for e in reversed(log) if e["event"] == "end")


def oracle_cost(env, seed):
    """The cost of the oracle policy's episode from the same seed, played on a separate environment instance so
    that the oracle's actions never enter the agent's log. It is the denominator of the verifier's normalized cost
    (design v2, section 6), so the verifier's own cost rule is applied to the oracle's log."""
    observation = env.reset(seed)
    policy = OraclePolicy(env)
    done = False
    while not done:
        observation, _cost, done, _info = env.step(policy.act(observation))
    return verify(env.log, released_from_env(env), env.oracle(), cost_oracle=None)["cost"]


class HydraulicRigGym(_EnvBase):
    """The hydraulic rig episode as a Gymnasium environment with text observations.

    config is an EpisodeConfig, facts_path the recorded cycles, and seed_offset is added to every seed so that
    parallel instances fed the same seed sequence play disjoint episodes. reset(seed) returns the briefing and the
    asset profile as one text with an info dict; step(action) takes a dict or a JSON string of an action of the
    environment and returns the answer envelope as JSON, the reward, the terminated and truncated flags and info.
    terminated marks an end by the state of the episode (finish, budget, failure), truncated an end by the horizon
    or the step cap, or an end whose verifier could not run (reward 0.0 in the tuple, None in info). info carries
    the cost and the time of every step and, on the terminal step, the end reason, the verifier's verdict, its
    reward and the oracle's cost; info is for the trainer and never part of what the policy reads.

    The attribute observation keeps the answer dict the environment returned last, so that an in-process policy
    such as policies.OraclePolicy acts on the same episode as a text policy. The attribute env is the wrapped
    HydraulicRigEnv; its log and its oracle are hidden state for the verifier, never for the policy.
    """

    def __init__(self, config, facts_path, seed_offset=0):
        super().__init__()
        self.config = config
        self.facts_path = str(facts_path)
        self.seed_offset = seed_offset
        cycles, blocks = load_world(self.facts_path)
        self.env = HydraulicRigEnv(cycles, blocks, config)
        self._oracle_env = HydraulicRigEnv(cycles, blocks, config)
        self.episodes = 0
        self.seed = None
        self.steps = 0
        self.task_text = None
        self.cost_oracle = None
        self.observation = None
        if GYMNASIUM_AVAILABLE:
            self.observation_space = Text(TEXT_MAX_LENGTH, charset=string.printable)
            self.action_space = Text(TEXT_MAX_LENGTH, charset=string.printable)

    def reset(self, seed=None, options=None):
        """Start the episode of seed plus the seed offset. Without a seed the instance plays consecutive seeds
        from the offset, which is what a training loop that calls reset() expects. options is gymnasium's and
        unused."""
        if seed is None:
            seed = self.episodes
        self.seed = seed + self.seed_offset
        self.episodes += 1
        self.steps = 0
        self.cost_oracle = oracle_cost(self._oracle_env, self.seed)
        self.observation = self.env.reset(self.seed)
        profile = self.env.public_profile()
        self.task_text = briefing(profile, self.config)
        return self.task_text + "\n\n" + json.dumps(profile), {"seed": self.seed}

    def step(self, action):
        """Apply one action and return (observation_text, reward, terminated, truncated, info)."""
        self.observation, cost, done, _info = self.env.step(parse_action(action))
        self.steps += 1
        text = json.dumps(self.observation)
        info = {"cost": cost, "time": self.env.time, "steps": self.steps}
        if not done:
            return text, 0.0, False, False, info
        reason = end_reason(self.env.log)
        verdict = verify(
            self.env.log, released_from_env(self.env), self.env.oracle(), self.cost_oracle, grade=None,
            scope=self.config.verifier_scope,
        )
        reward = terminal_reward(verdict, self.config.horizon)
        info.update({"end_reason": reason, "verdict": verdict, "reward": reward, "cost_oracle": self.cost_oracle})
        # gymnasium needs a number in the tuple; a verifier that could not run gives 0.0 here, None in info and a
        # truncation rather than a terminal state, which keeps a missing verdict apart from a score of zero
        truncated = reason in TRUNCATION_REASONS or reward is None
        return text, 0.0 if reward is None else float(reward), not truncated, truncated, info
