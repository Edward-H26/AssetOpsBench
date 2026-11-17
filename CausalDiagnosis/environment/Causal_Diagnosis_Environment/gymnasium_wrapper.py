"""Gymnasium wrapper of the hydraulic rig environment: text observations, dict or JSON text actions, and the
verifier's reward on the terminal step (CausalDiagnosis/docs/environment.md, section Gym and OpenEnv).

The wrapper exists for policy learning. An episode is a function of the seed, the configuration and the action
sequence, it contains no language model call, and it is scored by verifier/ from the log and the hidden state,
never from the answer text. Observations are text so that a language model policy reads what an agent of the
action server reads: reset returns the briefing followed by the asset profile as JSON, and every step returns the
answer envelope of the action as JSON. The reward is 0 until the terminal step and then the verifier's reward; the
cost of each step is reported in info and never subtracted from the reward, so that the outcome term stays the
signal. A verifier that could not run gives no reward: the tuple then carries 0.0 with
truncated set and info["reward"] is None, so that a trainer can drop the episode instead of scoring it zero.

The wrapper plays the development starts of the asset by default (development_split.development_starts), so a trainer
never meets a start of the test set, and starts="test" plays the configuration's own start set, the test set of a preset.

gymnasium is optional and imported lazily. With it installed the class is a gymnasium.Env with Text spaces; without
it the class is a duck typed stand in with the same reset and step methods. The package group rl of pyproject.toml
installs it with openenv-core for the tests of both wrappers.
"""
import json
import string

from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
from CausalDiagnosis.datasets.splits.development_split import development_starts, on_starts
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, reward as terminal_reward, verify
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory import make_env

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
STARTS = ("development", "test")


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
    that the oracle's actions never enter the agent's log. It is the denominator of the verifier's normalized cost,
    so the verifier's own cost rule is applied to the oracle's log, in the unit of the row: its
    number of calls on the free base row, where every price is zero."""
    observation = env.reset(seed)
    policy = OraclePolicy(env)
    done = False
    while not done:
        observation, _cost, done, _info = env.step(policy.act(observation))
    return verify(env.log, released_from_env(env), env.oracle(), cost_oracle=None)["cost"]


def played_config(config, facts_path, starts):
    """The configuration a wrapper plays. starts="development": config narrowed to the development starts of its asset, within
    config.allowed when that is set, and a start fixed in config.start must be one of them. starts="test": config as it is."""
    if starts not in STARTS:
        raise ValueError(f"starts must be one of {', '.join(STARTS)}, not {starts!r}")
    if starts == "test":
        return config
    env = make_env(config, facts_path)
    try:
        development = development_starts(env.profile, set(env.blocks))
    except ValueError as e:  # development_split.SplitRefused
        raise ValueError(f"the asset {config.asset_id} has no development starts, so the wrapper needs starts='test': {e}") from None
    if config.allowed is not None:
        allowed = {tuple(c) for c in config.allowed}
        development = [c for c in development if c in allowed]
        if not development:
            raise ValueError("no development start lies in config.allowed")
    if config.start is not None and tuple(config.start["combination"]) not in development:
        raise ValueError(f"the start {list(config.start['combination'])} of config.start is no development start of the asset {config.asset_id}")
    return on_starts(config, development)


class HydraulicRigGym(_EnvBase):
    """The hydraulic rig episode as a Gymnasium environment with text observations.

    config is an EpisodeConfig, facts_path the recorded cycles, and seed_offset is added to every seed so that
    parallel instances fed the same seed sequence play disjoint episodes. reset(seed) returns the briefing and the
    asset profile as one text with an info dict; step(action) takes a dict or a JSON string of an action of the
    environment and returns the answer envelope as JSON, the reward, the terminated and truncated flags and info.
    terminated marks an end by the state of the episode (finish, budget, failure), truncated an end by the horizon
    or the step cap, or an end whose verifier could not run (reward 0.0 in the tuple, None in info). info carries
    the cost and the time of every step, the environment steps so far under steps, and, on the terminal step, the end reason, the verifier's verdict, its
    reward and the oracle's cost; info is for the trainer and never part of what the policy reads.

    The attribute observation keeps the answer dict the environment returned last, so that an in-process policy
    such as policies.OraclePolicy acts on the same episode as a text policy. The attribute env is the wrapped
    HydraulicRigEnv; its log and its oracle are hidden state for the verifier, never for the policy.

    label_of(env), called after reset, gives the separability of the start in the verifier's form, {"separable": bool, ...,
    "e18": {"separable": bool, ...}}, by default ports.start_label, the label that the package of the episode's asset
    declares: for hyd_1 start_grade of diagnosability_label.py, the independent label with the server's E18 grade, each table
    computed once per rules key, and None for an asset that declares none. The verifier's success, and with it the reward,
    credits a declaration of insufficient evidence that cites a reading on a start that both mark inseparable;
    with label_of None, or a label of None, every start counts as separable.

    starts is "development" (the default) or "test"; played_config gives the configuration the gym plays, which config then
    holds.
    """

    def __init__(self, config, facts_path, seed_offset=0, label_of=ports.start_label, starts="development"):
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES
        if config.clock_mode in THINKING_MODES:
            raise ValueError(f"the clock mode {config.clock_mode} charges thinking time, which an episode played in process does not "
                             "measure; play it through the driver's servers")
        super().__init__()
        self.facts_path = str(facts_path)
        self.starts = starts
        self.config = config = played_config(config, self.facts_path, starts)
        self.seed_offset = seed_offset
        self.label_of = label_of
        self.label = None
        self.env = make_env(config, self.facts_path)
        self._oracle_env = make_env(config, self.facts_path)
        self.episodes = 0
        self.seed = None
        self.steps = 0
        self.task_text = None
        self.cost_oracle = None
        self._oracle_costs = {}  # seed -> the cost of the oracle episode of the seed, which this gym plays once per seed
        self.observation = None
        if GYMNASIUM_AVAILABLE:
            self.observation_space = Text(TEXT_MAX_LENGTH, charset=string.printable)
            self.action_space = Text(TEXT_MAX_LENGTH, charset=string.printable)

    def reset(self, seed=None, options=None):
        """Start the episode of seed plus the seed offset. Without a seed the instance plays consecutive seeds
        from the offset, which is what a training loop that calls reset() expects. options is gymnasium's and
        unused."""
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)  # seeds np_random from the seed, as gymnasium's checker asks
        if seed is None:
            seed = self.episodes
        self.seed = seed + self.seed_offset
        self.episodes += 1
        self.steps = 0
        if self.seed not in self._oracle_costs:
            self._oracle_costs[self.seed] = oracle_cost(self._oracle_env, self.seed)
        self.cost_oracle = self._oracle_costs[self.seed]
        self.observation = self.env.reset(self.seed)
        self.label = self.label_of(self.env) if self.label_of else None
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
            self.env.log, released_from_env(self.env), self.env.oracle(), self.cost_oracle, grade=self.label,
            scope=self.config.verifier_scope,
        )
        reward = terminal_reward(verdict, self.config.horizon)
        info.update({"end_reason": reason, "verdict": verdict, "reward": reward, "cost_oracle": self.cost_oracle})
        # gymnasium needs a number in the tuple; a verifier that could not run gives 0.0 here, None in info and a
        # truncation rather than a terminal state, which keeps a missing verdict apart from a score of zero
        truncated = reason in TRUNCATION_REASONS or reward is None
        return text, 0.0 if reward is None else float(reward), not truncated, truncated, info
