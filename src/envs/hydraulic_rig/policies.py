"""Reference policies for v0: random and oracle (docs/hydraulic_rig/environment.md). Greedy lives in greedy.py with the per-tier classifiers.

RandomPolicy is a fixed mixture, not a uniform draw over legal actions: in the investigate phase it submits a random
diagnosis with probability 0.15, or whenever no tier is affordable, and otherwise measures an affordable tier; in the repair phase it draws r uniformly:
r < 0.4 replaces a random component if affordable (an unaffordable draw falls through), r < 0.7 measures an affordable
tier, and it finishes otherwise."""
import random

from .data import COMPONENTS, HEALTHY, LEVELS


class RandomPolicy:
    def __init__(self, seed):
        # A stream separate from the environment's: the same integer seeds env.reset, and two generators seeded
        # alike emit the same numbers, which tied this policy's first decision to the hidden combination.
        self.rng = random.Random(f"random-policy:{seed}")

    def act(self, obs):
        left = obs["budget_left"]
        tiers = [t for t, p in obs["prices"]["measure"].items() if p <= left]
        if obs["phase"] == "investigate":
            if not tiers or self.rng.random() < 0.15:
                return {"type": "submit", "diagnosis": {c: self.rng.choice(LEVELS[c]) for c in COMPONENTS}}
            return {"type": "measure", "tier": self.rng.choice(tiers)}
        r = self.rng.random()
        if r < 0.4:
            comp = self.rng.choice(COMPONENTS)
            if obs["prices"]["parts"][comp] + obs["prices"]["downtime"] <= left:
                return {"type": "replace", "component": comp}
        if r < 0.7 and tiers:
            return {"type": "measure", "tier": self.rng.choice(tiers)}
        return {"type": "finish"}


class OraclePolicy:
    """Reads the hidden state (env.initial, env.hidden), which no agent observes. It is the reference row for the upper
    bound on diagnosis and recovery and for the lowest cost of an episode that recovers; only the tests and the drivers
    run_policies.py, scenario_anchors.py and run_episodes.py (its in-process oracle row) under benchmarks/hydraulic_rig/
    construct it, never an agent harness."""

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        if obs["phase"] == "investigate":
            return {"type": "submit", "diagnosis": dict(zip(COMPONENTS, self.env.initial))}
        for i, c in enumerate(COMPONENTS):
            if self.env.hidden[i] != HEALTHY[c]:
                return {"type": "replace", "component": c}
        return {"type": "finish"}


def run_episode(env, policy, seed):
    """Run one seeded episode to its end; the environment's step cap guarantees termination, so an agent that never
    submits scores as unsubmitted instead of getting a submit written for it."""
    obs = env.reset(seed)
    while not env.done:
        obs, cost, done, info = env.step(policy.act(obs))
    return env.log
