"""Two scripted policies on the presets with visits and an emergency card or drops, which read the hidden state as the
oracle policy does and are no agent. Each first reads one cycle of the cheapest sensor group, so that its diagnosis follows a
reading and its failure class names what it did after. EmergencyThenFinish submits the true diagnosis, orders an emergency
repair of every faulty component in priority order and finishes. WaitThenOneVisit submits the true diagnosis, waits until the
last drop that the schedule or the hazard table brings before the first close of the oracle policy, creates one visit for
every component faulty by then, waits for its closes and finishes. The first is the path that a hold in force at the end must not pass, the
second a passing path beside the oracle policy, which is no reference policy, so no test sets it against the oracle. The
seeds are those of the reference policies, the evaluation pool and the replicates of every start. Every test asserts a
relation and no value of the data."""
from CausalDiagnosis.utils.paths import CONFIGS, preset_path
import copy
import json
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import EVALUATION_POOL, REPLICATES, combination_seeds, sampled_combination_seeds
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, reward, verify

PRESETS = CONFIGS
ROWS = ("repair_rules", "deterioration", "scheduled")  # the presets with visits and an emergency card or drops
DROPS = ("scheduled_drop", "hazard_drop")


def _faulty(env):
    """The components faulty now, smaller priority number first."""
    config = env.config
    faulty = [c for c, level in zip(config.components, env.hidden) if level != config.healthy[c]]
    return sorted(faulty, key=lambda c: (config.priority[c], config.components.index(c)))


def _first_read(env):
    """One cycle of the cheapest sensor group, by price then by name."""
    prices = env.config.prices["read"]
    return {"type": "read_sensors", "sensor_group": min(prices, key=lambda g: (prices[g], g)), "cycles": 1}


def _diagnosis(env):
    return {"type": "submit_diagnosis", "diagnosis": dict(zip(env.config.components, env.hidden)), "evidence_ids": []}


def _order(component, mode):
    return {"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": [], "mode": mode}


def _drops_after_the_first_order(env):
    """Whether a level drops at or after the cycle of the first work order of the episode."""
    created = [e["t_end"] for e in env.log if e["event"] == "work_order_created"]
    return bool(created) and any(e["event"] in DROPS and e["t_end"] >= min(created) for e in env.log)


def _wait_for_closes(env):
    """A wait until the last open order closes, or None when no order is open."""
    open_orders = env.clock.open_orders()
    if not open_orders:
        return None
    return {"type": "wait_for_notification", "timeout": max(1, max(o.close_t for o in open_orders) - env.time)}


class EmergencyThenFinish:
    """One read, the true diagnosis, an emergency order on every faulty component in priority order, the finish."""

    def __init__(self, env):
        self.env = env

    def start(self, obs):
        self.read, self.queue = False, None

    def act(self, obs):
        if not self.read:
            self.read = True
            return _first_read(self.env)
        if self.queue is None:
            self.queue = _faulty(self.env)
            return _diagnosis(self.env)
        if self.queue:
            return _order(self.queue.pop(0), "emergency")
        return {"type": "finish", "summary": "Every faulty component got an emergency repair."}


class WaitThenOneVisit:
    """One read, the true diagnosis, a wait until the last drop that the schedule or the hazard table brings before the first close of
    the oracle policy, one visit for every component faulty by then, a wait for its closes, the finish. The drops before that
    close are those of an episode in which nothing has closed yet, so a copy of the episode that the oracle policy plays
    shows them, as the oracle's own foresight of the drops would."""

    def __init__(self, env):
        self.env = env

    def start(self, obs):
        twin = copy.deepcopy(self.env, memo={id(self.env.cycles): self.env.cycles, id(self.env.blocks): self.env.blocks})
        oracle = OraclePolicy(twin)
        oracle.start(obs)
        while not twin.done:
            obs = twin.step(oracle.act(obs))[0]
        closes = [e["t_end"] for e in twin.log if e["event"] == "work_order_closed" and e.get("source") == "agent"]
        first_close = min(closes, default=self.env.config.horizon)
        drops = [e["t_end"] for e in twin.log if e["event"] in DROPS and e["t_end"] < first_close]
        self.ready_t = max(drops) + 1 if drops else 0
        self.phase = "read"
        self.queue = []

    def act(self, obs):
        env = self.env
        if self.phase == "read":
            self.phase = "diagnose"
            return _first_read(env)
        if self.phase == "diagnose":
            self.phase = "wait_drops"
            return _diagnosis(env)
        if self.phase == "wait_drops":
            if env.time < self.ready_t:
                return {"type": "wait_for_notification", "timeout": self.ready_t - env.time}
            self.phase, self.queue = "order", _faulty(env)
        if self.phase == "order" and self.queue:
            return _order(self.queue.pop(0), "standard")
        self.phase = "closes"
        wait = _wait_for_closes(env)
        if wait is not None:
            return wait
        return {"type": "finish", "summary": "One visit repaired every faulty component."}


@pytest.fixture(scope="module")
def played(world):
    """Per row and seed the verdict and the reward of the oracle policy and of the two scripted policies, each with the
    cost of the oracle policy's episode as the normalizer, as the drivers score it."""
    cycles, blocks = world
    out = {}
    for row in ROWS:
        config = compose(json.loads(preset_path(row).read_text(encoding="utf-8")))
        probe = HydraulicRigEnv(cycles, blocks, config)
        seeds = sorted(set(combination_seeds(probe, EVALUATION_POOL)) | set(sampled_combination_seeds(probe, replicates=REPLICATES)))
        for seed in seeds:
            cost = oracle_cost(HydraulicRigEnv(cycles, blocks, config), seed)
            for name, make in (("oracle", OraclePolicy), ("emergency", EmergencyThenFinish), ("one_visit", WaitThenOneVisit)):
                env = HydraulicRigEnv(cycles, blocks, config)
                env.reset(seed)
                log = run_episode(env, make(env), seed)
                verdict = verify(log, released_from_env(env), env.oracle(), cost, scope=config.verifier_scope)
                out[(row, seed, name)] = (verdict, reward(verdict, config.horizon), env)
    return out


def test_the_emergency_path_does_not_pass_and_earns_less_than_the_oracle(played):
    """An emergency repair holds its component only for its hold, so the path of emergency orders and the finish never passes.
    The path ends at its finish, or at the budget when the budget cannot pay every emergency order. Wherever it reaches its
    finish its class is temporary_repair_expired, whatever level drops later, since each of its holds has expired or still holds
    at the end. On every seed the path earns less than the oracle policy."""
    seen = 0
    for (row, seed, name), (verdict, earned, env) in played.items():
        if name != "emergency":
            continue
        seen += 1
        assert verdict["verified"] and not verdict["pass"], (row, seed)
        assert verdict["end_reason"] in ("finish", "budget"), (row, seed, verdict["end_reason"])
        if verdict["end_reason"] == "finish":
            assert verdict["failure_class"] == "temporary_repair_expired", (row, seed, verdict["failure_class"])
        assert earned < played[(row, seed, "oracle")][1], (row, seed)
    assert seen


def test_the_one_visit_path_is_verified_and_passes_wherever_no_level_drops_after_its_first_order(played):
    """WaitThenOneVisit names the true levels and orders one visit for every component faulty once the drops before the first
    close of the oracle policy have come. Every episode of it is verified, and where every order closes before the horizon and
    no level drops at or after the cycle of its first order, the visit restores what the diagnosis named, so it passes."""
    seen = 0
    for (row, seed, name), (verdict, _earned, env) in played.items():
        if name != "one_visit":
            continue
        seen += 1
        assert verdict["verified"], (row, seed)
        created = any(e["event"] == "work_order_created" for e in env.log)
        closed = all(order.status == "closed" for order in env.clock.orders)
        if created and closed and not _drops_after_the_first_order(env):
            assert verdict["pass"], (row, seed, verdict["failure_class"])
    assert seen
