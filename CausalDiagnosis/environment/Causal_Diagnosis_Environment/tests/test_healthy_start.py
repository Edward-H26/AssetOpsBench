"""The start without a fault and its control preset: a fault rule may begin at zero faulty components, the preset
healthy_control plays the combination with every component at its healthy level at one entry offset per replicate, and the
verdict scores such a start by the rules of every start, with the work orders counted as unnecessary and the components that
a diagnosis names off their healthy level counted as false alarms. No text tells the control from another row. Every test
asserts a relation and no value of the data."""
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import REPLICATES, combination_folds, sampled_combination_seeds, start_state
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, reward, verify

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import events, finish, order, read

from CausalDiagnosis.utils.paths import CONFIGS, preset_path
SEED = 0


def preset(name, **fields):
    return compose({**json.loads(preset_path(name).read_text(encoding="utf-8")), **fields})


def healthy(config):
    return {c: config.healthy[c] for c in config.components}


def scored(env, seed=SEED):
    """The verdict of a finished episode and its reward, with the cost of the oracle policy's episode as the normalizer."""
    cost = oracle_cost(type(env)(list(env.cycles.values()), env.blocks, env.config), seed)
    verdict = verify(env.log, released_from_env(env), env.oracle(), cost, scope=env.config.verifier_scope)
    return verdict, reward(verdict, env.config.horizon)


@pytest.fixture
def control(world):
    cycles, blocks = world

    def build(engine=HydraulicRigEnv, seed=SEED):
        env = engine(cycles, blocks, preset("healthy_control"))
        env.reset(seed)
        return env
    return build


def test_the_control_preset_draws_the_healthy_combination_at_one_entry_offset_per_replicate(control):
    env = control()
    assert env.candidates == [tuple(healthy(env.config).values())]
    seeds = sampled_combination_seeds(env, replicates=REPLICATES)
    states = []
    for seed in seeds:
        env.reset(seed)
        states.append(start_state(env))
    assert len(seeds) == REPLICATES and len({offset for _combination, _window, offset in states}) == REPLICATES
    assert all(combination == env.candidates[0] for combination, _window, _offset in states)


@pytest.mark.parametrize("engine", (HydraulicRigEnv, RecordedRigEnv))
def test_on_a_healthy_start_an_agent_that_orders_nothing_passes(control, engine):
    env = control(engine)
    env.step({"type": "submit_diagnosis", "diagnosis": healthy(env.config), "evidence_ids": []})
    finish(env)
    verdict, earned = scored(env)
    assert verdict["verified"] and verdict["pass"] and verdict["false_alarms"] == 0 and verdict["unnecessary"] == 0
    assert verdict["failure_class"] == "none" and earned is not None


@pytest.mark.parametrize("orders", (1, 2))
def test_on_a_healthy_start_every_work_order_is_counted_and_fails_the_pass(control, orders):
    env = control()
    read(env)  # a reading before the diagnosis, so that the class names the orders
    env.step({"type": "submit_diagnosis", "diagnosis": healthy(env.config), "evidence_ids": []})
    for component in env.config.components[:orders]:
        order(env, component)
    finish(env)
    verdict, _earned = scored(env)
    assert verdict["unnecessary"] == orders and not verdict["pass"] and verdict["failure_class"] == "unnecessary_replacement"


@pytest.mark.parametrize("named", (1, 2))
def test_on_a_healthy_start_a_diagnosis_that_names_a_fault_is_a_false_alarm(control, named):
    env = control()
    diagnosis = healthy(env.config)
    for component in env.config.components[:named]:
        diagnosis[component] = next(level for level in env.config.levels[component] if level != env.config.healthy[component])
    env.step({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": []})
    finish(env)
    verdict, _earned = scored(env)
    assert verdict["false_alarms"] == named and not verdict["pass"]


def test_false_alarms_count_the_components_healthy_at_the_write_on_any_start(make_env):
    """On a start with one fault a healthy component named faulty counts once, a wrong level on the faulty component counts not
    at all, no diagnosis counts nothing, and after a drop before the write the dropped component is no false alarm."""
    def played(diagnosis_of, **fields):
        env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 1}, **fields)
        env.reset(SEED)
        if "schedule" in fields:
            while not events(env.log, "scheduled_drop"):
                env.step({"type": "wait_for_notification", "timeout": 1})
        diagnosis = diagnosis_of(env)
        if diagnosis is not None:
            env.step({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": []})
        finish(env)
        return verify(env.log, released_from_env(env), env.oracle(), None)["false_alarms"]

    def truth(env):
        return dict(zip(env.config.components, env.hidden))

    def other_level(env, component):
        return next(level for level in env.config.levels[component] if level not in (env.hidden[env.config.components.index(component)],
                                                                                       env.config.healthy[component]))

    def faulty(env):
        return next(c for c, level in truth(env).items() if level != env.config.healthy[c])

    def healthy_one(env):
        return next(c for c, level in truth(env).items() if level == env.config.healthy[c])

    assert played(lambda env: {**truth(env), healthy_one(env): other_level(env, healthy_one(env))}) == 1
    assert played(lambda env: {**truth(env), faulty(env): other_level(env, faulty(env))}) == 0
    assert played(lambda env: None) == 0
    schedule = {"mode": "seeded", "components": 1, "first_drop": [1, 1], "interval": 50}
    assert played(lambda env: truth(env), schedule=schedule) == 0  # the dropped component is named at its level at the write


def test_every_reference_policy_plays_the_control_to_a_verified_end(world, control):
    """The five reference policies play the replicate seeds of the control to a verified end, and the oracle policy passes
    there with no false alarm and no unnecessary order."""
    from CausalDiagnosis.experiments.episode_runners.run_reference_policies import POLICIES, play, policy_table

    cycles, blocks = world
    config = preset("healthy_control")
    table = policy_table(cycles, blocks)
    for seed in sampled_combination_seeds(control(), replicates=REPLICATES):
        for name in POLICIES:
            verdict, _earned = play(cycles, blocks, config, table[name], seed)
            assert verdict["verified"], (name, seed)
            if name == "oracle":
                assert verdict["pass"] and verdict["false_alarms"] == 0 and verdict["unnecessary"] == 0, seed


def test_the_briefing_of_the_control_is_the_briefing_of_the_base_preset_on_both_surfaces(world):
    """The fault rule of the starts reaches no text an agent reads, so at every guidance level the briefing of the control is
    the briefing of the base preset, on the rig's surface and on the AssetOpsBench rows."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing as rows_briefing
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing as rig_briefing

    cycles, blocks = world
    for level in (1, 2, 3):
        texts = {}
        for name in ("base", "healthy_control"):
            config = dataclasses.replace(preset(name), guidance_level=level)
            rig, rows = HydraulicRigEnv(cycles, blocks, config), RecordedRigEnv(cycles, blocks, config)
            rig.reset(SEED)
            rows.reset(SEED)
            texts[name] = (rig_briefing(rig.public_profile(), config), rows_briefing(rows.public_profile(), config))
        assert texts["base"] == texts["healthy_control"], level


def test_no_fold_holds_a_start_without_a_fault(control):
    with pytest.raises(ValueError, match="every start holds at least one fault"):
        combination_folds(control(), 2, 0)


def test_where_the_oracle_pays_nothing_the_normalized_cost_and_the_reward_are_undefined(world):
    """Under a cost in money the oracle policy pays nothing on the start without a fault, since it reads nothing and orders
    nothing, so the normalized cost and the reward of every episode of that start are undefined, while the pass stands."""
    cycles, blocks = world
    env = HydraulicRigEnv(cycles, blocks, preset("priced", fault_rule={"min_faulty": 0, "max_faulty": 0}))
    env.reset(SEED)
    read(env, next(iter(env.config.sensor_groups)), 1)
    env.step({"type": "submit_diagnosis", "diagnosis": healthy(env.config), "evidence_ids": []})
    finish(env)
    verdict, earned = scored(env)
    assert env.config.cost_unit() == "money" and verdict["cost"] > 0
    assert verdict["pass"] and verdict["cost_oracle"] == 0 and verdict["cost_normalized"] is None and earned is None
