"""The degradation hazard (CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_5_deterioration_and_failure.py): the stochastic mode replays from the seed with the same
actions and meets other drops under other actions, the oracle passes in it, and the collateral damage of the replacement
of a healthy component follows its probability. No test states a value computed from the rig data."""
import json

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import hazard_rate
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import schedule_visit
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.environment.reference_policies.policies import DoNothingPolicy, OraclePolicy, run_episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import _asset_truth, _charged_ineffective, released_from_env, verify

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, order, submit_truth

STOCHASTIC = {"mode": "stochastic", "base_rate": 0.01, "depth_factor": 1.0, "collateral": 0.0}  # the rates of the asset file
PUMP_ONLY = (100, 100, 1, 130)


def hazard(make_env, base_rate=0.01, **overrides):
    return make_env(hazard={**STOCHASTIC, "base_rate": base_rate}, **overrides)


def drops(log):
    return [(e["t_start"], e["component"], e["to_level"], e["cause"]) for e in events(log, "hazard_drop")]


def test_the_same_seed_and_the_same_actions_replay_the_stochastic_episode_exactly(make_env):
    env, scheduled = hazard(make_env, base_rate=0.05), make_env()
    seen = 0
    for seed in range(5):
        first = json.dumps(run_episode(env, DoNothingPolicy(), seed))
        assert json.dumps(run_episode(env, DoNothingPolicy(), seed)) == first
        seen += len(drops(json.loads(first)))
        assert all(env.public_event(e) is None for e in events(env.log, "hazard_drop"))  # the hazard is hidden
        scheduled.reset(seed)
        env.reset(seed)
        # the hazard draws from a generator of its own, so the start is the one of the scheduled mode
        assert (env.initial, env.window_start, env.entry_offset) == (scheduled.initial, scheduled.window_start, scheduled.entry_offset)
    assert seen


def test_another_action_sequence_meets_other_drops_on_a_seed_constructed_from_the_hazard_table(make_env):
    """With the pump one level below healthy, waiting keeps its drop rate at two base rates while a repair brings it back
    to one. The seed is the first one whose table lets no dependency of the pump drop before the repair closes, lets the
    pump itself not drop before then, and holds after the close a first uniform below the waiting rate that lies above
    the repaired rate: the waiting run must drop the pump in that cycle and the repairing run must not."""
    env = hazard(make_env, base_rate=0.05, start={"combination": list(PUMP_ONLY), "entry_offset": 0})
    config = env.config
    pump = COMPONENTS.index("pump")
    close_t = config.durations["lead_time"]["pump"] + config.durations["repair_downtime"]["pump"]  # alone in its visit, ordered at 0
    repaired_rate, waiting_rate = hazard_rate(config, "pump", HEALTHY["pump"]), hazard_rate(config, "pump", PUMP_ONLY[pump])
    for seed in range(1000):
        env.reset(seed)
        table = env.clock.hazard["drop"]
        dependencies_hold = all(table[t][COMPONENTS.index(d)] >= hazard_rate(config, d, HEALTHY[d])
                                for t in range(close_t) for d in config.depends_on("pump"))
        later = [t for t in range(close_t, config.horizon) if table[t][pump] < waiting_rate]
        if dependencies_hold and all(table[t][pump] >= waiting_rate for t in range(close_t)) and later and table[later[0]][pump] >= repaired_rate:
            break
    else:
        raise AssertionError("no seed below 1000 separates the two runs")
    drop_t = later[0]
    env.reset(seed)
    advance(env, config.horizon)
    waiting = drops(env.log)
    env.reset(seed)
    submit_truth(env)
    order(env, "pump")
    advance(env, config.horizon)
    repairing = drops(env.log)
    assert env.log[-1]["final"][pump] == HEALTHY["pump"] or any(c == "pump" for _t, c, _l, _cause in repairing)
    assert (drop_t, "pump") in [(t, c) for t, c, _l, _cause in waiting]
    assert (drop_t, "pump") not in [(t, c) for t, c, _l, _cause in repairing]


def test_the_oracle_passes_in_the_stochastic_mode(make_env):
    env = hazard(make_env)
    seen = 0
    for seed in range(30):
        log = run_episode(env, OraclePolicy(env), seed)
        v = verify(log, released_from_env(env), env.oracle(), None)
        assert log[-1]["reason"] == "finish" and v["pass"] and v["success"] and v["failure_class"] == "none", seed
        seen += len(drops(log))
    assert seen  # some episodes drop a component the oracle then repairs again


def test_an_order_the_hazard_made_ineffective_is_not_charged_and_the_oracle_orders_after_that_drop(make_env):
    """With the pump one level below healthy and its dependencies healthy, an order on the pump at cycle 0 closes ineffective when
    the hazard drops a dependency before the close, which the agent could not foresee, so the verifier does not charge the class
    repair_before_dependency for it. The oracle reads that drop from the table of uniforms drawn at reset: it orders the pump
    only after the drop and passes with no ineffective order. The seed is the first one whose table drops a
    dependency of the pump before the close of an order created at cycle 0."""
    env = hazard(make_env, base_rate=0.05, start={"combination": list(PUMP_ONLY), "entry_offset": 0})
    config = env.config
    close_t = config.durations["lead_time"]["pump"] + config.durations["repair_downtime"]["pump"]  # alone in its visit, ordered at 0
    dependencies = [COMPONENTS.index(d) for d in config.depends_on("pump")]
    for seed in range(1000):
        env.reset(seed)
        table = env.clock.hazard["drop"]
        dropping = [t for t in range(close_t) for i in dependencies if table[t][i] < hazard_rate(config, COMPONENTS[i], HEALTHY[COMPONENTS[i]])]
        if dropping:
            break
    else:
        raise AssertionError("no seed below 1000 drops a dependency of the pump before the close")
    env.reset(seed)
    submit_truth(env)
    order(env, "pump")
    advance(env, close_t)
    assert events(env.log, "order_ineffective") and not _charged_ineffective(env.log, _asset_truth(env.log))
    log = run_episode(env, OraclePolicy(env), seed)
    v = verify(log, released_from_env(env), env.oracle(), None)
    assert v["pass"] and not events(log, "order_ineffective"), seed
    assert all(e["t_start"] > dropping[0] for e in events(log, "work_order_created") if e["component"] == "pump"), seed


def test_the_oracle_places_a_visit_a_cycle_later_when_the_table_would_drop_a_repaired_component_before_the_visit_ends(make_env):
    """With the cooler and the accumulator degraded, the visit of cycle 0 closes the accumulator before the cooler. On a seed
    whose table of uniforms drops the repaired accumulator before the cooler's close, the oracle places the visit a cycle later,
    so that the repair closes past the drop, and orders the accumulator once. The seed is the first one whose
    table drops the repaired accumulator there and drops no healthy component before the visit could end."""
    env = hazard(make_env, start={"combination": [20, 100, 0, 90], "entry_offset": 0})
    config = env.config
    acc = COMPONENTS.index("accumulator")
    members = [("cooler", "standard"), ("accumulator", "standard")]
    plan = {members[i][0]: close for i, _start, close in schedule_visit(config, 0, members)}
    assert plan["accumulator"] < plan["cooler"]  # the accumulator comes first in the visit
    healthy = [COMPONENTS.index(c) for c in COMPONENTS if c not in ("cooler", "accumulator")]
    for seed in range(5000):
        env.reset(seed)
        table = env.clock.hazard["drop"]
        relapse = any(table[u][acc] < hazard_rate(config, "accumulator", HEALTHY["accumulator"]) for u in range(plan["accumulator"], plan["cooler"]))
        quiet = all(table[u][k] >= hazard_rate(config, COMPONENTS[k], HEALTHY[COMPONENTS[k]]) for u in range(plan["cooler"] + 3) for k in healthy)
        if relapse and quiet:
            break
    else:
        raise AssertionError("no seed below 5000 drops the repaired accumulator before the cooler's close")
    log = run_episode(env, OraclePolicy(env), seed)
    v = verify(log, released_from_env(env), env.oracle(), None)
    orders = events(log, "work_order_created")
    assert v["pass"] and not events(log, "failure"), seed
    assert min(e["t_start"] for e in orders) >= 1 and [e["component"] for e in orders].count("accumulator") == 1, seed


def test_collateral_zero_means_no_drop_after_replacing_a_healthy_component_and_one_means_a_drop(make_env):
    pump = COMPONENTS.index("pump")
    for collateral in (0.0, 1.0):
        env = make_env(start={"combination": [3, 90, 0, 130], "entry_offset": 0}, hazard={**STOCHASTIC, "mode": "scheduled", "collateral": collateral})
        env.reset(0)
        submit_truth(env)
        order(env, "pump")  # the pump is healthy, so the replacement only risks damage
        advance(env, 10)
        close_t = events(env.log, "work_order_closed")[-1]["t_start"]
        damage = [e for e in events(env.log, "hazard_drop") if e["cause"] == "collateral"]
        if collateral == 0.0:
            assert not damage and env.hidden[pump] == HEALTHY["pump"]
        else:
            assert [(e["t_start"], e["component"], e["to_level"]) for e in damage] == [(close_t, "pump", env.hidden[pump])]
            assert env.hidden[pump] != HEALTHY["pump"] and env.public_event(damage[0]) is None
