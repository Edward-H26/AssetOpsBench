"""The graded fields of the verdict and the exact match of the faulty set, stated for any asset.
graded_top1 is the share of the components faulty at the cycle of the first diagnosis that it names at their true level,
and 0 when it puts a healthy component at another level or when there is no diagnosis. graded_pass is the share of the
components faulty at any time of the episode that are healthy at its end, a component faulty at the write counting only
when the diagnosis also named it at its true level, and 0 when the diagnosis holds a false alarm or any work order was
unnecessary. With no faulty component the two fields equal Top-1 and pass, so on a start with at most one fault and no
drop both equal Top-1 and pass, which the tests pin over every case below. graded_top1_n and graded_pass_n are the
denominators, so a reader recovers each share as an exact fraction. faulty_set_right compares the set of components off
their healthy level and ignores the levels. The last tests play the environment: a diagnosis is scored against the state
at the cycle it is submitted in every clock mode."""
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import events, finish
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.test_verifier import (COMPONENTS, COOLER_FAULTY, HEALTHY, HEALTHY_LIST, HOT, INSIDE, LEAD_TIME, REPAIR_DOWNTIME, TEMPERATURE, TWO_FAULTY,
                            Episode, do_nothing, oracle_like, verdict_of)

CITED = [f"obs/1/{TEMPERATURE}"]
DROP_CYCLE = 2  # the cycle of the one scheduled drop of the constructed episode


def _episode(hidden, diagnosis, repaired, extra_orders=()):
    ep = Episode(hidden)
    ep.read(HOT if hidden[0] != HEALTHY["cooler"] else INSIDE)
    if diagnosis is not None:
        ep.diagnose(diagnosis, CITED)
    orders = [ep.order(c, evidence_ids=CITED) for c in (*repaired, *extra_orders)]
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    for entry in orders:
        ep.close(entry["id"], entry["component"])
    ep.finish("done")
    ep.end()
    return ep


def _truth(hidden):
    return dict(zip(COMPONENTS, hidden))


def _one_fault_cases():
    truth = _truth(COOLER_FAULTY)
    wrong_level = {**truth, "cooler": 20}
    false_alarm = {**truth, "valve": 73}
    return [oracle_like(), do_nothing(), _episode(COOLER_FAULTY, truth, ["cooler"]), _episode(COOLER_FAULTY, wrong_level, ["cooler"]),
            _episode(COOLER_FAULTY, false_alarm, ["cooler"]), _episode(COOLER_FAULTY, truth, []),
            _episode(COOLER_FAULTY, truth, ["cooler"], extra_orders=["pump"]), _episode(COOLER_FAULTY, None, ["cooler"])]


def _healthy_cases():
    healthy = dict(HEALTHY)
    return {"named": _episode(HEALTHY_LIST, healthy, []), "silent": _episode(HEALTHY_LIST, None, []),
            "alarmed": _episode(HEALTHY_LIST, {**healthy, "valve": 73}, []), "ordered": _episode(HEALTHY_LIST, healthy, [], extra_orders=["pump"])}


def test_on_one_fault_without_a_drop_the_graded_fields_equal_top1_and_pass():
    for ep in _one_fault_cases():
        v = verdict_of(ep)
        assert v["graded_top1"] == float(v["diagnosis_all_correct"]) and v["graded_pass"] == float(v["pass"]), v["failure_class"]


def test_on_two_faults_the_graded_fields_give_half_credit():
    truth = _truth(TWO_FAULTY)
    one_right = {**truth, "accumulator": HEALTHY["accumulator"]}
    v = verdict_of(_episode(TWO_FAULTY, one_right, ["cooler"]))
    assert v["graded_top1"] == 0.5 and v["graded_pass"] == 0.5 and not v["diagnosis_all_correct"] and not v["pass"]
    v = verdict_of(_episode(TWO_FAULTY, truth, ["cooler"]))  # both named, one repaired
    assert v["graded_top1"] == 1.0 and v["graded_pass"] == 0.5
    v = verdict_of(_episode(TWO_FAULTY, truth, ["cooler", "accumulator"]))
    assert v["graded_top1"] == v["graded_pass"] == 1.0 and v["pass"]


def test_a_false_alarm_or_an_unnecessary_order_zeroes_the_graded_fields():
    truth = _truth(TWO_FAULTY)
    v = verdict_of(_episode(TWO_FAULTY, {**truth, "pump": 1}, ["cooler", "accumulator"]))
    assert v["graded_top1"] == 0.0 and v["graded_pass"] == 0.0
    v = verdict_of(_episode(TWO_FAULTY, truth, ["cooler", "accumulator"], extra_orders=["pump"]))
    assert v["graded_top1"] == 1.0 and v["graded_pass"] == 0.0


def test_a_fault_that_appears_during_the_episode_counts_in_graded_pass_by_its_end_state():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(_truth(COOLER_FAULTY), CITED)
    ep.drop("pump", 1)
    orders = [ep.order("cooler", evidence_ids=CITED)]
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close(orders[0]["id"], "cooler")
    ep.finish("done")
    ep.end()
    v = verdict_of(ep, oracle=ep.oracle(work_orders=[("cooler", 120)]))
    assert v["graded_top1"] == 1.0 and v["graded_pass"] == 0.5 and not v["pass"]


def test_the_fields_are_exact_shares_of_the_faulty_count():
    v = verdict_of(_episode(TWO_FAULTY, _truth(TWO_FAULTY), ["cooler"]))
    assert (v["graded_top1_n"], v["graded_pass_n"]) == (2, 2) and v["graded_pass"] * v["graded_pass_n"] == 1
    assert v["graded_top1"] * v["graded_top1_n"] == 2


def test_without_a_faulty_component_the_graded_fields_equal_top1_and_pass():
    """An empty share is no success: with no faulty component at the write graded_top1 is Top-1, and with none at any
    time graded_pass is pass, so an agent that submits nothing on a healthy asset earns neither."""
    verdicts = {name: verdict_of(ep) for name, ep in _healthy_cases().items()}
    assert all((v["graded_top1_n"], v["graded_pass_n"]) == (0, 0) for v in verdicts.values())
    assert all(v["graded_top1"] == float(v["diagnosis_all_correct"]) and v["graded_pass"] == float(v["pass"]) for v in verdicts.values())
    assert verdicts["named"]["pass"] and (verdicts["named"]["graded_top1"], verdicts["named"]["graded_pass"]) == (1.0, 1.0)
    assert (verdicts["silent"]["graded_top1"], verdicts["silent"]["graded_pass"]) == (0.0, 0.0)
    assert (verdicts["alarmed"]["graded_top1"], verdicts["alarmed"]["graded_pass"]) == (0.0, 0.0)
    assert verdicts["ordered"]["graded_top1"] == 1.0 and verdicts["ordered"]["graded_pass"] == 0.0 and verdicts["ordered"]["unnecessary"] == 1


def test_faulty_set_right_ignores_levels_and_reads_the_truth_at_the_write():
    truth = _truth(COOLER_FAULTY)
    wrong_level = verdict_of(_episode(COOLER_FAULTY, {**truth, "cooler": 20}, ["cooler"]))
    assert wrong_level["faulty_set_right"] is True and wrong_level["diagnosis_all_correct"] is False
    assert wrong_level["checks"]["hard"]["faulty_set_right"] is True
    superset = verdict_of(_episode(COOLER_FAULTY, {**truth, "valve": 73}, ["cooler"]))
    subset = verdict_of(_episode(TWO_FAULTY, {**_truth(TWO_FAULTY), "accumulator": HEALTHY["accumulator"]}, ["cooler"]))
    silent = verdict_of(_episode(COOLER_FAULTY, None, []))
    assert [v["faulty_set_right"] for v in (superset, subset, silent)] == [False, False, False]
    # after a drop before the write the set of that cycle counts: the cooler and the pump are faulty then
    oracle_orders = [("cooler", 120), ("pump", 120)]
    late = Episode(COOLER_FAULTY)
    late.read(HOT)
    late.drop("pump", 1)
    late.diagnose({**truth, "pump": 2}, CITED)  # the set of the write, with a wrong level of the pump
    late.finish("done")
    late.end()
    v = verdict_of(late, oracle=late.oracle(work_orders=oracle_orders))
    assert v["faulty_set_right"] is True and v["diagnosis_all_correct"] is False
    stale = Episode(COOLER_FAULTY)
    stale.read(HOT)
    stale.drop("pump", 1)
    stale.diagnose(truth, CITED)  # the set of the start, which the drop left behind
    stale.finish("done")
    stale.end()
    assert verdict_of(stale, oracle=stale.oracle(work_orders=oracle_orders))["faulty_set_right"] is False


def test_a_right_diagnosis_has_the_right_faulty_set():
    verdicts = [verdict_of(ep) for ep in (*_one_fault_cases(), *_healthy_cases().values())]
    assert any(v["diagnosis_all_correct"] for v in verdicts) and any(not v["diagnosis_all_correct"] for v in verdicts)
    assert all(v["faulty_set_right"] for v in verdicts if v["diagnosis_all_correct"])
    assert all(isinstance(v["faulty_set_right"], bool) for v in verdicts)


def _write_at_the_drop(world, mode):
    """An episode whose schedule drops one component at DROP_CYCLE, with the levels of that cycle submitted at that cycle."""
    cycles, blocks = world
    schedule = {**EpisodeConfig().schedule, "mode": "seeded", "components": 1, "first_drop": [DROP_CYCLE, DROP_CYCLE]}
    env = HydraulicRigEnv(cycles, blocks, EpisodeConfig(clock_mode=mode, schedule=schedule))
    env.reset(0)
    while env.time < DROP_CYCLE:
        env.step({"type": "wait_for_notification", "timeout": 1})
    before = dict(zip(COMPONENTS, env.hidden))
    env.step({"type": "submit_diagnosis", "diagnosis": before, "evidence_ids": []})
    finish(env)
    return env, before


def test_a_drop_inside_the_cycle_of_the_write_does_not_enter_its_truth(world):
    """In the instant mode the submission takes a cycle, the cycle of the drop here. The entry of the diagnosis stands
    before the drop in the log, so the diagnosis is scored against the levels it was submitted under."""
    env, before = _write_at_the_drop(world, "instant")
    names = [e["event"] for e in env.log]
    entry, drop = events(env.log, "submit_diagnosis")[0], events(env.log, "scheduled_drop")[0]
    assert names.index("submit_diagnosis") < names.index("scheduled_drop")
    assert (entry["t_start"], entry["t_end"], drop["t_start"]) == (DROP_CYCLE, DROP_CYCLE + 1, DROP_CYCLE)
    verdict = verify(env.log, released_from_env(env), env.oracle(), None)
    assert verdict["truth_at_write"] == before and verdict["diagnosis_all_correct"] and verdict["write_t"] == DROP_CYCLE
    # a submission takes no cycle in the declared mode, where the entry starts and ends at the cycle of the write
    env, before = _write_at_the_drop(world, "declared")
    entry = events(env.log, "submit_diagnosis")[0]
    verdict = verify(env.log, released_from_env(env), env.oracle(), None)
    assert entry["t_start"] == entry["t_end"] == DROP_CYCLE == verdict["write_t"] and verdict["truth_at_write"] == before


def test_a_write_is_scored_at_its_submission_cycle_and_the_oracle_wins_in_the_instant_mode(world):
    """In the instant mode a submit takes a cycle, and a drop in that cycle no longer counts against the
    write. Deterioration seed 44 drops the valve in the oracle's write cycle, which failed the oracle before."""
    import dataclasses
    import sys
    from pathlib import Path

    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
    from CausalDiagnosis.experiments.episode_runners.run_reference_policies import play, policy_table

    cycles, blocks = world
    config = dataclasses.replace(load_config("deterioration")[1], clock_mode="instant")
    verdict, _reward = play(cycles, blocks, config, policy_table(cycles, blocks)["oracle"], 44)
    assert verdict["pass"] and verdict["diagnosis_all_correct"] and verdict["failure_class"] == "none"
    for mode in ("declared", "generation_time"):
        verdict, _reward = play(cycles, blocks, dataclasses.replace(config, clock_mode=mode), policy_table(cycles, blocks)["oracle"], 44)
        assert verdict["pass"]
