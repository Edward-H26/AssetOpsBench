"""CausalDiagnosis/evaluation/metrics/mcp_metrics.py, the measures of the CausalDiagnosis MCP (CausalDiagnosis/docs/metrics.md): each measure
on hand-made episode folders whose values are made up here and checked by hand, the transcript of each folder written by the
writer of the servers (servers/transcript.py), the cells of a row of the table MCP, and episodes played through the real servers
in process, on which the measures agree with the log. No test states a value computed from the rig data."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from CausalDiagnosis.evaluation.metrics import mcp_metrics
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, RIG_SERVER
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, default_facts_path
from CausalDiagnosis.mcp_servers.utils import transcript

START = [100, 80, 0, 90]  # the valve and the accumulator off their healthy levels
HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]
PRICED = {"read": {"tier1": 1, "tier2": 3}, "parts": {"valve": 20, "accumulator": 20}, "misdiagnosis": 25}
FREE = {"read": {"tier1": 0}, "parts": {"valve": 0}}  # a row that prices nothing: its cost unit is the call
ADVICE = ("visits", "advised", "follow_first", "follow_any", "harm", "deviate", "good_deviation", "first_visit_follows")
needs_servers = pytest.mark.skipif(not Path(default_facts_path()).exists() or importlib.util.find_spec("mcp") is None,
                                   reason="needs the rig facts file and the mcp package")


def combo(levels):
    return dict(zip(COMPONENTS, levels))


def rank(**answer):
    """A call of the diagnostics server that answered a ranking: it takes no step, and its line keeps the text."""
    return (CAUSAL_SERVER, "rank_root_causes", {}, json.dumps(answer), False, False)


def act(tool, stepped=True, server=RIG_SERVER, is_error=False, **args):
    """A call of an action server, which took a step of the environment unless stepped is false."""
    return (server, tool, args, json.dumps({"result": {}}), is_error, stepped)


def branch(plan, pid, levels=START, act_now=0, price=0, downtime=0):
    return {"combination": combo(levels), "cost_to_restored": {"act_now": act_now},
            "repair_plan": {"steps": [{"component": c} for c in plan], "plan_prediction_id": pid, "price": price, "downtime_cost": downtime}}


def reset(start=START, prices=None, config=None):
    return {"event": "reset", "step": None, "t_start": 0, "t_end": 0, "cost": 0, "hidden": list(start),
            "config": config if config is not None else ({"prices": prices} if prices else {})}


def entry(event, t, step=None, **fields):
    return {"event": event, "step": step, "t_start": t, "t_end": t, "cost": 0, **fields}


def folder(tmp_path, log, calls, name="ep", released=None, predictions=None):
    """An episode folder: its hidden log, released evidence and predictions, and its transcript written by the servers' own
    writer, line 0 the start and line k the k-th call. calls None writes no transcript."""
    ep = tmp_path / name
    (ep / "hidden").mkdir(parents=True)
    for rel, value in (("hidden/log.json", log), ("released.json", released or {}), ("predictions.json", predictions or [])):
        (ep / rel).write_text(json.dumps(value), encoding="utf-8")
    if calls is not None:
        transcript.begin(ep, wall=0.0)
        for server, tool, arguments, text, is_error, stepped in calls:
            transcript.answered(ep, server, tool, arguments, text, is_error, stepped=stepped, keep_text=server == CAUSAL_SERVER)
    return ep


def measures(ep):
    return mcp_metrics.episode_measures(ep, None, "joint", {})


def base_config():
    return load_config("base")[1].to_json()  # a row without a clock, whose orders close inside their own call


# ----- uptake and harm of the advice -----

def test_a_visit_follows_the_first_plan_of_the_last_answer_before_it_and_harms_only_when_that_plan_is_not_the_faulty_set_at_the_visit(tmp_path):
    """The first visit, two orders one after another, orders the plan of the first branch, the valve and the accumulator, which is
    the faulty set of the start, so it follows without harm. A wait ends that run. The second visit orders the cooler, no plan of
    the answer before it, while every part is healthy after the closes, so it deviates and the deviation is not right. The order
    the engine refused took a step and joins no visit."""
    calls = [rank(branches=[branch(["valve", "accumulator"], "pred/b1"), branch(["valve"], "pred/b2")]),
             act("create_work_order", component="valve", prediction_id="pred/b1"),
             act("create_work_order", component="accumulator", prediction_id="pred/b1"),
             act("wait_for_notification", timeout=6),
             rank(branches=[branch(["pump"], "pred/b3")]),
             act("create_work_order", component="cooler"),
             act("create_work_order", component="pump")]
    log = [reset(), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1, prediction_id="pred/b1"),
           entry("work_order_created", 0, 2, id="wo-2", component="accumulator", visit=1, prediction_id="pred/b1"),
           entry("work_order_closed", 4, id="wo-1", component="valve", source="agent", combination_after=[100, 100, 0, 90]),
           entry("work_order_closed", 5, id="wo-2", component="accumulator", source="agent", combination_after=HEALTHY_LIST),
           entry("wait", 6, 3), entry("work_order_created", 6, 4, id="wo-3", component="cooler", visit=2),
           entry("refused", 6, 5, reason="the budget is spent")]
    out = measures(folder(tmp_path, log, calls))
    assert out["unaligned"] is False
    assert [out[k] for k in ADVICE] == [2, 2, 1, 1, 0, 1, 0, True]
    assert out["plans"] == []  # no stored prediction carries pred/b1


def test_a_followed_first_plan_that_misses_the_fault_is_harm_and_a_deviation_to_the_faulty_set_is_right(tmp_path):
    """The pump alone is faulty. The first branch repairs the valve: the first visit follows it, which is harm. The second visit,
    under the same answer, orders the pump, the plan of the second branch and the faulty set, a right deviation that still follows a
    plan of the answer."""
    start = [100, 100, 1, 130]
    calls = [rank(branches=[branch(["valve"], "pred/b1", start), branch(["pump"], "pred/b2", start)]),
             act("create_work_order", component="valve"), act("wait_for_notification", timeout=4), act("create_work_order", component="pump")]
    log = [reset(start), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1),
           entry("work_order_closed", 3, id="wo-1", component="valve", source="agent", combination_after=start),
           entry("wait", 4, 2), entry("work_order_created", 4, 3, id="wo-2", component="pump", visit=2)]
    out = measures(folder(tmp_path, log, calls))
    assert [out[k] for k in ADVICE] == [2, 2, 1, 2, 1, 1, 1, True]


def test_orders_created_one_after_another_form_one_visit_whatever_visits_the_environment_books(tmp_path):
    """On a row whose orders close in their own call every order books a visit of its own, so a plan of two components is one run
    of two orders and one visit here, which follows the first plan. A call of the MCP between the two orders leaves the run whole."""
    calls = [rank(branches=[branch(["valve", "accumulator"], "pred/b1")]), act("create_work_order", component="valve"),
             rank(branches=[branch(["accumulator"], "pred/b2")]), act("create_work_order", component="accumulator")]
    log = [reset(), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1),
           entry("work_order_closed", 0, id="wo-1", component="valve", source="agent", combination_after=[100, 100, 0, 90]),
           entry("work_order_created", 0, 2, id="wo-2", component="accumulator", visit=2),
           entry("work_order_closed", 0, id="wo-2", component="accumulator", source="agent", combination_after=HEALTHY_LIST)]
    out = measures(folder(tmp_path, log, calls))
    assert [out[k] for k in ADVICE] == [1, 1, 1, 1, 0, 0, 0, True]


def test_a_refused_order_between_two_orders_keeps_the_visit_whole_and_the_episode_aligned(tmp_path):
    calls = [rank(branches=[branch(["valve", "accumulator"], "pred/b1")]), act("create_work_order", component="valve"),
             act("create_work_order", component="pump"), act("create_work_order", component="accumulator")]
    log = [reset(), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1),
           entry("refused", 0, 2, reason="the budget is spent"),
           entry("work_order_created", 0, 3, id="wo-2", component="accumulator", visit=1)]
    ep = folder(tmp_path, log, calls)
    call_list, aligned = mcp_metrics.calls(ep, log)
    made = mcp_metrics.decisions(call_list, log)
    assert aligned and [(d.kind, d.n, d.index) for d in made] == [("order", 2, 1), ("order", 4, 3)]
    [visit] = mcp_metrics.visits(made, call_list, log)
    assert (visit.components, visit.first, visit.orders) == (frozenset({"valve", "accumulator"}), 1, ("wo-1", "wo-2"))
    out = measures(ep)
    assert out["unaligned"] is False and [out[k] for k in ADVICE] == [1, 1, 1, 1, 0, 0, 0, True]


def test_a_no_call_step_between_two_orders_keeps_one_visit_as_the_environment_books_it(tmp_path):
    """A turn that calls no tool of the rig takes a step of the clock and no call of the environment, so the run of orders stays
    open across it (visits.keeps_run) and the two orders form one visit of both components."""
    calls = [rank(branches=[branch(["valve", "accumulator"], "pred/b1")]), act("create_work_order", component="valve"),
             act("no_call"), act("create_work_order", component="accumulator")]
    log = [reset(), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1),
           entry("no_call", 0, 2),
           entry("work_order_created", 0, 3, id="wo-2", component="accumulator", visit=1)]
    ep = folder(tmp_path, log, calls)
    call_list, aligned = mcp_metrics.calls(ep, log)
    made = mcp_metrics.decisions(call_list, log)
    assert aligned and [(d.kind, d.n) for d in made] == [("order", 2), ("order", 4)]
    [visit] = mcp_metrics.visits(made, call_list, log)
    assert visit.components == frozenset({"valve", "accumulator"}) and visit.orders == ("wo-1", "wo-2")


def test_a_call_that_took_no_step_is_no_decision_and_no_break(tmp_path):
    """On a row without a clock a call of a tool the row does not list and a refused call after the end take no step: they stand
    between two orders of one plan and split nothing, and the answer after them is placed after the closes of both orders."""
    calls = [rank(branches=[branch(["valve", "accumulator"], "pred/b1")]), act("create_work_order", component="valve"),
             act("no_such_tool", stepped=False, is_error=True), act("get_current_time", stepped=False),
             act("create_work_order", component="accumulator"), rank(candidates=[{"combination": combo(HEALTHY_LIST), "probability": 1.0}])]
    log = [reset(config=base_config()), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1),
           entry("work_order_closed", 0, id="wo-1", component="valve", source="agent", combination_after=[100, 100, 0, 90]),
           entry("work_order_created", 0, 2, id="wo-2", component="accumulator", visit=2),
           entry("work_order_closed", 0, id="wo-2", component="accumulator", source="agent", combination_after=HEALTHY_LIST)]
    ep = folder(tmp_path, log, calls)
    call_list, aligned = mcp_metrics.calls(ep, log)
    assert aligned and [call.index for call in call_list] == [None, 1, None, None, 3, None]
    assert [d.kind for d in mcp_metrics.decisions(call_list, log)] == ["order", "order"]
    out = measures(ep)
    assert [out[k] for k in ADVICE] == [1, 1, 1, 1, 0, 0, 0, True] and out["unplaced"] == 0
    assert mcp_metrics.answer_truths(call_list, log, mcp_metrics.asset_truth(log)) == {1: combo(START), 6: combo(HEALTHY_LIST)}


def test_an_episode_whose_transcript_and_log_disagree_is_unaligned_and_measures_nothing_that_needs_the_match(tmp_path):
    """Two stepped calls against three steps, a step with two entries, and a folder whose log holds steps and no transcript."""
    answer = rank(belief_time=0, candidates=[{"combination": combo(START), "probability": 1.0}],
                  branches=[branch(["valve", "accumulator"], "pred/b1")])
    orders = [entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1, prediction_id="pred/b1"),
              entry("work_order_created", 0, 2, id="wo-2", component="accumulator", visit=1, prediction_id="pred/b1")]
    cases = {"short": ([answer, act("create_work_order", component="valve"), act("create_work_order", component="accumulator")],
                       [reset(), *orders, entry("read", 1, 3, sensor_group="tier1")], 1),
             "twice": ([answer, act("create_work_order", component="valve"), act("create_work_order", component="accumulator"), act("read_sensors")],
                       [reset(), *orders, entry("read", 1, 2, sensor_group="tier1")], 1),
             "none": (None, [reset(), *orders], 0)}
    for name, (calls, log, answers) in cases.items():
        ep = folder(tmp_path, log, calls, name, predictions=[{"id": "pred/b1", "components": ["valve", "accumulator"], "recovery": 0.5}])
        out = measures(ep)
        assert out["unaligned"] is True, name
        assert "advised" not in out and "plans" not in out and "reads_read_first" not in out and "diagnosis_follows" not in out, name
        assert out["unplaced"] == answers and out["ranking"] == [], name
        assert all(call.index is None for call in mcp_metrics.calls(ep, log)[0]), name


def test_the_measures_read_no_file_that_a_runner_writes(tmp_path):
    calls = [rank(branches=[branch(["valve", "accumulator"], "pred/b1")]), act("create_work_order", component="valve"),
             act("create_work_order", component="accumulator")]
    log = [reset(), entry("work_order_created", 0, 1, id="wo-1", component="valve", visit=1),
           entry("work_order_created", 0, 2, id="wo-2", component="accumulator", visit=1)]
    ep = folder(tmp_path, log, calls)
    before = measures(ep)
    # a runner's record that holds other calls, other answers and an error for every call
    contradiction = [{"server": RIG_SERVER, "tool": "read_sensors", "tool_args": {}, "response": "{}", "error": "lost"}] * 5
    (ep / "trajectory.json").write_text(json.dumps(contradiction), encoding="utf-8")
    assert measures(ep) == before and before["advised"] == 1


# ----- uptake of the advised reads and of the first candidate -----

def test_a_read_under_the_read_first_flag_counts_and_follows_the_advice_when_it_reads_the_separating_test_or_the_head_of_the_batch(tmp_path):
    """Three reads come under the flag; the group of the first is the separating test and that of the third the head of the
    read batch, so two of three follow. An answer that carries no flag reads as no flag, so the read after it counts nothing,
    nor does the last read, which comes after an answer whose flag is false."""
    flag = mcp_metrics.READ_FIRST
    calls = [rank(**{flag: True, "separating_test": {"sensor_group": "tier2"}, "read_batch": [{"sensor_group": "tier3"}]}),
             act("read_sensors", sensor_group="tier2"), act("read_sensors", sensor_group="tier1"), act("read_sensors", sensor_group="tier3"),
             rank(separating_test={"sensor_group": "tier1"}), act("read_sensors", sensor_group="tier1"),
             rank(**{flag: False, "separating_test": {"sensor_group": "tier2"}}), act("read_sensors", sensor_group="tier2")]
    log = [reset(), *(entry("read", n, n, sensor_group=g, cost=1) for n, g in enumerate(("tier2", "tier1", "tier3", "tier1", "tier2"), 1))]
    out = measures(folder(tmp_path, log, calls))
    assert (out["reads_read_first"], out["reads_advised"]) == (3, 2)
    # a read of cycles paid before costs nothing on a row that prices in money, so it is no decision to read
    log[2] = {**log[2], "cost": 0}
    out = measures(folder(tmp_path, log, calls, "free"))
    assert (out["reads_read_first"], out["reads_advised"]) == (2, 2)


def test_a_diagnosis_follows_when_it_writes_the_first_candidate_of_the_last_answer_before_it(tmp_path):
    for name, written, follows in (("same", START, True), ("other", [100, 73, 0, 90], False)):
        calls = [rank(candidates=[{"combination": combo(START), "probability": 0.6}, {"combination": combo([100, 73, 0, 90]), "probability": 0.4}]),
                 act("submit_diagnosis", **combo(written))]
        log = [reset(), entry("submit_diagnosis", 0, 1, diagnosis=combo(written))]
        assert measures(folder(tmp_path, log, calls, name))["diagnosis_follows"] is follows


# ----- the truth at the moment of each answer, for the ranking and the table A12 -----

DROPPED = [100, 80, 1, 90]


def test_on_a_row_with_a_clock_an_answer_is_scored_against_the_combination_before_the_cycle_moves_on(tmp_path):
    """The clock stands at the `belief_time` of an answer until the next action moves it on, so the drop of the pump in cycle 3,
    logged before the wait that ran that cycle, counts for the answers after the wait and not for those at cycle 1. An answer
    that names no cycle is placed after the calls that took a step before it. At cycle 1 the first candidate is right. At cycle 5
    it misses the pump: AC@1 is 0, the true faulty set stands second, its reciprocal rank is one half, and the first candidate's
    two faulty parts are faulty and cover two of the three. A healthy first candidate has no precision and recalls nothing."""
    log = [reset(), entry("read", 1, 1, sensor_group="tier1"), entry("hazard_drop", 3, component="pump", to_level=1),
           {**entry("wait", 5, 2), "t_start": 1}]
    both = [{"combination": combo(START), "probability": 0.7}, {"combination": combo(DROPPED), "probability": 0.3}]
    calls = [act("read_sensors", sensor_group="tier1"), rank(belief_time=1, candidates=both),
             rank(belief_time=1, candidates=[{"combination": combo(HEALTHY_LIST), "probability": 0.9}]), act("wait_for_notification", timeout=4),
             rank(belief_time=5, candidates=both), rank(candidates=both)]
    ep = folder(tmp_path, log, calls)
    asset = mcp_metrics.asset_truth(log)
    call_list, aligned = mcp_metrics.calls(ep, log)
    truths = mcp_metrics.answer_truths(call_list, log, asset)
    assert aligned and truths == {2: combo(START), 3: combo(START), 5: combo(DROPPED), 6: combo(DROPPED)}
    answers = [(call.n, call.answer) for call in call_list if call.answer is not None]
    late = {"ac1": 0.0, "ack": 1.0, "rr": 0.5, "precision": 1.0, "recall": 2 / 3}
    assert mcp_metrics.ranking_measures(answers, truths, asset) == [
        {"ac1": 1.0, "ack": 1.0, "rr": 1.0, "precision": 1.0, "recall": 1.0},
        {"ac1": 0.0, "ack": 0.0, "rr": 0.0, "precision": None, "recall": 0.0}, late, late]
    # the table A12 reads the same answers: the accepted reads before each, its first candidate, its probability, its truth
    assert mcp_metrics.ranked_answers(ep) == [(1, START, 0.7, START), (1, HEALTHY_LIST, 0.9, START), (1, START, 0.7, DROPPED),
                                              (1, START, 0.7, DROPPED)]


def test_on_a_row_without_a_clock_an_answer_counts_the_repairs_of_the_orders_before_it(tmp_path):
    """On the base row every order closes inside its own call, so the answer after the diagnosis sees the start, the answer after
    the order on the valve sees the valve repaired, and the answer after both orders and a read sees a healthy asset, although all
    of them share cycle 0."""
    log = [reset(config=base_config()), entry("submit_diagnosis", 0, 1, diagnosis=combo(START)), entry("work_order_created", 0, 2, component="valve"),
           entry("work_order_closed", 0, component="valve", source="agent", combination_after=[100, 100, 0, 90]),
           entry("work_order_created", 0, 3, component="accumulator"),
           entry("work_order_closed", 0, component="accumulator", source="agent", combination_after=HEALTHY_LIST),
           entry("read", 0, 4, sensor_group="tier1")]
    calls = [act("submit_diagnosis", **combo(START)), rank(belief_time=0, candidates=[{"combination": combo(START), "probability": 1.0}]),
             act("create_work_order", component="valve"), rank(belief_time=0, candidates=[{"combination": combo(START), "probability": 1.0}]),
             act("create_work_order", component="accumulator"), act("read_sensors", sensor_group="tier1"),
             rank(belief_time=0, candidates=[{"combination": combo(HEALTHY_LIST), "probability": 1.0}])]
    ep = folder(tmp_path, log, calls)
    call_list, aligned = mcp_metrics.calls(ep, log)
    assert aligned and mcp_metrics.answer_truths(call_list, log, mcp_metrics.asset_truth(log)) == {
        2: combo(START), 4: combo([100, 100, 0, 90]), 7: combo(HEALTHY_LIST)}
    assert [readings for readings, _top, _p, _truth in mcp_metrics.ranked_answers(ep)] == [0, 0, 1]


# ----- production lost while thinking -----

def test_the_production_lost_while_thinking_sums_the_charges_of_the_cycles_charged_before_each_action():
    """The first read thought two cycles before it started at cycle 3, cycles 1 and 2, the second one cycle before cycle 6, cycle 5.
    Each cycle lost 5 of the 35 the episode lost, and the failure at cycle 2 fell in a thinking cycle."""
    log = [reset(), {"event": "read", "step": 1, "t_start": 3, "t_end": 4, "think_cycles": 2, "cost": 1},
           {"event": "failure", "step": None, "t_start": 2, "t_end": 2, "cost": 50},
           {"event": "read", "step": 2, "t_start": 6, "t_end": 7, "think_cycles": 1, "cost": 1},
           {"event": "read", "step": 3, "t_start": 7, "t_end": 8, "think_cycles": 0, "cost": 1},
           {"event": "end", "step": None, "t_start": 8, "t_end": 8, "downtime_total": 35, "downtime_by_cycle": [{"t": t, "charge": 5} for t in range(7)]}]
    assert mcp_metrics.thinking_loss(log) == {"thinking_cycles": 3, "think_loss": 15.0, "downtime_total": 35.0, "think_loss_share": 15 / 35,
                                              "think_failures": 1}


# ----- the predictions of a plan acted on -----

def _priced_episode(closes=True, after=()):
    """A diagnosis of the start, one visit at cycle 2 that orders the valve and the accumulator under pred/b1 at 30 and 25, the
    closes at cycles 4 and 5, which restore the asset, a read at cycle 6 at 3, and 5 lost in each of cycles 0 to 4. The end
    entry's own cost is the total and stays out of the sum. after holds entries logged after the restore."""
    log = [reset(prices=PRICED), entry("read", 0, 1, sensor_group="tier1", cost=1), entry("submit_diagnosis", 2, 2, diagnosis=combo(START)),
           entry("work_order_created", 2, 3, id="wo-1", component="valve", visit=1, prediction_id="pred/b1", cost=30),
           entry("work_order_created", 2, 4, id="wo-2", component="accumulator", visit=1, prediction_id="pred/b1", cost=25),
           entry("work_order_closed", 4, id="wo-1", component="valve", source="agent", combination_after=[100, 100, 0, 90]),
           *([entry("work_order_closed", 5, id="wo-2", component="accumulator", source="agent", combination_after=HEALTHY_LIST)] if closes else []),
           *after, entry("read", 6, 5, sensor_group="tier2", cost=3),
           {"event": "end", "step": None, "t_start": 9, "t_end": 9, "cost": 999, "downtime_total": 25,
            "downtime_by_cycle": [{"t": t, "charge": 5} for t in range(5)] + [{"t": t, "charge": 7} for t in range(5, 9) if after]}]
    released = {"rules": {"prices": PRICED}, "observations": [{"n": 1, "t": 0, "values": {"F1": 7.0, "F2": 5.5}},
                                                                {"n": 2, "t": 6, "values": {"F1": 11.0, "F2": 6.5}}]}
    predictions = [{"id": "pred/b1", "components": ["valve", "accumulator"], "recovery": 0.8, "observations": 1,
                    "predicted": {"F1": [10.0, 8.0, 12.0, 2.0], "F2": [5.0, 4.0, 6.0, 2.0]}}]
    # two branches share the plan, and the one whose combination is the diagnosis is the branch acted on
    answer = {"branches": [branch(["valve", "accumulator"], "pred/b1", [100, 73, 0, 90], act_now=500, price=60, downtime=20),
                           branch(["valve", "accumulator"], "pred/b1", START, act_now=100, price=60, downtime=20)]}
    visit = mcp_metrics.Visit(2, frozenset({"valve", "accumulator"}), ("pred/b1",), 3, answer, ("wo-1", "wo-2"))
    return log, released, predictions, visit


VERIFIED = {"status": "verified", "observation": 2, "observed_t": 6, "consistent": True, "predicted_combination": combo(HEALTHY_LIST)}


def test_a_verified_plan_scores_recovery_coverage_readings_and_the_cost_to_the_restore_against_the_oracle():
    """Recovery: every part is healthy at the verifying reading of cycle 6, so the target is 1. Coverage: F1 at 11 lies in [8, 12],
    F2 at 6.5 outside [4, 6]. Readings: sigma is (high - mean) / quantile, 1 and 0.5, the MCP misses by 1 and 3, holding the reading
    the plan saw misses by 4 and 2. Cost: the window runs from the first order to the close of cycle 5 that restores the asset, so
    30 + 25 paid in it and 15 lost in cycles 2 to 4 make 70, and the read of cycle 6 lies outside. Acting now at 100 errs by 30
    and the price with its downtime at 80 by 10, each over the oracle's 50."""
    log, released, predictions, visit = _priced_episode()
    asset = mcp_metrics.asset_truth(log)
    assert mcp_metrics.restored_at(log, visit, asset) == 6 and mcp_metrics.cost_to_restore(log, visit, asset) == 70.0
    plans = mcp_metrics.prediction_measures([visit], log, released, predictions, {"pred/b1": VERIFIED}, {"diagnosis_all_correct": True, "cost_oracle": 50}, asset)
    assert plans == [{"id": "pred/b1", "components": ["valve", "accumulator"], "recovery": 0.8, "target": 1.0, "verdict_consistent": True,
                      "predicted_right": True, "covered": 1, "summaries": 2, "error_mcp": 4.0, "error_hold": 6.0, "errors": 2,
                      "cost_error": pytest.approx(30 / 50), "price_error": pytest.approx(10 / 50)}]


def test_the_cost_of_a_right_plan_does_not_move_with_a_drop_after_the_restore():
    """The same episode with a drop of the pump after the restore, its repair at 40 and 7 lost in each later cycle: the window of
    the plan closes at the same entry, so the cost error and the price error stay as they were."""
    later = (entry("hazard_drop", 6, component="pump", to_level=1),
             entry("work_order_created", 6, 6, id="wo-3", component="pump", visit=2, cost=40),
             entry("work_order_closed", 8, id="wo-3", component="pump", source="agent", combination_after=HEALTHY_LIST))
    errors = []
    for after in ((), later):
        log, released, predictions, visit = _priced_episode(after=after)
        asset = mcp_metrics.asset_truth(log)
        assert mcp_metrics.restored_at(log, visit, asset) == 6
        [plan] = mcp_metrics.prediction_measures([visit], log, released, predictions, {}, {"diagnosis_all_correct": True, "cost_oracle": 50}, asset)
        errors.append((plan["cost_error"], plan["price_error"]))
    assert errors[0] == errors[1] == (pytest.approx(30 / 50), pytest.approx(10 / 50))


def test_the_cost_of_a_plan_that_leaves_a_fault_runs_to_the_repair_that_restores_the_asset_and_else_to_the_end():
    """A plan that repairs the valve alone leaves the accumulator faulty: its window runs to the close of a later order that
    repairs the accumulator, and without that order to the end of the episode. A plan that orders the valve on a healthy asset
    closes its window at the close of its own order, not at its first entry."""
    def log_of(start, *rest, charges=10):
        return [reset(start, prices=PRICED), *rest,
                {"event": "end", "step": None, "t_start": 12, "t_end": 12, "cost": 999, "downtime_by_cycle": [{"t": t, "charge": 2} for t in range(charges)]}]

    first = entry("work_order_created", 2, 1, id="wo-1", component="valve", visit=1, cost=30)
    valve = entry("work_order_closed", 4, id="wo-1", component="valve", source="agent", combination_after=[100, 100, 0, 90])
    second = entry("work_order_created", 6, 2, id="wo-2", component="accumulator", visit=2, cost=25)
    accumulator = entry("work_order_closed", 8, id="wo-2", component="accumulator", source="agent", combination_after=HEALTHY_LIST)
    visit = mcp_metrics.Visit(2, frozenset({"valve"}), (), 1, None, ("wo-1",))
    log = log_of(START, first, valve, entry("read", 5, 3, sensor_group="tier1", cost=1), second, accumulator)
    asset = mcp_metrics.asset_truth(log)
    # 30 + 1 + 25 paid up to the close of cycle 8, and 2 lost in each of cycles 2 to 7
    assert mcp_metrics.restored_at(log, visit, asset) == 5 and mcp_metrics.cost_to_restore(log, visit, asset) == 56.0 + 12.0
    log = log_of(START, first, valve, entry("read", 5, 2, sensor_group="tier1", cost=1))
    # never restored: every entry from the first order on, the end entry aside, and every charged cycle from cycle 2
    assert mcp_metrics.restored_at(log, visit, asset) is None and mcp_metrics.cost_to_restore(log, visit, asset) == 31.0 + 16.0
    on_healthy = entry("work_order_closed", 4, id="wo-1", component="valve", source="agent", combination_after=HEALTHY_LIST)
    log = log_of(HEALTHY_LIST, first, on_healthy, entry("read", 5, 2, sensor_group="tier1", cost=1))
    assert mcp_metrics.restored_at(log, visit, asset) == 2 and mcp_metrics.cost_to_restore(log, visit, asset) == 30.0 + 4.0


def test_a_wrong_diagnosis_adds_its_price_to_the_cost_and_a_refuted_right_prediction_is_a_false_alarm():
    log, released, predictions, visit = _priced_episode()
    asset = mcp_metrics.asset_truth(log)
    refuted = {**VERIFIED, "consistent": False}
    [plan] = mcp_metrics.prediction_measures([visit], log, released, predictions, {"pred/b1": refuted}, {"diagnosis_all_correct": False, "cost_oracle": 50}, asset)
    assert (plan["verdict_consistent"], plan["predicted_right"]) == (False, True)
    assert (plan["cost_error"], plan["price_error"]) == (pytest.approx((100 - 95) / 50), pytest.approx((80 - 95) / 50))


def test_without_a_verdict_the_target_is_the_first_cycle_after_the_last_close_and_an_open_order_leaves_it_empty():
    log, released, predictions, visit = _priced_episode()
    asset = mcp_metrics.asset_truth(log)
    [plan] = mcp_metrics.prediction_measures([visit], log, released, predictions, {}, {"cost_oracle": 50}, asset)
    assert plan["target"] == 1.0 and "verdict_consistent" not in plan and "covered" not in plan
    log, released, predictions, visit = _priced_episode(closes=False)
    [plan] = mcp_metrics.prediction_measures([visit], log, released, predictions, {}, {"cost_oracle": 50}, asset)
    assert plan["target"] is None


def test_on_a_row_that_prices_nothing_a_plan_has_no_cost_error_and_the_row_no_cost_cell():
    start = [100, 80, 0, 130]
    log = [reset(start, prices=FREE), entry("read", 0, 1),
           entry("work_order_created", 1, 2, id="wo-1", component="valve", visit=1, prediction_id="pred/v"),
           entry("work_order_closed", 3, id="wo-1", component="valve", source="agent", combination_after=HEALTHY_LIST), entry("read", 4, 3)]
    asset = mcp_metrics.asset_truth(log)
    answer = {"branches": [branch(["valve"], "pred/v", start, act_now=3, price=0, downtime=0)]}
    visit = mcp_metrics.Visit(1, frozenset({"valve"}), ("pred/v",), 2, answer, ("wo-1",))
    predictions = [{"id": "pred/v", "components": ["valve"], "recovery": 0.9}]
    assert mcp_metrics.cost_unit_of(log, {}) == "calls"
    [plan] = mcp_metrics.prediction_measures([visit], log, {"rules": {"prices": FREE}}, predictions, {}, {"cost_oracle": 2}, asset)
    assert plan["target"] == 1.0 and "cost_error" not in plan and "price_error" not in plan
    cells = mcp_metrics.row_cells([{"block": "[0]", "pass": True, "mcp": {"plans": [plan]}}], resamples=20, seed=0)
    assert [(cells[k], cells[f"{k}_n"]) for k in ("cost_bias", "cost_mae", "price_mae", "cost_skill")] == [("", 0)] * 4
    assert cells["recovery_bias_n"] == 1


# ----- the cells of a row of the table MCP -----

def _episode(agent, server, seed, think, mcp=None, passed=True):
    return {"agent": agent, "model": "m", "config": "deterioration", "clock_mode": "advanced_real_time", "surface": "rig", "guidance_level": 2,
            "verifier_scope": "writes", "prices": "default", "ranking_model": "joint", "abstention_rule": "rule", "training_variant": "as_released",
            "diagnostics_server": server, "seed": seed, "scenario_id": "", "block": json.dumps([seed]), "pass": passed, "think_loss": think,
            "mcp": {"think_loss": think, **(mcp or {})}}


ON = [_episode("causal_m", True, 0, 3.0, {"advised": 2, "follow_first": 1, "follow_any": 2, "harm": 1, "deviate": 1, "good_deviation": 1,
                                          "first_visit_follows": True, "downtime_total": 10.0,
                                          "plans": [{"recovery": 0.8, "target": 1.0, "cost_error": 0.5, "price_error": 1.0}]}),
      _episode("causal_m", True, 1, 5.0, {"advised": 1, "follow_first": 1, "follow_any": 1, "harm": 0, "deviate": 0, "good_deviation": 0,
                                          "first_visit_follows": True, "downtime_total": 10.0,
                                          "plans": [{"recovery": 0.4, "target": 0.0, "cost_error": -0.5, "price_error": -1.0}]}, passed=False)]


def test_the_cells_of_a_row_pool_the_items_of_its_episodes_before_the_ratio():
    """Uptake is 2 of 3 visits, harm 1 of 2 followed first plans, the right deviations 1 of 1, the pass after a following first visit
    1 of 2, and nothing deviated on a first visit. The Brier skill of 0.8 and 0.4 against 1 and 0 is 1 - 0.1 / 0.25, the bias the
    mean of -0.2 and 0.4, the absolute cost error 0.5 against 1 for the price alone, and thinking lost 8 of 20."""
    cells = mcp_metrics.row_cells(ON, resamples=50, seed=0)
    expected = {"uptake": 2 / 3, "uptake_any": 1.0, "harm": 0.5, "good_deviation": 1.0, "pass_follow": 0.5, "recovery_bss": 0.6, "recovery_bias": 0.1,
                "cost_bias": 0.0, "cost_mae": 0.5, "price_mae": 1.0, "cost_skill": 0.5, "think_loss_share": 0.4}
    assert {k: cells[k] for k in expected} == {k: pytest.approx(v, abs=1e-6) for k, v in expected.items()}
    # the count beside a share is its denominator, the 3 advised visits behind the uptake, and the points of the other statistics
    assert (cells["pass_deviate"], cells["pass_deviate_n"], cells["uptake_n"], cells["recovery_bss_n"], cells["unaligned"]) == ("", 0, 3, 2, 0)
    assert all(cells[f"{k}_low"] <= cells[k] <= cells[f"{k}_high"] and cells[f"{k}_se"] >= 0 for k in expected)


OFF = [_episode("causal_m", False, 0, 1.0), _episode("causal_m", False, 1, 2.0)]


def test_the_table_holds_the_agents_with_the_mcp_and_pairs_the_thinking_loss_against_the_agent_without_it():
    from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate

    rows = aggregate.table_mcp(ON + OFF, {"resamples": 50, "seed": 0})
    assert [(row["agent"], row["diagnostics_server"]) for row in rows] == [("causal_m", True)]
    [with_mcp] = rows
    assert (with_mcp["n"], with_mcp["think_loss"], with_mcp["think_loss_diff"], with_mcp["pairs"]) == (2, 4.0, 2.5, 2)
    assert "think_loss_p" not in with_mcp and set(aggregate.with_standard_errors(aggregate.MCP_COLUMNS)) >= set(with_mcp)


# ----- the appendix table of paper_tables/ -----

def _mcp_row(agent, preset, row_number, **cells):
    from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate

    config = aggregate.preset_config(preset)
    return {"agent": agent, "model": "m", "config": preset, "diagnostics_server": "True", "clock_mode": aggregate.config_clock_mode(preset), "surface": "rig",
            **{k: str(v) for k, v in aggregate.MAIN_LEVELS.items()}, **{k: str(getattr(config, k)) for k in aggregate.PRESET_LEVELS},
            "row_number": row_number, **cells}


def test_the_appendix_table_prints_the_cells_of_mcp_with_intervals_and_no_value_where_the_row_prices_nothing_or_charges_no_thinking():
    """Two groups of the three agents with the MCP. On the preset of all six layers every cell and the thinking difference print
    with their intervals. On the preset of the first experiment the row prices nothing, so its cost cells are empty and print
    the word for no value, and its clock charges no thinking, so the thinking column prints the same word."""
    from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables

    estimate = {f"{field}{part}": value for _, field in main_tables.MCP_CELLS for part, value in (("", 0.5), ("_low", 0.25), ("_high", 0.75))}
    priced_nothing = {k: ("" if k.startswith("cost_") else v) for k, v in estimate.items()}
    rows = [_mcp_row("causal_m", main_tables.TRAINING_PRESET, 1, **estimate, think_loss_diff=2.5, think_loss_diff_low=1.0, think_loss_diff_high=4.0),
            _mcp_row("causal_m", main_tables.AGENTS_PRESET, 2, **priced_nothing, think_loss_diff=0.0, think_loss_diff_low=0.0, think_loss_diff_high=0.0)]
    sheet = main_tables.Sheet({"MCP": rows}, "m")
    table = main_tables.mcp_table(sheet)
    lines = {tuple(labels): [cell.full for cell in cells] for group in table.groups for labels, cells in group}
    ours = "CausalDiagnosis agent + CausalDiagnosis MCP"
    assert [len(group) for group in table.groups] == [3, 3]
    assert lines[("All six big layers", ours)] == ["0.50 [0.25, 0.75]"] * 6 + ["+2.50 [+1.00, +4.00]"]
    assert lines[("Layer 0", ours)] == ["0.50 [0.25, 0.75]"] * 3 + ["none"] * 4
    assert lines[("All six big layers", "Plan-and-Execute + CausalDiagnosis MCP")] == ["no run"] * 7
    assert [cell.short for cell in table.groups[0][2][1]] == ["0.50"] * 6 + ["+2.50 [+1.00, +4.00]"]
    assert {(entry["table"], entry["csv"], entry["row"]) for entry in sheet.cited} == {("mcpvalue", "MCP.csv", 1), ("mcpvalue", "MCP.csv", 2)}


def test_the_module_names_one_flag_and_no_row_that_plays_the_mcp_alone():
    from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
    from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import RootCauses

    # the one key of the flag is a field of the reply of rank_root_causes, whatever the name the server gives it
    assert mcp_metrics.READ_FIRST in RootCauses.model_fields and not hasattr(mcp_metrics, "MCP_ALONE") and not hasattr(mcp_metrics, "READ_FIRST_KEYS")
    assert not any(hasattr(mcp_metrics, name) for name in ("_reply", "_refused", "_accepted", "_reached", "matched", "truth_lists"))
    entry = next(table for table in aggregate.tables() if table.id == "MCP")
    words = " ".join(str(getattr(entry, field)) for field in ("varies", "hypothesis", "metric", "reading", "rule"))
    words += " " + main_tables.mcp_table(main_tables.Sheet({"MCP": []}, "m")).caption
    old = "insufficient" + "_evidence"  # spelled in two parts, so that the rename of the flag leaves this name as it is
    # the word alone stands in the baseline of the cost prediction, the price and the downtime of the plan alone, so the test reads the
    # phrase that names a row of the MCP on its own
    assert "mcp alone" not in words.lower() and old not in words


# ----- episodes played through the real servers -----

SETTINGS = {"kind": "memory"}
SITE = {"site_name": "MAIN", "asset_id": "hyd_1"}
CE = "CE_CoolingEfficiency_percent_1Hz"
PLAYED_PRESETS = ("base", "priced")  # the preset of the first experiment, without a receipt and a clock, and a preset with prices


async def _ask(app, tool, arguments):
    """One tool call through the handler that the MCP package calls for a client. Returns the text the client receives."""
    import mcp.types as types

    from CausalDiagnosis.mcp_servers.utils.transcript_hooks import lowlevel_of, sent_text

    request = types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name=tool, arguments=arguments))
    return sent_text((await lowlevel_of(app).request_handlers[types.CallToolRequest](request)).root)


async def _script(path, start):
    """The calls of a scripted agent through the servers of one episode on the AssetOpsBench rows, built in process as their
    launchers build them: a reading and a ranking, an order before the diagnosis, which the engine refuses, the diagnosis of
    the first candidate, the orders of the first plan of the ranking that has steps, a reading, the finish, a call after the
    end and a call of a tool the row does not list. Returns the ranking and the index of the branch whose plan was ordered."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import build_server
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server as build_diagnostics
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    apps = {server: build_server(path, server, SETTINGS) for server in ("iot", "wo", TASK_SERVER)}
    diagnostics = build_diagnostics(episode.released_path(path), left_out=start)
    await _ask(apps["iot"], "history", {**SITE, "sensors": [CE], "limit": 2})
    ranking = json.loads(await _ask(diagnostics, "rank_root_causes", {"top_k": 5}))
    planned = next(k for k, b in enumerate(ranking["branches"]) if b["repair_plan"]["steps"])
    plan = ranking["branches"][planned]["repair_plan"]
    await _ask(apps["wo"], "generate_work_order", {"description": "replace", "asset_num": f"hyd_1_{plan['steps'][0]['component']}", "site_id": "MAIN"})
    await _ask(apps[TASK_SERVER], "submit_diagnosis", {**ranking["candidates"][0]["combination"], "evidence_ids": ""})
    for step in plan["steps"]:
        await _ask(apps["wo"], "generate_work_order", {"description": "replace", "asset_num": f"hyd_1_{step['component']}", "site_id": "MAIN",
                                                       "aob_source": {"prediction_id": plan["plan_prediction_id"] or ""}})
    await _ask(apps["iot"], "history", {**SITE, "sensors": [CE], "limit": 2})
    await _ask(apps[TASK_SERVER], "finish", {"summary": "done"})
    await _ask(apps["iot"], "sites", {})
    await _ask(apps["iot"], "submit_diagnosis", {**ranking["candidates"][0]["combination"], "evidence_ids": ""})
    return ranking, planned


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """{preset: (episode folder, its hidden log, the ranking, the index of the branch whose plan was ordered, the start)}, the
    hidden log written from the replay of the state file, as the driver's verify_episode writes it."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as rows
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    out = {}
    for preset in PLAYED_PRESETS:
        ep = tmp_path_factory.mktemp(preset)
        path = ep / "episode.json"
        state = rows.start(path, 7, load_config(preset)[1], default_facts_path())
        episode.hand_over(path)
        start = tuple(episode.replay(state, env_class=RecordedRigEnv).initial)
        EpisodeStore(path, SETTINGS).create()
        ranking, planned = asyncio.run(_script(path, start))
        log = episode.replay(episode.load_state(path), env_class=RecordedRigEnv).log
        (ep / "hidden").mkdir()
        (ep / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
        out[preset] = ep, log, ranking, planned, start
    return out


@needs_servers
@pytest.mark.parametrize("preset", PLAYED_PRESETS)
def test_on_episodes_played_through_the_real_servers_the_measures_agree_with_the_log(played, preset):
    """The acceptance case of the fault of the old measures, which read a refusal behind a receipt line as an accepted call: on
    both rows the transcript and the log agree, every answer is placed, the refused order before the diagnosis is no decision,
    the orders of the plan form one visit that follows that plan, and the diagnosis follows the first candidate."""
    ep, log, ranking, planned, start = played[preset]
    out = mcp_metrics.episode_measures(ep, None, "joint", {})
    assert out["unaligned"] is False and out["unplaced"] == 0
    call_list, aligned = mcp_metrics.calls(ep, log)
    made = mcp_metrics.decisions(call_list, log)
    refused = [call for call in call_list if call.index is not None and log[call.index]["event"] == "refused"]
    assert aligned and refused and not {call.n for call in refused} & {d.n for d in made}
    assert [d.kind for d in made] == ["read", "diagnosis", *["order"] * len(ranking["branches"][planned]["repair_plan"]["steps"]), "read"]
    asset = mcp_metrics.asset_truth(log)
    planned_set = mcp_metrics.plan_components(ranking["branches"][planned])
    needed = frozenset(asset.faulty(asset.levels(list(start))))
    assert (out["visits"], out["advised"], out["follow_any"], out["follow_first"]) == (1, 1, 1, int(planned == 0))
    assert out["harm"] == int(planned == 0 and planned_set != needed) and out["diagnosis_follows"] is True


@needs_servers
def test_the_default_reply_of_the_ranking_holds_every_field_that_the_measures_read(played):
    """The fields that the measures read, on the replies of the played episodes as the transcript holds them: a later default
    reply that drops or renames one fails here. Where the advice of a reply names a rank, it is the first branch."""
    for preset in PLAYED_PRESETS:
        ep, log, _ranking, _planned, _start = played[preset]
        replies = [call.answer for call in mcp_metrics.calls(ep, log)[0] if call.answer is not None]
        assert replies, preset
        for reply in replies:
            assert {"candidates", "belief_time", mcp_metrics.READ_FIRST, "branches"} <= set(reply), preset
            assert all({"combination", "probability"} <= set(c) for c in reply["candidates"]), preset
            if reply[mcp_metrics.READ_FIRST]:
                assert "sensor_group" in (reply.get("separating_test") or {}) or "sensor_group" in (reply.get("read_batch") or [{}])[0], preset
            for b in reply["branches"]:
                assert "combination" in b and "act_now" in b["cost_to_restored"], preset
                assert {"steps", "plan_prediction_id", "price", "downtime_cost"} <= set(b["repair_plan"]), preset
                assert all("component" in step for step in b["repair_plan"]["steps"]), preset
            # the step of the server's rule names a rank only when it advises the diagnosis, and a read or a declaration names none
            if isinstance(reply.get("advice"), dict) and reply["advice"].get("rank") is not None:
                assert reply["advice"]["rank"] == 1, preset


def test_the_cells_of_a_row_resample_the_clusters_of_the_tables_and_follow_the_registered_minimum():
    """The unit of resampling is the cluster that aggregate.cluster_of gives, so one cluster leaves no spread, and below the registered
    smallest number of clusters a cell prints its estimate alone."""
    one = mcp_metrics.row_cells(ON, 50, 0, lambda e: "one")
    assert one["uptake_low"] == one["uptake_high"] == one["uptake"]
    few = mcp_metrics.row_cells(ON, 50, 0, None, 30)
    assert few["uptake"] == pytest.approx(2 / 3, abs=1e-6) and (few["uptake_low"], few["uptake_high"], few["uptake_se"]) == ("", "", "")
