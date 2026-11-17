"""The visit rule of the environment (utils/public_rules/visits.py): the standard orders that an agent
creates one after another, with no other call between them, form one technician visit with one visit charge, whatever the
thinking time between the calls; any other call of the environment ends the run, a refused call included; a late order joins
after the repairs of its visit that have begun. No test states a value computed from the rig data."""
import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import schedule_visit

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import events, order, read, submit_truth

PUMP_AND_VALVE = (100, 80, 2, 130)  # two faulty components of the asset file, the pump depending on the valve


def anchored(make_env, **overrides):
    return make_env(start={"combination": list(PUMP_AND_VALVE), "entry_offset": 0}, **overrides)


def ordered_with(env, between, think=0):
    """The valve, then `between` (nothing, a read, a refused call), then the pump with `think` thinking cycles before it."""
    submit_truth(env)
    order(env, "valve")
    if between == "read":
        read(env, "tier1", 1)
    elif between == "refused":
        env.step({"type": "get_failure_modes", "component": "no such component"})
    env.step({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []}, think_cycles=think)
    return [(e["visit"], e["visit_charge"]) for e in events(env.log, "work_order_created")]


@pytest.mark.parametrize("clock_mode", ["generation_time", "declared"])
@pytest.mark.parametrize("think", [0, 1, 3])
def test_consecutive_orders_form_one_visit_whatever_the_thinking_between_them(make_env, clock_mode, think):
    env = anchored(make_env, clock_mode=clock_mode)
    env.reset(0)
    charge = env.config.prices["visit"]
    assert ordered_with(env, None, think) == [(1, charge), (1, 0)]


@pytest.mark.parametrize("between", ["read", "refused"])
def test_any_call_between_two_orders_ends_the_run_and_opens_a_second_visit(make_env, between):
    env = anchored(make_env, clock_mode="generation_time")
    env.reset(0)
    charge = env.config.prices["visit"]
    assert ordered_with(env, between) == [(1, charge), (2, charge)]


def test_an_order_after_thinking_joins_the_visit_and_waits_for_its_own_lead_time(make_env):
    """The pump ordered three cycles after the valve, nothing between: one visit, and the technician starts the repairs not begun
    at the latest creation cycle plus lead time, the valve first by its priority number, as schedule_visit gives for orders of
    different cycles."""
    env = anchored(make_env, clock_mode="generation_time")
    env.reset(0)
    ordered_with(env, None, think=3)
    config = env.config
    valve, pump = (next(o for o in env.clock.orders if o.component == c) for c in ("valve", "pump"))
    assert valve.visit == pump.visit and (valve.created_t, pump.created_t) == (0, 3)
    lead, down = config.durations["lead_time"], config.durations["repair_downtime"]
    if valve.status == "open":
        start = max(0 + lead["valve"], 3 + lead["pump"])
        assert (valve.execute_t, valve.close_t, pump.execute_t, pump.close_t) == (start, start + down["valve"], start + down["valve"],
                                                                                  start + down["valve"] + down["pump"])


def test_a_late_order_is_repaired_after_the_repair_of_its_visit_that_has_begun(make_env):
    """With no lead times, the valve's repair begins at once; the pump ordered while it runs joins the visit, pays no visit
    charge and is repaired after it, at the latest of its own creation cycle and the valve's close."""
    zero_lead = {"lead_time": {c: 0 for c in ("cooler", "valve", "pump", "accumulator")}}
    env = anchored(make_env, clock_mode="generation_time")
    durations = {**env.config.durations, **zero_lead}
    env = anchored(make_env, clock_mode="generation_time", durations=durations)
    env.reset(0)
    down = env.config.durations["repair_downtime"]
    if down["valve"] < 2:
        pytest.skip("the asset's valve repair is too short to be running two cycles later")
    created = ordered_with(env, None, think=1)
    valve, pump = (next(o for o in env.clock.orders if o.component == c) for c in ("valve", "pump"))
    assert created == [(1, env.config.prices["visit"]), (1, 0)] and valve.status == "in_repair"
    assert (valve.execute_t, valve.close_t) == (0, down["valve"])
    assert (pump.execute_t, pump.close_t) == (max(1, down["valve"]), max(1, down["valve"]) + down["pump"])


def test_orders_of_one_cycle_keep_the_schedule_of_one_cycle(make_env):
    """For orders created in one cycle the rule gives the schedule it gave before: arrival after the longest lead time, the
    repairs back to back in priority order."""
    env = anchored(make_env)
    config = env.config
    lead, down = config.durations["lead_time"], config.durations["repair_downtime"]
    members = [("pump", "standard"), ("valve", "standard")]
    arrival = 5 + max(lead["pump"], lead["valve"])
    first, second = sorted(members, key=lambda m: config.priority[m[0]])
    plan = schedule_visit(config, 5, members)
    assert [(members[i][0], start, close) for i, start, close in plan] == [
        (first[0], arrival, arrival + down[first[0]]), (second[0], arrival + down[first[0]], arrival + down[first[0]] + down[second[0]])]


def test_a_no_call_step_between_two_orders_keeps_one_visit_in_the_environment_the_board_and_the_released_orders(make_env):
    """A no_call step, the step a turn without a call to a tool of the rig takes, is no call of
    the environment, so like thinking it ends no run of orders. Two orders with a no_call step between them make one visit and
    one visit charge in the environment, in the agent's mirror of the run (Board.note_step) and in the released orders that the
    diagnostics server reads."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
    from CausalDiagnosis.utils.public_rules.visits import keeps_run

    env = anchored(make_env, clock_mode="generation_time")
    env.reset(0)
    charge = env.config.prices["visit"]
    submit_truth(env)
    order(env, "valve")
    env.step({"type": "no_call"})
    env.step({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []})
    created = events(env.log, "work_order_created")
    assert keeps_run("no_call") and not keeps_run("read_sensors")
    assert [(e["visit"], e["visit_charge"]) for e in created] == [(1, charge), (1, 0)]
    board = Board()
    board.note_step("order", {"result": {"work_order": {"visit": 1, "mode": "standard"}}})
    run = board.run
    board.note_step("no_call", {})
    assert run is not None and board.run == run  # the mirror keeps the run open over the no_call step
    board.note_step("read", {})
    assert board.run is None  # any call of the environment ends it
    released = env.public_work_orders()  # the open orders every envelope releases, which the diagnostics server reads
    assert len(released) == 2 and {o["visit"] for o in released} == {1}
