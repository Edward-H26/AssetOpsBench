"""The clock processes (CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/episode_clock.py) through the environment: scheduled drops, the work order
lifecycle, downtime accrual, the failure event, and the oracle set (oracle.py) under each of them."""
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order, worst_level
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, faulty_components, finish, order, read, repaired, submit_truth, summary, truth

HAND_START = (3, 90, 0, 130)  # cooler at its worst level, valve at its first level below healthy
LEAD = EpisodeConfig().durations["lead_time"]
DOWN = EpisodeConfig().durations["repair_downtime"]


def anchored(make_env, combination=HAND_START, **overrides):
    return make_env(start={"combination": list(combination), "entry_offset": 0}, **overrides)


def served_combinations(log):
    return [(o["t"], tuple(o["combination"])) for e in events(log, "read") for o in e["served"]]


def test_a_scheduled_drop_switches_the_served_block_at_the_scheduled_cycle(make_env, world):
    cycles, blocks = world
    drop_t, interval = 4, 3
    env = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [drop_t, drop_t], "interval": interval})
    env.reset(0)
    schedule = env.log[0]["schedule"]
    component = schedule[0]["component"]
    assert component in ("pump", "accumulator")  # a healthy component is preferred, so the episode changes
    assert [d["t"] for d in schedule] == [drop_t + i * interval for i in range(len(schedule))]
    assert [d["to_level"] for d in schedule] == degradation_order(env.config, component)[1:]
    for _ in range(drop_t + 2):
        read(env, "tier1", 1)
    dropped = list(HAND_START)
    dropped[COMPONENTS.index(component)] = schedule[0]["to_level"]
    served = served_combinations(env.log)
    assert [combo for t, combo in served if t < drop_t] == [HAND_START] * drop_t
    assert [combo for t, combo in served if t >= drop_t] == [tuple(dropped)] * 2
    assert all(o["cycle"] in blocks[tuple(o["combination"])] for e in events(env.log, "read") for o in e["served"])
    drop = events(env.log, "scheduled_drop")[0]
    assert drop["t_start"] == drop["t_end"] == drop_t and drop["step"] is None and drop["class"] == "system" and drop["cost"] == 0
    assert (drop["component"], drop["from_level"], drop["to_level"]) == (component, HEALTHY[component], schedule[0]["to_level"])
    assert env.combination_at(drop_t - 1) == HAND_START and env.combination_at(drop_t) == tuple(dropped)
    advance(env, interval)
    assert env.hidden[COMPONENTS.index(component)] == schedule[1]["to_level"]


def test_a_work_order_executes_after_the_lead_time_stops_the_rig_and_closes_with_the_repaired_block(make_env, world):
    cycles, blocks = world
    env = anchored(make_env)
    env.reset(0)
    submit_truth(env)
    obs, cost, *_ = order(env, "cooler")
    created = obs["result"]["work_order"]
    lead, down = LEAD["cooler"], DOWN["cooler"]
    assert cost == env.config.order_price("cooler") + env.config.prices["visit"] and created["created_t"] == 0 and created["expected_close_t"] == lead + down
    assert created["status"] == "open" and created["source"] == "agent" and created["closed_t"] is None
    assert events(env.log, "work_order_created")[0]["unnecessary"] is False and events(env.log, "work_order_created")[0]["class"] == "write"
    stopped_rate = env.config.downtime_rate["stopped"]
    running_rate = sum(env.config.level_rate(c, level) for c, level in zip(COMPONENTS, HAND_START))
    statuses, charges = [], []
    for t in range(lead + down):
        before = env.cost
        obs, *_ = advance(env, 1)
        charges.append(env.cost - before)
        statuses.append([(o["id"], o["status"]) for o in obs["open_work_orders"]])
    assert statuses[:lead] == [[("wo-1", "open")]] * lead and statuses[lead:-1] == [[("wo-1", "in_repair")]] * (down - 1) and statuses[-1] == []
    assert charges[:lead] == [running_rate] * lead and charges[lead:] == [stopped_rate] * down
    executed = events(env.log, "work_order_executed")[0]
    closed = events(env.log, "work_order_closed")[0]
    assert executed["id"] == closed["id"] == "wo-1" and executed["t_start"] == lead and closed["t_start"] == closed["t_end"] == lead + down
    assert closed["source"] == "agent" and closed["combination_after"] == list(repaired(HAND_START, "cooler")) == env.hidden
    assert env.combination_at(lead + down - 1) == HAND_START and env.combination_at(lead + down) == repaired(HAND_START, "cooler")
    read(env, "tier1", 1)
    observation = events(env.log, "read")[0]["served"][0]
    assert observation["cycle"] in blocks[repaired(HAND_START, "cooler")] and observation["t"] == lead + down
    obs, *_ = summary(env, 0)
    assert obs["result"]["work_orders_closed"][0]["closed_t"] == lead + down
    obs, *_ = order(env, "pump")  # healthy at creation
    assert events(env.log, "work_order_created")[-1]["unnecessary"] is True and obs["result"]["work_order"]["id"] == "wo-2"


def test_a_read_during_the_repair_downtime_serves_no_recording_for_the_stopped_cycles(make_env):
    env = anchored(make_env, durations={**EpisodeConfig().durations, "lead_time": {c: 0 for c in COMPONENTS}, "repair_downtime": {c: 3 for c in COMPONENTS}})
    env.reset(0)
    submit_truth(env)
    order(env, "valve")
    obs, cost, *_ = read(env, "tier1", 5)
    assert events(env.log, "read")[0]["cost"] == 2 * env.config.prices["read"]["tier1"]  # the two cycles that returned a reading
    assert obs["result"]["stopped"] == 3 and len(obs["result"]["served"]) == 2
    assert [o["t"] for o in obs["result"]["served"]] == [3, 4] and obs["open_work_orders"] == [] and obs["time"] == 5
    assert all(tuple(o["combination"]) == repaired(HAND_START, "valve") for o in events(env.log, "read")[0]["served"])
    assert [p for p in env.clock.played if p[1] is None] == [(0, None), (1, None), (2, None)]


def test_a_repair_without_downtime_closes_as_the_technician_arrives(make_env):
    env = anchored(make_env, durations={**EpisodeConfig().durations, "repair_downtime": {c: 0 for c in COMPONENTS}})
    env.reset(0)
    submit_truth(env)
    order(env, "valve")
    advance(env, LEAD["valve"] + 1)
    closed = events(env.log, "work_order_closed")[0]
    assert closed["t_start"] == LEAD["valve"] and not events(env.log, "work_order_executed")
    assert all(index is not None for _t, index in env.clock.played)  # the rig never stopped
    assert env.hidden == list(repaired(HAND_START, "valve"))


def test_downtime_accrues_per_degraded_component_at_the_configured_rates(make_env):
    env = anchored(make_env)
    env.reset(0)
    rate = env.config.level_rate("cooler", 3) + env.config.level_rate("valve", 90)
    advance(env, 6)
    assert env.cost == 6 * rate and env.clock.downtime == [{"t": t, "charge": rate} for t in range(6)]
    # the budget counter moves with prices alone and the downtime goes to the hidden ledger
    assert env.envelope()["budget_left"] == env.config.budget and env.ledger == {"downtime": 6 * rate, "penalties": 0}
    both_fixed = anchored(make_env, allowed=None)
    both_fixed.reset(0)
    submit_truth(both_fixed)
    for c in faulty_components(HAND_START):
        order(both_fixed, c)
    advance(both_fixed, 20)
    assert both_fixed.hidden == [HEALTHY[c] for c in COMPONENTS]
    before = both_fixed.cost
    advance(both_fixed, 5)
    assert both_fixed.cost == before  # a healthy running rig accrues nothing


def test_a_failure_fires_after_patience_cycles_at_the_worst_level_and_strict_ends_the_episode(make_env):
    patience, penalty = 4, 50
    lenient = anchored(make_env, failure={"patience": patience, "strict": False, "penalty": penalty})
    lenient.reset(0)
    before = lenient.cost
    obs, cost, done, info = advance(lenient, 12)
    failures = events(lenient.log, "failure")
    assert not done and len(failures) == 1 and failures[0]["component"] == "cooler" and failures[0]["t_start"] == patience - 1
    assert failures[0]["cost"] == penalty and cost == lenient.cost - before and cost > penalty
    assert worst_level(lenient.config, "cooler") == HAND_START[0]
    strict = anchored(make_env, failure={"patience": patience, "strict": True, "penalty": penalty})
    strict.reset(0)
    obs, cost, done, info = advance(strict, 12)
    assert done and info["reason"] == "failure" and strict.time == patience and obs["time"] == patience and obs["episode_over"]
    assert [e["event"] for e in strict.log[-3:]] == ["failure", "wait", "end"]
    # the repair of the cooler closes after its lead time and its downtime: under a patience equal to the two the order is
    # open at the window end, so the failure fires there and the repair closes one cycle later, and one cycle more of
    # patience lets the repair close in time
    repair = LEAD["cooler"] + DOWN["cooler"]
    still_open = anchored(make_env, failure={"patience": repair, "strict": True, "penalty": penalty})
    still_open.reset(0)
    submit_truth(still_open)
    order(still_open, "cooler")
    obs, cost, done, info = advance(still_open, 12)
    assert done and info["reason"] == "failure" and [e["t_start"] for e in events(still_open.log, "failure")] == [repair - 1]
    assert [e["t_start"] for e in events(still_open.log, "work_order_closed")] == [repair]
    averted = anchored(make_env, failure={"patience": repair + 1, "strict": True, "penalty": penalty})
    averted.reset(0)
    submit_truth(averted)
    order(averted, "cooler")
    obs, cost, done, info = advance(averted, 12)
    assert not done and not events(averted.log, "failure") and averted.hidden[0] == HEALTHY["cooler"]


def test_the_oracle_names_the_faulty_set_with_window_ends(make_env):
    patience = EpisodeConfig().failure["patience"]
    env = anchored(make_env)
    env.reset(0)
    oracle = env.oracle()
    assert oracle == {"diagnosis": truth(env), "finish": True,
                      "work_orders": [{"component": "cooler", "window_end": patience - 1}, {"component": "valve", "window_end": env.config.horizon}]}
    drop_t = 20
    scheduled = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [drop_t, drop_t], "interval": 5})
    scheduled.reset(0)
    schedule = scheduled.log[0]["schedule"]
    component = schedule[0]["component"]
    last_drop = max(d["t"] for d in schedule)
    expected_end = min(last_drop + patience - 1, scheduled.config.horizon)
    orders = {o["component"]: o["window_end"] for o in scheduled.oracle()["work_orders"]}
    assert orders == {"cooler": patience - 1, "valve": scheduled.config.horizon, component: expected_end}
    late = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [59, 59], "interval": 10}, horizon=60)
    late.reset(0)
    assert {o["component"] for o in late.oracle()["work_orders"]} == {"cooler", "valve", late.log[0]["schedule"][0]["component"]}
    beyond = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [60, 60], "interval": 10}, horizon=60)
    beyond.reset(0)
    assert {o["component"] for o in beyond.oracle()["work_orders"]} == {"cooler", "valve"}  # a drop at the horizon never happens
    early_end = anchored(make_env, schedule={"mode": "seeded", "components": 1, "first_drop": [20, 20], "interval": 10})
    early_end.reset(0)
    advance(early_end, 20)
    finish(early_end)  # the episode ends at the drop's cycle, which the clock never passed
    assert {o["component"] for o in early_end.oracle()["work_orders"]} == {"cooler", "valve"} and not events(early_end.log, "scheduled_drop")
