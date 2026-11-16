"""The repair rules of the environment (CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_4_repair_rules.py and env.py): the dependency rule with its
ineffective orders, the visits with one charge each, the fix history drawn from the seed, the repair manual and the
emergency mode with its hold, each through the environment and, where a failure class follows, through the verifier.
No test states a value computed from the rig data."""
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, finish, order, read, submit_truth, summary

PUMP_AND_VALVE = (100, 80, 2, 130)  # the pump depends on the valve under the dependencies of the asset file
COOLER_ONLY = (20, 100, 0, 130)  # the cooler depends on nothing
HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]
# the seeded processes of CausalDiagnosis/configs/development/timed.json
TIMED_PRESET = {"schedule": {"mode": "seeded", "components": 1, "first_drop": [5, 30], "interval": 10}}


def anchored(make_env, combination, **overrides):
    return make_env(start={"combination": list(combination), "entry_offset": 0}, **overrides)


def emergency(env, component, mode="emergency"):
    return env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": [], "mode": mode})


def verdict(env):
    """The verdict of a finished episode; the failure class does not depend on the cost reference."""
    return verify(env.log, released_from_env(env), env.oracle(), None)


def stopped_cycles(env):
    return [t for t, index in env.clock.played if index is None]


# ----- dependencies -----
def test_an_order_on_the_pump_while_the_valve_is_degraded_closes_ineffective_and_the_class_fires(make_env):
    env = anchored(make_env, PUMP_AND_VALVE)
    env.reset(0)
    read(env, "tier1", 1)
    submit_truth(env)
    obs, cost, *_ = order(env, "pump")
    assert cost == env.config.order_price("pump") + env.config.prices["visit"]  # the price is charged as usual
    close_t = obs["result"]["work_order"]["expected_close_t"]
    advance(env, close_t - env.time)
    closed, ineffective = events(env.log, "work_order_closed"), events(env.log, "order_ineffective")
    assert [e["id"] for e in closed] == [e["id"] for e in ineffective] == ["wo-1"] and closed[0]["t_start"] == ineffective[0]["t_start"] == close_t
    assert ineffective[0]["degraded_dependencies"] == ["valve"] and env.hidden == list(PUMP_AND_VALVE) == closed[0]["combination_after"]
    assert len(stopped_cycles(env)) == env.config.durations["repair_downtime"]["pump"]  # and so is the downtime
    # the agent sees an ordinary closed order and has to learn from the readings that the repair did not take
    assert env.public_event(ineffective[0]) is None and "order_ineffective" not in {e["event"] for e in env.public_events()}
    listed = summary(env, 0)[0]["result"]["work_orders_closed"]
    assert [(o["id"], o["status"]) for o in listed] == [("wo-1", "closed")]
    finish(env, "The pump was replaced and the rig is restored.")
    v = verdict(env)
    # the class comes before the contradiction of the summary, which an agent that never saw the repair fail trips too
    assert not v["recovered"] and v["summary_check"]["contradicts_log"] and v["failure_class"] == "repair_before_dependency"


def test_the_oracle_passes_with_the_dependencies_on(make_env):
    env = make_env()
    assert env.config.depends_on("pump")  # the dependency rule is on by default
    needs_a_dependency = 0
    for start in env.candidates:
        single = make_env(allowed=[list(start)])
        log = run_episode(single, OraclePolicy(single), 0)
        v = verdict(single)
        assert v["pass"] and v["failure_class"] == "none" and v["checks"]["hard"]["ok"] and not events(log, "order_ineffective"), start
        levels = dict(zip(COMPONENTS, start))
        needs_a_dependency += levels["pump"] != HEALTHY["pump"] and any(levels[d] != HEALTHY[d] for d in single.config.depends_on("pump"))
    assert needs_a_dependency  # some starts repair a dependency and the pump in one visit
    timed = make_env(**TIMED_PRESET)
    for seed in range(30):
        log = run_episode(timed, OraclePolicy(timed), seed)
        v = verdict(timed)
        assert v["pass"] and v["checks"]["hard"]["work_orders_ok"] and not events(log, "order_ineffective") and not events(log, "refused"), seed


# ----- visits -----
def test_two_orders_of_one_cycle_form_one_visit_close_in_priority_order_and_charge_one_visit(make_env):
    env = anchored(make_env, PUMP_AND_VALVE)
    env.reset(0)
    submit_truth(env)
    config = env.config
    first, first_cost, *_ = order(env, "pump")  # created before the valve, repaired after it
    second, second_cost, *_ = order(env, "valve")
    assert first_cost == config.order_price("pump") + config.prices["visit"] and second_cost == config.order_price("valve")
    created = events(env.log, "work_order_created")
    assert [(e["visit"], e["visit_charge"]) for e in created] == [(1, config.prices["visit"]), (1, 0)]
    assert config.priority["valve"] < config.priority["pump"]
    lead, down = config.durations["lead_time"], config.durations["repair_downtime"]
    arrival = max(lead["pump"], lead["valve"])
    assert first["result"]["work_order"]["expected_close_t"] == arrival + down["pump"]  # before the valve joined the visit
    expected = {o["id"]: o["expected_close_t"] for o in second["open_work_orders"]}
    assert expected == {"wo-2": arrival + down["valve"], "wo-1": arrival + down["valve"] + down["pump"]}
    advance(env, expected["wo-1"])
    assert [e["id"] for e in events(env.log, "work_order_executed")] == ["wo-2", "wo-1"]
    assert [(e["id"], e["t_start"]) for e in events(env.log, "work_order_closed")] == [("wo-2", expected["wo-2"]), ("wo-1", expected["wo-1"])]
    assert stopped_cycles(env) == list(range(arrival, expected["wo-1"]))  # the repairs follow each other
    assert env.hidden == HEALTHY_LIST and not events(env.log, "order_ineffective")


def test_two_orders_in_different_cycles_charge_two_visits(make_env):
    env = anchored(make_env, PUMP_AND_VALVE)
    env.reset(0)
    submit_truth(env)
    config = env.config
    _obs, valve_cost, *_ = order(env, "valve")
    advance(env, 1)
    _obs, pump_cost, *_ = order(env, "pump")
    assert valve_cost == config.order_price("valve") + config.prices["visit"] and pump_cost == config.order_price("pump") + config.prices["visit"]
    assert [(e["visit"], e["visit_charge"]) for e in events(env.log, "work_order_created")] == [(1, config.prices["visit"]), (2, config.prices["visit"])]
    advance(env, 10)
    assert env.hidden == HEALTHY_LIST and not events(env.log, "order_ineffective")  # the valve's visit closed first


def test_a_mode_other_than_standard_or_emergency_is_refused_and_an_empty_mode_is_standard(make_env):
    env = anchored(make_env, PUMP_AND_VALVE)
    env.reset(0)
    submit_truth(env)
    for mode in ("urgent", 1, ["emergency"]):
        obs, cost, done, info = emergency(env, "valve", mode)
        assert info == {"refused": "mode must be standard or emergency"} and cost == 0
    obs, cost, *_ = emergency(env, "valve", "")
    assert obs["result"]["work_order"]["mode"] == "standard" and events(env.log, "work_order_created")[0]["mode"] == "standard"


# ----- the fix history -----
def test_the_fix_history_is_drawn_from_the_seed_apart_from_every_other_draw_and_follows_its_rule(make_env):
    env = make_env(**TIMED_PRESET)
    off = make_env(**TIMED_PRESET, fix_history={"healthy_probability": 0.0, "healthy_ago": [5, 25], "degraded_probability": 0.0, "degraded_ago": [30, 80]})
    rule = env.config.fix_history
    counts = {"healthy": [0, 0], "degraded": [0, 0]}  # components seen, past orders drawn
    for seed in range(200):
        env.reset(seed)
        history, drawn = env.fix_history, (env.initial, env.window_start, env.entry_offset, env.clock.schedule)
        assert env.log[0]["fix_history"] == history
        env.reset(seed)
        assert env.fix_history == history  # reproducible from the seed
        off.reset(seed)
        assert off.fix_history == [] and (off.initial, off.window_start, off.entry_offset, off.clock.schedule) == drawn
        assert [e["id"] for e in history] == [f"hist/{n}" for n in range(1, len(history) + 1)]
        assert [e["closed_t"] for e in history] == sorted(e["closed_t"] for e in history) and len({e["component"] for e in history}) == len(history)
        for c, level in zip(COMPONENTS, env.initial):
            kind = "healthy" if level == HEALTHY[c] else "degraded"
            entries = [e for e in history if e["component"] == c]
            counts[kind][0] += 1
            counts[kind][1] += len(entries)
            low, high = rule[f"{kind}_ago"]
            assert all(-high <= e["closed_t"] <= -low and (e["mode"], e["outcome"]) == ("standard", "closed") for e in entries), (seed, c)
    # mildly informative by construction: a past order is drawn a little more often for a healthy component
    for kind, (components, drawn_orders) in counts.items():
        assert abs(drawn_orders / components - rule[f"{kind}_probability"]) < 0.1, kind


def test_get_fix_history_and_lookup_fix_are_free_reads_and_refuse_an_unknown_component(make_env):
    env = anchored(make_env, PUMP_AND_VALVE)
    env.reset(3)
    config = env.config
    obs, cost, *_ = env.step({"type": "get_fix_history"})
    assert cost == 0 and obs["time"] == 0 and obs["result"] == {"history": env.fix_history}
    obs, cost, *_ = env.step({"type": "lookup_fix", "component": "pump"})
    manual = obs["result"]
    assert cost == 0 and obs["time"] == 0 and manual == env.fix_manual("pump")
    assert manual["dependencies"] == config.depends_on("pump") and manual["priority"] == config.priority["pump"]
    standard, urgent = manual["standard"], manual["emergency"]
    assert (standard["fix"], standard["price"], standard["visit_charge"]) == (config.prices["fix_kind"]["pump"], config.order_price("pump"), config.prices["visit"])
    assert (standard["lead_time"], standard["repair_downtime"]) == (config.durations["lead_time"]["pump"], config.durations["repair_downtime"]["pump"])
    assert (urgent["price"], urgent["lead_time"], urgent["hold"]) == (config.order_price("pump", "emergency"), config.emergency["lead"], config.emergency["hold"])
    assert all(f"the {d}" in manual["effect"] for d in manual["dependencies"])
    cooler = env.step({"type": "lookup_fix", "component": "cooler"})[0]["result"]
    assert cooler["dependencies"] == [] and "provided" not in cooler["effect"]
    for bad in ({"type": "lookup_fix", "component": "turbine"}, {"type": "lookup_fix", "component": ""}, {"type": "lookup_fix"}):
        assert env.step(bad)[3] == {"refused": "unknown component"}
    assert [e["class"] for e in events(env.log, "fix_history") + events(env.log, "fix_lookup")] == ["read"] * 3


def test_hist_identifiers_pass_the_provenance_check_once_the_history_was_served(make_env):
    env = anchored(make_env, COOLER_ONLY)
    for seed in range(100):
        env.reset(seed)
        if env.fix_history:
            break
    cited = [e["id"] for e in env.fix_history]

    def play(serve_history):
        env.reset(seed)
        if serve_history:
            env.step({"type": "get_fix_history"})
        read(env, "tier1", 1)
        submit_truth(env, cited)
        order(env, "cooler", evidence=cited[:1])
        advance(env, 10)
        finish(env)
        return verdict(env)["checks"]["provenance"]

    assert play(True) == {"ok": True, "cited": len(cited) + 1, "unmatched": []}
    released = released_from_env(env)
    assert next(e for e in released["events"] if e["event"] == "fix_history")["history"] == env.fix_history
    assert play(False)["unmatched"] == sorted(cited)  # an identifier the agent was never served is not evidence


# ----- the emergency mode -----
def test_an_emergency_order_comes_after_the_emergency_lead_costs_the_surcharge_restores_and_expires_after_the_hold(make_env):
    env = anchored(make_env, COOLER_ONLY)
    env.reset(0)
    read(env, "tier1", 1)
    submit_truth(env)
    config, created_t = env.config, env.time
    lead, down, hold = config.emergency["lead"], config.durations["repair_downtime"]["cooler"], config.emergency["hold"]
    obs, cost, *_ = emergency(env, "cooler")
    assert cost == config.order_price("cooler") * config.emergency["surcharge"] + config.prices["visit"]
    assert obs["result"]["work_order"]["mode"] == "emergency" and obs["result"]["work_order"]["expected_close_t"] == created_t + lead + down
    advance(env, lead + down)
    executed, closed = events(env.log, "work_order_executed"), events(env.log, "work_order_closed")
    assert executed[0]["t_start"] == created_t + lead and closed[0]["t_start"] == created_t + lead + down and env.hidden == HEALTHY_LIST
    advance(env, hold)
    assert env.hidden == HEALTHY_LIST and not events(env.log, "emergency_expired")  # healthy for the whole hold
    advance(env, 1)  # the component returns as the clock passes the end of the hold, as a drop does
    expired = events(env.log, "emergency_expired")
    assert [(e["id"], e["t_start"], e["to_level"]) for e in expired] == [("wo-1", closed[0]["t_start"] + hold, COOLER_ONLY[0])]
    assert env.hidden == list(COOLER_ONLY) and env.public_event(expired[0]) is None
    finish(env, "The cooler was repaired.")
    v = verdict(env)
    assert not v["recovered"] and v["failure_class"] == "temporary_repair_expired"
    # a standard order that closes while the hold lasts makes the repair permanent, and it is not an unnecessary order
    held = anchored(make_env, COOLER_ONLY)
    held.reset(0)
    submit_truth(held)
    emergency(held, "cooler")
    advance(held, lead + down)
    order(held, "cooler")
    advance(held, hold + 5)
    assert not events(held.log, "emergency_expired") and held.hidden == HEALTHY_LIST
    assert [e["unnecessary"] for e in events(held.log, "work_order_created")] == [False, False]


# ----- a hold in force at the end of the episode -----
def test_an_episode_that_ends_inside_an_emergency_hold_is_not_recovered_and_does_not_pass(make_env):
    """The true diagnosis, an emergency order on the one faulty component and the finish, which runs the clock to the close of
    the order: the component then stands at its healthy level by the hold alone. The end entry names the hold with the level
    it replaced, and the verifier scores that level, so the episode is not recovered and does not pass."""
    env = anchored(make_env, COOLER_ONLY)
    env.reset(0)
    read(env, "tier1", 1)
    submit_truth(env)
    emergency(env, "cooler")
    finish(env, "The cooler was repaired.")
    end = env.log[-1]
    assert end["event"] == "end" and end["held"] == {"cooler": COOLER_ONLY[0]} and env.hidden == HEALTHY_LIST
    v = verdict(env)
    assert v["diagnosis_all_correct"] and not v["recovered"] and not v["pass"] and v["graded_pass"] == 0.0
    assert v["checks"]["outcome"]["held"] == ["cooler"] and v["failure_class"] == "temporary_repair_expired"
    assert v["final"] == dict(zip(COMPONENTS, HEALTHY_LIST))  # the verdict keeps the levels of the end entry


def test_an_emergency_order_followed_by_a_standard_order_that_closes_is_recovered(make_env):
    """A standard order that closes while the hold lasts ends the hold, so nothing is held at the end and the repair counts."""
    env = anchored(make_env, COOLER_ONLY)
    env.reset(0)
    read(env, "tier1", 1)
    submit_truth(env)
    config = env.config
    emergency(env, "cooler")
    advance(env, config.emergency["lead"] + config.durations["repair_downtime"]["cooler"])
    order(env, "cooler")
    finish(env, "The cooler was repaired.")
    assert env.log[-1]["event"] == "end" and "held" not in env.log[-1] and env.hidden == HEALTHY_LIST
    v = verdict(env)
    assert v["recovered"] and v["pass"] and v["checks"]["outcome"]["held"] == [] and v["failure_class"] == "none"


def test_a_hold_in_force_at_the_end_is_hidden_from_every_public_view(make_env):
    env = anchored(make_env, COOLER_ONLY)
    env.reset(0)
    submit_truth(env)
    emergency(env, "cooler")
    finish(env)
    assert "held" in env.log[-1]
    assert "held" not in env.public_event(env.log[-1])
    assert all("held" not in e for e in released_from_env(env)["events"]) and all("held" not in e for e in env.public_events())
