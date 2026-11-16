"""One deadline for the failure card and the timing check: the card counts the cycles a component spends at its worst level
until a repair restores it, so an open work order no longer averts the failure, and the window end of the oracle is the
deadline of the card and of the timing check of the verifier alike. The oracle policy orders early enough that each repair
closes by that end, and plans its visits where one visit would let a component fail. Data-dependent assertions are
relations, never values."""
import dataclasses
import json
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.splits.development_split import development_starts
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing as rows_briefing
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, finish, order, submit_truth

VALVE_START = {"combination": [100, 90, 0, 130], "entry_offset": 0}  # the valve one level below healthy, never at its worst
PUMP_WORST = {"combination": [100, 90, 2, 130], "entry_offset": 0}  # the pump at its worst level, a dependency of it degraded
CLOSED_REPAIR = "only a repair on it that has closed by then averts it"
from CausalDiagnosis.utils.paths import CONFIGS
EVERY_WORST = (3, 73, 2, 90)  # a development start with every component at its worst level


def _scheduled(make_env, patience, interval=1):
    """An episode in which a scheduled drop takes one healthy component to its worst level, with the strict failure card."""
    env = make_env(start=VALVE_START, schedule={"mode": "seeded", "components": 1, "first_drop": [2, 2], "interval": interval},
                   failure={"patience": patience, "strict": True, "penalty": 50})
    env.reset(0)
    component = env.log[0]["schedule"][0]["component"]
    window_end = {o["component"]: o["window_end"] for o in env.oracle()["work_orders"]}[component]
    return env, component, window_end


def _diagnose_now(env):
    return env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENTS, env.hidden)), "evidence_ids": []})


def _close_t(event):
    return event["t_end"] if event.get("t_end") is not None else event["t_start"]


def test_an_order_open_at_the_window_end_no_longer_averts_the_failure(make_env):
    """An order created at the window end is open at the failure check of that cycle, and the failure fires there all the
    same: the strict card ends the episode before the repair closes, so the episode does not pass."""
    env, component, window_end = _scheduled(make_env, patience=4)
    advance(env, window_end)
    _diagnose_now(env)
    order(env, component)
    obs, _cost, done, info = advance(env, 10)
    assert done and info["reason"] == "failure"
    assert [(e["component"], e["t_start"]) for e in events(env.log, "failure")] == [(component, window_end)]
    assert events(env.log, "work_order_created")[0]["t_start"] == window_end and not events(env.log, "work_order_closed")
    verdict = verify(env.log, released_from_env(env), env.oracle(), cost_oracle=None)
    assert not verdict["pass"] and verdict["failures"] == [component]


def test_a_repair_that_closes_by_the_window_end_averts_the_failure_and_meets_the_timing_check(make_env):
    """The valve, degraded from the start, is repaired at once. The order on the scheduled component placed at its window end
    less the lead time and the repair downtime closes at the window end, before the failure check of that cycle, so nothing
    fails, the episode passes and the timing row of that order is met at its bound."""
    env, component, window_end = _scheduled(make_env, patience=8)
    lead_and_downtime = env.config.lead_time(component) + env.config.durations["repair_downtime"][component]
    _diagnose_now(env)
    order(env, "valve")
    advance(env, window_end - lead_and_downtime)
    order(env, component)
    advance(env, lead_and_downtime + 1)
    finish(env)
    assert not events(env.log, "failure")
    verdict = verify(env.log, released_from_env(env), env.oracle(), cost_oracle=None)
    row = verdict["checks"]["timing"]["orders"][0]
    assert verdict["pass"] and verdict["checks"]["timing"]["ok"] and row["component"] == component and row["closed_t"] == window_end


def test_an_ineffective_close_leaves_the_count_running(make_env):
    """The pump is at its worst level from the start while a component it depends on is degraded, so its repair closes without
    effect. The count of its cycles at the worst level never stops, the open order included, and the failure fires at the
    window end of the oracle, not patience cycles after the close."""
    patience = 8
    env = make_env(start=PUMP_WORST, failure={"patience": patience, "strict": True, "penalty": 50})
    env.reset(0)
    window_end = {o["component"]: o["window_end"] for o in env.oracle()["work_orders"]}["pump"]
    submit_truth(env)
    order(env, "pump")
    obs, _cost, done, info = advance(env, patience + 10)
    ineffective = events(env.log, "order_ineffective")
    assert ineffective and ineffective[0]["component"] == "pump" and ineffective[0]["t_start"] < window_end
    assert done and info["reason"] == "failure"
    assert [(e["component"], e["t_start"]) for e in events(env.log, "failure")] == [("pump", window_end)] == [("pump", patience - 1)]


def test_the_oracle_orders_before_the_last_drop_when_an_order_after_it_would_close_late(make_env):
    """With a patience shorter than the lead time and the repair downtime that follow the last drop, an order placed after the
    drop closes after the window end, so the oracle orders the component earlier, from the latest cycle whose repair still
    closes by the window end and after the drop. Every episode passes with no failure, every order closes by the window end of
    its component, and on some seeds the order of the scheduled component comes before its last drop."""
    env = make_env(start=VALVE_START, schedule={"mode": "seeded", "components": 1, "first_drop": [2, 2], "interval": 3},
                   failure={"patience": 4, "strict": True, "penalty": 50})
    early = 0
    for seed in range(12):
        log = run_episode(env, OraclePolicy(env), seed)
        oracle = env.oracle()
        verdict = verify(log, released_from_env(env), oracle, cost_oracle=None)
        assert verdict["pass"] and verdict["checks"]["timing"]["ok"] and not events(log, "failure"), seed
        ends = {o["component"]: o["window_end"] for o in oracle["work_orders"]}
        closes = {e["id"]: _close_t(e) for e in events(log, "work_order_closed")}
        last_drop = max(d["t"] for d in log[0]["schedule"])
        scheduled = log[0]["schedule"][0]["component"]
        for created in events(log, "work_order_created"):
            assert closes[created["id"]] <= ends[created["component"]], (seed, created["component"])
            if created["component"] == scheduled:
                assert closes[created["id"]] - 1 >= last_drop, seed  # the repair closes after the last drop, so it lasts
                early += created["t_start"] < last_drop
    assert early


def test_both_briefings_state_that_only_a_closed_repair_averts_the_failure(make_env, world):
    """The sentence of the failure card on both surfaces, the rig's and the AssetOpsBench rows', stands wherever the card is on,
    and the old exemption of an open order stands nowhere. The end rules are part of the second guidance level."""
    cycles, blocks = world
    for failure in ({"patience": 10, "strict": True, "penalty": 50}, {"patience": 10, "strict": False, "penalty": 50},
                    {"patience": 0, "strict": False, "penalty": 0}):
        env = make_env(failure=failure, guidance_level=2)
        env.reset(0)
        rows = RecordedRigEnv(cycles, blocks, env.config)
        rows.reset(0)
        for text in (briefing(env.public_profile(), env.config), rows_briefing(rows.public_profile(), env.config)):
            assert (CLOSED_REPAIR in text) is (failure["patience"] > 0) and "without an open work order" not in text


def test_the_oracle_plans_its_visits_where_one_visit_would_let_a_component_fail(world):
    """On the full environment the oracle delays its one visit so that no hazard draw undoes a repair, and on some seeds that
    delay would carry a close past a window end. There the oracle plays the plan of order cycles that its copies of the
    environment show to pass with no failure. On a development start with every component at its worst level, every seed
    passes with no failure, and on some seed the one-visit schedule would have failed. Development starts only."""
    cycles, blocks = world
    assert EVERY_WORST in set(development_starts(get_profile("hyd_1"), set(blocks)))
    preset = compose(json.loads((CONFIGS / "layer_presets" / "deterioration.json").read_text(encoding="utf-8")))
    config = dataclasses.replace(preset, allowed=[list(EVERY_WORST)], fault_rule={"min_faulty": 4, "max_faulty": 4})
    planned = 0
    for seed in range(24):
        env = HydraulicRigEnv(cycles, blocks, config)
        oracle = OraclePolicy(env)
        log = run_episode(env, oracle, seed)
        verdict = verify(log, released_from_env(env), env.oracle(), cost_oracle=None)
        assert verdict["pass"] and not events(log, "failure"), seed
        planned += oracle.plan is not None
    assert planned


def test_a_repair_that_ends_no_run_at_the_worst_level_meets_the_timing_check_on_development_starts(world):
    """A component at its worst level from the start is repaired in time, dropped by the hazard to a level above its worst
    and repaired again after the window end of its first run. With one window per component the oracle missed the timing
    check there with a pass and no failure on some seed of each of these two development starts. The second close ends no
    run at the worst level, so it reads the horizon, and every seed passes with no failure and meets the timing check.
    Development starts only."""
    cycles, blocks = world
    preset = compose(json.loads((CONFIGS / "layer_presets" / "deterioration.json").read_text(encoding="utf-8")))
    development = set(development_starts(get_profile("hyd_1"), set(blocks)))
    for start in ((3, 73, 2, 130), (3, 100, 1, 90)):
        assert start in development
        n = preset.n_faulty(start)
        config = dataclasses.replace(preset, allowed=[list(start)], fault_rule={"min_faulty": n, "max_faulty": n})
        for seed in range(24):
            env = HydraulicRigEnv(cycles, blocks, config)
            log = run_episode(env, OraclePolicy(env), seed)
            verdict = verify(log, released_from_env(env), env.oracle(), cost_oracle=None)
            assert verdict["pass"] and not events(log, "failure") and verdict["checks"]["timing"]["ok"], (start, seed)
