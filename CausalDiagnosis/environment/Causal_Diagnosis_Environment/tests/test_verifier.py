"""The verifier on synthetic logs that follow the log schema of the environment entry by entry: one test per check and
per failure class, the reward order, pass^k against the estimator's formula, and the scope "all" against "writes". The
last test plays the oracle policy and the do nothing policy on the real environment when it is present."""
from CausalDiagnosis.utils.paths import CONFIGS
import json
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.tests import knowledge_tables as knowledge
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import ABSTENTION_CREDIT, FAILURE_CLASSES, good, pass_power_k, read_counts, released_from_env, reward, verify

HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]
# A start with the cooler at its worst recorded level and every other component healthy (design levels, not data).
COOLER_FAULTY = [3, 100, 0, 130]
# A start with the cooler and the accumulator faulty.
TWO_FAULTY = [3, 100, 0, 90]
TEMPERATURE = "TS1_Temperature_C_1Hz_mean"  # indicates the cooler
PRESSURE = "PS1_Pressure_bar_100Hz_mean"  # indicates the valve and the accumulator
BANDS = {TEMPERATURE: {"low": 0.0, "high": 10.0}, PRESSURE: {"low": 0.0, "high": 10.0}}
INSIDE = {TEMPERATURE: 5.0, PRESSURE: 5.0}
HOT = {TEMPERATURE: 20.0, PRESSURE: 5.0}  # the temperature outside its normal range
HIGH_PRESSURE = {TEMPERATURE: 5.0, PRESSURE: 20.0}
PRICES = {"read": {"tier1": 1, "tier2": 3, "tier3": 10}, "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20},
          "repair_charge": 20}
LEAD_TIME, REPAIR_DOWNTIME, HORIZON = 3, 1, 60
TASK_TEXT = "Find the condition of every component of hyd_1 and restore the rig."


class Episode:
    """Builds a log entry by entry as the environment writes it, with the hidden combination followed
    through the system events, and derives the released evidence (format 3) and the oracle from the same entries."""

    def __init__(self, hidden, seed=0):
        self.t = 0
        self.step = 0
        self.n = 0
        self.orders = 0
        self.current = dict(zip(COMPONENTS, hidden))
        self.log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "class": "system", "cost": 0, "seed": seed,
                     "hidden": list(hidden), "schedule": {"mode": "none"},
                     "config": {"prices": PRICES, "horizon": HORIZON}}]

    # agent calls
    def _call(self, event, cls, duration=0, cost=0, **fields):
        self.step += 1
        entry = {"event": event, "step": self.step, "t_start": self.t, "t_end": self.t + duration, "class": cls, "cost": cost, **fields}
        self.t += duration
        self.log.append(entry)
        return entry

    def profile(self):
        return self._call("profile", "read")

    def read(self, values=INSIDE, group="tier1", cycles=1):
        served = []
        for _ in range(cycles):
            self.n += 1
            served.append({"n": self.n, "t": self.t + len(served), "cycle": 1000 + self.n, "combination": list(self.current.values()),
                           "repeated": False, "values": dict(values)})
        return self._call("read", "read", cycles, cycles * PRICES["read"][group], sensor_group=group, cycles=cycles, served=served)

    def advance(self, cycles):
        return self._call("advance", "read", cycles, 0, cycles=cycles)

    def summary(self, since, alarms=(), closed=()):
        return self._call("summary", "read", since=since, alarms=list(alarms), work_orders_closed=list(closed))

    def failure_modes(self, component=""):
        return self._call("failure_modes", "read", component=component)

    def diagnose(self, diagnosis, evidence_ids=()):
        return self._call("submit_diagnosis", "write", diagnosis=dict(diagnosis), evidence_ids=list(evidence_ids))

    def declare(self, evidence_ids=(), test=None):
        return self._call("declare_insufficient_evidence", "write", separating_test=test or {"sensor_group": "tier3", "cycles": 1},
                          evidence_ids=list(evidence_ids))

    def order(self, component, prediction_id="", evidence_ids=()):
        self.orders += 1
        return self._call("work_order_created", "write", 0, PRICES["parts"][component] + PRICES["repair_charge"],
                          id=f"wo-{self.orders}", component=component, prediction_id=prediction_id, evidence_ids=list(evidence_ids),
                          expected_close_t=self.t + LEAD_TIME + REPAIR_DOWNTIME, unnecessary=self.current[component] == HEALTHY[component])

    def finish(self, summary=""):
        return self._call("finish", "write", summary=summary)

    def refuse(self, action, reason, cls="write"):
        return self._call("refused", cls, action=action, reason=reason)

    # system events at the current time
    def _system(self, event, **fields):
        entry = {"event": event, "step": None, "t_start": self.t, "t_end": self.t, "class": "system", **{"cost": 0, **fields}}
        self.log.append(entry)
        return entry

    def drop(self, component, to_level):
        entry = self._system("scheduled_drop", component=component, from_level=self.current[component], to_level=to_level)
        self.current[component] = to_level
        return entry

    def hazard_drop(self, component, to_level):
        entry = self._system("hazard_drop", component=component, from_level=self.current[component], to_level=to_level, cause="hazard")
        self.current[component] = to_level
        return entry

    def close(self, order_id, component, source="agent"):
        self._system("work_order_executed", id=order_id)
        self.current[component] = HEALTHY[component]
        return self._system("work_order_closed", id=order_id, component=component, source=source, combination_after=list(self.current.values()))

    def ineffective(self, order_id, component, degraded):
        """A close that did not take: the order closes with the combination unchanged and the hidden event names the
        degraded dependencies."""
        self._system("work_order_executed", id=order_id)
        self._system("work_order_closed", id=order_id, component=component, source="agent", combination_after=list(self.current.values()))
        return self._system("order_ineffective", id=order_id, component=component, degraded_dependencies=list(degraded))

    def expire(self, order_id, component, to_level):
        self.current[component] = to_level
        return self._system("emergency_expired", id=order_id, component=component, to_level=to_level,
                            combination_after=list(self.current.values()))

    def failure(self, component, penalty=50):
        return self._system("failure", component=component, cost=penalty)

    def end(self, reason="finish", downtime=(), held=None):
        """The end entry, with held the components that an emergency hold keeps at their healthy level, each with the level
        the hold replaced, as the environment writes it while such a hold is in force."""
        downtime = [{"t": t, "charge": charge} for t, charge in downtime]
        total = sum(d["charge"] for d in downtime)
        prices = sum(e["cost"] for e in self.log)
        return self._system("end", reason=reason, time=self.t, final=list(self.current.values()), cost=prices + total,
                            downtime_total=total, downtime_by_cycle=downtime, **({"held": dict(held)} if held else {}))

    # the derived files
    def released(self, bands=BANDS):
        """The public view the action server releases: the entries after reset without the hidden fields, without the
        scheduled drops (the schedule is hidden) and without the component of a failure."""
        hidden_keys = {"cycle", "combination", "combination_after", "unnecessary", "hidden", "schedule", "final", "held"}
        events = []
        for e in self.log[1:]:
            if e["event"] in ("scheduled_drop", "hazard_drop", "order_ineffective", "emergency_expired"):
                continue
            public = {k: v for k, v in e.items() if k not in hidden_keys and not (e["event"] == "failure" and k == "component")}
            if e["event"] == "read":
                public["served"] = [{k: v for k, v in s.items() if k not in hidden_keys} for s in e["served"]]
            events.append(public)
        observations = [{"n": s["n"], "t": s["t"], "sensor_group": e["sensor_group"], "values": s["values"]}
                        for e in events if e["event"] == "read" for s in e["served"]]
        return {"format": 3, "asset_id": "hyd_1", "task_text": TASK_TEXT, "guidance_level": 1,
                "rules": {"components": list(COMPONENTS), "healthy": dict(HEALTHY), "prices": PRICES, "horizon": HORIZON,
                          "normal_ranges": bands},
                "status": {"time": self.t, "episode_over": self.log[-1]["event"] == "end"}, "events": events, "observations": observations}

    def oracle(self, work_orders=None, reads=None):
        hidden = self.log[0]["hidden"]
        if work_orders is None:
            work_orders = [(c, HORIZON) for c, level in zip(COMPONENTS, hidden) if level != HEALTHY[c]]
        oracle = {"diagnosis": dict(zip(COMPONENTS, hidden)), "work_orders": [{"component": c, "window_end": w} for c, w in work_orders],
                  "finish": True}
        if reads is not None:
            oracle["reads"] = reads
        return oracle


def oracle_like(hidden=COOLER_FAULTY, summary="Replaced the cooler; the rig is restored."):
    """A read, the true diagnosis citing it, one work order per faulty component, time until they close, finish."""
    ep = Episode(hidden)
    ep.read(HOT if hidden[0] != HEALTHY["cooler"] else INSIDE)
    ep.diagnose(dict(zip(COMPONENTS, hidden)), [f"obs/1/{TEMPERATURE}"])
    faulty = [c for c, level in zip(COMPONENTS, hidden) if level != HEALTHY[c]]
    orders = [ep.order(c, evidence_ids=[f"obs/1/{TEMPERATURE}"]) for c in faulty]
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    for entry in orders:
        ep.close(entry["id"], entry["component"])
    ep.finish(summary)
    ep.end()
    return ep


def do_nothing(hidden=COOLER_FAULTY):
    ep = Episode(hidden)
    ep.advance(HORIZON)
    ep.finish("Nothing was done.")
    ep.end(reason="finish", downtime=[(t, 2) for t in range(HORIZON)])
    return ep


def verdict_of(ep, cost_oracle=None, **kwargs):
    oracle_cost = cost_oracle if cost_oracle is not None else verify(oracle_like(ep.log[0]["hidden"]).log, oracle_like(ep.log[0]["hidden"]).released(),
                                                                       ep.oracle(), None)["cost"]
    return verify(ep.log, ep.released(), kwargs.pop("oracle", None) or ep.oracle(), oracle_cost, **kwargs)


# ----- the checks -----

def test_an_oracle_like_log_passes_every_check_and_is_json():
    ep = oracle_like()
    v = verdict_of(ep)
    assert v["verified"] and v["all_checks_ok"] and v["pass"] and v["failure_class"] == "none"
    assert v["diagnosis_per_component"] == {c: True for c in COMPONENTS} and v["recovered"] and v["unnecessary"] == 0
    assert v["cost_normalized"] == 1.0 and v["end_reason"] == "finish" and v["steps"] == 5 and v["refusals"] == 0
    assert v["turns_to_correct_diagnosis"] == 2 and v["abstained"] is False and v["time_used"] == 1 + LEAD_TIME + REPAIR_DOWNTIME
    assert json.loads(json.dumps(v)) == v


def test_the_count_gate_needs_one_commitment_one_finish_and_no_unknown_write():
    without_finish = oracle_like()
    without_finish.log = [e for e in without_finish.log if e["event"] != "finish"]
    v = verdict_of(without_finish)
    assert not v["checks"]["count_gate"]["ok"] and "finish" in v["checks"]["count_gate"]["problems"][0]

    two_commitments = Episode(COOLER_FAULTY)
    two_commitments.read(HOT)
    two_commitments.declare()
    two_commitments.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    two_commitments.finish()
    two_commitments.end()
    v = verdict_of(two_commitments)
    assert not v["checks"]["count_gate"]["ok"] and v["checks"]["count_gate"]["writes"]["declare_insufficient_evidence"] == 1

    unknown_write = oracle_like()
    unknown_write.log.insert(2, {"event": "reset_rig", "step": 2, "t_start": 1, "t_end": 1, "class": "write", "cost": 0})
    v = verdict_of(unknown_write)
    assert not v["checks"]["count_gate"]["ok"] and "reset_rig" in v["checks"]["count_gate"]["problems"][0]

    refused_write = oracle_like()
    refused_write.log.insert(2, {"event": "refused", "step": 2, "t_start": 1, "t_end": 1, "class": "write",
                                 "action": {"type": "create_work_order", "component": "cooler"}, "reason": "no diagnosis yet"})
    assert verdict_of(refused_write)["checks"]["count_gate"]["ok"], "a refused action is not a write"


def test_the_hard_checks_compare_levels_per_component_and_work_orders_as_unordered_lists():
    wrong_valve = Episode(TWO_FAULTY)
    wrong_valve.read(HOT)
    wrong_valve.diagnose({"cooler": 3, "valve": 90, "pump": 0, "accumulator": 90})
    wrong_valve.order("accumulator")
    wrong_valve.order("cooler")
    wrong_valve.advance(LEAD_TIME + REPAIR_DOWNTIME)
    wrong_valve.close("wo-1", "accumulator")
    wrong_valve.close("wo-2", "cooler")
    wrong_valve.finish()
    wrong_valve.end()
    v = verdict_of(wrong_valve)
    assert v["diagnosis_per_component"] == {"cooler": True, "valve": False, "pump": True, "accumulator": True}
    assert not v["diagnosis_all_correct"] and not v["checks"]["hard"]["ok"]
    assert v["checks"]["hard"]["work_orders_ok"], "the order of two repairs does not matter"
    assert v["recovered"] and v["pass"] is False and v["turns_to_correct_diagnosis"] is None

    extra_order = oracle_like()
    extra_order.log.insert(3, dict(extra_order.log[3], id="wo-9", component="pump", unnecessary=True))
    v = verdict_of(extra_order)
    assert not v["checks"]["hard"]["work_orders_ok"] and v["checks"]["hard"]["work_orders_agent"] == ["cooler", "pump"]

    no_order = Episode(COOLER_FAULTY)
    no_order.read(HOT)
    no_order.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    no_order.finish()
    no_order.end()
    v = verdict_of(no_order)
    assert not v["checks"]["hard"]["work_orders_ok"] and not v["recovered"] and v["diagnosis_all_correct"]


def test_a_second_order_on_a_component_keeps_the_hard_check():
    """An agent may order a component again after an ineffective or expired repair, a legal recovery: the hard check compares
    the set of components ordered with the oracle's, and the orders themselves stay listed."""
    repeated = oracle_like()
    repeated.log.insert(3, dict(repeated.log[3], id="wo-9"))
    hard = verdict_of(repeated)["checks"]["hard"]
    assert hard["work_orders_agent"] == ["cooler", "cooler"] and hard["work_orders_oracle"] == ["cooler"]
    assert hard["work_orders_ok"] and hard["ok"]


def test_the_timing_check_applies_to_scheduled_components_and_accepts_a_close_at_the_window_end():
    def scheduled(close_delay, with_close=True):
        ep = Episode(HEALTHY_LIST)
        ep.advance(5)
        ep.drop("pump", 2)
        ep.read(INSIDE)
        ep.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)))
        ep.order("pump")
        ep.advance(close_delay)
        if with_close:
            ep.close("wo-1", "pump")
        ep.finish()
        ep.end()
        oracle = ep.oracle(work_orders=[("pump", 15)])  # the pump would fail at cycle 15 under the patience rule
        return verify(ep.log, ep.released(), oracle, 10.0)

    assert scheduled(LEAD_TIME + REPAIR_DOWNTIME)["checks"]["timing"]["ok"]
    at_window_end = scheduled(15 - 6)
    assert at_window_end["checks"]["timing"]["orders"][0]["closed_t"] == 15 and at_window_end["checks"]["timing"]["ok"]
    late = scheduled(20)
    assert not late["checks"]["timing"]["ok"] and late["checks"]["timing"]["orders"][0]["window_end"] == 15
    never = scheduled(20, with_close=False)
    assert not never["checks"]["timing"]["ok"] and never["checks"]["timing"]["orders"][0]["closed_t"] is None
    # a work order on a component the schedule never touched is not timed
    assert verdict_of(oracle_like())["checks"]["timing"] == {"ok": True, "orders": []}


def _order_on_a_pump_at_its_worst_level_from_the_start(close_delay):
    """The verdict of a rig whose pump stands at its worst level from the start and never drops, with an order on the pump that
    closes close_delay cycles after it, against an oracle whose window for the pump ends at cycle 9."""
    ep = Episode([HEALTHY["cooler"], HEALTHY["valve"], 2, HEALTHY["accumulator"]])
    ep.read(INSIDE)
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "pump": 2})
    ep.order("pump")
    ep.advance(close_delay)
    ep.close("wo-1", "pump")
    ep.finish()
    ep.end()
    return verify(ep.log, ep.released(), ep.oracle(work_orders=[("pump", 9)]), 10.0)


def test_an_order_on_a_component_at_its_worst_level_from_the_start_gets_its_row_beside_the_checks_alone():
    """The timing check times the components that degrade during the episode, so an order on a pump at its worst level from the
    start has no row there; worst_level_timing gives it its row against the oracle's window end, late after it and in time by it,
    while the checks, all_checks_ok, pass and every score are the same for the late and the timely close."""
    late, in_time = _order_on_a_pump_at_its_worst_level_from_the_start(20), _order_on_a_pump_at_its_worst_level_from_the_start(LEAD_TIME + REPAIR_DOWNTIME)
    for verdict in (late, in_time):
        assert verdict["checks"]["timing"] == {"ok": True, "orders": []}
    rows = late["worst_level_timing"]["orders"]
    assert not late["worst_level_timing"]["ok"] and [(r["component"], r["window_end"], r["ok"]) for r in rows] == [("pump", 9, False)]
    assert in_time["worst_level_timing"]["ok"] and in_time["worst_level_timing"]["orders"][0]["closed_t"] <= 9
    same = ("checks", "all_checks_ok", "pass", "success", "recovered", "unnecessary")
    assert {key: late[key] for key in same} == {key: in_time[key] for key in same}
    # a component off its worst level gets no row: an order on a valve one level below its healthy level
    ep = Episode([HEALTHY["cooler"], 90, HEALTHY["pump"], HEALTHY["accumulator"]])
    ep.read(INSIDE)
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "valve": 90})
    ep.order("valve")
    ep.advance(20)
    ep.close("wo-1", "valve")
    ep.finish()
    ep.end()
    assert verify(ep.log, ep.released(), ep.oracle(work_orders=[("valve", None)]), 10.0)["worst_level_timing"] == {"ok": True, "orders": []}


def _order_on_a_pump_that_a_hazard_drop_degraded(close_delay):
    """The verdict of a healthy rig whose pump a hazard drop takes to its worst level at cycle 5, with an order on the pump
    at cycle 6 that closes close_delay cycles later, against an oracle whose window for the pump ends at cycle 15."""
    ep = Episode(HEALTHY_LIST)
    ep.advance(5)
    ep.hazard_drop("pump", 2)
    ep.read(INSIDE)
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "pump": 2})
    ep.order("pump")
    ep.advance(close_delay)
    ep.close("wo-1", "pump")
    ep.finish()
    ep.end()
    return verify(ep.log, ep.released(), ep.oracle(work_orders=[("pump", 15)]), 10.0)


def test_an_order_on_a_component_that_a_hazard_drop_degraded_passes_the_timing_check_when_it_closes_in_time():
    """The timing check covers every component that degrades during the episode, so an order on a component
    that only a hazard drop degraded gets its row, and a close before the window end passes."""
    timing = _order_on_a_pump_that_a_hazard_drop_degraded(LEAD_TIME + REPAIR_DOWNTIME)["checks"]["timing"]
    assert timing == {"ok": True, "orders": [{"id": "wo-1", "component": "pump", "closed_t": 10, "window_end": 15, "ok": True}]}


def test_an_order_on_a_component_that_a_hazard_drop_degraded_fails_the_timing_check_when_it_closes_after_the_window_end():
    """A close after the cycle at which the component that a hazard drop degraded would fail misses the
    oracle's window, as it does on a component that the schedule dropped."""
    verdict = _order_on_a_pump_that_a_hazard_drop_degraded(20)
    assert verdict["checks"]["timing"] == {"ok": False, "orders": [{"id": "wo-1", "component": "pump", "closed_t": 26, "window_end": 15,
                                                                 "ok": False}]}
    assert not verdict["all_checks_ok"]


def _pump_degraded_twice(second_level, second_close, patience=10, failed=False):
    """The verdict of a rig whose pump is at its worst level from the start, under a failure card of the given patience. The
    first order on the pump closes at cycle 5, inside the window of that run, or at cycle 11 after a failure event at cycle 9
    when failed is set. The hazard takes the pump to second_level at cycle 13, and a second order closes at second_close."""
    ep = Episode([HEALTHY["cooler"], HEALTHY["valve"], 2, HEALTHY["accumulator"]])
    ep.log[0]["config"]["failure"] = {"patience": patience, "strict": False, "penalty": 50}
    ep.read(INSIDE)
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "pump": 2})
    ep.order("pump")
    if failed:
        ep.advance(9 - ep.t)
        ep.failure("pump")
        ep.advance(11 - ep.t)
    else:
        ep.advance(5 - ep.t)
    ep.close("wo-1", "pump")
    ep.advance(13 - ep.t)
    ep.hazard_drop("pump", second_level)
    ep.order("pump")
    ep.advance(second_close - ep.t)
    ep.close("wo-2", "pump")
    ep.finish()
    ep.end()
    window = patience - 1 if patience else HORIZON
    return verify(ep.log, ep.released(), ep.oracle(work_orders=[("pump", window)]), 10.0)


def test_a_close_that_ends_no_run_at_the_worst_level_reads_the_horizon():
    """The pump is repaired inside the window of its run at the worst level, the hazard then takes it one level below healthy,
    above its worst level, and the second order closes after the window end of the first run. That close ends no run at the
    worst level, where the failure card counts nothing, so its row reads the horizon and is met."""
    rows = _pump_degraded_twice(1, 20)["checks"]["timing"]["orders"]
    assert [(r["id"], r["closed_t"], r["window_end"], r["ok"]) for r in rows] == [("wo-1", 5, 9, True), ("wo-2", 20, HORIZON, True)]


def test_a_second_run_at_the_worst_level_has_its_own_window():
    """The hazard takes the repaired pump back to its worst level at cycle 13, which opens a second run whose window ends at
    cycle 13 + patience - 1, where the failure card would fire on it. A close by that end is met and a close after it is not,
    whatever the window of the first run."""
    met = _pump_degraded_twice(2, 22)["checks"]["timing"]
    assert met["ok"] and [r["window_end"] for r in met["orders"]] == [9, 13 + 10 - 1]
    late = _pump_degraded_twice(2, 23)["checks"]["timing"]
    assert not late["ok"] and [(r["window_end"], r["ok"]) for r in late["orders"]] == [(9, True), (13 + 10 - 1, False)]


def test_without_the_failure_card_every_run_reads_the_horizon():
    """With the failure card off the oracle's window is the horizon, and so is the window of every later run."""
    rows = _pump_degraded_twice(2, 50, patience=0)["checks"]["timing"]["orders"]
    assert [(r["window_end"], r["ok"]) for r in rows] == [(HORIZON, True), (HORIZON, True)]


def test_a_run_that_opens_after_the_component_failed_reads_the_horizon():
    """The failure card fails a component once, so a run at the worst level that opens after the failure of its component
    has no cycle at which the card would fire, and its window is the horizon. The late first repair still misses its run."""
    rows = _pump_degraded_twice(2, 50, failed=True)["checks"]["timing"]["orders"]
    assert [(r["closed_t"], r["window_end"], r["ok"]) for r in rows] == [(11, 9, False), (50, HORIZON, True)]


def test_the_end_of_an_emergency_hold_opens_a_run_with_its_own_window():
    """The hazard takes a healthy pump to its worst level at cycle 2, which opens a run whose window ends at cycle 11. An
    emergency repair closes at cycle 4 and holds the pump at its healthy level until cycle 12, when the hold expires and the
    pump comes back to its worst level: a second run opens there, with its window end at cycle 21. A standard order that
    closes at cycle 16 reads that window and is met, whatever the window of the first run."""
    ep = Episode(HEALTHY_LIST)
    ep.log[0]["config"]["failure"] = {"patience": 10, "strict": False, "penalty": 50}
    ep.read(INSIDE)
    ep.advance(2 - ep.t)
    ep.hazard_drop("pump", 2)
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "pump": 2})
    ep.order("pump")
    ep.advance(4 - ep.t)
    ep.close("wo-1", "pump")
    ep.advance(12 - ep.t)
    ep.expire("wo-1", "pump", 2)
    ep.order("pump")
    ep.advance(16 - ep.t)
    ep.close("wo-2", "pump")
    ep.finish()
    ep.end()
    rows = verify(ep.log, ep.released(), ep.oracle(work_orders=[("pump", 2 + 10 - 1)]), 10.0)["checks"]["timing"]["orders"]
    assert [(r["id"], r["closed_t"], r["window_end"], r["ok"]) for r in rows] == [("wo-1", 4, 11, True), ("wo-2", 16, 12 + 10 - 1, True)]


def test_the_causality_check_wants_the_commitment_before_every_work_order():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.order("cooler")
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    assert not v["checks"]["causality"]["ok"] and v["checks"]["causality"]["first_order_index"] < v["checks"]["causality"]["commit_index"]
    assert verdict_of(oracle_like())["checks"]["causality"]["ok"]


def test_the_outcome_reads_the_final_combination_and_counts_unnecessary_orders():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    ep.order("pump")  # healthy at creation
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "pump")
    ep.close("wo-2", "cooler")
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    assert v["recovered"] and v["unnecessary"] == 1 and v["checks"]["outcome"]["unnecessary_orders"] == ["wo-1"] and not v["pass"]
    assert v["final"] == dict(HEALTHY)

    open_order = Episode(COOLER_FAULTY)
    open_order.read(HOT)
    open_order.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    open_order.order("cooler")
    open_order.finish()
    open_order.end()
    v = verdict_of(open_order)
    assert not v["recovered"] and v["final"]["cooler"] == 3 and v["end_reason"] == "finish"


def test_the_provenance_check_knows_every_identifier_kind_and_rejects_a_fabricated_one():
    ep = Episode(TWO_FAULTY)
    ep.read(HOT)
    ep.summary(0, alarms=[{"channel": TEMPERATURE, "first_t": 0}])
    ep.failure_modes("cooler")
    cited = [f"obs/1/{TEMPERATURE}", f"alarm/0/{TEMPERATURE}", f"fmea/{knowledge.FMEA_SOURCE}/cooler_efficiency_critical"]
    ep.diagnose(dict(zip(COMPONENTS, TWO_FAULTY)), cited)
    ep.order("cooler", prediction_id="pred/1", evidence_ids=["wo/wo-0"])
    ep.order("accumulator", evidence_ids=["wo/wo-1"])
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")
    ep.close("wo-2", "accumulator")
    ep.finish()
    ep.end()
    predictions = [{"id": "pred/1", "action": "replace:cooler", "recovery": 0.5}]
    v = verify(ep.log, ep.released(), ep.oracle(), 10.0, predictions=predictions)
    assert v["checks"]["provenance"] == {"ok": False, "cited": 6, "unmatched": ["wo/wo-0"]}
    assert v["failure_class"] == "unmatched_evidence" and reward(v, HORIZON) == reward(as_honest_failure(v), HORIZON) < 0.0
    assert v["prediction_verdict"] == [{"work_order": "wo-1", "prediction_id": "pred/1", "found": True, "predicted_recovery": 0.5,
                                        "recovery_error": 0.5, "verification": None}]
    without_predictions = verify(ep.log, ep.released(), ep.oracle(), 10.0)
    assert without_predictions["checks"]["provenance"]["unmatched"] == ["pred/1", "wo/wo-0"]
    # a code of a component whose failure modes were not requested is not released evidence
    ep.log[4]["evidence_ids"].append(f"fmea/{knowledge.FMEA_SOURCE}/pump_internal_leakage_severe")
    assert f"fmea/{knowledge.FMEA_SOURCE}/pump_internal_leakage_severe" in verify(ep.log, ep.released(), ep.oracle(), 10.0)["checks"]["provenance"]["unmatched"]


def test_the_provenance_check_releases_only_the_failure_modes_the_tool_serves():
    """get_failure_modes serves the codes that describe a fault, critical or degraded, so a code of a healthy state or of a
    transient, which the profile lists and the tool never serves, is no released evidence after a call of the tool."""
    served = sorted(f"fmea/{knowledge.FMEA_SOURCE}/{e['code']}" for e in knowledge.FAULT_CODES)
    unserved = sorted(f"fmea/{knowledge.FMEA_SOURCE}/{e['code']}" for e in knowledge.FAILURE_CODES if e not in knowledge.FAULT_CODES)
    assert unserved
    ep = Episode(TWO_FAULTY)
    ep.read(HOT)
    ep.failure_modes("")
    ep.diagnose(dict(zip(COMPONENTS, TWO_FAULTY)), [f"obs/1/{TEMPERATURE}", *served, *unserved])
    ep.finish()
    ep.end()
    provenance = verify(ep.log, ep.released(), ep.oracle(), 10.0)["checks"]["provenance"]
    assert provenance["unmatched"] == unserved and provenance["cited"] == 1 + len(served) + len(unserved)


# ----- the failure classes -----

def test_premature_diagnosis_is_a_diagnosis_with_no_reading_before_it():
    ep = Episode(COOLER_FAULTY)
    ep.profile()
    ep.diagnose({"cooler": 20, "valve": 100, "pump": 0, "accumulator": 130})
    ep.finish()
    ep.end()
    assert verdict_of(ep)["failure_class"] == "premature_diagnosis"
    # the oracle policy diagnoses without reading too, but a passed episode has no failure class
    no_read = oracle_like()
    no_read.log = [e for e in no_read.log if e["event"] != "read"]
    for e in no_read.log:
        if e["event"] in ("submit_diagnosis", "work_order_created"):
            e["evidence_ids"] = []
    v = verdict_of(no_read)
    assert v["pass"] and v["failure_class"] == "none"


def test_unmatched_evidence_comes_before_an_unnecessary_replacement():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), ["obs/7/nothing"])
    ep.order("pump")
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "pump")
    ep.close("wo-2", "cooler")
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    assert v["failure_class"] == "unmatched_evidence" and v["unnecessary"] == 1
    ep.log[2]["evidence_ids"] = [f"obs/1/{TEMPERATURE}"]
    assert verdict_of(ep)["failure_class"] == "unnecessary_replacement"


SEPARABLE, INSEPARABLE = {"separable": True, "cost": 12, "e18": {"separable": True}}, {"separable": False, "cost": None, "e18": {"separable": False}}
# a start the label marks inseparable while the server's E18 grade marks it separable, and the reverse
E18_SEPARABLE, LABEL_SEPARABLE = {**INSEPARABLE, "e18": {"separable": True}}, {**SEPARABLE, "e18": {"separable": False}}


def declared_episode():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.declare([f"obs/1/{TEMPERATURE}"])
    ep.finish("The evidence does not separate the candidates; one tier three reading would.")
    ep.end()
    return ep


def test_abstention_is_a_failure_on_a_separable_start_and_correct_on_an_inseparable_one():
    ep = declared_episode()
    separable = verdict_of(ep, grade=SEPARABLE)
    assert separable["abstained"] and not separable["abstention_correct"] and separable["failure_class"] == "abstention_on_separable"
    assert separable["separable"] is True and not separable["success"] and not separable["commitment_on_inseparable"]
    inseparable = verdict_of(ep, grade=INSEPARABLE)
    assert inseparable["abstention_correct"] and inseparable["success"] and inseparable["failure_class"] == "none" and not inseparable["pass"]
    assert inseparable["diagnosis_per_component"] == {c: None for c in COMPONENTS}
    # an unknown grade counts as separable, since it is no evidence that the declaration was right
    unknown = verdict_of(ep, grade=None)
    assert unknown["separable"] is True and not unknown["abstention_correct"] and unknown["failure_class"] == "abstention_on_separable"
    with pytest.raises(ValueError):
        verdict_of(ep, grade=12)


def test_success_is_a_pass_on_every_start_or_a_declaration_on_a_start_the_label_marks_inseparable():
    """The four cases on a start the label marks not separable and the other labels: a correct commitment and a
    correct declaration are successes, a wrong commitment and a declaration on a separable start are not, and the two
    descriptive flags that En5 reads keep their meaning."""
    committed = oracle_like()
    for grade in (SEPARABLE, None):
        v = verdict_of(committed, grade=grade)
        assert v["pass"] and v["success"] and not v["commitment_on_inseparable"] and v["failure_class"] == "none"
    v = verdict_of(committed, grade=INSEPARABLE)
    assert v["pass"] and v["success"] and v["commitment_on_inseparable"] and v["failure_class"] == "none"
    assert v["separable"] is False and not v["abstained"] and not v["abstention_correct"]
    declared = verdict_of(declared_episode(), grade=INSEPARABLE)
    assert declared["success"] and declared["abstention_correct"] and not declared["pass"] and not declared["commitment_on_inseparable"]
    wrong = Episode(COOLER_FAULTY)
    wrong.read(HOT)
    wrong.diagnose(dict(zip(COMPONENTS, [100, 100, 0, 130])), [f"obs/1/{TEMPERATURE}"])
    wrong.finish("Nothing is wrong.")
    wrong.end()
    v = verdict_of(wrong, grade=INSEPARABLE)
    assert not v["pass"] and not v["success"] and v["commitment_on_inseparable"] and not v["abstention_correct"]
    on_separable = verdict_of(declared_episode(), grade=SEPARABLE)
    assert not on_separable["pass"] and not on_separable["success"] and on_separable["abstained"] and not on_separable["abstention_correct"]


# ----- a declaration counts only when it cites a reading released before it -----

CITED = [f"obs/1/{TEMPERATURE}"]
NOT_A_READING = [f"alarm/0/{TEMPERATURE}", f"fmea/{knowledge.FMEA_SOURCE}/cooler_efficiency_critical"]


def outcome_term(verdict):
    """The reward without its cost and delay terms."""
    return reward(verdict, HORIZON, w_c=0.0, w_t=0.0)


def declaring(*steps):
    """An episode of the given steps on a start with a faulty cooler, then a finish."""
    ep = Episode(COOLER_FAULTY)
    for step in steps:
        step(ep)
    ep.finish("The evidence does not separate the candidates.")
    ep.end()
    return ep


def read(ep):
    ep.read(HOT)


def read_alarm_and_modes(ep):
    ep.read(HOT)
    ep.summary(0, alarms=[{"channel": TEMPERATURE, "first_t": 0}])
    ep.failure_modes("cooler")


WITHOUT_A_READING = {
    "no read at all": declaring(lambda ep: ep.declare()),
    "a read it does not cite": declaring(read, lambda ep: ep.declare()),
    "identifiers that are no reading": declaring(read_alarm_and_modes, lambda ep: ep.declare(NOT_A_READING)),
    "a reading released after it": declaring(lambda ep: ep.declare(CITED), read),
}


@pytest.mark.parametrize("ep", WITHOUT_A_READING.values(), ids=WITHOUT_A_READING)
@pytest.mark.parametrize("grade", [INSEPARABLE, SEPARABLE, None], ids=["inseparable", "separable", "unknown"])
def test_a_declaration_without_a_reading_released_before_it_earns_no_outcome_under_any_label(ep, grade):
    v = verdict_of(ep, grade=grade)
    assert v["checks"]["provenance"]["ok"], v["checks"]["provenance"]  # every citation is an identifier the episode issued
    assert v["abstained"] and not v["abstention_cites_reading"] and not v["abstention_correct"] and not v["success"]
    assert v["failure_class"] == "abstention_without_reading" and outcome_term(v) == 0.0


def test_a_declaration_that_cites_a_reading_earns_the_credit_only_on_a_start_the_label_marks_inseparable():
    ep = declared_episode()
    correct = verdict_of(ep, grade=INSEPARABLE)
    assert correct["abstention_cites_reading"] and correct["abstention_correct"] and correct["success"] and not correct["pass"]
    assert correct["failure_class"] == "none" and outcome_term(correct) == ABSTENTION_CREDIT == 0.5
    for grade in (SEPARABLE, None):
        v = verdict_of(ep, grade=grade)
        assert v["abstention_cites_reading"] and not v["abstention_correct"] and not v["success"]
        assert v["failure_class"] == "abstention_on_separable" and outcome_term(v) == 0.0
    # no declaration, nothing to cite
    assert not verdict_of(oracle_like(), grade=INSEPARABLE)["abstention_cites_reading"]


def test_a_declaration_earns_the_credit_only_where_the_label_and_e18_both_mark_the_start_inseparable():
    """A declaration that cites a reading is correct, with success and the credit, only on a start that the
    label and E18, the server's first commitment majority, both mark inseparable, and the failure class says which of the two
    judged the start separable, the label first. A correct commitment succeeds whatever both say."""
    ep = declared_episode()
    both = verdict_of(ep, grade=INSEPARABLE)
    assert both["abstention_correct"] and both["success"] and both["failure_class"] == "none" and outcome_term(both) == ABSTENTION_CREDIT
    assert both["separable"] is False and both["separable_e18"] is False
    for grade, judged, failure in ((E18_SEPARABLE, (False, True), "abstention_on_e18_separable"), (LABEL_SEPARABLE, (True, False), "abstention_on_separable"),
                                   (SEPARABLE, (True, True), "abstention_on_separable")):
        v = verdict_of(ep, grade=grade)
        assert (v["separable"], v["separable_e18"]) == judged and v["abstention_cites_reading"] and v["abstained"]
        assert not v["abstention_correct"] and not v["success"] and v["failure_class"] == failure and outcome_term(v) == 0.0
    assert all(verdict_of(oracle_like(), grade=grade)["success"] for grade in (INSEPARABLE, E18_SEPARABLE, LABEL_SEPARABLE, SEPARABLE))


def test_a_grade_without_the_e18_verdict_raises_and_an_unknown_grade_credits_no_declaration():
    """A missing E18 is a failed lookup, as a missing label is, so the verifier refuses a grade without it rather
    than score a declaration under the label alone, while an unknown grade counts as separable under both labels."""
    ep = declared_episode()
    for grade in ({"separable": False, "cost": None}, {**INSEPARABLE, "e18": None}, {**INSEPARABLE, "e18": {"first_share": 0.2}}):
        with pytest.raises(ValueError, match="E18"):
            verdict_of(ep, grade=grade)
    unknown = verdict_of(ep, grade=None)
    assert unknown["separable"] is True and unknown["separable_e18"] is True and unknown["failure_class"] == "abstention_on_separable"


def test_an_unknown_citation_still_fails_provenance_and_earns_no_outcome():
    alone = verdict_of(declaring(read, lambda ep: ep.declare(["obs/99/made_up"])), grade=INSEPARABLE)
    assert alone["checks"]["provenance"]["unmatched"] == ["obs/99/made_up"] and alone["failure_class"] == "unmatched_evidence"
    assert not alone["abstention_cites_reading"] and not alone["abstention_correct"] and outcome_term(alone) == 0.0
    # beside a real reading the declaration stays correct and the success flag stands, and the provenance rule withholds the outcome
    beside = verdict_of(declaring(read, lambda ep: ep.declare([*CITED, "obs/99/made_up"])), grade=INSEPARABLE)
    assert not beside["checks"]["provenance"]["ok"] and beside["failure_class"] == "unmatched_evidence"
    assert beside["abstention_correct"] and beside["success"] and outcome_term(beside) == 0.0


def test_a_closing_summary_that_claims_recovery_or_an_unmade_replacement_contradicts_the_log():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    ep.finish("The cooler was replaced and the rig is restored to normal operation.")
    ep.end()
    v = verdict_of(ep)
    assert v["summary_check"] == {"heuristic": True, "claims_recovery": True, "claimed_replacements": ["cooler"],
                                  "unsupported_replacements": ["cooler"], "claims_no_alarms": False, "alarms_released": False,
                                  "contradicts_log": True}
    assert v["failure_class"] == "summary_contradicts_log"
    # a denial in the sentence is not a claim, so the same failed episode has no class of the taxonomy
    ep.log[-2]["summary"] = "The cooler was not replaced; the rig is not restored."
    v = verdict_of(ep)
    assert v["summary_check"]["contradicts_log"] is False and v["failure_class"] == "none"
    # a true statement about one healthy component is not a claim that the rig recovered
    ep.log[-2]["summary"] = "The valve is healthy. The cooler needs a work order."
    v = verdict_of(ep)
    assert v["summary_check"]["claims_recovery"] is False and v["failure_class"] == "none"


# A closing summary that claims no alarm was triggered contradicts the log when the released change summaries list
# alarms, and the verifier sets contradicts_log.
NO_ALARM_SUMMARY = ("The hydraulic pump condition-monitoring rig, asset hyd_1, has been diagnosed with issues in its cooler and valve components. "
                  "The cooler is experiencing reduced efficiency, with a current level of 3 (critical) and 20 (degraded), while the valve is at "
                  "level 73 (severe degradation). The pump and accumulator are within healthy ranges (pump: 0, accumulator: 90). A work order "
                  "(wo-1) was created to address the cooler's issues, which has since been completed. No alarms were triggered during the "
                  "monitoring period. The budget was fully utilized, and the horizon of 60 cycles was not reached.")


def test_a_closing_summary_that_claims_no_alarm_contradicts_a_released_alarm():
    def episode(alarms):
        ep = Episode(COOLER_FAULTY)
        ep.read(HOT)
        ep.summary(0, alarms=alarms)
        ep.diagnose({"cooler": 20, "valve": 100, "pump": 0, "accumulator": 130}, [f"obs/1/{TEMPERATURE}"])  # a wrong level fails it
        ep.finish(NO_ALARM_SUMMARY)
        ep.end()
        return verdict_of(ep)
    v = episode([{"channel": TEMPERATURE, "first_t": 0}])
    assert v["summary_check"]["claims_no_alarms"] and v["summary_check"]["alarms_released"] and v["summary_check"]["contradicts_log"]
    assert not v["summary_check"]["claims_recovery"] and v["failure_class"] == "summary_contradicts_log"
    quiet = episode([])
    assert quiet["summary_check"]["claims_no_alarms"] and not quiet["summary_check"]["contradicts_log"]
    assert quiet["failure_class"] != "summary_contradicts_log"
    # a sentence about part of the rig is no claim about the episode
    assert not verdict_of(_with_summary(Episode(COOLER_FAULTY), "No alarm was raised on the pump channels."))["summary_check"]["claims_no_alarms"]


def _with_summary(ep, text):
    ep.read(HOT)
    ep.summary(0, alarms=[{"channel": TEMPERATURE, "first_t": 0}])
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    ep.finish(text)
    ep.end()
    return ep


def test_a_diagnosis_is_scored_against_the_hidden_combination_at_the_cycle_of_its_write():
    dropped = dict(zip(COMPONENTS, COOLER_FAULTY), valve=90)

    def episode(diagnosis):
        ep = Episode(COOLER_FAULTY)
        ep.read(HOT)
        ep.advance(5)
        ep.drop("valve", 90)  # the schedule drops the valve before the write
        ep.diagnose(diagnosis, [f"obs/1/{TEMPERATURE}"])
        for c in ("cooler", "valve"):
            ep.order(c)
        ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
        ep.close("wo-1", "cooler")
        ep.close("wo-2", "valve")
        ep.finish("Replaced the cooler and the valve.")
        ep.end()
        return verdict_of(ep, oracle=ep.oracle(work_orders=[("cooler", HORIZON), ("valve", HORIZON)]))
    v = episode(dropped)
    assert v["write_t"] == 6 and v["truth_at_write"] == dropped and v["hidden_start"] == dict(zip(COMPONENTS, COOLER_FAULTY))
    assert v["diagnosis_all_correct"] and v["pass"] and v["turns_to_correct_diagnosis"] == 3
    start = episode(dict(zip(COMPONENTS, COOLER_FAULTY)))  # the start levels, which the drop has made stale
    assert start["diagnosis_per_component"]["valve"] is False and not start["pass"] and start["turns_to_correct_diagnosis"] is None
    assert verdict_of(do_nothing())["write_t"] is None and verdict_of(do_nothing())["truth_at_write"] is None


def test_three_refusals_are_a_repeated_malformed_call():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    for _ in range(3):
        ep.refuse({"type": "read_sensors", "sensor_group": "tier4", "cycles": 1}, "unknown sensor group", cls="read")
    ep.diagnose({"cooler": 20, "valve": 100, "pump": 0, "accumulator": 130}, [f"obs/1/{TEMPERATURE}"])
    ep.finish("The cooler needs attention.")
    ep.end()
    v = verdict_of(ep)
    assert v["refusals"] == 3 and v["steps"] == 6 and v["failure_class"] == "repeated_malformed_call"
    ep.log.pop(2)
    assert verdict_of(ep)["failure_class"] == "none"


def test_anchoring_names_the_components_of_the_first_anomalous_channel_when_the_truth_differs():
    # the first anomaly is a pressure channel, which indicates the valve and the accumulator; the agent calls both faulty
    # although only the accumulator is
    ep = Episode([100, 100, 0, 90])
    ep.read(HIGH_PRESSURE)
    ep.diagnose({"cooler": 100, "valve": 73, "pump": 0, "accumulator": 90}, [f"obs/1/{PRESSURE}"])
    ep.finish("Valve and accumulator faults.")
    ep.end()
    v = verdict_of(ep)
    assert v["first_anomaly"] == {"channel": PRESSURE, "t": 0, "source": "read", "indicates": ["accumulator", "valve"]}
    assert v["failure_class"] == "anchoring_on_first_anomaly"
    # the same first anomaly seen through a change summary instead of a reading
    alarm = Episode([100, 100, 0, 90])
    alarm.summary(0, alarms=[{"channel": PRESSURE, "first_t": 0}])
    alarm.read(INSIDE)
    alarm.diagnose({"cooler": 100, "valve": 73, "pump": 0, "accumulator": 90}, [f"alarm/0/{PRESSURE}"])
    alarm.finish()
    alarm.end()
    v = verdict_of(alarm)
    assert v["first_anomaly"]["source"] == "alarm" and v["failure_class"] == "anchoring_on_first_anomaly"


def test_the_first_anomaly_of_a_reading_is_the_channel_furthest_outside_its_band_whatever_the_names():
    """A reading shows its channels at once, so the anomaly an agent anchors on is the strongest one, the channel furthest
    outside its normal range in normal range widths, and not the one whose name sorts first: here the pressure channel sorts first but stands
    just outside its normal range while the temperature stands two range widths outside, and an agent that calls the cooler alone
    faulty anchored on the temperature."""
    ep = Episode([100, 100, 0, 90])
    ep.read({TEMPERATURE: 30.0, PRESSURE: 10.5})
    ep.diagnose({"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130}, [f"obs/1/{TEMPERATURE}"])
    ep.finish("Cooler fault.")
    ep.end()
    v = verdict_of(ep)
    assert v["first_anomaly"] == {"channel": TEMPERATURE, "t": 0, "source": "read", "indicates": ["cooler"]}
    assert v["failure_class"] == "anchoring_on_first_anomaly"


def test_an_ineffective_close_is_a_repair_before_its_dependency_and_no_repair():
    ep = Episode([3, 80, 0, 130])
    ep.read(HOT)  # the temperature outside its normal range at cycle 0 indicates the cooler
    ep.diagnose(dict(zip(COMPONENTS, [3, 80, 0, 130])), [f"obs/1/{TEMPERATURE}"])
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.ineffective("wo-1", "cooler", ["valve"])  # the verifier reads the hidden event, not the dependency graph
    ep.finish("The cooler was replaced and the rig is restored.")
    ep.end()
    v = verdict_of(ep)
    assert not v["recovered"] and v["repair_delay"]["cooler"] is None and v["summary_check"]["contradicts_log"]
    assert v["failure_class"] == "repair_before_dependency"
    assert "order_ineffective" not in {e["event"] for e in ep.released()["events"]}
    ep.log.insert(3, dict(ep.log[3], id="wo-9", component="pump", unnecessary=True))
    assert verdict_of(ep)["failure_class"] == "unnecessary_replacement", "an order on a healthy component comes first"


def test_an_order_made_ineffective_by_a_drop_after_its_creation_is_not_a_repair_before_its_dependency():
    pump_only = [100, 100, 2, 130]  # the pump faulty, the valve and the accumulator it depends on healthy
    ep = Episode(pump_only)
    ep.read(INSIDE)
    ep.diagnose(dict(zip(COMPONENTS, pump_only)), [f"obs/1/{TEMPERATURE}"])
    ep.order("pump")
    ep.advance(2)
    ep.hazard_drop("valve", 90)  # the stochastic hazard degrades the valve while the pump waits for its technician
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME - 2)
    ep.ineffective("wo-1", "pump", ["valve"])
    ep.finish("The pump order closed.")
    ep.end()
    v = verdict_of(ep)
    assert not v["recovered"] and v["failure_class"] == "none"
    # the valve degraded when the pump was ordered is the agent's error
    early = Episode([100, 90, 2, 130])
    early.read(INSIDE)
    early.diagnose(dict(zip(COMPONENTS, [100, 90, 2, 130])), [f"obs/1/{TEMPERATURE}"])
    early.order("pump")
    early.advance(LEAD_TIME + REPAIR_DOWNTIME)
    early.ineffective("wo-1", "pump", ["valve"])
    early.finish("The pump order closed.")
    early.end()
    assert verdict_of(early)["failure_class"] == "repair_before_dependency"
    # a valve repaired in the same visit and dropped again before the pump closes is not the agent's error either
    again = Episode([100, 90, 2, 130])
    again.read(INSIDE)
    again.diagnose(dict(zip(COMPONENTS, [100, 90, 2, 130])), [f"obs/1/{TEMPERATURE}"])
    again.order("valve")
    again.order("pump")
    again.advance(LEAD_TIME + REPAIR_DOWNTIME)
    again.close("wo-1", "valve")
    again.drop("valve", 90)
    again.advance(1)
    again.ineffective("wo-2", "pump", ["valve"])
    again.finish("The pump order closed.")
    again.end()
    assert verdict_of(again)["failure_class"] == "none"


def test_an_emergency_repair_that_expired_without_a_later_repair_is_a_temporary_repair_expired():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    ep.order("cooler")
    ep.advance(2)
    ep.close("wo-1", "cooler")
    ep.advance(10)
    ep.expire("wo-1", "cooler", COOLER_FAULTY[0])
    ep.finish("The cooler was repaired.")
    ep.end()
    v = verdict_of(ep)
    assert not v["recovered"] and v["failure_class"] == "temporary_repair_expired"
    # a standard repair after the expiry restores the cooler; the wrong diagnosis then fails the episode for another class
    again = Episode(COOLER_FAULTY)
    again.read(HOT)
    again.diagnose({"cooler": 20, "valve": 100, "pump": 0, "accumulator": 130}, [f"obs/1/{TEMPERATURE}"])
    again.order("cooler")
    again.advance(2)
    again.close("wo-1", "cooler")
    again.advance(10)
    again.expire("wo-1", "cooler", COOLER_FAULTY[0])
    again.order("cooler")
    again.advance(LEAD_TIME + REPAIR_DOWNTIME)
    again.close("wo-2", "cooler")
    again.finish("The cooler was repaired twice.")
    again.end()
    v = verdict_of(again)
    assert v["recovered"] and v["failure_class"] == "recovery_without_diagnosis"


def test_recovery_without_a_correct_diagnosis_is_the_last_class():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose({"cooler": 20, "valve": 100, "pump": 0, "accumulator": 130}, [f"obs/1/{TEMPERATURE}"])
    ep.order("cooler", evidence_ids=[f"obs/1/{TEMPERATURE}"])
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")
    ep.finish("Cooler replaced.")
    ep.end()
    v = verdict_of(ep)
    assert v["recovered"] and not v["diagnosis_all_correct"] and v["failure_class"] == "recovery_without_diagnosis"
    assert set(FAILURE_CLASSES) >= {"premature_diagnosis", "unmatched_evidence", "unnecessary_replacement", "repair_before_dependency",
                                    "temporary_repair_expired", "abstention_without_reading", "abstention_on_separable", "abstention_on_e18_separable",
                                    "summary_contradicts_log", "repeated_malformed_call", "anchoring_on_first_anomaly",
                                    "recovery_without_diagnosis"} and len(FAILURE_CLASSES) == 12


# ----- scores with hand computed values -----

def test_detection_and_repair_delays_follow_the_schedule_the_readings_and_the_closures():
    ep = Episode(HEALTHY_LIST)
    ep.advance(4)
    ep.drop("cooler", 20)
    ep.advance(2)
    ep.read(HOT)  # at cycle 6, the first reading outside its normal range; the read takes one cycle
    ep.advance(2)
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "cooler": 20}, [f"obs/1/{TEMPERATURE}"])  # the condition at the write, after the drop
    ep.order("cooler")  # at cycle 9, five cycles after the drop
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")  # at cycle 13, seven cycles after the reading
    ep.finish()
    ep.end()
    v = verify(ep.log, ep.released(), ep.oracle(work_orders=[("cooler", HORIZON)]), 40.0)
    assert v["detection_delay"] == {"cooler": 5, "valve": None, "pump": None, "accumulator": None}
    assert v["repair_delay"] == {"cooler": 9, "valve": None, "pump": None, "accumulator": None}  # from the drop at 4, not the reading at 6
    assert v["diagnosis_all_correct"], "the diagnosis is scored against the condition at its write, after the drop"
    assert v["pass"] and v["failure_class"] == "none"


def test_the_time_to_work_order_runs_from_the_onset_to_the_first_order_at_or_after_it():
    """The time to work order of the paper. The cooler is faulty from the start, so it counts from cycle
    zero. The valve is healthy at the start and the agent orders it before it drops, an order that does not count, so its
    time runs from the scheduled drop to the next order. The pump leaves its healthy level by a hazard drop and counts from
    it, and the order on the accumulator, which never degraded, measures nothing."""
    ep = Episode([20, 100, 0, 130])
    ep.advance(1)
    ep.order("valve")  # cycle 1, on the healthy valve
    ep.advance(2)
    ep.order("cooler")  # cycle 3, three cycles after the start
    ep.advance(1)
    ep.drop("valve", 90)  # cycle 4
    ep.advance(1)
    ep.hazard_drop("pump", 1)  # cycle 5
    ep.advance(1)
    ep.order("pump")  # cycle 6, one cycle after the hazard drop
    ep.advance(3)
    ep.order("valve")  # cycle 9, five cycles after the scheduled drop
    ep.order("accumulator")  # cycle 9, on the healthy accumulator
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    assert v["detection_delay"] == {"cooler": 3, "valve": 5, "pump": 1, "accumulator": None}


def test_a_reading_of_the_history_before_cycle_zero_moves_no_onset_below_it():
    """The AssetOpsBench rows serve recorded cycles before the episode, and such a reading of a fault present at the start
    leaves its repair delay where not reading leaves it, counted from cycle zero."""
    delays = []
    for read_history in (False, True):
        ep = Episode(COOLER_FAULTY)
        if read_history:
            entry = ep.read(HOT, group="tier3")
            entry["served"][0]["t"], entry["t_end"], ep.t = -10, 0, 0  # a recorded cycle before the episode, read at once
        ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
        ep.order("cooler")
        ep.close("wo-1", "cooler")  # the order of a row without a clock closes inside its call
        ep.finish()
        ep.end()
        delays.append(verify(ep.log, ep.released(), ep.oracle(), 40.0)["repair_delay"]["cooler"])
    assert delays == [0, 0]


def test_a_fault_that_begins_during_the_episode_counts_from_its_drop_to_its_restoring_close():
    """The time to repair of the paper runs from the onset of the fault in the hidden record to the close that restores the
    component. The accumulator is faulty from the start and moves the pressure, which also indicates the valve, the agent
    replaces the healthy valve, and the valve drops later. The close on the healthy valve restores nothing, and no reading dates
    the valve's fault, so the delay runs from the drop to the close that restores the valve."""
    ep = Episode([100, 100, 0, 90])
    ep.read(HIGH_PRESSURE, group="tier3")  # cycle 0, the pressure outside its normal range because of the accumulator
    ep.diagnose({"cooler": 100, "valve": 100, "pump": 0, "accumulator": 90}, [f"obs/1/{PRESSURE}"])
    ep.advance(2)
    ep.order("valve")  # cycle 3, on the healthy valve
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "valve")  # cycle 7, a close that restores nothing
    ep.advance(3)
    ep.drop("valve", 90)  # cycle 10, the valve leaves its healthy level
    ep.advance(4)
    ep.read(HIGH_PRESSURE, group="tier3")  # cycle 14, the first reading after the drop
    ep.order("valve")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-2", "valve")  # cycle 19, the close that restores the valve
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    assert v["repair_delay"] == {"cooler": None, "valve": 19 - 10, "pump": None, "accumulator": None}


def test_an_order_placed_before_a_fault_and_closed_after_it_restores_it_from_the_drop():
    """An order placed on the healthy valve closes after the valve dropped, so the close restores the valve: its time to repair
    runs from the drop to the close with a reading after the close or without one, and the order, created before the drop,
    is no work order of that fault."""
    delays = []
    for read_after_the_close in (False, True):
        ep = Episode([100, 100, 0, 90])
        ep.read(HIGH_PRESSURE, group="tier3")  # cycle 0, before the valve drops
        ep.diagnose({"cooler": 100, "valve": 100, "pump": 0, "accumulator": 90}, [f"obs/1/{PRESSURE}"])
        ep.advance(2)
        ep.order("valve")  # cycle 3, on the healthy valve
        ep.advance(2)
        ep.drop("valve", 90)  # cycle 5
        ep.advance(2)
        ep.close("wo-1", "valve")  # cycle 7, the close that restores the valve
        if read_after_the_close:
            ep.read(HIGH_PRESSURE, group="tier3")  # the reading of cycle 7, recorded after the close
        ep.finish()
        ep.end()
        v = verdict_of(ep)
        delays.append((v["repair_delay"]["valve"], v["detection_delay"]["valve"]))
    assert delays == [(7 - 5, None), (7 - 5, None)]


def test_a_drop_in_the_cycle_of_the_close_starts_a_new_occurrence_and_leaves_the_ended_one_its_delay():
    """A scheduled component can drop again in the cycle in which its repair closes. The close ends the first occurrence, whose
    time to repair runs from the first drop, and the second drop begins an occurrence of its own that no close ends."""
    ep = Episode(HEALTHY_LIST)
    ep.read(INSIDE)
    ep.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)))
    ep.advance(2)
    ep.order("valve")  # cycle 3, on the healthy valve
    ep.advance(2)
    ep.drop("valve", 90)  # cycle 5
    ep.advance(2)
    ep.close("wo-1", "valve")  # cycle 7, the close that restores the valve
    ep.drop("valve", 90)  # cycle 7 again, the next drop of the schedule
    ep.read(HIGH_PRESSURE, group="tier3")  # the reading of cycle 7 shows the new fault
    ep.finish()
    ep.end()
    assert verdict_of(ep)["repair_delay"]["valve"] == 7 - 5
    occurrences = [o for o in verifier.fault_occurrences(ep.log, dict(zip(COMPONENTS, HEALTHY_LIST)), verifier._asset_truth(ep.log)) if o[0] == "valve"]
    assert len(occurrences) == 2 and occurrences[1][2] is None


def test_a_reading_taken_without_moving_the_clock_leaves_the_onset_at_the_drop():
    """On a row whose reads take no time a reading carries the cycle of the events that follow it. Readings do not date a fault,
    so the delay runs from the drop at cycle 5 to the close at cycle 9."""
    def read_at_once(ep):
        entry = ep.read(HIGH_PRESSURE, group="tier3")
        entry["t_end"] = ep.t = entry["t_start"]  # a read of zero duration leaves the clock where it stood

    ep = Episode([100, 100, 0, 90])  # the accumulator holds the pressure outside its normal range from the start
    ep.advance(5)
    read_at_once(ep)  # cycle 5, before the valve drops
    ep.drop("valve", 90)  # cycle 5
    ep.diagnose({"cooler": 100, "valve": 90, "pump": 0, "accumulator": 90}, [f"obs/1/{PRESSURE}"])
    ep.order("valve")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    read_at_once(ep)  # cycle 9, before the close
    ep.close("wo-1", "valve")  # cycle 9
    ep.finish()
    ep.end()
    assert verdict_of(ep)["repair_delay"]["valve"] == 9 - 5


def test_cost_adds_prices_and_downtime_and_is_normalized_by_the_oracle_cost():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT, group="tier3", cycles=2)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")
    ep.finish()
    ep.end(downtime=[(0, 2), (1, 2), (2, 5)])
    v = verify(ep.log, ep.released(), ep.oracle(), 40.0)
    prices = 2 * PRICES["read"]["tier3"] + PRICES["parts"]["cooler"] + PRICES["repair_charge"]
    assert (v["cost_prices"], v["penalties"], v["downtime_total"], v["cost"]) == (prices, 0, 9, prices + 9)
    assert v["cost_normalized"] == (prices + 9) / 40.0 and v["cost"] == ep.log[-1]["cost"]
    assert verify(ep.log, ep.released(), ep.oracle(), None)["cost_normalized"] is None
    # a failure event carries its penalty to the hidden ledger, which the budget never pays
    failed = Episode(COOLER_FAULTY)
    failed.advance(10)
    failed.failure("cooler", penalty=50)
    failed.finish()
    failed.end(downtime=[(t, 2) for t in range(10)])
    v = verify(failed.log, failed.released(), failed.oracle(), 40.0)
    assert (v["cost_prices"], v["penalties"], v["downtime_total"], v["cost"]) == (0, 50, 20, 70) and v["failures"] == ["cooler"]
    assert v["cost"] == failed.log[-1]["cost"]


# ----- the reward -----

def as_honest_failure(verdict):
    """The same verdict with the provenance check passed and the episode failed, the honest failure."""
    checks = {**verdict["checks"], "provenance": {**verdict["checks"]["provenance"], "ok": True, "unmatched": []}}
    return {**verdict, "checks": checks, "pass": False, "abstention_correct": False, "success": False}


def test_the_reward_credits_a_correct_declaration_with_a_part_of_the_outcome_term():
    ep = declared_episode()
    reference = verify(oracle_like().log, oracle_like().released(), ep.oracle(), None)["cost"]
    correct = verify(ep.log, ep.released(), ep.oracle(), reference, grade=INSEPARABLE)
    wrong = verify(ep.log, ep.released(), ep.oracle(), reference, grade=SEPARABLE)
    # a declaration leaves the fault in place, so its delay term is full, w_t = 0.1
    cost_term = 0.2 * correct["cost_normalized"] + 0.1
    assert correct["unrestored"] and abs(reward(correct, HORIZON) - (0.5 - cost_term)) < 1e-12
    assert abs(reward(wrong, HORIZON) - (0.0 - cost_term)) < 1e-12
    committed = oracle_like()
    on_inseparable = verify(committed.log, committed.released(), committed.oracle(), reference, grade=INSEPARABLE)
    on_separable = verify(committed.log, committed.released(), committed.oracle(), reference, grade=SEPARABLE)
    # a correct commitment earns the whole outcome term whatever the label says
    assert abs(reward(on_separable, HORIZON) - reward(on_inseparable, HORIZON)) < 1e-12
    assert outcome_term(on_inseparable) == outcome_term(on_separable) == 1.0


def test_the_declaration_credit_switch_of_the_reward_scores_a_correct_declaration_as_a_failure_and_leaves_a_pass_alone():
    """The online loop of GEPA scores its stream with the declaration credit set to zero, pass
    alone in the outcome term, through the one switch of the reward, while the default keeps ABSTENTION_CREDIT."""
    ep = declared_episode()
    reference = verify(oracle_like().log, oracle_like().released(), ep.oracle(), None)["cost"]
    correct = verify(ep.log, ep.released(), ep.oracle(), reference, grade=INSEPARABLE)
    cost_term = 0.2 * correct["cost_normalized"] + 0.1  # the full delay term of a fault left in place
    assert abs(reward(correct, HORIZON, declaration_credit=0.0) - (0.0 - cost_term)) < 1e-12
    assert abs(reward(correct, HORIZON) - (ABSTENTION_CREDIT - cost_term)) < 1e-12
    assert reward(correct, HORIZON, w_c=0.0, w_t=0.0, declaration_credit=0.0) == 0.0 and outcome_term(correct) == ABSTENTION_CREDIT
    committed = oracle_like()
    passed = verify(committed.log, committed.released(), committed.oracle(), reference)
    assert reward(passed, HORIZON, declaration_credit=0.0) == reward(passed, HORIZON)


def test_on_one_start_a_repair_earns_more_than_a_declaration_after_reading_and_that_more_than_a_wrong_answer():
    repair = oracle_like()
    reference = verify(repair.log, repair.released(), repair.oracle(), None)["cost"]
    wrong = Episode(COOLER_FAULTY)
    wrong.read(HOT)
    wrong.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)), CITED)
    wrong.finish("Nothing is wrong.")
    wrong.end()
    rewards = [reward(verify(ep.log, ep.released(), ep.oracle(), reference, grade=INSEPARABLE), HORIZON) for ep in (repair, declared_episode(), wrong)]
    assert rewards[0] > rewards[1] > rewards[2]


def test_the_reward_orders_the_oracle_above_do_nothing_and_gates_on_provenance():
    oracle = oracle_like()
    reference = verify(oracle.log, oracle.released(), oracle.oracle(), None)["cost"]
    r_oracle = reward(verify(oracle.log, oracle.released(), oracle.oracle(), reference), HORIZON)
    idle = do_nothing()
    r_idle = reward(verify(idle.log, idle.released(), idle.oracle(), reference), HORIZON)
    # the oracle like episode reads at cycle 0 and its order closes five cycles later, which is its repair delay
    assert abs(r_oracle - (1.0 - 0.2 * 1.0 - 0.1 * 5 / HORIZON)) < 1e-12 and r_idle < 0 < r_oracle
    fabricated = oracle_like()
    fabricated.log[2]["evidence_ids"] = ["obs/99/made_up"]
    # the made up identifier costs the outcome term and leaves the cost and delay terms as they were
    r_fabricated = reward(verify(fabricated.log, fabricated.released(), fabricated.oracle(), reference), HORIZON)
    assert abs(r_fabricated - (r_oracle - 1.0)) < 1e-12
    assert reward({"verified": False, "error": "the log has no reset event"}, HORIZON) is None


def test_a_provenance_failure_scores_as_a_failed_episode_never_above_an_honest_failure():
    """A cited identifier that the released evidence does not hold scores the episode as a failure, the
    outcome term 0 with the cost and delay terms kept. The log never scores above the same log as an honest failure,
    with nonzero cost terms it scores below zero, the reward of the earlier rule, and the failure class, the success flag
    and the provenance check stay as they were."""
    honest = oracle_like()
    reference = verify(honest.log, honest.released(), honest.oracle(), None)["cost"]
    fabricated = oracle_like()
    fabricated.log[2]["evidence_ids"] = ["obs/99/made_up"]
    v = verify(fabricated.log, fabricated.released(), fabricated.oracle(), reference)
    assert v["checks"]["provenance"]["unmatched"] == ["obs/99/made_up"] and v["failure_class"] == "unmatched_evidence"
    assert v["pass"] and v["success"]
    failed = as_honest_failure(v)
    assert reward(v, HORIZON) <= reward(failed, HORIZON) and abs(reward(v, HORIZON) - reward(failed, HORIZON)) < 1e-12
    cost_terms = 0.2 * v["cost_normalized"] + 0.1 * 5 / HORIZON  # the order closes five cycles after the reading
    assert cost_terms > 0 and abs(reward(v, HORIZON) + cost_terms) < 1e-12
    # without cost and delay weights both score zero, so only the outcome term separates the honest log from the made up one
    assert reward(v, HORIZON, w_c=0.0, w_t=0.0) == reward(failed, HORIZON, w_c=0.0, w_t=0.0) == 0.0
    r_honest = reward(verify(honest.log, honest.released(), honest.oracle(), reference), HORIZON)
    assert abs(r_honest - reward(v, HORIZON) - 1.0) < 1e-12


def test_the_reward_formula_with_the_declared_weights_by_hand():
    ep = Episode(HEALTHY_LIST)
    ep.advance(4)
    ep.drop("cooler", 20)
    ep.read(HOT)  # the reading at cycle 4 leaves its normal range; the read takes one cycle
    ep.diagnose({**dict(zip(COMPONENTS, HEALTHY_LIST)), "cooler": 20}, [f"obs/1/{TEMPERATURE}"])  # the condition at the write
    ep.advance(1)
    ep.order("cooler")  # at cycle 6, two cycles after the drop
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")  # at cycle 10, six cycles after the reading
    ep.finish()
    ep.end()
    v = verify(ep.log, ep.released(), ep.oracle(work_orders=[("cooler", HORIZON)]), v_cost := 41.0)
    assert v["pass"] and (v["detection_delay"]["cooler"], v["repair_delay"]["cooler"]) == (2, 6)
    expected = 1.0 - 0.2 * (41.0 / v_cost) - 0.1 * 6 / HORIZON
    assert abs(reward(v, HORIZON) - expected) < 1e-12
    assert abs(reward(v, HORIZON, w_c=0.3, w_t=0.0) - (1.0 - 0.3)) < 1e-12  # 0.3 * COST_CAP stays below the gap


def test_pass_power_k_is_the_estimator_of_tau_bench():
    # one task solved three times in four trials: C(3, 2) / C(4, 2) = 3 / 6
    assert pass_power_k(3, 4, 2) == 0.5
    assert pass_power_k(4, 4, 4) == 1.0 and pass_power_k(0, 4, 1) == 0.0 and pass_power_k(2, 4, 1) == 0.5
    # two tasks: the mean of C(3, 2) / C(4, 2) and C(4, 2) / C(4, 2)
    assert pass_power_k([3, 4], 4, 2) == (0.5 + 1.0) / 2
    for bad in ((5, 4, 2), (3, 4, 0), (3, 4, 5), ([], 4, 1)):
        with pytest.raises(ValueError):
            pass_power_k(*bad)


# ----- the verifier scope and the unverifiable cases -----

def test_scope_all_holds_the_sensor_reads_per_group_against_the_greedy_policy():
    ep = oracle_like()
    assert read_counts(ep.log) == {"tier1": 1}
    # the listing calls of the AssetOpsBench rows, a read that released nothing and a wait are no sensor reads
    assert read_counts([{"event": "aob_read"}, {"event": "advance"}, {"event": "read", "sensor_group": "tier1", "served": []}, {"event": "work_order_created"}]) == {}
    assert read_counts([{"event": "read", "sensor_group": "tier1", "served": [{}, {}, {}]}]) == {"tier1": 1}  # a read of three cycles counts once
    writes = verify(ep.log, ep.released(), ep.oracle(), 40.0, scope="writes")
    above = verify(ep.log, ep.released(), ep.oracle(reads={"tier3": 5}), 40.0, scope="all")
    assert writes["checks"]["count_gate"]["ok"] and writes["read_gate"] is None and writes["checks"]["count_gate"]["reference_reads"] is None
    assert not above["checks"]["count_gate"]["ok"] and above["read_gate"] is False and not above["all_checks_ok"]
    assert above["checks"]["count_gate"]["problems"] == ["sensor reads above the greedy policy's in tier1"]
    assert above["pass"] == writes["pass"], "the scope changes the gate, not the outcome"
    matched = verify(ep.log, ep.released(), ep.oracle(reads=read_counts(ep.log)), 40.0, scope="all")
    assert matched["checks"]["count_gate"]["ok"] and matched["all_checks_ok"] and matched["read_gate"] is True
    # an agent that reads less than the greedy policy in every group holds the gate, one that reads more in one group fails it
    less = verify(ep.log, ep.released(), ep.oracle(reads={"tier1": 3, "tier3": 1}), 40.0, scope="all")
    assert less["read_gate"] is True and less["checks"]["count_gate"]["ok"]
    twice = Episode(COOLER_FAULTY)
    twice.read(HOT)
    twice.read(HOT)
    twice.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    twice.finish("Two reads.")
    twice.end()
    more = verify(twice.log, twice.released(), twice.oracle(reads={"tier1": 1}), 40.0, scope="all")
    assert read_counts(twice.log) == {"tier1": 2} and more["read_gate"] is False
    with pytest.raises(ValueError):
        verify(ep.log, ep.released(), ep.oracle(), 40.0, scope="reads")


def test_a_public_log_a_mismatched_oracle_and_a_wrong_released_format_are_unverifiable():
    ep = oracle_like()
    public = [{k: v for k, v in e.items() if k not in ("hidden", "unnecessary")} for e in ep.log]
    assert verify(public, ep.released(), ep.oracle(), 40.0) == {"verified": False, "error": "the log lacks its hidden fields; the verifier needs env.log, not the public view", "scope": "writes"}
    other_start = dict(ep.oracle(), diagnosis=dict(zip(COMPONENTS, TWO_FAULTY)))
    assert verify(ep.log, ep.released(), other_start, 40.0)["verified"] is False
    assert verify(ep.log, dict(ep.released(), format=1), ep.oracle(), 40.0)["verified"] is False
    assert verify([], ep.released(), ep.oracle(), 40.0)["verified"] is False
    assert reward(verify([], ep.released(), ep.oracle(), 40.0), HORIZON) is None


# ----- the real environment, when it is present -----

def _environment():
    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    try:
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
        from CausalDiagnosis.environment.reference_policies.policies import DoNothingPolicy, OraclePolicy, run_episode
    except ImportError as exc:
        pytest.skip(f"the v2 environment is not available yet: {exc}")
    return EpisodeConfig, HydraulicRigEnv, DoNothingPolicy, OraclePolicy, run_episode


def test_on_the_environment_the_oracle_policy_passes_and_outranks_do_nothing():
    EpisodeConfig, HydraulicRigEnv, DoNothingPolicy, OraclePolicy, run_episode = _environment()
    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    config = EpisodeConfig()
    for seed in range(3):
        env = HydraulicRigEnv(cycles, blocks, config)
        oracle_log = run_episode(env, OraclePolicy(env), seed)
        oracle, released = env.oracle(), released_from_env(env)
        reference = verify(oracle_log, released, oracle, None)["cost"]
        best = verify(oracle_log, released, oracle, reference)
        assert best["verified"] and best["pass"] and best["all_checks_ok"] and best["failure_class"] == "none", best
        assert best["cost"] == oracle_log[-1]["cost"], "the verifier's cost equals the budget the environment charged"
        idle_env = HydraulicRigEnv(cycles, blocks, config)
        idle_log = run_episode(idle_env, DoNothingPolicy(), seed)
        idle = verify(idle_log, released_from_env(idle_env), idle_env.oracle(), reference)
        assert idle["verified"] and not idle["pass"]
        assert reward(best, config.horizon) > reward(idle, config.horizon)
        assert json.loads(json.dumps(best)) == best


def _free(EpisodeConfig):
    """The prices of the free base row (CausalDiagnosis/configs/layer_presets/base.json): every read, part, repair and visit at 0."""
    prices = EpisodeConfig().prices
    return {**prices, "read": {g: 0 for g in prices["read"]}, "parts": {c: 0 for c in prices["parts"]}, "repair_charge": 0, "visit": 0}


def test_the_repair_delay_of_a_fault_present_at_the_start_counts_from_cycle_zero():
    EpisodeConfig, HydraulicRigEnv, *_rest = _environment()
    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    start = [3, 90, 0, 130]  # the cooler and the valve off their healthy level from the first cycle
    delays = []
    for first_read in (False, True):
        env = HydraulicRigEnv(cycles, blocks, EpisodeConfig(start={"combination": start, "entry_offset": 0}))
        env.reset(0)
        if first_read:
            env.step({"type": "read_sensors", "sensor_group": "tier3", "cycles": 2})
        env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENTS, start)), "evidence_ids": []})
        for component in ("cooler", "valve"):
            env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})
        while len([e for e in env.log if e["event"] == "work_order_closed"]) < 2:
            env.step({"type": "wait_for_notification", "timeout": 10})
        env.step({"type": "finish", "summary": "done"})
        v = verify(env.log, released_from_env(env), env.oracle(), None)
        closes = {e["component"]: e["t_end"] for e in env.log if e["event"] == "work_order_closed"}
        # without a reading the delay still counts from cycle zero, so not reading never shortens it
        assert v["repair_delay"] == {"cooler": closes["cooler"], "valve": closes["valve"], "pump": None, "accumulator": None}
        delays.append(v["repair_delay"]["cooler"])
    assert delays[0] > 0 and delays[1] > delays[0]  # the read took its cycles, and the delay counted them


def _seed_15_rule_only_episode():
    """The episode of seed 15 on the real-time clock row with the decisions of the rule-only CausalDiagnosis agent: the
    cooler and the accumulator are faulty from the start, the valve stays healthy, and the agent reads the three groups,
    diagnoses the start and replaces all three. Returns the environment, its released evidence and its verdict."""
    EpisodeConfig, HydraulicRigEnv, *_rest = _environment()
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers

    preset = CONFIGS / "layer_presets" / "real_time.json"
    cycles = load_cycles(default_facts_path())
    env = HydraulicRigEnv(cycles, build_blocks(cycles), layers.compose(json.loads(preset.read_text(encoding="utf-8"))))
    env.reset(15)
    start = dict(zip(COMPONENTS, env.log[0]["hidden"]))
    assert start == {"cooler": 20, "valve": 100, "pump": 0, "accumulator": 100}
    for group in ("tier1", "tier2", "tier3"):
        env.step({"type": "read_sensors", "sensor_group": group, "cycles": 1}, think_cycles=1)
    env.step({"type": "submit_diagnosis", "diagnosis": start, "evidence_ids": []}, think_cycles=1)
    for component in ("cooler", "accumulator", "valve"):
        env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []}, think_cycles=1)
    env.step({"type": "finish", "summary": "done"}, think_cycles=1)
    released = released_from_env(env)
    return env, released, verify(env.log, released, env.oracle(), None)


def test_a_component_that_never_left_its_healthy_level_carries_no_repair_delay():
    """On the seed 15 episode the reads show channels that indicate the valve outside their normal ranges because of the other
    two faults. The order on the healthy valve restores nothing, so the valve carries no repair delay, while the two
    faults count from cycle zero."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.datasets.dataset_loader.ground_truth import LevelsTruth
    env, released, v = _seed_15_rule_only_episode()
    bands, truth = released["rules"]["normal_ranges"], LevelsTruth(get_profile("hyd_1", engine=False))
    assert any("valve" in truth.indicated(field) and field in bands and not bands[field]["low"] <= value <= bands[field]["high"]
               for e in released["events"] if e["event"] == "read" for served in e["served"] for field, value in served["values"].items())
    closes = {e["component"]: e["t_end"] for e in env.log if e["event"] == "work_order_closed"}
    assert v["unnecessary"] == 1 and set(closes) == {"cooler", "accumulator", "valve"}
    assert v["repair_delay"] == {"cooler": closes["cooler"], "valve": None, "pump": None, "accumulator": closes["accumulator"]}


def test_on_seed_15_a_fault_present_at_the_start_has_its_time_to_work_order_from_cycle_zero():
    """On the seed 15 episode the time to work order of the
    cooler and the accumulator, faulty from the start with no scheduled drop, runs from cycle zero to the agent's first
    order on each, and the valve, which never left its healthy level, has none although it got an order."""
    env, _released, v = _seed_15_rule_only_episode()
    first_order = {}
    for e in env.log:
        if e["event"] == "work_order_created":
            first_order.setdefault(e["component"], e["t_start"])
    assert set(first_order) == {"cooler", "accumulator", "valve"} and first_order["cooler"] > 0
    assert v["detection_delay"] == {"cooler": first_order["cooler"], "valve": None, "pump": None,
                                    "accumulator": first_order["accumulator"]}


def test_on_the_free_base_row_every_cost_counts_calls_so_the_reward_is_defined():
    EpisodeConfig, HydraulicRigEnv, DoNothingPolicy, OraclePolicy, run_episode = _environment()
    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    free = EpisodeConfig(prices=_free(EpisodeConfig), budget=None)
    env = HydraulicRigEnv(cycles, blocks, free)
    oracle_log = run_episode(env, OraclePolicy(env), 0)
    oracle, released = env.oracle(), released_from_env(env)
    reference = verify(oracle_log, released, oracle, None)
    calls = len({e["step"] for e in oracle_log if e.get("step") is not None})
    assert (reference["cost_unit"], reference["cost"]) == ("calls", calls) and calls > 0 and reference["cost_prices"] == 0
    best = verify(oracle_log, released, oracle, reference["cost"])
    assert best["cost_normalized"] == 1.0 and reward(best, free.horizon) is not None
    idle_env = HydraulicRigEnv(cycles, blocks, free)
    idle = verify(run_episode(idle_env, DoNothingPolicy(), 0), released_from_env(idle_env), idle_env.oracle(), reference["cost"])
    assert idle["cost_unit"] == "calls" and reward(best, free.horizon) > reward(idle, free.horizon)
    priced_env = HydraulicRigEnv(cycles, blocks, EpisodeConfig())
    priced = verify(run_episode(priced_env, OraclePolicy(priced_env), 0), released_from_env(priced_env), priced_env.oracle(), None)
    assert priced["cost_unit"] == "money" and priced["cost"] == priced_env.log[-1]["cost"]


def test_a_synthetic_end_entry_with_a_hold_scores_the_component_at_its_level_before_the_repair():
    """A hold in force at the end counts the component at the level the hold replaced: the episode is not recovered, the
    outcome check names the held component and the class is temporary_repair_expired. The same episode without the hold
    passes, and its outcome check holds an empty list."""
    def played(held):
        ep = Episode(COOLER_FAULTY)
        ep.read(HOT)
        ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
        ep.order("cooler", evidence_ids=[f"obs/1/{TEMPERATURE}"])
        ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
        ep.close("wo-1", "cooler")
        ep.finish("The cooler was repaired.")
        ep.end(held=held)
        return verdict_of(ep)

    held = played({"cooler": COOLER_FAULTY[0]})
    assert not held["recovered"] and not held["pass"] and held["checks"]["outcome"]["held"] == ["cooler"]
    assert held["failure_class"] == "temporary_repair_expired" and held["final"] == dict(zip(COMPONENTS, HEALTHY_LIST))
    plain = played(None)
    assert plain["recovered"] and plain["pass"] and plain["checks"]["outcome"]["held"] == [] and plain["failure_class"] == "none"


def test_an_episode_without_a_commitment_has_not_committed_on_an_inseparable_start():
    """commitment_on_inseparable says that the agent committed on a start that both labels mark inseparable, so an episode
    with neither a diagnosis nor a declaration has it false, a diagnosis on the same start has it true, and a declaration,
    which commits to no combination, has it false."""
    inseparable = {"separable": False, "e18": {"separable": False}}
    assert verdict_of(do_nothing(), grade=inseparable)["commitment_on_inseparable"] is False
    assert verdict_of(oracle_like(), grade=inseparable)["commitment_on_inseparable"] is True
    declared = Episode(COOLER_FAULTY)
    declared.read(HOT)
    declared.declare([f"obs/1/{TEMPERATURE}"])
    declared.finish("The evidence cannot separate the candidates.")
    declared.end()
    assert verdict_of(declared, grade=inseparable)["commitment_on_inseparable"] is False


def test_a_read_that_served_nothing_is_no_reading_before_the_diagnosis():
    """A read counts as a reading only when it served one, so a read whose cycles the horizon or a repair left empty, then a
    diagnosis, is a premature diagnosis."""
    ep = Episode(COOLER_FAULTY)
    ep._call("read", "read", 0, 0, sensor_group="tier1", cycles=1, served=[])
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    ep.finish("Nothing was read.")
    ep.end()
    assert verdict_of(ep)["failure_class"] == "premature_diagnosis"


def test_a_log_whose_reset_entry_holds_no_prices_is_not_verified():
    """The cost unit of a row comes from the prices of the reset entry, which every log of the environment holds, so a log
    without them is refused as unverifiable rather than scored in a unit guessed for it."""
    ep = oracle_like()
    del ep.log[0]["config"]["prices"]
    verdict = verify(ep.log, ep.released(), ep.oracle(), None)
    assert not verdict["verified"] and "holds no prices" in verdict["error"]


def test_a_log_without_an_end_event_is_not_verified():
    """An episode is verified once it has ended, by the environment or by a finish, so the outcome always reads the levels
    of an end state: a log whose last entries hold no end event, as a runner that raised leaves it, is unverifiable and earns
    no reward, and the same log with its end event is verified."""
    ep = oracle_like()
    ended = verify(ep.log, ep.released(), ep.oracle(), None)
    assert ended["verified"]
    open_log = [e for e in ep.log if e["event"] != "end"]
    verdict = verify(open_log, ep.released(), ep.oracle(), None)
    assert not verdict["verified"] and "no end event" in verdict["error"] and reward(verdict, HORIZON) is None


def test_a_good_episode_is_a_pass_whose_cited_evidence_the_record_holds_and_the_reward_reads_that_one_predicate():
    """One definition of a good episode for the reward and the filter of the training data: a pass with a made-up identifier is not
    good and earns no outcome term, while pass itself leaves provenance out."""
    verdict = {"verified": True, "pass": True, "abstention_correct": False, "cost_normalized": 1.0, "checks": {"provenance": {"ok": True}},
               "detection_delay": {}, "repair_delay": {}, "unrestored": []}
    made_up = {**verdict, "checks": {"provenance": {"ok": False}}}
    assert good(verdict) is True and good(made_up) is False and good({**verdict, "pass": False}) is False and good({}) is False
    assert reward(verdict, 10) == 1.0 - 0.2 * 1.0 and reward(made_up, 10) == 0.0 - 0.2 * 1.0
    declared = {**verdict, "pass": False, "abstention_correct": True}
    assert reward(declared, 10) == ABSTENTION_CREDIT - 0.2 and reward({**declared, "checks": {"provenance": {"ok": False}}}, 10) == -0.2


# ----- one onset per fault occurrence: the time to work order, the time to repair and the declared time read it -----

def _cycle(log, event, nth=0, **fields):
    """The t_start of the nth log entry of the event whose fields match."""
    return [e for e in log if e["event"] == event and all(e.get(k) == v for k, v in fields.items())][nth]["t_start"]


def test_an_unread_fault_that_began_in_the_episode_counts_its_time_to_repair_from_the_drop():
    """No reading of the agent shows the valve's fault, and the repair delay still runs from the hidden drop to the close."""
    ep = Episode(HEALTHY_LIST)
    ep.read(INSIDE)
    ep.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)))
    ep.advance(2)
    ep.drop("valve", 90)
    ep.advance(1)
    ep.order("valve")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "valve")
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    onset = _cycle(ep.log, "scheduled_drop", component="valve")
    assert v["repair_delay"]["valve"] == _cycle(ep.log, "work_order_closed", component="valve") - onset
    assert v["detection_delay"]["valve"] == _cycle(ep.log, "work_order_created", component="valve") - onset


def test_an_expired_emergency_repair_starts_a_new_onset_for_both_delays():
    """The cooler is restored, the emergency repair expires, and the agent orders it again late: the second occurrence runs
    from the expiry in both measures, and each field holds the larger of the two occurrences."""
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    ep.order("cooler")
    ep.advance(2)
    ep.close("wo-1", "cooler")
    ep.advance(4)
    ep.expire("wo-1", "cooler", COOLER_FAULTY[0])
    ep.advance(9)
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-2", "cooler")
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    expiry = _cycle(ep.log, "emergency_expired")
    first_order, second_order = _cycle(ep.log, "work_order_created"), _cycle(ep.log, "work_order_created", 1)
    first_close, second_close = _cycle(ep.log, "work_order_closed"), _cycle(ep.log, "work_order_closed", 1)
    assert v["detection_delay"]["cooler"] == max(first_order - 0, second_order - expiry) == second_order - expiry
    assert v["repair_delay"]["cooler"] == max(first_close - 0, second_close - expiry) == second_close - expiry


def test_a_second_fault_after_a_restoring_close_counts_from_its_own_onset():
    """The cooler drops, is restored, drops again and is ordered late: the second occurrence runs from the second drop, and an
    order placed before that drop counts for no occurrence of it."""
    ep = Episode(HEALTHY_LIST)
    ep.read(INSIDE)
    ep.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)))
    ep.advance(1)
    ep.drop("cooler", 20)
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")
    ep.advance(3)
    ep.drop("cooler", 20)
    ep.advance(8)
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-2", "cooler")
    ep.finish()
    ep.end()
    v = verdict_of(ep)
    second_drop = _cycle(ep.log, "scheduled_drop", 1)
    assert v["detection_delay"]["cooler"] == _cycle(ep.log, "work_order_created", 1) - second_drop
    assert v["repair_delay"]["cooler"] == _cycle(ep.log, "work_order_closed", 1) - second_drop
