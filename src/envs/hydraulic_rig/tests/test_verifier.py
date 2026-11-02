"""The verifier on synthetic logs that follow the log schema of the environment entry by entry: one test per check and
per failure class, the reward order, pass^k against the estimator's formula, and the scope "all" against "writes". The
last test plays the oracle policy and the do nothing policy on the real environment when it is present."""
import json
from pathlib import Path

import pytest

from envs.hydraulic_rig import knowledge
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.verifier import FAILURE_CLASSES, pass_power_k, read_counts, released_from_env, reward, verify

HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]
# A start with the cooler at its worst recorded level and every other component healthy (design levels, not data).
COOLER_FAULTY = [3, 100, 0, 130]
# A start with the cooler and the accumulator faulty.
TWO_FAULTY = [3, 100, 0, 90]
TEMPERATURE = "TS1_Temperature_C_1Hz_mean"  # indicates the cooler
PRESSURE = "PS1_Pressure_bar_100Hz_mean"  # indicates the valve and the accumulator
BANDS = {TEMPERATURE: {"low": 0.0, "high": 10.0}, PRESSURE: {"low": 0.0, "high": 10.0}}
INSIDE = {TEMPERATURE: 5.0, PRESSURE: 5.0}
HOT = {TEMPERATURE: 20.0, PRESSURE: 5.0}  # the temperature outside its band
HIGH_PRESSURE = {TEMPERATURE: 5.0, PRESSURE: 20.0}
PRICES = {"read": {"tier1": 1, "tier2": 3, "tier3": 10}, "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20},
          "repair_charge": 20}
LEAD_TIME, REPAIR_DOWNTIME, HORIZON = 3, 1, 60
TASK_TEXT = "Find the condition of every component of hyd_1 and restore the rig."


class Episode:
    """Builds a log entry by entry as the environment writes it (plan section 3), with the hidden combination followed
    through the system events, and derives the released evidence (format 2) and the oracle from the same entries."""

    def __init__(self, hidden, seed=0):
        self.t = 0
        self.step = 0
        self.n = 0
        self.orders = 0
        self.current = dict(zip(COMPONENTS, hidden))
        self.log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "class": "system", "cost": 0, "seed": seed,
                     "hidden": list(hidden), "schedule": {"mode": "none"}, "external_events": {"mode": "none"},
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

    def close(self, order_id, component, source="agent"):
        self._system("work_order_executed", id=order_id)
        self.current[component] = HEALTHY[component]
        return self._system("work_order_closed", id=order_id, component=component, source=source, combination_after=list(self.current.values()))

    def external(self, component):
        self._system("external_replacement", component=component, work_order_id="wo-ext-1")
        return self.close("wo-ext-1", component, source="external")

    def failure(self, component, penalty=50):
        return self._system("failure", component=component, cost=penalty)

    def end(self, reason="finish", downtime=()):
        downtime = [{"t": t, "charge": charge} for t, charge in downtime]
        total = sum(d["charge"] for d in downtime)
        prices = sum(e["cost"] for e in self.log)
        return self._system("end", reason=reason, time=self.t, final=list(self.current.values()), cost=prices + total,
                            downtime_total=total, downtime_by_cycle=downtime)

    # the derived files
    def released(self, bands=BANDS):
        """The public view the action server releases: the entries after reset without the hidden fields, without the
        scheduled drops (the schedule is hidden) and without the component of a failure."""
        hidden_keys = {"cycle", "combination", "combination_after", "unnecessary", "hidden", "schedule", "external_events", "final"}
        events = []
        for e in self.log[1:]:
            if e["event"] == "scheduled_drop":
                continue
            public = {k: v for k, v in e.items() if k not in hidden_keys and not (e["event"] == "failure" and k == "component")}
            if e["event"] == "read":
                public["served"] = [{k: v for k, v in s.items() if k not in hidden_keys} for s in e["served"]]
            events.append(public)
        observations = [{"n": s["n"], "t": s["t"], "sensor_group": e["sensor_group"], "values": s["values"]}
                        for e in events if e["event"] == "read" for s in e["served"]]
        return {"format": 2, "asset_id": "hyd_1", "task_text": TASK_TEXT, "guidance_level": 1,
                "rules": {"components": list(COMPONENTS), "healthy": dict(HEALTHY), "prices": PRICES, "horizon": HORIZON,
                          "healthy_bands": bands},
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


def test_an_unfinished_log_is_verified_from_the_system_events_and_fails_the_count_gate():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)))
    ep.order("cooler")
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")
    v = verdict_of(ep)
    assert v["verified"] and v["recovered"] and v["end_reason"] is None and v["time_used"] == 1 + LEAD_TIME + REPAIR_DOWNTIME
    assert not v["checks"]["count_gate"]["ok"] and v["pass"]


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
    assert v["failure_class"] == "unmatched_evidence" and reward(v, HORIZON) == 0.0
    assert v["prediction_verdict"] == [{"work_order": "wo-1", "prediction_id": "pred/1", "found": True, "predicted_recovery": 0.5,
                                        "recovery_error": 0.5, "verification": None}]
    without_predictions = verify(ep.log, ep.released(), ep.oracle(), 10.0)
    assert without_predictions["checks"]["provenance"]["unmatched"] == ["pred/1", "wo/wo-0"]
    # a code of a component whose failure modes were not requested is not released evidence
    ep.log[4]["evidence_ids"].append(f"fmea/{knowledge.FMEA_SOURCE}/pump_internal_leakage_severe")
    assert f"fmea/{knowledge.FMEA_SOURCE}/pump_internal_leakage_severe" in verify(ep.log, ep.released(), ep.oracle(), 10.0)["checks"]["provenance"]["unmatched"]


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


SEPARABLE, INSEPARABLE = {"separable": True, "cost": 12}, {"separable": False, "cost": None}


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


def test_success_follows_the_tri_state_grade_and_pass_stays_equation_two():
    committed = oracle_like()
    for grade in (SEPARABLE, None):
        v = verdict_of(committed, grade=grade)
        assert v["pass"] and v["success"] and not v["commitment_on_inseparable"] and v["failure_class"] == "none"
    v = verdict_of(committed, grade=INSEPARABLE)
    assert v["pass"] and not v["success"] and v["commitment_on_inseparable"] and v["failure_class"] == "none"
    assert v["separable"] is False and not v["abstained"] and not v["abstention_correct"]
    declared = verdict_of(declared_episode(), grade=INSEPARABLE)
    assert declared["success"] and not declared["pass"] and not declared["commitment_on_inseparable"]


def test_a_closing_summary_that_claims_recovery_or_an_unmade_replacement_contradicts_the_log():
    ep = Episode(COOLER_FAULTY)
    ep.read(HOT)
    ep.diagnose(dict(zip(COMPONENTS, COOLER_FAULTY)), [f"obs/1/{TEMPERATURE}"])
    ep.finish("The cooler was replaced and the rig is restored to normal operation.")
    ep.end()
    v = verdict_of(ep)
    assert v["summary_check"] == {"heuristic": True, "claims_recovery": True, "claimed_replacements": ["cooler"],
                                  "unsupported_replacements": ["cooler"], "contradicts_log": True}
    assert v["failure_class"] == "summary_contradicts_log"
    # a denial in the sentence is not a claim, so the same failed episode has no class of the taxonomy
    ep.log[-2]["summary"] = "The cooler was not replaced; the rig is not restored."
    v = verdict_of(ep)
    assert v["summary_check"]["contradicts_log"] is False and v["failure_class"] == "none"
    # a true statement about one healthy component is not a claim that the rig recovered
    ep.log[-2]["summary"] = "The valve is healthy. The cooler needs a work order."
    v = verdict_of(ep)
    assert v["summary_check"]["claims_recovery"] is False and v["failure_class"] == "none"


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
    assert set(FAILURE_CLASSES) >= {"premature_diagnosis", "unmatched_evidence", "unnecessary_replacement", "abstention_on_separable",
                                    "summary_contradicts_log", "repeated_malformed_call", "anchoring_on_first_anomaly",
                                    "recovery_without_diagnosis"} and len(FAILURE_CLASSES) == 8


# ----- scores with hand computed values -----

def test_detection_and_repair_delays_follow_the_schedule_the_readings_and_the_closures():
    ep = Episode(HEALTHY_LIST)
    ep.advance(4)
    ep.drop("cooler", 20)
    ep.advance(2)
    ep.read(HOT)  # at cycle 6, the first reading outside its band; the read takes one cycle
    ep.advance(2)
    ep.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)), [f"obs/1/{TEMPERATURE}"])  # the start, which the schedule changed later
    ep.order("cooler")  # at cycle 9, five cycles after the drop
    ep.advance(LEAD_TIME + REPAIR_DOWNTIME)
    ep.close("wo-1", "cooler")  # at cycle 13, seven cycles after the reading
    ep.finish()
    ep.end()
    v = verify(ep.log, ep.released(), ep.oracle(work_orders=[("cooler", HORIZON)]), 40.0)
    assert v["detection_delay"] == {"cooler": 5, "valve": None, "pump": None, "accumulator": None}
    assert v["repair_delay"] == {"cooler": 7, "valve": None, "pump": None, "accumulator": None}
    assert v["diagnosis_all_correct"], "the diagnosis is scored against the start, which was healthy"
    assert v["pass"] and v["failure_class"] == "none"


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
    # a failure event carries its penalty, which the budget pays like a price
    failed = Episode(COOLER_FAULTY)
    failed.advance(10)
    failed.failure("cooler", penalty=50)
    failed.finish()
    failed.end(downtime=[(t, 2) for t in range(10)])
    v = verify(failed.log, failed.released(), failed.oracle(), 40.0)
    assert (v["cost_prices"], v["penalties"], v["downtime_total"], v["cost"]) == (0, 50, 20, 70) and v["failures"] == ["cooler"]
    assert v["cost"] == failed.log[-1]["cost"]


# ----- the reward -----

def test_the_reward_credits_success_so_a_correct_abstention_earns_the_outcome_term():
    ep = declared_episode()
    reference = verify(oracle_like().log, oracle_like().released(), ep.oracle(), None)["cost"]
    correct = verify(ep.log, ep.released(), ep.oracle(), reference, grade=INSEPARABLE)
    wrong = verify(ep.log, ep.released(), ep.oracle(), reference, grade=SEPARABLE)
    cost_term = 0.2 * correct["cost_normalized"]
    assert abs(reward(correct, HORIZON) - (1.0 - cost_term)) < 1e-12 and abs(reward(wrong, HORIZON) - (0.0 - cost_term)) < 1e-12
    committed = oracle_like()
    on_inseparable = verify(committed.log, committed.released(), committed.oracle(), reference, grade=INSEPARABLE)
    on_separable = verify(committed.log, committed.released(), committed.oracle(), reference, grade=SEPARABLE)
    # the same episode loses exactly the outcome term when the start was inseparable and the agent committed anyway
    assert abs(reward(on_separable, HORIZON) - reward(on_inseparable, HORIZON) - 1.0) < 1e-12


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
    assert reward(verify(fabricated.log, fabricated.released(), fabricated.oracle(), reference), HORIZON) == 0.0
    assert reward({"verified": False, "error": "the log has no reset event"}, HORIZON) is None


def test_the_reward_formula_with_the_declared_weights_by_hand():
    ep = Episode(HEALTHY_LIST)
    ep.advance(4)
    ep.drop("cooler", 20)
    ep.read(HOT)  # the reading at cycle 4 leaves its band; the read takes one cycle
    ep.diagnose(dict(zip(COMPONENTS, HEALTHY_LIST)), [f"obs/1/{TEMPERATURE}"])
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
    assert abs(reward(v, HORIZON, w_c=0.5, w_t=0.0) - (1.0 - 0.5)) < 1e-12


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

def test_scope_all_adds_the_oracle_policy_reads_to_the_count_gate():
    ep = oracle_like()
    assert read_counts(ep.log) == {"read": 1, "advance": 1}
    writes = verify(ep.log, ep.released(), ep.oracle(), 40.0, scope="writes")
    all_actions = verify(ep.log, ep.released(), ep.oracle(), 40.0, scope="all")
    assert writes["checks"]["count_gate"]["ok"] and not all_actions["checks"]["count_gate"]["ok"]
    assert all_actions["checks"]["count_gate"]["problems"] == ["reads differ from the oracle policy's reads"]
    assert all_actions["pass"] == writes["pass"], "the scope changes the gate, not the outcome"
    matched = verify(ep.log, ep.released(), ep.oracle(reads=read_counts(ep.log)), 40.0, scope="all")
    assert matched["checks"]["count_gate"]["ok"] and matched["all_checks_ok"]
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
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    try:
        from envs.hydraulic_rig.config import EpisodeConfig
        from envs.hydraulic_rig.env import HydraulicRigEnv
        from envs.hydraulic_rig.policies import DoNothingPolicy, OraclePolicy, run_episode
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
