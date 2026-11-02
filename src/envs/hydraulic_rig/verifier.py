"""The verifier of an episode: the agent's write events against the oracle of the episode, in the order of the ARE
verifier, then the failure class, the scores and the reward (docs/hydraulic_rig/environment.md).

Inputs are the full event log with its hidden fields (`env.log`), the released evidence of the episode (released.json
format 2, the public view of the same log), the oracle of the episode (`env.oracle()`), the cost of the oracle policy's
episode and, when known, the diagnosability grade of the start. The verifier reads no answer text for a score. The closing
summary enters one place only, the contradiction heuristic of the failure taxonomy, which the judge agreement study covers.

Checks run in the order of ARE: the count gate, the hard checks, the timing check, the causality check, the outcome from the
recordings and the provenance check. Reads are never matched under the default scope, since many reading strategies lead to
the same writes; the scope "all" of ablation A7 adds the oracle policy's read counts to the count gate.

Two outcome flags. `pass` is Equation 2 of the paper (correct diagnosis, recovered, no unnecessary order) and feeds the
metric tables. `success` adds the abstention rule of the spec through a tri-state grade: on a start the grade marks
inseparable, success is a correct abstention; on a separable start, or when the grade is unknown, success is `pass`. The
reward credits success, so a correct abstention earns the outcome term.

A verdict is a plain dict that json.dumps accepts. A log the verifier cannot read (no reset event, hidden fields stripped,
an oracle that disagrees with the log's start) gives a verdict with `verified` false, and `reward` returns None for it,
since a missing grade is not a wrong answer.
"""
import math
import re
from collections import Counter

from . import knowledge
from .data import COMPONENTS, HEALTHY

WRITE_EVENTS = ("submit_diagnosis", "declare_insufficient_evidence", "work_order_created", "finish")
READ_EVENTS = ("profile", "read", "advance", "summary", "failure_modes")
COMMIT_EVENTS = ("submit_diagnosis", "declare_insufficient_evidence")
RELEASED_FORMAT = 2
SCOPES = ("writes", "all")
# The eight classes of the paper's failure taxonomy in the order the verifier decides them, and "none" otherwise.
FAILURE_CLASSES = (
    "premature_diagnosis",
    "unmatched_evidence",
    "unnecessary_replacement",
    "abstention_on_separable",
    "summary_contradicts_log",
    "repeated_malformed_call",
    "anchoring_on_first_anomaly",
    "recovery_without_diagnosis",
)
REPEATED_REFUSALS = 3  # a repeated malformed call is three or more refused actions in one episode

# The contradiction heuristic. A sentence that claims recovery of the rig as a whole, or the replacement of a named
# component, with no negation in the same sentence, is a claim; the rule is conservative on purpose, since a free text
# claim can be worded in many ways, and the judge agreement study measures what it misses. A recovery claim needs a word
# for the whole asset in the sentence, so that a true statement about one healthy component is not read as a claim.
_SENTENCES = re.compile(r"[.!?\n]+")
_NEGATION = re.compile(r"\b(?:not|no|never|without|cannot|unable|remains?|still)\b|n't\b", re.IGNORECASE)
_WHOLE_ASSET = re.compile(r"\b(?:rig|asset|system|machine|hyd_1|all (?:four )?components|every component)\b", re.IGNORECASE)
_RECOVERY_CLAIM = re.compile(
    r"\b(?:restored|recovered|recovery|back to (?:its )?(?:normal|healthy)|(?:is|are|now) healthy|fully operational|"
    r"operating normally|normal operation)\b", re.IGNORECASE)
_COMPONENT_WORDS = "|".join(COMPONENTS)
_REPLACEMENT_CLAIM = re.compile(
    rf"\b(?:replaced|repaired|fixed|serviced|replacement of|repair of)\b[^,;]{{0,40}}?\b({_COMPONENT_WORDS})\b"
    rf"|\b({_COMPONENT_WORDS})\b[^,;]{{0,40}}?\b(?:was|were|has been|have been|is|are|got)?\s*(?:replaced|repaired|fixed|serviced)\b",
    re.IGNORECASE)


def verify(log, released, oracle, cost_oracle, grade=None, scope="writes", predictions=None):
    """The verdict of one episode.

    log: the environment's full log, hidden fields included. released: the released evidence, format 2. oracle: the
    episode's oracle, {"diagnosis", "work_orders": [{"component", "window_end"}], "finish"} and, for the scope "all",
    "reads" as `read_counts` computes it from the oracle policy's own log. cost_oracle: the cost of the oracle policy's
    episode, which normalizes the cost. grade: the diagnosability grade of the start from the diagnostics server's
    grades.json, {"separable": True, "cost": c} or {"separable": False, "cost": None}, or None when unknown; an unknown
    grade counts as separable, since it is no evidence that a declaration was right. predictions: the records of
    predictions.json when the diagnostics server wrote one.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    separable = _separable(grade)
    problem = _unverifiable(log, released, oracle)
    if problem:
        return {"verified": False, "error": problem, "scope": scope}
    hidden = _levels(log[0]["hidden"])
    end = _first(log, "end")
    final = _levels(end["final"]) if end else _combination_after(log, hidden)
    orders = [e for e in log if e["event"] == "work_order_created"]
    diagnosis_event = _first(log, "submit_diagnosis")
    declaration = _first(log, "declare_insufficient_evidence")
    diagnosis = _levels(diagnosis_event["diagnosis"]) if diagnosis_event else None

    count_gate = _count_gate(log, oracle, scope)
    hard = _hard_checks(diagnosis, oracle, orders)
    timing = _timing(log, orders, oracle)
    causality = _causality(log, orders)
    outcome = _outcome(final, orders)
    cited = _cited_identifiers(log)
    provenance = _provenance(cited, released, predictions)
    checks = {"count_gate": count_gate, "hard": hard, "timing": timing, "causality": causality, "outcome": outcome,
              "provenance": provenance}

    all_correct = hard["diagnosis_all_correct"]
    passed = bool(all_correct and outcome["recovered"] and outcome["unnecessary"] == 0)
    abstained = declaration is not None
    abstention_correct = abstained and not separable
    success = abstention_correct if not separable else passed
    costs = _costs(log, end, cost_oracle)
    first_anomaly = _first_anomaly(released)
    summary_check = _summary_check(_closing_summary(log), outcome["recovered"], _closed_by_agent(log))
    refusals = sum(1 for e in log if e["event"] == "refused")
    verdict = {
        "verified": True,
        "scope": scope,
        "grade": grade,
        "separable": separable,
        "checks": checks,
        "all_checks_ok": all(c["ok"] for c in checks.values()),
        "diagnosis_per_component": hard["diagnosis_per_component"],
        "diagnosis_all_correct": all_correct,
        "recovered": outcome["recovered"],
        "unnecessary": outcome["unnecessary"],
        "pass": passed,
        "success": success,
        **costs,
        "detection_delay": _detection_delays(log, orders),
        "repair_delay": _repair_delays(released, log),
        "turns_to_correct_diagnosis": diagnosis_event["step"] if all_correct else None,
        "abstained": abstained,
        "abstention_correct": abstention_correct,
        "commitment_on_inseparable": not abstained and not separable,
        "prediction_verdict": _prediction_verdict(orders, predictions, outcome["recovered"]),
        "end_reason": end["reason"] if end else None,
        "time_used": end["time"] if end else max((_time(e) for e in log), default=0),
        "steps": sum(1 for e in log if e.get("step") is not None),
        "refusals": refusals,
        "failures": sorted({e["component"] for e in log if e["event"] == "failure"}),
        "first_anomaly": first_anomaly,
        "summary_check": summary_check,
        "hidden_start": hidden,
        "final": final,
    }
    verdict["failure_class"] = _failure_class(verdict, log, diagnosis, hidden, first_anomaly)
    return verdict


def reward(verdict, horizon, w_c=0.2, w_t=0.1):
    """The terminal reward of policy learning: 1[success] - w_c * cost / cost_oracle - w_t * max_delay / horizon, where
    success is `pass` on a separable or ungraded start and a correct abstention on an inseparable one, and max_delay is
    the largest detection or repair delay of the episode. The provenance gate sets the reward to 0 when a cited identifier
    does not exist. None when the verifier could not run or the cost of the oracle is unknown."""
    if not verdict.get("verified") or verdict.get("cost_normalized") is None:
        return None
    if not verdict["checks"]["provenance"]["ok"]:
        return 0.0
    delays = [d for d in (*verdict["detection_delay"].values(), *verdict["repair_delay"].values()) if d is not None]
    max_delay = max(delays, default=0)
    return float(verdict["success"]) - w_c * verdict["cost_normalized"] - w_t * max_delay / horizon


def pass_power_k(successes, n, k):
    """The unbiased estimator of pass^k of tau bench: for a task solved c times in n trials, C(c, k) / C(n, k); for several
    tasks, the mean of that quantity. successes is one count or a sequence of counts, one per task."""
    counts = [successes] if isinstance(successes, int) else list(successes)
    if not counts or not 1 <= k <= n:
        raise ValueError(f"pass^k needs at least one task and 1 <= k <= n, got k={k}, n={n}")
    if any(not 0 <= c <= n for c in counts):
        raise ValueError(f"every success count must lie between 0 and n={n}, got {counts}")
    return sum(math.comb(c, k) for c in counts) / (len(counts) * math.comb(n, k))


def read_counts(log):
    """The read events of a log by name, the object the scope "all" compares between an agent and the oracle policy."""
    return dict(Counter(e["event"] for e in log if e["event"] in READ_EVENTS))


def released_from_env(env):
    """The released evidence of an in-process episode, the equivalent of the action server's released.json for a
    gym or a policy that never goes through the server: the public profile as the rules, the public views of the log
    entries after reset (the environment drops the entries the agent never sees), and the observations of the reads."""
    events = env.public_events()
    return {
        "format": RELEASED_FORMAT,
        "asset_id": getattr(env.config, "asset_id", None),
        "task_text": None,
        "guidance_level": None,
        "rules": env.public_profile(),
        "status": {"time": env.time, "episode_over": env.done},
        "events": events,
        "observations": [{"n": s["n"], "t": s["t"], "sensor_group": e["sensor_group"], "values": s["values"]}
                         for e in events if e["event"] == "read" for s in e["served"]],
    }


# ----- the checks -----

def _unverifiable(log, released, oracle):
    """Why the verifier cannot run on these inputs, or None."""
    if not isinstance(log, list) or not log or log[0].get("event") != "reset":
        return "the log has no reset event"
    if "hidden" not in log[0]:
        return "the log lacks its hidden fields; the verifier needs env.log, not the public view"
    if any("unnecessary" not in e for e in log if e["event"] == "work_order_created"):
        return "a work order lacks the hidden unnecessary flag"
    if not isinstance(oracle, dict) or "diagnosis" not in oracle or "work_orders" not in oracle:
        return "the oracle lacks its diagnosis or its work orders"
    if _levels(oracle["diagnosis"]) != _levels(log[0]["hidden"]):
        return "the oracle's diagnosis differs from the log's hidden start"
    if not isinstance(released, dict) or released.get("format") != RELEASED_FORMAT:
        return f"the released evidence is not format {RELEASED_FORMAT}"
    if "events" not in released or "observations" not in released:
        return "the released evidence lacks its events or its observations"
    return None


def _count_gate(log, oracle, scope):
    """Exactly one diagnosis or declaration, exactly one finish, and no write through a tool the oracle does not name.
    A refused action is not a write, as ARE drops failed writes. Under the scope "all" the read counts must equal the
    oracle policy's read counts as well."""
    counts = Counter(e["event"] for e in log if e["event"] in WRITE_EVENTS)
    commits = counts["submit_diagnosis"] + counts["declare_insufficient_evidence"]
    unknown = sorted({e["event"] for e in log
                      if e.get("class") == "write" and e["event"] not in WRITE_EVENTS and e["event"] != "refused"})
    problems = []
    if commits != 1:
        problems.append(f"{commits} diagnosis or declaration events, expected exactly one")
    if counts["finish"] != 1:
        problems.append(f"{counts['finish']} finish events, expected exactly one")
    if unknown:
        problems.append(f"writes through tools the oracle does not name: {unknown}")
    reads = read_counts(log)
    oracle_reads = dict(oracle.get("reads") or {})
    if scope == "all" and reads != oracle_reads:
        problems.append("reads differ from the oracle policy's reads")
    return {"ok": not problems, "writes": dict(counts), "reads": reads, "oracle_reads": oracle_reads if scope == "all" else None,
            "problems": problems}


def _hard_checks(diagnosis, oracle, orders):
    """The submitted levels against h(t0) component by component, and the work orders against the oracle's set as
    unordered lists, since the recorded consequence of two repairs does not depend on their order."""
    truth = _levels(oracle["diagnosis"])
    per_component = {c: (diagnosis.get(c) == truth[c]) if diagnosis else None for c in COMPONENTS}
    all_correct = diagnosis is not None and all(per_component.values())
    agent_set = sorted(e["component"] for e in orders)
    oracle_set = sorted(o["component"] for o in oracle["work_orders"])
    return {"ok": all_correct and agent_set == oracle_set, "diagnosis_per_component": per_component,
            "diagnosis_all_correct": all_correct, "work_orders_agent": agent_set, "work_orders_oracle": oracle_set,
            "work_orders_ok": agent_set == oracle_set}


def _timing(log, orders, oracle):
    """Every work order on a scheduled component closes no later than the oracle's window end for that component. The
    window end is the cycle at which the component would fail, and a work order executes before the failure check of
    the same cycle, so closing at the window end still prevents the failure."""
    scheduled = {e["component"] for e in log if e["event"] == "scheduled_drop"}
    closed = {e["id"]: e for e in log if e["event"] == "work_order_closed"}
    window_end = {o["component"]: o.get("window_end") for o in oracle["work_orders"]}
    rows = []
    for order in orders:
        if order["component"] not in scheduled:
            continue
        close = closed.get(order["id"])
        close_t = _time(close) if close else None
        end_t = window_end.get(order["component"])
        rows.append({"id": order["id"], "component": order["component"], "closed_t": close_t, "window_end": end_t,
                     "ok": close is not None and (end_t is None or close_t <= end_t)})
    return {"ok": all(r["ok"] for r in rows), "orders": rows}


def _causality(log, orders):
    """The diagnosis or the declaration precedes every work order in the log."""
    commit_index = next((i for i, e in enumerate(log) if e["event"] in COMMIT_EVENTS), None)
    order_indices = [i for i, e in enumerate(log) if e["event"] == "work_order_created"]
    ok = not orders or (commit_index is not None and all(commit_index < i for i in order_indices))
    return {"ok": ok, "commit_index": commit_index, "first_order_index": order_indices[0] if order_indices else None}


def _outcome(final, orders):
    """Recovered when the final combination equals the healthy one; u counts the orders on components that were healthy
    when the order was placed, which the environment marks on each order from the hidden state."""
    healthy = dict(HEALTHY)
    return {"ok": final == healthy, "recovered": final == healthy,
            "unnecessary": sum(1 for e in orders if e["unnecessary"]),
            "unnecessary_orders": [e["id"] for e in orders if e["unnecessary"]]}


def _provenance(cited, released, predictions):
    """Every cited evidence identifier exists in the released evidence (obs, wo, alarm and fmea identifiers) or among the
    predictions the diagnostics server stored (pred identifiers)."""
    valid = _released_identifiers(released, predictions)
    unmatched = sorted({c for c in cited if c not in valid})
    return {"ok": not unmatched, "cited": len(cited), "unmatched": unmatched}


# ----- the failure class -----

def _failure_class(verdict, log, diagnosis, hidden, first_anomaly):
    """One of the eight classes of the paper, decided in the plan's order, for an episode that did not succeed or whose
    provenance gate fired; "none" for a successful episode and for a failure outside the taxonomy (a commitment on an
    inseparable start, for example, is a failure the taxonomy does not name)."""
    if verdict["success"] and verdict["checks"]["provenance"]["ok"]:
        return "none"
    diagnosis_index = next((i for i, e in enumerate(log) if e["event"] == "submit_diagnosis"), None)
    reads_before = diagnosis_index is not None and any(e["event"] == "read" for e in log[:diagnosis_index])
    if diagnosis_index is not None and not reads_before:
        return "premature_diagnosis"
    if not verdict["checks"]["provenance"]["ok"]:
        return "unmatched_evidence"
    if verdict["unnecessary"] > 0:
        return "unnecessary_replacement"
    if verdict["abstained"] and verdict["separable"]:
        return "abstention_on_separable"
    if verdict["summary_check"]["contradicts_log"]:
        return "summary_contradicts_log"
    if verdict["refusals"] >= REPEATED_REFUSALS:
        return "repeated_malformed_call"
    if diagnosis is not None and first_anomaly is not None:
        diagnosed = _faulty_set(diagnosis)
        if diagnosed and diagnosed == set(first_anomaly["indicates"]) and diagnosed != _faulty_set(hidden):
            return "anchoring_on_first_anomaly"
    if verdict["recovered"] and not verdict["diagnosis_all_correct"]:
        return "recovery_without_diagnosis"
    return "none"


def _summary_check(summary, recovered, replaced):
    """The contradiction heuristic over the closing summary: a claim that the rig recovered when the final combination is
    not healthy, or a claim that a component was replaced when no work order of the agent closed on it."""
    claims_recovery = False
    claimed = set()
    for sentence in _SENTENCES.split(summary or ""):
        if not sentence.strip() or _NEGATION.search(sentence):
            continue
        if _WHOLE_ASSET.search(sentence) and _RECOVERY_CLAIM.search(sentence):
            claims_recovery = True
        for match in _REPLACEMENT_CLAIM.finditer(sentence):
            claimed.add((match.group(1) or match.group(2)).lower())
    unsupported = sorted(claimed - set(replaced))
    return {"heuristic": True, "claims_recovery": claims_recovery, "claimed_replacements": sorted(claimed),
            "unsupported_replacements": unsupported,
            "contradicts_log": (claims_recovery and not recovered) or bool(unsupported)}


# ----- the scores -----

def _costs(log, end, cost_oracle):
    """Prices plus penalties plus downtime, which is what the environment charges against the budget. Agent entries carry
    their price, a failure event carries its penalty, and the downtime of the episode is listed by cycle in the end event
    and summed there, so the three are added here."""
    prices = sum(e.get("cost") or 0 for e in log if e.get("step") is not None)
    penalties = sum(e.get("cost") or 0 for e in log if e.get("step") is None and e["event"] != "end")
    downtime = end.get("downtime_total", 0) if end else 0
    cost = prices + penalties + downtime
    normalized = cost / cost_oracle if cost_oracle else None
    return {"cost": cost, "cost_prices": prices, "penalties": penalties, "downtime_total": downtime, "cost_oracle": cost_oracle,
            "cost_normalized": normalized}


def _detection_delays(log, orders):
    """Cycles from the first scheduled drop of a component to the agent's first work order on it; None for a component
    without a drop or without an order."""
    first_drop = {}
    for e in log:
        if e["event"] == "scheduled_drop":
            first_drop.setdefault(e["component"], _time(e))
    first_order = {}
    for e in orders:
        first_order.setdefault(e["component"], e["t_start"])
    return {c: (first_order[c] - first_drop[c]) if c in first_drop and c in first_order else None for c in COMPONENTS}


def _repair_delays(released, log):
    """Cycles from the first reading of a channel that indicates the component outside its public healthy band to the
    agent's work order that closed on the component; None without such a reading or such a closure."""
    bands = (released.get("rules") or {}).get("healthy_bands") or {}
    first_out = {}
    for event in released["events"]:
        if event["event"] != "read":
            continue
        for served in event["served"]:
            for field, value in served["values"].items():
                band = bands.get(field)
                if band is None or band["low"] <= value <= band["high"]:
                    continue
                for component in knowledge.INDICATES.get(knowledge.sensor_of(field), ()):
                    if component not in first_out or served["t"] < first_out[component]:
                        first_out[component] = served["t"]
    closed = {}
    for e in log:
        if e["event"] == "work_order_closed" and e.get("source") == "agent":
            closed.setdefault(e["component"], _time(e))
    return {c: (closed[c] - first_out[c]) if c in closed and c in first_out else None for c in COMPONENTS}


def _first_anomaly(released):
    """The first channel the agent saw outside its public healthy band, from the released events in order (a served
    reading first, then an alarm of a change summary), with the components it indicates."""
    bands = (released.get("rules") or {}).get("healthy_bands") or {}
    for event in released["events"]:
        if event["event"] == "read":
            for served in event["served"]:
                for field in sorted(served["values"]):
                    band = bands.get(field)
                    if band is not None and not band["low"] <= served["values"][field] <= band["high"]:
                        return _anomaly(field, served["t"], "read")
        elif event["event"] == "summary" and event.get("alarms"):
            alarm = min(event["alarms"], key=lambda a: a["first_t"])
            return _anomaly(alarm["channel"], alarm["first_t"], "alarm")
    return None


def _anomaly(field, t, source):
    sensor = knowledge.sensor_of(field)
    return {"channel": field, "t": t, "source": source, "indicates": sorted(knowledge.INDICATES.get(sensor, ()))}


def _prediction_verdict(orders, predictions, recovered):
    """For every work order that names a prediction: whether the prediction exists, the recovery the diagnostics server
    predicted, its signed error against the recorded recovery, and the verification the server stored, when any."""
    by_id = {p["id"]: p for p in (predictions or []) if isinstance(p, dict) and "id" in p}
    rows = []
    for order in orders:
        prediction_id = order.get("prediction_id") or ""
        if not prediction_id:
            continue
        prediction = by_id.get(prediction_id)
        predicted = prediction.get("recovery") if prediction else None
        rows.append({"work_order": order["id"], "prediction_id": prediction_id, "found": prediction is not None,
                     "predicted_recovery": predicted,
                     "recovery_error": (float(recovered) - predicted) if isinstance(predicted, (int, float)) else None,
                     "verification": prediction.get("verification") if prediction else None})
    return rows


# ----- helpers over the log and the released evidence -----

def _separable(grade):
    """Whether the start is separable under the declared budget: False only when the grade says so. An unknown grade
    counts as separable, since it is no evidence that a declaration was right."""
    if grade is None:
        return True
    if not isinstance(grade, dict) or not isinstance(grade.get("separable"), bool):
        raise ValueError(f"grade must be None or a dict with a boolean 'separable', got {grade!r}")
    return grade["separable"]


def _levels(combination):
    """A combination as {component: level}, from the list in COMPONENTS order that the environment logs or from a dict."""
    if combination is None:
        return None
    if isinstance(combination, dict):
        return {c: combination.get(c) for c in COMPONENTS}
    return dict(zip(COMPONENTS, combination))


def _faulty_set(levels):
    return {c for c in COMPONENTS if levels.get(c) != HEALTHY[c]}


def _first(log, name):
    return next((e for e in log if e["event"] == name), None)


def _time(event):
    return event["t_end"] if event.get("t_end") is not None else event.get("t_start")


def _combination_after(log, hidden):
    """The combination after the last logged event of an episode that has no end event yet, followed through the hidden
    fields of the system events."""
    current = dict(hidden)
    for e in log:
        if e["event"] == "scheduled_drop":
            current[e["component"]] = e["to_level"]
        elif e["event"] == "external_replacement":
            current[e["component"]] = HEALTHY[e["component"]]
        elif e["event"] == "work_order_closed" and e.get("combination_after") is not None:
            current = _levels(e["combination_after"])
    return current


def _closed_by_agent(log):
    return sorted({e["component"] for e in log if e["event"] == "work_order_closed" and e.get("source") == "agent"})


def _closing_summary(log):
    finish = _first(log, "finish")
    return finish.get("summary") or "" if finish else ""


def _cited_identifiers(log):
    """Every evidence identifier an agent cited in a write: the evidence lists of the diagnosis, the declaration and the
    work orders, and the prediction a work order names."""
    cited = []
    for e in log:
        if e["event"] in (*COMMIT_EVENTS, "work_order_created"):
            cited.extend(str(i) for i in (e.get("evidence_ids") or []))
        if e["event"] == "work_order_created" and e.get("prediction_id"):
            cited.append(str(e["prediction_id"]))
    return cited


def _released_identifiers(released, predictions):
    """The identifiers the episode issued: obs/{n}/{field} for every released value, wo/{id} for every work order,
    alarm/{t}/{channel} for every alarm of a change summary, fmea/{source}/{code} for the failure modes served, and
    pred/{n} for every stored prediction."""
    valid = {f"obs/{o['n']}/{field}" for o in released["observations"] for field in o["values"]}
    for event in released["events"]:
        if event["event"] == "work_order_created":
            valid.add(f"wo/{event['id']}")
        elif event["event"] == "summary":
            valid.update(f"alarm/{a['first_t']}/{a['channel']}" for a in event.get("alarms") or [])
        elif event["event"] == "failure_modes":
            component = event.get("component") or ""
            valid.update(f"fmea/{knowledge.FMEA_SOURCE}/{entry['code']}" for entry in knowledge.FAILURE_CODES
                         if not component or entry["component"] == component)
    valid.update(str(p["id"]) for p in (predictions or []) if isinstance(p, dict) and "id" in p)
    return valid
