"""Deterministic scoring from the episode log alone (docs/hydraulic_rig/environment.md).

The diagnosis is scored against the initial hidden combination (the fault the agent was asked to find);
recovery against the combination after the agent's repairs. cost_before_submit is the cost at the submit event
(None if the agent never submitted); "cost to the correct diagnosis" is this value on Top-1-correct episodes.
steps counts every step call (measure, replace, submit, refused, finish) and equals the environment's own counter;
steps_before_submit counts up to and including the submit event.
"""
from .data import COMPONENTS, HEALTHY


def score(log):
    reset = log[0]
    assert reset["event"] == "reset"
    initial = tuple(reset["hidden"])
    combo = list(initial)
    cost = 0
    measures = {1: 0, 2: 0, 3: 0}
    replaced = []
    unnecessary = 0
    submitted = None
    cost_before_submit = None
    steps = 0
    steps_before_submit = None
    refusals = 0
    end_reason = None
    for e in log[1:]:
        cost += e.get("cost", 0)
        if e["event"] in ("measure", "replace", "submit", "refused", "finish"):
            steps += 1  # every step call, refusals and the accepted finish included (the "number of tool calls" metric)
        if e["event"] == "measure":
            measures[e["tier"]] += 1
        elif e["event"] == "replace":
            replaced.append(e["component"])
            unnecessary += int(e["unnecessary"])
            combo[COMPONENTS.index(e["component"])] = HEALTHY[e["component"]]
        elif e["event"] == "submit":
            submitted = e
            cost_before_submit = cost
            steps_before_submit = steps
        elif e["event"] == "refused":
            refusals += 1
        elif e["event"] == "end":
            end_reason = e["reason"]
    final = tuple(combo)
    insufficient = bool(submitted and submitted.get("insufficient_evidence"))
    # An insufficient-evidence outcome is its own outcome (logged, graded in a later version), never a Top-1 guess.
    diagnosis = submitted.get("diagnosis") if submitted and not insufficient else None
    per_component = {c: (diagnosis.get(c) == initial[i]) if diagnosis else None for i, c in enumerate(COMPONENTS)}
    all_correct = bool(diagnosis) and all(per_component.values())
    diagnosed_faulty = {c for c in COMPONENTS if diagnosis and diagnosis.get(c) != HEALTHY[c]} if diagnosis else set()
    return {
        "submitted": submitted is not None,
        "insufficient_evidence": insufficient,
        "diagnosis_per_component": per_component,
        "diagnosis_all_correct": all_correct,
        "recovered": final == tuple(HEALTHY[c] for c in COMPONENTS),
        "repairs_match_diagnosis": bool(diagnosis) and set(replaced) <= diagnosed_faulty,
        "cost": cost,
        "cost_before_submit": cost_before_submit,
        "steps": steps,
        "steps_before_submit": steps_before_submit,
        "measures_tier1": measures[1], "measures_tier2": measures[2], "measures_tier3": measures[3],
        "replacements": len(replaced),
        "unnecessary_replacements": unnecessary,
        "refusals": refusals,
        "end_reason": end_reason,
        "initial": initial,
        "final": final,
    }
