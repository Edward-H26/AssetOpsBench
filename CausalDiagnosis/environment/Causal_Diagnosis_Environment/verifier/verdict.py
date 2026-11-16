from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import VERIFIER_SCOPES, cost_unit
from . import episode_log, episode_costs, failure_class, summary_consistency, episode_checks, delays


def verify(log, released, oracle, cost_oracle, grade=None, scope="writes", predictions=None):
    """The verdict of one episode.

    log: the environment's full log, hidden fields included. released: the released evidence, format 3. oracle: the
    episode's oracle, {"diagnosis", "work_orders": [{"component", "window_end"}], "finish"} and, for the scope "all",
    "reads" as `read_counts` computes it from the greedy policy's own log. cost_oracle: the cost of the oracle policy's
    episode, which normalizes the cost. grade: the separability of the start, a dict with the independent label's boolean
    "separable" and under "e18" the server's E18 grade with a boolean "separable", such as `separability.start_grade` returns,
    or None when unknown; an unknown grade counts as separable under both, since it is no evidence that a declaration was
    right, and a grade without the E18 verdict raises. predictions: the records of
    predictions.json when the diagnostics server wrote one.
    """
    if scope not in VERIFIER_SCOPES:
        raise ValueError(f"scope must be one of {VERIFIER_SCOPES}, got {scope!r}")
    separable = _separable(grade)
    separable_e18 = _separable_e18(grade)
    asset = episode_log._asset_truth(log)
    problem = episode_checks._unverifiable(log, released, oracle, asset)
    if problem:
        return {"verified": False, "error": problem, "scope": scope}
    hidden = asset.levels(log[0]["hidden"])
    end = episode_log._first(log, "end")
    final = asset.levels(end["final"])
    held = episode_log._held(end, asset)
    # the levels that outlast the episode: a component that an emergency hold keeps healthy at the end counts at the level
    # the hold replaced, so a repair that rests on a hold is no recovery
    lasting = {c: held.get(c, final[c]) for c in asset.components}
    orders = [e for e in log if e["event"] == "work_order_created"]
    diagnosis_index = next((i for i, e in enumerate(log) if e["event"] == "submit_diagnosis"), None)
    diagnosis_event = log[diagnosis_index] if diagnosis_index is not None else None
    declaration = episode_log._first(log, "declare_insufficient_evidence")
    diagnosis = asset.levels(diagnosis_event["diagnosis"]) if diagnosis_event else None
    truth_at_write = episode_log._combination_after(log[:diagnosis_index], hidden, asset) if diagnosis_event else None

    count_gate = episode_checks._count_gate(log, oracle, scope)
    hard = episode_checks._hard_checks(diagnosis, truth_at_write, oracle, orders, asset)
    timing = episode_checks._timing(log, orders, oracle)
    causality = episode_checks._causality(log, orders)
    outcome = episode_checks._outcome(lasting, orders, asset, held)
    cited = episode_log._cited_identifiers(log)
    provenance = episode_checks._provenance(cited, released, predictions, asset)
    checks = {"count_gate": count_gate, "hard": hard, "timing": timing, "causality": causality, "outcome": outcome,
              "provenance": provenance}

    all_correct = hard["diagnosis_all_correct"]
    passed = bool(all_correct and outcome["recovered"] and outcome["unnecessary"] == 0)
    abstained = declaration is not None
    cites_reading = abstained and episode_log._cites_reading(log)
    abstention_correct = cites_reading and not separable and not separable_e18
    success = passed or abstention_correct
    costs = episode_costs._costs(log, end, cost_oracle, cost_unit(episode_log._reset_config(log)["prices"]))
    first_anomaly = failure_class._first_anomaly(released, asset)
    summary_check = summary_consistency._summary_check(episode_log._closing_summary(log), outcome["recovered"], episode_log._closed_by_agent(log), episode_log._alarms_released(released),
                                   asset)
    refusals = sum(1 for e in log if e["event"] == "refused")
    verdict = {
        "verified": True,
        "scope": scope,
        "grade": grade,
        "separable": separable,
        "separable_e18": separable_e18,
        "checks": checks,
        "all_checks_ok": all(c["ok"] for c in checks.values()),
        "read_gate": count_gate["read_gate"],
        "diagnosis_per_component": hard["diagnosis_per_component"],
        "diagnosis_all_correct": all_correct,
        "faulty_set_right": hard["faulty_set_right"],
        "recovered": outcome["recovered"],
        "unnecessary": outcome["unnecessary"],
        "false_alarms": len(episode_checks._false_alarms(diagnosis, hard["diagnosis_per_component"], truth_at_write, asset)),
        "pass": passed,
        **episode_checks._graded(log, diagnosis, truth_at_write, hidden, lasting, outcome, hard, passed, asset),
        "success": success,
        **costs,
        **episode_costs._misdiagnosis_field(log),
        "detection_delay": delays._detection_delays(log, hidden, asset),
        "repair_delay": delays._repair_delays(log, hidden, asset),
        "unrestored": delays._unrestored(log, hidden, asset),
        "turns_to_correct_diagnosis": diagnosis_event["step"] if all_correct else None,
        "abstained": abstained,
        "abstention_cites_reading": cites_reading,
        "abstention_correct": abstention_correct,
        "commitment_on_inseparable": diagnosis_event is not None and not separable,
        "prediction_verdict": _prediction_verdict(orders, predictions, outcome["recovered"]),
        "end_reason": end["reason"],
        "time_used": end["time"],
        "steps": sum(1 for e in log if e.get("step") is not None),
        "refusals": refusals,
        "failures": sorted({e["component"] for e in log if e["event"] == "failure"}),
        "first_anomaly": first_anomaly,
        "summary_check": summary_check,
        "hidden_start": hidden,
        "write_t": diagnosis_event["t_start"] if diagnosis_event else None,
        "truth_at_write": truth_at_write,
        "final": final,
    }
    verdict["failure_class"] = failure_class._failure_class(verdict, log, diagnosis, truth_at_write, first_anomaly, asset)
    return verdict


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
    """Whether the start is separable under the declared budget: False only when the label says so. An unknown label
    counts as separable, since it is no evidence that a declaration was right."""
    if grade is None:
        return True
    if not isinstance(grade, dict) or not isinstance(grade.get("separable"), bool):
        raise ValueError(f"grade must be None or a dict with a boolean 'separable', got {grade!r}")
    return grade["separable"]


def _separable_e18(grade):
    """Whether the server's E18 grade, the second label, marks the start separable, the majority of its
    first commitments as grade.separable reads it. An unknown grade counts as separable, as for the label, and a grade without
    the E18 verdict raises, since scoring a declaration under the label alone would credit it on a start the server's rule
    solves."""
    if grade is None:
        return True
    e18 = grade.get("e18")
    if not isinstance(e18, dict) or not isinstance(e18.get("separable"), bool):
        raise ValueError(f"grade must carry the server's E18 verdict, a dict with a boolean 'separable' under 'e18', got {e18!r}")
    return e18["separable"]
