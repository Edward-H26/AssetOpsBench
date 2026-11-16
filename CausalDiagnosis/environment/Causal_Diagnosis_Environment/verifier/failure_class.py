from . import episode_log


# The twelve classes of the failure taxonomy in the order the verifier decides them, and "none" otherwise. The two repair
# classes follow the unnecessary replacement, since all three are decided from the hidden fields of the agent's orders, and
# precede the summary check, which an agent that never saw the repair fail would otherwise trip first.
FAILURE_CLASSES = (
    "premature_diagnosis",
    "unmatched_evidence",
    "unnecessary_replacement",
    "repair_before_dependency",
    "temporary_repair_expired",
    "abstention_without_reading",
    "abstention_on_separable",
    "abstention_on_e18_separable",
    "summary_contradicts_log",
    "repeated_malformed_call",
    "anchoring_on_first_anomaly",
    "recovery_without_diagnosis",
)
REPEATED_REFUSALS = 3  # a repeated malformed call is three or more refused actions in one episode


# ----- the failure class -----
def _failure_class(verdict, log, diagnosis, truth_at_write, first_anomaly, asset):
    """One of the twelve classes of the taxonomy in the order of FAILURE_CLASSES, for an episode that did not succeed or whose
    provenance gate fired; "none" for a successful episode and for a failure outside the taxonomy (a wrong commitment that
    no class describes, for example)."""
    if verdict["success"] and verdict["checks"]["provenance"]["ok"]:
        return "none"
    diagnosis_index = next((i for i, e in enumerate(log) if e["event"] == "submit_diagnosis"), None)
    # a read counts only when it served a reading, so a read that the horizon or a repair left empty is none
    reads_before = diagnosis_index is not None and any(e["event"] == "read" and e.get("served") for e in log[:diagnosis_index])
    if diagnosis_index is not None and not reads_before:
        return "premature_diagnosis"
    if not verdict["checks"]["provenance"]["ok"]:
        return "unmatched_evidence"
    if verdict["unnecessary"] > 0:
        return "unnecessary_replacement"
    if _charged_ineffective(log, asset):
        return "repair_before_dependency"
    if _expired_without_repair(log, asset):
        return "temporary_repair_expired"
    # before the label, since an agent decides whether it cites a reading, and an unknown label says nothing about that
    if verdict["abstained"] and not verdict["abstention_cites_reading"]:
        return "abstention_without_reading"
    if verdict["abstained"] and verdict["separable"]:
        return "abstention_on_separable"
    if verdict["abstained"] and verdict["separable_e18"]:
        return "abstention_on_e18_separable"
    if verdict["summary_check"]["contradicts_log"]:
        return "summary_contradicts_log"
    if verdict["refusals"] >= REPEATED_REFUSALS:
        return "repeated_malformed_call"
    if diagnosis is not None and first_anomaly is not None:
        diagnosed = asset.faulty(diagnosis)
        if diagnosed and diagnosed == set(first_anomaly["indicates"]) and diagnosed != asset.faulty(truth_at_write):
            return "anchoring_on_first_anomaly"
    if verdict["recovered"] and not verdict["diagnosis_all_correct"]:
        return "recovery_without_diagnosis"
    return "none"


def _expired_without_repair(log, asset):
    """Whether an emergency repair expired and no later repair restored its component, no agent order having closed on it
    with the component healthy afterward, or whether a hold was still in force at the end, which scores its component at
    the level before the repair."""
    if episode_log._first(log, "end").get("held"):
        return True
    for i, expired in enumerate(log):
        if expired["event"] != "emergency_expired":
            continue
        c = expired["component"]
        repaired = any(e["event"] == "work_order_closed" and e["component"] == c
                       and asset.levels(e.get("combination_after") or [])[c] == asset.healthy[c] for e in log[i + 1:])
        if not repaired:
            return True
    return False


def _charged_ineffective(log, asset):
    """Whether an ineffective close is the agent's error: one of the dependencies that blocked the close was degraded when
    the order was created and stayed degraded until the close. A dependency that was healthy at some state in between
    dropped after the agent ordered, by the hazard, the schedule or the expiry of an emergency repair, which the order
    could not foresee, so that close is not charged as a repair before its dependency."""
    hidden = asset.levels(log[0]["hidden"])
    created = {e["id"]: i for i, e in enumerate(log) if e["event"] == "work_order_created"}
    for j, e in enumerate(log):
        if e["event"] != "order_ineffective":
            continue
        i = created.get(e["id"], 0)
        state, states = episode_log._combination_after(log[:i], hidden, asset), []
        for k in range(i, j):
            state = episode_log._next_combination(state, log[k], asset)
            states.append(state)
        if any(all(state[d] != asset.healthy[d] for state in states) for d in e["degraded_dependencies"]):
            return True
    return False


def _first_anomaly(released, asset):
    """The first channel the agent saw outside its public normal range, from the released events in order (a served
    reading, an alarm of a change summary or an alarm notification), with the components it indicates. A served reading shows
    its channels at once, so within one it is the channel furthest outside its normal range in normal range widths, ties by name, the
    strongest anomaly an agent can anchor on, and not the name that sorts first."""
    bands = (released.get("rules") or {}).get("normal_ranges") or {}
    for event in released["events"]:
        if event["event"] == "read":
            for served in event["served"]:
                beyond = {field: _excursion(value, bands[field]) for field, value in served["values"].items() if field in bands}
                outside = sorted(field for field, x in beyond.items() if x > 0)
                if outside:
                    return _anomaly(max(outside, key=beyond.get), served["t"], "read", asset)
        elif event["event"] == "summary" and event.get("alarms"):
            alarm = min(event["alarms"], key=lambda a: a["first_t"])
            return _anomaly(alarm["channel"], alarm["first_t"], "alarm", asset)
        elif event["event"] == "alarm":
            return _anomaly(event["channel"], event["t_end"], "alarm", asset)
    return None


def _excursion(value, band):
    """How far a value lies outside its normal range in normal range widths, 0 inside it, and in the units of the value for a normal range of no width."""
    width = band["high"] - band["low"]
    beyond = max(band["low"] - value, value - band["high"], 0.0)
    return beyond / width if width > 0 else beyond


def _anomaly(field, t, source, asset):
    return {"channel": field, "t": t, "source": source, "indicates": sorted(asset.indicated(field))}
