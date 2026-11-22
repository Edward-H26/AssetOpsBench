from collections import Counter
from CausalDiagnosis.datasets.dataset_loader.ground_truth import misdiagnosed
from . import episode_log


def read_counts(log):
    """The sensor reads of a log per sensor group, the object the scope "all" compares between an agent and the greedy policy.
    A read counts once whatever cycles it covers, since counting cycles would measure the tool shape of AssetOpsBench's history
    call and not the reading strategy, and a multi-cycle read still pays its price in the cost. It counts only when it released
    a reading, so a free re-read and the listing and lookup calls of the rows stay out."""
    return dict(Counter(e["sensor_group"] for e in log if e["event"] == "read" and e["served"]))


# ----- the checks -----
def _unverifiable(log, released, oracle, asset):
    """Why the verifier cannot run on these inputs, or None."""
    if not isinstance(log, list) or not log or log[0].get("event") != "reset":
        return "the log has no reset event"
    if "hidden" not in log[0]:
        return "the log lacks its hidden fields; the verifier needs env.log, not the public view"
    if episode_log._first(log, "end") is None:
        return "the log has no end event, so the episode has not ended"
    if not isinstance(episode_log._reset_config(log).get("prices"), dict):
        # the cost unit of the row comes from the prices of the reset entry, which every log of the environment holds
        return "the reset entry of the log holds no prices, so the cost unit of its row is unknown"
    if any("unnecessary" not in e for e in log if e["event"] == "work_order_created"):
        return "a work order lacks the hidden unnecessary flag"
    if not isinstance(oracle, dict) or "diagnosis" not in oracle or "work_orders" not in oracle:
        return "the oracle lacks its diagnosis or its work orders"
    if asset.levels(oracle["diagnosis"]) != asset.levels(log[0]["hidden"]):
        return "the oracle's diagnosis differs from the log's hidden start"
    if not isinstance(released, dict) or released.get("format") != episode_log.RELEASED_FORMAT:
        return f"the released evidence is not format {episode_log.RELEASED_FORMAT}"
    if "events" not in released or "observations" not in released:
        return "the released evidence lacks its events or its observations"
    return None


def _count_gate(log, oracle, scope):
    """Exactly one diagnosis or declaration, exactly one finish, and no write through a tool the oracle does not name.
    A refused action is not a write, as ARE drops failed writes. Under the scope "all" the read gate holds as well: no
    sensor group read more often than the greedy policy reads it on the same start."""
    counts = Counter(e["event"] for e in log if e["event"] in episode_log.WRITE_EVENTS)
    commits = counts["submit_diagnosis"] + counts["declare_insufficient_evidence"]
    unknown = sorted({e["event"] for e in log
                      if e.get("class") == "write" and e["event"] not in episode_log.WRITE_EVENTS and e["event"] != "refused"})
    problems = []
    if commits != 1:
        problems.append(f"{commits} diagnosis or declaration events, expected exactly one")
    if counts["finish"] != 1:
        problems.append(f"{counts['finish']} finish events, expected exactly one")
    if unknown:
        problems.append(f"writes through tools the oracle does not name: {unknown}")
    reads = read_counts(log)
    reference = dict(oracle.get("reads") or {})
    above = sorted(group for group, count in reads.items() if count > reference.get(group, 0))
    if scope == "all" and above:
        problems.append(f"sensor reads above the greedy policy's in {', '.join(above)}")
    return {"ok": not problems, "writes": dict(counts), "reads": reads, "reference_reads": reference if scope == "all" else None,
            "read_gate": not above if scope == "all" else None, "problems": problems}


def _hard_checks(diagnosis, truth_at_write, oracle, orders, asset):
    """The submitted levels against the hidden combination at the cycle of the write, component by component, and the
    components of the work orders against the oracle's set as sets, since the outcome of two repairs in the hidden state does
    not depend on their order, and a second order on a component after an ineffective or expired repair is a legal recovery. faulty_set_right says whether the components that the diagnosis puts off their healthy level
    are exactly the components off their healthy level at the write, whatever the levels, and is false without a
    diagnosis."""
    per_component = asset.diagnosis_correct(diagnosis, truth_at_write)
    all_correct = diagnosis is not None and not misdiagnosed(asset, diagnosis, truth_at_write)
    faulty_set_right = diagnosis is not None and asset.faulty(diagnosis) == asset.faulty(truth_at_write)
    agent_set = sorted(e["component"] for e in orders)
    oracle_set = sorted(o["component"] for o in oracle["work_orders"])
    orders_ok = set(agent_set) == set(oracle_set)
    return {"ok": all_correct and orders_ok, "diagnosis_per_component": per_component,
            "diagnosis_all_correct": all_correct, "faulty_set_right": faulty_set_right, "work_orders_agent": agent_set,
            "work_orders_oracle": oracle_set, "work_orders_ok": orders_ok}


def _timing(log, orders, oracle):
    """Every work order on a component that degrades during the episode, by a scheduled drop or by a hazard drop, closes
    no later than the window end of the run at the worst level that is open at its close. A run is a stretch of cycles at
    the worst level, and its window end is the cycle at which the failure card would fail the component on it, the one
    deadline of the card and of this check, since the card counts the cycles of each run from zero. A close is processed
    before the failure check of its cycle, so a repair that closes at the window end and restores the component still
    prevents the failure, while an order that is open but not closed by then does not. _run_windows gives the window end
    of every close, and the row of an order that never closed shows the oracle's window end."""
    degrading = {e["component"] for e in log if e["event"] in ("scheduled_drop", "hazard_drop")}
    closes = episode_log._run_windows(log, oracle)
    window_end = {o["component"]: o.get("window_end") for o in oracle["work_orders"]}
    rows = [_timing_row(order, closes, window_end) for order in orders if order["component"] in degrading]
    return {"ok": all(r["ok"] for r in rows), "orders": rows}


def _timing_row(order, closes, window_end):
    """The row of one work order: its close and the window end of the run at the worst level open at it, the oracle's window end
    for an order that never closed, and whether it closed by then."""
    close_t, end_t = closes.get(order["id"], (None, window_end.get(order["component"])))
    return {"id": order["id"], "component": order["component"], "closed_t": close_t, "window_end": end_t,
            "ok": close_t is not None and (end_t is None or close_t <= end_t)}


def _worst_level_timing(log, orders, oracle):
    """The row of _timing for every work order on a component that stands at its worst level, the last level of its degradation
    order in the asset file, at the start of the episode or at an event of the log, whether or not a drop degrades it during the
    episode, so that an order on a component at its worst level from the start meets the window end that the oracle gives it: the
    cycle at which the failure card would fail the component, its patience after the run opens. An order needs the lead time and
    the repair downtime that the asset declares before it closes. The rows are reported beside the checks and enter neither
    pass, nor all_checks_ok, nor a score, since the timing check and the plans of the reference policies read _timing alone."""
    closes, at_worst = episode_log._worst_level_runs(log, oracle)
    window_end = {o["component"]: o.get("window_end") for o in oracle["work_orders"]}
    rows = [_timing_row(order, closes, window_end) for order in orders if order["component"] in at_worst]
    return {"ok": all(r["ok"] for r in rows), "orders": rows}


def _causality(log, orders):
    """The diagnosis or the declaration precedes every work order in the log."""
    commit_index = next((i for i, e in enumerate(log) if e["event"] in episode_log.COMMIT_EVENTS), None)
    order_indices = [i for i, e in enumerate(log) if e["event"] == "work_order_created"]
    ok = not orders or (commit_index is not None and all(commit_index < i for i in order_indices))
    return {"ok": ok, "commit_index": commit_index, "first_order_index": order_indices[0] if order_indices else None}


def _false_alarms(diagnosis, per_component, at_write, asset):
    """The components that are at their healthy level at the cycle of the write and that the diagnosis puts at another
    level, in component order, and none without a diagnosis."""
    if diagnosis is None:
        return []
    return [c for c in asset.components if at_write[c] == asset.healthy[c] and not per_component[c]]


def _graded(log, diagnosis, truth_at_write, hidden, final, outcome, hard, passed, asset):
    """The graded Top-1 and pass, for any asset. graded_top1 is the share of the components faulty at the
    cycle of the first diagnosis that it names at their true level, 0 without a diagnosis or when it holds a false alarm.
    graded_pass is the share of the components faulty at any time of the episode, at the start, at the write or after a
    drop, that are healthy at the end, a component faulty at the write counting only when the diagnosis named it at its
    true level, and 0 when the diagnosis holds a false alarm or a work order was unnecessary. Where no component is faulty
    at the write graded_top1 equals Top-1, and where none is faulty at any time graded_pass equals pass, so an empty share
    is no success of an agent that submitted nothing. On a start with at most one fault and no drop the two fields
    therefore equal Top-1 and pass. The _n fields hold the denominators, the faulty counts, so each share is an exact
    fraction."""
    healthy, per_component = asset.healthy, hard["diagnosis_per_component"]
    at_write = truth_at_write if diagnosis is not None else hidden
    faulty_at_write = [c for c in asset.components if at_write[c] != healthy[c]]
    false_alarm = bool(_false_alarms(diagnosis, per_component, at_write, asset))
    right = [c for c in faulty_at_write if diagnosis is not None and per_component[c]]
    changed = {e["component"] for e in log if e["event"] in ("scheduled_drop", "hazard_drop", "emergency_expired")}
    ever = [c for c in asset.components if hidden[c] != healthy[c] or c in faulty_at_write or c in changed]
    if not faulty_at_write:
        top1 = float(bool(hard["diagnosis_all_correct"]))
    elif false_alarm or diagnosis is None:
        top1 = 0.0
    else:
        top1 = len(right) / len(faulty_at_write)
    credited = [c for c in ever if final[c] == healthy[c] and (c not in faulty_at_write or c in right)]
    if not ever:
        graded_pass = float(passed)
    elif false_alarm or outcome["unnecessary"] > 0:
        graded_pass = 0.0
    else:
        graded_pass = len(credited) / len(ever)
    return {"graded_top1": top1, "graded_top1_n": len(faulty_at_write), "graded_pass": graded_pass, "graded_pass_n": len(ever)}


def _outcome(lasting, orders, asset, held=()):
    """Recovered when the levels that outlast the episode equal the healthy ones, the final combination with every component
    that an emergency hold keeps at its healthy level at the end put back at the level the hold replaced; held lists those
    components, sorted. u counts the orders on components that were healthy when the order was placed, which the
    environment marks on each order from the hidden state."""
    recovered = asset.recovered(lasting)
    return {"ok": recovered, "recovered": recovered,
            "unnecessary": sum(1 for e in orders if e["unnecessary"]),
            "unnecessary_orders": [e["id"] for e in orders if e["unnecessary"]], "held": sorted(held)}


def _provenance(cited, released, predictions, asset):
    """Every cited evidence identifier exists in the released evidence (obs, wo, alarm and fmea identifiers) or among the
    predictions the diagnostics server stored (pred identifiers)."""
    valid = episode_log._released_identifiers(released, predictions, asset)
    unmatched = sorted({c for c in cited if c not in valid})
    return {"ok": not unmatched, "cited": len(cited), "unmatched": unmatched}
