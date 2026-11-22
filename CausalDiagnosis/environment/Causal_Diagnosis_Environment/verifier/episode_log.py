from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.dataset_loader.ground_truth import LevelsTruth
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET


WRITE_EVENTS = ("submit_diagnosis", "declare_insufficient_evidence", "work_order_created", "finish")
COMMIT_EVENTS = ("submit_diagnosis", "declare_insufficient_evidence")
RELEASED_FORMAT = 3  # the format of the released evidence, which the action server writes (CausalDiagnosis/mcp_servers/Environment_Actions_MCP/episode_state.py)


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


def _run_windows(log, oracle):
    """{order id: (close cycle, window)} for every close of a work order in the log (_worst_level_runs)."""
    return _worst_level_runs(log, oracle)[0]


def _worst_level_runs(log, oracle):
    """(closes, at_worst): closes is {order id: (close cycle, window)} for every close of a work order in the log, and at_worst
    the components that stand at their worst level at the start or reach it at an event of the log. The window is the one of the run at the
    worst level of the order's component that is open at the close, and the horizon when the component is not at its worst
    level then. The levels follow the log from the reset entry through the drops and the combination after every close and
    every expiry of an emergency hold. A run opens when its component reaches the last level of its degradation order, at
    the start or at an event of the log, and ends at the event that moves it off that level. Its window is the cycle at which
    it opens plus the patience of the reset configuration less one, capped at the horizon, the cycle at which the failure
    card would fire on it. The first run of a component takes the oracle's window end, which oracle.py computes by the same
    rule. A run that opens after the component failed takes the horizon, since the card fails a component once, and with
    the card off every run takes the horizon."""
    config = _reset_config(log)
    asset = _asset_truth(log)
    horizon = config.get("horizon")
    patience = (config.get("failure") or {}).get("patience", 0)
    degradation = get_profile(config.get("asset_id") or DEFAULT_ASSET, engine=False).degradation
    worst = {c: levels[-1] for c, levels in degradation.items()}
    first = {o["component"]: o.get("window_end") for o in oracle["work_orders"]}
    started, failed, window = set(), set(), {}

    def open_run(component, t):
        if component in failed or patience <= 0:
            end = horizon
        else:
            end = t + patience - 1 if horizon is None else min(t + patience - 1, horizon)
        window[component] = end if component in started else first.get(component, end)
        started.add(component)

    current = asset.levels(log[0]["hidden"])
    for c in asset.components:
        if current[c] == worst.get(c):
            open_run(c, 0)
    closes = {}
    for e in log[1:]:
        if e["event"] == "work_order_closed":
            closes[e["id"]] = (_time(e), window.get(e["component"], horizon))
        elif e["event"] == "failure":
            failed.add(e["component"])
        after = _next_combination(current, e, asset)
        for c in asset.components:
            if after[c] == worst.get(c) and current[c] != worst.get(c):
                open_run(c, _time(e))
            elif current[c] == worst.get(c) and after[c] != worst.get(c):
                window.pop(c, None)
        current = after
    return closes, frozenset(started)


def _asset_truth(log):
    """The truth model of the asset that the configuration of the log's reset entry names, or of hyd_1, the default asset of
    the configuration, when the entry names none, as in the synthetic logs of the tests: the truth model that the evidence
    kind of the asset declares, LevelsTruth when it declares none."""
    profile = get_profile(_reset_config(log).get("asset_id") or DEFAULT_ASSET, engine=False)
    return (kinds.kind_of(profile.evidence_kind).part("truth") or LevelsTruth)(profile)


def _reset_config(log):
    """The configuration of the log's reset entry, {} when the log has no reset entry or the entry no configuration."""
    first = log[0] if isinstance(log, list) and log and isinstance(log[0], dict) else {}
    return (first.get("config") or {}) if first.get("event") == "reset" else {}


def _first(log, name):
    return next((e for e in log if e["event"] == name), None)


def _held(end, asset):
    """{component: the level an emergency hold replaced} of the holds in force at the end, in component order, empty without
    a hold."""
    held = end.get("held") or {}
    return {c: held[c] for c in asset.components if c in held}


def _time(event):
    return event["t_end"] if event.get("t_end") is not None else event.get("t_start")


def _combination_after(log, hidden, asset):
    """The combination after the events of a log or of a part of it, followed through the hidden fields of the system
    events, such as the levels at the cycle of a write."""
    current = dict(hidden)
    for e in log:
        current = _next_combination(current, e, asset)
    return current


def _next_combination(current, e, asset):
    """The combination after one log entry: a deterioration transition sets the level of its component, and a close or the
    expiry of an emergency repair carries the whole combination after it."""
    if e["event"] in ("scheduled_drop", "hazard_drop"):
        return {**current, e["component"]: e["to_level"]}
    if e["event"] in ("work_order_closed", "emergency_expired") and e.get("combination_after") is not None:
        return asset.levels(e["combination_after"])
    return current


def _alarms_released(released):
    """Whether the agent was told of an alarm, by a change summary or by a notification."""
    return any((e["event"] == "summary" and e.get("alarms")) or e["event"] == "alarm" for e in released["events"])


def _closed_by_agent(log):
    return sorted({e["component"] for e in log if e["event"] == "work_order_closed" and e.get("source") == "agent"})


def _closing_summary(log):
    finish = _first(log, "finish")
    return finish.get("summary") or "" if finish else ""


def _cites_reading(log):
    """Whether the declaration of the log cites a reading released before it, an obs identifier of a read entry that precedes it.
    The identifier of an alarm, a failure mode or a work order is no reading, and a reading released later
    cannot have informed the declaration, although the provenance check holds the identifier."""
    index = next(i for i, e in enumerate(log) if e["event"] == "declare_insufficient_evidence")
    read = {f"obs/{o['n']}/{field}" for e in log[:index] if e["event"] == "read" for o in e["served"] for field in o["values"]}
    return any(str(i) in read for i in log[index].get("evidence_ids") or [])


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


def _released_identifiers(released, predictions, asset):
    """The identifiers the episode issued: obs/{n}/{field} for every released value, wo/{id} for every work order the
    released events name, alarm/{t}/{channel} for every alarm of a change summary or of a notification, fmea/{source}/{code}
    for the failure modes served, hist/{n} for every past order of a served fix history, and pred/{n} for every stored
    prediction."""
    valid = {f"obs/{o['n']}/{field}" for o in released["observations"] for field in o["values"]}
    valid.update(f"wo/{order_id}" for order_id in _released_work_orders(released))
    for event in released["events"]:
        if event["event"] == "summary":
            valid.update(f"alarm/{a['first_t']}/{a['channel']}" for a in event.get("alarms") or [])
        elif event["event"] == "alarm":
            valid.add(f"alarm/{event['t_end']}/{event['channel']}")
        elif event["event"] == "failure_modes":
            valid.update(asset.fmea_ids(event.get("component") or ""))
        elif event["event"] == "fix_history":
            valid.update(str(entry["id"]) for entry in event.get("history") or [] if isinstance(entry, dict) and entry.get("id"))
    valid.update(str(p["id"]) for p in (predictions or []) if isinstance(p, dict) and "id" in p)
    return valid


def _released_work_orders(released):
    """Every work order identifier the released events name: the orders the agent created, the orders executed or closed,
    and the closures a change summary lists. The agent may cite any of them, since the envelope and summarize_changes_since
    released them (CausalDiagnosis/docs/environment-actions-mcp.md, Evidence identifiers), not only the orders it created itself."""
    ids = set()
    for event in released["events"]:
        if event["event"] in ("work_order_created", "work_order_executed", "work_order_closed") and event.get("id"):
            ids.add(str(event["id"]))
        ids.update(str(o["id"]) for o in event.get("work_orders_closed") or [] if isinstance(o, dict) and o.get("id"))
    return ids
