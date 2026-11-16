import json
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS as AOB_SERVERS, CAUSAL_SERVER
from . import formatting


MEANS_SHOWN, TOP_SHOWN, SHORT_ARGUMENT = 3, 3, 48
RECORD_FIELDS = ("asset_id", "timestamp")  # the fields of a history row that carry no value


def describe_args(call):
    if not isinstance(call.args, dict):
        return f"`{call.tool}({json.dumps(call.args)})`"
    parts = []
    for key, value in call.args.items():
        if key == "evidence_ids":
            parts.append(f"evidence_ids=<{len(formatting.id_list(value))} ids>")
        elif key == "summary" and isinstance(value, str):
            parts.append(f"summary=<{len(value)} characters>")
        elif key == "sensors" and isinstance(value, list):
            parts.append(f"sensors=<{len(value)} fields>")
        else:
            text = json.dumps(value)
            parts.append(f"{key}={text if len(text) <= SHORT_ARGUMENT else text[:SHORT_ARGUMENT - 3] + '...'}")
    return f"`{call.tool}({', '.join(parts)})`"


def describe_answer(call, ep):
    if call.error:
        return f"error {formatting.quoted(call.error, 160)}"
    if call.answer is None:
        return f"answer {formatting.quoted(call.text, 160)}" if call.text else "no answer"
    if call.server == CAUSAL_SERVER:
        return diagnostics_answer(call.tool, call.answer, ep.components)
    if call.server in AOB_SERVERS:
        return aob_answer(call, ep)
    return action_answer(call, ep.bands, ep.components)


def aob_answer(call, ep):
    """What an answer of AssetOpsBench's servers or of the task server said. The task tools answer with the engine's result
    as the rig's action server does, so action_answer tells them, while the reads and the order of iot, fmsr and wo, the
    clock and the wait are told from AssetOpsBench's text."""
    data, describe = call.answer, AOB_ANSWERS.get(call.tool)
    if "text" in data:
        text = f"answer {formatting.quoted(data['text'], 160)}"
    elif describe is None or call.result is None:
        return action_answer(call, ep.bands, ep.components)
    else:
        text = describe(call, ep)
    return text + (", the episode is over" if data.get("episode_over") else "")


def history_answer(call, ep):
    """The rows of a history window with the group its sensors make up and the last row's means against their normal ranges, which
    the released rules key by the mean's name in the rig's record (`<channel>_mean`)."""
    rows = call.result.get("observations") or []
    if not rows:
        return "no row in the window"
    span = row_span(rows) + (", the last shown" if len(rows) > 1 else "")
    means = [(f, v) for f, v in rows[-1].items()
             if f not in RECORD_FIELDS and not f.endswith("_std") and isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = [m for m in means if formatting.outside_band(f"{m[0]}_mean", m[1], ep.bands)]
    shown = (out + [m for m in means if m not in out])[:MEANS_SHOWN]
    return (f"{ep.group_of(call) or 'sensors as asked'}, {formatting.plural(len(rows), 'row')} at {span}, {len(out)} of {len(means)} means outside the normal range, "
            + ", ".join(f"{formatting.tag(f)} {formatting.num(v)} {'out' if formatting.outside_band(f'{f}_mean', v, ep.bands) else 'in'}" for f, v in shown))


def row_span(rows):
    """The timestamps of the first and the last row of a history window, one when they are the same."""
    first, last = rows[0].get("timestamp"), rows[-1].get("timestamp")
    return first if first == last else f"{first} to {last}"


def aob_order_answer(call, ep):
    result = call.result
    asset = (result.get("work_order") or {}).get("assetnum")
    component = ep.children.get(asset)
    return (f"{result.get('wonum')} on " + (f"the {component} ({asset})" if component else str(asset))
            + f", status {result.get('status')}")


def workorders_answer(call, ep):
    result = call.result
    orders = result.get("work_orders") or []
    statuses = {}
    for order in orders:
        statuses[order.get("status")] = statuses.get(order.get("status"), 0) + 1
    return (f"{formatting.plural(result.get('total', len(orders)), 'work order')} at site {result.get('site_id')}"
            + (", status " + formatting.joined(f"{status} for {count}" for status, count in statuses.items()) if statuses else ""))


AOB_ANSWERS = {
    "history": history_answer,
    "generate_work_order": aob_order_answer,
    "list_workorders": workorders_answer,
    "stream_extent": lambda c, ep: (f"{formatting.plural(c.result.get('total_records'), 'recorded row')} from {c.result.get('start_time')} to "
                                    f"{c.result.get('end_time')}"),
    "get_failure_modes": lambda c, ep: f"{len(c.result.get('failure_modes') or [])} failure modes of the {c.result.get('asset_class')}",
    "get_current_time": lambda c, ep: (f"cycle {formatting.num(c.result.get('time'))} at {c.result.get('current_datetime')}, horizon "
                                       f"{formatting.plural(c.result.get('horizon'), 'cycle')}"),
    "wait_for_notification": lambda c, ep: (f"clock at cycle {formatting.num(c.result.get('time'))} after a wait of {formatting.plural(c.result.get('waited'), 'cycle')}, "
                                            f"ended by {c.result.get('ended_by')}"),
}


def action_answer(call, bands, components):
    data = call.answer
    if "refused" in data:
        text = f"refused, reason {formatting.quoted(data['refused'])}"
    else:
        result = call.result or {}
        describe = {"read_sensors": read_answer, "wait_for_notification": advance_answer,
                    "summarize_changes_since": summary_answer,
                    "get_asset_profile": lambda r, d, b, c: "the asset profile of part 1",
                    "get_fix_history": lambda r, d, b, c: f"{formatting.plural(len(r.get('history') or []), 'work order')} before the start",
                    "get_failure_modes": lambda r, d, b, c: (f"{len(r.get('failure_modes') or [])} failure modes of "
                                                              + (f"the {r['component']}" if r.get("component") else "every component")),
                    "submit_diagnosis": lambda r, d, b, c: (f"diagnosis {formatting.short_levels(r.get('diagnosis'), c)} stored with "
                                                             f"{len(r.get('evidence_ids') or [])} evidence ids"),
                    "declare_insufficient_evidence": lambda r, d, b, c: (f"declaration stored, separating test "
                                                                          f"{(r.get('separating_test') or {}).get('sensor_group')} for "
                                                                          f"{formatting.plural((r.get('separating_test') or {}).get('cycles'), 'cycle')}"),
                    "create_work_order": order_answer,
                    "finish": lambda r, d, b, c: "summary stored"}.get(call.tool)
        text = describe(result, data, bands, components) if describe else formatting.quoted(json.dumps(result), 160)
    if data.get("episode_over"):
        text += ", the episode is over"
    return text


def read_answer(result, data, bands, components):
    served, stopped = result.get("served") or [], result.get("stopped") or 0
    if not served:
        return f"{result.get('sensor_group')}, no cycle served, {formatting.plural(stopped, 'cycle')} stopped for a repair"
    first, last = served[0], served[-1]
    span = (f"obs {first['n']} at cycle {first['t']}" if len(served) == 1
            else f"obs {first['n']} to {last['n']} at cycles {first['t']} to {last['t']}, the last shown")
    means = [(f, v) for f, v in last["values"].items() if f.endswith("_mean")]
    out = [m for m in means if formatting.outside_band(*m, bands)]
    shown = (out + [m for m in means if not formatting.outside_band(*m, bands)])[:MEANS_SHOWN]
    text = (f"{result.get('sensor_group')} {span}, {len(out)} of {len(means)} means outside the normal range, "
            + ", ".join(f"{formatting.tag(f)} {formatting.num(v)} {'out' if formatting.outside_band(f, v, bands) else 'in'}" for f, v in shown))
    if stopped:
        text += f", {formatting.plural(stopped, 'cycle')} stopped for a repair"
    return text


def advance_answer(result, data, bands, components):
    """The clock and the open orders after a wait, whose notifications the table shows in their own column."""
    orders = data.get("open_work_orders") or []
    return (f"clock at cycle {formatting.num(result.get('time'))}, "
            + ("open orders " + formatting.joined(f"{o['id']} {str(o['status']).replace('_', ' ')}" for o in orders) if orders else "no open order"))


def notifications_text(envelope):
    """The public events the action server returned with a call, blank when it reported none at all."""
    events = envelope.get("notifications")
    if events is None:
        return ""
    return formatting.joined(event_text(e) for e in events) or "none"


def event_text(event):
    if event.get("message"):
        return f"{str(event['message']).rstrip('.')} (cycle {formatting.num(event.get('t'))})"
    text = str(event.get("event", "")).replace("_", " ")
    if event.get("id"):
        text += f" {event['id']}"
    if event.get("component"):
        text += f" on the {event['component']}"
    return text + f" at cycle {formatting.num(event.get('t_end', event.get('t')))}"


def summary_answer(result, data, bands, components):
    alarms, closed = result.get("alarms") or [], result.get("work_orders_closed") or []
    text = f"since cycle {formatting.num(result.get('since'))}, {formatting.plural(len(alarms), 'alarm')}"
    if alarms:
        text += f" from cycle {min(a['first_t'] for a in alarms)}"
    return text + (", closed " + formatting.joined(f"{o['id']} on the {o['component']} at cycle {formatting.num(o.get('closed_t'))} by the {o.get('source')}"
                                        for o in closed) if closed else ", no order closed")


def order_answer(result, data, bands, components):
    order = result.get("work_order") or {}
    return (f"{order.get('id')} on the {order.get('component')}, created at cycle {formatting.num(order.get('created_t'))}, expected to close at "
            f"cycle {formatting.num(order.get('expected_close_t'))}")


def scored_combination(data, components):
    """The most probable combination after the repair, which verify_outcome scores the reading against, with its mass,
    or nothing for a result that names none."""
    if not data.get("predicted_combination"):
        return ""
    return (f", predicted combination {formatting.short_levels(data['predicted_combination'], components)}, the most probable after the repair, with mass "
            f"{formatting.prob(data.get('predicted_mass'))}")


def diagnostics_answer(tool, data, components):
    if "error" in data and len(data) == 1:
        return f"error {formatting.quoted(data['error'], 160)}"
    if tool == "rank_root_causes":
        top = [f"{formatting.short_levels(c.get('combination'), components)} {formatting.prob(c.get('probability'))}" for c in (data.get("candidates") or [])[:TOP_SHOWN]]
        text = f"top {len(top)} of {formatting.num(data.get('n_candidates'))}, " + ", ".join(top) + f", read first {formatting.num(data.get('read_first'))}"
        test = data.get("separating_test")
        if test:
            text += f", separating test {test.get('sensor_group')} for {formatting.plural(test.get('cycles'), 'cycle')} at price {formatting.num(test.get('price'))}"
        return text
    if tool == "verify_outcome":
        return (f"{data.get('prediction_id')} status {formatting.quoted(data.get('status'))}, consistent {formatting.num(data.get('consistent'))}, recovery predicted "
                f"{formatting.prob(data.get('recovery_predicted'))}, mass of the healthy combination in the belief now "
                f"{formatting.prob(data.get('recovery_posterior'))}" + scored_combination(data, components))
    if tool == "estimate_intervention_effect":
        return (f"{data.get('prediction_id')} for `{data.get('action')}`, predicted recovery {formatting.prob(data.get('recovery'))}, price {formatting.num(data.get('price'))}, "
                f"duration {formatting.plural(data.get('duration'), 'cycle')}")
    return f"message {formatting.quoted(data.get('message', ''), 200)}"
