"""Public work-order feedback shared by every agent, independent of diagnostic advice.

Completion means the technician finished the order, not that the repair was effective.
Only public scheduling fields enter these messages; no hidden condition or repair outcome does.
"""
import json


ORDER_FIELDS = ("id", "component", "created_t", "expected_close_t", "status", "closed_t")


def public_order(order):
    return {key: order.get(key) for key in ORDER_FIELDS}


def progress(envelope, *, wait_available):
    """The next observation may arrive before a scheduled close, for example an alarm.
    Waiting hands control to the existing event clock; it does not force a successful repair.
    """
    orders = [public_order(o) for o in envelope.get("open_work_orders", [])]
    closed = [{key: n.get(key) for key in ("id", "component", "t")}
              for n in envelope.get("notifications", []) if n.get("event") == "work_order_closed"]
    if not orders and not closed:
        return None
    message = ("Existing work orders remain accepted and pending; a duplicate request does not cancel them. "
               "You may investigate other issues" if orders else "Work orders completed. Completion does not establish equipment recovery")
    if orders and wait_available:
        message += ", or call wait_for_notification to yield until the next public event or timeout"
    if closed:
        message += ". Check post-repair measurements to assess recovery"
    return {"current_cycle": envelope["time"], "open_work_orders": orders,
            "completed_in_this_reply": closed, "message": message + "."}


def accepted_reply(text, order, cycle):
    """Keep the WO result shape while making its lifecycle explicit at creation."""
    body = json.loads(text)
    # The producer already synced the WO database to this engine state. The original
    # generate_work_order result was captured before that sync and still says WAPPR.
    status = {"open": "APPR", "in_repair": "INPRG", "closed": "COMP"}.get(order.get("status"))
    if status:
        body["status"] = status
        if isinstance(body.get("work_order"), dict):
            body["work_order"]["status"] = status
    body["operation_status"] = "accepted"
    body["maintenance"] = {"current_cycle": cycle, "work_order": public_order(order)}
    body["message"] = (f"Work order {order['id']} accepted. " +
                       ("The order completed; equipment recovery still requires verification."
                        if order.get("status") == "closed" else
                        f"It is pending, not failed; scheduled completion is cycle {order.get('expected_close_t')}. "
                        "Creation does not mean the repair has completed."))
    return json.dumps(body, indent=2, ensure_ascii=False)
