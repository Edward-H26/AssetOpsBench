import math
from CausalDiagnosis.utils.public_rules.repair_completion import schedule


UNLIMITED = math.inf  # a budget, horizon or step cap the manifest leaves out, as on the base row where every tool is free and no clock runs


def channels_outside(bands, values):
    """The channel means of one served cycle that lie outside their public normal range, as [field, "below" or "above"] in field
    order; a field without a normal range, such as a std summary, is not judged."""
    out = []
    for name, value in sorted(values.items()):
        band = bands.get(name)
        if band is not None and not band["low"] <= value <= band["high"]:
            out.append([name, "below" if value < band["low"] else "above"])
    return out


def band_verdict(bands, cycles):
    """Whether a reading is consistent with a restored rig, from the normal ranges the profile serves and the channel means of every cycle
    the reading served after the close: it is not when any channel mean with a normal range lies outside it. This is the verification
    without the CausalDiagnosis MCP. A normal range spans the smallest to the largest mean of the stable healthy
    recordings, so a restored rig shows no mean outside and a fault still present shows at least one, and the rule holds no
    threshold of its own, so a change of the normal ranges needs no change here."""
    return not any(channels_outside(bands, values) for values in cycles)


def limit_of(value):
    """A budget, a horizon or a step cap as a number to compare against, unlimited when the manifest leaves it out."""
    return UNLIMITED if value is None else value


def read_prices(profile):
    """The read price per cycle of every sensor group, zero for a group the manifest does not price (the base
    row prices nothing)."""
    prices = (profile.get("prices") or {}).get("read") or {}
    return {group: prices.get(group, 0) or 0 for group in profile.get("sensor_groups") or {}}


def order_prices(profile):
    """The price of a work order per component, the part plus the repair charge, zero where the manifest prices neither."""
    prices = profile.get("prices") or {}
    parts, charge = prices.get("parts") or {}, prices.get("repair_charge", 0) or 0
    return {c: (parts.get(c, 0) or 0) + charge for c in profile.get("components") or []}


def durations_of(profile):
    """The durations of the manifest with zero for what it leaves out, since without a clock nothing takes time; a wait of
    one cycle still moves a clock by one cycle."""
    given = profile.get("durations") or {}
    components = profile.get("components") or []
    return {"read_per_cycle": given.get("read_per_cycle", 0) or 0, "advance_per_cycle": given.get("advance_per_cycle", 1) or 1,
            "lead_time": {c: (given.get("lead_time") or {}).get(c, 0) or 0 for c in components},
            "repair_downtime": {c: (given.get("repair_downtime") or {}).get(c, 0) or 0 for c in components}}


def profile_level(profile, component, level):
    """The level of the profile that a written level names, whatever type the profile gives its levels: the same value, or the
    same written form, so that a level written as text names the number the profile lists; None when the profile lists no such
    level for the component."""
    if isinstance(level, bool):
        return None
    for value in (profile.get("levels") or {}).get(component, ()):
        if value == level or str(value) == str(level).strip():
            return value
    return None


def visit_closes(profile, members):
    """{component: close cycle} of the orders of one visit, members (component, created cycle, close of a repair begun or None) in
    creation order, as the environment schedules the visit from what the profile publishes (servers/public_rules/visits.py): the
    repairs begun keep their close, and the technician starts the others at the latest creation cycle plus lead time among them,
    after the repairs begun, one after the other, the smaller priority number of the processes block first and ties in the order
    given. A profile without priorities repairs in the order given."""
    durations = durations_of(profile)
    priority = (profile.get("processes") or {}).get("priority") or {}
    return {c: close for c, _created, _execute, close in schedule(members, durations["lead_time"], durations["repair_downtime"], priority)}
