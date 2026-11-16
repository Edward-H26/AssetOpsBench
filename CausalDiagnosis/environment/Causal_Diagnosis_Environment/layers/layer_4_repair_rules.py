"""Big layer 4, the repair rules: a repair takes only when the components it
depends on are at their healthy level at its close, the standard orders created one after another with no other call
between them form one technician visit that repairs them in priority order with one visit charge, an emergency order arrives sooner at a surcharge but restores
its component only for a hold, and the fix history lists the work orders on the rig before the episode. The repair manual
lookup_fix states them. Off, nothing depends on anything, every priority is 1, a visit costs nothing, the emergency hold is
0 and the history is empty, the off parameters of the rule cards. An emergency order needs the lead times of the advanced
real-time clock, since it arrives sooner than a standard order only where standard orders wait their lead time, and the
dependencies need the priorities, since a visit repairs a dependency first only by its smaller priority number."""
from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.utils.public_rules.visits import visit_schedule

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_5_deterioration_and_failure as deterioration
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.small_switch import Switch, asset_value

NAME = "repair_rules"
FIELDS = ("dependencies", "priority", "prices.visit", "emergency", "fix_history")
CARDS = ("fix_history", "emergency", "dependencies", "visits")  # the rule cards of the layer, in the order the briefing states them
HISTORY_PREFIX = "hist/"


def _off(kind):
    return rules.kind_of(kind).off


SWITCHES = {s.name: s for s in (
    Switch("dependencies", ("dependencies",), off=lambda profile: {"dependencies": _off("dependencies")(profile)["dependencies"]},
           on=asset_value("dependencies"), needs=(("priority", "a visit repairs the components a repair depends on before it only "
                                                               "where their priority numbers are smaller"),)),
    Switch("priority", ("priority",), off=lambda profile: {"priority": _off("dependencies")(profile)["priority"]}, on=asset_value("priority")),
    Switch("visit", ("prices.visit",), off=lambda profile: {"prices.visit": _off("visits")(profile)["charge"]},
           on=lambda profile: {"prices.visit": profile.config_fields()["prices"]["visit"]}),
    Switch("emergency", ("emergency",), off=lambda profile: {"emergency": _off("emergency")(profile)}, on=asset_value("emergency"),
           needs=(("lead_time", "an emergency order arrives sooner than a standard order only where standard orders wait their "
                                "lead time"),)),
    Switch("fix_history", ("fix_history",), off=lambda profile: {"fix_history": _off("fix_history")(profile)}, on=asset_value("fix_history")),
)}


def emergency_on(config):
    """A hold of 0 cycles switches the emergency mode off, the off value of its rule card."""
    return config.emergency["hold"] > 0


def repair_card_on(config):
    """The dependency or the visit card is on, the two repair rules the briefing states for a visit: a component depends
    on another, or a visit has a charge. Visit semantics follow this card and not the clock."""
    return bool(config.dependencies) or config.prices["visit"] > 0


def depends_on(config, component):
    """The components that must be at their healthy level when an order on the component closes; none when the
    mapping is empty, which switches the repair rule off."""
    return list(config.dependencies.get(component, []))


def cards(config):
    """The repair rule cards the episode switches on, in the order of CARDS."""
    history = config.fix_history["healthy_probability"] > 0 or config.fix_history["degraded_probability"] > 0
    on = {"fix_history": history, "emergency": emergency_on(config), "dependencies": bool(config.dependencies),
          "visits": config.prices["visit"] > 0}
    return [card for card in CARDS if on[card]]


def draw_fix_history(rng, config, hidden):
    """The past work orders on the rig before t0, oldest first: [{"id", "component", "closed_t", "mode", "outcome"}] with
    the identifiers hist/1, hist/2 and so on and negative closing cycles. A component at its healthy level at t0 has one
    past order with probability healthy_probability, closed a number of cycles drawn in healthy_ago before t0; a degraded
    one with probability degraded_probability, closed a number drawn in degraded_ago before. The default windows coincide,
    so the closing date tells nothing about the kind and only the two probabilities weigh the hypotheses. rng is a
    generator of its own, so the history changes no other draw of the episode."""
    rule = config.fix_history
    entries = []
    for c, level in zip(config.components, hidden):
        kind = "healthy" if level == config.healthy[c] else "degraded"
        if rng.random() < rule[f"{kind}_probability"]:
            entries.append({"component": c, "closed_t": -rng.randint(*rule[f"{kind}_ago"])})
    entries.sort(key=lambda e: (e["closed_t"], config.components.index(e["component"])))
    return [{"id": f"{HISTORY_PREFIX}{n}", **e, "mode": "standard", "outcome": "closed"} for n, e in enumerate(entries, 1)]


def schedule_visit(config, created_t, members, after=None):
    """The repairs of one visit created at one cycle (servers/public_rules/visits.py): the technician arrives at the creation
    cycle plus the longest lead time of the visit, or at the close of the repair begun before them (after), and executes the
    repairs back to back, smaller priority number first, each with its own repair downtime. members are (component, mode) pairs
    in creation order; returns (index into members, execute_t, close_t) in execution order."""
    return schedule_orders(config, [(c, mode, created_t) for c, mode in members], after)


def schedule_orders(config, members, after=None):
    """The repairs of the orders of one visit that have not begun, members (component, mode, created cycle) in creation order,
    after the close of the last repair of the visit begun (after, or None), by the visit rule of servers/public_rules/visits.py;
    returns (index into members, execute_t, close_t) in execution order."""
    return visit_schedule([(created_t, config.lead_time(c, mode), config.durations["repair_downtime"][c], config.priority[c])
                           for c, mode, created_t in members], after)


def close(clock, order, t):
    """Close an agent order. Its component is restored only when every component it depends on is at its healthy level;
    otherwise the order is ineffective, charged as usual with the level unchanged, and the hidden event order_ineffective
    names the degraded dependencies. An emergency order restores for emergency.hold cycles and keeps the level it
    replaced for the expiry, and a standard order ends such a hold. The replacement of a healthy component may damage it,
    the collateral damage of the deterioration layer."""
    config = clock.config
    c, healthy = order.component, config.healthy[order.component]
    degraded = [d for d in depends_on(config, c) if clock.levels[d] != config.healthy[d]]
    replaced_healthy = clock.levels[c] == healthy and c not in clock.temporary
    ineffective = bool(degraded) and not replaced_healthy
    if not degraded:
        if order.mode == "emergency":
            replaced = clock.temporary.get(c, {}).get("level", clock.levels[c])
            if replaced != healthy:
                clock.temporary[c] = {"level": replaced, "expires_t": t + config.emergency["hold"], "id": order.id}
        else:
            clock.temporary.pop(c, None)
        clock.set_level(c, healthy, t)
    events = [clock.mark_closed(order, t)]
    if ineffective:
        events.append({"event": "order_ineffective", "t": t, "id": order.id, "component": c, "degraded_dependencies": degraded})
    if replaced_healthy:
        events += deterioration.collateral(clock, c, t)
    return events


def expire_holds(clock, t):
    """The emergency repairs whose hold ends at cycle t expire: the component returns to the level it had before the
    repair, the hidden event emergency_expired."""
    events = []
    for c in [c for c in clock.config.components if c in clock.temporary and clock.temporary[c]["expires_t"] <= t]:
        hold = clock.temporary.pop(c)
        clock.set_level(c, hold["level"], t)
        events.append({"event": "emergency_expired", "t": t, "id": hold["id"], "component": c, "to_level": hold["level"],
                       "combination_after": list(clock.combination)})
    return events


def manual(config, component, channels):
    """The answer of lookup_fix: the standard fix of a component with its charges and times, its priority, the
    components that must be at their healthy level for the fix to take, the channels that indicate it, the expected effect
    and, when the emergency
    mode is on, the emergency option, since a manual should not offer an order the episode refuses. channels are the
    public fields of the channels that indicate the component."""
    c = config
    needs = depends_on(c, component)
    condition = (f", provided that {' and '.join(f'the {d}' for d in needs)} {'is' if len(needs) == 1 else 'are'} at the healthy "
                 "level at the close, else the order closes without effect" if needs else "")
    downtime = c.durations["repair_downtime"][component]
    manual = {
        "component": component,
        "standard": {"mode": "standard", "fix": c.prices["fix_kind"][component], "part": c.prices["parts"][component],
                     "repair_charge": c.prices["repair_charge"], "price": c.order_price(component), "visit_charge": c.prices["visit"],
                     "lead_time": c.lead_time(component), "repair_downtime": downtime},
        "priority": c.priority[component],
        "dependencies": needs,
        "indicating_channels": list(channels),
        "effect": (f"At the close of the order the {component} returns to its healthy level {c.healthy[component]} and the channels "
                   f"that indicate it ({', '.join(channels)}) return to their normal ranges{condition}."),
    }
    if emergency_on(c):
        manual["emergency"] = {"mode": "emergency", "price": c.order_price(component, "emergency"), "visit_charge": c.prices["visit"],
                               "lead_time": c.lead_time(component, "emergency"), "repair_downtime": downtime, "hold": c.emergency["hold"],
                               "effect": (f"A temporary restoration: the {component} stays at its healthy level for {c.emergency['hold']} "
                                          "cycles after the close and then returns to its level before the repair, unless a standard "
                                          "order on it closes in between.")}
    return manual
