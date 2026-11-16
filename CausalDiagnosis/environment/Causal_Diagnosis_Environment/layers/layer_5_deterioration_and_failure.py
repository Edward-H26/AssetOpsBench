"""Big layer 5, deterioration and failure. The deterioration mode drops components level
by level during the episode, by a fixed schedule drawn at reset or by random hazard rates whose chance grows with the levels a
component sits below its healthy level. The collateral chance lets a replacement of a healthy component damage it, at 0 in
every preset. The failure card fails a component that has stayed at its worst level for its patience, open orders or not, adds
its penalty to the hidden ledger and ends the episode, a card that is on whenever the layer is on. Off, the mode is
none, the collateral chance 0 and the patience 0, the off parameters of the rule cards."""
import copy
import math

from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import is_number

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.small_switch import Switch

NAME = "deterioration"
FIELDS = ("schedule", "hazard", "failure")
MODES = ("none", "schedule", "hazard")
CARDS = ("schedule", "hazard")  # the deterioration cards, in the order the briefing states them
HAZARD_KEYS = ("mode", "base_rate", "depth_factor")  # the keys of the hazard card the mode sets, the collateral chance a switch of its own


def _mode(mode):
    """The schedule card and the hazard keys of the mode under a deterioration mode: both off for none, and for schedule or
    hazard that card with the asset's parameters in its drawing mode and the other card off."""
    def values(profile):
        schedule, hazard = rules.kind_of("schedule").off(profile), rules.kind_of("hazard").off(profile)
        if mode == "schedule":
            schedule = {**profile.config_fields()["schedule"], "mode": "seeded"}
        elif mode == "hazard":
            hazard = {**profile.config_fields()["hazard"], "mode": "stochastic"}
        return {"schedule": schedule, **{f"hazard.{key}": hazard[key] for key in HAZARD_KEYS}}
    return values


def _parameters(mapping, profile):
    """A mapping setting names the mode, schedule or hazard, and parameters of that card merged over the asset's."""
    mode = mapping.get("mode")
    extra = {k: v for k, v in mapping.items() if k != "mode"}
    keys = {"schedule": set(rules.kind_of("schedule").keys), "hazard": set(HAZARD_KEYS)}.get(mode, set()) - {"mode"}
    if mode not in CARDS or not set(extra) <= keys:
        raise ValueError(f"the small switch deterioration takes on, off, {', '.join(MODES)} or a mapping of the mode schedule or "
                         f"hazard with parameters of that card, not {mapping!r}")
    values = _mode(mode)(profile)
    if mode == "schedule":
        values["schedule"] = {**values["schedule"], **copy.deepcopy(extra)}
    else:
        values.update({f"hazard.{key}": copy.deepcopy(value) for key, value in extra.items()})
    return values


def _failure_on(profile):
    """The asset's failure card with the strict end, so that a failure ends the episode wherever the layer is on,
    while the asset file keeps its card."""
    return {"failure": {**profile.config_fields()["failure"], "strict": True}}


def _failure_parameters(mapping, profile):
    """A mapping setting of the failure switch. {"scale": x}, x a positive finite number, gives the on card with the patience
    ceil(x times the asset's patience), the rounding the optional pilot of cluster_run_checks.py also uses, so that no preset types a patience; a
    scaled patience below one cycle, or at or above the asset's horizon, where the failure rule never fires, is refused by
    name. Any other mapping replaces entries of the on card."""
    card = _failure_on(profile)["failure"]
    if "scale" not in mapping:
        if not mapping or not set(mapping) <= set(card):
            raise ValueError(f"the small switch failure takes on, off, the mapping {{'scale': x}} or a mapping of entries of its "
                             f"card {sorted(card)}, not {mapping!r}")
        return {"failure": {**card, **copy.deepcopy(mapping)}}
    scale = mapping["scale"]
    if set(mapping) != {"scale"} or not is_number(scale) or scale <= 0:
        raise ValueError(f"the small switch failure takes the mapping {{'scale': x}} with x a positive number, not {mapping!r}")
    # rounded before the ceiling, since a product such as 0.3 times 10 is 3.0000000000000004 in floating point
    patience = math.ceil(round(scale * card["patience"], 9))
    horizon = profile.config_fields()["horizon"]
    if not 1 <= patience < horizon:
        raise ValueError(f"the scale {scale} gives the failure card the patience {patience}, outside 1 to {horizon - 1} cycles, and a "
                         f"failure rule whose patience reaches the horizon {horizon} never fires")
    return {"failure": {**card, "patience": patience}}


SWITCHES = {s.name: s for s in (
    Switch("deterioration", ("schedule", *(f"hazard.{key}" for key in HAZARD_KEYS)), off=_mode("none"), on=_mode("hazard"),
           modes={m: _mode(m) for m in MODES}, parameters=_parameters),
    Switch("collateral", ("hazard.collateral",), off=lambda profile: {"hazard.collateral": rules.kind_of("hazard").off(profile)["collateral"]},
           on=lambda profile: {"hazard.collateral": profile.config_fields()["hazard"]["collateral"]}),
    Switch("failure", ("failure",), off=lambda profile: {"failure": rules.kind_of("failure").off(profile)}, on=_failure_on,
           parameters=_failure_parameters),
)}


def mode(config):
    """The deterioration mode of a configuration: hazard while the hazard draws, schedule while the schedule does, else none."""
    if config.hazard["mode"] == "stochastic":
        return "hazard"
    return "schedule" if config.schedule["mode"] != "none" else "none"


def cards(config):
    """The deterioration cards the episode switches on, whose parameters the processes block publishes: the
    schedule card while it draws transitions, the hazard card while it draws random drops or collateral damage."""
    schedule, hazard = config.schedule, config.hazard
    on = {"schedule": schedule["mode"] != "none" and schedule["components"] > 0,
          "hazard": (hazard["mode"] == "stochastic" and hazard["base_rate"] > 0) or hazard["collateral"] > 0}
    return [card for card in CARDS if on[card]]


def failure_on(config):
    """A patience of 0 cycles switches the failure rule off, the off value of its rule card."""
    return config.failure["patience"] > 0


def failure_charge(config):
    """(penalty, strict) of a failure: the penalty it adds to the hidden ledger and whether it ends the episode."""
    return config.failure["penalty"], config.failure["strict"]


def degradation_order(config, component):
    """The levels of a component from healthy to worst, which the schedule walks down, as the profile of the configured
    asset lists them, for hyd_1 cooler 100, 20, 3; valve 100, 90, 80, 73; pump 0, 1, 2; accumulator 130, 115, 100, 90."""
    return list(get_profile(config.asset_id, engine=False).degradation[component])


def worst_level(config, component):
    return degradation_order(config, component)[-1]


def draw_schedule(rng, config, hidden):
    """The scheduled drops of the episode: [{"t", "component", "to_level"}] in time order, empty in mode none.

    The components are drawn among those healthy at t0 first, so that a scheduled episode changes, and among the faulty
    ones only when more components are asked for than are healthy. Each drawn component gets a first drop cycle uniform
    in first_drop and then a drop every interval cycles until its worst level."""
    rule = config.schedule
    if rule["mode"] == "none" or rule["components"] == 0:
        return []
    levels = dict(zip(config.components, hidden))
    healthy = [c for c in config.components if levels[c] == config.healthy[c]]
    faulty = [c for c in config.components if levels[c] != config.healthy[c]]
    chosen = rng.sample(healthy, min(rule["components"], len(healthy)))
    chosen += rng.sample(faulty, rule["components"] - len(chosen))
    drops = []
    for c in chosen:
        first = rng.randint(*rule["first_drop"])
        order = degradation_order(config, c)
        below = order[order.index(levels[c]) + 1:]
        drops += [{"t": first + i * rule["interval"], "component": c, "to_level": level} for i, level in enumerate(below)]
    return sorted(drops, key=lambda d: (d["t"], config.components.index(d["component"])))


def hazard_rate(config, component, level):
    """The chance per cycle that a component at a level drops one level under the stochastic hazard: base_rate times one
    plus depth_factor times the levels it sits below its healthy level, so that running degraded brings the next drop
    sooner."""
    depth = degradation_order(config, component).index(level)
    return config.hazard["base_rate"] * (1 + config.hazard["depth_factor"] * depth)


def draw_hazard(rng, config):
    """The uniforms of the hazard, drawn once at reset: drop[t][i] decides the stochastic drop of component i during cycle
    t, and collateral[t][i] whether the replacement of the healthy component i that closes at cycle t damages it. Because
    the table is fixed by the seed, a seed and an action sequence replay exactly, while another action sequence, which
    leaves the components at other levels, meets other drops. rng is a generator of its own."""
    n = len(config.components)
    return {"drop": [[rng.random() for _ in range(n)] for _ in range(config.horizon)],
            "collateral": [[rng.random() for _ in range(n)] for _ in range(config.horizon + 1)]}


def hazard_drop(clock, component, t, cause):
    """One level down for a component, by the stochastic hazard or as collateral damage of a replacement; a drop sets
    the level itself, so a hold ends without expiry."""
    order = degradation_order(clock.config, component)
    to_level = order[order.index(clock.levels[component]) + 1]
    event = {"event": "hazard_drop", "t": t, "component": component, "from_level": clock.levels[component], "to_level": to_level, "cause": cause}
    clock.hazard_drops.append({"t": t, "component": component, "to_level": to_level, "cause": cause})
    clock.temporary.pop(component, None)
    clock.set_level(component, to_level, t)
    return event


def drops(clock, t):
    """The deterioration of cycle t: the scheduled drops of the cycle, then under the stochastic hazard the drop of every
    component whose uniform of the cycle falls below its hazard rate."""
    events = []
    for drop in clock.schedule:
        if drop["t"] == t:
            events.append({"event": "scheduled_drop", "t": t, "component": drop["component"],
                           "from_level": clock.levels[drop["component"]], "to_level": drop["to_level"]})
            clock.temporary.pop(drop["component"], None)  # the drop sets its absolute level, so a hold ends without expiry
            clock.set_level(drop["component"], drop["to_level"], t)
    if clock.config.hazard["mode"] == "stochastic":
        for i, c in enumerate(clock.config.components):
            if clock.levels[c] != worst_level(clock.config, c) and clock.hazard["drop"][t][i] < hazard_rate(clock.config, c, clock.levels[c]):
                events.append(hazard_drop(clock, c, t, "hazard"))
    return events


def collateral(clock, component, t):
    """The collateral damage of the replacement of a healthy component that closes at cycle t, a hazard drop when the
    component's uniform of the cycle falls below the collateral chance, off at 0."""
    if clock.hazard["collateral"][t][clock.config.components.index(component)] < clock.config.hazard["collateral"]:
        return [hazard_drop(clock, component, t, "collateral")]
    return []


def failure_check(clock, t):
    """A component at its worst level for `patience` consecutive cycles fails once, unless a patience of 0 switches the rule
    off. The count runs whether or not a work order on the component is open and starts again only when the component leaves
    its worst level, which an effective repair or an emergency hold does, so a repair averts the failure only by closing by
    the window end of oracle.py, the deadline that the timing check of the verifier reads as well."""
    events = []
    if not failure_on(clock.config):
        return events
    for c in clock.config.components:
        if clock.levels[c] == worst_level(clock.config, c):
            clock.streak[c] += 1
        else:
            clock.streak[c] = 0
        if clock.streak[c] >= clock.config.failure["patience"] and c not in clock.failed:
            clock.failed.add(c)
            events.append({"event": "failure", "t": t, "component": c})
    return events
