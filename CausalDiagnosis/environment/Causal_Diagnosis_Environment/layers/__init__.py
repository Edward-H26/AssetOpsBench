"""The six big layers of the environment (CausalDiagnosis/docs/environment.md, section Layers).

The base is always on, and five big layers join it one at a time on the ladder of the environment table: priced reads and
writes, the real-time clock, the advanced real-time clock, the repair rules, and deterioration and failure. Each big layer
is one module that owns its configuration paths, the small switches that set them one by one with their off and on values,
and the rules of the layer, which the engine, the rig's action server and the metering layer of the AssetOpsBench rows call.
OWNER gives every field of EpisodeConfig, or every key of the two tables that layers share, exactly one owner.

A preset is a composition: the big layers it turns on, the small switches it sets otherwise, and the fields of the base it
sets. compose starts from the asset's default configuration with every switch of every layer at its off value, turns the
named layers on, applies the switch settings and the base fields, and validates the result, so a layer a preset leaves out
stays off whatever the asset's defaults say. A small switch that has no meaning without a switch of a lower layer, such as
the emergency order without the lead times, makes the loader refuse a composition that turns it on alone, and so does a
composition whose cost unit does not fit it (unit_problems).
"""
from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_3_advanced_real_time as advanced_real_time, layer_0_base as base, layer_5_deterioration_and_failure as deterioration, layer_1_priced as priced, layer_2_real_time as real_time, layer_4_repair_rules as repair_rules
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.small_switch import set_path

LAYERS = (base, priced, real_time, advanced_real_time, repair_rules, deterioration)
LADDER = tuple(layer.NAME for layer in LAYERS)
SPLIT = ("prices", "durations")  # the two tables whose keys belong to different layers
OWNER = {path: layer.NAME for layer in LAYERS for path in layer.FIELDS}
SWITCHES = {name: layer for layer in LAYERS for name in layer.SWITCHES}  # small switch -> its big layer
PRESET_KEYS = ("name", "description", "layers", "switches")


def _layer(name):
    return next(layer for layer in LAYERS if layer.NAME == name)


def _owner_of_field(field):
    """The big layers that own a field of EpisodeConfig or keys of it."""
    return sorted({OWNER[path] for path in OWNER if path.partition(".")[0] == field})


def problems(spec):
    """Every problem of a composition at once, as the configuration validates its fields."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig  # imported here, since the configuration imports this package

    if not isinstance(spec, dict):
        return ["a composition must be a mapping"]
    out = []
    names = spec.get("layers", [])
    if not isinstance(names, list):
        out.append("layers must be a list of big layers")
        names = []
    for name in names:
        if name == base.NAME:
            out.append("the base is always on and is not named under layers")
        elif name not in LADDER:
            out.append(f"unknown big layer {name!r}, one of {list(LADDER[1:])}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        out.append(f"big layers {repeated} are named more than once")
    switches = spec.get("switches", {})
    if not isinstance(switches, dict):
        out.append("switches must map small switches to their settings")
        switches = {}
    unknown = sorted(n for n in switches if n not in SWITCHES)
    if unknown:
        out.append(f"unknown small switch {unknown}, one of {sorted(SWITCHES)}")
    fields = {f for f in EpisodeConfig.__dataclass_fields__}
    for key in spec:
        if key in PRESET_KEYS:
            continue
        if key not in fields:
            out.append(f"{key} is no field of EpisodeConfig")
        elif OWNER.get(key) != base.NAME:
            owners = _owner_of_field(key)
            through = sorted(n for n, layer in SWITCHES.items() if layer.NAME in owners and any(
                p.partition(".")[0] == key for p in layer.SWITCHES[n].paths))
            out.append(f"{key} belongs to the big layer {' and '.join(owners)}, set it through the switches {through}")
    return out


def unmet(data, profile):
    """The needs of the small switches that a configuration's JSON turns on without the lower switch they need, each with
    the big layer of the missing switch and the reason."""
    out = []
    for layer in LAYERS:
        for switch in layer.SWITCHES.values():
            for need, reason in switch.needs if switch.is_on(data, profile) else ():
                below = SWITCHES[need]
                if not below.SWITCHES[need].is_on(data, profile):
                    out.append(f"the small switch {switch.name} of the big layer {layer.NAME} needs the small switch {need} of the "
                               f"big layer {below.NAME}, since {reason}")
    return out


def unit_problems(data):
    """The cost unit of a configuration's JSON that does not fit it, for any asset. A cost in money needs a price on the work
    order of every component, its part, the repair charge or the visit, since the oracle policy reads nothing and would pay
    nothing on a start with a fault whose order is free, which leaves the normalized cost and the reward undefined. A cost
    in calls needs a hidden ledger that nothing charges, no downtime rate, no penalty of a failure rule that is on and no
    misdiagnosis price, since a cost in calls leaves that ledger out. The rule reads values and not switches, so it holds
    for an asset whose file prices a part at zero as well."""
    prices = data["prices"]
    if priced.cost_unit(prices) == "money":
        free = [c for c in data["components"] if prices["parts"][c] + prices["repair_charge"] + prices["visit"] == 0]
        if free:
            return [f"the cost unit is money, since a price is above zero, while a work order on {', '.join(free)} costs nothing, so "
                    "the oracle policy, which reads nothing, would pay nothing on a start with that fault: give every work order a "
                    "price with part_prices, repair_charge or visit"]
        return []
    rates = [r for c in data["components"] for r in data["downtime_rate"][c].values()] + [data["downtime_rate"][rules.STOPPED]]
    penalty = data["failure"]["patience"] > 0 and data["failure"]["penalty"] > 0
    if any(r > 0 for r in rates) or penalty:
        return ["the cost unit is calls, since every price is zero, while the downtime rates or the penalty of the failure rule "
                "charge a hidden ledger that a cost in calls leaves out: give the work orders a price with part_prices, "
                "repair_charge or visit, or switch downtime_rate and failure off"]
    if (prices.get("misdiagnosis") or 0) > 0:
        return ["the cost unit is calls, since every price of a read and of a work order is zero, while the misdiagnosis "
                "price charges a hidden ledger that a cost in calls leaves out: give the work orders a price with "
                "part_prices, repair_charge or visit, or switch misdiagnosis off"]
    return []


def compose(spec):
    """The EpisodeConfig of a composition {"layers", "switches", <base fields>}; ValueError lists every problem at once."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET, EpisodeConfig

    found = problems(spec)
    if found:
        raise ValueError("invalid composition: " + "; ".join(found))
    profile = get_profile(spec.get("asset_id", DEFAULT_ASSET))
    data = profile.default_config().to_json()
    named = set(spec.get("layers", []))
    for layer in LAYERS[1:]:
        for switch in layer.SWITCHES.values():
            values = switch.on(profile) if layer.NAME in named else switch.off(profile)
            for path, value in values.items():
                set_path(data, path, value)
    for name, setting in spec.get("switches", {}).items():
        for path, value in SWITCHES[name].SWITCHES[name].values(setting, profile).items():
            set_path(data, path, value)
    for key, value in spec.items():
        if key not in PRESET_KEYS:
            data[key] = value
    found = unmet(data, profile) + unit_problems(data)
    if found:
        raise ValueError("invalid composition: " + "; ".join(found))
    return EpisodeConfig.from_json(data)


def without(spec, name):
    """The composition without a big layer and without the settings of its small switches, the row below it on the ladder
    when the layer is the last of the row."""
    out = dict(spec)
    out["layers"] = [n for n in spec.get("layers", []) if n != name]
    if "switches" in spec:
        out["switches"] = {n: s for n, s in spec["switches"].items() if SWITCHES.get(n) is None or SWITCHES[n].NAME != name}
    return out


def layers_on(config):
    """The big layers above the base with at least one small switch away from its off value, in ladder order."""
    profile = get_profile(config.asset_id)
    data = config.to_json()
    return tuple(layer.NAME for layer in LAYERS[1:] if any(switch.is_on(data, profile) for switch in layer.SWITCHES.values()))


def lists_tools(name, config):
    """Whether the task tools that a big layer adds are listed for a configuration on the AssetOpsBench rows:
    the two time tools of the real-time clock where its switch time_tools is on, the repair manual of the repair rules where
    that layer is on and the clock runs, by the clock mode of layer 2 or a duration of layer 3, since only the clock lets an
    agent of these rows wait out a lead time, and the tools of the base always."""
    if name == real_time.NAME:
        return config.time_tools
    if name == repair_rules.NAME:
        return repair_rules.NAME in layers_on(config) and real_time.has_clock(config.clock_mode, config.durations)
    return True


def row_of(config):
    """The row of the ladder a configuration plays on the AssetOpsBench rows, which sets
    the receipt line and its notifications: without a clock the base row while nothing has a price and no budget stops the
    episode, else the priced row, and with a clock the repair rules row where a repair rule is on, the deterioration row
    where the deterioration layer is on as well, else the advanced real-time clock row where an action takes declared time
    and the real-time clock row otherwise. On every ladder preset the tools of the row are the tools lists_tools lists."""
    if not real_time.has_clock(config.clock_mode, config.durations):
        return base.NAME if priced.free(config) else priced.NAME
    on = layers_on(config)
    if repair_rules.NAME in on:
        return deterioration.NAME if deterioration.NAME in on else repair_rules.NAME
    return advanced_real_time.NAME if advanced_real_time.durations_on(config) else real_time.NAME
