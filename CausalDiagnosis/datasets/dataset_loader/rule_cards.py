"""The rule library of the asset layer (CausalDiagnosis/docs/adding-a-dataset.md, section Rules of the asset file).

A rule card is {"kind": <name>, "parameters": {...}}. An asset profile picks the cards it needs and fills their
parameters. A rule that only one asset needs is a card of a declared kind, and a kind that no built in kind covers is a
plugin registered by name, written by a person. The kinds and the keys of their parameters are declared in this module
and nowhere else. Each kind also states where its parameters land in the episode configuration, so that the
configuration an asset profile builds equals, field by field and byte by byte, the configuration the engine reads.
"""
import copy
from dataclasses import dataclass

from CausalDiagnosis.datasets.dataset_loader import evidence_kinds  # the module function kinds() below lists the rule kinds

RECORDED = ("recorded_levels",)


@dataclass(frozen=True)
class Bind:
    """Where a parameter of a card lands in the episode configuration. parameter None stands for the whole parameter
    mapping in the key order of its kind. key names one entry of a configuration table, after the entry it follows in that
    table, and merge adds the entries of a mapping to the table instead of replacing it."""

    field: str
    parameter: str = None
    key: str = None
    after: str = None
    merge: bool = False


@dataclass(frozen=True)
class RuleKind:
    """A kind of rule card: its name, the exact keys of its parameters, the evidence kinds it applies to, where its
    parameters land in the configuration, an optional check(parameters, profile) for what the configuration does not
    check itself, and for a kind with binds off(profile), the parameters that switch the rule off, which the
    configuration takes when the profile or the preset selects no card of the kind."""

    name: str
    keys: tuple
    applies_to: tuple = RECORDED
    binds: tuple = ()
    check: object = None
    off: object = None


STOPPED = "stopped"  # the key of the stopped rate inside EpisodeConfig.downtime_rate, config.STOPPED


# The off parameters of each kind use the switches that the engine reads, a mode of none, a zero probability, rate,
# charge, hold or patience, and an empty dependency mapping.
def _history_off(profile):
    return {"healthy_probability": 0.0, "healthy_ago": [0, 0], "degraded_probability": 0.0, "degraded_ago": [0, 0]}


def _emergency_off(profile):
    return {"lead": 0, "surcharge": 1.0, "hold": 0}  # a hold of 0 switches the emergency mode off, EpisodeConfig.emergency_on


def _schedule_off(profile):
    return {"mode": "none", "components": 0, "first_drop": [0, 0], "interval": 1}


def _hazard_off(profile):
    return {"mode": "scheduled", "base_rate": 0.0, "depth_factor": 0.0, "collateral": 0.0}


def _dependencies_off(profile):
    return {"dependencies": {}, "priority": {c: 1 for c in profile.components}}


def _visits_off(profile):
    return {"charge": 0}


def _downtime_off(profile):
    return {"rates": {c: {str(level): 0 for level in profile.levels[c]} for c in profile.components}, "stopped": 0}


def _failure_off(profile):
    return {"patience": 0, "strict": False, "penalty": 0}  # a patience of 0 switches the rule off, EpisodeConfig.failure_on


BUILTIN_KINDS = {kind.name: kind for kind in (
    RuleKind("fix_history", ("healthy_probability", "healthy_ago", "degraded_probability", "degraded_ago"), binds=(Bind("fix_history"),),
             off=_history_off),
    RuleKind("emergency", ("lead", "surcharge", "hold"), binds=(Bind("emergency"),), off=_emergency_off),
    RuleKind("schedule", ("mode", "components", "first_drop", "interval"), binds=(Bind("schedule"),), off=_schedule_off),
    RuleKind("hazard", ("mode", "base_rate", "depth_factor", "collateral"), binds=(Bind("hazard"),), off=_hazard_off),
    RuleKind("dependencies", ("dependencies", "priority"), binds=(Bind("dependencies", "dependencies"), Bind("priority", "priority")),
             off=_dependencies_off),
    RuleKind("visits", ("charge",), binds=(Bind("prices", "charge", key="visit", after="repair_charge"),), off=_visits_off),
    RuleKind("downtime", ("rates", "stopped"), binds=(Bind("downtime_rate", "rates", merge=True), Bind("downtime_rate", "stopped", key=STOPPED)),
             off=_downtime_off),
    RuleKind("failure", ("patience", "strict", "penalty"), binds=(Bind("failure"),), off=_failure_off),
)}
_PLUGINS = {}


def register_kind(kind):
    """Register a plugin kind by its name. A plugin is any object with name, keys and applies_to, and optionally binds,
    check and off. The name of a built in kind or of an earlier plugin is refused."""
    for attribute in ("name", "keys", "applies_to"):
        if not hasattr(kind, attribute):
            raise TypeError(f"a rule kind needs {attribute}")
    if kind.name in BUILTIN_KINDS or kind.name in _PLUGINS:
        raise ValueError(f"the rule kind {kind.name!r} is already declared")
    _PLUGINS[kind.name] = kind
    return kind


def unregister_kind(name):
    """Remove a plugin kind, as the tests that register one do. Built in kinds cannot be removed."""
    if name in BUILTIN_KINDS:
        raise ValueError(f"the built in rule kind {name!r} cannot be removed")
    _PLUGINS.pop(name, None)


def kind_of(name):
    """The kind of the name, built in or plugin, or None."""
    return BUILTIN_KINDS.get(name) or _PLUGINS.get(name)


def kinds():
    """The names of every kind, the built in ones first."""
    return tuple(BUILTIN_KINDS) + tuple(_PLUGINS)


def applies(kind, evidence_kind):
    """Whether a rule kind applies to an evidence kind: one that its applies_to names, and for a rule kind of the recorded
    levels every evidence kind that plays episodes, since the parameters of its cards land in the episode configuration."""
    if evidence_kind in kind.applies_to:
        return True
    played = evidence_kinds.kind_of(evidence_kind)
    return all(name in kind.applies_to for name in RECORDED) and played is not None and played.plays_episodes


def card_problems(card, profile=None, where="card"):
    """Every problem of one card at once: its shape, its kind, the keys of its parameters, whether its kind applies to the
    profile's evidence kind, and the check of its kind. profile is the raw profile mapping, read for the evidence kind and
    the observation kinds."""
    if not isinstance(card, dict) or set(card) != {"kind", "parameters"}:
        return [f"{where} must be an object with exactly the keys kind and parameters"]
    kind = kind_of(card["kind"])
    if kind is None:
        return [f"{where} has the unknown kind {card['kind']!r}; the kinds are {list(kinds())}, and a new kind is a plugin registered by name"]
    parameters = card["parameters"]
    if not isinstance(parameters, dict):
        return [f"{where} ({kind.name}): parameters must be an object"]
    problems = []
    missing = [k for k in kind.keys if k not in parameters]
    unknown = sorted(k for k in parameters if k not in kind.keys)
    if missing:
        problems.append(f"{where} ({kind.name}): missing parameters {missing}")
    if unknown:
        problems.append(f"{where} ({kind.name}): unknown parameters {unknown}, the keys of {kind.name} are {list(kind.keys)}")
    evidence_kind = (profile or {}).get("evidence_kind")
    if evidence_kind is not None and not applies(kind, evidence_kind):
        problems.append(f"{where} ({kind.name}): the kind applies to {list(kind.applies_to)}, not to {evidence_kind}")
    if not problems and getattr(kind, "check", None) is not None:
        problems += [f"{where} ({kind.name}): {p}" for p in kind.check(parameters, profile)]
    return problems


def ordered(parameters, keys):
    """The parameters in the key order of their kind, so that the configuration keeps its byte order whatever order a
    drafted card lists them in."""
    return {k: copy.deepcopy(parameters[k]) for k in keys}


def _insert(table, key, value, after):
    """The table with key set to value, placed right after the entry after when the key is new and after is present."""
    if key in table or after not in table:
        return {**table, key: value}
    out = {}
    for k, v in table.items():
        out[k] = v
        if k == after:
            out[key] = value
    return out


def apply_card(fields, card):
    """Write the parameters of one card into the configuration fields, in place, by the binds of its kind."""
    kind = kind_of(card["kind"])
    parameters = card["parameters"]
    for bind in getattr(kind, "binds", ()):
        value = ordered(parameters, kind.keys) if bind.parameter is None else copy.deepcopy(parameters[bind.parameter])
        if bind.merge:
            fields[bind.field] = {**fields.get(bind.field, {}), **value}
        elif bind.key is not None:
            fields[bind.field] = _insert(fields.get(bind.field, {}), bind.key, value, bind.after)
        else:
            fields[bind.field] = value
    return fields


def read_card(config, kind):
    """The parameters of a card of the kind as an episode configuration holds them, the inverse of apply_card, so that the
    public view of an episode shows the values its preset set. config is the JSON of an EpisodeConfig. None for a kind
    whose parameters have no place in the configuration, whose card then shows its own parameters."""
    if not kind.binds:
        return None
    keyed = {(b.field, b.key) for b in kind.binds if b.key is not None}
    parameters = {}
    for bind in kind.binds:
        if bind.parameter is None:
            parameters.update(ordered(config[bind.field], kind.keys))
        elif bind.merge:
            parameters[bind.parameter] = {k: copy.deepcopy(v) for k, v in config[bind.field].items() if (bind.field, k) not in keyed}
        elif bind.key is not None:
            parameters[bind.parameter] = copy.deepcopy(config[bind.field][bind.key])
        else:
            parameters[bind.parameter] = copy.deepcopy(config[bind.field])
    return ordered(parameters, kind.keys)
