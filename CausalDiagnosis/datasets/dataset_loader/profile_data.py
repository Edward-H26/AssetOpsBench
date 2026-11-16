import copy
import itertools
import json
from dataclasses import dataclass, fields
from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, rule_cards as rules
from CausalDiagnosis.datasets.dataset_loader.recordings import REPO


FORMAT = 1
FAULT_SEVERITIES = ("critical", "degraded")  # the severities of a failure mode that describes a fault
# the kinds whose parameters the engine configuration reads; a kind that the profile states no card of is off
CONFIG_KINDS = tuple(name for name, kind in rules.BUILTIN_KINDS.items() if kind.binds)
# the kinds whose parameters the profile publishes, with the failure card, and the configuration fields their cards bind,
# which the processes block of the public profile and of released.json holds and the diagnostics server reads; the visit
# charge and the downtime rates reach the profile in its prices and downtime_rate
PROCESS_KINDS = ("schedule", "hazard", "emergency", "dependencies", "fix_history", "failure")
PROCESS_FIELDS = tuple(bind.field for name in PROCESS_KINDS for bind in rules.kind_of(name).binds)


class ProfileError(ValueError):
    """A profile that fails validation, with every problem in .problems."""

    def __init__(self, source, problems):
        self.problems = list(problems)
        super().__init__(f"{source}: invalid asset profile: " + "; ".join(self.problems))


@dataclass(frozen=True)
class AssetProfile:
    """The public, static tables of one asset class and its rule cards. Build it with load_profile or from_json, which
    validate first, and treat its tables as read only."""

    asset_id: str
    evidence_kind: str
    components: tuple
    levels: dict
    healthy: dict
    degradation: dict
    observation_kinds: dict
    channels: dict
    statistics: dict
    defaults: dict
    rules: tuple
    sampler: dict
    failure_modes: tuple
    knowledge: dict
    records: dict
    text: dict
    scoring: str

    @classmethod
    def from_json(cls, data, source="profile", engine=True):
        problems = profile_validation.validate(data, engine)
        if problems:
            raise ProfileError(source, problems)
        return cls._build(data)

    @classmethod
    def _build(cls, data):
        values = {f.name: copy.deepcopy(data[f.name]) for f in fields(cls)}
        values["components"] = tuple(values["components"])
        values["levels"] = {c: tuple(v) for c, v in values["levels"].items()}
        values["degradation"] = {c: tuple(v) for c, v in values["degradation"].items()}
        values["rules"] = tuple(values["rules"])
        values["failure_modes"] = tuple(values["failure_modes"])
        return cls(**values)

    def to_json(self):
        """The asset file this profile reads back from, every tuple a list."""
        return {"format": FORMAT, **{f.name: _plain(getattr(self, f.name)) for f in fields(self)}}

    # ----- views -----
    def card(self, kind):
        """The card of the kind, or None when the profile does not use it."""
        return next((c for c in self.rules if c["kind"] == kind), None)

    def healthy_state(self):
        """The healthy combination in component order, the reference of the normal ranges, of recovery and of anomaly scores."""
        return tuple(self.healthy[c] for c in self.components)

    def combinations(self):
        """Every combination of levels in itertools.product order over the components, the grid classifiers.COMBOS holds."""
        return tuple(itertools.product(*(self.levels[c] for c in self.components)))

    def field_name(self, channel):
        """The public field name of a channel, as the served values and the evidence identifiers spell it."""
        return self.channels[channel]["field"]

    def plain_statistic(self):
        """The served statistic listed ahead of the others, which carries the normal range and the alarm and which a telemetry
        record of the AssetOpsBench rows stores under the bare field of a channel."""
        return self.statistics["served"][0]

    def channel_of(self, name):
        """The channel of a channel name, a public field or a summary field, or None. A field starts with the prefix of its
        channel's field, which need not be the channel's name (P1 is EPS1 on hyd_1)."""
        if name in self.channels:
            return name
        prefix = name.split("_", 1)[0]
        return next((c for c in self.channels if self.field_name(c).split("_", 1)[0] == prefix), None)

    def unit_and_rate(self, channel):
        """(unit, sampling rate) of a channel from its field, ('l_per_min', '10Hz') for FS1_VolumeFlow_l_per_min_10Hz."""
        parts = self.field_name(channel).split("_")
        return "_".join(parts[2:-1]), parts[-1]

    def fault_modes(self):
        """The failure modes that describe a fault, critical or degraded, in table order."""
        return tuple(m for m in self.failure_modes if m["severity"] in FAULT_SEVERITIES)

    def level_code(self, component, level):
        """The failure mode code of a component at a level, or None."""
        return next((m["code"] for m in self.failure_modes
                     if m["level"] is not None and (m["component"], m["level"]) == (component, level)), None)

    def registry_entry(self):
        """{asset_id, asset_type, description, site}: the asset's entry in the registry file the profile names, a sample of
        AssetOpsBench's asset profiles, or the profile's fallback when the file or the entry is absent. The entry is read once
        per profile and kept on it outside its fields, so equality, to_json and the digests of a profile do not see it, and it
        is read again only when the registry file or the fallback of the profile differ from those it was read under. Every
        call returns a copy."""
        source = (self.text["registry_file"], self.text["fallback"])
        kept = self.__dict__.get("_registry_entry")
        if kept is None or kept[0] != source:
            kept = (copy.deepcopy(source), self._read_registry_entry())
            object.__setattr__(self, "_registry_entry", kept)  # the dataclass is frozen, and the entry is no field of it
        return dict(kept[1])

    def _read_registry_entry(self):
        if self.text["registry_file"] is not None:
            try:
                for entry in json.loads((REPO / self.text["registry_file"]).read_text(encoding="utf-8")):
                    if entry.get("assetnum") == self.asset_id:
                        return {"asset_id": self.asset_id, "asset_type": entry.get("assettype"),
                                "description": entry.get("description"), "site": entry.get("siteid")}
            except (OSError, ValueError):
                pass
        return {"asset_id": self.asset_id, **self.text["fallback"]}

    def config_fields(self):
        """The fields of EpisodeConfig that the asset decides, from its tables and its cards, with the off parameters of
        every kind that it states no card of. The other fields belong to the preset and keep the engine's defaults."""
        kind = kinds.kind_of(self.evidence_kind)
        if kind is None or not kind.plays_episodes:
            raise ValueError(f"{self.asset_id}: an asset of the {self.evidence_kind} kind has no EpisodeConfig, since the engine plays no "
                             "episode of that kind")
        d = self.defaults
        out = {
            "asset_id": self.asset_id,
            "components": list(self.components),
            "levels": {c: list(self.levels[c]) for c in self.components},
            "healthy": {c: self.healthy[c] for c in self.components},
            "sensor_groups": {g: list(spec["channels"]) for g, spec in self.observation_kinds.items()},
            "prices": copy.deepcopy(d["prices"]),
            "durations": copy.deepcopy(d["durations"]),
            "budget": d["limits"]["budget"],
            "horizon": d["limits"]["horizon"],
            "max_steps": d["limits"]["max_steps"],
            "fault_rule": copy.deepcopy(self.sampler["fault_rule"]),
            "clock_mode": d["clock"]["clock_mode"],
            "seconds_per_cycle": d["clock"]["seconds_per_cycle"],
            "think_seconds_per_cycle": d["clock"]["think_seconds_per_cycle"],
            "seconds_per_token": d["clock"].get("seconds_per_token"),
        }
        for name in CONFIG_KINDS:
            rules.apply_card(out, self.card(name) or {"kind": name, "parameters": rules.kind_of(name).off(self)})
        for card in self.rules:
            if card["kind"] not in CONFIG_KINDS:
                rules.apply_card(out, card)
        return out

    def default_config(self):
        """The episode configuration of the asset's defaults, field by field equal to EpisodeConfig() for hyd_1."""
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig  # imported here, so that the configuration module can import this one

        with profile_registry._served(self):
            return EpisodeConfig(**self.config_fields())


def processes_block(config):
    """{configuration field: value} of the published rules, the processes block of the public profile and of
    released.json. It holds the fields that the published kinds bind, with the values of the configuration, so a kind that the
    profile states no card of shows the off parameters its configuration runs with. config is an
    EpisodeConfig or its JSON. The fault rule of the starts lives in the sampler and never enters the block."""
    data = config if isinstance(config, dict) else config.to_json()
    return {field: copy.deepcopy(data[field]) for field in PROCESS_FIELDS}


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return copy.deepcopy(value)


# Imported after the definitions: this module uses them only when a function runs, and they may load
# this module first (the order of the package is in __init__.py).
from . import profile_registry, profile_validation
