"""The configuration of one hydraulic rig episode (docs/hydraulic_rig/environment.md, section Configuration).

EpisodeConfig holds every parameter an episode depends on besides its seed: the asset and its components, the sensor
groups, the prices, the durations and the downtime rates of the clock, the budget, the horizon and the step cap, the
public fault rule, the hidden processes drawn at reset, the failure rule, the clock mode and the briefing. The defaults
are the placeholders of the design and every run prints them as design parameters. Validation lists every problem at
once, since a bad table is a configuration error and not an action the environment refuses.
"""
import copy
import datetime
import math
from dataclasses import dataclass, field, fields

from .data import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS

CLOCK_MODES = ("declared", "generation_time", "instant")
PROCESS_MODES = ("none", "seeded")
GUIDANCE_LEVELS = (1, 2, 3)
VERIFIER_SCOPES = ("writes", "all")
STOPPED = "stopped"  # the downtime_rate key charged per cycle while the rig is stopped for a repair


def default_sensor_groups():
    return {f"tier{t}": list(sensors) for t, sensors in TIER_SENSORS.items()}


def default_prices():
    return {"read": {"tier1": 1, "tier2": 3, "tier3": 10},
            "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20},
            "repair_charge": 20}


def default_durations():
    return {"read_per_cycle": 1, "advance_per_cycle": 1,
            "lead_time": {"cooler": 3, "valve": 3, "pump": 3, "accumulator": 3},
            "repair_downtime": {"cooler": 1, "valve": 1, "pump": 2, "accumulator": 1}}


def default_downtime_rate():
    return {"cooler": {"100": 0, "20": 1, "3": 2},
            "valve": {"100": 0, "90": 1, "80": 2, "73": 3},
            "pump": {"0": 0, "1": 1, "2": 2},
            "accumulator": {"130": 0, "115": 1, "100": 2, "90": 3},
            STOPPED: 5}


@dataclass
class EpisodeConfig:
    asset_id: str = "hyd_1"
    components: list = field(default_factory=lambda: list(COMPONENTS))
    levels: dict = field(default_factory=lambda: {c: list(LEVELS[c]) for c in COMPONENTS})
    healthy: dict = field(default_factory=lambda: dict(HEALTHY))
    sensor_groups: dict = field(default_factory=default_sensor_groups)
    prices: dict = field(default_factory=default_prices)
    durations: dict = field(default_factory=default_durations)
    downtime_rate: dict = field(default_factory=default_downtime_rate)
    budget: float = 300
    horizon: int = 60
    max_steps: int = 60
    fault_rule: dict = field(default_factory=lambda: {"min_faulty": 1, "max_faulty": 2})
    allowed: list = None
    start: dict = None
    schedule: dict = field(default_factory=lambda: {"mode": "none", "components": 1, "first_drop": [5, 40], "interval": 10})
    external_events: dict = field(default_factory=lambda: {"mode": "none", "probability": 0.3, "window": [5, 50]})
    failure: dict = field(default_factory=lambda: {"patience": 10, "strict": False, "penalty": 50})
    clock_mode: str = "declared"
    seconds_per_cycle: float = 60
    guidance_level: int = 1
    question: str = None
    anchors: dict = None
    verifier_scope: str = "writes"
    diagnostics_server: bool = True

    def __post_init__(self):
        self.validate()

    # ----- JSON -----
    def to_json(self):
        """A JSON-ready copy: every tuple becomes a list, so that from_json(json.loads(json.dumps(...))) gives an equal config."""
        return _listed({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_json(cls, data):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown episode config keys {unknown}")
        return cls(**_listed(copy.deepcopy(data)))

    # ----- derived values the environment reads -----
    def n_faulty(self, combination):
        return sum(1 for c, level in zip(self.components, combination) if level != self.healthy[c])

    def within_fault_rule(self, combination):
        return self.fault_rule["min_faulty"] <= self.n_faulty(combination) <= self.fault_rule["max_faulty"]

    def is_combination(self, combination):
        return (isinstance(combination, (list, tuple)) and len(combination) == len(self.components)
                and all(level in self.levels[c] for c, level in zip(self.components, combination)))

    def read_price(self, sensor_group, cycles):
        return cycles * self.prices["read"][sensor_group]

    def order_price(self, component):
        return self.prices["parts"][component] + self.prices["repair_charge"]

    def level_rate(self, component, level):
        """The downtime rate per cycle of a component at a level; the table keys are the levels as JSON spells them."""
        return self.downtime_rate[component][str(level)]

    def date_of(self, t):
        """The ISO date of cycle t under the anchor (one recorded cycle per day), or None without an anchor."""
        if self.anchors is None:
            return None
        return (datetime.date.fromisoformat(self.anchors["date0"]) + datetime.timedelta(days=t)).isoformat()

    # ----- validation -----
    def validate(self):
        problems = []
        problems += self._check_asset()
        problems += self._check_sensor_groups()
        problems += self._check_prices()
        problems += self._check_durations()
        problems += self._check_downtime_rate()
        problems += self._check_episode_limits()
        problems += self._check_starts()
        problems += self._check_processes()
        problems += self._check_modes()
        if problems:
            raise ValueError("invalid episode config: " + "; ".join(problems))

    def _check_asset(self):
        problems = []
        if not isinstance(self.asset_id, str) or not self.asset_id:
            problems.append("asset_id must be a non-empty string")
        if not isinstance(self.components, list) or not self.components or len(set(self.components)) != len(self.components):
            return problems + ["components must be a non-empty list of distinct names"]
        for c in self.components:
            levels = self.levels.get(c) if isinstance(self.levels, dict) else None
            if not isinstance(levels, list) or not levels or not all(_is_int(v) for v in levels) or len(set(levels)) != len(levels):
                problems.append(f"levels of {c} must be a non-empty list of distinct integers")
            elif not isinstance(self.healthy, dict) or self.healthy.get(c) not in levels:
                problems.append(f"the healthy level of {c} must be one of its levels")
        return problems

    def _check_sensor_groups(self):
        if not isinstance(self.sensor_groups, dict) or not self.sensor_groups:
            return ["sensor_groups must be a non-empty mapping"]
        problems = []
        for group, sensors in self.sensor_groups.items():
            if not isinstance(group, str) or not group:
                problems.append(f"sensor group name {group!r} must be a non-empty string")
            if not isinstance(sensors, list) or not sensors or len(set(sensors)) != len(sensors) or any(s not in SENSORS for s in sensors):
                problems.append(f"sensor group {group} must be a non-empty list of distinct channels among {list(SENSORS)}")
        return problems

    def _check_prices(self):
        if not isinstance(self.prices, dict):
            return ["prices must be a mapping"]
        problems = []
        read = self.prices.get("read")
        if not isinstance(read, dict):
            problems.append("prices.read must map every sensor group to a price")
        else:
            unknown = [g for g in read if g not in self.sensor_groups]
            if unknown:
                problems.append(f"prices.read names groups {unknown} that have no sensors")
            unpriced = [g for g in self.sensor_groups if g not in read]
            if unpriced:
                problems.append(f"sensor groups {unpriced} have no read price")
            # a free reading would let an agent read the whole recording for nothing, so a read price must be positive
            not_positive = [g for g, p in read.items() if not _is_number(p) or p <= 0]
            if not_positive:
                problems.append(f"read prices of {not_positive} are not positive finite numbers")
        parts = self.prices.get("parts")
        if not isinstance(parts, dict):
            problems.append("prices.parts must map every component to a price")
        else:
            bad = [c for c in self.components if not _is_number(parts.get(c)) or parts[c] < 0]
            if bad:
                problems.append(f"part prices of {bad} are missing or not non-negative finite numbers")
        if not _is_number(self.prices.get("repair_charge")) or self.prices["repair_charge"] < 0:
            problems.append("prices.repair_charge must be a non-negative finite number")
        return problems

    def _check_durations(self):
        if not isinstance(self.durations, dict):
            return ["durations must be a mapping"]
        problems = []
        if not _is_int(self.durations.get("read_per_cycle")) or self.durations["read_per_cycle"] < 0:
            problems.append("durations.read_per_cycle must be a non-negative integer")
        # advancing time has to move the clock, or the horizon could never be reached
        if not _is_int(self.durations.get("advance_per_cycle")) or self.durations["advance_per_cycle"] < 1:
            problems.append("durations.advance_per_cycle must be a positive integer")
        for key in ("lead_time", "repair_downtime"):
            table = self.durations.get(key)
            if not isinstance(table, dict):
                problems.append(f"durations.{key} must map every component to a number of cycles")
                continue
            bad = [c for c in self.components if not _is_int(table.get(c)) or table[c] < 0]
            if bad:
                problems.append(f"durations.{key} of {bad} are missing or not non-negative integers")
        return problems

    def _check_downtime_rate(self):
        if not isinstance(self.downtime_rate, dict):
            return ["downtime_rate must be a mapping"]
        problems = []
        for c in self.components:
            rates = self.downtime_rate.get(c)
            levels = self.levels.get(c) if isinstance(self.levels, dict) else None
            if not isinstance(rates, dict) or not isinstance(levels, list):
                problems.append(f"downtime_rate of {c} must map every level, as a string, to a rate")
                continue
            if set(rates) != {str(level) for level in levels}:
                problems.append(f"downtime_rate of {c} must have exactly the keys {[str(level) for level in levels]}")
            bad = [k for k, r in rates.items() if not _is_number(r) or r < 0]
            if bad:
                problems.append(f"downtime rates {bad} of {c} are not non-negative finite numbers")
        if not _is_number(self.downtime_rate.get(STOPPED)) or self.downtime_rate[STOPPED] < 0:
            problems.append(f"downtime_rate.{STOPPED} must be a non-negative finite number")
        return problems

    def _check_episode_limits(self):
        problems = []
        if not _is_number(self.budget) or self.budget <= 0:
            problems.append("budget must be a positive finite number")
        if not _is_int(self.horizon) or self.horizon < 1:
            problems.append("horizon must be a positive integer")
        if not _is_int(self.max_steps) or self.max_steps < 1:
            problems.append("max_steps must be a positive integer")
        rule = self.fault_rule
        if not isinstance(rule, dict) or not _is_int(rule.get("min_faulty")) or not _is_int(rule.get("max_faulty")):
            problems.append("fault_rule must give integer min_faulty and max_faulty")
        elif not 1 <= rule["min_faulty"] <= rule["max_faulty"] <= len(self.components):
            problems.append(f"the fault rule needs 1 <= min_faulty <= max_faulty <= {len(self.components)}, got {rule}")
        return problems

    def _check_starts(self):
        problems = []
        if self.allowed is not None:
            if not isinstance(self.allowed, list) or not self.allowed or not all(self.is_combination(k) for k in self.allowed):
                problems.append("allowed must be null or a non-empty list of combinations with one valid level per component")
        if self.start is not None:
            if not isinstance(self.start, dict) or not self.is_combination(self.start.get("combination")):
                problems.append("start.combination must give one valid level per component")
            elif not self.within_fault_rule(self.start["combination"]):
                problems.append("start.combination must satisfy the fault rule the agent is told")
            if not isinstance(self.start, dict) or not _is_int(self.start.get("block_offset")) or self.start["block_offset"] < 0:
                problems.append("start.block_offset must be a non-negative integer")
        return problems

    def _check_processes(self):
        problems = []
        s = self.schedule
        if not isinstance(s, dict) or s.get("mode") not in PROCESS_MODES:
            problems.append(f"schedule.mode must be one of {list(PROCESS_MODES)}")
        else:
            if not _is_int(s.get("components")) or not 0 <= s["components"] <= len(self.components):
                problems.append(f"schedule.components must be an integer between 0 and {len(self.components)}")
            if not _is_cycle_range(s.get("first_drop")):
                problems.append("schedule.first_drop must be [first, last] with 0 <= first <= last")
            if not _is_int(s.get("interval")) or s["interval"] < 1:
                problems.append("schedule.interval must be a positive integer")
        e = self.external_events
        if not isinstance(e, dict) or e.get("mode") not in PROCESS_MODES:
            problems.append(f"external_events.mode must be one of {list(PROCESS_MODES)}")
        else:
            if not _is_number(e.get("probability")) or not 0 <= e["probability"] <= 1:
                problems.append("external_events.probability must be a number between 0 and 1")
            if not _is_cycle_range(e.get("window")):
                problems.append("external_events.window must be [first, last] with 0 <= first <= last")
        f = self.failure
        if not isinstance(f, dict):
            problems.append("failure must be a mapping with patience, strict and penalty")
        else:
            if not _is_int(f.get("patience")) or f["patience"] < 1:
                problems.append("failure.patience must be a positive integer")
            if not isinstance(f.get("strict"), bool):
                problems.append("failure.strict must be a boolean")
            if not _is_number(f.get("penalty")) or f["penalty"] < 0:
                problems.append("failure.penalty must be a non-negative finite number")
        return problems

    def _check_modes(self):
        problems = []
        if self.clock_mode not in CLOCK_MODES:
            problems.append(f"clock_mode must be one of {list(CLOCK_MODES)}")
        if not _is_number(self.seconds_per_cycle) or self.seconds_per_cycle <= 0:
            problems.append("seconds_per_cycle must be a positive finite number")
        if self.guidance_level not in GUIDANCE_LEVELS:
            problems.append(f"guidance_level must be one of {list(GUIDANCE_LEVELS)}")
        if self.question is not None and not isinstance(self.question, str):
            problems.append("question must be null or a string")
        if self.anchors is not None:
            try:
                datetime.date.fromisoformat(self.anchors["date0"])
            except (TypeError, KeyError, ValueError):
                problems.append("anchors must be null or {\"date0\": an ISO date}")
        if self.verifier_scope not in VERIFIER_SCOPES:
            problems.append(f"verifier_scope must be one of {list(VERIFIER_SCOPES)}")
        if not isinstance(self.diagnostics_server, bool):
            problems.append("diagnostics_server must be a boolean")
        return problems


def _is_int(value):
    # bool is a subclass of int, so True and False are excluded explicitly
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_cycle_range(value):
    return isinstance(value, list) and len(value) == 2 and all(_is_int(v) for v in value) and 0 <= value[0] <= value[1]


def _listed(value):
    """The value with every tuple turned into a list, recursively, as JSON would spell it."""
    if isinstance(value, dict):
        return {k: _listed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listed(v) for v in value]
    return value
