"""The configuration of one hydraulic rig episode (CausalDiagnosis/docs/environment.md, section Configuration).

EpisodeConfig holds every parameter an episode depends on besides its seed: the asset and its components, the sensor
groups, the prices, the durations and the downtime rates of the clock, the repair rules (dependencies, priorities and the
emergency option), the budget, the horizon and the step cap, the fault rule of the starts, the hidden processes drawn at reset
with the degradation hazard, the fix history, the failure rule, the clock mode and the briefing. Every field belongs to the
base or to one of the five big layers of CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers, whose modules hold the rules that the
helpers below apply. The defaults are design parameters, and every run records them.
The fields an asset decides take their defaults from the hyd_1 asset file
(CausalDiagnosis/datasets/hyd_1/asset.json), and the sensor groups draw from the channels of the asset profile of asset_id.
Validation lists every problem at once, since a bad table is a configuration error and not an action the environment
refuses.
"""
import copy
import datetime
from dataclasses import dataclass, field, fields

from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import is_int, is_number
from CausalDiagnosis.datasets.dataset_loader.rule_cards import STOPPED  # the downtime_rate key charged per cycle while the rig is stopped for a repair

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_3_advanced_real_time as advanced_real_time, layer_5_deterioration_and_failure as deterioration, layer_1_priced as priced, layer_2_real_time as real_time, layer_4_repair_rules as repair_rules
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_1_priced import cost_unit  # noqa: F401  (the drivers and the diagnostics server import the rule from here)
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import CLOCK_MODES, has_clock  # noqa: F401  (the diagnostics server's belief_filter.py imports has_clock from here)

PROCESS_MODES = ("none", "seeded")
HAZARD_MODES = ("scheduled", "stochastic")
GUIDANCE_LEVELS = (1, 2, 3)
VERIFIER_SCOPES = ("writes", "all")  # the scopes of the verifier, which verifier/ reads from here
FIX_KINDS = ("replace", "service")  # the standard fix of a component, as the price table names it
ORDER_MODES = ("standard", "emergency")
DEFAULT_ASSET = "hyd_1"  # the asset whose profile gives the defaults of EpisodeConfig


def asset_channels(asset_id):
    """The channels of the asset profile of asset_id, which every sensor group of its configurations draws from, or None
    when no profile is registered or bundled under that id."""
    try:
        return tuple(get_profile(asset_id, engine=False).channels)
    except KeyError:
        return None


# the fields that the default asset decides, as its profile builds them from its tables and its cards, read once at import
_DEFAULTS = get_profile(DEFAULT_ASSET, engine=False).config_fields()


def _default(name):
    """A fresh copy of one default field, so that no two configurations share a table."""
    return copy.deepcopy(_DEFAULTS[name])


def default_sensor_groups():
    return _default("sensor_groups")


def default_prices():
    return _default("prices")


def default_durations():
    return _default("durations")


def default_downtime_rate():
    return _default("downtime_rate")


def default_dependencies():
    """The components that must be healthy for a repair of each component to take, a design parameter of the asset file."""
    return _default("dependencies")


def default_priority():
    return _default("priority")


@dataclass
class EpisodeConfig:
    asset_id: str = DEFAULT_ASSET
    components: list = field(default_factory=lambda: _default("components"))
    levels: dict = field(default_factory=lambda: _default("levels"))
    healthy: dict = field(default_factory=lambda: _default("healthy"))
    sensor_groups: dict = field(default_factory=default_sensor_groups)
    prices: dict = field(default_factory=default_prices)
    durations: dict = field(default_factory=default_durations)
    downtime_rate: dict = field(default_factory=default_downtime_rate)
    dependencies: dict = field(default_factory=default_dependencies)
    priority: dict = field(default_factory=default_priority)
    emergency: dict = field(default_factory=lambda: _default("emergency"))
    budget: float = _DEFAULTS["budget"]
    horizon: int = _DEFAULTS["horizon"]
    max_steps: int = _DEFAULTS["max_steps"]
    fault_rule: dict = field(default_factory=lambda: _default("fault_rule"))
    allowed: list = None
    start: dict = None
    schedule: dict = field(default_factory=lambda: _default("schedule"))
    hazard: dict = field(default_factory=lambda: _default("hazard"))
    fix_history: dict = field(default_factory=lambda: _default("fix_history"))
    failure: dict = field(default_factory=lambda: _default("failure"))
    clock_mode: str = _DEFAULTS["clock_mode"]
    seconds_per_cycle: float = _DEFAULTS["seconds_per_cycle"]
    think_seconds_per_cycle: float = _DEFAULTS["think_seconds_per_cycle"]
    seconds_per_token: float = None
    time_tools: bool = True
    guidance_level: int = 1
    question: str = None
    anchors: dict = None
    verifier_scope: str = "writes"
    diagnostics_server: bool = True
    environment_mode: str = None

    def __post_init__(self):
        self.seconds_per_token = real_time.token_rate(self)
        self.validate()

    # ----- JSON -----
    def to_json(self):
        """A JSON-ready copy: every tuple becomes a list, so that from_json(json.loads(json.dumps(...))) gives an equal config.
        seconds_per_token, None outside the generated_tokens mode, is left out there, so that the JSON of a configuration of
        every other mode keeps its bytes."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.seconds_per_token is None:
            del data["seconds_per_token"]
        if self.environment_mode is None:
            del data["environment_mode"]
        return _listed(data)

    @classmethod
    def from_json(cls, data):
        """The configuration of a JSON object, such as the config of a run.json or a state file, every key at its stored value
        and validated as a configuration built in code is."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown episode config keys {unknown}")
        return cls(**_listed(copy.deepcopy(data)))

    # ----- derived values the environment reads, the rules of the big layers called where they live -----
    def n_faulty(self, combination):
        return sum(1 for c, level in zip(self.components, combination) if level != self.healthy[c])

    def within_fault_rule(self, combination):
        return self.fault_rule["min_faulty"] <= self.n_faulty(combination) <= self.fault_rule["max_faulty"]

    def is_combination(self, combination):
        return (isinstance(combination, (list, tuple)) and len(combination) == len(self.components)
                and all(level in self.levels[c] for c, level in zip(self.components, combination)))

    def read_price(self, sensor_group, cycles):
        return priced.read_price(self, sensor_group, cycles)

    def order_price(self, component, mode="standard"):
        return priced.order_price(self, component, mode)

    def lead_time(self, component, mode="standard"):
        return advanced_real_time.lead_time(self, component, mode)

    def tool_price(self, tool):
        return priced.tool_price(self, tool)

    def tool_duration(self, tool):
        return advanced_real_time.tool_duration(self, tool)

    def emergency_on(self):
        return repair_rules.emergency_on(self)

    def failure_on(self):
        return deterioration.failure_on(self)

    def cost_unit(self):
        return cost_unit(self.prices)

    def has_clock(self):
        return has_clock(self.clock_mode, self.durations)

    def repair_card_on(self):
        return repair_rules.repair_card_on(self)

    def downtime_on(self):
        return advanced_real_time.downtime_on(self)

    def depends_on(self, component):
        return repair_rules.depends_on(self, component)

    def level_rate(self, component, level):
        return advanced_real_time.level_rate(self, component, level)

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
        problems += self._check_repairs()
        problems += self._check_episode_limits()
        problems += self._check_starts()
        processes = self._check_processes()
        problems += processes
        if not processes:
            problems += self._check_kind()
        problems += self._check_modes()
        if not problems:
            from .runtime_mode import problems as runtime_mode_problems
            problems += runtime_mode_problems(self)
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
            if not isinstance(levels, list) or not levels or not all(is_int(v) for v in levels) or len(set(levels)) != len(levels):
                problems.append(f"levels of {c} must be a non-empty list of distinct integers")
                continue
            if not isinstance(self.healthy, dict) or self.healthy.get(c) not in levels:
                problems.append(f"the healthy level of {c} must be one of its levels")
            # a fault is a level off the healthy one, so a component of one level can never be faulty
            if len(levels) < 2:
                problems.append(f"levels of {c} must hold a second level beside the healthy one")
        return problems

    def _check_sensor_groups(self):
        if not isinstance(self.sensor_groups, dict) or not self.sensor_groups:
            return ["sensor_groups must be a non-empty mapping"]
        channels = asset_channels(self.asset_id)
        if channels is None:
            return [f"asset_id {self.asset_id!r} names no registered or bundled asset profile, whose channels the sensor groups draw from"]
        problems = []
        for group, sensors in self.sensor_groups.items():
            if not isinstance(group, str) or not group:
                problems.append(f"sensor group name {group!r} must be a non-empty string")
            if not isinstance(sensors, list) or not sensors or len(set(sensors)) != len(sensors) or any(s not in channels for s in sensors):
                problems.append(f"sensor group {group} must be a non-empty list of distinct channels among {list(channels)}")
        # a read of every channel, a history read without a sensor list among them, is charged to one group that covers them
        # (CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_1_priced.py cheapest_group), so one group must hold every channel
        if not problems and not any(set(sensors) == set(channels) for sensors in self.sensor_groups.values()):
            problems.append(f"sensor_groups must hold one group of every channel of {self.asset_id}, the group that a read of every "
                            "channel is charged to")
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
            # a zero price is the free reading of the base row, which the step cap and the horizon still bound
            negative = [g for g, p in read.items() if not is_number(p) or p < 0]
            if negative:
                problems.append(f"read prices of {negative} are not non-negative finite numbers")
        parts = self.prices.get("parts")
        if not isinstance(parts, dict):
            problems.append("prices.parts must map every component to a price")
        else:
            bad = [c for c in self.components if not is_number(parts.get(c)) or parts[c] < 0]
            if bad:
                problems.append(f"part prices of {bad} are missing or not non-negative finite numbers")
        if not is_number(self.prices.get("repair_charge")) or self.prices["repair_charge"] < 0:
            problems.append("prices.repair_charge must be a non-negative finite number")
        if not is_number(self.prices.get("visit")) or self.prices["visit"] < 0:
            problems.append("prices.visit must be a non-negative finite number")
        if not is_number(self.prices.get("misdiagnosis")) or self.prices["misdiagnosis"] < 0:
            problems.append("prices.misdiagnosis must be a non-negative finite number")
        kinds = self.prices.get("fix_kind")
        if not isinstance(kinds, dict) or any(kinds.get(c) not in FIX_KINDS for c in self.components):
            problems.append(f"prices.fix_kind must name the standard fix of every component, one of {list(FIX_KINDS)}")
        return problems

    def _check_durations(self):
        if not isinstance(self.durations, dict):
            return ["durations must be a mapping"]
        problems = []
        if not is_int(self.durations.get("read_per_cycle")) or self.durations["read_per_cycle"] < 0:
            problems.append("durations.read_per_cycle must be a non-negative integer")
        # wait_for_notification runs the clock one cycle at a time, so a waited cycle lasts exactly one cycle
        if self.durations.get("advance_per_cycle") != 1 or not is_int(self.durations["advance_per_cycle"]):
            problems.append("durations.advance_per_cycle must be 1, since a wait runs the clock one cycle at a time")
        for key in ("lead_time", "repair_downtime"):
            table = self.durations.get(key)
            if not isinstance(table, dict):
                problems.append(f"durations.{key} must map every component to a number of cycles")
                continue
            bad = [c for c in self.components if not is_int(table.get(c)) or table[c] < 0]
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
            bad = [k for k, r in rates.items() if not is_number(r) or r < 0]
            if bad:
                problems.append(f"downtime rates {bad} of {c} are not non-negative finite numbers")
        if not is_number(self.downtime_rate.get(STOPPED)) or self.downtime_rate[STOPPED] < 0:
            problems.append(f"downtime_rate.{STOPPED} must be a non-negative finite number")
        return problems

    def _check_repairs(self):
        problems = []
        priority = self.priority
        if not isinstance(priority, dict) or set(priority) != set(self.components) or not all(is_int(p) and p >= 1 for p in priority.values()):
            problems.append("priority must map every component to a positive integer")
            priority = None
        dependencies = self.dependencies
        if not isinstance(dependencies, dict):
            problems.append("dependencies must be a mapping, empty to switch the repair rule off")
        elif dependencies:
            if set(dependencies) != set(self.components):
                problems.append("dependencies must name every component once, or be empty to switch the repair rule off")
            for c, needed in dependencies.items():
                if (not isinstance(needed, list) or not all(isinstance(d, str) for d in needed) or len(set(needed)) != len(needed)
                        or any(d not in self.components or d == c for d in needed)):
                    problems.append(f"dependencies of {c} must be a list of distinct other components")
                elif priority is not None and c in priority:
                    # a visit repairs in priority order, so a dependency repaired in the same visit closes first
                    late = [d for d in needed if priority[d] >= priority[c]]
                    if late:
                        problems.append(f"dependencies {late} of {c} need a smaller priority number than {c}")
        e = self.emergency
        if not isinstance(e, dict):
            problems.append("emergency must be a mapping with lead, surcharge and hold")
        else:
            if not is_int(e.get("lead")) or e["lead"] < 0:
                problems.append("emergency.lead must be a non-negative integer")
            if not is_number(e.get("surcharge")) or e["surcharge"] < 1:
                problems.append("emergency.surcharge must be a finite number of at least 1")
            if not is_int(e.get("hold")) or e["hold"] < 0:
                problems.append("emergency.hold must be a non-negative integer, 0 switching the emergency mode off")
        return problems

    def _check_episode_limits(self):
        problems = []
        if self.budget is not None and (not is_number(self.budget) or self.budget <= 0):
            problems.append("budget must be null, for an episode without a budget stop, or a positive finite number")
        if not is_int(self.horizon) or self.horizon < 1:
            problems.append("horizon must be a positive integer")
        if not is_int(self.max_steps) or self.max_steps < 1:
            problems.append("max_steps must be a positive integer")
        rule = self.fault_rule
        if not isinstance(rule, dict) or not is_int(rule.get("min_faulty")) or not is_int(rule.get("max_faulty")):
            problems.append("fault_rule must give integer min_faulty and max_faulty")
        elif not 0 <= rule["min_faulty"] <= rule["max_faulty"] <= len(self.components):
            # a rule from zero admits the combination with every component healthy, the start of the control
            problems.append(f"the fault rule needs 0 <= min_faulty <= max_faulty <= {len(self.components)}, got {rule}")
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
                problems.append("start.combination must satisfy the fault rule of the starts")
            if not isinstance(self.start, dict) or not is_int(self.start.get("entry_offset")) or self.start["entry_offset"] < 0:
                problems.append("start.entry_offset must be a non-negative integer")
        return problems

    def _check_processes(self):
        problems = []
        s = self.schedule
        if not isinstance(s, dict) or s.get("mode") not in PROCESS_MODES:
            problems.append(f"schedule.mode must be one of {list(PROCESS_MODES)}")
        else:
            if not is_int(s.get("components")) or not 0 <= s["components"] <= len(self.components):
                problems.append(f"schedule.components must be an integer between 0 and {len(self.components)}")
            if not _is_cycle_range(s.get("first_drop")):
                problems.append("schedule.first_drop must be [first, last] with 0 <= first <= last")
            if not is_int(s.get("interval")) or s["interval"] < 1:
                problems.append("schedule.interval must be a positive integer")
        z = self.hazard
        if not isinstance(z, dict) or z.get("mode") not in HAZARD_MODES:
            problems.append(f"hazard.mode must be one of {list(HAZARD_MODES)}")
        else:
            for key in ("base_rate", "collateral"):
                if not is_number(z.get(key)) or not 0 <= z[key] <= 1:
                    problems.append(f"hazard.{key} must be a number between 0 and 1")
            if not is_number(z.get("depth_factor")) or z["depth_factor"] < 0:
                problems.append("hazard.depth_factor must be a non-negative finite number")
            if z["mode"] == "stochastic" and isinstance(s, dict) and s.get("mode") != "none":
                problems.append("hazard.mode stochastic replaces the scheduled drops, so schedule.mode must be none")
        h = self.fix_history
        if not isinstance(h, dict):
            problems.append("fix_history must be a mapping with healthy_probability, healthy_ago, degraded_probability and degraded_ago")
        else:
            for kind in ("healthy", "degraded"):
                if not is_number(h.get(f"{kind}_probability")) or not 0 <= h[f"{kind}_probability"] <= 1:
                    problems.append(f"fix_history.{kind}_probability must be a number between 0 and 1")
                if not _is_cycle_range(h.get(f"{kind}_ago")):
                    problems.append(f"fix_history.{kind}_ago must be [first, last] with 0 <= first <= last")
        f = self.failure
        if not isinstance(f, dict):
            problems.append("failure must be a mapping with patience, strict and penalty")
        else:
            if not is_int(f.get("patience")) or f["patience"] < 0:
                problems.append("failure.patience must be a non-negative integer, 0 switching the failure rule off")
            if not isinstance(f.get("strict"), bool):
                problems.append("failure.strict must be a boolean")
            if not is_number(f.get("penalty")) or f["penalty"] < 0:
                problems.append("failure.penalty must be a non-negative finite number")
        return problems

    def _check_kind(self):
        """What the evidence kind of the asset allows: a kind that plays no episode has no configuration, and the records of a
        kind that replays no repair show no change of the asset that no work order made, so its deterioration cards are off."""
        try:
            profile = get_profile(self.asset_id, engine=False)
        except KeyError:
            return []  # the check of the sensor groups names an asset id without a profile
        kind = kinds.kind_of(profile.evidence_kind)
        if not kind.plays_episodes:
            return [f"{self.asset_id}: an asset of the {kind.name} kind has no EpisodeConfig, since the engine plays no episode of that kind"]
        if not kind.replays_repairs and deterioration.cards(self):
            return [f"an asset of the {kind.name} kind holds no recording after a change that no work order made, so the deterioration "
                    "cards must be off"]
        return []

    def _check_modes(self):
        problems = []
        if self.clock_mode not in CLOCK_MODES:
            problems.append(f"clock_mode must be one of {list(CLOCK_MODES)}")
        if not is_number(self.seconds_per_cycle) or self.seconds_per_cycle <= 0:
            problems.append("seconds_per_cycle must be a positive finite number")
        if not is_number(self.think_seconds_per_cycle) or self.think_seconds_per_cycle <= 0:
            problems.append("think_seconds_per_cycle must be a positive finite number")
        if self.clock_mode == real_time.TOKEN_MODE and (not is_number(self.seconds_per_token) or self.seconds_per_token <= 0):
            problems.append(f"the clock mode {real_time.TOKEN_MODE} needs seconds_per_token in the clock of the asset file, a positive "
                            "finite number")
        if not isinstance(self.time_tools, bool):
            problems.append("time_tools must be a boolean")
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


def _is_cycle_range(value):
    return isinstance(value, list) and len(value) == 2 and all(is_int(v) for v in value) and 0 <= value[0] <= value[1]


def _listed(value):
    """The value with every tuple turned into a list, recursively, as JSON would spell it."""
    if isinstance(value, dict):
        return {k: _listed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listed(v) for v in value]
    return value
