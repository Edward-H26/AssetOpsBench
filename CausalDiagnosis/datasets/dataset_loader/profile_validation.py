import keyword
import math
import re
from dataclasses import fields
from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, dataset_packages as ports, rule_cards as rules
from . import profile_data, profile_registry


SCORING_MODES = ("program", "judge")
SEVERITIES = ("critical", "degraded", "healthy", "info")
ASSET_ID = re.compile(r"[A-Za-z0-9_]+")
RECORD_NAME = re.compile(r"[A-Za-z0-9]+")  # a channel or a statistic inside the feature names of a data file
CONFIG_PREFIX = "invalid episode config: "
DEFAULT_TABLES = {"prices": ("read", "parts", "repair_charge", "fix_kind", "misdiagnosis"),
                  "durations": ("read_per_cycle", "advance_per_cycle", "lead_time", "repair_downtime"),
                  "limits": ("budget", "horizon", "max_steps"), "clock": ("clock_mode", "seconds_per_cycle", "think_seconds_per_cycle")}
# the keys a table may hold besides its own: the rate of the generated_tokens clock, which only an asset that plays that mode needs
OPTIONAL_DEFAULTS = {"clock": ("seconds_per_token",)}
DATA_KEYS = ("data_env", "release_dir", "release_file")  # the keys of records that name the data file, after the keys of the evidence kind
SOURCE_KEYS = ("page", "fetch")  # the official page of the data and the one command that fetches and builds it, stated together
TEXT_KEYS = ("noun", "short_noun", "episode_noun", "unit_noun", "whole_asset_patterns", "registry_file", "fallback")
# the sentences that an asset may state in place of the ones the briefings build from its nouns
OPTIONAL_TEXT_KEYS = ("question", "opening", "scoring", "level_meaning", "readings")
FAILURE_MODE_KEYS = ("component", "level", "code", "severity", "action")


# ----- validation -----
def is_int(value):
    """An int that is no bool, the one test of an integer of the asset layer and the engine, since bool is a subclass of int."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value):
    """A finite int or float that is no bool, the one test of a number of the asset layer and the engine."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _names(value):
    return isinstance(value, list) and all(isinstance(v, str) and v for v in value) and len(set(value)) == len(value)


def _check_tables(data, kind):
    """The components, their levels, healthy levels and degradation orders. components comes back None when it is
    missing or unusable, so that the checks that read it are skipped instead of repeating its one problem."""
    if "components" not in data:
        return [], None
    components = data["components"]
    if not _names(components):
        return ["components must be a list of distinct non-empty names"], None
    if kind.plays_episodes and not components:
        return [f"a {kind.name} profile needs at least one component"], None
    problems, good_levels, good_healthy = [], {}, {}
    # the diagnosis tool takes one argument per component beside evidence_ids
    unusable = [c for c in components if not c.isidentifier() or keyword.iskeyword(c) or c == "evidence_ids"]
    if unusable:
        problems.append(f"components {unusable} must be identifiers other than keywords and evidence_ids")
    if "levels" in data:
        levels = data["levels"]
        if not isinstance(levels, dict) or set(levels) != set(components):
            problems.append(f"levels must name exactly the components {components}")
        else:
            for c in components:
                v = levels[c]
                if not isinstance(v, list) or not v or not all(is_int(x) for x in v) or len(set(v)) != len(v):
                    problems.append(f"levels of {c} must be a non-empty list of distinct integers")
                else:
                    good_levels[c] = v
                    # a fault is a level off the healthy one, so a component of one level can never be faulty
                    if kind.plays_episodes and len(v) < 2:
                        problems.append(f"levels of {c} must hold a second level beside the healthy one")
    if "healthy" in data:
        healthy = data["healthy"]
        if not isinstance(healthy, dict) or set(healthy) != set(components):
            problems.append(f"healthy must name exactly the components {components}")
        else:
            for c in good_levels:
                if healthy[c] in good_levels[c]:
                    good_healthy[c] = healthy[c]
                else:
                    problems.append(f"the healthy level of {c} must be one of its levels")
    if "degradation" in data:
        degradation = data["degradation"]
        if not isinstance(degradation, dict) or set(degradation) != set(components):
            problems.append(f"degradation must name exactly the components {components}")
        else:
            for c in good_levels:
                order = degradation[c]
                if not isinstance(order, list) or not all(is_int(x) for x in order) or len(order) != len(good_levels[c]) \
                        or set(order) != set(good_levels[c]):
                    problems.append(f"degradation of {c} must list every level of {c} once")
                elif c in good_healthy and order[0] != good_healthy[c]:
                    problems.append(f"degradation of {c} must start at the healthy level")
    return problems, components


def _check_observations(data, kind):
    """The channels, the observation kinds and the statistics. channels comes back as the set of declared channels, or
    None when they are missing or unusable."""
    problems, known, recorded = [], None, kind.plays_episodes
    if "channels" in data:
        channels = data["channels"]
        if not isinstance(channels, dict) or (recorded and not channels):
            problems.append("channels must map every channel to its public field and its quantity" if recorded else "channels must be an object")
        else:
            for name, spec in channels.items():
                if not isinstance(spec, dict) or set(spec) != {"field", "quantity"} or not all(isinstance(spec[k], str) and spec[k] for k in spec):
                    problems.append(f"channel {name} must give exactly a non-empty field and quantity")
            public = [s.get("field") for s in channels.values() if isinstance(s, dict)]
            if len(set(map(str, public))) != len(public):
                problems.append("the public fields of the channels must be distinct")
            # feature names read <Quantity>_<Channel>_<statistic> from the right, so a channel holds no underscore
            bad = [name for name in channels if not RECORD_NAME.fullmatch(name)]
            if recorded and bad:
                problems.append(f"channels {bad} must be letters and digits, as the feature names of the data file spell them")
            known = set(channels)
    if "observation_kinds" in data:
        kinds = data["observation_kinds"]
        if not isinstance(kinds, dict) or not kinds:
            problems.append("observation_kinds must be a non-empty object")
        elif recorded:
            for name, spec in kinds.items():
                group = spec.get("channels") if isinstance(spec, dict) and set(spec) == {"channels"} else None
                if not _names(group) or not group:
                    problems.append(f"observation kind {name} must give exactly a non-empty list of distinct channels")
                elif known is not None and set(group) - known:
                    problems.append(f"observation kind {name} names channels {sorted(set(group) - known)} that channels does not declare")
    if "statistics" in data:
        statistics = data["statistics"]
        if not recorded:
            if not isinstance(statistics, dict):
                problems.append("statistics must be an object")
        elif not isinstance(statistics, dict) or set(statistics) != {"recorded", "served"} or not _names(statistics.get("recorded")) \
                or not all(RECORD_NAME.fullmatch(s) for s in statistics["recorded"]) or not _names(statistics.get("served")) \
                or not statistics["served"] or set(statistics["served"]) - set(statistics["recorded"]):
            problems.append("statistics must give the recorded statistics, letters and digits, and the served ones among them")
    return problems, known


def _check_defaults(data, kind):
    if "defaults" not in data:
        return []
    defaults = data["defaults"]
    if not isinstance(defaults, dict):
        return ["defaults must be an object"]
    if not kind.plays_episodes:
        return []
    problems = []
    if set(defaults) != set(DEFAULT_TABLES):
        problems.append(f"defaults must hold exactly the tables {list(DEFAULT_TABLES)}")
    for table, keys in DEFAULT_TABLES.items():
        value = defaults.get(table)
        optional = OPTIONAL_DEFAULTS.get(table, ())
        if isinstance(value, dict) and not set(keys) <= set(value) <= set(keys) | set(optional):
            problems.append(f"defaults.{table} must hold exactly {list(keys)}" + (f", and may hold {list(optional)}" if optional else ""))
        elif table in defaults and not isinstance(value, dict):
            problems.append(f"defaults.{table} must be an object")
    return problems


def _check_rules(data):
    if "rules" not in data:
        return []
    cards = data["rules"]
    if not isinstance(cards, list):
        return ["rules must be a list of cards"]
    problems = []
    for i, card in enumerate(cards):
        problems += rules.card_problems(card, data, where=f"rules[{i}]")
    names = [c.get("kind") for c in cards if isinstance(c, dict)]
    repeated = sorted({n for n in names if isinstance(n, str) and names.count(n) > 1})
    if repeated:
        problems.append(f"rules name the kinds {repeated} more than once")
    return problems


def _check_sampler(data, kind):
    if "sampler" not in data:
        return []
    sampler = data["sampler"]
    if not isinstance(sampler, dict):
        return ["sampler must be an object"]
    if not kind.plays_episodes:
        return []
    rule = sampler.get("fault_rule")
    if set(sampler) != {"fault_rule"} or not isinstance(rule, dict) or set(rule) != {"min_faulty", "max_faulty"} \
            or not all(is_int(v) for v in rule.values()):
        return ["sampler must hold exactly fault_rule with integer min_faulty and max_faulty"]
    return []


def _check_failure_modes(data, components):
    if "failure_modes" not in data:
        return []
    modes = data["failure_modes"]
    if not isinstance(modes, list):
        return ["failure_modes must be a list"]
    levels = data.get("levels") if isinstance(data.get("levels"), dict) else {}
    problems, codes = [], []
    for i, mode in enumerate(modes):
        if not isinstance(mode, dict) or set(mode) != set(FAILURE_MODE_KEYS):
            problems.append(f"failure_modes[{i}] must give exactly {list(FAILURE_MODE_KEYS)}")
            continue
        if not isinstance(mode["code"], str) or not mode["code"]:
            problems.append(f"failure_modes[{i}] must give a non-empty code")
        else:
            codes.append(mode["code"])
        if mode["severity"] not in SEVERITIES:
            problems.append(f"failure_modes[{i}] has the severity {mode['severity']!r}, not one of {list(SEVERITIES)}")
        if components is None:
            continue
        if mode["component"] in components:
            if isinstance(levels.get(mode["component"]), list) and mode["level"] not in levels[mode["component"]]:
                problems.append(f"failure_modes[{i}] gives the level {mode['level']!r}, not a level of {mode['component']}")
        elif mode["level"] is not None:
            problems.append(f"failure_modes[{i}] names {mode['component']!r}, which is no component, so its level must be null")
    if len(set(codes)) != len(codes):
        problems.append("the failure mode codes must be distinct")
    return problems


def _check_knowledge(data, kind, components, channels):
    if "knowledge" not in data:
        return []
    knowledge = data["knowledge"]
    if not isinstance(knowledge, dict):
        return ["knowledge must be an object"]
    if not kind.plays_episodes:
        return []
    problems = []
    if set(knowledge) != {"fmea_source", "association_source", "indicates", "correlations"}:
        problems.append("knowledge must hold exactly fmea_source, association_source, indicates and correlations")
    indicates = knowledge.get("indicates")
    if not isinstance(indicates, dict) or not all(_names(v) for v in indicates.values()) \
            or (channels is not None and set(indicates) - channels) \
            or (components is not None and not all(set(v) <= set(components) for v in indicates.values())):
        problems.append("knowledge.indicates must map declared channels to lists of components")
    correlations = knowledge.get("correlations")
    if not isinstance(correlations, dict) or not all(isinstance(v, dict) and all(is_number(x) and -1 <= x <= 1 for x in v.values())
                                                      for v in correlations.values()) \
            or (components is not None and set(correlations) - set(components)) \
            or (channels is not None and not all(set(v) <= channels for v in correlations.values())):
        problems.append("knowledge.correlations must map components to channel correlations between -1 and 1")
    return problems


def _check_text(data, kind, channels):
    if "text" not in data:
        return []
    text = data["text"]
    if not isinstance(text, dict):
        return ["text must be an object"]
    recorded = kind.plays_episodes
    keys = TEXT_KEYS + (("example_channel",) if recorded else ())
    problems = []
    if set(text) - set(OPTIONAL_TEXT_KEYS) != set(keys):
        problems.append(f"text must hold exactly {list(keys)}")
    for key in ("noun", "short_noun", "episode_noun", "unit_noun", *OPTIONAL_TEXT_KEYS):
        if key in text and (not isinstance(text[key], str) or not text[key]):
            problems.append(f"text.{key} must be a non-empty string")
    if recorded:
        if kind.loop is not None and "readings" in text:
            problems.append(f"text.readings is stated by the evidence kind {kind.name}, whose readings loop")
        elif kind.loop is None and "readings" not in text:
            problems.append(f"text.readings must say what a reading of the asset is, since the evidence kind {kind.name} declares no loop")
    patterns = text.get("whole_asset_patterns")
    if not _names(patterns):
        problems.append("text.whole_asset_patterns must be a list of distinct regular expressions")
    elif recorded and not patterns:
        # the summary check compiles the patterns into one, and an empty list would match every sentence
        problems.append("text.whole_asset_patterns must hold at least one pattern, the words of the summary check for the whole asset")
    else:
        for p in patterns:
            try:
                re.compile(p)
            except re.error as e:
                problems.append(f"text.whole_asset_patterns holds {p!r}, which does not compile ({e})")
    if text.get("registry_file") is not None and not isinstance(text.get("registry_file"), str):
        problems.append("text.registry_file must be a path relative to the fork root or null")
    fallback = text.get("fallback")
    if not isinstance(fallback, dict) or set(fallback) != {"asset_type", "description", "site"}:
        problems.append("text.fallback must give exactly asset_type, description and site")
    if recorded and channels is not None and text.get("example_channel") not in channels:
        problems.append("text.example_channel must be a declared channel")
    return problems


def _check_records(data, kind):
    if "records" not in data:
        return []
    records = data["records"]
    if not isinstance(records, dict):
        return ["records must be an object"]
    if not kind.plays_episodes:
        return []
    problems = []
    keys = kind.record_keys + DATA_KEYS
    extra = set(records) - set(keys)
    if set(keys) - set(records) or extra - set(SOURCE_KEYS):
        problems.append(f"records must hold {list(keys)} and may add {list(SOURCE_KEYS)}")
    if extra & set(SOURCE_KEYS) and not all(isinstance(records.get(key), str) and records.get(key) for key in SOURCE_KEYS):
        problems.append("records.page and records.fetch are stated together, each as a text that is not empty")
    return problems + (kind.part("check")(data) if kind.check is not None else [])


def _asset_type(data):
    """The asset type of the fallback entry of a raw profile, or None when the text table holds none."""
    text = data.get("text")
    fallback = text.get("fallback") if isinstance(text, dict) else None
    asset_type = fallback.get("asset_type") if isinstance(fallback, dict) else None
    return asset_type if isinstance(asset_type, str) else None


def config_problems(profile):
    """The problems the engine configuration finds in the configuration a recorded levels profile builds."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig

    try:
        with profile_registry._served(profile):
            EpisodeConfig(**profile.config_fields())
    except ValueError as e:
        text = str(e)
        found = text[len(CONFIG_PREFIX):].split("; ") if text.startswith(CONFIG_PREFIX) else [text]
        return [f"the configuration built from the profile: {p}" for p in found]
    except (KeyError, TypeError) as e:
        return [f"the configuration built from the profile cannot be assembled: {e!r}"]
    return []


def validate(data, engine=True):
    """Every problem of a raw profile mapping at once: its fields, its tables and their consistency, its cards, and the
    configuration it builds. An empty list means the profile may be used. engine=False leaves out the check of the
    configuration, which imports the engine configuration and so cannot run while the engine modules are imported."""
    if not isinstance(data, dict):
        return ["the profile must be a JSON object"]
    expected = ("format", *(f.name for f in fields(profile_data.AssetProfile)))
    problems = []
    missing = [k for k in expected if k not in data]
    unknown = sorted(k for k in data if k not in expected)
    if missing:
        problems.append(f"missing fields {missing}")
    if unknown:
        problems.append(f"unknown fields {unknown}")
    if "format" in data and data["format"] != profile_data.FORMAT:
        problems.append(f"format must be {profile_data.FORMAT}")
    if not isinstance(data.get("asset_id"), str) or not ASSET_ID.fullmatch(data["asset_id"]):
        problems.append("asset_id must be letters, digits and underscores")
    elif data["asset_id"] in _layer_names():
        problems.append(f"asset_id {data['asset_id']!r} is the name of a module of the asset layer, which a folder of that name "
                        "would hide from the import of CausalDiagnosis.datasets")
    if data.get("scoring") not in SCORING_MODES:
        problems.append(f"scoring must be one of {list(SCORING_MODES)}")
    kind = kinds.kind_of(data.get("evidence_kind"))
    if kind is None:
        # the evidence kind decides which tables the profile holds, so the rest cannot be checked
        return problems + [f"evidence_kind must be one of {list(kinds.names())}"]
    table_problems, components = _check_tables(data, kind)
    observation_problems, channels = _check_observations(data, kind)
    problems += table_problems + observation_problems
    problems += _check_defaults(data, kind)
    problems += _check_rules(data)
    problems += _check_sampler(data, kind)
    problems += _check_failure_modes(data, components)
    problems += _check_knowledge(data, kind, components, channels)
    problems += _check_records(data, kind)
    problems += _check_text(data, kind, channels)
    if not problems and kind.plays_episodes and engine:
        problems += config_problems(profile_data.AssetProfile._build(data))
    return problems


def _layer_names():
    """The names that CausalDiagnosis.datasets resolves to its own modules and folders, which an asset folder of the same name would
    shadow, since the import of a package finds a folder before a file of one name."""
    return {path.stem for path in profile_registry.ASSETS_DIR.glob("*.py") if path.stem != "__init__"} | {
        path.name for path in profile_registry.ASSETS_DIR.iterdir() if path.is_dir() and not (path / ports.ASSET_FILE).exists()}
