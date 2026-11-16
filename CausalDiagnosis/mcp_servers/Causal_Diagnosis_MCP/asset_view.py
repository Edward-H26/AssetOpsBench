"""The asset view of the diagnostics server and the protocol of its diagnostics model (version 7,
CausalDiagnosis/docs/causal-diagnosis-mcp.md, section The asset view).

This is the one module of the package that reads the tables of an asset. profile_view builds the view of the recorded
levels kind from the asset profile of CausalDiagnosis/datasets (the components, levels, healthy levels and degradation orders, the
grid, the channel fields and the default episode configuration) and from the rules that released.json carries. The
parameters of the rule cards, which are the schedule, the stochastic hazard, the emergency
option, the dependencies, the priorities, the history rule and the failure card, are read from the processes block of the released rules
when it is present and from the processes block of the profile's default configuration otherwise, and the view records
which source each came from. The view never carries the fault rule of the starts, the allowed starts, a start, a seed or a
split, so no formula of the server can read how the environment samples its starts. Only start_sets and development_blocks
read the fault rule, through the one split of the environment, for the fit and for the offline table, which no tool calls.
The recorded blocks and the channel
knowledge come from the profile too, and the folds and the held out discriminants from the classifiers of the environment,
run on the grid of the view.
"""
from dataclasses import dataclass
import numpy as np
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader.evidence_source import BLOCK_CAP, SEGMENT, segment_positions
from CausalDiagnosis.datasets.dataset_loader.profile_data import PROCESS_FIELDS, processes_block
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set
from CausalDiagnosis.environment.reference_policies.tier_classifiers import cross_fitted_predictions, stratified_folds
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from CausalDiagnosis.utils import frameworks


STATISTICS = ("mean", "std")  # the summaries a reading serves per channel
PUBLIC_RULE_KEYS = ("prices", "durations", "downtime_rate", "budget", "horizon", "max_steps", "clock_mode")
PRIVATE_KEYS = ("fault_rule", "allowed", "start", "seed")  # never part of a view, whatever a caller hands in
LIVE_WINDOW = SEGMENT  # the width of the live window that the environment serves, the window of the window means
__all__ = [
    "ProfileView", "asset_folder", "asset_view", "data_variable", "default_facts_path", "folds_of", "is_default_asset",
    "level_predictions", "profile_view", "recorded_blocks", "served_pairs", "served_segments", "tool_words"
]


@dataclass(frozen=True)
class ProfileView:
    """The public, static description of the asset that the server reasons over, with the rules of the episode.

    components fixes the order of every combination tuple, levels and healthy are per component, degradation lists the
    levels of a component from healthy to worst, combinations is the grid in itertools.product order, sensor_groups maps a
    group to its channel fields, summary_fields lists every channel field with every statistic, rules holds the public
    prices, durations, downtime rates, budget, horizon, step cap, clock mode and normal ranges, processes holds the
    parameters of the rule cards and processes_source says for each key whether it came from the released rules or from
    the defaults. reads_record says whether the row reads a record of recorded cycles."""
    asset_id: str
    components: tuple
    levels: dict
    healthy: dict
    degradation: dict
    combinations: tuple
    sensor_groups: dict
    statistics: tuple
    summary_fields: tuple
    rules: dict
    processes: dict
    processes_source: dict

    # ----- the grid -----
    def healthy_combination(self):
        return tuple(self.healthy[c] for c in self.components)

    def faulty(self, combination):
        return [c for c, v in zip(self.components, combination) if v != self.healthy[c]]

    def n_faulty(self, combination):
        return len(self.faulty(combination))

    # ----- fields and channels -----
    def group_fields(self, group):
        """The summary fields of the channels of one sensor group, in the order of summary_fields."""
        channels = set(self.sensor_groups[group])
        return tuple(f for f in self.summary_fields if channel_field(f) in channels)

    @property
    def reads_record(self):
        """Whether the row reads a record: its released tool table lists a tool that the contract of a declared tool surface
        classes as a read of recorded cycles (the action read_recorded). On such a row a cycle index names one recording,
        since the record holds one per cycle, and a read of a recorded cycle moves no clock. A view without a tool table,
        as the default view of an asset, reads none."""
        return any(frameworks.code("SURFACES", surface).records_reads(row) for surface in frameworks.surfaces()
                   for row in self.rules.get("tools", ()))

    def _profile(self):
        return get_profile(self.asset_id)

    def told_indicating_channels(self, events):
        """{component: the field names of the channels that indicate it} as far as the released events show the agent was told
        them, which is what the agent's board knows (Board.indicating_channels): an answer of lookup_fix (event fix_lookup) names
        them for its component on every row, and a read of the failure modes (event failure_modes, of one component or of every
        one) names them for each component with a failure mode, except on a row that reads a record, whose failure modes come
        from AssetOpsBench's fmsr server without channels. A component the agent was told nothing of has none, so that the server
        judges a verifying reading as the agent does (public_rules.repair_completion.verifies), which then asks for every channel."""
        profile = self._profile()
        with_modes = {m["component"] for m in profile.fault_modes()}
        told = set()
        for e in events:
            if e.get("event") == "fix_lookup" and e.get("component") in self.components:
                told.add(e["component"])
            elif e.get("event") == "failure_modes" and not self.reads_record:
                told |= with_modes & ({e["component"]} if e.get("component") else set(self.components))
        return {c: {profile.field_name(s) for s in profile.channels if c in profile.knowledge["indicates"].get(s, ())} if c in told else set()
                for c in self.components}

    def channel_name(self, sensor):
        """PS1 -> PS1_Pressure_bar_100Hz, the field name of a channel."""
        return self._profile().field_name(sensor)

    def sensor_of(self, name):
        """The short channel name of a channel, a field or a summary field, or None."""
        return self._profile().channel_of(name)

    def unit_and_rate(self, sensor):
        return self._profile().unit_and_rate(sensor)

    def indicates(self, sensor):
        """The components a channel indicates, as diagnostic evidence rather than a causal edge."""
        return tuple(self._profile().knowledge["indicates"].get(sensor, ()))

    def level_code(self, component, level):
        """The failure mode code of a component at a level, or None."""
        return self._profile().level_code(component, level)

    def band(self, summary_field):
        """The public normal range of a channel mean from the released rules, keyed by the summary field or by the
        channel field; None for a standard deviation, which has no public normal range, and without released rules."""
        bands = self.rules.get("normal_ranges") or {}
        band = bands.get(summary_field)
        if band is None and statistic_of(summary_field) == "mean":
            band = bands.get(channel_field(summary_field))
        return band

    # ----- the rule cards -----
    def dependencies(self, component):
        return tuple((self.processes.get("dependencies") or {}).get(component, ()))

    def priority(self, component):
        return int(self.processes["priority"][component])

    def lead_time(self, component, mode="standard"):
        emergency = self.processes.get("emergency") or {}
        return int(emergency["lead"]) if mode == "emergency" else int(self.rules["durations"]["lead_time"][component])

    def repair_downtime(self, component):
        return int(self.rules["durations"]["repair_downtime"][component])

    def visit_charge(self):
        return float(self.rules["prices"].get("visit", 0))


def channel_field(summary_field):
    """PS1_Pressure_bar_100Hz_mean -> PS1_Pressure_bar_100Hz."""
    return summary_field.rsplit("_", 1)[0]


def statistic_of(summary_field):
    return summary_field.rsplit("_", 1)[1]


def _asset_rules(profile):
    """The public rules and the processes block of the asset's default episode, the source of a view built without released
    evidence."""
    config = profile.default_config().to_json()
    rules = {key: config[key] for key in PUBLIC_RULE_KEYS}
    rules["sensor_groups"] = {g: [profile.field_name(s) for s in sensors] for g, sensors in config["sensor_groups"].items()}
    rules["statistics"] = list(STATISTICS)
    return rules, processes_block(config)


def profile_view(rules=None):
    """The view of the asset under the released rules, or under the asset's default rules when rules is None, built from the
    asset profile. Released components, levels and healthy levels must be the profile's own. The fault rule of the starts
    and every other private key of a configuration are dropped whatever the caller hands in."""
    profile = get_profile((rules or {}).get("asset_id", DEFAULT_ASSET))
    defaults, default_processes = _asset_rules(profile)
    released = {k: v for k, v in (rules or {}).items() if k not in PRIVATE_KEYS}
    public = {**defaults, **{k: v for k, v in released.items() if k not in ("components", "levels", "healthy", "processes")}}
    components = tuple(profile.components)
    levels = {c: tuple(profile.levels[c]) for c in components}
    healthy = {c: profile.healthy[c] for c in components}
    tables = {"components": list(components), "levels": {c: list(v) for c, v in levels.items()}, "healthy": healthy}
    mismatched = [key for key, table in tables.items() if key in released and released[key] != table]
    if mismatched:
        raise ValueError(f"the released {', '.join(mismatched)} are not those of the asset profile {profile.asset_id}")
    given = released.get("processes") or {}
    processes = {key: given[key] if key in given else default_processes[key] for key in PROCESS_FIELDS}
    source = {key: "released" if key in given else "defaults" for key in PROCESS_FIELDS}
    statistics = tuple(public.get("statistics") or STATISTICS)
    channels = sorted({f for fields in public["sensor_groups"].values() for f in fields})
    summary_fields = tuple(sorted(f"{channel}_{stat}" for channel in channels for stat in statistics))
    return ProfileView(asset_id=profile.asset_id, components=components, levels=levels, healthy=healthy,
                       degradation={c: tuple(profile.degradation[c]) for c in components}, combinations=profile.combinations(),
                       sensor_groups={g: tuple(f) for g, f in public["sensor_groups"].items()}, statistics=statistics,
                       summary_fields=summary_fields, rules=public, processes=processes, processes_source=source)


def asset_view(asset_id=None):
    """The view of an asset under its default rules, the default asset when asset_id is None."""
    return profile_view({"asset_id": asset_id} if asset_id else None)


def is_default_asset(asset_id):
    """Whether asset_id names the default asset, the one that models.json and grades.json beside the server serve, None
    standing for it."""
    return asset_id in (None, DEFAULT_ASSET)


# ----- the recorded evidence of the recorded levels kind -----
def recorded_blocks(view, facts_path=None):
    """{combination: array (cycles, summaries)} over the stable cycles of every recorded combination, the summaries in the
    order of view.summary_fields, and the number of records of the file. Only the fit and the offline reports call it,
    never a tool."""
    profile = get_profile(view.asset_id)
    cycles = records.load_cycles(facts_path or records.default_facts_path(profile), profile)
    by_index = {c.index: c for c in cycles}
    names = _facts_names(cycles[0], profile)
    missing = [f for f in view.summary_fields if f not in names]
    if missing:
        raise ValueError(f"the facts file lacks the summaries {missing}")
    blocks = {}
    for combination, indices in records.build_blocks(cycles).items():
        blocks[tuple(combination)] = np.array([[by_index[i].features[names[f]] for f in view.summary_fields] for i in indices])
    return blocks, len(cycles)


def start_sets(view, recorded):
    """(the development starts, the starts of the test set) of the asset among its recorded combinations, each in grid order,
    by the one split of the environment (CausalDiagnosis.datasets.splits.development_split): the test set is what the fault rule of the
    profile allows, less the combinations that the split moves to development. The offline table calls it, and the fit through
    development_blocks, never a tool."""
    profile, recorded = get_profile(view.asset_id), [tuple(h) for h in recorded]
    return development_starts(profile, recorded), starts_of_test_set(profile, recorded)


def development_blocks(view, recorded):
    """The development blocks of the asset: every recorded combination outside the test set, in grid order. They are the
    development starts and the recorded combinations that are a start of neither set, such as the healthy one. The form of the
    model is chosen on them. Only the fit calls it, never a tool."""
    recorded = {tuple(h) for h in recorded}
    test = set(start_sets(view, recorded)[1])
    return [h for h in view.combinations if h in recorded and h not in test]


def served_segments():
    """The block positions of every segment that the evidence source serves as the live window of an entry offset, for the
    rollouts of the grade and of the calibration report, which serve the recordings as the environment does."""
    return [segment_positions(index) for index in range(BLOCK_CAP)]


def served_pairs():
    """For every entry offset, the block positions of the recordings before cycle 0 and of the live window that the evidence
    source serves with it, for the rollouts of the grade, which start from what a row has recorded before it plays."""
    return [(segment_positions(index), segment_positions(index + LIVE_WINDOW)) for index in range(BLOCK_CAP)]


def _facts_names(cycle, profile):
    """Public summary field -> the feature name of the facts file, Pressure_PS1_mean for PS1_Pressure_bar_100Hz_mean."""
    names = {}
    for name in records.group_features(cycle, tuple(profile.channels), profile):
        if name.endswith(tuple(f"_{s}" for s in STATISTICS)):
            names[records.public_name(name, profile)] = name
    return names


def default_facts_path(asset_id=None):
    """The data file of an asset, the default asset when asset_id is None: the file that the environment variable of its
    profile names when the variable is set and not empty, else its release file."""
    return records.default_facts_path(get_profile(asset_id or DEFAULT_ASSET))


def data_variable(asset_id=None):
    """The environment variable that the profile of an asset, the default asset when asset_id is None, names for its data
    file, which the fit reads, so that a text of the server names it for any asset."""
    return get_profile(asset_id or DEFAULT_ASSET).records["data_env"]


def asset_folder(asset_id):
    """The folder of the asset on the search path of CausalDiagnosis.datasets, where a port keeps what it generates, None for an asset
    registered without one. The modules of this package reach the asset layer through this module alone."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports

    return ports.asset_dirs().get(asset_id)


def require_same_data(file, stored_sha256, facts_sha256, rebuild):
    """Refuse a file built from the recordings of another data file than the one in use (records.require_same_data). The
    modules of this package reach the asset layer through this module alone."""
    records.require_same_data(file, stored_sha256, facts_sha256, rebuild)


def file_sha256(path):
    """The SHA-256 of a file, read in chunks (records.file_sha256), through the one module of this package that reaches the
    asset layer."""
    return records.file_sha256(path)


def folds_of(view, n_folds, seed):
    """The fold of every combination, the stratified folds the cross-fitted classifiers of the environment use, so the
    cross fit of the effect model and the separability label of the environment split the grid alike."""
    return dict(stratified_folds(n_folds, seed, grid=(view.components, view.levels, view.combinations)))


def level_predictions(x, cycle_combo, fold_of, space=None):
    """Held out predicted level indices per cycle by the cross-fitted discriminants of the environment, the observation
    model of the urn ranking model, on the grid of the space when given."""
    grid = None if space is None else (space.components, dict(zip(space.components, space.levels)))
    return cross_fitted_predictions(x, cycle_combo, fold_of, grid=grid)


def tool_words(asset_id=None):
    """The words the texts of the tools name for an asset, the default asset when asset_id is None, so that the texts come from
    the asset and name no asset of the server's own: the nouns of the text block of the asset file, the sensor
    groups of its default rules, one channel with its public field and one component."""
    view = asset_view(asset_id)
    text = getattr(get_profile(view.asset_id), "text", None) or {}
    groups = list(view.sensor_groups)
    channel = view.sensor_of(view.group_fields(groups[0])[0])
    return {"noun": text.get("noun", "industrial asset"), "short_noun": text.get("short_noun", "asset"), "groups": groups,
            "channel": channel, "field": view.channel_name(channel), "component": view.components[0]}
