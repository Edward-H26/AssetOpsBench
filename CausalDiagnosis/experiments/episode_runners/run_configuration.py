from . import servers_of_a_run
import dataclasses
import functools
import json
from pathlib import Path
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports, recordings as records  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import environment_factory as world  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ARM_PRESETS, HEALTHY_CONTROL_PRESET, LADDER, TOKEN_PRESETS  # noqa: E402
from CausalDiagnosis.utils.paths import CONFIGS, preset_files


PRICE_TABLES = CONFIGS / "prices"
# the rows of the ladder, the presets of the ablation arms beside them, the healthy control and the token twins of the clock presets
# (experiment_tables.py), on which a model agent runs on AssetOpsBench's servers: the default of --servers of the driver and the guard of the
# tables (results_index.refused)
ROW_PRESETS = (*LADDER, *ARM_PRESETS, HEALTHY_CONTROL_PRESET, *TOKEN_PRESETS)


def load_world(facts_path=None, asset_id=DEFAULT_ASSET):
    """The data file of an asset, its records and their blocks per combination, as the loader of the evidence kind of the asset
    reads them (CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory.load_world): for the recorded levels kind the recorded cycles and their stable blocks."""
    profile = get_profile(asset_id)
    facts = Path(facts_path or records.default_facts_path(profile))
    try:
        cycles, blocks = world.load_world(asset_id, str(facts))
    except records.DataFileError as e:
        raise SystemExit(str(e)) from None
    return facts, cycles, blocks


@functools.lru_cache(maxsize=None)
def fold_line_empty(config_name, fold, fraction, master_seed=42, training_split="registered"):
    """Whether the line of the model of `fold` on a preset plays no episode at the share (start_states.fold_line_empty), from
    the preset, its recordings and the split of its asset: the one predicate that pending_grid_lines.py and the completeness of the
    tables share, kept per arguments, since a preset and its recordings do not change in a process."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.experiments.table_definitions.training_variants import asset_training

    if fold is None or start_states.fraction_plan(fraction)[0] == "whole":
        return False
    _name, config = load_config(config_name)
    _facts, cycles, blocks = load_world(None, config.asset_id)
    split = asset_training(config.asset_id, cycles.facts_sha256, split=training_split).split
    return start_states.fold_line_empty(HydraulicRigEnv(cycles, blocks, config), split, fold, fraction, master_seed)


def scenario_set(asset_id):
    """The scenario set that a framework declares for an asset under SCENARIO_SETS, the object its entry names, imported on
    first use, or None for an asset that no framework declares one for. The object gives what an anchored preset reads: DATE0,
    ASSUMPTIONS, PROFILE, scenarios_path, load_scenarios, anchor and start_of. Two sets for one asset stop the run, since the
    preset names no set."""
    from CausalDiagnosis.utils import frameworks

    declared = sorted(name for name, entry in frameworks.scenario_sets().items() if entry.get("asset_id") == asset_id)
    if len(declared) > 1:
        raise SystemExit(f"the frameworks declare the scenario sets {', '.join(declared)} for the asset {asset_id}, and an anchored "
                         "preset plays one set of an asset")
    return frameworks.code("SCENARIO_SETS", declared[0]) if declared else None


def _preset_file(spec, folder, kind):
    path = Path(spec)
    if path.suffix == ".json" and path.exists():
        return path
    found = preset_files(folder) if folder == CONFIGS else {p.stem: p for p in sorted(folder.glob("*.json"))}
    if spec in found:
        return found[spec]
    names = sorted(found)
    raise SystemExit(f"unknown {kind} {spec!r}: give a path to a JSON file or one of {names}")


def load_config(spec, asset_id=None):
    """(name, EpisodeConfig) of a preset name under configs/ or a JSON path. A preset is a composition: the big
    layers it turns on, the small switches it sets otherwise and the fields of the base it sets, which
    CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.compose builds over the asset's defaults and validates, so run.json records the whole. asset_id,
    when given, is the registered asset the preset is composed over: a preset that names no asset takes it, and a preset that
    names another asset is refused by name, so that a run plays the one asset that its configuration names."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers

    path = _preset_file(spec, CONFIGS, "config preset")
    preset = json.loads(path.read_text(encoding="utf-8"))
    if asset_id is not None:
        named = preset.get("asset_id")
        if named is not None and named != asset_id:
            raise SystemExit(f"{path}: the preset names the asset {named}, and --asset names {asset_id}; name one of them")
        try:
            get_profile(asset_id)
        except KeyError:
            raise SystemExit(f"--asset {asset_id} names no registered asset; the asset folders found are {', '.join(ports.asset_ids())}") from None
        preset = {**preset, "asset_id": asset_id}
    try:
        config = layers.compose(preset)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}") from None
    return preset.get("name", path.stem), config


def _empty_paths(value, prefix=""):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _empty_paths(v, f"{prefix}{k}/")]
    return [prefix.rstrip("/")] if value is None else []


def load_price_table(spec):
    """A price table of configs/prices/ or a JSON path, refused while any value is empty (the sme.json template)."""
    path = _preset_file(spec, PRICE_TABLES, "price table")
    table = json.loads(path.read_text(encoding="utf-8"))
    sections = {k: table[k] for k in ("prices", "durations", "downtime_rate") if k in table}
    if "prices" not in sections:
        raise SystemExit(f"{path}: a price table needs a 'prices' section")
    empty = _empty_paths(sections)
    if empty:
        raise SystemExit(f"{path}: the table {table.get('name', path.stem)!r} has empty values at {empty}; "
                         "fill them after the session with the subject matter experts")
    return {"name": table.get("name", path.stem), "source": table.get("source", ""), **sections}


def apply_price_table(config, table):
    """The configuration with the table's prices (and its durations and downtime rates when the table gives them),
    validated again through the JSON round trip of EpisodeConfig."""
    replaced = dataclasses.replace(config, **{k: table[k] for k in ("prices", "durations", "downtime_rate") if k in table})
    return type(config).from_json(replaced.to_json())
