"""A first asset file for a new asset of the recorded levels kind (CausalDiagnosis/docs/adding-a-dataset.md, section Steps of a port).

    PYTHONPATH=src python -m CausalDiagnosis.datasets.new_dataset_scaffold <asset_id> --component <name>=<levels from healthy to worst> [--component ...]

The command writes the folder CausalDiagnosis/datasets/<asset_id> with the two files of an asset: the package __init__.py, empty,
which an asset of the recorded levels kind without a label or a table of its own keeps as it is, and asset.json with every
field of the asset file, the components with their levels, healthy levels and degradation orders as given, and the mark
"TO FILL: <path>" in every other field, an open field. It then runs validate and prints the open fields left and the problems
that validate reports. It refuses an asset id that is not letters, digits and underscores, one that names a module or a
folder of the asset layer, which the package of the asset would hide from the import of CausalDiagnosis.datasets, and a folder that
exists, so a bundled asset is never overwritten. It adds no rule kind and changes nothing in the engine.
"""
import argparse
import json

from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, dataset_packages as ports
from CausalDiagnosis.datasets.dataset_loader.profile_validation import ASSET_ID, DATA_KEYS, DEFAULT_TABLES, FAILURE_MODE_KEYS, SOURCE_KEYS, TEXT_KEYS, _layer_names, validate
from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR
from CausalDiagnosis.datasets.dataset_loader.profile_data import FORMAT

KIND = "recorded_levels"  # the evidence kind of the draft
OPEN = "TO FILL: "  # the mark of an open field, with the path of the field after it
PER_COMPONENT = {("prices", "parts"), ("prices", "fix_kind"), ("durations", "lead_time"), ("durations", "repair_downtime")}
KNOWLEDGE_KEYS = ("fmea_source", "association_source", "indicates", "correlations")


def fill(path):
    return OPEN + path


def per_component(components, path):
    return {c: fill(f"{path}.{c}") for c in components}


def draft(asset_id, degradation):
    """The asset file of the components that degradation maps to their levels from healthy to worst, with an open field
    for every value the caller did not give."""
    components = list(degradation)
    defaults = {table: {key: per_component(components, f"defaults.{table}.{key}") if (table, key) in PER_COMPONENT
                        else fill(f"defaults.{table}.{key}") for key in keys}
                for table, keys in DEFAULT_TABLES.items()}
    modes = [(c, level) for c in components for level in degradation[c]]
    failure_modes = [{"component": c, "level": level,
                      **{key: fill(f"failure_modes[{i}].{key}") for key in FAILURE_MODE_KEYS if key not in ("component", "level")}}
                     for i, (c, level) in enumerate(modes)]
    text = {key: fill(f"text.{key}") for key in (*TEXT_KEYS, "example_channel")}
    text.update(whole_asset_patterns=[fill("text.whole_asset_patterns[0]")],
                fallback={key: fill(f"text.fallback.{key}") for key in ("asset_type", "description", "site")})
    records = {key: fill(f"records.{key}") for key in (*kinds.kind_of(KIND).record_keys, *DATA_KEYS, *SOURCE_KEYS)}
    records["label_keys"] = per_component(components, "records.label_keys")
    return {
        "format": FORMAT,
        "asset_id": asset_id,
        "evidence_kind": KIND,
        "components": components,
        "levels": {c: sorted(levels) for c, levels in degradation.items()},
        "healthy": {c: levels[0] for c, levels in degradation.items()},
        "degradation": {c: list(levels) for c, levels in degradation.items()},
        "observation_kinds": fill("observation_kinds"),
        "channels": fill("channels"),
        "statistics": {"recorded": fill("statistics.recorded"), "served": fill("statistics.served")},
        "defaults": defaults,
        "rules": fill("rules"),
        "sampler": {"fault_rule": {"min_faulty": fill("sampler.fault_rule.min_faulty"), "max_faulty": fill("sampler.fault_rule.max_faulty")}},
        "failure_modes": failure_modes,
        "knowledge": {key: fill(f"knowledge.{key}") for key in KNOWLEDGE_KEYS},
        "records": records,
        "text": text,
        "scoring": fill("scoring"),
    }


def open_fields(value):
    """The paths of the open fields a profile still holds, in file order."""
    if isinstance(value, dict):
        return [p for v in value.values() for p in open_fields(v)]
    if isinstance(value, list):
        return [p for v in value for p in open_fields(v)]
    return [value[len(OPEN):]] if isinstance(value, str) and value.startswith(OPEN) else []


def scaffold(asset_id, degradation, assets_dir=ASSETS_DIR):
    """Write the empty package and the draft into a new folder <assets_dir>/<asset_id> and return the path of its asset
    file."""
    if not ASSET_ID.fullmatch(asset_id):
        raise SystemExit(f"asset id {asset_id!r} must be letters, digits and underscores")
    if asset_id in _layer_names():
        raise SystemExit(f"asset id {asset_id!r} names a module or a folder of the asset layer, which the package of the asset would hide")
    folder = assets_dir / asset_id
    if folder.exists():
        raise SystemExit(f"{folder} exists, and the scaffold never overwrites an asset")
    folder.mkdir()
    (folder / "__init__.py").write_text("", encoding="utf-8")
    path = folder / ports.ASSET_FILE
    path.write_text(json.dumps(draft(asset_id, degradation), indent=1) + "\n", encoding="utf-8")
    return path


def report(data):
    """The lines that say what is still to fill: the open fields, then every problem that validate reports."""
    left, problems = open_fields(data), validate(data)
    lines = [f"{len(left)} open fields to fill:", *(f"  {p}" for p in left)]
    return lines + ([f"validate reports {len(problems)} problems:", *(f"  {p}" for p in problems)] if problems else ["validate: valid"])


def component_levels(text):
    """name=levels from healthy to worst, such as bearing=0,1,2."""
    name, _, levels = text.partition("=")
    try:
        values = [int(v) for v in levels.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} must read name=levels, integers from healthy to worst separated by commas")
    if not name:
        raise argparse.ArgumentTypeError(f"{text!r} names no component")
    return name, values


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a first asset file with open fields and print what is still to fill.")
    ap.add_argument("asset_id", help="letters, digits and underscores, the name of the new folder under CausalDiagnosis/datasets/")
    ap.add_argument("--component", action="append", type=component_levels, required=True,
                    help="name=levels from healthy to worst, such as bearing=0,1,2, once per component in the order of the diagnosis")
    args = ap.parse_args(argv)
    names = [name for name, _ in args.component]
    if len(set(names)) != len(names):
        ap.error(f"the components {names} repeat a name")
    path = scaffold(args.asset_id, dict(args.component))
    print(f"wrote {path.parent / '__init__.py'} and {path}")
    print("\n".join(report(json.loads(path.read_text(encoding="utf-8")))))


if __name__ == "__main__":
    main()
