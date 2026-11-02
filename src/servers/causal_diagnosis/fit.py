"""Fit the models of the diagnostics server once and write them to models.json beside this file, a file generated from
the public rig data and never committed; ensure_models fits it on first use in the drivers and the tests
(docs/hydraulic_rig/causal-diagnosis-server.md).

Format 2 holds the interventional effect model of E1 with its resampled fits, the healthy reference of E3, the conformal
calibration scores of E14, and the v1 discriminant and urn model kept for ablation A12, all fitted on one pool: the stable
cycles of every combination outside the evaluation split, which is every combination the default public fault rule can
start an episode from. The same models serve every episode, so they carry no information about the hidden combination.

usage: HYD_FACTS_PATH=<facts file> python -m servers.causal_diagnosis.fit [--interactions select|none] [--resamples N] [--seed S]
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from envs.hydraulic_rig.classifiers import COMBOS, stratified_folds
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, build_blocks, default_facts_path, load_cycles
from servers.json_files import write_json_atomically

from .discriminant_urn import fit_discriminant_urn
from .effect_model import FOLD_SEED, FOLDS, PoolData, combo_key, fit_effect_model, n_faulty
from .posterior import DEFAULT_RANKING_MODEL, RANKING_MODELS

MODELS_PATH = Path(__file__).resolve().with_name("models.json")
MODELS_FORMAT = 2
EVALUATION_FAULT_RULE = {"min_faulty": 1, "max_faulty": 2}  # the default public fault rule; its combinations are the evaluation split
RESAMPLES, RESAMPLE_SEED = 200, 0


def split(fault_rule=EVALUATION_FAULT_RULE):
    """The fitting pool and the evaluation combinations under the fault rule."""
    low, high = fault_rule["min_faulty"], fault_rule["max_faulty"]
    pool = [h for h in COMBOS if not low <= n_faulty(h) <= high]
    evaluation = [h for h in COMBOS if low <= n_faulty(h) <= high]
    return pool, evaluation


def fit(facts_path, interactions="select", resamples=RESAMPLES, seed=RESAMPLE_SEED):
    cycles = load_cycles(facts_path)
    blocks = build_blocks(cycles)
    pool, evaluation = split()
    effect = fit_effect_model(cycles, blocks, pool, interactions=interactions, resamples=resamples, seed=seed)
    data = PoolData(cycles, blocks, pool)
    fold_all = stratified_folds(FOLDS, FOLD_SEED)
    urn = fit_discriminant_urn(data, {h: fold_all[h] for h in data.pool})
    low, high = EVALUATION_FAULT_RULE["min_faulty"], EVALUATION_FAULT_RULE["max_faulty"]
    return {
        "format": MODELS_FORMAT,
        "generated": datetime.date.today().isoformat(),
        "facts_sha256": hashlib.sha256(Path(facts_path).read_bytes()).hexdigest(),
        "records": len(cycles),
        "pool": {"rule": (f"stable cycles of every combination whose number of faulty components is outside {low} to {high}, "
                          "the evaluation split of the default public fault rule"),
                 "combinations": [combo_key(h) for h in pool], "evaluation": [combo_key(h) for h in evaluation],
                 "cycles": int(sum(len(x) for x in data.arrays)), "folds": FOLDS, "fold_seed": FOLD_SEED},
        "components": list(COMPONENTS), "levels": {c: list(LEVELS[c]) for c in COMPONENTS}, "healthy": dict(HEALTHY),
        "ranking_models": list(RANKING_MODELS), "default_ranking_model": DEFAULT_RANKING_MODEL,
        "effect": effect.to_json(),
        "discriminant_urn": urn,
    }


def write_models(path=MODELS_PATH, **options):
    out = fit(default_facts_path(), **options)
    write_json_atomically(path, out)
    return out


def ensure_models(path=MODELS_PATH):
    """Fit the models when the file is missing or in an older format."""
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("format") != MODELS_FORMAT:
        write_models(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="fit models.json of the diagnostics server from the rig data")
    parser.add_argument("--interactions", choices=("select", "none"), default="select")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=RESAMPLE_SEED)
    parser.add_argument("--out", type=Path, default=MODELS_PATH)
    args = parser.parse_args(argv)
    out = write_models(args.out, interactions=args.interactions, resamples=args.resamples, seed=args.seed)
    effect = out["effect"]
    print(f"wrote {args.out}: format {out['format']}, pool of {len(out['pool']['combinations'])} combinations, "
          f"{len(effect['interactions'])} interaction pair(s) selected, {len(effect['resampled'])} resampled fits, "
          f"{args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
