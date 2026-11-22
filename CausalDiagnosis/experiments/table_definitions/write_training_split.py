"""Writes the registered training split of an asset (CausalDiagnosis/docs/adding-a-dataset.md, step 6): the general rule of
CausalDiagnosis/datasets/splits/train_test_folds.py with K folds and a seed on the start combinations of the test set of the asset, the recorded combinations its
fault rule allows less those that the development rule moved to development (development_split.starts_of_test_set). Each
start combination is written with `fold`, the held-out fold whose model tests it, and `models`, the held-out folds whose models
train on it, and each single fault with its fold, and facts_sha256 names the data file whose recordings the split was dealt from,
so that a run on the recordings of another data file refuses it (training.asset_training). The file is written once, before
any run, and the training records read it.
configs/training/hyd_1/folds.json is the output of this command for hyd_1 with K = 2 and seed 0, which a test checks byte for byte.
With --starts development the same rule is dealt over the development starts of the asset (development_split.development_starts),
the development split of a trial, which rft.json names under folds.development_file. configs/training/hyd_1/
folds_development.json is its output for hyd_1 with K = 2 and seed 0, checked byte for byte as well.

    PYTHONPATH=src python -m CausalDiagnosis.experiments.table_definitions.write_training_split --asset <asset_id> --k <K> --out <folds file>
        [--seed 0] [--starts registered|development] [--profile <asset file>] [--facts <data file>]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the fork root, which holds CausalDiagnosis/
from CausalDiagnosis.datasets.dataset_loader import recordings as records  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, load_profile, register_profile, unregister_profile  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set  # noqa: E402
from CausalDiagnosis.datasets.splits.train_test_folds import fault_folds  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import combination_atoms  # noqa: E402

RULE = "CausalDiagnosis.datasets.splits.train_test_folds.fault_folds through CausalDiagnosis/experiments/table_definitions/write_training_split.py"
ABOUT = ("the training split of the asset by the general rule of CausalDiagnosis/datasets/splits/train_test_folds.py: the model that holds out fold h trains on "
         "the start combinations whose models name h and plays the test episodes of the start combinations of fold h "
         "(CausalDiagnosis/docs/experiments.md, section Training in the environment)")


DEALT_OVER = {"registered": starts_of_test_set, "development": development_starts}  # the starts that each split is dealt over


def split_record(env, asset_id, k, seed, test=None, dealt_over="registered", name="folds"):
    """The split of the starts test, by default the start combinations of env, which fault_folds reads back from the file alone.
    dealt_over names the starts that test holds, those of the test set for the registered split or the development starts, and
    name is the stem of the file the record goes to, since every file under configs/ names itself by its stem."""
    components = list(env.config.components)
    starts = {combination: combination_atoms(env, combination) for combination in sorted(env.candidates if test is None else test)}
    split, atom_fold = fault_folds(starts, components, k, seed)
    atoms = sorted(atom_fold.items(), key=lambda item: (components.index(item[0][0]), item[0][1]))
    return {"name": name, "about": ABOUT, "rule": RULE, "asset_id": asset_id, "k": k, "seed": seed, "starts": dealt_over,
            "facts_sha256": env.facts_sha256,
            "components": components,
            "combinations": [{"combination": list(combination), "fold": fold, "trains": bool(models), "models": list(models)}
                             for combination, (fold, models) in sorted(split.items())],
            "atoms": [{"component": component, "level": level, "fold": fold} for (component, level), fold in atoms]}


def fold_counts(record):
    """Per held-out fold h, the start combinations its model tests and those it trains on, the counts from which the training
    start states of rft.json and the lines of the training table follow."""
    rows = record["combinations"]
    return {h: (sum(row["fold"] == h for row in rows), sum(h in row["models"] for row in rows)) for h in range(1, record["k"] + 1)}


def _environment(profile, facts):
    path = Path(facts or records.default_facts_path(profile))
    try:
        cycles = records.load_cycles(str(path), profile)
    except records.DataFileError as e:
        raise SystemExit(str(e)) from None
    return HydraulicRigEnv(cycles, records.build_blocks(cycles), profile.default_config())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Writes a training split of an asset, the registered one or the development one of a "
                                                 "trial.")
    parser.add_argument("--asset", required=True, help="the asset_id of the asset file")
    parser.add_argument("--k", type=int, required=True, help="the number of folds, at least 2 and at most the single faults")
    parser.add_argument("--seed", type=int, default=0, help="the seed of the dealing, which the written split records")
    parser.add_argument("--profile", help="an asset file to read in place of the bundled one, such as a draft of a port")
    parser.add_argument("--facts", help="the data file, else the one the profile names")
    parser.add_argument("--starts", choices=tuple(DEALT_OVER), default="registered",
                        help="the starts the split is dealt over: those of the test set, or the development starts for a trial")
    parser.add_argument("--out", required=True, help="the folds file to write")
    a = parser.parse_args(argv)
    draft = register_profile(load_profile(a.profile)) if a.profile else None
    try:
        if draft is not None and draft.asset_id != a.asset:
            raise SystemExit(f"--profile holds the asset {draft.asset_id}, not {a.asset}")
        profile = get_profile(a.asset)
        try:
            env = _environment(profile, a.facts)
            record = split_record(env, profile.asset_id, a.k, a.seed, DEALT_OVER[a.starts](profile, set(env.source.blocks)), a.starts,
                                  Path(a.out).stem)
        except ValueError as e:
            raise SystemExit(f"no split written: {e}") from None
    finally:
        if draft is not None:
            unregister_profile(draft.asset_id)
    out = Path(a.out)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(record['combinations'])} {a.starts} start combinations of {record['asset_id']} in {a.k} folds, seed {a.seed}")
    for h, (tested, training) in fold_counts(record).items():
        print(f"held-out fold {h}: {tested} start combinations tested, {training} start combinations to train on")
    return record


if __name__ == "__main__":
    main()
