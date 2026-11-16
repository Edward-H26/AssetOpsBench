from CausalDiagnosis.utils.paths import CONFIGS, ROOT
import functools
import json
from pathlib import Path
from typing import NamedTuple


TRAINING_DIR = CONFIGS / "training"  # one folder per asset, configs/training/<asset_id>/
CONFIG_NAME = "rft.json"  # the registered numbers of the training of an asset, beside its folds file
DEFAULT_ASSET = "hyd_1"  # the asset of a run whose configuration names none, CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config.DEFAULT_ASSET (a test keeps both)
# the splits of an asset: the registered split of the test set, and the development split of a trial, the fold rule
# dealt over the development starts, each in the folds file that the folds block of rft.json names under its key
REGISTERED, DEVELOPMENT = "registered", "development"
SPLITS = (REGISTERED, DEVELOPMENT)
FOLDS_KEYS = {REGISTERED: "file", DEVELOPMENT: "development_file"}
TRAINING_SPLIT_KEY = "training_split"  # the run.json key of the split, and that of the records of the data, the adapters and the blocks


class AssetTraining(NamedTuple):
    """The training files of one asset: its registered numbers and its split, the output of write_training_split.py fixed before any run."""
    asset_id: str
    config_file: Path
    config: dict
    folds_file: Path
    split: dict  # {start combination: (fold, the held-out folds whose models train on it)}
    fault_folds: dict  # {(component, level): fold} of every single fault
    fold_ids: tuple  # 1 to K
    components: list
    facts_sha256: str  # the SHA-256 of the data file whose recordings the split was dealt from, None when the file names none
    training_split: str  # REGISTERED or DEVELOPMENT


# what a caller passes that holds no recordings, as the grids and the tables do: it compares no data file with the split
WITHOUT_RECORDINGS = object()


@functools.lru_cache(maxsize=None)
def asset_training(asset_id=DEFAULT_ASSET, facts_sha256=WITHOUT_RECORDINGS, split=REGISTERED):
    """The training files of an asset under configs/training/<asset_id>/ (CausalDiagnosis/docs/adding-a-dataset.md, step 6), with the folds
    file of the split, the registered one or the development one of a trial. A caller that
    holds the recordings of a run names their SHA-256, records.Recordings.facts_sha256, and a split dealt from the recordings of
    another data file, or from recordings that name no file, is refused with the command that deals it again, since its start
    combinations are those of that file."""
    config_file = TRAINING_DIR / asset_id / CONFIG_NAME
    if not config_file.exists():
        raise SystemExit(f"the asset {asset_id} has no training configuration {config_file}, see CausalDiagnosis/docs/adding-a-dataset.md")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if split not in SPLITS:
        raise SystemExit(f"the training split is one of {SPLITS}, not {split!r}")
    named = config["folds"].get(FOLDS_KEYS[split])
    if not named:
        raise SystemExit(f"{config_file} names no folds file of the {split} split under folds.{FOLDS_KEYS[split]}, see CausalDiagnosis/docs/adding-a-dataset.md")
    folds_file = config_file.parent / named
    record = json.loads(folds_file.read_text(encoding="utf-8"))
    if record.get("asset_id", asset_id) != asset_id:
        raise SystemExit(f"{folds_file} holds the split of {record.get('asset_id')}, not of {asset_id}")
    if record.get("starts", REGISTERED) != split:
        raise SystemExit(f"{folds_file} was dealt over the {record.get('starts')} starts, and the {split} split reads it")
    split = {tuple(row["combination"]): (row["fold"], tuple(row["models"])) for row in record["combinations"]}
    fault_folds = {(row["component"], row["level"]): row["fold"] for row in record["atoms"]}
    training = AssetTraining(asset_id, config_file, config, folds_file, split, fault_folds, tuple(range(1, record["k"] + 1)),
                             list(record["components"]), record.get("facts_sha256"), split)
    if facts_sha256 is not WITHOUT_RECORDINGS:
        # imported here: train_lora.py imports this module in the training environment, which holds no recordings
        from CausalDiagnosis.datasets.dataset_loader import recordings as records

        root = ROOT  # the fork root, from which the command is run
        shown = folds_file.relative_to(root) if folds_file.is_relative_to(root) else folds_file
        records.require_same_data(shown, training.facts_sha256, facts_sha256,
                                  f"Deal it again with `python -m CausalDiagnosis.experiments.table_definitions.write_training_split --asset {asset_id} --k {record['k']} "
                                  f"--seed {record['seed']} --starts {split} --out {shown}`.")
    return training


def fold_ids(asset_id=DEFAULT_ASSET, split=REGISTERED):
    """The folds of a split of an asset, 1 to K, read from its folds file."""
    return asset_training(asset_id, split=split).fold_ids


def asset_of(meta):
    """The asset of a run: asset_id of run.json, then the asset of its configuration, then the default asset for a run.json
    older than both keys. The one rule, which results_index, replay_check and the training modules read."""
    return meta.get("asset_id") or (meta.get("config") or {}).get("asset_id") or DEFAULT_ASSET


# the values of the default asset, for the grids and the tables of the hydraulic rig: no function that a run calls reads them,
# since each of those takes the asset of the run and reads asset_training and numbers of that asset
_DEFAULT = asset_training()
CONFIG_FILE, CONFIG, FOLDS_FILE = _DEFAULT.config_file, _DEFAULT.config, _DEFAULT.folds_file
FOLD_COMPONENTS, FOLDS, FAULT_FOLDS, TRAINING_FOLDS = _DEFAULT.components, _DEFAULT.split, _DEFAULT.fault_folds, _DEFAULT.fold_ids
LEVEL_EVENTS = ("hazard_drop", "scheduled_drop")  # the clock's events that set the level of one component


def check_fold(fold, asset_id, split=REGISTERED):
    if fold not in fold_ids(asset_id, split):
        raise SystemExit(f"the {split} split of {asset_id} names the folds {fold_ids(asset_id, split)}, got {fold!r}")


def fold_of(combination, asset_id=DEFAULT_ASSET, split=REGISTERED):
    """The fold of a start combination in a split of the asset, the registered one by default, as scores.csv or a hidden record
    writes it, the fold whose model plays its test episodes."""
    training = asset_training(asset_id, split=split)
    try:
        return training.split[tuple(combination)][0]
    except KeyError:
        raise SystemExit(f"{combination} is no start combination of the {split} split {training.folds_file}") from None


def combinations_played(log, components):
    """Every combination the asset played in an episode, in order, read from its hidden log: the start that reset drew, then the
    combination after each change of a level that the clock logged. A drop of the hazard or of the schedule names the component
    and its new level, and a closed work order and an expired emergency repair name the whole combination after them."""
    combination, played = None, []
    for entry in log:
        if entry.get("event") == "reset":
            combination = list(entry["hidden"])
        elif entry.get("combination_after") is not None:
            combination = list(entry["combination_after"])
        elif entry.get("event") in LEVEL_EVENTS:
            combination[components.index(entry["component"])] = entry["to_level"]
        else:
            continue
        if tuple(combination) not in played:
            played.append(tuple(combination))
    return played


def visits_test_combination(log, fold, asset_id, split=REGISTERED):
    """Whether a training episode of the model of fold `fold`, the fold it holds out, played a combination that holds a single fault
    of that fold, in any combination and not only in the start combinations of the test set. The split is closed under repair, so
    only a drop of the layer 5 hazard can lead there. Such an episode is left out of the data and of the stream of a loop, and the
    table of what each agent trained on counts it."""
    training = asset_training(asset_id, split=split)
    return any(training.fault_folds.get((component, level)) == fold
               for combination in combinations_played(log, training.components)
               for component, level in zip(training.components, combination))
