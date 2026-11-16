import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
from . import held_out_folds, variant_names


def run_reference(run_dir):
    """A run as a record names it: its folder name and the SHA-256 of its run.json."""
    return {"run": Path(run_dir).name, "run_json_sha256": sha256_file(Path(run_dir) / "run.json")}


def adapter_name_of_arm(agent, arm, fold, asset_id):
    """The served name of the adapter trained on one arm of an agent's data of one fold, the name of its folder under adapters/."""
    held_out_folds.check_fold(fold, asset_id)
    return f"rft_{agent}" + ("" if arm == variant_names.FILTERED else f"_{variant_names.UNFILTERED}") + f"_f{fold}"


def adapter_name(agent, variant, fold, asset_id):
    """The served name of the adapter a variant of a fold plays, None for a variant on the base weights."""
    return adapter_name_of_arm(agent, variant_names.ARM_OF[variant], fold, asset_id) if variant in variant_names.ARM_OF else None


def block_name(agent, base_variant, fold, asset_id, card="verifier"):
    """The name of the block that the loop of GEPA on an agent, a base variant (as_released or ft), a fold and a feedback card
    exports, the score-only card on the base weights alone."""
    held_out_folds.check_fold(fold, asset_id)
    if card not in variant_names.CARDS or (card == "score_only" and base_variant != variant_names.AS_RELEASED):
        raise SystemExit(f"no block of the training table comes from a loop under the card {card!r} on {base_variant!r}: the "
                         "score-only control plays the base weights alone")
    prefix = "gepa_score_only" if card == "score_only" else "gepa" if base_variant == variant_names.AS_RELEASED else "gepa_ft"
    return f"{prefix}_{agent}_f{fold}"


def data_name(agent, fold, asset_id):
    """The folder under data/ that rejection_sampling_examples.py writes for an agent and a fold."""
    held_out_folds.check_fold(fold, asset_id)
    return f"{agent}_f{fold}"


def training_root(root=None):
    folder = root or os.environ.get(variant_names.TRAINING_ROOT_ENV)
    if not folder:
        raise SystemExit(f"a trained variant reads its adapter or block under the training folder, so set {variant_names.TRAINING_ROOT_ENV}")
    return Path(folder)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_record(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path} cannot be read as a training record: {exc}") from None


def check_fold_convention(record, source):
    """Refuses a training record that does not state that its fold number names the fold its model holds out."""
    found = (record or {}).get(variant_names.FOLD_CONVENTION_KEY)
    if found != variant_names.FOLD_CONVENTION:
        raise SystemExit(f"{source} was written under the earlier fold convention, in which the fold number named the fold a model "
                         f"trained on ({variant_names.FOLD_CONVENTION_KEY} {found!r}, not {variant_names.FOLD_CONVENTION!r}), so a model could play the starts "
                         "it trained on: move the training folder aside and build it again")


def check_split(record, source, split):
    """Refuses a data manifest, an adapter manifest or a block record of another split than the run's, so the files of a trial on
    the development split never enter a run of the registered split, nor the reverse. A record that names no split is registered."""
    found = (record or {}).get(held_out_folds.TRAINING_SPLIT_KEY) or held_out_folds.REGISTERED
    if found != split:
        raise SystemExit(f"{source} belongs to the {found} split, and the run plays the {split} split")


def check_asset(record, source, asset_id):
    """Refuses a data manifest, an adapter manifest or a block record that names another asset than the run's: each asset has a
    training folder of its own."""
    found = (record or {}).get("asset_id")
    if found != asset_id:
        raise SystemExit(f"{source} belongs to the asset {found!r}, and the run plays {asset_id}: each asset has a training folder of "
                         "its own")


def check_served_after_training(name, ended, served_started):
    """Refuses a serving job that started before the training job of the adapter ended, since it serves other weights under the
    adapter's name. Both times carry their offset from UTC, as train_lora.py and the endpoint file write them."""
    try:
        trained, served = datetime.fromisoformat(ended), datetime.fromisoformat(served_started)
    except (TypeError, ValueError):
        raise SystemExit(f"the training of the adapter {name} ended at {ended!r} and the server started at {served_started!r}, "
                         "times that cannot be compared") from None
    if trained.tzinfo is None or served.tzinfo is None:
        raise SystemExit(f"the training of the adapter {name} ended at {ended} and the server started at {served_started}, and a time "
                         "without its offset from UTC cannot be compared")
    if served < trained:
        raise SystemExit(f"the server started at {served_started}, before the training of the adapter {name} ended at {ended}, so it "
                         "serves other weights under that name; start the serving job again")


def adapter_of(agent, variant, fold, asset_id, root=None, served_started=None, split=held_out_folds.REGISTERED):
    """The adapter of a variant and a fold as run.json records it, or None: checked against the manifest of its training job and
    the data manifest of the agent and fold it trained on, with the counts that the table of what each agent trained on prints, the
    episodes sampled, kept by the filter and drawn for the control, the examples and the optimizer steps of this adapter's arm, and
    the episodes left out of the data. A manifest of another split than the run's is refused (check_split). served_started is the
    start of the serving job from its endpoint file: a server that started
    before the training job ended is refused, and without a start time the record says that nothing was compared."""
    name = adapter_name(agent, variant, fold, asset_id)
    if name is None:
        return None
    root = training_root(root)
    folder = root / "adapters" / name
    manifest_path, data_path = folder / variant_names.ADAPTER_MANIFEST, root / "data" / data_name(agent, fold, asset_id) / variant_names.DATA_MANIFEST
    weights = folder / "adapter_model.safetensors"
    data_record = read_record(data_path) if data_path.exists() else None
    if data_record is not None:
        check_fold_convention(data_record, data_path)
        check_asset(data_record, data_path, asset_id)
        check_split(data_record, data_path, split)
    gate = (data_record or {}).get("gate")
    if gate is not None and gate.get("second_round_due"):
        # the gate is not final while a round of sampling is due, so neither an adapter nor a marker stands for the fold yet
        raise SystemExit(f"the data of {agent} on fold {fold} waits for a further round of sampling, so no run of {variant} starts yet")
    if gate is not None and gate.get("trained") is False:
        # no adapter trains on too few kept episodes, so the variant's run of this fold is a marker that the tables
        # replace by the runs without training of the fold, and an adapter folder beside such a gate is a contradiction
        if folder.exists():
            raise SystemExit(f"the data of {agent} on fold {fold} stayed below the registered minimum and {folder} holds an adapter "
                             "anyway, so the run cannot tell what it plays")
        return {"name": None, "trained": False, "reason": "below_minimum", "alias": variant_names.AS_RELEASED, "arm": variant_names.ARM_OF[variant], "fold": fold,
                "asset_id": asset_id, held_out_folds.TRAINING_SPLIT_KEY: split, "data_manifest_sha256": sha256_file(data_path), "gate": gate}
    if not manifest_path.exists() or not weights.exists():
        raise SystemExit(f"--training-variant {variant} plays the adapter {name}, and {folder} holds no {variant_names.ADAPTER_MANIFEST} with its weights")
    manifest = read_record(manifest_path)
    check_fold_convention(manifest, manifest_path)
    check_asset(manifest, manifest_path, asset_id)
    check_split(manifest, manifest_path, split)
    if (manifest.get("agent"), manifest.get("arm"), manifest.get("fold")) != (agent, variant_names.ARM_OF[variant], fold):
        raise SystemExit(f"the adapter {name} was trained for the agent {manifest.get('agent')!r} on the arm {manifest.get('arm')!r} of "
                         f"fold {manifest.get('fold')!r}, and --training-variant {variant} of the agent {agent!r} on fold {fold} needs "
                         f"the arm {variant_names.ARM_OF[variant]!r}")
    data_sha = sha256_file(data_path) if data_path.exists() else None
    if data_sha is None or manifest.get("data_manifest_sha256") != data_sha:
        raise SystemExit(f"the adapter {name} trained on the data manifest {manifest.get('data_manifest_sha256')}, and "
                         f"data/{data_name(agent, fold, asset_id)}/{variant_names.DATA_MANIFEST} of the training folder is {data_sha}, so its counts "
                         "cannot be told")
    data, counts = read_record(data_path), manifest.get("counts") or {}
    arm = data["arms"][variant_names.ARM_OF[variant]]
    if served_started is not None:
        check_served_after_training(name, manifest.get("ended"), served_started)
    return {"name": name, "path": f"adapters/{name}", "arm": variant_names.ARM_OF[variant], "fold": fold, "asset_id": asset_id, held_out_folds.TRAINING_SPLIT_KEY: split,
            "weights_sha256": sha256_file(weights),
            "manifest_sha256": sha256_file(manifest_path), "data_manifest_sha256": data_sha,
            "trained_ended": manifest.get("ended"), "served_started": served_started, "served_checked": served_started is not None,
            "stamp": manifest.get("stamp"), "built_from": manifest.get("built_from"),
            "training": {"episodes_sampled": data["episodes_sampled"], "kept_by_filter": data["arms"][variant_names.FILTERED]["episodes"],
                         "control_trajectories": data["arms"][variant_names.UNFILTERED]["episodes"], "examples": counts.get("kept"),
                         "examples_over_max_length": counts.get("dropped_over_max_length"), "optimizer_steps": manifest.get("steps"),
                         variant_names.VISITS: arm.get(variant_names.VISITS), "gate": data.get("gate")}}


def block_of(agent, variant, fold, asset_id, root=None, split=held_out_folds.REGISTERED):
    """(text, record) of the block a variant of a fold plays, checked against its export record, or ("", None) for a variant
    without one."""
    if variant not in variant_names.BLOCK_OF:
        return "", None
    base, card = variant_names.BLOCK_OF[variant]
    playing = adapter_of(agent, variant, fold, asset_id, root, split=split)
    if variant_names.is_marker(playing):
        return "", None  # no adapter on this fold, so no loop ran on it and the marker run plays nothing
    name = block_name(agent, base, fold, asset_id, card)
    text_path = training_root(root) / "blocks" / f"{name}.txt"
    record_path = text_path.with_suffix(".json")
    if not text_path.exists() or not record_path.exists():
        raise SystemExit(f"--training-variant {variant} plays the block {name}, and {text_path.parent} holds no {name}.txt with its record")
    text = text_path.read_text(encoding="utf-8")
    record = read_record(record_path)
    check_fold_convention(record, record_path)
    check_asset(record, record_path, asset_id)
    check_split(record, record_path, split)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    expected = (agent, base, fold, card, digest)
    if (record.get("agent"), record.get("base_variant"), record.get("fold"), record.get("card"), record.get("sha256")) != expected:
        raise SystemExit(f"the block {name} does not match its record: agent {record.get('agent')!r}, base variant "
                         f"{record.get('base_variant')!r}, fold {record.get('fold')!r}, card {record.get('card')!r}, recorded SHA-256 "
                         f"{record.get('sha256')} against {digest}")
    # a block fits the weights its loop played on, so the ft_gepa variant plays it only on the adapter of that loop
    tuned_on = (record.get("adapter") or {}).get("weights_sha256")
    playing = (playing or {}).get("weights_sha256")
    if tuned_on != playing:
        raise SystemExit(f"the block {name} was written by a loop on the adapter weights {tuned_on}, and --training-variant {variant} "
                         f"plays the weights {playing}")
    return text, {"name": name, "path": f"blocks/{name}.txt", "fold": fold, held_out_folds.TRAINING_SPLIT_KEY: split, "card": card, "sha256": digest,
                  "characters": len(text),
                  "run": record.get("run"), "candidate": record.get("candidate"), **{key: record.get(key) for key in variant_names.LOOP_COUNTS},
                  "stamp": record.get("stamp"), "built_from": record.get("built_from")}


def built_from(record):
    """The runs behind an adapter record or a block record, each with the SHA-256 of its run.json: the sampling runs of the data of an
    adapter, and for a block the run of its loop with the runs behind the adapter the loop played on, which the replay check reads."""
    return list((record or {}).get("built_from") or [])
