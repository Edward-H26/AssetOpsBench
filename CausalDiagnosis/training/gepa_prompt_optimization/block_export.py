import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.experiments.table_definitions.training_variants import FOLD_CONVENTION, FOLD_CONVENTION_KEY, REGISTERED, TRAINED_AGENTS, TRAINING_SPLIT_KEY, asset_of, asset_training, block_name, built_from, check_fold_convention, loop_counts, run_reference, training_root  # noqa: E402
from . import run_folders, candidate_proposals, online_loop, loop_settings


def export(argv):
    """Writes the final block of a finished loop to blocks/<name>.txt of the training folder with its record, the block that the
    gepa, gepa_score_only or ft_gepa variant of run_episodes/ plays on the loop's fold, the score-only card only from a loop
    on the base weights and under a name of its own, and refuses to replace a
    different block. The record names the fold, the adapter the loop played on and the counts of the loop that the table of what
    each agent trained on prints."""
    ap = argparse.ArgumentParser(description="the final block of a finished online loop of GEPA, for the variants of the training table")
    ap.add_argument("run", help="the run folder of gepa_prompt_optimization.py")
    ap.add_argument("--training-root", default=None, help="the training folder, HYD_TRAINING by default")
    a = ap.parse_args(argv)
    run_dir = Path(a.run)
    meta, result = read_json(run_dir / "run.json", {}), read_json(run_dir / run_folders.RESULT, {})
    gepa = meta.get("gepa") or {}
    card = gepa.get("card")
    if meta.get("mode") != run_folders.MODE or result.get("status") != "finished" or card not in online_loop.FEEDBACK:
        raise SystemExit(f"{run_dir} is no finished online loop under a feedback card, and only such a loop gives a variant its block")
    if not result.get("proposals_answered"):
        raise SystemExit(f"{run_dir}: the reflection model answered none of the {result.get('proposals')} proposals of the loop, so "
                         "its final block is the starting block and no variant plays it as a block of GEPA")
    check_fold_convention(gepa, run_dir / "run.json")
    agent, variant, fold, asset_id = gepa.get("agent"), gepa.get("training_variant"), gepa.get("training_fold"), asset_of(meta)
    split = meta.get(TRAINING_SPLIT_KEY) or REGISTERED
    if agent not in TRAINED_AGENTS or variant not in loop_settings.LOOP_VARIANTS or fold not in asset_training(asset_id, split=split).fold_ids:
        raise SystemExit(f"{run_dir} names the agent {agent!r}, the variant {variant!r} and the fold {fold!r}, which no block of the "
                         "training table belongs to")
    final = result["final_candidate"]
    block = (run_dir / "candidates" / f"{final}_{candidate_proposals.BLOCK}.txt").read_text(encoding="utf-8")
    name = block_name(agent, variant, fold, asset_id, card)
    folder = training_root(a.training_root) / "blocks"
    target = folder / f"{name}.txt"
    if target.exists() and target.read_text(encoding="utf-8") != block:
        raise SystemExit(f"{target} holds another block; move it aside before a second loop replaces it")
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(block, encoding="utf-8")
    # the stream episodes are the training data of the gepa, gepa_score_only and ft_gepa columns, and the loop left out those that reached
    # a combination the fold's model is judged on, which only a drop of the layer 5 hazard can do
    record = {"agent": agent, "base_variant": variant, "fold": fold, FOLD_CONVENTION_KEY: FOLD_CONVENTION, "asset_id": asset_id,
              TRAINING_SPLIT_KEY: split, "card": card,
              "run": run_dir.name,
              "candidate": final, "characters": len(block),
              "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(), "adapter": meta.get("adapter"), **loop_counts(result),
              "stamp": meta.get("stamp"), "built_from": [run_reference(run_dir), *built_from(meta.get("adapter"))],
              "exported": datetime.now().isoformat(timespec="seconds")}
    write_json(target.with_suffix(".json"), record)
    print(f"wrote {target}: the block of candidate {final}, {len(block)} characters")
    return target
