"""Stable machine-group splits, with input identities and no answer fields."""

import hashlib
import json
import math
from pathlib import Path


def input_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(dataset):
    result = {}
    for qa_id in dataset.question_ids():
        asset = dataset.public_fact(qa_id).get("asset_id")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"{qa_id}: a machine split requires asset_id")
        result.setdefault(asset, []).append(qa_id)
    if len(result) < 2:
        raise ValueError("a machine split requires at least two distinct assets")
    return result


def make_split(dataset, facts, qa, dev_fraction=0.2, seed="0"):
    if isinstance(dev_fraction, bool) or not math.isfinite(dev_fraction) or not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    by_asset = groups(dataset)
    ordered = sorted(by_asset, key=lambda asset: hashlib.sha256(json.dumps([str(seed), asset]).encode()).hexdigest())
    count = max(1, min(len(ordered) - 1, round(len(ordered) * dev_fraction)))
    partitions = {"dev": ordered[:count], "test": ordered[count:]}
    return {"format": "maintenance_qa_machine_split", "group_by": "asset_id", "seed": str(seed),
            "dev_fraction": dev_fraction, "inputs": {"facts": input_hash(facts), "qa": input_hash(qa)},
            "partitions": {name: {"assets": sorted(assets),
                                  "qa_ids": sorted(qa_id for asset in assets for qa_id in by_asset[asset])}
                           for name, assets in partitions.items()}}


def select_split(dataset, facts, qa, manifest, subset):
    if manifest.get("format") != "maintenance_qa_machine_split" or manifest.get("group_by") != "asset_id":
        raise ValueError("unsupported split manifest")
    if manifest.get("inputs") != {"facts": input_hash(facts), "qa": input_hash(qa)}:
        raise ValueError("split input hashes do not match the run inputs")
    by_asset = groups(dataset)
    parts = manifest.get("partitions", {})
    if set(parts) != {"dev", "test"} or subset not in parts:
        raise ValueError("split must contain dev and test partitions")
    seen_assets, seen_questions = set(), set()
    for part in parts.values():
        assets, questions = part["assets"], part["qa_ids"]
        if not assets or len(set(assets)) != len(assets) or seen_assets.intersection(assets):
            raise ValueError("split assets must be nonempty, unique and disjoint")
        if not set(assets) <= by_asset.keys():
            raise ValueError("split contains an unknown asset")
        expected = {qa_id for asset in assets for qa_id in by_asset[asset]}
        if set(questions) != expected or len(questions) != len(expected) or seen_questions.intersection(questions):
            raise ValueError("split questions do not match their assets")
        seen_assets.update(assets)
        seen_questions.update(questions)
    if seen_assets != by_asset.keys():
        raise ValueError("split must cover every asset")
    return list(parts[subset]["qa_ids"])
