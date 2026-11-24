"""Audit private release records without printing cases or reference answers."""

import hashlib
import json
from pathlib import Path

from .download import REPOSITORY, REVISION, REQUIRED, relative_path


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON value")


def read_json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)


def _records(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = read_json(line)
        key = record.get("case_id")
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError("case identifiers must be nonempty and unique")
        result[key] = record
    if not result:
        raise ValueError("case file must be nonempty")
    return result


def audit_release(folder):
    folder = Path(folder)
    manifest = read_json((folder / "source.json").read_text(encoding="utf-8"))
    if (manifest.get("repository"), manifest.get("revision"), manifest.get("status")) != (REPOSITORY, REVISION, "downloaded"):
        raise ValueError("release manifest does not identify a completed pinned download")
    if not set(REQUIRED) <= manifest["files"].keys():
        raise ValueError("release manifest lacks required files")
    for name, identity in manifest["files"].items():
        content = folder.joinpath(*relative_path(name).parts).read_bytes()
        if len(content) != identity["bytes"] or hashlib.sha256(content).hexdigest() != identity["sha256"]:
            raise ValueError(f"release bytes differ from manifest: {name}")
    all_cases, train, test = [_records(folder / "data" / name) for name in ("all.jsonl", "train.jsonl", "test.jsonl")]
    if train.keys() & test.keys() or train.keys() | test.keys() != all_cases.keys():
        raise ValueError("official train and test must be disjoint and cover all cases")
    if any(all_cases[key] != record for split in (train, test) for key, record in split.items()):
        raise ValueError("split records differ from all.jsonl")
    tools = read_json((folder / "tools/tools_schema.json").read_text(encoding="utf-8"))["tools"]
    names = [tool["name"] for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tool definitions")
    problems, images, missing, multimodal = [], set(), set(), 0
    for key, record in all_cases.items():
        case = record["case"]
        if case.get("case_id") != key or case["environment_setup"].get("initial_query") != record["initial_query"]:
            problems.append({"case_id": key, "problem": "embedded case identity or query differs"})
        actions = case["hidden_environment"]
        action_ids = [action["action_id"] for action in actions]
        if len(set(action_ids)) != len(action_ids):
            problems.append({"case_id": key, "problem": "duplicate action identifiers"})
        if any(action["match_tool"] not in names for action in actions):
            problems.append({"case_id": key, "problem": "action names an unknown tool"})
        dag = case["golden_dag"]
        needed = set(dag["mandatory_nodes"]) | set(dag["dependencies"])
        needed.update(parent for parents in dag["dependencies"].values() for parent in parents)
        if not needed <= set(action_ids):
            problems.append({"case_id": key, "problem": "DAG references an absent action"})
        dependencies = {node: set(dag["dependencies"].get(node, ())) for node in needed}
        while dependencies:
            ready = {node for node, parents in dependencies.items() if not parents}
            if not ready:
                problems.append({"case_id": key, "problem": "cyclic DAG"})
                break
            dependencies = {node: parents - ready for node, parents in dependencies.items() if node not in ready}
        multimodal += bool(record.get("is_multimodal"))
        paths = set(record.get("image_paths", ()))
        paths.update(action.get("observation", {}).get("image_path") for action in actions)
        for name in paths - {"", None}:
            path = relative_path(name)
            if not path.parts or path.parts[0] != "images":
                raise ValueError("image reference must stay under images/")
            images.add(name)
            if not folder.joinpath(*path.parts).is_file():
                missing.add(name)
    return {"repository": REPOSITORY, "revision": REVISION, "cases": len(all_cases), "train": len(train),
            "test": len(test), "tools": len(tools), "multimodal_cases": multimodal,
            "image_references": len(images), "missing_images": len(missing), "problems": problems,
            "integration_status": "release_audit_only"}
