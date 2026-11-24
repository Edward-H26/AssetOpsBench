"""Local input upload boundaries and version-local runtime identity."""

import ast
import fnmatch
import os
import re
import tempfile
from pathlib import Path

from CausalDiagnosis.utils.paths import GENERATED, ROOT
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.runner import code_files


def upload_policy():
    path = ROOT / "CausalDiagnosis/experiments/run_verification/upload_manifest.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {"SKIPPED_DIRS", "SKIPPED_PATHS", "SKIPPED_NAMES", "LOCAL_FILES", "DATA_FILES"}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in {"skipped", "shipped"}
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)]
    namespace = {"Path": Path, "os": os, "re": re, "fnmatch": fnmatch,
                 "GENERATED_TABLES": set(), "UPLOAD_MANIFEST": "UPLOAD_MANIFEST.txt"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_local_inputs_and_extracted_code_are_not_shipped_as_source():
    policy = upload_policy()
    source = "CausalDiagnosis/debug/general_dataset/versions/v001/industryasseteqa/runner.py"
    data = "CausalDiagnosis/datasets/industryasseteqa/data/"
    pin = data + "CHECKSUMS.txt"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in (source, data + "pdm/facts.jsonl", data + "pdm/upstream/extractor.py", pin):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artificial content", encoding="utf-8")
        assert policy["shipped"](root) == sorted([source, pin])
    assert any(fnmatch.fnmatchcase(data + "pdm/upstream/extractor.py", rule) for rule in GENERATED)


def test_runtime_manifest_names_the_local_loop_without_the_retired_shared_module():
    files = code_files()
    root = "CausalDiagnosis/debug/general_dataset/versions/v001/industryasseteqa/"
    assert root + "tool_loop.py" in files and root + "runner.py" in files
    assert "CausalDiagnosis/agents/utils/tool_call_loop.py" not in files
    assert all("/data/" not in name for name in files)
    assert all(len(digest) == 64 for digest in files.values())
