"""The import rules of the layers (CausalDiagnosis/docs/adding-a-dataset.md), over every Python file of src, CausalDiagnosis and benchmarks
outside the tests folders, so that the package of a port is read the day it lands.

Rule 1: no module outside CausalDiagnosis/datasets imports the package of an asset, but the modules of the table BOUND, each bound to one asset
for its reason. Rule 2: a module of the asset layer, a module of CausalDiagnosis/datasets or of an asset package, imports
the engine, a server, an agent or a driver only inside a function, and the __init__.py of an asset package imports from
CausalDiagnosis.datasets alone, since the engine imports the package while it loads the profile. Rule 3: a module of CausalDiagnosis/environment or of
CausalDiagnosis/datasets outside the asset packages imports a server, an agent or a driver only in the statements of the table LAYER_IMPORTS, each with its
reason. The package of one asset is no general path, so the label of a port may call a server of its own inside a
function, as rule 2 allows."""
import ast
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports

from CausalDiagnosis.datasets.tests.conftest import ROOT

ASSETS = "CausalDiagnosis/datasets"
SCANNED = ("src", "CausalDiagnosis", "benchmarks")
DRIVER_SIDE = tuple(f"CausalDiagnosis.{name}" for name in ("mcp_servers", "agents", "experiments", "evaluation", "training",
                                                          "episode_viewer", "examples"))
ENGINE_SIDE = ("CausalDiagnosis.environment", "servers", "agent", *DRIVER_SIDE)  # what the asset layer imports inside a function alone
OTHER_LAYERS = ("servers", "agent", *DRIVER_SIDE)  # what the environment and the datasets import in the rows of LAYER_IMPORTS alone
# a module outside the asset layer that imports the package of one asset: (asset, reason)
BOUND = {
    "CausalDiagnosis/environment/reference_policies/tier_classifiers.py": ("hyd_1", "the reference classifiers of the rig, behind the label of its starts"),
    "CausalDiagnosis/frameworks/AssetOpsBench/scenarios/scenario_anchors.py": ("hyd_1", "the anchored starts of the rig, from its scenario records"),
    "CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/hyd_1_sample_documents.py": ("hyd_1", "the sample documents of the rig that the latency probe and the tests of the tool servers read"),
}
# (file, imported module, the function that must hold the statement or None): reason
LAYER_IMPORTS = {
    ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py", "CausalDiagnosis.utils.public_rules.visits", None):
        "the visit rule of the work orders, which the engine, the agent and the diagnostics server read from one module",
    ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_4_repair_rules.py", "CausalDiagnosis.utils.public_rules.visits", None):
        "the schedule of a visit, which the engine, the agent and the diagnostics server read from one module",
    ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_costs.py", "CausalDiagnosis.utils.public_rules.visits", None):
        "the rule of a no_call step, which the verifier, the environment and the drivers read from one module",
    ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/token_clock.py", "CausalDiagnosis.utils.atomic_writes_and_locks", None):
        "the ledger of generated tokens is written and read under the lock that the servers use for their files",
    ("CausalDiagnosis/environment/reference_policies/diagnosability_label.py", "CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP", "start_grade"):
        "the label of the rig nests the E18 grade of the CausalDiagnosis server, read inside start_grade alone",
}


def _module(rel):
    parts = Path(rel).with_suffix("").parts
    return ".".join(parts[1:] if parts[0] == "src" else parts)


def _targets(node, rel):
    """The modules an import statement names, a relative import resolved against the package of its file: each name of
    an import, and for a from-import the module with each name after it, which may be a module or one of its attributes."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:
        package = _module(rel).split(".")[:-node.level]
        base = ".".join(package + ([node.module] if node.module else []))
    return [f"{base}.{alias.name}" if base else alias.name for alias in node.names]


def _under(module, prefixes):
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _imports(tree):
    """(import node, the names of the functions that hold it) of every import statement of a module."""
    found = []

    def visit(node, functions):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append((child, functions))
            inner = functions + (child.name,) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else functions
            visit(child, inner)

    visit(tree, ())
    return found


def _listed(rel, targets, functions):
    return any(rel == file and all(_under(t, (module,)) for t in targets) and (function is None or function in functions)
               for file, module, function in LAYER_IMPORTS)


def violations(root):
    """Every import statement of the tree at root that breaks one of the three rules, as "file:line rule N: modules"."""
    root = Path(root)
    asset_ids = sorted(path.parent.name for path in (root / ASSETS).glob(f"*/{ports.ASSET_FILE}"))
    packages = [f"{ports.PACKAGE}.{asset_id}" for asset_id in asset_ids]
    found = []
    for top in SCANNED:
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if "/tests/" in rel or rel.startswith("tests/"):
                continue
            parts = Path(rel).parts
            in_layer = rel.startswith(ASSETS + "/")
            in_package = in_layer and len(parts) > 3 and parts[2] in asset_ids
            for node, functions in _imports(ast.parse(path.read_text(encoding="utf-8"))):
                targets = _targets(node, rel)
                where = f"{rel}:{node.lineno}"
                for package, asset_id in zip(packages, asset_ids):
                    hits = [t for t in targets if _under(t, (package,))]
                    if hits and not in_layer and BOUND.get(rel, (None,))[0] != asset_id:
                        found.append(f"{where} rule 1: {hits}")
                if in_layer and not functions:
                    engine = [t for t in targets if _under(t, ENGINE_SIDE)]
                    if engine:
                        found.append(f"{where} rule 2: {engine} at module level")
                    if in_package and parts[3] == "__init__.py" and not all(_under(t, (ports.PACKAGE,)) for t in targets):
                        found.append(f"{where} rule 2: {targets} in the package of an asset")
                other = [t for t in targets if _under(t, OTHER_LAYERS)]
                if rel.startswith(("CausalDiagnosis/environment/", ASSETS + "/")) and not in_package and other and not _listed(rel, other, functions):
                    found.append(f"{where} rule 3: {other}")
    return found


def test_the_tree_keeps_the_three_import_rules():
    assert sorted(path.parent.name for path in (ROOT / ASSETS).glob(f"*/{ports.ASSET_FILE}")) == ports.asset_ids()
    assert violations(ROOT) == []


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


ASSET = {"CausalDiagnosis/datasets/made_up_1/asset.json": "{}\n", "CausalDiagnosis/datasets/made_up_1/__init__.py": "from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds\n"}
PLANTED = {
    "rule 1": ({"CausalDiagnosis/environment/Causal_Diagnosis_Environment/engine.py": "import json\nfrom CausalDiagnosis.datasets.made_up_1 import source\n",
                "CausalDiagnosis/datasets/made_up_1/source.py": "from CausalDiagnosis.datasets.made_up_1 import labels\n"},
               "CausalDiagnosis/environment/Causal_Diagnosis_Environment/engine.py:2 rule 1: ['CausalDiagnosis.datasets.made_up_1.source']"),
    "rule 2": ({"CausalDiagnosis/datasets/helper.py": "def late():\n    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import env\n\n\nfrom CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import episode_clock as clock\n",
                "CausalDiagnosis/datasets/made_up_1/labels.py": ("def label_of(env, rules=None):\n    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier\n"
                                                        "    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade\n")},
               "CausalDiagnosis/datasets/helper.py:5 rule 2: ['CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.episode_clock'] at module level"),
    "rule 3": ({"CausalDiagnosis/environment/Causal_Diagnosis_Environment/thing.py": "\nfrom CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade\n",
                "CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py": "from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract\n",
                "CausalDiagnosis/environment/reference_policies/diagnosability_label.py": "def start_grade(env):\n    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade import separable\n"},
               "CausalDiagnosis/environment/Causal_Diagnosis_Environment/thing.py:2 rule 3: ['CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade']"),
}


@pytest.mark.parametrize("rule", sorted(PLANTED))
def test_each_import_rule_reports_a_planted_violation(rule, tmp_path):
    files, expected = PLANTED[rule]
    found = violations(_write(tmp_path, {**ASSET, **files}))
    assert found == [expected]
