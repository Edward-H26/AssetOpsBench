"""Build descriptive QA with the upstream extractor and question builder."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from CausalDiagnosis.utils.paths import ROOT

from .data import QADataset
from .download import FILES


def build_descriptive(raw, destination):
    """Run upstream code from the locally available publication branch.

    No source files of the current checkout are replaced. Extracted upstream code
    and its outputs stay in the selected data directory with their code identity.
    """
    raw, destination = Path(raw).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("build destination must be new")
    missing = [name for name in ("pandas", "numpy", "joblib") if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(f"upstream build dependencies are missing from this Python environment: {', '.join(missing)}")
    manifest = json.loads((raw / "source.json").read_text(encoding="utf-8"))
    for name in FILES:
        digest = hashlib.sha256((raw / name).read_bytes()).hexdigest()
        if digest != manifest["files"][name]["sha256"]:
            raise ValueError(f"{name}: raw file disagrees with source.json")
    root = ROOT
    revision = subprocess.run(["git", "rev-parse", "origin/IndustryAssetEQA"], cwd=root,
                              check=True, capture_output=True, text=True).stdout.strip()
    scripts = {}
    for name in ("ts_fact_extractor.py", "qa_builder_ts.py"):
        scripts[name] = subprocess.run(["git", "show", f"{revision}:IndustryAssetEQA/src/utils/{name}"],
                                       cwd=root, check=True, capture_output=True).stdout
    destination.mkdir(parents=True)
    tools = destination / "upstream"
    tools.mkdir()
    for name, content in scripts.items():
        (tools / name).write_bytes(content)
    facts, questions = destination / "facts.jsonl", destination / "qa_descriptive.jsonl"
    source = {"raw": manifest, "upstream_revision": revision,
              "scripts": {name: hashlib.sha256(content).hexdigest() for name, content in scripts.items()},
              "task_type": "descriptive", "status": "building"}
    record = destination / "build.json"
    record.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    extract = [sys.executable, str(tools / "ts_fact_extractor.py")]
    for option, name in (("telemetry", "PdM_telemetry.csv"), ("failures", "PdM_failures.csv"),
                         ("errors", "PdM_errors.csv"), ("maint", "PdM_maint.csv"), ("machines", "PdM_machines.csv")):
        extract.extend((f"--{option}", str(raw / name)))
    subprocess.run([*extract, "--out", str(facts)], cwd=destination, check=True)
    subprocess.run([sys.executable, str(tools / "qa_builder_ts.py"), "--facts", str(facts),
                    "--out", str(questions), "--mode", "descriptive"], cwd=destination, check=True)
    QADataset.load(facts, questions)
    source["status"] = "built"
    source["outputs"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (facts, questions)}
    record.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return destination
