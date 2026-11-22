import hashlib
import os
import platform
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_configuration import scenario_set
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.utils import frameworks
from . import hashing, serving_endpoints


def data_files(facts_path, asset_id=None, anchored=False):
    """{name: {"path", "sha256"}} of the data a run reads: the recordings, the fitted models of the diagnostics server for the
    asset, the tables that the package of the asset declares under TABLES other than that models file (the label and grade tables
    of the rig), each keyed by its attribute in lower case without _path, and the fetched scenarios for a run whose preset
    anchors its episodes. asset_id None is the engine's default asset, as in a record that holds no such key. A path that the
    asset has none of records sha256 None."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit

    asset_id = asset_id or DEFAULT_ASSET
    tables = {}
    for reference in ports.declared(asset_id, "TABLES", ()):
        key = reference.partition(":")[2].lower().removesuffix("_path")
        path = ports.resolve(reference, f"{asset_id}: TABLES")
        if key == "models" and Path(path) == Path(fit.models_path(asset_id)):
            continue  # the models file of the asset, which the record holds under its own key
        if key in tables or key in ("facts", "models", "scenarios"):
            raise SystemExit(f"{asset_id}: TABLES names two tables under the key {key}, the second {reference}")
        tables[key] = path
    paths = {"facts": facts_path, "models": fit.models_path(asset_id), **dict(sorted(tables.items()))}
    if anchored:
        scenarios = scenario_set(asset_id)
        paths["scenarios"] = None if scenarios is None else scenarios.scenarios_path()
    return {name: {"path": None if path is None else str(path), "sha256": None if path is None else hashing.sha256_of(path)} for name, path in paths.items()}


def installed_sha256():
    """The SHA-256 over the sorted lines name==version of every distribution of the running interpreter, equal for equal
    environments and needing no tool on the node."""
    from importlib import metadata

    lines = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]})
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def platform_of(root=ROOT):
    """run.json platform: the Python version, the hash of the installed distributions and of the lock, and the hash of the
    CouchDB image that the job exports after it hashes the image, None outside such a job."""
    return {"python": platform.python_version(), "installed_sha256": installed_sha256(), "lock_sha256": hashing.sha256_of(Path(root) / "uv.lock"),
            **{name.lower(): os.environ.get(name) for _framework, name in frameworks.names("JOB_ENVIRONMENT")}}


def data_record(facts_path, asset_id=None):
    """run.json data: {name: SHA-256} of the data files of the run of the asset and of the generated tables as they are at the
    start, the fitted models without the date of their build."""
    files = data_files(facts_path, asset_id)
    models = files["models"]["path"]
    return {**{name: entry["sha256"] for name, entry in files.items()}, "models": None if models is None else hashing.models_sha256(models)}


def platform_command(facts_path=None, model=None, judge_model=None, asset_id=None):
    """The platform record of the node it runs on, an optional record of a node: what a run of the asset records there as
    platform and data, the data on the data file of the asset unless facts_path names another, and the serving identity of the
    agent and the judge endpoints (platform_serving), each with the GPU card of its serving job and under the served name of its
    setting or of its endpoint file unless model or judge_model names another. asset_id None is the engine's default asset."""
    from CausalDiagnosis.datasets.dataset_loader import recordings as records
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET

    asset_id = asset_id or DEFAULT_ASSET
    facts = facts_path or records.default_facts_path(get_profile(asset_id))
    return {"asset_id": asset_id, "platform": platform_of(), "data": data_record(facts, asset_id),
            "serving": {"agent": serving_endpoints.platform_serving("agent", model), "judge": serving_endpoints.platform_serving("judge", judge_model)}}
