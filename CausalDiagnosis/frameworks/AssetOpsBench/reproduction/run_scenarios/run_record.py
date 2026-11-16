import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_sets import family_of
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_folder_names import PE_PORT, PORT
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import LAYER_SCENARIO_SHA256, ROOT, SCENARIO_REVISION, sha256_of
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import fetch_public_servers_and_data as layer_b
from CausalDiagnosis.experiments.run_verification.run_ledger.serving_endpoints import endpoint_models  # noqa: E402  the endpoint's /models answer, where SGLang names the served model
from agent.runner import DEFAULT_SERVER_PATHS  # noqa: E402


# The code of the fork a run depends on, hashed into run.json beside the commits of layer B, since the cluster copy of the fork is an upload without .git.
CODE = ("src/agent/plan_execute", "src/agent/_prompts.py", "src/agent/runner.py", "src/llm", "src/observability/persistence.py",
        "src/evaluation/evaluator.py", "src/evaluation/scorers/llm_judge.py", "CausalDiagnosis/frameworks/AssetOpsBench/reproduction", "CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper",
        "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen", "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/agents/react",
        "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/agents/react/prompts", "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/agents/reviewer_agent",
        "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/agents/assessment_agent", "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/tools/jsonreader",
        "CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen/utils", "CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def fork_identity():
    """git describe of the fork (None for an upload without .git) and the SHA-256 prefixes of the code a run depends on."""
    try:
        described = subprocess.run(["git", "describe", "--always", "--tags", "--dirty"], cwd=ROOT, capture_output=True,
                                   text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        described = None
    files = []
    for entry in (*CODE, *(f"src/servers/{name}/main.py" for name in DEFAULT_SERVER_PATHS)):
        path = ROOT / entry
        files += sorted(path.glob("*.py")) if path.is_dir() else [path]
    return {"git_describe": described,
            "code_sha256": {str(f.relative_to(ROOT)): sha256_of(f)[:12] for f in files if f.exists()}}


def port_settings(servers):
    """What run.json records of the paper-era Agent-As-Tool on the servers of a run."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_runner import run_settings

    return run_settings(list(servers), "repro")


def pe_port_settings(servers):
    """What run.json records of the paper-era Plan-Execute on the servers of a run."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import run_settings

    return run_settings(list(servers), "repro")


def run_meta(a, run_dir, set_name, ids, servers, found, shared, api_key):
    digest = sha256_of(a.scenarios)
    pinned = digest == LAYER_SCENARIO_SHA256
    families = {}
    for sid in ids:
        families.setdefault(family_of(sid), []).append(sid)
    return {
        "format": 1, "kind": "aob_repro", "run": Path(run_dir).name, "set": set_name, "scenario_ids": ids, "families": families,
        "repeats": a.repeats, "agents": a.agents, "episodes_planned": a.repeats * len(ids) * len(a.agents),
        "episodes_done": 0, "agent_errors": 0, "model": a.model, "base_url": a.base_url, "temperature": 0.0,
        PORT: port_settings(servers) if PORT in a.agents else None,
        PE_PORT: pe_port_settings(servers) if PE_PORT in a.agents else None,
        "endpoint_models": endpoint_models(a.base_url, api_key),
        "episode_timeout_s": a.episode_timeout,
        "fmsr_model": f"litellm_proxy/{a.model}", "reload_db": not a.no_reload,
        "layer_b": layer_b.provenance(a.layer_b) if a.layer_b else None,
        "scenarios": {"path": str(Path(a.scenarios).resolve()), "sha256": digest, "pinned": pinned,
                      "revision": SCENARIO_REVISION if pinned else None},
        "servers": {name: {"script": script or f"servers.{name}.main", **found[name]} for name, script in servers.items()},
        "server_env": sorted(shared), "fork": fork_identity(),
        "versions": {"python": platform.python_version(), **{p: package_version(p) for p in ("mcp", "openai")}},
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.now().isoformat(timespec="seconds"), "ended": None,
    }
