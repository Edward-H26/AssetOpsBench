import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import ROOT, SRC
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import fetch_public_servers_and_data as layer_b
from agent.plan_execute.executor import _list_tools  # noqa: E402
from agent.runner import DEFAULT_SERVER_PATHS  # noqa: E402


# The settings the released servers read (INSTRUCTIONS.md, Environment Variables), handed to every launcher when set here.
SERVER_ENV = ("COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD", "IOT_DBNAME", "ASSET_DBNAME", "WO_DBNAME",
              "FAILURE_CODE_DBNAME", "FAILURE_MODE_DBNAME", "VIBRATION_DBNAME", "CATALOG_DBNAME", "MODEL_CATALOG_DBNAME",
              "FEATURE_CATALOG_DBNAME", "TSFM_STORE", "TSFM_WORKDIR", "LOG_LEVEL")
_LAUNCHER = '''"""The {name} MCP server of one AssetOpsBench reproduction run, written by CausalDiagnosis/frameworks/AssetOpsBench/reproduction/run_scenarios.py. A stdio
server inherits almost no environment from its client, so the settings it reads are set here before it is imported."""
import os
import sys

sys.dont_write_bytecode = True
os.environ.update({env!r})
sys.path[:0] = {src!r}
{start}
'''
RELEASED_START = "from servers.{name}.main import main\n\nmain()"
SCRIPT_START = "import runpy\n\nrunpy.run_path({script!r}, run_name=\"__main__\")"


def parse_servers(specs):
    """{name: script or None}: the six servers of layer B, or exactly the servers that --server names."""
    if not specs:
        return {name: None for name in DEFAULT_SERVER_PATHS}
    servers = {}
    for spec in specs:
        name, sep, script = spec.partition("=")
        if not (sep and name and script):
            raise SystemExit(f"--server {spec!r}: give NAME=SCRIPT")
        servers[name] = str(Path(script).resolve())
    return servers


def server_env(model, base_url, api_key):
    """(settings of every server, settings of the FMSR server): the CouchDB and TSFM settings of this process, and for the
    FMSR server's generate tools the agent's model at the agent's endpoint through LiteLLM, as the FMSR server of main-0.x
    reads them."""
    shared = {k: os.environ[k] for k in SERVER_ENV if k in os.environ}
    fmsr = {"FMSR_MODEL_ID": f"litellm_proxy/{model}", "LITELLM_BASE_URL": base_url, "LITELLM_API_KEY": api_key}
    return shared, fmsr


def write_launchers(folder, servers, shared, fmsr, src=SRC):
    """{name: launcher path}. A server of the tree under src starts from its entry point, whose relative imports need that
    tree first on the path, and a --server script runs as __main__."""
    paths = {}
    for name, script in servers.items():
        env = {**shared, **(fmsr if name == "fmsr" else {})}
        start = SCRIPT_START.format(script=script) if script else RELEASED_START.format(name=name)
        path = Path(folder) / f"{name}_server.py"
        path.write_text(_LAUNCHER.format(name=name, env=env, src=[str(ROOT), str(src)], start=start), encoding="utf-8")
        paths[name] = path
    return paths


def check_servers(launchers):
    """Lists the tools of every server once before the first episode, so that a server that cannot start (a missing package,
    an unset setting) stops the run here instead of appearing as an unavailable server inside every plan."""
    found = {}
    for name, path in launchers.items():
        try:
            tools = asyncio.run(_list_tools(path))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"server {name} did not start from {path}: {type(exc).__name__}: {exc}")
        if not tools:
            raise SystemExit(f"server {name} lists no tool")
        found[name] = {"tools": len(tools), "listing_chars": len(json.dumps(tools))}
    return found


def reload_couchdb(layer):
    """Drops the databases of layer B and loads them again with the loaders of main-0.x, the reset the fork's scenario suite
    runner makes before each scenario. They run in the layer's folder, so that a .env of the fork does not reach them.
    Returns its seconds."""
    started = time.perf_counter()
    for command in layer_b.load_commands(layer):
        done = subprocess.run(command, cwd=layer, capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"CouchDB reload failed with exit {done.returncode}: {done.stderr[-2000:]}")
    return time.perf_counter() - started
