"""Shared pieces of the hydraulic rig drivers (CausalDiagnosis/docs/experiments.md): the fork's import path, the servers
with their tool sets, the configuration presets and price tables, the per episode launcher files, the run folder files
and the columns of scores.csv.

The agent runners spawn a server from its launcher path, as ``python -m <dotted path>`` from the repository root when the
path lies inside the repository and as ``python <launcher path>`` otherwise, with almost no environment variables, so a
launcher cannot rely on PYTHONPATH or on an installed package. Each episode folder therefore gets one small launcher file
per server that puts the fork's ``src`` on the import path itself and binds the server process to the folder's state
file, the same binding that ``CausalDiagnosis/mcp_servers/Environment_Actions_MCP/launcher.py`` and ``CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/launcher.py`` use.
"""
from . import servers_of_a_run
import json
from pathlib import Path
from CausalDiagnosis.utils.paths import ROOT
from CausalDiagnosis.experiments.run_verification.public_record import public_value  # noqa: E402  the form of a public record
from . import run_folder_layout


STATE_FILES = {servers_of_a_run.RIG_SERVER: run_folder_layout.STATE_FILE, servers_of_a_run.CAUSAL_SERVER: run_folder_layout.RELEASED_FILE, **{server: surface["state_file"] for surface in servers_of_a_run.SURFACES.values() for server in surface["servers"]}}
LAUNCHER_MODULES = {servers_of_a_run.RIG_SERVER: "CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main", servers_of_a_run.CAUSAL_SERVER: "CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main",
                    **{server: surface["launch_module"] for surface in servers_of_a_run.SURFACES.values() for server in surface["servers"]}}
_LAUNCHER = '''"""Per-episode entry point of the {server} MCP server, written by the benchmark driver beside the episode's files. The
paths of the machine and the credentials of the store stand in the private file beside it, which no public file holds."""
import time
_STARTED = (time.time(), time.monotonic(), time.thread_time())  # the start s of the process for the thinking clock
import json  # noqa: E402
import sys  # noqa: E402
if not sys.flags.isolated:
    raise SystemExit("the {server} server runs only isolated, as python -I <launcher>, so that no environment variable or user site reaches it")
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True  # a server process writes nothing beside the episode files but its own state
PRIVATE = json.loads(Path(__file__).resolve().with_name({private!r}).read_text(encoding="utf-8"))
sys.path[:0] = PRIVATE["src"]
{assets}
from {module} import main  # noqa: E402

main(Path(__file__).resolve().with_name({state!r}){extra}, started=_STARTED)
'''
# the private file of the launchers of a folder: the source folder, the asset folders, the variables and the options that name a
# path of this machine or a credential (public_record), read by every launcher of the folder, never published (ledger.IGNORED_NAMES)
LAUNCH_PRIVATE = "launch_private.json"
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import DEFAULT_RANKING_MODEL  # noqa: E402  the driver and the server share one default


DEFAULT_ABSTENTION_RULE = "decision"


def asset_search_path():
    """The folders of the asset packages that this process searches beyond the folder of the package CausalDiagnosis.datasets itself, in search
    order, such as the folder of a port installed or mounted beside the tree."""
    import CausalDiagnosis.datasets

    own = Path(CausalDiagnosis.datasets.__file__).resolve().parent
    return [str(Path(folder)) for folder in CausalDiagnosis.datasets.__path__ if Path(folder).resolve() != own]


def launch_options(server, ranking_model=DEFAULT_RANKING_MODEL, abstention=None, store=None, left_out=None, models_path=None):
    """The keyword arguments that the launcher of a server passes to the main function of its module, in the order the launcher
    names them, the one source of the launcher text and of a server that a replay builds in its own process. The diagnostics
    server is always told the ranking model, so that the model the run records is the model the server runs whatever the
    server's own default is, the models file of the asset of the run when one is given, left_out, the combination the episode
    started in, whose block every fit of the server leaves out and which it uses for nothing else, and the abstention rule with
    its cut (abstention = (rule, cut)) only when they differ from the default rule. A metered server is told its name and the
    store of the run, since its process gets no environment of its own."""
    options = {}
    if server == servers_of_a_run.CAUSAL_SERVER:
        options["ranking_model"] = ranking_model
        if models_path is not None:
            options["models_path"] = str(models_path)
        options["left_out"] = left_out
        if abstention and abstention[0] != DEFAULT_ABSTENTION_RULE:
            options.update(abstention_rule=abstention[0], abstention_cut=float(abstention[1]))
    if server in servers_of_a_run.METERED_SERVERS:
        options.update(servers_of_a_run.tool_surface(servers_of_a_run.surface_of(server)).launch_options(server, store))
    return options


def write_launchers(ep_dir, servers=(servers_of_a_run.RIG_SERVER, servers_of_a_run.CAUSAL_SERVER), ranking_model=DEFAULT_RANKING_MODEL, abstention=None, store=None,
                    left_out=None, models_path=None, data_file=None):
    """The launcher files of an episode folder, as {server name: absolute path} for a runner's server_paths, each calling the
    main function of its server's module with the options of launch_options. Every launcher puts the folders of the asset
    packages that this process searches beyond the folder of
    the package itself (asset_search_path) on the search path of the server process, so that a port mounted beside the tree is
    found there too; without such folders a launcher holds no line for them. data_file, (variable, path), sets in every server
    process the variable that the asset file names for its data file to the data file of the run, so that code of the evidence
    kind that looks the data file up reads the file that the driver reads."""
    paths = {}
    folders = asset_search_path()
    private = {"src": [str(ROOT), str(servers_of_a_run.SRC)], "asset_folders": folders, "environment": {data_file[0]: str(data_file[1])} if data_file is not None else {},
               "options": {}}
    assets = ("import CausalDiagnosis.datasets  # noqa: E402\n"
              "CausalDiagnosis.datasets.__path__.extend(PRIVATE[\"asset_folders\"])  # the folders of the asset packages beside the package's own\n"
              ) if folders else ""
    if data_file is not None:
        assets += "import os  # noqa: E402\nos.environ.update(PRIVATE[\"environment\"])  # the data file of the run\n"
    for server in servers:
        options = launch_options(server, ranking_model=ranking_model, abstention=abstention, store=store, left_out=left_out,
                                 models_path=models_path)
        # an option whose public form differs from it (a path, a store with a URL or a password) is read from the private file
        hidden = {key: value for key, value in options.items() if public_value({key: value}) != {key: value} or key == "models_path"}
        extra = "".join(f", {key}={value!r}" if key not in hidden else f", {key}=PRIVATE[\"options\"][{server!r}][{key!r}]"
                        for key, value in options.items())
        if hidden:
            private["options"][server] = hidden
        path = Path(ep_dir) / f"{server}_server.py"
        path.write_text(_LAUNCHER.format(server=server, module=LAUNCHER_MODULES[server], private=LAUNCH_PRIVATE, assets=assets,
                                         state=STATE_FILES[server], extra=extra),
                        encoding="utf-8")
        path.chmod(0o600)
        paths[server] = path.resolve()
    target = Path(ep_dir) / LAUNCH_PRIVATE
    target.write_text(json.dumps(private, indent=1, default=str) + "\n", encoding="utf-8")
    target.chmod(0o600)  # it holds the random CouchDB password of the job
    return paths
