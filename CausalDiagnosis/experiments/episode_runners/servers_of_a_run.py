import sys
from pathlib import Path
from CausalDiagnosis.utils.paths import ROOT


SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from CausalDiagnosis.utils import frameworks  # noqa: E402


RIG_SERVER, CAUSAL_SERVER = "hydraulic_rig", "causal_diagnosis"
# the tool surfaces the frameworks declare, read as data (utils/frameworks.py): their servers, state files and launch modules
SURFACES = frameworks.surfaces()
IOT_SERVER, FMSR_SERVER, WO_SERVER, TASK_SERVER = SURFACES["aob"]["servers"]
METERED_SERVERS = tuple(server for surface in SURFACES.values() for server in surface["servers"])  # the servers behind a metering layer
RIG_TOOLS = frozenset({"get_asset_profile", "read_sensors", "get_current_time", "wait_for_notification", "summarize_changes_since", "get_failure_modes",
                       "get_fix_history", "lookup_fix", "submit_diagnosis", "declare_insufficient_evidence", "create_work_order", "finish"})
CAUSAL_TOOLS = frozenset({"get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome"})
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import RANKING_MODELS  # noqa: E402  the driver and the server share one default


def contract_tools(server, row=None, config=None):
    """The tool names a server lists: the rig's contract for the rig's two servers (a fixed list while the contract
    module does not exist), and the contract of the AssetOpsBench rows for the metered servers, those a configuration's
    big layers list when config is given, else those of the given row or of every row."""
    if server in METERED_SERVERS:
        return tool_surface(surface_of(server)).contract_tools(server, row, config)
    fallback = {RIG_SERVER: RIG_TOOLS, CAUSAL_SERVER: CAUSAL_TOOLS}[server]
    try:
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
    except ImportError:
        return fallback
    return frozenset(contract.tools_of(server)) or fallback


def registered_servers(server_set, diagnostics):
    """The servers of a run: the rig's action server, or AssetOpsBench's servers with the task server behind the
    metering layer, and the diagnostics server when the configuration lists it."""
    servers = (RIG_SERVER,) if server_set == "rig" else SURFACES[server_set]["servers"] if server_set in SURFACES else METERED_SERVERS
    return (*servers, CAUSAL_SERVER) if diagnostics else servers


def surface_of(server):
    """The declared tool surface whose servers hold a server name."""
    return next(name for name, surface in SURFACES.items() if server in surface["servers"])


def tool_surface(name):
    """The object of a declared tool surface, its module imported on first use (the entry of its declaration)."""
    return frameworks.code("SURFACES", name)
