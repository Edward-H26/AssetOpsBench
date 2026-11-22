"""The mapping from the agent's action kinds to the concrete tools of the layer it runs on.

The agent plans in action kinds, a read of a sensor group over cycles, an order on a component, a diagnosis, a declaration,
a wait and a finish, plus the controller's own kinds, the free reads of the opening, the change summary and the three
diagnostics calls. A surface binds every kind to a server, a tool name and the arguments the tool takes, and reads the
tool's answer back into the one envelope the controller consumes, so that the rows of the paper that run AssetOpsBench's
own servers behind the metering layer serve the same controller as the hydraulic rig's action server does. The time
tools follow the System app of ARE, get_current_time and wait_for_notification(timeout), the latter the only way the agent lets
the clock move.

The rig surface is the hydraulic rig's action server, whose envelope carries the clock, the budget and the step counter
with every answer. The AssetOpsBench surface is the iot, fmsr and wo servers of AssetOpsBench plus the task server of
the metering layer (CausalDiagnosis/frameworks/AssetOpsBench/tool_servers). There a read is history over a window of
timestamps, which the surface computes from one anchor in cycles, the current time on the clock rows and the newest
recorded row elsewhere, so that a read asks for exactly the cycles the plan wants and the price per returned cycle is
the one the plan expected; an order is generate_work_order on the child asset of the component; the
receipt line the layer appends from the priced row on is the envelope; and on the base row, which appends none, the
surface counts the calls the step cap counts. The change summary of the rig has no tool there, so the surface answers
it from the notifications the receipts carried.
"""
from dataclasses import dataclass

from CausalDiagnosis.agents.utils.tool_answers import json_object

RIG_SERVER, CAUSAL_SERVER = "hydraulic_rig", "causal_diagnosis"  # the server names the drivers register (CausalDiagnosis/experiments/episode_runners/episode_launchers.py, CausalDiagnosis/experiments/episode_runners/run_configuration.py, CausalDiagnosis/experiments/episode_runners/run_folder_layout.py, CausalDiagnosis/experiments/episode_runners/servers_of_a_run.py, CausalDiagnosis/utils/csv_rows.py and CausalDiagnosis/utils/json_files.py)
DIAGNOSTICS_KINDS = ("rank", "estimate", "verify")


@dataclass(frozen=True)
class Binding:
    server: str
    tool: str
    arguments: object  # keyword fields of the kind -> the tool's arguments


def _write(levels, evidence):
    return {**levels, "evidence_ids": ",".join(evidence)}


HYDRAULIC_RIG = {
    "profile": Binding(RIG_SERVER, "get_asset_profile", lambda: {}),
    "failure_modes": Binding(RIG_SERVER, "get_failure_modes", lambda component="": {"component": component}),
    "fix_history": Binding(RIG_SERVER, "get_fix_history", lambda: {}),
    "read": Binding(RIG_SERVER, "read_sensors", lambda sensor_group, cycles: {"sensor_group": sensor_group, "cycles": cycles}),
    "changes": Binding(RIG_SERVER, "summarize_changes_since", lambda since: {"cycle": since}),
    "diagnose": Binding(RIG_SERVER, "submit_diagnosis", _write),
    "declare": Binding(RIG_SERVER, "declare_insufficient_evidence",
                       lambda sensor_group, cycles, evidence: {"sensor_group": sensor_group, "cycles": cycles, "evidence_ids": ",".join(evidence)}),
    "order": Binding(RIG_SERVER, "create_work_order",
                     lambda component, prediction_id, evidence: {"component": component, "prediction_id": prediction_id, "evidence_ids": ",".join(evidence)}),
    "wait": Binding(RIG_SERVER, "wait_for_notification", lambda timeout: {"timeout": timeout}),
    "finish": Binding(RIG_SERVER, "finish", lambda summary: {"summary": summary}),
    # the diagnostics tools answer at the level raw, the whole answer that the board reads field by field
    "rank": Binding(CAUSAL_SERVER, "rank_root_causes",
                    lambda top_k=None, branch_detail=None: {**{k: v for k, v in (("top_k", top_k), ("branch_detail", branch_detail)) if v is not None},
                                                            "response_format": "raw"}),
    "estimate": Binding(CAUSAL_SERVER, "estimate_intervention_effect", lambda action: {"action": action, "response_format": "raw"}),
    "verify": Binding(CAUSAL_SERVER, "verify_outcome", lambda prediction_id: {"prediction_id": prediction_id, "response_format": "raw"}),
}


DECLARE_TOOL, SUMMARY_TOOL = "declare_insufficient_evidence", "summarize_changes_since"  # the change summary is served on the level with the CausalDiagnosis MCP only, the declaration on both levels
def not_json(tool, text):
    """The reason for an answer that is not a JSON object, with its text, where a server writes its error as text."""
    shown = (text or "").strip()
    return f"the answer of {tool} is not a JSON object" + (f": {shown}" if shown else "")


def bind(tool_map, kind, **fields):
    """(server, tool, arguments) of an action kind under a tool map."""
    binding = tool_map[kind]
    return binding.server, binding.tool, binding.arguments(**fields)


class Surface:
    """What every surface gives the texts of the agent: the noun of the asset, a channel tag to give as an example and the
    statistics a reading serves, which the driver takes from the asset file (its short noun, its example channel and its served
    statistics, the plain one first); a surface built without them says asset, gives no example and names the values value."""

    def __init__(self, noun="asset", example_tag=None, statistics=("value",)):
        self.noun, self.example_tag, self.statistics = noun, example_tag, tuple(statistics)

    @property
    def statistic(self):
        """The plain statistic, listed ahead of the others, which the texts name in the plural with an s appended and which a
        record of the AssetOpsBench rows stores under the bare field of a channel."""
        return self.statistics[0]

    served = None  # the tools the opening lists for the run, set by listed; None serves every tool, as the rig's action server does

    def listed(self, tools, where=None):
        """The served list of the row's level, from the opening listing of every server, {server: [tools]} (EpisodeSessions.tools),
        which is no call, is not metered and takes no step."""
        if tools is None:  # no listing given, as the in-process stand-ins of the tests: every tool stays served, as before the levels
            return
        served = {tool.name for server_tools in tools.values() for tool in server_tools}
        if not served:  # an opening listing that names no tool is a broken episode, never a row that serves nothing
            raise ValueError(f"the opening listing of {where or 'the episode'} names no tool of the servers {sorted(tools)}")
        self.served = served

    def serves(self, tool):
        """Whether the row serves a tool: every tool before the listing, as on the rig's action server, else the listed ones."""
        return self.served is None or tool in self.served

    def unserved_kinds(self):
        """The step kinds of the plan whose tool the row does not serve, which leave the prompts and which the plan check refuses."""
        return frozenset(kind for kind, tool in OPTIONAL_KIND_TOOLS.items() if not self.serves(tool))

    def summary_fields(self):
        """The words of a read for the fields of the statistics served after the plain one, each a sensor with the statistic
        appended; none where the asset serves the plain statistic alone."""
        extra = [f"_{s}" for s in self.statistics[1:]]
        return f" and their {' and '.join(extra)} fields" if extra else ""


class RigSurface(Surface):
    """The hydraulic rig's action server: every kind bound by HYDRAULIC_RIG and the envelope read as the server returns it."""

    action_servers = (RIG_SERVER,)
    evidence_form = "obs/<n>/<field>"
    reads_take_time = True  # the action server plays the cycles a read asks for, each for its declared duration
    prompt_tools = ""  # the system prompt names the rig's tools already
    prompt_changes = ", reads the change summary"
    summary_is_call = True  # the change summary after a wait is summarize_changes_since, a call the step cap counts

    def bind(self, kind, board, **fields):
        return bind(HYDRAULIC_RIG, kind, **fields)

    def parse(self, kind, tool, text, board, **fields):
        """The envelope of an answered call; an answer that is not a JSON object reads as a refusal."""
        data = json_object(text)
        if data is None:
            data = {"error": not_json(tool, text)}
        if "error" in data and "refused" not in data and "result" not in data:
            return {"refused": data["error"]}
        return data

    def failure_mode_requests(self, board):
        return [{"component": c} for c in board.profile["components"]]  # by name, as the AssetOpsBench surface asks fmsr

    def opening_kinds(self, board):
        return []

    def can_observe_after(self, board, cycle):
        return True  # a wait moves the clock, so a reading after a close can always be bought

    def local(self, kind, board, **fields):
        raise KeyError(f"{kind} has a tool on the hydraulic rig")


TIME_TOOL, WAIT_TOOL = "get_current_time", "wait_for_notification"  # the time tools of the task server, served on the level with the CausalDiagnosis MCP
# the step kinds a level or a layer may leave out, by the tool that serves each; every other kind is served on every row: read
# (history), diagnose (submit_diagnosis), order (generate_work_order), finish, profile, failure_modes, fix_history and extent
OPTIONAL_KIND_TOOLS = {"declare": DECLARE_TOOL, "wait": WAIT_TOOL, "time": TIME_TOOL}
# the clause of a clocked row that serves no wait: the cycles still pass, and what moves them
CLOCK_WITHOUT_WAIT = (" Time passes with your thinking on this row, which serves no wait: a history window past the present plays the "
                      "cycles it reaches where reads take time, and a work order closes after its lead time and downtime.")
