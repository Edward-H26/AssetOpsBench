"""The four fixture declarations of the freeze, written as data, each with its listing (CausalDiagnosis/docs/general-agent-tasks.md,
step 0): the rig on its own action server; the rig on a row of AssetOpsBench's servers, with a stub of the binding that holds its
carried map and the rows it states; the made-up port bottler_1, from the rows of the plan page with the defaults of a row applied;
and one with a single tool. Every task builds and tests against them.

The rows hold the defaults that the builder of the declaration applies (the form of an answer of the kind, and commits on the
write that ends where no row states commits); Declaration.table applies the default of candidate, which needs the listing. The
listings of the rig and of the AssetOpsBench rows were taken from the servers of the tree on the presets base and real_time and
are kept under data/, each listed tool with its name, description and input schema in listing order. The rules of each kind are
stubs of KindRules here. Vocabulary.identifiers is empty in all four: the identifier forms are constants beside the answer forms,
the ledger of the kind and the modules, which the assembler joins."""
import json
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary
from CausalDiagnosis.utils.public_rules.plan_rules import KindRules

DATA = Path(__file__).resolve().parent / "data"
RIG_SERVER, DIAGNOSTICS_SERVER = "hydraulic_rig", "causal_diagnosis"
AOB_ROLES = {"iot": "observe", "fmsr": "knowledge", "wo": "act", "task": "task"}  # the block roles of the contract of the AssetOpsBench rows
AOB_ROWS = ("base", "real_time")  # the rows whose listings data/ holds
# the fields of each advice kind and where its answer carries a warrant, as the backend interface of the diagnostics server states them
ADVICE_FIELDS = {
    "evidence": ("message", "evidence_ids", "items", "items[].evidence_id", "items[].field", "items[].value"),
    "ranking": ("message", "evidence_ids", "candidates", "candidates[].rank", "candidates[].combination", "candidates[].probability",
                "candidates[].faults", "candidates[].evidence_ids"),
    "plan_assessment": ("message", "evidence_ids", "prediction_id", "action", "recovery"),
    "outcome_check": ("message", "evidence_ids", "prediction_id", "status", "consistent"),
}
WARRANT_FIELDS = {
    "evidence": (),
    "ranking": ("branches[].repair_plan.plan_prediction_id", "branches[].repair_plan.steps[].prediction_id", "branches[].faulty_set",
                "branches[].repair_plan.steps[].component", "branches[].predicted_outcome.recovery_if_acted"),
    "plan_assessment": ("prediction_id", "action", "recovery"),
    "outcome_check": (),
}


class RecordedLevelsRules(KindRules):
    """A stub of the rules of recorded levels."""

    default_answer = "object"


class ScriptedCasesRules(KindRules):
    """A stub of the rules of scripted cases, with the phrase that the texts of the kind write into every answer to a call that
    was not carried out."""

    default_answer = "text"
    refusal_marks = ("not carried out",)


def listing(name):
    """The listing of data/listing_<name>.json, {server: [listed tool]}, each listed tool with the attributes name, description
    and inputSchema, as the sessions of an episode list them."""
    raw = json.loads((DATA / f"listing_{name}.json").read_text(encoding="utf-8"))
    return {server: [SimpleNamespace(**tool) for tool in tools] for server, tools in raw.items()}


def _listed(listed, server, name):
    return next(tool for tool in listed[server] if tool.name == name)


def _components(listed):
    """The components of the rig: the required arguments of its answer write that no fill writes, in the order of the schema."""
    schema = _listed(listed, RIG_SERVER, "submit_diagnosis").inputSchema
    return tuple(name for name in schema["required"] if name != "evidence_ids")


def _advice(kind):
    return {"kind": kind, "fields": ADVICE_FIELDS[kind], "warrant": WARRANT_FIELDS[kind], "example": None}


def rig_rows():
    """The sixteen rows of the engine's contract with the keys of the plan page: twelve of the action server, four of the
    diagnostics server."""
    components = _components(listing("rig"))
    rig = {
        "get_asset_profile": dict(role="task", facts=True),
        "read_sensors": dict(role="observe", values={"sensor_group": "sensor_groups"}, series="sensor_group", answer="records",
                             minimum={"cycles": 1}, omitted={"cycles": "fixed:1"}),
        "get_current_time": dict(role="task", withheld=True),
        "wait_for_notification": dict(role="task", waits="timeout", minimum={"timeout": 1}, omitted={"timeout": "fixed:1"}),
        "summarize_changes_since": dict(role="task"),
        "get_failure_modes": dict(role="knowledge", values={"component": "components"}, orients=True, each="component"),
        "get_fix_history": dict(role="knowledge"),
        "lookup_fix": dict(role="knowledge", values={"component": "components"}, withheld=True),
        "submit_diagnosis": dict(role="task", write="answer", commits=True, candidate=components,
                                 values={c: f"levels.{c}" for c in components}, omitted={c: f"healthy.{c}" for c in components},
                                 fills={"evidence_ids": "evidence"}),
        "declare_insufficient_evidence": dict(role="task", write="answer", commits=True, requires_observation=True,
                                              values={"sensor_group": "sensor_groups"}, minimum={"cycles": 1},
                                              omitted={"cycles": "fixed:1"}, fills={"evidence_ids": "evidence"}),
        "create_work_order": dict(role="act", write="repair", target="component", values={"component": "components"},
                                  after=(("submit_diagnosis", "declare_insufficient_evidence"),),
                                  fills={"prediction_id": "warrant", "evidence_ids": "evidence", "mode": "fixed:standard"}),
        "finish": dict(role="task", write="answer", ends=True, fills={"summary": "closing_summary"}),
    }
    diagnostics = {
        "get_episode_evidence": dict(role="advice", advice="evidence"),
        "rank_root_causes": dict(role="advice", advice="ranking", call={"detail": "branch_detail", "top": "top_k"}),
        "estimate_intervention_effect": dict(role="advice", advice="plan_assessment", call={"action": "action"}),
        "verify_outcome": dict(role="advice", advice="outcome_check", call={"warrant": "prediction_id"}),
    }
    return {**{name: ToolRow(name=name, server=RIG_SERVER, **keys) for name, keys in rig.items()},
            **{name: ToolRow(name=name, server=DIAGNOSTICS_SERVER, **keys) for name, keys in diagnostics.items()}}


RIG_VOCABULARY = Vocabulary(noun="hydraulic test rig", short_noun="rig", example_field="PS1",
                            terms={"time_unit": "cycle", "time_limit": "horizon", "series": "sensor group", "target": "component",
                                   "repair": "work order", "value": "mean"})
# the public cards of the asset file as the configuration of the preset base holds them
RIG_CARDS = (
    {"kind": "fix_history", "parameters": {"healthy_probability": 0.0, "healthy_ago": [0, 0], "degraded_probability": 0.0, "degraded_ago": [0, 0]}},
    {"kind": "emergency", "parameters": {"lead": 0, "surcharge": 1.0, "hold": 0}},
    {"kind": "schedule", "parameters": {"mode": "none", "components": 0, "first_drop": [0, 0], "interval": 1}},
    {"kind": "hazard", "parameters": {"mode": "scheduled", "base_rate": 0.0, "depth_factor": 0.0, "collateral": 0.0}},
    {"kind": "dependencies", "parameters": {"dependencies": {}, "priority": {"cooler": 1, "valve": 1, "pump": 1, "accumulator": 1}}},
    {"kind": "visits", "parameters": {"charge": 0}},
    {"kind": "downtime", "parameters": {"rates": {"cooler": {"3": 0, "20": 0, "100": 0}, "valve": {"73": 0, "80": 0, "90": 0, "100": 0},
                                                  "pump": {"0": 0, "1": 0, "2": 0}, "accumulator": {"90": 0, "100": 0, "115": 0, "130": 0}},
                                        "stopped": 0}},
    {"kind": "failure", "parameters": {"patience": 0, "strict": False, "penalty": 0}},
)
RIG_EVENTS = ("work_order_executed", "work_order_closed", "failure", "alarm")  # the notifying events of the engine's contract


def _rig_advice(rows):
    return {name: _advice(row.advice) for name, row in rows.items() if row.role == "advice"}


def rig_declaration():
    """The rig on its own action server, with the diagnostics server."""
    rows = rig_rows()
    return Declaration(asset_id="hyd_1", kind="recorded_levels", surface=None, vocabulary=RIG_VOCABULARY, rows=rows, listed_rows={},
                       server_roles={}, cards=RIG_CARDS, advice=_rig_advice(rows), events=RIG_EVENTS, rules=RecordedLevelsRules(),
                       binding=None)


class AobBindingStub:
    """A stub of the binding of the AssetOpsBench rows, for one row of them: the table tools it carries, each by its listed tool,
    the rows it states for them with a description and a schema, and the tools it needs answered before each. A stated row keeps
    every key of the engine's row for the same tool, but get_failure_modes holds no each (the server takes one component per
    call) and get_current_time is not withheld; it carries no lookup_fix. The clock, the wait and the change summary are carried
    where the row lists them. write and read are those of the binding of the plan, which this stub does not hold."""

    def __init__(self, row_listing):
        task = {tool.name for tool in row_listing["task"]}
        carried = {"read_sensors": ("iot", "history"), "get_failure_modes": ("fmsr", "get_failure_modes"),
                   "get_fix_history": ("wo", "list_workorders"), "create_work_order": ("wo", "generate_work_order"),
                   "stream_extent": ("iot", "stream_extent")}
        for name in ("get_asset_profile", "get_current_time", "wait_for_notification", "summarize_changes_since", "submit_diagnosis",
                     "declare_insufficient_evidence", "finish"):
            if name in task:
                carried[name] = ("task", name)
        self.carried = carried
        self.flags = {"reads_take_time": False, "wait_served": "wait_for_notification" in task}
        self._rows = self._stated(row_listing)

    def _stated(self, row_listing):
        engine, rig = rig_rows(), listing("rig")
        rows = {}
        for name, (server, tool) in self.carried.items():
            if name == "stream_extent":
                described = _listed(row_listing, server, tool)
                rows[name] = ToolRow(name=name, server=server, role="task", description=described.description,
                                     schema={"type": "object", "properties": {}})
                continue
            described = _listed(rig, RIG_SERVER, name)
            row = engine[name]
            keys = {"each": None} if name == "get_failure_modes" else {"withheld": False} if name == "get_current_time" else {}
            rows[name] = ToolRow(**{**row.__dict__, **keys, "server": server, "description": described.description,
                                    "schema": described.inputSchema})
        return rows

    def rows(self):
        return dict(self._rows)

    def needs(self, tool):
        """The tools that the binding needs answered before it can write the arguments of the tool or read its answer: the facts
        tool for the four tools whose site, asset, fields or child asset come from the facts, and before a read the clock where
        the row lists it and the extent of the record."""
        facts = ("get_asset_profile",) if tool in ("stream_extent", "read_sensors", "get_fix_history", "create_work_order") else ()
        if tool == "read_sensors":
            return facts + tuple(name for name in ("get_current_time", "stream_extent") if name in self.carried)
        return facts

    def write(self, tool, args, view):
        raise NotImplementedError

    def read(self, tool, args, text, is_error, view):
        raise NotImplementedError


def _aob_listed_rows(row_listing):
    """(server, name) -> the row of every tool that a server of the AssetOpsBench rows lists: the role of its server, the write
    class repair on generate_work_order, and on a task tool the write class and the keys of the engine's row of that name."""
    engine = rig_rows()
    rows = {}
    for server in AOB_ROLES:
        for tool in row_listing[server]:
            if server == "task" and tool.name in engine:
                rows[(server, tool.name)] = ToolRow(**{**engine[tool.name].__dict__, "server": server, "role": "task"})
            else:
                write = "repair" if (server, tool.name) == ("wo", "generate_work_order") else None
                rows[(server, tool.name)] = ToolRow(name=tool.name, server=server, role=AOB_ROLES[server], write=write)
    return rows


def aob_declaration(row="base"):
    """The rig on a row of AssetOpsBench's servers, with the diagnostics server: the rows that the binding stub states for the
    row, beside the four rows of the diagnostics server, the listed rows of the four servers and their block roles."""
    row_listing = listing(f"aob_{row}")
    binding = AobBindingStub(row_listing)
    diagnostics = {name: r for name, r in rig_rows().items() if r.server == DIAGNOSTICS_SERVER}
    rows = {**binding.rows(), **diagnostics}
    return Declaration(asset_id="hyd_1", kind="recorded_levels", surface="aob", vocabulary=RIG_VOCABULARY, rows=rows,
                       listed_rows=_aob_listed_rows(row_listing), server_roles=dict(AOB_ROLES), cards=RIG_CARDS,
                       advice=_rig_advice(diagnostics), events=RIG_EVENTS, rules=RecordedLevelsRules(), binding=binding)


PORT_SERVER = "bottler_1"  # the shared action server of scripted cases, under the asset's name


def _schema(properties, required):
    return {"type": "object", "properties": properties, "required": list(required)}


PORT_LISTING = {
    PORT_SERVER: [
        {"name": "read_gauge", "description": "Read one gauge of the filler.",
         "inputSchema": _schema({"gauge": {"type": "string", "enum": ["flow", "level", "torque"]}}, ["gauge"])},
        {"name": "inspect_nozzle", "description": "Look at one nozzle, or open it.",
         "inputSchema": _schema({"nozzle": {"type": "string"}, "mode": {"type": "string", "enum": ["look", "open"]},
                                 "note": {"type": "string"}}, ["nozzle"])},
        {"name": "lock_out", "description": "Isolate the supply.",
         "inputSchema": _schema({"valve": {"type": "string", "enum": ["main", "bypass"]}}, [])},
        {"name": "restore_supply", "description": "Make the supply live again.", "inputSchema": _schema({}, [])},
        {"name": "swap_part", "description": "Replace one part.",
         "inputSchema": _schema({"part": {"type": "string"}, "grade": {"type": "string", "enum": ["new", "rebuilt"]},
                                 "reason": {"type": "string"}}, ["part"])},
        {"name": "test_run", "description": "Run the filler on a batch of bottles.",
         "inputSchema": _schema({"bottles": {"type": "string", "enum": ["10", "50"]}}, ["bottles"])},
        {"name": "file_report", "description": "File the report of the case, which ends it.",
         "inputSchema": _schema({"cause": {"type": "string"}}, ["cause"])},
    ],
    DIAGNOSTICS_SERVER: [
        {"name": "get_episode_evidence", "description": "Every observation released so far, with its identifier.",
         "inputSchema": _schema({"tool": {"type": "string"}}, ["tool"])},
        {"name": "check_repair_plan", "description": "Check a planned sequence of calls against the published requirements.",
         "inputSchema": _schema({"plan": {"type": "string"}}, ["plan"])},
    ],
}
# the precondition card of the port's asset file
PORT_CARDS = ({"kind": "precondition", "parameters": {
    "state": "supply", "values": ["live", "isolated"], "initial": "entry",
    "requirements": {"read_gauge": "live", "lock_out": "live", "restore_supply": "isolated", "swap_part": "isolated", "test_run": "live",
                     "inspect_nozzle": {"argument": "mode", "values": {"open": "isolated"}}},
    "on_violation": "end_episode"}},)


def port_listing():
    return {server: [SimpleNamespace(**tool) for tool in tools] for server, tools in PORT_LISTING.items()}


def port_declaration():
    """The made-up port bottler_1 with the scripted backend of the diagnostics server: seven rows of its action server and the two
    advice rows of the backend, the form text of the kind on its observe and act tools, and commits on its write that ends, since
    no row states commits."""
    action = {
        "read_gauge": dict(role="observe", words=("normal", "low", "high")),
        "inspect_nozzle": dict(role="observe", words=("clear", "clogged", "worn")),
        "lock_out": dict(role="act", write="state", sets="isolated"),
        "restore_supply": dict(role="act", write="state", sets="live"),
        "swap_part": dict(role="act", write="repair", target="part"),
        "test_run": dict(role="act", write="probe"),
    }
    rows = {name: ToolRow(name=name, server=PORT_SERVER, answer="text", **keys) for name, keys in action.items()}
    rows["file_report"] = ToolRow(name="file_report", server=PORT_SERVER, role="task", write="answer", ends=True, commits=True)
    rows["get_episode_evidence"] = ToolRow(name="get_episode_evidence", server=DIAGNOSTICS_SERVER, role="advice", advice="evidence")
    rows["check_repair_plan"] = ToolRow(name="check_repair_plan", server=DIAGNOSTICS_SERVER, role="advice",
                                        advice="plan_assessment", call={"calls": "plan"})
    advice = {"get_episode_evidence": _advice("evidence"), "check_repair_plan": _advice("plan_assessment")}
    return Declaration(asset_id="bottler_1", kind="scripted_cases", surface=None,
                       vocabulary=Vocabulary(noun="bottle filler", short_noun="filler", terms={"target": "part", "repair": "swap_part",
                                                                                              "state": "supply"}),
                       rows=rows, listed_rows={}, server_roles={}, cards=PORT_CARDS, advice=advice, events=(),
                       rules=ScriptedCasesRules(), binding=None)


SINGLE_SERVER = "single_tool"
SINGLE_LISTING = {SINGLE_SERVER: [{"name": "answer", "description": "Give the answer, which ends the episode.",
                                   "inputSchema": _schema({"text": {"type": "string"}}, ["text"])}]}


def single_listing():
    return {server: [SimpleNamespace(**tool) for tool in tools] for server, tools in SINGLE_LISTING.items()}


def single_declaration():
    """One tool, of the role task and the write class answer, that ends the episode and so commits by the default."""
    row = ToolRow(name="answer", server=SINGLE_SERVER, role="task", write="answer", ends=True, commits=True)
    return Declaration(asset_id="single_tool", kind="scripted_cases", surface=None, vocabulary=Vocabulary(noun="test object", short_noun="object"),
                       rows={"answer": row}, listed_rows={}, server_roles={}, cards=(), advice={}, events=(), rules=ScriptedCasesRules(),
                       binding=None)


def declarations():
    """name -> (declaration, listing) of the four fixtures, and the declaration of the row real_time beside that of the base row."""
    return {"rig": (rig_declaration(), listing("rig")), "aob": (aob_declaration("base"), listing("aob_base")),
            "aob_real_time": (aob_declaration("real_time"), listing("aob_real_time")), "port": (port_declaration(), port_listing()),
            "single": (single_declaration(), single_listing())}
