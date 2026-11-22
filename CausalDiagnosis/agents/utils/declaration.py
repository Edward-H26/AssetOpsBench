"""The declaration of a run: the tools of the servers of a run with their roles and keys, the words of the asset, the rule cards in
force, the advice records and the rules of the kind, read before the run folder exists, and its join with the listing of the
servers into the table of one episode (CausalDiagnosis/docs/general-agent-plan.md, section The declaration).

The module imports the standard library alone and is shared code of the three designs and of the driver, so a run of another
design that reads it loads no file of an agent. A data class here may gain a field with a default, a function a keyword parameter
with a default, and a class a method, and a reader of the records of an agent ignores a key it does not know.

A listing is what the sessions of an episode list, {server: [listed tool]}, in listing order. A listed tool is an object with the
attributes name, description and inputSchema, as an MCP tool is, or a mapping with the same three keys.
"""
from dataclasses import MISSING, dataclass, field, fields, replace


class DeclarationError(Exception):
    """A fault of a declaration, by name: a listed tool with no row and no role of its server, a row whose keys do not fit, a
    tool name that two servers of a listing offer."""


@dataclass(frozen=True)
class ToolRow:
    """One tool: a row of the contract of its server, joined with the listing once an episode has opened its sessions."""

    name: str
    server: str
    role: str  # observe, knowledge, act, task or advice
    # empty in a Declaration; in a Table as the server listed it, or as the binding states it for a tool it carries
    description: str = ""
    schema: dict = field(default_factory=dict)  # the input schema, empty in a Declaration and filled as the description
    write: str | None = None  # the write class, repair, probe, state or answer; None for a read
    commits: bool = False
    ends: bool = False
    sets: str | None = None
    target: str | None = None
    words: tuple = ()
    facts: bool = False  # the tool whose answer publishes the public facts
    orients: bool = False  # a knowledge read whose answer the differential waits for
    each: str | None = None  # the argument that, left empty or left out, stands for one call per legal value of it
    withheld: bool = False  # listed, and kept out of the table that the CausalDiagnosis agent plans on
    minimum: dict = field(default_factory=dict)  # argument -> the least count that a step may write for it
    after: tuple = ()  # each item a tool name or a tuple of names, any one of which satisfies it
    values: dict = field(default_factory=dict)  # argument -> path of the facts whose keys or items are its legal values
    fills: dict = field(default_factory=dict)  # argument -> evidence, warrant, closing_summary or fixed:<value>
    candidate: tuple = ()  # on a commit write: the arguments that state the answer
    # argument -> what it takes when a step or a candidate leaves it out: a path of the facts, or fixed:<value>
    omitted: dict = field(default_factory=dict)
    waits: str | None = None  # the argument up to which the tool lets time pass
    answer: str = "object"  # the form of its answer: object, records or text
    series: str | None = None  # the argument that names the series of an observation
    requires_observation: bool = False
    advice: str | None = None  # the advice kind of a tool of the role advice
    call: dict = field(default_factory=dict)  # on an advice tool: input of its module -> argument of the tool

def _row_json(row):
    """A row as the record of a run holds it: name, server and role, and every other field where it differs from its default,
    description and schema left out, tuples as lists."""
    data = {"name": row.name, "server": row.server, "role": row.role}
    for f in fields(row):
        if f.name in ("name", "server", "role", "description", "schema"):
            continue
        value = getattr(row, f.name)
        if value != (f.default_factory() if f.default_factory is not MISSING else f.default):
            data[f.name] = _json_value(value)
    return data


def _json_value(value):
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Vocabulary:
    noun: str
    short_noun: str
    example_field: str | None = None  # a field name to give as an example
    identifiers: tuple = ()  # the forms of the identifiers a step may cite, each (form, what it names)
    terms: dict = field(default_factory=dict)  # words of the kind: time_unit, time_limit, series, target, repair, value, state


@dataclass(frozen=True)
class Declaration:
    """The declaration of one run, built from files alone before the run folder exists."""

    asset_id: str
    kind: str
    surface: str | None  # the surface of the run; None on the asset's own action server
    vocabulary: Vocabulary
    # name -> ToolRow of the asset's own tools; on a surface with a binding the rows that it states, with the rows of the diagnostics
    # server
    rows: dict
    listed_rows: dict  # (server, name) -> ToolRow of the listed tools of a framework surface; {} where the surface is the asset's own server
    server_roles: dict  # server -> role, the block roles of a contract
    cards: tuple  # the public rule cards in force, each {"kind", "parameters"}
    advice: dict  # tool -> advice record {"kind", "fields", "warrant", "example"}; {} where the run has no diagnostics server
    events: tuple  # the names of the events that an answer can notify
    rules: object  # the rules of the kind
    binding: object | None  # the binding of the surface; None where every tool is called as listed

    def table(self, listing):
        """The table of an episode on which the CausalDiagnosis agent plans: every row whose tool the listing serves and that is not
        withheld, in listing order, each with the description and the schema that the listing gives it. A tool that the binding
        carries counts as served where the listed tool that carries it is listed, at its place, and keeps the description and the
        schema that the binding states. Where no row of the declaration states candidate, a commit write takes as its candidate
        arguments the required arguments of its schema that no fill writes. A listed tool that has no row, carries no tool, has no
        row of the framework surface and no role of its server raises a DeclarationError by name."""
        carried = {carrier: name for name, carrier in (getattr(self.binding, "carried", None) or {}).items()}
        tools = {}
        for server, tool in _listed(listing):
            name = tool["name"]
            if (server, name) in carried:
                if carried[(server, name)] not in self.rows:
                    raise DeclarationError(f"the binding carries {carried[(server, name)]} by {server}.{name} and states no row of it")
                row = self.rows[carried[(server, name)]]
            elif name in self.rows and self.rows[name].server == server:
                row = replace(self.rows[name], description=tool["description"], schema=tool["schema"])
            elif (server, name) in self.listed_rows or server in self.server_roles:
                continue
            else:
                raise DeclarationError(f"the server {server} lists the tool {name}, which has no row and whose server has no role")
            if not row.withheld:
                tools[row.name] = row
        return Table(self, self._completed(tools))

    def listed(self, listing):
        """Every listed tool under its listed name, with its listed description and schema: its row is its row of the framework
        surface, else the row of that name of the same server; its role is the role that the contract gives its server, else the
        role of its row; its write class and row keys are those of its row. The defaults that need the listing are applied as in
        table. A listed tool with neither a row nor a role of its server, and a tool name that two servers of the listing offer,
        raise a DeclarationError by name."""
        tools, servers = {}, {}
        for server, tool in _listed(listing):
            name = tool["name"]
            if name in servers:
                raise DeclarationError(f"the tool {name} is listed by two servers, {servers[name]} and {server}")
            servers[name] = server
            own = self.listed_rows.get((server, name))
            if own is None and name in self.rows and self.rows[name].server == server:
                own = self.rows[name]
            role = self.server_roles.get(server, own.role if own is not None else None)
            if role is None:
                raise DeclarationError(f"the server {server} lists the tool {name}, which has no row and whose server has no role")
            row = own if own is not None else ToolRow(name=name, server=server, role=role)
            tools[name] = replace(row, name=name, server=server, role=role, description=tool["description"], schema=tool["schema"])
        return Table(self, self._completed(tools))

    def _completed(self, tools):
        """The rows of a table with the default of candidate applied where no row of the declaration states the key."""
        stated = any(row.candidate for row in (*self.rows.values(), *self.listed_rows.values()))
        if stated:
            return tools
        return {name: replace(row, candidate=_required_unfilled(row)) if row.commits and not row.candidate else row
                for name, row in tools.items()}

    def to_json(self):
        """What run.json records of the declaration: the rows, the vocabulary, the kinds of the cards, the advice kinds, the events
        and whether a binding stands."""
        return {"asset_id": self.asset_id, "kind": self.kind, "surface": self.surface,
                "vocabulary": _json_value({"noun": self.vocabulary.noun, "short_noun": self.vocabulary.short_noun,
                                           "example_field": self.vocabulary.example_field,
                                           "identifiers": self.vocabulary.identifiers, "terms": self.vocabulary.terms}),
                "rows": {name: _row_json(row) for name, row in self.rows.items()},
                "cards": [card["kind"] for card in self.cards],
                "advice": {tool: record["kind"] for tool, record in self.advice.items()},
                "events": list(self.events), "binding": self.binding is not None}


@dataclass(frozen=True)
class Table:
    """The tools of one episode."""

    declaration: Declaration
    tools: dict  # name -> ToolRow with description and schema, in listing order

    def of_role(self, role):
        return tuple(name for name, row in self.tools.items() if row.role == role)

    def of_write(self, write):
        return tuple(name for name, row in self.tools.items() if row.write == write)

    def commit_tools(self):
        return tuple(name for name, row in self.tools.items() if row.commits)

    def end_tool(self):
        return next((name for name, row in self.tools.items() if row.ends), None)

    def facts_tool(self):
        return next((name for name, row in self.tools.items() if row.facts), None)

    def advice_kinds(self):
        """The kinds of the advice tools of the table, each once, in listing order."""
        return tuple(dict.fromkeys(row.advice for row in self.tools.values() if row.role == "advice" and row.advice))


def _listed(listing):
    """(server, {name, description, schema}) of every listed tool, in listing order, from MCP tools or mappings."""
    for server, tools in listing.items():
        for tool in tools:
            if isinstance(tool, dict):
                name, description, schema = tool.get("name"), tool.get("description"), tool.get("inputSchema")
            else:
                name, description, schema = tool.name, getattr(tool, "description", None), getattr(tool, "inputSchema", None)
            yield server, {"name": name, "description": description or "", "schema": dict(schema or {})}


def _required_unfilled(row):
    """The required arguments of the listed schema of a row that no fill writes, in the order of the schema."""
    return tuple(name for name in row.schema.get("required") or () if name not in row.fills)
