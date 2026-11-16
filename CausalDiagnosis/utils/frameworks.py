"""The registry of the frameworks under CausalDiagnosis/frameworks/: what each framework declares, read as data, and the code
it names, imported on first use.

A framework folder holds a declaration in its __init__.py (frameworks/README.md): module-level assignments of plain literals
(dicts, tuples, strings, numbers, booleans, None), which this module reads with ast.literal_eval and never imports, so a core
module may read a declaration while it is imported and still loads no module under frameworks/. Code is named by a string
"module:attribute", the field "entry" of an entry or a top-level reference field, and imported only when a run asks for it.
A field this core does not know is ignored with one message per framework and field; a known field whose value is no literal
or has the wrong shape stops the registry with the framework and the field, since the core would misread it. Two frameworks
that declare one name in a field of named entries stop the registry, since a run could not tell which one it names.
"""
import ast
import importlib
import sys
from functools import lru_cache

from CausalDiagnosis.utils.paths import PACKAGE

FRAMEWORKS = PACKAGE / "frameworks"
DECLARATION = "__init__.py"
# fields of named entries: {name: {key: literal}}, each entry may name its code by "entry": "module:attribute"
NAMED_FIELDS = ("AGENTS", "SURFACES", "JUDGES", "SCENARIO_SETS", "RUN_KINDS")
# fields of plain names: a tuple of strings
NAME_FIELDS = ("CONFIGS", "JOB_ENVIRONMENT")
# fields of references to code: {name: "module:attribute"}
REFERENCE_FIELDS = ("PROFILE_CHECKS", "DIGEST_FAMILIES")
COMMAND_FIELD = "COMMANDS"  # {command: module string}, started as python -m starts the module
LIST_FIELD = "LISTS"  # {list name: tuple of paths relative to the framework folder}
# the tool surface of a framework at a level: the actions that answer an episode, which every level serves on the action server of
# the surface, and what each level adds to the framework's own tools (framework: none of the environment's actions; full: the environment's actions and
# the diagnostics server)
ANSWER_ACTIONS = ("submit_diagnosis", "finish")
LEVELS = {"framework": (), "full": ("environment", "diagnostics")}
KNOWN_FIELDS = (*NAMED_FIELDS, *NAME_FIELDS, *REFERENCE_FIELDS, COMMAND_FIELD, LIST_FIELD)
_warned = set()


class FrameworkError(Exception):
    """A declaration the core cannot read: a known field that is no literal or has the wrong shape, or a name two frameworks
    declare."""


def frameworks():
    """The names of the framework folders that hold a declaration, sorted."""
    if not FRAMEWORKS.is_dir():
        return ()
    return tuple(sorted(p.name for p in FRAMEWORKS.iterdir() if p.is_dir() and (p / DECLARATION).is_file()))


@lru_cache(maxsize=None)
def declaration(name):
    """{field: value} of the known fields that the framework's __init__.py assigns at module level, read without importing it.
    A field that is not assigned reads as empty."""
    path = FRAMEWORKS / name / DECLARATION
    fields = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        field = node.targets[0].id
        if not field.isupper():
            continue
        if field not in KNOWN_FIELDS:
            _ignore(name, field)
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            raise FrameworkError(f"frameworks/{name}: field {field} is not a literal (line {node.lineno})") from None
        fields[field] = _checked(name, field, value)
    return fields


def _ignore(name, field):
    if (name, field) not in _warned:
        _warned.add((name, field))
        print(f"frameworks/{name}: field {field} is not known to this core and is ignored", file=sys.stderr)


def _checked(name, field, value):
    def fail(shape):
        raise FrameworkError(f"frameworks/{name}: field {field} must be {shape}")

    if field in NAMED_FIELDS:
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, dict) for k, v in value.items()):
            fail("a dict of names to dicts")
        for key, entry in value.items():
            if "entry" in entry and not _is_reference(entry["entry"]):
                fail(f"a dict whose entry {key} names its code as 'module:attribute'")
            if field == "SURFACES":
                for problem in _surface_problems(entry):
                    raise FrameworkError(f"frameworks/{name}: field {field} entry {key}: {problem}")
    elif field in NAME_FIELDS:
        if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
            fail("a tuple of strings")
    elif field in REFERENCE_FIELDS:
        if not isinstance(value, dict) or not all(isinstance(k, str) and _is_reference(v) for k, v in value.items()):
            fail("a dict of names to 'module:attribute' strings")
    elif field == COMMAND_FIELD:
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) and v and ":" not in v for k, v in value.items()):
            fail("a dict of commands to module strings")
    elif field == LIST_FIELD:
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, tuple) and all(isinstance(p, str) for p in v)
                                                  for k, v in value.items()):
            fail("a dict of list names to tuples of paths")
    return value


def _surface_problems(entry):
    """The shape problems of the keys of a surface that name what it serves of the environment's actions (action_server_name, covers,
    withheld_actions, levels, gateway), each named by its key; a surface without them serves no action of the environment. Whether a
    covered or withheld name is an action of the contract of the environment's action server is checked by the tests, since this module reads no
    layer above it."""
    servers = entry.get("servers", ())
    action_server = entry.get("action_server_name")
    if "action_server_name" in entry and action_server not in servers:
        yield "action_server_name must be one of its servers"
    covers = entry.get("covers", {})
    if not isinstance(covers, dict) or any(not isinstance(k, str) or k in ANSWER_ACTIONS for k in covers):
        yield "covers must map actions that answer no episode (not in ANSWER_ACTIONS) to tools"
    else:
        for action, tools in covers.items():
            if not isinstance(tools, tuple) or not tools or any(not isinstance(t, str) or t.count(".") != 1 for t in tools):
                yield f"covers {action} must be a non-empty tuple of 'server.tool'"
            elif any(t.split(".")[0] not in servers or t.split(".")[0] == action_server for t in tools):
                yield f"covers {action} must name tools of its servers other than action_server_name"
    withheld = entry.get("withheld_actions", ())
    if not isinstance(withheld, tuple) or any(not isinstance(a, str) or a in ANSWER_ACTIONS or a in covers for a in withheld):
        yield "withheld_actions must be a tuple of actions outside covers and ANSWER_ACTIONS"
    levels = entry.get("levels", ())
    if "levels" in entry and (not isinstance(levels, tuple) or not levels or any(level not in LEVELS for level in levels)):
        yield f"levels must be a non-empty tuple of {', '.join(LEVELS)}"
    if "gateway" in entry and not _is_reference(entry["gateway"]):
        yield "gateway must name its code as 'module:attribute'"


def _is_reference(value):
    return isinstance(value, str) and value.count(":") == 1 and all(value.split(":"))


def entries(field):
    """{name: entry} of a field of named entries over every framework, each entry with the key "framework" added. A name that
    two frameworks declare stops the registry."""
    if field not in NAMED_FIELDS:
        raise KeyError(f"{field} is not a field of named entries ({', '.join(NAMED_FIELDS)})")
    out = {}
    for name in frameworks():
        for key, entry in declaration(name).get(field, {}).items():
            if key in out:
                raise FrameworkError(f"{field} {key} is declared by frameworks/{out[key]['framework']} and frameworks/{name}")
            out[key] = {**entry, "framework": name}
    return out


def agents():
    """{agent name: entry} of every declared agent kind."""
    return entries("AGENTS")


def surfaces():
    """{surface name: entry} of every declared tool surface."""
    return entries("SURFACES")


def judges():
    """{judge name: entry} of every declared judge."""
    return entries("JUDGES")


def scenario_sets():
    """{scenario set name: entry} of every declared scenario set."""
    return entries("SCENARIO_SETS")


def run_kinds():
    """{run kind: entry} of every declared kind of run."""
    return entries("RUN_KINDS")


def commands():
    """{group: {command: module string}} of every framework that declares commands; the group is the folder name in lower
    case (frameworks/AssetOpsBench gives assetopsbench)."""
    return {name.lower(): dict(declaration(name)[COMMAND_FIELD]) for name in frameworks() if declaration(name).get(COMMAND_FIELD)}


def names(field):
    """((framework, name), ...) of a field of plain names over every framework, in framework order."""
    if field not in NAME_FIELDS:
        raise KeyError(f"{field} is not a field of plain names ({', '.join(NAME_FIELDS)})")
    return tuple((name, value) for name in frameworks() for value in declaration(name).get(field, ()))


def references(field):
    """((framework, name, "module:attribute"), ...) of a reference field over every framework."""
    if field not in REFERENCE_FIELDS:
        raise KeyError(f"{field} is not a reference field ({', '.join(REFERENCE_FIELDS)})")
    return tuple((name, key, ref) for name in frameworks() for key, ref in declaration(name).get(field, {}).items())


def lists(name):
    """{list name: tuple of paths relative to CausalDiagnosis/frameworks/<name>/} that the framework declares for the identity
    lists (experiment_tables.lists_of)."""
    return declaration(name).get(LIST_FIELD, {})


def framework_dir(name):
    """The folder of a framework."""
    return FRAMEWORKS / name


def resolve(reference):
    """The object a "module:attribute" string names, its module imported on first use."""
    if not _is_reference(reference):
        raise FrameworkError(f"{reference!r} names no code as 'module:attribute'")
    module, attribute = reference.split(":")
    return getattr(importlib.import_module(module), attribute)


def code(field, key):
    """The object that the entry key of a field of named entries names by "entry"."""
    entry = entries(field)[key]
    if "entry" not in entry:
        raise FrameworkError(f"{field} {key} of frameworks/{entry['framework']} names no code")
    return resolve(entry["entry"])


def surface_profile_problems(profile, surface):
    """The problems a declared surface finds in an asset profile (PLAN.md 4.6): the check that the surface's entry of
    PROFILE_CHECKS names, run on the profile; none for a surface that declares no check."""
    checks = {key: ref for _name, key, ref in references("PROFILE_CHECKS")}
    if surface not in surfaces():
        raise KeyError(f"no framework declares the surface {surface!r}")
    return list(resolve(checks[surface])(profile)) if surface in checks else []
