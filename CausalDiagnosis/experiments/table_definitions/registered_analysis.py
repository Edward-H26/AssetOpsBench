"""The one reader of the registered analysis file, registered_analysis.json beside this module: what the experiment tests, what it
only describes, and every number or rule that a table, a check of the development steps or the judge reads. The file is written by
hand and never generated, since it is the declaration that the generated tables are checked against.
Every reader of a registered value calls this module, and no other module opens the file. A value that is null is open with the
user: value() stops and names it, and missing() lists every such key, which the start check of a run of record refuses to pass (run_episodes/)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "registered_analysis.json"
FORMAT = 1
DIRECTIONS = ("above", "below", "two-sided")


class OpenValue(SystemExit):
    """A registered value that the user has not set yet, raised by value() with the dotted name of the key."""


def _path(path):
    return Path(path) if path is not None else PATH


def load(path=None):
    """The registered file as a dict, after the checks of check(), read from PATH unless a path is given."""
    data = json.loads(_path(path).read_text(encoding="utf-8"))
    problems = check(data)
    if problems:
        raise ValueError(f"{_path(path).name}: " + "; ".join(problems))
    return data


# the five numbers of the reward (the block reward), which common.episode_reward passes to the verifier
REWARD_KEYS = ("w_c", "w_t", "cost_cap", "declaration_credit", "stream_declaration_credit")
# the numbers that the tables of registered runs read, each of which must be set before such tables are built
TABLE_VALUES = ("descriptive.floor_of_pass", "counting.max_unscored_share", "counting.fraction_of_record", "intervals.min_clusters", "families.parallel_time.margin_pass",
                "families.parallel_time.margin_cost", *(f"reward.{key}" for key in REWARD_KEYS))


def reward_numbers(path=None):
    """The five numbers of the reward as {key: value}, from the block reward of the registered file."""
    block = load(path)["reward"]
    return {key: block[key] for key in REWARD_KEYS}


def check(data, root_kind=None):
    """The problems of a registered file: the format, the groups of primary, a direction, a size and measures for every family and no
    role of its own, and every question tied to one table and to families or tests that exist; for the tables of a registered root
    (root_kind record) also every number of TABLE_VALUES left open."""
    problems = []
    if root_kind == "record":
        for name in TABLE_VALUES:
            node = data
            for key in name.split("."):
                node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                problems.append(f"{name} is not set")
    if data.get("format") != FORMAT:
        problems.append(f"format must be {FORMAT}")
    families = data.get("families") or {}
    problems += _group_problems(data.get("primary") or {}, families)
    for name, family in families.items():
        if "role" in family:
            problems.append(f"family {name}: the groups of primary give its role, so it holds no role of its own")
        if family.get("direction") not in DIRECTIONS:
            problems.append(f"family {name}: direction must be one of {', '.join(DIRECTIONS)}")
        if not isinstance(family.get("size"), int) or family["size"] < 1:
            problems.append(f"family {name}: size must be a whole number above zero")
        if not family.get("measures"):
            problems.append(f"family {name}: no measure")
    for question in data.get("questions") or []:
        if not isinstance(question.get("table"), str) or not question["table"]:
            problems.append(f"question {question.get('id')}: it names no table")
        for name in question.get("families") or []:
            if name not in families:
                problems.append(f"question {question.get('id')}: no family {name}")
        for test in question.get("tests") or []:
            family, _, test_name = test.partition(":")
            names = [entry[0] for entry in (families.get(family) or {}).get("tests") or [] if isinstance(entry, list)]
            if test_name not in names:
                problems.append(f"question {question.get('id')}: no test {test}")
    return problems


def _group_problems(primary, families):
    """The problems of the block primary: at least one group, each with a name and families of the file, no family in two groups."""
    groups_found = primary.get("groups") or []
    if not groups_found:
        return ["primary names no group"]
    problems, seen = [], {}
    for group in groups_found:
        name = group.get("name")
        if not isinstance(name, str) or not name:
            problems.append("a group of primary has no name")
        if not group.get("families"):
            problems.append(f"group {name}: no family")
        for member in group.get("families") or []:
            if member not in families:
                problems.append(f"group {name}: no family {member}")
            elif member in seen:
                problems.append(f"family {member}: in the groups {seen[member]} and {name}")
            seen.setdefault(member, name)
    return problems


def families(path=None):
    return load(path)["families"]


def family(name, path=None):
    found = families(path).get(name)
    if found is None:
        raise KeyError(f"the registered file names no family {name}")
    return found


def groups(path=None):
    """The groups of primary, each the families that one question tests together and Holm corrects as one family."""
    return load(path)["primary"]["groups"]


def role(name, path=None):
    """primary for a family that a group of primary names, secondary for every other family of the file."""
    family(name, path)
    return "primary" if any(name in group["families"] for group in groups(path)) else "secondary"


def size(name, path=None):
    return family(name, path)["size"]


def measures(name, path=None):
    return tuple(family(name, path)["measures"])


def level(path=None):
    return float(load(path)["level"])


def questions(path=None):
    return load(path)["questions"]


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _leaves(child, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, node


def missing(path=None):
    """The dotted names of the keys whose value is still null, the numbers and choices open with the user."""
    return sorted(name for name, leaf in _leaves(load(path)) if leaf is None)


def value(name, path=None):
    """The value of a dotted key, as development.valid_share. A null value stops the caller with the name of the key."""
    node = load(path)
    for part in name.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"the registered file has no key {name}")
        node = node[part]
    if node is None:
        raise OpenValue(f"the registered value {name} is not set yet: the user sets it in {PATH.name} before it is read")
    return node


def sha256(path=None):
    """The SHA-256 of the file, which every result page prints at its foot and the index of the tables records."""
    return hashlib.sha256(_path(path).read_bytes()).hexdigest()
