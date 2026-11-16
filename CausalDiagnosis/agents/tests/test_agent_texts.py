"""No text an agent reads states a level of a component of a registered asset.

The rule, for any asset. A registered asset is one with a file CausalDiagnosis/datasets/<asset>/asset.json, so a new asset is covered
as soon as its file is added, with no change here. A text states a level of a component when it holds one of four forms:

- a name of the component, its own name or its label column in the data file, followed on the same line by one of that
  component's levels as a whole number, with nothing but words between the two, as "pump 1", "pump=1", '"pump": 1',
  "pump at level 1" and "the pump that was replaced is now at 1" do. The first number after the name is the one that counts;
- the same pair in the other order, as "level 1 of the pump";
- the code of a failure mode that belongs to one level of one component;
- one whole number per component, in the order of the components and separated by commas, each a level of its component,
  which is a full combination without names. An asset with one component has no such sequence.

The texts an agent reads, as files. Under CausalDiagnosis/agents and src/agent (the prompts, the controller rules, the in-context
examples, the vendored engine with its prompt templates), CausalDiagnosis/mcp_servers and src/servers (the briefings, the tool
descriptions, the fixed sentences of a tool answer), CausalDiagnosis/utils, and the briefings and tool contracts of
CausalDiagnosis/environment named in AGENT_TEXT_FILES, outside the tests folders, a Python file is read in two ways. The three named forms are searched in the
whole file without its comments, code and string literals and docstrings alike, so that a rule written in code that names a
level is found, and the run of numbers is searched in its string literals alone, docstrings included, since plain code holds
such runs as indexes, ranges and counters. Every other file there that decodes as text is read in full, whatever its suffix,
and so is a Python file that does not parse. Of the Python files under CausalDiagnosis/environment and
CausalDiagnosis/datasets every string literal outside a docstring is
read, since those are the strings the environment itself puts into a tool answer. Comments are read nowhere. Left out, since
no agent reads them as text: the asset files, which define the levels and are served as data, and the tables the diagnostics
server builds for itself (models.json and grades.json, wherever they lie). A file of a port that no agent reads, such as
recordings, fitted tables or caches, lies in the folder of its asset under CausalDiagnosis/datasets, where only Python literals are
read, or outside these folders. A made-up asset of an example is not registered, so its values pass unless they state a level of a
component of a registered asset. A text made at run time, such as an example filled from the tool contract or a block that a
training loop wrote, is checked where it is made.

Accepted coincidences. An asset may hold CausalDiagnosis/datasets/<asset>/accepted_texts.json, a list of entries {"file": a scanned
file relative to the repository root, "text": the matched text, "count": how many hits of that text in that file it accepts, "why": one
sentence}. For that asset the scan drops those hits when their number is the count of the entry. An entry whose count is not
that number, or that lacks a key, fails the test, so the list cannot go stale and a new occurrence cannot ride on an old entry.
A text that states a level of the asset is reworded where it stands, and only a text that no agent can take for such a
statement is accepted, so a port edits no file outside its own folder.
"""
import ast
import io
import json
import re
import tokenize
from pathlib import Path

import pytest

from CausalDiagnosis.utils.paths import ROOT
AGENT_ROOTS = ("CausalDiagnosis/agents", "CausalDiagnosis/mcp_servers", "CausalDiagnosis/utils", "src/agent", "src/servers")  # Python files: the named forms in the code without comments, the runs in the literals with docstrings
ENV_ROOTS = ("CausalDiagnosis/environment", "CausalDiagnosis/datasets")  # Python files: every string literal outside a docstring
SKIPPED_PARTS = {"tests", "__pycache__"}
GENERATED = {"models.json", "grades.json"}  # the tables the diagnostics server builds for itself, which no agent reads
ASSETS = "CausalDiagnosis/datasets"  # the folder of the registered assets, one asset.json each
ACCEPTED = "accepted_texts.json"  # the accepted coincidences of an asset, in the folder of its asset.json
AGENT_TEXT_FILES = tuple("CausalDiagnosis/environment/Causal_Diagnosis_Environment/" + name for name in (
    "briefing.py", "aob_briefing.py", "tool_contract.py", "aob_tool_contract.py", "tool_contract.json", "aob_tool_contract.json"))
# the texts an agent reads that live in the environment, read as under the agent roots
ENTRY_KEYS = ("file", "text", "count", "why")
WORD = "[A-Za-z0-9_]"
NUMBER_RUN = re.compile(r"(?<![\w.])-?\d+(?:\s*,\s*-?\d+)+(?![\w.])")  # whole numbers separated by commas, as far as the run goes


def registered_assets(src):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((src / ASSETS).glob("*/asset.json"))]


def _longest_first(values):
    return "|".join(re.escape(str(value)) for value in sorted(set(values), key=lambda value: (-len(str(value)), str(value))))


def leveled_codes(asset):
    """The failure mode codes of an asset that belong to one level of one of its components."""
    return [mode["code"] for mode in asset.get("failure_modes") or [] if mode.get("level") is not None and mode.get("component") in asset["components"]]


def named_level(asset):
    """The pattern of the three named forms for an asset, or None for an asset that has neither a component nor such a code."""
    labels = (asset.get("records") or {}).get("label_keys") or {}
    alternatives = []
    for component in asset["components"]:
        name = _longest_first([component, labels.get(component, component)])
        level = _longest_first(asset["levels"][component])
        alternatives.append(rf"(?<!{WORD})(?:{name})(?!{WORD})(?:[^\w\n]{{1,4}}[A-Za-z]+)*[^\w\n]{{1,4}}(?:{level})(?!{WORD})(?!\.\d)")
        alternatives.append(rf"(?<!{WORD})levels?\s+(?:{level})\s+(?:of|for|on)\s+(?:the\s+)?(?:{name})(?!{WORD})")
    if leveled_codes(asset):
        alternatives.append(rf"(?<!{WORD})(?:{_longest_first(leveled_codes(asset))})(?!{WORD})")
    return re.compile("|".join(alternatives), re.IGNORECASE) if alternatives else None


def finder(asset, runs=True):
    """A function from a text to the places that state a level of a component of the asset, as sorted (offset, matched text).
    With runs off it finds the three named forms alone."""
    pattern, components = named_level(asset), asset["components"]

    def find(text):
        found = [(match.start(), match.group(0)) for match in (pattern.finditer(text) if pattern else ())]
        if runs:
            found += _runs_in(asset, components, text)
        return sorted(found)

    return find


def _runs_in(asset, components, text):
    """(offset, matched text) of every run of whole numbers that is a full combination of the asset's levels."""
    found = []
    for match in NUMBER_RUN.finditer(text):
        values = [int(value) for value in re.split(r"\s*,\s*", match.group(0))]
        if len(values) == len(components) and all(value in asset["levels"][c] for value, c in zip(values, components)):
            found.append((match.start(), match.group(0)))
    return found


def _kept(path, src):
    return path.is_file() and not SKIPPED_PARTS & set(path.relative_to(src).parts)


def _parses(path):
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return False
    return True


def _decodes(path):
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _agent_paths(src, pattern):
    """The files under the agent roots that match pattern, then the agent texts of the environment that match it."""
    named = [src / name for name in AGENT_TEXT_FILES if (src / name).is_file() and (src / name).match(pattern)]
    return [path for root in AGENT_ROOTS for path in sorted((src / root).rglob(pattern))] + named


def _env_paths(src):
    """The Python files under the environment roots, the agent texts of the environment left out."""
    every = sorted(path for root in ENV_ROOTS for path in (src / root).rglob("*.py"))
    return [path for path in every if path.relative_to(src).as_posix() not in AGENT_TEXT_FILES]


def whole_files(src):
    """The files read in full: under the agent roots every file that decodes as text and is no Python file that parses, under
    the environment roots every Python file that does not parse, the tests folders and the generated tables left out."""
    found = [path for path in _agent_paths(src, "*")
             if _kept(path, src) and path.name not in GENERATED and _decodes(path) and not (path.suffix == ".py" and _parses(path))]
    found += [path for path in _env_paths(src) if _kept(path, src) and not _parses(path)]
    return found


def literal_files(src):
    """(path, docstrings, code) of every Python file that parses: under the agent roots its docstrings are read and its code
    is read for the named forms, under the environment roots neither."""
    return ([(path, True, True) for path in _agent_paths(src, "*.py") if _kept(path, src) and _parses(path)]
            + [(path, False, False) for path in _env_paths(src) if _kept(path, src) and _parses(path)])


def string_literals(source, docstrings=False):
    """(line, text) of every string literal of a Python source, a docstring only when asked. The fixed parts of an f-string
    count one by one, so a part that a value fills at run time is not read."""
    tree = ast.parse(source)
    skipped = set() if docstrings else {id(node.body[0].value) for node in ast.walk(tree)
                                        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
                                        and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                                        and isinstance(node.body[0].value.value, str)}
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skipped]


def code_text(source):
    """The source with every comment blanked, line for line, so that an offset in it falls on the line of the file."""
    lines = source.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            (row, start), (_row, end) = token.start, token.end
            lines[row - 1] = lines[row - 1][:start] + " " * (end - start) + lines[row - 1][end:]
    return "".join(lines)


def _asset_folders(src):
    return {json.loads(path.read_text(encoding="utf-8"))["asset_id"]: path.parent for path in sorted((src / ASSETS).glob("*/asset.json"))}


def accepted(src, asset_id):
    """The entries of the accepted coincidences of an asset, an empty list when its folder holds no such file."""
    path = _asset_folders(src)[asset_id] / ACCEPTED
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _hits(src, asset):
    """(file, line, matched text) of every stated level of one asset in the texts an agent reads."""
    find, named = finder(asset), finder(asset, runs=False)
    found = []
    for path in whole_files(src):
        text = path.read_text(encoding="utf-8")
        found += [(str(path.relative_to(src)), text.count("\n", 0, offset) + 1, matched) for offset, matched in find(text)]
    for path, docstrings, code in literal_files(src):
        source, rel = path.read_text(encoding="utf-8"), str(path.relative_to(src))
        if code:
            text = code_text(source)
            found += [(rel, text.count("\n", 0, offset) + 1, matched) for offset, matched in named(text)]
            found += [(rel, line, matched) for line, literal in string_literals(source, docstrings)
                      for _offset, matched in _runs_in(asset, asset["components"], literal)]
        else:
            found += [(rel, line, matched) for line, literal in string_literals(source, docstrings) for _offset, matched in find(literal)]
    return sorted(found)


def _entry_problem(entry):
    if not isinstance(entry, dict) or set(entry) != set(ENTRY_KEYS):
        return f"an entry needs exactly the keys {', '.join(ENTRY_KEYS)}: {entry!r}"
    if any(entry[key] in ("", None) for key in ENTRY_KEYS):
        return f"an entry has an empty value: {entry!r}"
    if not isinstance(entry["count"], int) or isinstance(entry["count"], bool) or entry["count"] < 1:
        return f"the count of an entry is a whole number of one or more: {entry!r}"
    return None


def scan(src):
    """Every stated level in the texts an agent reads under a source root, each as "<file>:<line> states <text> of <asset>", with
    the hits that the asset's own folder accepts left out, then one line per entry of an accepted file that is malformed or
    whose count is not the number of its hits. The line of a hit in a string literal is the line the literal starts on."""
    found, problems = [], []
    folders = _asset_folders(src)
    for asset in registered_assets(src):
        asset_id = asset["asset_id"]
        hits = _hits(src, asset)
        where = f"{folders[asset_id].relative_to(src)}/{ACCEPTED}"
        dropped = set()
        for entry in accepted(src, asset_id):
            problem = _entry_problem(entry)
            if problem:
                problems.append(f"{where}: {problem}")
                continue
            matching = [hit for hit in hits if (hit[0], hit[2]) == (entry["file"], entry["text"])]
            if len(matching) == entry["count"]:
                dropped.update(matching)
            else:
                problems.append(f"{where}: {entry['text']!r} in {entry['file']} accepts {entry['count']} hit(s), and the scan finds {len(matching)}")
        found += [f"{file}:{line} states {matched!r} of {asset_id}" for file, line, matched in hits if (file, line, matched) not in dropped]
    return found + problems


@pytest.fixture(scope="module")
def assets():
    found = registered_assets(ROOT)
    assert found, "no asset file under CausalDiagnosis/datasets"
    return found


def test_a_planted_level_is_found_in_every_written_form_and_an_innocent_text_passes(assets):
    checked = 0
    for asset in assets:
        components = asset["components"]
        if not components:
            continue
        checked += 1
        find, first = finder(asset), components[0]
        levels = asset["levels"][first]
        level, no_level = levels[0], max(levels) + 1
        stating = [f"submit the diagnosis {first} {level}", f"{first}={level}", f'{{"{first}": {level}}}', f"{first}: {level}", f"{first} at level {level}",
                   f"the {first} is at {level}.", f"{first.upper()} {level}", f"level {level} of the {first}",
                   f"the {first} that the technician of the second visit replaced is now at {level}"]
        label = ((asset.get("records") or {}).get("label_keys") or {}).get(first)
        if label:
            stating.append(f"{label}={level}")
        if leveled_codes(asset):
            stating.append(f"the failure mode {leveled_codes(asset)[0]} is the cause")
        if len(components) > 1:
            stating.append("[" + ", ".join(str(asset["levels"][c][-1]) for c in components) + "]")
        for text in stating:
            assert find(text), (asset["asset_id"], text)
        passing = [f"the {first} and its levels in the profile", f"submit the diagnosis {first} <level> with the evidence", f"{first} {no_level}",
                   f"{first} {level}.5", f"{first}_spare {level}", f"spare{first} {level}",
                   f"the {first} waited {no_level} cycles and was then set to {level}", f"the {first}\n{level} cycles later",
                   "[" + ", ".join(str(asset["levels"][c][-1]) for c in components) + f", {level}]"]
        passing += [f"declare that 2 cycles of the sensor group {group} would separate the candidates" for group in asset.get("observation_kinds") or {}]
        for text in passing:
            assert not find(text), (asset["asset_id"], text, find(text))
    assert checked, "no registered asset has a component"


def test_the_scan_reports_a_planted_level_with_its_file_and_line_and_reads_no_test_comment_or_generated_table(tmp_path):
    src = tmp_path / "fork"
    files = {
        "CausalDiagnosis/datasets/toy/asset.json": json.dumps({"asset_id": "toy", "components": ["fan", "belt"], "levels": {"fan": [0, 2], "belt": [1, 5]},
                                                  "failure_modes": [{"component": "belt", "level": 5, "code": "belt_worn"}]}),
        "CausalDiagnosis/datasets/toy/accepted_texts.json": json.dumps([
            {"file": "CausalDiagnosis/agents/vendor/notes.md", "text": "fan 0", "count": 2, "why": "two numbered steps of a vendored list"},
            {"file": "CausalDiagnosis/agents/loop/prompts.py", "text": "belt 1", "count": 1, "why": "a line that was reworded since"},
            {"file": "CausalDiagnosis/agents/vendor/list.md", "text": "fan 2", "count": 1, "why": "the list had one such item"}]),
        "CausalDiagnosis/agents/loop/prompts.py": 'ROLE = "You plan a repair."\n\nEXAMPLE = "Action Input 1: fan=2, evidence_ids=ev-3"\n',
        "CausalDiagnosis/agents/loop/rules.py": ('def plan(call):\n    call("submit_diagnosis", fan=2)\n    start = {"belt": 5}\n'
                                '    # the fan stays at 2 here, a comment that no agent reads\n    combination = (2, 5)\n'
                                '    return start, combination, "the combination 2, 5"\n'),
        "CausalDiagnosis/agents/loop/guide.prompt": "Diagnose belt 5 when the belt slips.\n",
        "CausalDiagnosis/agents/vendor/notes.md": "Step fan 0 of the list.\nStep fan 0 again.\n",
        "CausalDiagnosis/agents/vendor/list.md": "item fan 2\nitem fan 2\n",
        "CausalDiagnosis/agents/loop/tests/test_loop.py": 'START = {"fan": 2, "belt": 5}\n',
        "CausalDiagnosis/mcp_servers/task/main.py": 'def submit():\n    """Submit a diagnosis, for example belt 5."""\n',
        "CausalDiagnosis/mcp_servers/task/tool_contract.json": json.dumps({"submit": {"description": "a diagnosis such as [2, 5]"}}),
        "CausalDiagnosis/mcp_servers/task/models.json": json.dumps({"fan": 2, "belt": 5}),
        "CausalDiagnosis/environment/toy/env.py": ('def refuse(component, level):\n    """Refuses a level, such as fan 2."""\n    if level is None:\n'
                            '        return "the failure mode belt_worn has no order"\n    return f"{component} {level} is not a level"\n'),
    }
    for name, text in files.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (src / "CausalDiagnosis/mcp_servers/task/blob.bin").write_bytes(b"\xff\xfe belt 5\n")  # bytes that do not decode are no text
    assert scan(src) == ["CausalDiagnosis/agents/loop/guide.prompt:1 states 'belt 5' of toy", "CausalDiagnosis/agents/loop/prompts.py:3 states 'fan=2' of toy",
                         "CausalDiagnosis/agents/loop/rules.py:2 states 'fan=2' of toy", "CausalDiagnosis/agents/loop/rules.py:3 states 'belt\": 5' of toy",
                         "CausalDiagnosis/agents/loop/rules.py:6 states '2, 5' of toy", "CausalDiagnosis/agents/vendor/list.md:1 states 'fan 2' of toy",
                         "CausalDiagnosis/agents/vendor/list.md:2 states 'fan 2' of toy", "CausalDiagnosis/environment/toy/env.py:4 states 'belt_worn' of toy",
                         "CausalDiagnosis/mcp_servers/task/main.py:2 states 'belt 5' of toy", "CausalDiagnosis/mcp_servers/task/tool_contract.json:1 states '2, 5' of toy",
                         "CausalDiagnosis/datasets/toy/accepted_texts.json: 'belt 1' in CausalDiagnosis/agents/loop/prompts.py accepts 1 hit(s), and the scan finds 0",
                         "CausalDiagnosis/datasets/toy/accepted_texts.json: 'fan 2' in CausalDiagnosis/agents/vendor/list.md accepts 1 hit(s), and the scan finds 2"]
    (src / "CausalDiagnosis/datasets/toy/accepted_texts.json").write_text(json.dumps([{"file": "CausalDiagnosis/agents/loop/rules.py", "text": "fan=2", "count": 0, "why": "x"},
                                                                       {"file": "CausalDiagnosis/agents/loop/rules.py", "text": "fan=2"}]), encoding="utf-8")
    assert scan(src)[-2:] == ["CausalDiagnosis/datasets/toy/accepted_texts.json: the count of an entry is a whole number of one or more: "
                              "{'file': 'CausalDiagnosis/agents/loop/rules.py', 'text': 'fan=2', 'count': 0, 'why': 'x'}",
                              "CausalDiagnosis/datasets/toy/accepted_texts.json: an entry needs exactly the keys file, text, count, why: "
                              "{'file': 'CausalDiagnosis/agents/loop/rules.py', 'text': 'fan=2'}"]


def test_no_text_an_agent_reads_states_a_level_of_a_component_of_a_registered_asset(assets):
    read_in_full = {str(path.relative_to(ROOT)) for path in whole_files(ROOT)}
    literals_of = {str(path.relative_to(ROOT)): (docstrings, code) for path, docstrings, code in literal_files(ROOT)}
    # the scan cannot pass by reading nothing: it holds the prompts of the CausalDiagnosis agent, the examples of the ported baselines,
    # both briefings and the tool texts of the CausalDiagnosis MCP with their code, both tool contracts in full, and the environment's own answers
    for name in ("CausalDiagnosis/agents/Causal_Diagnosis_Agent/prompts.py", "CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/in_context_examples.py",
                 "CausalDiagnosis/environment/Causal_Diagnosis_Environment/briefing.py", "CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_briefing.py",
                 "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/main.py"):
        assert literals_of.get(name) == (True, True), name
    for name in ("CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json", "CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.json"):
        assert name in read_in_full, name
    assert literals_of.get("CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py") == (False, False) and not read_in_full & set(literals_of)
    assert any(named_level(asset) is not None for asset in assets)
    found = scan(ROOT)
    assert not found, "\n".join(found)
