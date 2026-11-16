"""The command table of CausalDiagnosis/__main__.py: every program a person starts has a command, every command names a
program, the table imports nothing before a command is chosen, and a program started through the table sees the same
command line as one started with python -m <module>."""
import ast
import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from CausalDiagnosis import __main__ as table
from CausalDiagnosis.utils.paths import PACKAGE, ROOT, SRC

# Programs with a main that no person starts, each with its reason.
LEFT_OUT = {
    "mcp_servers/Causal_Diagnosis_MCP/main.py": "the server of an episode, started by the launcher beside its state file",
    "mcp_servers/Environment_Actions_MCP/main.py": "the server of an episode, started by the launcher beside its state file",
}
FRAMEWORKS = "frameworks"  # a framework's programs reach the table through its declaration


def _env():
    return {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(SRC))), "PYTHONDONTWRITEBYTECODE": "1"}


def _run(*argv):
    return subprocess.run([sys.executable, *argv], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=300)


def _has_main_guard(path):
    with warnings.catch_warnings():  # escape sequences of other files are not this test's subject
        warnings.simplefilter("ignore", SyntaxWarning)
        body = ast.parse(Path(path).read_text()).body
    return any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) for node in body)


def _table_modules():
    modules = {table.TABLES}
    for commands in table.COMMANDS.values():
        modules |= {table.P + module for module, _ in commands.values()}
    return modules


def _module_file(module):
    spec = importlib.util.find_spec(module)
    return spec.origin if spec.submodule_search_locations is None else os.path.join(spec.submodule_search_locations[0], "__main__.py")


def test_every_command_names_a_program_with_a_main():
    missing = []
    for module in _table_modules():
        module = module.replace("<asset_id>", "hyd_1")
        if importlib.util.find_spec(module) is None or not _has_main_guard(_module_file(module)):
            missing.append(module)
    assert missing == []


def test_every_command_of_a_framework_names_a_program_with_a_main():
    declared = table.framework_commands()
    missing = [module for commands in declared.values() for module in commands.values()
               if importlib.util.find_spec(module) is None or not _has_main_guard(_module_file(module))]
    assert declared and missing == []


def test_every_program_with_a_main_has_a_command_or_a_reason():
    named = {module.replace(".", "/") for module in _table_modules()}
    loose = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        if "tests" in rel.parts or rel.parts[0] == FRAMEWORKS or rel.name == "__main__.py" and rel.parent == PACKAGE.relative_to(PACKAGE):
            continue
        if not _has_main_guard(path) or str(rel) in LEFT_OUT:
            continue
        stem = "CausalDiagnosis/" + str(rel.with_suffix("")).removesuffix("/__main__")
        dataset_program = rel.parts[0] == "datasets" and len(rel.parts) == 3 and rel.name == "download_and_build.py"
        if stem not in named and not dataset_program:
            loose.append(str(rel))
    assert loose == []


def test_the_table_imports_no_module_of_the_package_before_a_command_is_chosen():
    done = _run("-c", "import sys, CausalDiagnosis.__main__; "
                      "print(sorted(m for m in sys.modules if m.startswith('CausalDiagnosis')))")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "['CausalDiagnosis', 'CausalDiagnosis.__main__']"


def test_a_program_sees_the_same_command_line_through_the_table_as_with_python_m():
    through_table = _run("-m", "CausalDiagnosis", "experiments", "run-ledger", "--help")
    direct = _run("-m", "CausalDiagnosis.experiments.run_verification.run_ledger", "--help")
    assert through_table.returncode == direct.returncode == 0, through_table.stderr
    assert through_table.stdout == direct.stdout


def test_a_command_passes_its_arguments_and_exit_status():
    assert _run("-m", "CausalDiagnosis", "utils", "paths", "GRIDS").stdout == _run("-m", "CausalDiagnosis.utils.paths", "GRIDS").stdout
    assert _run("-m", "CausalDiagnosis", "utils", "paths").returncode != 0


@pytest.mark.parametrize("argv, expected", [
    (["tables", "list", "--runs", "r"], (table.TABLES, ["--list", "--runs", "r"])),
    (["tables", "all", "--runs", "r"], (table.TABLES, ["--runs", "r"])),
    (["tables", "agents", "--out", "o"], (table.TABLES, ["--table", "agents", "--out", "o"])),
    (["datasets", "download", "fan_1", "--download"], ("CausalDiagnosis.datasets.fan_1.download_and_build", ["--download"])),
    (["planted", "say", "x"], ("CausalDiagnosis.utils.paths", ["x"])),
])
def test_the_table_dispatches_to_the_module(monkeypatch, argv, expected):
    started = []
    monkeypatch.setattr(table, "start", lambda module, rest: started.append((module, rest)) or 0)
    monkeypatch.setattr(table, "framework_commands", lambda: {"planted": {"say": "CausalDiagnosis.utils.paths"}})
    assert table.main(argv) == 0 and started == [expected]


@pytest.mark.parametrize("argv", [[], ["nogroup"], ["experiments", "nocommand"], ["datasets", "download"]])
def test_an_unknown_or_missing_name_lists_the_choices_with_status_2(monkeypatch, capsys, argv):
    monkeypatch.setattr(table, "framework_commands", lambda: {})
    assert table.main(argv) == 2
    assert "python -m CausalDiagnosis" in capsys.readouterr().err


def test_a_group_lists_every_command_with_its_line(monkeypatch, capsys):
    monkeypatch.setattr(table, "framework_commands", lambda: {"planted": {"say": "CausalDiagnosis.utils.paths"}})
    assert table.main(["experiments", "--help"]) == 0
    out = capsys.readouterr().out
    assert all(f"  {name} " in out for name in table.COMMANDS["experiments"])
    assert table.main(["planted"]) == 2
    assert "The locations of the repository" in capsys.readouterr().err
