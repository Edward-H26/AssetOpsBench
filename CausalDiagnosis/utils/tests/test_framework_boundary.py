"""The boundary between the core and the frameworks (frameworks/README.md, utils/frameworks.py): no module of the core imports a
module of a framework or names one in its code, so a framework is reached by name through its declaration alone. Docstrings and
comments may name a framework's files for a reader; a string the code uses may not. Tests are exempt."""
import ast

from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.utils.paths import PACKAGE


def _core_modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative.startswith("frameworks/") or "/tests/" in f"/{relative}" or path.name.startswith("test_") or path.name == "conftest.py":
            continue
        yield relative, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _docstrings(tree):
    nodes = [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))]
    return {id(n.body[0].value) for n in nodes if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}


def test_no_core_module_imports_or_names_a_module_of_a_framework():
    names = frameworks.frameworks()
    assert names, "no framework is declared, so the test would pass on nothing"
    dotted = tuple(f"CausalDiagnosis.frameworks.{name}" for name in names)
    slashed = tuple(f"frameworks/{name}/" for name in names)
    found = []
    for relative, tree in _core_modules():
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [f"{relative}:{node.lineno} imports {a.name}" for a in node.names if a.name.startswith(dotted)]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(dotted):
                found.append(f"{relative}:{node.lineno} imports {node.module}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if any(d in node.value for d in dotted) or any(s in node.value for s in slashed):
                    found.append(f"{relative}:{node.lineno} names {node.value[:80]!r}")
    assert found == []


def test_the_code_list_of_a_rescoring_names_the_judge_of_every_declared_judge_from_its_declaration():
    """rescore_under_verifier_scope.CODE hashes the module of each declared judge, read from the declaration's entry, so the core
    names no file of a framework."""
    from CausalDiagnosis.experiments.run_verification import rescore_under_verifier_scope as rescore

    judges = [entry["entry"].partition(":")[0].replace(".", "/") + ".py" for entry in frameworks.judges().values() if "entry" in entry]
    assert judges and all(path in rescore.CODE for path in judges)
    assert all((PACKAGE.parent / path).is_file() for path in rescore.CODE)


def test_the_help_of_the_judge_option_names_no_file_of_a_framework():
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    files = {entry["entry"].partition(":")[0].rpartition(".")[2] + ".py" for entry in frameworks.judges().values() if "entry" in entry}
    help_text = next(action.help for action in run_episodes.build_parser()._actions if action.dest == "judge")
    assert files and not any(name in help_text for name in files)


def test_the_start_page_of_an_episode_names_the_metering_layer_of_its_surface_from_the_declaration():
    """walkthrough.gateway_folder reads the folder of the metering layer from the gateway that the surface of the called servers
    declares, so the core page names no folder of a framework."""
    from types import SimpleNamespace

    from CausalDiagnosis.episode_viewer import walkthrough

    gated = [entry for entry in frameworks.surfaces().values() if "gateway" in entry]
    assert gated
    for entry in gated:
        folder = walkthrough.gateway_folder(SimpleNamespace(calls=[SimpleNamespace(server=entry["servers"][0])]))
        assert entry["gateway"].partition(":")[0].replace(".", "/").startswith(folder) and (PACKAGE.parent / folder).is_dir()
