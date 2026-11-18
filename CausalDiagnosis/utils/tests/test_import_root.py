"""The import guard of the package: a process with the package folder on its path stops at the first import, and one
process that imports every module of the package holds no file of the fork under two names. A launcher module is left out: it is the
template copied beside the state file of an episode, and importing it starts the server of that episode."""
import json
import os
import subprocess
import sys

from CausalDiagnosis.experiments.table_definitions.tests.test_code_lists import _fork_files
from CausalDiagnosis.utils.paths import PACKAGE, ROOT, SRC

EVERY_MODULE = (
    "import collections, importlib, json, os, pkgutil, sys\n"
    "import CausalDiagnosis\n"
    "for info in pkgutil.walk_packages(CausalDiagnosis.__path__, 'CausalDiagnosis.'):\n"
    "    if '.tests' not in info.name and not info.name.endswith('.launcher'):\n"
    "        importlib.import_module(info.name)\n"
    "root = os.path.dirname(os.path.dirname(os.path.realpath(CausalDiagnosis.__file__))) + os.sep\n"
    "names = collections.defaultdict(list)\n"
    "for name, module in list(sys.modules.items()):\n"
    "    path = getattr(module, '__file__', None)\n"
    "    if path and os.path.realpath(path).startswith(root):\n"
    "        names[os.path.relpath(os.path.realpath(path), root)].append(name)\n"
    "print(json.dumps({path: sorted(found) for path, found in sorted(names.items()) if len(found) > 1}))\n"
)


def _env(*entries):
    return {**os.environ, "PYTHONPATH": os.pathsep.join(str(entry) for entry in entries), "PYTHONDONTWRITEBYTECODE": "1"}


def test_a_process_with_the_package_folder_on_its_path_stops_at_the_first_import(tmp_path):
    done = subprocess.run([sys.executable, "-c", "import CausalDiagnosis"], cwd=tmp_path, env=_env(PACKAGE, ROOT, SRC),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "stands on the import path" in done.stderr


def test_no_module_of_the_package_is_loaded_under_two_names(tmp_path):
    """Every file under the root that the process loaded under two names fails, except a file of the interpreter's own
    environment outside the listed roots (the rule of the code lists), which a package of that environment may register under
    a second name of its own."""
    assert _fork_files(["CausalDiagnosis/utils/paths.py"]) == ["CausalDiagnosis/utils/paths.py"]
    done = subprocess.run([sys.executable, "-c", EVERY_MODULE], cwd=tmp_path, env=_env(ROOT, SRC),
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    doubled = json.loads(done.stdout)
    assert {path: doubled[path] for path in _fork_files(sorted(doubled))} == {}
