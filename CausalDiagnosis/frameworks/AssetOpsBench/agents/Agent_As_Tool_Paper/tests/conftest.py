"""src/agent/__init__.py imports every SDK runner with its packages, which the slim test environment lacks, so when
that import fails the ``agent`` package is registered by its path with its __init__ skipped, as
CausalDiagnosis/experiments/episode_runners/run_episodes/ does. pytest also imports the __init__ of every package above a test file
during setup, under the names it derives from the repository root (``src.agent`` and ``src.agent.meta_agent``,
since src/__init__.py exists), so those two names get path registered placeholders in the same case, since otherwise pytest
would execute src/agent/__init__.py again under the other name. In an environment with every SDK the real package
imports and nothing is replaced. This folder has no __init__.py on purpose, so that pytest imports this file as a top
level module before any ``agent`` import, instead of as ``src.agent.meta_agent.tests.conftest``."""
from CausalDiagnosis.utils.paths import SRC
import importlib
import sys
import types
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
AGENT = SRC / "agent"

try:
    importlib.import_module("agent")
except ImportError:
    for name, path in (("agent", AGENT), ("src.agent", AGENT), ("src.agent.meta_agent", PACKAGE)):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
