"""src/agent/__init__.py imports every SDK runner with its packages, which the slim test environment lacks, so when that
import fails the ``agent`` package is registered by its path with its __init__ skipped, as the conftest of
CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/tests does and for the same reasons. This folder has no __init__.py on purpose."""
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
    for name, path in (("agent", AGENT), ("src.agent", AGENT), ("src.agent.plan_execute_paper", PACKAGE)):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
