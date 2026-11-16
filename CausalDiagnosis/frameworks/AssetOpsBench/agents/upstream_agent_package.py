"""Registers upstream's agent package (src/agent) by its path without running its __init__, which imports every SDK runner
with its packages (claude_agent, deep_agent, direct_llm_agent, openai_agent) that the modules of this framework do not need.
Every module of this framework that imports agent.* imports this module first. No bytecode is written, so the tree stays as
checked out.
"""
import sys
import types

from CausalDiagnosis.utils.paths import SRC


def register():
    sys.dont_write_bytecode = True
    package = types.ModuleType("agent")
    package.__path__ = [str(SRC / "agent")]
    sys.modules.setdefault("agent", package)


register()
