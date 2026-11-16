"""CausalDiagnosis: an industrial environment for language-model agents with its datasets, tool servers, agents,
experiments and evaluation. Programs start as python -m CausalDiagnosis.<module> from the repository root. The package
refuses to load when its own folder stands on the import path, where its folders would shadow the packages of the same
names (agents, datasets, evaluation)."""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.realpath(__file__))
if any(_os.path.realpath(_entry or _os.getcwd()) == _HERE for _entry in _sys.path):
    raise ImportError(f"{_HERE} stands on the import path; start the program as python -m CausalDiagnosis.<module> "
                      "from the repository root")
