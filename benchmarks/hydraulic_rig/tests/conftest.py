"""The drivers import the fork's src and their own folder by path; the tests run from the fork root with the same paths.
The models of the causal-diagnosis server are generated, not committed, so they are fitted once before the tests."""
import sys
from pathlib import Path

DRIVERS = Path(__file__).resolve().parents[1]
SRC = DRIVERS.parents[1] / "src"
for path in (str(SRC), str(DRIVERS)):
    if path not in sys.path:
        sys.path.insert(0, path)


import pytest  # noqa: E402

from envs.hydraulic_rig.data import default_facts_path  # noqa: E402
from servers.causal_diagnosis.fit import ensure_models  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def fitted_models():
    if Path(default_facts_path()).exists():
        ensure_models()
