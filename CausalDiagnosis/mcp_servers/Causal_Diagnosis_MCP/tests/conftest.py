"""The models of the diagnostics server are generated from the rig data, not committed, so they are fitted once before
the tests when the data file is present."""
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import ensure_models


@pytest.fixture(scope="session", autouse=True)
def fitted_models():
    if Path(default_facts_path()).exists():
        ensure_models()
