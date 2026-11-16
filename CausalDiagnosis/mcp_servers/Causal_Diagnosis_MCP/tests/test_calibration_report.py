"""The calibration report on a tiny configuration: it is written where asked, lists the protocol and the tables for every
ranking model over the combinations of its start set, and the healthy block under the verdict and the intervals per sensor
group, and states no value in this test."""
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import calibration_report as calibration
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import RANKING_MODELS

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)


def test_report_is_written_with_every_section(tmp_path):
    path = calibration.run(out_dir=tmp_path, readings=2, rollouts=1, seed=0)
    text = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path and path.suffix == ".md"
    for heading in ("## Protocol", "## Brier score", "## Share of rollouts whose top combination", "## Conformal coverage",
                    "## The healthy block under the verdict of E17", "## Calibration curve"):
        assert heading in text
    assert all(m in text for m in RANKING_MODELS) and "Nothing in it enters code, tests or docs" in text
    assert "| 1 |" in text and "| 2 |" in text
