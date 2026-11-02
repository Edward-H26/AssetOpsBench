"""The calibration report of section 5.6 on a tiny configuration: it is written where asked, lists the protocol and the
tables for every ranking model, and states no value in this test."""
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import default_facts_path
from servers.causal_diagnosis import calibration
from servers.causal_diagnosis.posterior import RANKING_MODELS

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)


def test_report_is_written_with_every_section(tmp_path):
    path = calibration.run(out_dir=tmp_path, readings=2, rollouts=1, seed=0)
    text = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path and path.suffix == ".md"
    for heading in ("## Protocol", "## Brier score", "## Share of rollouts whose top candidate", "## Conformal coverage", "## Calibration curve"):
        assert heading in text
    assert all(m in text for m in RANKING_MODELS) and "Nothing in it enters code, tests or docs" in text
    assert "| 1 |" in text and "| 2 |" in text
