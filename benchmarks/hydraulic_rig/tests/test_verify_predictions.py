"""Study A on two records of S1: every counterfactual replacement gives one row, the faulty replacement recovers the rig
and ends the episode, the healthy ones continue to a verified reading of the 17 channel means."""
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, default_facts_path
from verify_predictions import study, summarise

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)


def test_rows_cover_every_component_of_every_start_state_and_follow_the_recordings(tmp_path):
    rows = study(["S1"], per_record=1, max_records=2, work_dir=tmp_path)
    assert len(rows) == 2 * len(COMPONENTS)
    assert [r["replaced"] for r in rows[:4]] == list(COMPONENTS)
    faulty = [r for r in rows if r["truly_faulty"]]
    healthy = [r for r in rows if not r["truly_faulty"]]
    assert len(faulty) == 2 and len(healthy) == 6  # one faulty component per S1 start state
    for r in faulty:
        assert r["recovered_actual"] and r["episode_over_after_replace"]
        assert r["verify_status"].startswith("not yet observable") and r["n_inside"] + r["n_outside"] == 0
        assert 0 <= r["pred_addresses_fault"] <= 1 and 0 <= r["pred_recovery"] <= 1
    for r in healthy:
        assert not r["recovered_actual"] and not r["episode_over_after_replace"]
        assert r["verify_status"] == "verified" and r["n_inside"] + r["n_outside"] == 17
        assert 0 <= r["pred_addresses_fault"] <= 1
        assert r["n_outside_tolerated"] == 1 and r["consistent"] == (r["n_outside"] <= 1 and r["pred_recovery"] < 0.5)
    assert all(r["prediction_id"] == f"pred/{COMPONENTS.index(r['replaced']) + 1}" for r in rows)
    assert all(0 <= r["top_probability"] <= 1 and r["cost"] >= 54 for r in rows)


def test_summary_reports_each_scenario(tmp_path):
    rows = study(["S1"], per_record=1, max_records=1, work_dir=tmp_path)
    text = summarise(rows)
    assert "## S1: 1 start states of 1 records, 4 replacements" in text
    assert "| replacements with a verified reading | 3 of 4 |" in text
    assert "observed means inside the predicted band, healthy component replaced, same recordings continue (3)" in text
    assert "verifications the tool calls consistent, at most one mean outside, healthy component replaced" in text
