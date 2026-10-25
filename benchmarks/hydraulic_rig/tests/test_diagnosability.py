"""Study B on two records of S1: four rungs per start state with the ladder's readings and costs, one grade per state."""
from pathlib import Path

import pytest

from common import record_sizes_line, record_weighted_mean
from diagnosability import BLOCK, INSEPARABLE, LADDER, study, summarise
from envs.hydraulic_rig.data import default_facts_path
from servers.hydraulic_rig import episode

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)


def test_rungs_readings_costs_and_grades(tmp_path):
    rows = study(["S1"], per_record=1, max_records=2, work_dir=tmp_path)
    rungs = [name for name, _ in LADDER]
    assert len(rows) == 2 * len(rungs)
    for state_rows in (rows[:4], rows[4:]):
        assert [r["rung"] for r in state_rows] == rungs
        assert [r["readings"] for r in state_rows] == [1, 2, 3, 30]
        assert [r["cost"] for r in state_rows] == [1, 4, 14, 140]
        assert len({r["grade"] for r in state_rows}) == 1 and state_rows[0]["grade"] in {*rungs, INSEPARABLE}
        graded = state_rows[0]["grade"]
        if graded != INSEPARABLE:
            first = next(r for r in state_rows if r["rung"] == graded)
            assert first["top_is_truth"] and not first["insufficient_evidence"]
        assert all(0 <= r["truth_probability"] <= 1 and 0 <= r["top_probability"] <= 1 for r in state_rows)
    text = summarise(rows)
    assert "## S1: 2 start states of 2 records" in text and "| tiers 1 to 3, ten readings each | 30 | 140 |" in text
    assert "| grade | start states | share | share, records weighted equally |" in text and "Every record holds 1 start states." in text


def test_the_last_rung_reads_every_cycle_of_the_window_at_every_tier(tmp_path):
    study(["S1"], per_record=1, max_records=1, work_dir=tmp_path)
    (state_path,) = tmp_path.rglob("episode.json")
    log = episode.replay(episode.load_state(state_path)).log
    readings = [(e["tier"], e["cycle"]) for e in log if e["event"] == "measure"]
    window = {cycle for _, cycle in readings}
    assert len(window) == BLOCK and sorted(readings) == sorted((tier, cycle) for tier in (1, 2, 3) for cycle in window)


def test_record_weighted_mean_weights_records_not_rows():
    rows = [{"record": "a", "v": 1}, {"record": "a", "v": 1}, {"record": "a", "v": 1}, {"record": "b", "v": 0}]
    assert record_weighted_mean(rows, lambda r: r["v"]) == 0.5 and record_weighted_mean([], lambda r: 1) is None
    assert record_sizes_line(["a", "a", "b"]) == "Records hold 1 to 2 start states; the largest, a, holds 2 of the 3."
