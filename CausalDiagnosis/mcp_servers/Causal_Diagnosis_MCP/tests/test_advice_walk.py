"""The walk that follows the advice with no model (advice_walk.py), on a handful of development combinations of the base
preset and of the preset with prices: what the rule implies whatever the data, one read of the record and the diagnosis
of rank 1."""
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests import advice_walk as aw

pytestmark = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
EVERY = 24  # every k-th development combination in grid order, a handful of walks


@pytest.mark.parametrize("preset", ["base", "priced"])
@pytest.mark.parametrize("index", range(0, 96, EVERY))
def test_the_walk_reads_the_record_once_diagnoses_rank_1_and_passes_when_rank_1_is_the_start(index, preset):
    starts = aw.development_combinations()
    start = starts[index % len(starts)]
    view = profile_view()
    log, verdict, replies = aw.walk(start, preset=preset)
    assert all(r.advice is not None and r.advice.action != "declare" for r in replies)
    reads = [e for e in log if e["event"] == "read"]
    diagnosis = next(i for i, e in enumerate(log) if e["event"] == "submit_diagnosis")
    assert len(reads) == 1 and all(log.index(r) < diagnosis for r in reads)
    first = replies[-1].candidates[0].combination
    assert log[diagnosis]["diagnosis"] == dict(first)
    ordered = sorted(e["component"] for e in log if e["event"] == "work_order_created")
    assert ordered == sorted(c for c in view.components if first[c] != view.healthy[c])
    assert verdict["checks"]["count_gate"]["ok"] and verdict["checks"]["causality"]["ok"]
    assert verdict["pass"] == (tuple(first[c] for c in view.components) == tuple(start))
