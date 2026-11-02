"""The pilot files: scenarios_17.json against its source, the date to cycle arithmetic, and the anchored start on
synthetic cycles and on the facts file (relations only, no recorded value)."""
import json
from datetime import date
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, Cycle, HEALTHY, build_blocks, default_facts_path, load_cycles
from pilot import anchors
from pilot.build_scenarios import build, scenario_date

CLONE = Path(__file__).resolve().parents[4] / "hf_ibm-research_AssetOpsBench_clone" / "AssetOpsBench"


def test_scenarios_17_holds_the_dataset_records_with_their_dates():
    data = anchors.load_scenarios()
    scenarios = data["scenarios"]
    assert len(scenarios) == data["source"]["records"] == 17
    assert len(data["dates"]) == 10 and data["dates"] == sorted({s["date"] for s in scenarios})
    assert len({s["id"] for s in scenarios}) == 17
    for s in scenarios:
        assert s["date"] in s["text"] and s["date"] in s["failure_event"]
        assert s["asset_component"] in COMPONENTS and "hp_1" in s["text"]
        date.fromisoformat(s["date"])
    assert data["source"]["dataset"] == "ibm-research/AssetOpsBench" and len(data["source"]["sha256"]) == 64
    assert len(data["assumptions"]) == 2 and all("hp_1" in a or "ten dates" in a for a in data["assumptions"])


@pytest.mark.skipif(not CLONE.exists(), reason="the Hugging Face dataset clone is not checked out beside the fork")
def test_scenarios_17_equals_a_rebuild_from_the_clone():
    rebuilt = build(CLONE)
    stored = anchors.load_scenarios()
    assert rebuilt["scenarios"] == stored["scenarios"]
    assert rebuilt["source"]["sha256"] == stored["source"]["sha256"]
    assert rebuilt["dates"] == stored["dates"]


def test_scenario_date_needs_one_date_shared_by_text_and_event():
    assert scenario_date({"id": 1, "text": "on 2023-01-28", "failure_event": "at date 2023-01-28"}) == "2023-01-28"
    with pytest.raises(ValueError):
        scenario_date({"id": 1, "text": "2023-01-28 and 2023-01-29", "failure_event": "2023-01-28"})
    with pytest.raises(ValueError):
        scenario_date({"id": 1, "text": "2023-01-28", "failure_event": "2023-02-28"})


def test_cycle_of_counts_days_from_the_first_recorded_day():
    assert anchors.cycle_of("2022-07-01") == 0
    assert anchors.cycle_of("2022-07-02") == 1
    assert anchors.cycle_of("2023-01-28") == (date(2023, 1, 28) - date(2022, 7, 1)).days
    assert anchors.cycle_of("2022-08-01", date0="2022-07-31") == 1
    with pytest.raises(ValueError):
        anchors.cycle_of("2022-06-30")


def _synthetic_cycles():
    healthy = tuple(HEALTHY[c] for c in COMPONENTS)
    faulty = (3, 100, 0, 130)
    combos = [healthy, healthy, healthy, faulty, faulty, faulty, faulty, healthy]
    stable = [True, True, True, True, False, True, True, True]
    return [Cycle(i, combo, s, {}) for i, (combo, s) in enumerate(zip(combos, stable))]


def test_anchor_on_synthetic_cycles_gives_the_block_offset_of_the_day():
    cycles = _synthetic_cycles()
    blocks = {combo: [c.index for c in cycles if c.combo == combo and c.stable] for combo in {c.combo for c in cycles}}
    third_day = anchors.anchor("2022-07-04", cycles, blocks)
    assert third_day["cycle_index"] == 3 and third_day["combination"] == [3, 100, 0, 130] and third_day["n_faulty"] == 1
    assert third_day["stable"] and third_day["block_offset"] == 0 and third_day["block_length"] == 3
    unstable_day = anchors.anchor("2022-07-05", cycles, blocks)
    assert not unstable_day["stable"] and unstable_day["cycle_index"] == 4
    assert unstable_day["anchored_cycle_index"] == 3 and unstable_day["block_offset"] == 0
    assert anchors.start_of(third_day) == {"combination": [3, 100, 0, 130], "block_offset": 0}
    with pytest.raises(ValueError):
        anchors.anchor("2022-07-20", cycles, blocks)


@pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
def test_every_pilot_date_anchors_inside_a_stable_block_of_the_facts_file():
    cycles = load_cycles(str(default_facts_path()))
    blocks = build_blocks(cycles)
    by_index = {c.index: c for c in cycles}
    for day in anchors.load_scenarios()["dates"]:
        a = anchors.anchor(day, cycles, blocks)
        block = blocks[tuple(a["combination"])]
        assert block[a["block_offset"]] == a["anchored_cycle_index"] and a["block_length"] == len(block)
        assert by_index[a["anchored_cycle_index"]].combo == tuple(a["combination"]) and by_index[a["anchored_cycle_index"]].stable
        assert a["stable"] == (a["anchored_cycle_index"] == a["cycle_index"])
        assert 0 <= a["n_faulty"] <= len(COMPONENTS)


def test_assumptions_text_prints_both_assumptions():
    text = anchors.assumptions_text()
    assert text.count("\n- ") == 1 and text.startswith("- A1") and "A2" in text
    assert json.dumps(anchors.ASSUMPTIONS)
