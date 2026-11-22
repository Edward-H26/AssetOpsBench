"""The scenario anchors: the driver and the anchors import without the fetched
file, the fetch refuses bytes that differ from its pin and then writes nothing, the date to cycle arithmetic, and the anchored
start on synthetic cycles and, once the scenarios are fetched, on the facts file (relations only, no recorded value)."""
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import scenario_anchors as anchors, download_scenarios as fetch_scenarios
from CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios import scenario_date
from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, Cycle, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

from CausalDiagnosis.utils.paths import ROOT  # the root of the tree under test
FETCHED = anchors.scenarios_path().is_file()
NOT_FETCHED = f"the scenarios are not fetched, {anchors.scenarios_path()} is missing; run {anchors.FETCH_COMMAND}"


def test_the_driver_and_the_anchors_import_without_the_scenario_file(tmp_path):
    paths = [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, anchors.SCENARIOS_ENV: str(tmp_path / "absent.json"), "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
    code = ("from CausalDiagnosis.experiments.episode_runners import run_episodes\n"
            "from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import scenario_anchors as anchors\n"
            "print(len(anchors.ASSUMPTIONS), anchors.scenarios_path().exists())")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and done.stdout.split() == ["2", "False"], done.stderr[-3000:]


def test_a_missing_scenario_file_names_the_command(tmp_path, monkeypatch):
    monkeypatch.setenv(anchors.SCENARIOS_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match=re.escape(anchors.FETCH_COMMAND)):
        anchors.load_scenarios()
    with pytest.raises(FileNotFoundError, match=re.escape(anchors.FETCH_COMMAND)):
        run_episodes.episode_plan(EpisodeConfig(anchors={"date0": anchors.DATE0}), [0], [], {})


def _made_up(n):
    """n made-up records with the keys that the fetch reads, each naming one day in its text and its failure event."""
    return [{"id": 900 + i, "text": f"Made-up scenario {i} of hp_1 on 2022-09-{10 + i:02d}.",
             "failure_event": f"made-up fault at 2022-09-{10 + i:02d}"} for i in range(n)]


def _jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _opener(raw, calls):
    """A stand-in of urlopen that serves raw and records the address of every request, so that no test opens the network."""
    def opener(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(raw)
    return opener


def _pin(monkeypatch, raw, records):
    """The pin of the fetch set to the hash of raw and to a count of records."""
    monkeypatch.setattr(fetch_scenarios, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(fetch_scenarios, "RECORDS", records)


def test_the_fetch_refuses_a_file_with_another_checksum_or_another_count_and_writes_nothing(tmp_path, monkeypatch):
    out, raw = tmp_path / "scenarios.json", _jsonl(_made_up(3))
    with pytest.raises(ValueError) as refused:
        fetch_scenarios.main(["--out", str(out)], opener=_opener(raw, []))
    assert fetch_scenarios.SOURCE_SHA256 in str(refused.value) and hashlib.sha256(raw).hexdigest() in str(refused.value)
    assert not out.exists()
    _pin(monkeypatch, raw, 4)
    with pytest.raises(ValueError, match="holds 3 records"):
        fetch_scenarios.main(["--out", str(out)], opener=_opener(raw, []))
    assert not out.exists()


def test_the_fetch_builds_the_file_from_verified_bytes(tmp_path, monkeypatch):
    records = _made_up(3)[::-1]
    raw = _jsonl(records)
    _pin(monkeypatch, raw, 3)
    calls = []
    out = fetch_scenarios.main(["--out", str(tmp_path / "fetched.json")], opener=_opener(raw, calls))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert calls == [fetch_scenarios.source_url()] and fetch_scenarios.REVISION in calls[0]
    assert data["source"] == fetch_scenarios.source_block() and data["source"]["records"] == 3
    assert [s["id"] for s in data["scenarios"]] == sorted(r["id"] for r in records)
    assert data["dates"] == sorted({s["date"] for s in data["scenarios"]}) and all(s["date"] in s["text"] for s in data["scenarios"])
    # a file of a local clone, under the same checks, into the default path of the anchors
    source = tmp_path / "clone" / fetch_scenarios.SOURCE_FILE
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    monkeypatch.setenv(anchors.SCENARIOS_ENV, str(tmp_path / "default.json"))
    assert fetch_scenarios.main(["--from-file", str(source)], opener=_opener(raw, calls)) == tmp_path / "default.json"
    assert calls == [fetch_scenarios.source_url()] and anchors.load_scenarios() == data


def test_a_scenario_file_of_another_revision_is_refused(tmp_path, monkeypatch):
    raw = _jsonl(_made_up(2))
    _pin(monkeypatch, raw, 2)
    path = fetch_scenarios.main(["--out", str(tmp_path / "scenarios.json")], opener=_opener(raw, []))
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "source": {**data["source"], "revision": "0" * 40}}), encoding="utf-8")
    with pytest.raises(ValueError, match="download_scenarios.py pins"):
        anchors.load_scenarios(path)


def test_scenario_date_needs_one_date_shared_by_text_and_event():
    assert scenario_date({"id": 1, "text": "on 2022-09-15", "failure_event": "at date 2022-09-15"}) == "2022-09-15"
    with pytest.raises(ValueError):
        scenario_date({"id": 1, "text": "2022-09-15 and 2022-09-16", "failure_event": "2022-09-15"})
    with pytest.raises(ValueError):
        scenario_date({"id": 1, "text": "2022-09-15", "failure_event": "2022-10-15"})


def test_cycle_of_counts_days_from_the_first_recorded_day():
    assert anchors.cycle_of("2022-07-01") == 0
    assert anchors.cycle_of("2022-07-02") == 1
    assert anchors.cycle_of("2022-09-15") == (date(2022, 9, 15) - date(2022, 7, 1)).days
    assert anchors.cycle_of("2022-08-01", date0="2022-07-31") == 1
    with pytest.raises(ValueError):
        anchors.cycle_of("2022-06-30")


def _synthetic_cycles():
    healthy = tuple(HEALTHY[c] for c in COMPONENTS)
    faulty = (3, 100, 0, 130)
    combos = [healthy, healthy, healthy, faulty, faulty, faulty, faulty, healthy]
    stable = [True, True, True, True, False, True, True, True]
    return [Cycle(i, combo, s, {}) for i, (combo, s) in enumerate(zip(combos, stable))]


def test_anchor_on_synthetic_cycles_gives_the_block_position_of_the_day():
    cycles = _synthetic_cycles()
    blocks = {combo: [c.index for c in cycles if c.combo == combo and c.stable] for combo in {c.combo for c in cycles}}
    third_day = anchors.anchor("2022-07-04", cycles, blocks)
    assert third_day["cycle_index"] == 3 and third_day["combination"] == [3, 100, 0, 130] and third_day["n_faulty"] == 1
    assert third_day["stable"] and third_day["block_position"] == 0 and third_day["block_length"] == 3
    unstable_day = anchors.anchor("2022-07-05", cycles, blocks)
    assert not unstable_day["stable"] and unstable_day["cycle_index"] == 4
    assert unstable_day["anchored_cycle_index"] == 3 and unstable_day["block_position"] == 0
    # the named recording opens the live segment, half a block of positions after the entry offset
    assert anchors.start_of(third_day) == {"combination": [3, 100, 0, 130], "entry_offset": SEGMENT}
    with pytest.raises(ValueError):
        anchors.anchor("2022-07-20", cycles, blocks)


@pytest.mark.skipif(not FETCHED, reason=NOT_FETCHED)
def test_the_fetched_scenarios_hold_the_dataset_records_with_their_dates():
    data = anchors.load_scenarios()
    scenarios = data["scenarios"]
    assert data["source"] == fetch_scenarios.source_block() and len(scenarios) == fetch_scenarios.RECORDS
    assert [s["id"] for s in scenarios] == sorted({s["id"] for s in scenarios})
    assert data["dates"] == sorted({s["date"] for s in scenarios})
    for s in scenarios:
        assert s["date"] in s["text"] and s["date"] in s["failure_event"]
        assert s["asset_component"] in COMPONENTS and "hp_1" in s["text"]
        date.fromisoformat(s["date"])


@pytest.mark.skipif(not FETCHED, reason=NOT_FETCHED)
@pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
def test_every_scenario_date_anchors_inside_a_stable_block_of_the_facts_file():
    cycles = load_cycles(str(default_facts_path()))
    blocks = build_blocks(cycles)
    by_index = {c.index: c for c in cycles}
    source = HydraulicRigEnv(cycles, blocks, EpisodeConfig()).source
    for day in anchors.load_scenarios()["dates"]:
        a = anchors.anchor(day, cycles, blocks)
        block = blocks[tuple(a["combination"])]
        assert block[a["block_position"]] == a["anchored_cycle_index"] and a["block_length"] == len(block)
        source.forget()
        source.open(tuple(a["combination"]), 0, random.Random(0), anchors.start_of(a)["entry_offset"])
        assert source.play() == (a["anchored_cycle_index"], False)  # the first live cycle of the anchored start is the day's recording
        assert by_index[a["anchored_cycle_index"]].combo == tuple(a["combination"]) and by_index[a["anchored_cycle_index"]].stable
        assert a["stable"] == (a["anchored_cycle_index"] == a["cycle_index"])
        assert 0 <= a["n_faulty"] <= len(COMPONENTS)
