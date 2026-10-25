"""benchmarks/hydraulic_rig/build_blocks.py without the UCI archive: feature recomputation on synthetic cycles, the round trip through
the environment's loader, the --check comparison, verify_data_facts.py on a rebuilt file, and the block builder on the
IndustryAssetEQA records."""
import itertools
import json
import math
from pathlib import Path

import pytest

import build_blocks as bb
import verify_data_facts as vdf
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, build_blocks, default_facts_path, load_cycles, tier_features

needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")

# (cooler, valve, pump, accumulator, stable flag) per synthetic cycle; the third cycle is not stable
LABELS = [(3, 100, 0, 130, 0), (100, 100, 0, 130, 0), (100, 100, 0, 130, 1)]


def samples(sensor, cycle):
    k = SENSORS.index(sensor)
    return [k + cycle + v for v in (1.0, 2.0, 4.0, 7.0)]


def write_raw(folder):
    folder.mkdir()
    for s in SENSORS:
        (folder / f"{s}.txt").write_text("".join("\t".join(repr(v) for v in samples(s, i)) + "\n" for i in range(len(LABELS))))
    (folder / "profile.txt").write_text("".join("\t".join(str(v) for v in lab) + "\n" for lab in LABELS))
    return folder


def test_summary_statistics_match_a_hand_computation():
    m = bb.moments([1.0, 2.0, 4.0, 7.0])  # mean 3.5, squared deviations 6.25 + 2.25 + 0.25 + 12.25 = 21
    assert bb.summary(m, 0) == {"mean": 3.5, "std": math.sqrt(21 / 4), "min": 1.0, "max": 7.0}
    assert bb.summary(m, 1)["std"] == math.sqrt(21 / 3)


def test_recomputed_cycles_load_in_the_environment_and_form_blocks(tmp_path):
    labels, by_sensor = bb.raw_moments(write_raw(tmp_path / "raw"))
    cycles_path, blocks = bb.write_outputs(bb.records(labels, by_sensor, 0), tmp_path / "out")
    cycles = load_cycles(cycles_path)
    assert [c.index for c in cycles] == [0, 1, 2]
    assert cycles[0].combo == (3, 100, 0, 130) and cycles[0].stable and not cycles[2].stable
    assert cycles[1].features["Pressure_PS1_mean"] == 3.5 + 1
    assert cycles[2].features["Temperature_TS1_max"] == SENSORS.index("TS1") + 2 + 7.0
    assert set(tier_features(cycles[0], 1)) == {bb.feature_name(s, st) for s in TIER_SENSORS[1] for st in bb.STATS}
    assert blocks == {(3, 100, 0, 130): [0], (100, 100, 0, 130): [1]}
    saved = json.loads((tmp_path / "out" / "blocks.json").read_text())
    assert saved == {"components": list(COMPONENTS), "blocks": [{"combo": [3, 100, 0, 130], "cycles": [0]}, {"combo": [100, 100, 0, 130], "cycles": [1]}]}


@needs_facts
def test_feature_names_are_those_of_the_eqa_facts_file():
    first = load_cycles(default_facts_path())[0]
    assert {bb.feature_name(s, st) for s in SENSORS for st in bb.STATS} <= set(first.features)


def test_a_sensor_file_with_a_missing_row_is_refused(tmp_path):
    raw = write_raw(tmp_path / "raw")
    lines = (raw / "CP.txt").read_text().splitlines(keepends=True)
    (raw / "CP.txt").write_text("".join(lines[:-1]))
    with pytest.raises(ValueError, match="CP.txt has 2 rows"):
        bb.raw_moments(raw)


def test_a_ragged_row_is_refused(tmp_path):
    raw = write_raw(tmp_path / "raw")
    (raw / "FS1.txt").write_text((raw / "FS1.txt").read_text() + "1.0\t2.0\n")
    with pytest.raises(ValueError, match="FS1.txt: line 4 has 2 samples, expected 4"):
        bb.raw_moments(raw)


@pytest.mark.parametrize("ddof", [0, 1])
def test_check_accepts_matching_facts_and_names_their_std_convention(tmp_path, ddof):
    labels, by_sensor = bb.raw_moments(write_raw(tmp_path / "raw"))
    facts, _ = bb.write_outputs(bb.records(labels, by_sensor, ddof), tmp_path / "facts")
    assert bb.check(labels, by_sensor, facts) == ([], ddof)


def test_check_reports_a_changed_value_and_changed_labels(tmp_path):
    labels, by_sensor = bb.raw_moments(write_raw(tmp_path / "raw"))
    facts, _ = bb.write_outputs(bb.records(labels, by_sensor, 0), tmp_path / "facts")
    recs = [json.loads(line) for line in facts.read_text().splitlines()]
    next(f for f in recs[1]["features"] if f["name"] == "Vibration_VS1_max")["value"] += 0.01
    recs[2]["labels"]["valve_condition"] = 90
    facts.write_text("".join(json.dumps(r) + "\n" for r in recs))
    problems, ddof = bb.check(labels, by_sensor, facts)
    assert ddof == 0 and len(problems) == 2
    assert problems[0].startswith("cycle 1: Vibration_VS1_max") and problems[1].startswith("cycle 2: labels")


@pytest.mark.skipif(not vdf.DEFAULT_HYD1.exists(), reason="AssetOpsBench's hyd_1 file is not checked out")
def test_verify_data_facts_runs_only_the_hash_check_on_a_rebuilt_file(tmp_path, capsys):
    labels, by_sensor = bb.raw_moments(write_raw(tmp_path / "raw"))
    rebuilt, _ = bb.write_outputs(bb.records(labels, by_sensor, 0), tmp_path / "out")
    assert vdf.main(["--facts", str(rebuilt), "--hyd1", str(vdf.DEFAULT_HYD1)]) == 1
    out = capsys.readouterr().out
    assert vdf.NOT_EQA in out and "0 of 1 checks pass" in out


@needs_facts
def test_block_builder_on_the_eqa_records(tmp_path):
    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    bb.write_blocks(blocks, tmp_path / "blocks.json")
    saved = json.loads((tmp_path / "blocks.json").read_text())["blocks"]
    combos = [tuple(b["combo"]) for b in saved]
    assert sorted(combos) == sorted(itertools.product(*(LEVELS[c] for c in COMPONENTS)))  # all 144 combinations
    assert all(b["cycles"] == list(range(b["cycles"][0], b["cycles"][-1] + 1)) for b in saved)  # one contiguous block each
    assert sum(len(b["cycles"]) for b in saved) == sum(1 for c in cycles if c.stable)  # every stable cycle in one block
    assert blocks[tuple(HEALTHY[c] for c in COMPONENTS)]
