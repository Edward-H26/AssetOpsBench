"""The command that registers the training split of an asset (write_training_split.py, CausalDiagnosis/docs/adding-a-dataset.md step
6): on a made-up asset it writes a folds file that fault_folds reads back from the file alone and that puts every start of the
asset in exactly one test fold and names the data file it was dealt from, it refuses K beyond the single faults and writes
nothing, it stops with the sentence of the loader when the data file does not exist, and on the hydraulic rig with K = 2
its output is the registered configs/training/hyd_1/folds.json byte for byte, and with --starts development the development
split folds_development.json beside it, none of whose combinations is a test start."""
from CausalDiagnosis.utils.paths import CONFIGS
import json
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.table_definitions import write_training_split as write_folds
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.tests.conftest import HAS_DATA, TINY_FILE, write_recordings
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.splits.train_test_folds import fault_folds

FOLDS_FILE = CONFIGS / "training" / "hyd_1" / "folds.json"
DEVELOPMENT_FILE = FOLDS_FILE.with_name("folds_development.json")


def _made_up_asset(tmp_path, max_faulty):
    """The tiny asset of the asset layer tests, three components with four single faults, with starts of up to max_faulty faults,
    and a data file of every combination."""
    data = json.loads(TINY_FILE.read_text(encoding="utf-8"))
    data["sampler"]["fault_rule"]["max_faulty"] = max_faulty
    asset = tmp_path / "asset.json"
    asset.write_text(json.dumps(data), encoding="utf-8")
    facts = write_recordings(tmp_path / "cycles.jsonl", AssetProfile.from_json(data, str(asset)))
    return data, asset, facts


def test_the_split_of_a_made_up_asset_reads_back_and_tests_every_start_once(tmp_path, capsys):
    data, asset, facts = _made_up_asset(tmp_path, 3)
    out = tmp_path / "folds.json"
    write_folds.main(["--asset", data["asset_id"], "--k", "3", "--profile", str(asset), "--facts", str(facts), "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert (record["asset_id"], record["k"], record["seed"], record["components"]) == (data["asset_id"], 3, 0, data["components"])
    assert record["facts_sha256"] == records.file_sha256(facts)
    starts = {tuple(row["combination"]): tuple((c, level) for c, level in zip(record["components"], row["combination"])
                                               if level != data["healthy"][c]) for row in record["combinations"]}
    assert len(starts) == len(record["combinations"]) and max(len(atoms) for atoms in starts.values()) == 3
    split, atom_fold = fault_folds(starts, record["components"], record["k"], record["seed"])
    assert {tuple(row["combination"]): (row["fold"], tuple(row["models"])) for row in record["combinations"]} == split
    assert {(row["component"], row["level"]): row["fold"] for row in record["atoms"]} == atom_fold
    for row in record["combinations"]:
        assert row["fold"] in range(1, 4) and row["fold"] not in row["models"] and row["trains"] == bool(row["models"])
    assert {row["fold"] for row in record["combinations"]} == {1, 2, 3}
    counts = write_folds.fold_counts(record)
    assert sum(tested for tested, _training in counts.values()) == len(starts)
    assert "held-out fold 3:" in capsys.readouterr().out


def test_k_beyond_the_single_faults_of_the_asset_is_refused_and_writes_nothing(tmp_path):
    data, asset, facts = _made_up_asset(tmp_path, 1)
    out = tmp_path / "folds.json"
    with pytest.raises(SystemExit, match="exceeds the 4 single faults"):
        write_folds.main(["--asset", data["asset_id"], "--k", "5", "--profile", str(asset), "--facts", str(facts), "--out", str(out)])
    assert not out.exists()


def test_a_missing_data_file_stops_the_command_with_the_sentence_of_the_loader(tmp_path):
    """A data file that does not exist stops the command with the refusal of the loader, which names the file, and no folds
    file is written."""
    data, asset, _facts = _made_up_asset(tmp_path, 1)
    missing, out = tmp_path / "missing.jsonl", tmp_path / "folds.json"
    with pytest.raises(records.DataFileError) as refused:
        records.load_cycles(str(missing), AssetProfile.from_json(data, str(asset)))
    with pytest.raises(SystemExit) as stopped:
        write_folds.main(["--asset", data["asset_id"], "--k", "2", "--profile", str(asset), "--facts", str(missing), "--out", str(out)])
    assert str(stopped.value) == str(refused.value) and str(missing) in str(stopped.value)
    assert not out.exists()


def test_a_split_that_leaves_a_model_without_a_training_start_is_refused_and_writes_nothing(tmp_path):
    """The tiny asset with starts of three faults alone: under K = 2 each start holds a fault of both folds, so no model has a
    training start, and the command names the reason and writes no file."""
    data, asset, facts = _made_up_asset(tmp_path, 3)
    data["sampler"]["fault_rule"]["min_faulty"] = 3
    asset.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "folds.json"
    with pytest.raises(SystemExit, match="no split written: the split leaves the models of folds 1, 2 of 2 without a training start"):
        write_folds.main(["--asset", data["asset_id"], "--k", "2", "--profile", str(asset), "--facts", str(facts), "--out", str(out)])
    assert not out.exists()


@pytest.mark.skipif(not HAS_DATA, reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
def test_the_registered_split_of_the_rig_is_the_output_of_the_command(tmp_path):
    out = tmp_path / "folds.json"
    write_folds.main(["--asset", "hyd_1", "--k", "2", "--out", str(out)])
    assert out.read_bytes() == FOLDS_FILE.read_bytes()


@pytest.mark.skipif(not HAS_DATA, reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
def test_the_development_split_of_the_rig_is_the_output_of_the_command(tmp_path):
    """The same rule dealt over the development starts of the rig, which rft.json names beside the registered split: no
    combination of it is a start of the test set, whose combinations the registered split deals."""
    out = tmp_path / "folds_development.json"
    write_folds.main(["--asset", "hyd_1", "--k", "2", "--starts", "development", "--out", str(out)])
    assert out.read_bytes() == DEVELOPMENT_FILE.read_bytes()
    development, registered = (json.loads(path.read_text(encoding="utf-8")) for path in (DEVELOPMENT_FILE, FOLDS_FILE))
    assert (development["starts"], registered["starts"], development["k"]) == ("development", "registered", registered["k"])
    test_starts = {tuple(row["combination"]) for row in registered["combinations"]}
    assert development["combinations"] and not test_starts & {tuple(row["combination"]) for row in development["combinations"]}
