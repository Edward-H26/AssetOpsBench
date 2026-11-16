"""The data file of an asset and the files built from it (CausalDiagnosis/docs/environment.md, section The data file): the
resolver, the pin of a release file in CHECKSUMS.txt of its release folder, the named refusals of a missing or another
file, the hash that the recordings carry, the declaration of the official page and the command in the asset file, the
refusal of a file built from another data file, and the refusal that stops a test session. The tests write their own
recordings into the temporary folder of the test, with the tiny asset of conftest.py, and state no value of a data file."""
import copy
import json
import re

import pytest

from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports, recordings as records
from CausalDiagnosis.datasets.dataset_loader.evidence_kinds import kind_of
from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, get_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import DATA_KEYS, SOURCE_KEYS, validate
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

from CausalDiagnosis.datasets.tests.conftest import ROOT, mount_port, needs_data, write_recordings

PAGE = "https://example.org/datasets/tiny"
FETCH = "python benchmarks/tiny/fetch_tiny.py --download"
TOGETHER = "records.page and records.fetch are stated together, each as a text that is not empty"


@pytest.fixture
def sourced(tiny_data, tmp_path, monkeypatch):
    """The tiny profile with the official page and the command of its data, registered for the test, its release folder
    under the temporary folder of the test and the variable of its data unset."""
    data = copy.deepcopy(tiny_data)
    data["records"].update(page=PAGE, fetch=FETCH)
    profile = register_profile(AssetProfile.from_json(data, "the tiny asset with a source"))
    monkeypatch.setattr(records, "REPO", tmp_path)
    monkeypatch.delenv(profile.records["data_env"], raising=False)
    yield profile
    unregister_profile(profile.asset_id)


@pytest.fixture
def declared(tiny_data, tmp_path, monkeypatch):
    """One made-up asset folder that states the official page and the command of its data, in place of the bundled assets: its
    folder is mounted on the search path of CausalDiagnosis.datasets, and the search of asset folders sees that folder alone."""
    data = copy.deepcopy(tiny_data)
    data["asset_id"] = "declared_1"
    data["records"].update(page=PAGE, fetch=FETCH)
    assets = tmp_path / "assets"
    (assets / "declared_1").mkdir(parents=True)
    (assets / "declared_1" / "asset.json").write_text(json.dumps(data), encoding="utf-8")
    (assets / "declared_1" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(records, "REPO", tmp_path)
    monkeypatch.delenv(data["records"]["data_env"], raising=False)
    monkeypatch.delenv(records.WITHOUT_DATA, raising=False)
    with mount_port(assets):
        monkeypatch.setattr(ports, "_search_path", lambda: (str(assets),))
        yield get_profile("declared_1", engine=False)


def release(profile):
    """Recordings of the profile written to its release file."""
    path = records.release_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_recordings(path, profile)


def pin(profile, digest=None, name=None):
    """CHECKSUMS.txt of the release folder with one line, the hash of the release file unless another is given."""
    path = records.release_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{digest or records.file_sha256(path)}  {name or path.name}\n"
    (path.parent / records.CHECKSUMS_NAME).write_text(line, encoding="utf-8")


def changed(path):
    """The file with one label of its first record changed, so that it parses and has another SHA-256."""
    data = path.read_bytes()
    assert b'"stable_flag": 0' in data
    path.write_bytes(data.replace(b'"stable_flag": 0', b'"stable_flag": 1', 1))


def bundled():
    """(asset folder, asset file) of every asset under CausalDiagnosis/datasets of the recorded levels kind, one level deep."""
    found = []
    for path in sorted(ASSETS_DIR.glob("*/asset.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("evidence_kind") == "recorded_levels":
            found.append((path.parent, data))
    return found


def test_the_resolver_takes_the_variable_when_it_is_set_and_not_empty_else_the_release_file(sourced, monkeypatch, tmp_path):
    variable = sourced.records["data_env"]
    assert records.default_facts_path(sourced) == records.release_path(sourced)
    monkeypatch.setenv(variable, "")
    assert records.default_facts_path(sourced) == records.release_path(sourced)
    monkeypatch.setenv(variable, str(tmp_path / "elsewhere.jsonl"))
    assert records.default_facts_path(sourced) == tmp_path / "elsewhere.jsonl"


def test_the_release_file_is_read_only_when_it_equals_its_pin(sourced):
    path = release(sourced)
    digest = records.file_sha256(path)
    pin(sourced)
    cycles = records.load_cycles(path, sourced)
    assert isinstance(cycles, list) and len(cycles) > 0 and cycles.facts_sha256 == digest
    changed(path)
    with pytest.raises(records.DataFileError) as refused:
        records.load_cycles(path, sourced)
    text = str(refused.value)
    assert digest in text and records.file_sha256(path) in text and PAGE in text and FETCH in text
    path.write_bytes(path.read_bytes().replace(b'"stable_flag": 1', b'"stable_flag": 0', 1))
    pin(sourced, "0" * 64)
    with pytest.raises(records.DataFileError) as refused:
        records.load_cycles(path, sourced)
    text = str(refused.value)
    assert digest in text and "0" * 64 in text and PAGE in text and FETCH in text
    pin(sourced, digest, name="another_file.jsonl")
    with pytest.raises(records.DataFileError, match="pins no checksum for tiny_cycles.jsonl"):
        records.load_cycles(path, sourced)
    (path.parent / records.CHECKSUMS_NAME).unlink()
    with pytest.raises(records.DataFileError, match="pins no checksum for tiny_cycles.jsonl"):
        records.load_cycles(path, sourced)


def test_a_missing_data_file_is_refused_by_name_and_nothing_falls_back(sourced, monkeypatch, tmp_path):
    with pytest.raises(records.DataFileError) as refused:
        records.load_cycles(records.default_facts_path(sourced), sourced)
    assert refused.value.missing and PAGE in str(refused.value) and FETCH in str(refused.value)
    release(sourced)
    pin(sourced)
    variable, missing = sourced.records["data_env"], tmp_path / "missing.jsonl"
    monkeypatch.setenv(variable, str(missing))
    for read in (lambda: records.load_cycles(records.default_facts_path(sourced), sourced), lambda: records.check_data(sourced)):
        with pytest.raises(records.DataFileError) as refused:
            read()
        assert refused.value.missing and variable in str(refused.value) and str(missing) in str(refused.value)


def test_a_file_that_the_variable_or_a_path_names_is_read_as_given_and_carries_its_hash(sourced, monkeypatch, tmp_path):
    other = write_recordings(tmp_path / "development.jsonl", sourced, seed=1)
    digest = records.file_sha256(other)
    assert records.load_cycles(other, sourced).facts_sha256 == digest
    monkeypatch.setenv(sourced.records["data_env"], str(other))
    assert records.load_cycles(records.default_facts_path(sourced), sourced).facts_sha256 == digest
    assert records.check_data(sourced) == digest


def test_check_data_gives_the_hash_that_load_cycles_carries_and_refuses_what_it_refuses(sourced):
    with pytest.raises(records.DataFileError) as refused:
        records.check_data(sourced)
    assert refused.value.missing
    path = release(sourced)
    pin(sourced)
    assert records.check_data(sourced) == records.load_cycles(path, sourced).facts_sha256 == records.file_sha256(path)
    changed(path)
    with pytest.raises(records.DataFileError) as refused:
        records.check_data(sourced)
    assert not refused.value.missing and records.file_sha256(path) in str(refused.value)


def test_the_checksums_file_is_sha256sum_lines_relative_to_the_release_folder(sourced):
    assert records.checksums(sourced) == {}
    folder = records.release_path(sourced).parent
    folder.mkdir(parents=True)
    first, second = "a" * 64, "b" * 64
    (folder / records.CHECKSUMS_NAME).write_text(f"{first}  tiny_cycles.jsonl\n{second}  source/archive.zip\n", encoding="utf-8")
    assert records.checksums(sourced) == {"tiny_cycles.jsonl": first, "source/archive.zip": second}
    for bad in (f"{first} tiny_cycles.jsonl\n", f"{first[:63]}  tiny_cycles.jsonl\n"):
        (folder / records.CHECKSUMS_NAME).write_text(bad, encoding="utf-8")
        with pytest.raises(records.DataFileError, match="is no sha256sum line"):
            records.checksums(sourced)


def test_page_and_fetch_are_stated_together_or_not_at_all(tiny_data):
    both = copy.deepcopy(tiny_data)
    both["records"].update(page=PAGE, fetch=FETCH)
    assert validate(both) == []
    one = copy.deepcopy(both)
    one["records"].pop("fetch")
    empty = copy.deepcopy(both)
    empty["records"]["page"] = ""
    third = copy.deepcopy(both)
    third["records"]["mirror"] = PAGE
    assert validate(one) == [TOGETHER] and validate(empty) == [TOGETHER]
    assert validate(third) == [f"records must hold {list(kind_of(third['evidence_kind']).record_keys + DATA_KEYS)} and may add {list(SOURCE_KEYS)}"]


def test_every_bundled_asset_names_its_page_and_its_command_and_pins_its_release_file():
    assert bundled()
    for folder, data in bundled():
        spec = data["records"]
        assert spec.get("page", "").startswith("https://") and spec.get("fetch"), folder.name
        tokens = spec["fetch"].split()
        script = tokens[tokens.index("-m") + 1].replace(".", "/") + ".py"
        assert (ROOT / script).is_file(), (folder.name, script)
        pins = records.read_checksums(ROOT / spec["release_dir"] / records.CHECKSUMS_NAME)
        assert re.fullmatch(r"[0-9a-f]{64}", pins.get(spec["release_file"], "")), folder.name


@needs_data
def test_the_release_file_of_every_bundled_asset_is_built_and_equals_its_pin():
    for folder, data in bundled():
        spec = data["records"]
        path = ROOT / spec["release_dir"] / spec["release_file"]
        pins = records.read_checksums(path.parent / records.CHECKSUMS_NAME)
        assert path.is_file() and records.file_sha256(path) == pins[spec["release_file"]], folder.name


def test_a_derived_file_of_another_data_file_is_refused_by_name():
    first, second, again = "a" * 64, "b" * 64, "Build it again."
    assert records.require_same_data("table.json", first, first, again) is None
    with pytest.raises(records.DataFileError) as refused:
        records.require_same_data("table.json", second, first, again)
    assert first in str(refused.value) and second in str(refused.value) and again in str(refused.value)
    with pytest.raises(records.DataFileError, match="a data file that it does not name"):
        records.require_same_data("table.json", None, first, again)
    with pytest.raises(records.DataFileError, match="loaded from a data file"):
        records.require_same_data("table.json", first, None, again)


def test_the_refusal_of_a_session_names_every_bundled_asset_without_data_and_respects_the_switch(declared, monkeypatch):
    refusal = records.data_refusal()
    assert refusal and "declared_1" in refusal and FETCH in refusal and f"{records.WITHOUT_DATA}=1" in refusal
    assert records.data_header() == ["data file of declared_1: none"]
    monkeypatch.setenv(records.WITHOUT_DATA, "1")
    assert records.data_refusal() is None
    path = release(declared)
    pin(declared, "0" * 64)
    assert "0" * 64 in records.data_refusal()
    monkeypatch.delenv(records.WITHOUT_DATA)
    assert "0" * 64 in records.data_refusal()
    pin(declared)
    assert records.data_refusal() is None
    assert records.data_header() == [f"data file of declared_1: {path}, SHA-256 {records.file_sha256(path)}"]


def test_an_asset_file_that_does_not_load_stops_no_session(tiny_data, tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    (assets / "broken_json").mkdir(parents=True)
    (assets / "broken_json" / "asset.json").write_text("{ no json", encoding="utf-8")
    invalid = copy.deepcopy(tiny_data)
    invalid["asset_id"] = "broken_profile"
    invalid["records"].pop("release_file")
    invalid["records"].update(page=PAGE, fetch=FETCH)
    (assets / "broken_profile").mkdir()
    (assets / "broken_profile" / "asset.json").write_text(json.dumps(invalid), encoding="utf-8")
    (assets / "not-an-id").mkdir()
    (assets / "not-an-id" / "asset.json").write_text(json.dumps(tiny_data), encoding="utf-8")
    monkeypatch.setattr(records, "REPO", tmp_path)
    with mount_port(assets):
        monkeypatch.setattr(ports, "_search_path", lambda: (str(assets),))
        assert records.data_refusal() is None and records.data_header() == []


def test_an_environment_keeps_the_hash_of_the_file_its_recordings_came_from(tiny, tiny_recordings):
    cycles = records.load_cycles(tiny_recordings, tiny)
    blocks = records.build_blocks(cycles)
    assert HydraulicRigEnv(cycles, blocks, tiny.default_config()).facts_sha256 == records.file_sha256(tiny_recordings)
    assert HydraulicRigEnv(list(cycles), blocks, tiny.default_config()).facts_sha256 is None
