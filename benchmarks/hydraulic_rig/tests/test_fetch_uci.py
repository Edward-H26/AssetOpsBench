"""benchmarks/hydraulic_rig/fetch_uci.py without the network: the refusal without --download, safe extraction, an inner
archive, the checksum record, and a full run whose download is replaced by a local archive."""
import hashlib
import io
import shutil
import zipfile

import pytest

import fetch_uci as fu


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")
    monkeypatch.setattr(fu.urllib.request, "urlopen", refuse)


def raw_zip(path, prefix=""):
    with zipfile.ZipFile(path, "w") as zf:
        for name in fu.REQUIRED_FILES:
            zf.writestr(f"{prefix}{name}", "1\t2\n")
    return path


def test_refuses_to_run_without_the_download_flag(capsys):
    assert fu.main([]) == 2
    assert "Refusing to download" in capsys.readouterr().err


def test_refuses_a_destination_that_is_not_empty(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    assert fu.main(["--download", "--dest", str(tmp_path)]) == 2


def test_safe_extract_refuses_a_member_outside_the_destination(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as zf:
        zf.writestr("ok.txt", "x")
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="outside"):
        fu.safe_extract(tmp_path / "evil.zip", tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "out" / "ok.txt").exists()


def test_find_data_dir_opens_an_inner_archive(tmp_path):
    inner = io.BytesIO()
    raw_zip(inner, "hydraulic/")
    with zipfile.ZipFile(tmp_path / "outer.zip", "w") as zf:
        zf.writestr("hydraulic.zip", inner.getvalue())
    fu.safe_extract(tmp_path / "outer.zip", tmp_path / "extracted")
    found = fu.find_data_dir(tmp_path / "extracted")
    assert found.name == "hydraulic" and all((found / n).is_file() for n in fu.REQUIRED_FILES)


def test_find_data_dir_needs_all_eighteen_files(tmp_path):
    (tmp_path / "profile.txt").write_text("1\n")
    with pytest.raises(FileNotFoundError):
        fu.find_data_dir(tmp_path)


def test_record_checksum_appends_once_and_refuses_a_changed_hash(tmp_path):
    c = tmp_path / "CHECKSUMS.txt"
    assert fu.record_checksum(c, "uci/a.zip", "ab" * 32) is True
    assert fu.record_checksum(c, "uci/a.zip", "ab" * 32) is False
    assert c.read_text() == "ab" * 32 + "  uci/a.zip\n"
    with pytest.raises(ValueError, match="differs"):
        fu.record_checksum(c, "uci/a.zip", "cd" * 32)


def test_a_run_with_a_local_archive_records_the_hash_and_finds_the_files(tmp_path, monkeypatch, capsys):
    source = raw_zip(tmp_path / "source.zip")
    monkeypatch.setattr(fu, "download", lambda url, target: shutil.copyfile(source, target))
    monkeypatch.setattr(fu, "CHECKSUMS", tmp_path / "data" / "CHECKSUMS.txt")
    (tmp_path / "data").mkdir()
    dest = tmp_path / "data" / "uci_hydraulic"
    assert fu.main(["--download", "--dest", str(dest)]) == 0
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert (tmp_path / "data" / "CHECKSUMS.txt").read_text() == f"{digest}  uci_hydraulic/{fu.ARCHIVE_NAME}\n"
    out = capsys.readouterr().out
    assert f"raw files in {dest / 'extracted'}" in out and "first run" in out


def test_a_run_refuses_an_archive_whose_hash_differs_from_the_expected_one(tmp_path, monkeypatch):
    source = raw_zip(tmp_path / "source.zip")
    monkeypatch.setattr(fu, "download", lambda url, target: shutil.copyfile(source, target))
    monkeypatch.setattr(fu, "CHECKSUMS", tmp_path / "CHECKSUMS.txt")
    monkeypatch.setattr(fu, "EXPECTED_SHA256", "0" * 64)
    assert fu.main(["--download", "--dest", str(tmp_path / "dest")]) == 1
    assert not (tmp_path / "CHECKSUMS.txt").exists() and not (tmp_path / "dest" / "extracted").exists()
