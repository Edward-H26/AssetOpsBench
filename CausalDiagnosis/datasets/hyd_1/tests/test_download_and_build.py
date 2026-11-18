"""CausalDiagnosis/datasets/hyd_1/download_and_build.py without the network: the refusal without --download, safe extraction, an inner
archive, and the one command from an archive on the machine and from a stand-in of the download, every file checked
against the pins of a CHECKSUMS.txt that the test writes, and the command as a process of an isolated interpreter."""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1 import download_and_build as fu

# (cooler, valve, pump, accumulator, stable flag) per cycle of the made-up raw files; the third cycle is not stable
LABELS = [(3, 100, 0, 130, 0), (100, 100, 0, 130, 0), (100, 100, 0, 130, 1)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")
    monkeypatch.setattr(fu.download_retry.urllib.request, "urlopen", refuse)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_zip(path, prefix=""):
    with zipfile.ZipFile(path, "w") as zf:
        for name in fu.REQUIRED_FILES:
            zf.writestr(f"{prefix}{name}", "1\t2\n")
    return path


def test_refuses_to_run_without_the_download_flag(capsys):
    assert fu.main([]) == 2
    assert "Refusing to download" in capsys.readouterr().err


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


def buildable_zip(path):
    """A zip of the 18 raw files that the build reads: three cycles of four samples per channel and five label columns."""
    with zipfile.ZipFile(path, "w") as zf:
        for k, name in enumerate(fu.SENSOR_FILES):
            zf.writestr(name, "".join("\t".join(repr(k + i + v) for v in (1.0, 2.0, 4.0, 7.0)) + "\n" for i in range(len(LABELS))))
        zf.writestr("profile.txt", "".join("\t".join(str(v) for v in row) + "\n" for row in LABELS))
    return path


def repin(folder, name, digest):
    """CHECKSUMS.txt of the folder with the line of name pinned to digest."""
    lines = (folder / "CHECKSUMS.txt").read_text().splitlines()
    (folder / "CHECKSUMS.txt").write_text("".join(f"{digest}  {name}\n" if line.endswith(f"  {name}") else line + "\n" for line in lines))


@pytest.fixture
def data(tmp_path, monkeypatch):
    """(data folder, archive): a data folder in place of that of the rig, whose CHECKSUMS.txt pins a buildable archive and
    the two files of a first build of it."""
    archive = buildable_zip(tmp_path / fu.ARCHIVE_NAME)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp_path / "raw")
    fu.build_blocks.build(tmp_path / "raw", tmp_path / "first")
    folder = tmp_path / "data"
    folder.mkdir()
    lines = [f"{sha(archive)}  {fu.ARCHIVE_LINE}"] + [f"{sha(tmp_path / 'first' / name)}  {name}" for name in fu.build_blocks.BUILT]
    (folder / "CHECKSUMS.txt").write_text("".join(line + "\n" for line in lines))
    monkeypatch.setattr(fu, "DATA_DIR", folder)
    monkeypatch.setattr(fu, "CHECKSUMS", folder / "CHECKSUMS.txt")
    return folder, archive


def test_the_one_command_builds_checks_and_installs_both_files_from_an_archive_on_the_machine(data, capsys):
    folder, archive = data
    pins = (folder / "CHECKSUMS.txt").read_bytes()
    assert fu.main(["--archive", str(archive)]) == 0
    wanted = fu.records.read_checksums(folder / "CHECKSUMS.txt")
    assert all(sha(folder / name) == wanted[name] for name in fu.build_blocks.BUILT)
    assert capsys.readouterr().out.splitlines()[-1] == f"DATA_OK hyd_1 {folder / fu.build_blocks.CYCLES_FILE}"
    assert sorted(os.listdir(folder)) == sorted(["CHECKSUMS.txt", *fu.build_blocks.BUILT])
    assert (folder / "CHECKSUMS.txt").read_bytes() == pins


def test_the_download_is_checked_before_it_is_read_and_kept_for_the_next_run(data, tmp_path, monkeypatch, capsys):
    folder, archive = data
    calls = []
    monkeypatch.setattr(fu, "download", lambda url, target: calls.append(url) or shutil.copyfile(archive, target))
    dest = tmp_path / "kept"
    assert fu.main(["--download", "--dest", str(dest)]) == 0
    assert sorted(os.listdir(dest)) == [fu.ARCHIVE_NAME] and calls == [fu.URL]
    capsys.readouterr()
    assert fu.main(["--download", "--dest", str(dest)]) == 0
    assert calls == [fu.URL] and not any(line.startswith("archive ") for line in capsys.readouterr().out.splitlines())
    (folder / fu.build_blocks.CYCLES_FILE).unlink()
    assert fu.main(["--download", "--dest", str(dest)]) == 0
    assert calls == [fu.URL] and (folder / fu.build_blocks.CYCLES_FILE).is_file()


def test_an_archive_with_another_checksum_is_refused_and_nothing_is_built(data, tmp_path, monkeypatch, capsys):
    folder, archive = data
    repin(folder, fu.ARCHIVE_LINE, "0" * 64)
    assert fu.main(["--archive", str(archive)]) == 1
    err = capsys.readouterr().err
    assert sha(archive) in err and "0" * 64 in err
    assert not any((folder / name).exists() for name in fu.build_blocks.BUILT)
    monkeypatch.setattr(fu, "download", lambda url, target: shutil.copyfile(archive, target))
    dest = tmp_path / "fresh"
    assert fu.main(["--download", "--dest", str(dest)]) == 1
    assert list(dest.iterdir()) == [] and archive.is_file()
    assert not any((folder / name).exists() for name in fu.build_blocks.BUILT)


def test_a_build_with_another_checksum_is_refused_and_nothing_is_installed(data, capsys):
    folder, archive = data
    built = fu.records.read_checksums(folder / "CHECKSUMS.txt")[fu.build_blocks.CYCLES_FILE]
    repin(folder, fu.build_blocks.CYCLES_FILE, "0" * 64)
    assert fu.main(["--archive", str(archive)]) == 1
    err = capsys.readouterr().err
    assert fu.build_blocks.CYCLES_FILE in err and built in err and "0" * 64 in err
    assert sorted(os.listdir(folder)) == ["CHECKSUMS.txt"]


def test_the_check_mode_reads_what_is_built_and_touches_nothing(data, capsys):
    folder, archive = data
    assert fu.main(["--check"]) == 1
    assert fu.main(["--archive", str(archive)]) == 0
    before = {name: (folder / name).read_bytes() for name in fu.build_blocks.BUILT}
    capsys.readouterr()
    assert fu.main(["--check"]) == 0 and capsys.readouterr().out.splitlines()[-1].startswith("DATA_OK hyd_1 ")
    assert {name: (folder / name).read_bytes() for name in fu.build_blocks.BUILT} == before
    cycles = folder / fu.build_blocks.CYCLES_FILE
    pinned = sha(cycles)
    cycles.write_bytes(cycles.read_bytes().replace(b"1", b"2", 1))
    assert fu.main(["--check"]) == 1
    err = capsys.readouterr().err
    assert pinned in err and sha(cycles) in err


def test_without_the_checksums_file_nothing_is_fetched(data, tmp_path, monkeypatch, capsys):
    folder, _archive = data
    calls = []
    monkeypatch.setattr(fu, "download", lambda url, target: calls.append(url))
    (folder / "CHECKSUMS.txt").unlink()
    assert fu.main(["--download", "--dest", str(tmp_path / "d")]) == 1
    assert "does not exist" in capsys.readouterr().err and calls == []


def test_the_command_starts_in_an_isolated_interpreter_with_the_standard_library_alone():
    script = Path(fu.__file__).resolve()
    done = subprocess.run([sys.executable, "-I", "-S", "-B", str(script), "--help"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "--archive" in done.stdout and "--check" in done.stdout
