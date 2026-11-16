"""Fetch the UCI archive of the hydraulic rig hyd_1 and build its data file, every file checked against its pin.

The one command that builds the data file from a fresh clone: it fetches the archive of UCI dataset 447 from its official
page, checks its SHA-256, builds CausalDiagnosis/datasets/hyd_1/data/hyd_cycles.jsonl and blocks.json with
CausalDiagnosis/datasets/hyd_1/build_features_and_blocks.py, checks both against CausalDiagnosis/datasets/hyd_1/data/CHECKSUMS.txt and installs them.
The last line it prints is DATA_OK with the asset and the path of the data file.

Source: UCI Machine Learning Repository, dataset 447 "Condition monitoring of hydraulic systems" (Helwig et al.
2015), licence CC BY 4.0, DOI 10.24432/C5CW21.

Nothing is downloaded without --download, so the archive is fetched only on an explicit request. --archive FILE builds from
a copy of the archive on the machine, with no network, and --check only checks the two built files against their pins. A
second run finds the files in place and fetches nothing. The command never writes CHECKSUMS.txt: a pin that the command
wrote itself would pin nothing.
usage: python3 -I -m CausalDiagnosis.datasets.hyd_1.download_and_build --download [--dest DIR]
       python3 -I -m CausalDiagnosis.datasets.hyd_1.download_and_build --archive FILE
       python3 -I -m CausalDiagnosis.datasets.hyd_1.download_and_build --check"""
import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the fork root, which holds CausalDiagnosis/, also outside an editable install
from CausalDiagnosis.datasets.dataset_loader import recordings as records  # noqa: E402
from CausalDiagnosis.datasets.hyd_1 import records as hyd_1  # noqa: E402
from CausalDiagnosis.datasets.hyd_1 import build_features_and_blocks as build_blocks  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
ARCHIVE_NAME = "condition+monitoring+of+hydraulic+systems.zip"
ARCHIVE_LINE = "uci_hydraulic/" + ARCHIVE_NAME  # the name of the archive in CHECKSUMS.txt
SENSOR_FILES = tuple(f"{s}.txt" for s in (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"))
REQUIRED_FILES = SENSOR_FILES + ("profile.txt",)
DATA_DIR = hyd_1.release_path().parent  # the release folder of hyd_1, where this command installs the built files
DEFAULT_DEST = DATA_DIR / "uci_hydraulic"
CHECKSUMS = DATA_DIR / records.CHECKSUMS_NAME
USER_AGENT = "Mozilla/5.0"  # a generic agent string, never a person's name or address
AGAIN = "Run `python -m CausalDiagnosis.datasets.hyd_1.download_and_build --download`"


def download(url, target):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)


def safe_extract(zip_path, dest):
    """Extract every member under dest; a member whose path would leave dest is refused before anything is written."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise ValueError(f"{zip_path}: member {name!r} would be written outside {dest}")
        zf.extractall(dest)


def find_data_dir(root):
    """The folder under root that holds all 18 raw files; inner zip archives are extracted, each into its own folder,
    when the outer archive does not hold the files directly."""
    root = Path(root)

    def search():
        return sorted({p.parent for p in root.rglob("profile.txt") if all((p.parent / f).is_file() for f in REQUIRED_FILES)})

    found = search()
    if not found:
        for inner in sorted(root.rglob("*.zip")):
            safe_extract(inner, inner.with_suffix(""))
        found = search()
    if len(found) != 1:
        raise FileNotFoundError(f"expected one folder with {', '.join(REQUIRED_FILES)} under {root}, found {len(found)}")
    return found[0]


def shown(path):
    """A path relative to the root of the repository when it lies under it, else as given."""
    path = Path(path)
    resolved, repo = path.resolve(), records.REPO.resolve()
    return str(resolved.relative_to(repo)) if resolved.is_relative_to(repo) else str(path)


def pins():
    """The pinned SHA-256 of CHECKSUMS.txt, which must hold the archive and both built files."""
    if not CHECKSUMS.exists():
        raise records.DataFileError(f"{shown(CHECKSUMS)} does not exist, so no file can be checked against its pin.")
    wanted = records.read_checksums(CHECKSUMS)
    absent = [name for name in (ARCHIVE_LINE, *build_blocks.BUILT) if name not in wanted]
    if absent:
        raise records.DataFileError(f"{shown(CHECKSUMS)} pins no checksum for {', '.join(absent)}.")
    return wanted


def built_problem(wanted):
    """None when both built files lie in the data folder with their pinned SHA-256, else the sentence on the first that does not."""
    for name in build_blocks.BUILT:
        path = DATA_DIR / name
        if not path.exists():
            return f"{shown(path)} does not exist. {AGAIN} from the root of the repository."
        got = records.file_sha256(path)
        if got != wanted[name]:
            return f"{shown(path)} has the SHA-256 {got}, and {shown(CHECKSUMS)} pins {wanted[name]}. {AGAIN} again."
    return None


def checked(archive, wanted, downloaded):
    """The archive, after its SHA-256 equals its pin and before anything reads it. A wrong archive that this run downloaded
    is removed, and one that was there is left."""
    got = records.file_sha256(archive)
    if got != wanted[ARCHIVE_LINE]:
        if downloaded:
            Path(archive).unlink()
        raise records.DataFileError(f"{shown(archive)} has the SHA-256 {got}, and {shown(CHECKSUMS)} pins {wanted[ARCHIVE_LINE]}. "
                                    f"It is not the archive of {URL} that this repository was built on, and nothing was built.")
    return Path(archive)


def archive_for(dest, wanted):
    """The checked archive in dest: the one that lies there, else a download from the official page."""
    dest = Path(dest)
    archive = dest / ARCHIVE_NAME
    if archive.exists():
        return checked(archive, wanted, downloaded=False)
    dest.mkdir(parents=True, exist_ok=True)
    partial = dest / (ARCHIVE_NAME + ".part")
    try:
        download(URL, partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    checked(partial, wanted, downloaded=True)
    os.replace(partial, archive)
    return archive


def install(archive, wanted):
    """Extract the checked archive into a temporary folder, build both files into a second one inside the data folder,
    check each against its pin and move them into the data folder, the cycles file last. Nothing is installed when a
    built file differs from its pin."""
    with tempfile.TemporaryDirectory() as extracted:
        safe_extract(archive, extracted)
        raw = find_data_dir(extracted)
        with tempfile.TemporaryDirectory(dir=DATA_DIR) as out:
            build_blocks.build(raw, out)
            for name in build_blocks.BUILT:
                got = records.file_sha256(Path(out) / name)
                if got != wanted[name]:
                    raise records.DataFileError(f"The build of {name} has the SHA-256 {got}, and {shown(CHECKSUMS)} pins "
                                                f"{wanted[name]}. Nothing was installed.")
            for name in reversed(build_blocks.BUILT):
                os.replace(Path(out) / name, DATA_DIR / name)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="fetch the archive from the official page, unless it is in the folder of --dest")
    mode.add_argument("--archive", metavar="FILE", help="build from a copy of the archive on this machine, with no network")
    mode.add_argument("--check", action="store_true", help="only check the two built files against CHECKSUMS.txt")
    ap.add_argument("--dest", metavar="DIR", help=f"with --download, the folder that keeps the archive (default {shown(DEFAULT_DEST)})")
    a = ap.parse_args(argv)
    if not (a.download or a.archive or a.check):
        print(f"Refusing to download {URL} without --download: the archive is fetched only on an explicit "
              "request, so rerun with --download.", file=sys.stderr)
        return 2
    if a.dest is not None and not a.download:
        ap.error("--dest goes with --download")
    try:
        wanted = pins()
        problem = built_problem(wanted)
        if a.check and problem:
            raise records.DataFileError(problem)
        if problem:
            archive = Path(a.archive) if a.archive else archive_for(Path(a.dest) if a.dest else DEFAULT_DEST, wanted)
            if a.archive:
                checked(archive, wanted, downloaded=False)
            install(archive, wanted)
            print(f"archive {shown(archive)} SHA-256 {wanted[ARCHIVE_LINE]}, as pinned")
    except (records.DataFileError, ValueError, OSError) as e:
        print(e, file=sys.stderr)
        return 1
    for name in build_blocks.BUILT:
        print(f"built {shown(DATA_DIR / name)} SHA-256 {wanted[name]}, as pinned")
    print(f"DATA_OK {hyd_1.ASSET_ID} {shown(DATA_DIR / build_blocks.CYCLES_FILE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
