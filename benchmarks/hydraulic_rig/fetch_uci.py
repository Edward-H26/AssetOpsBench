"""Download the UCI hydraulic rig archive for the release path of docs/hydraulic_rig/environment.md, record its SHA-256 in
src/envs/hydraulic_rig/data/CHECKSUMS.txt, and extract it into a fresh folder for benchmarks/hydraulic_rig/build_blocks.py.

Source: UCI Machine Learning Repository, dataset 447 "Condition monitoring of hydraulic systems" (Helwig et al.
2015), licence CC BY 4.0, DOI 10.24432/C5CW21.

Nothing is downloaded without --download, so the archive is fetched only on an explicit request. The first run
with --download records the archive hash in src/envs/hydraulic_rig/data/CHECKSUMS.txt; copy it into
EXPECTED_SHA256 below, and every later run refuses an archive whose hash differs.
usage: python3 -I benchmarks/hydraulic_rig/fetch_uci.py --download [--dest src/envs/hydraulic_rig/data/uci_hydraulic]
then:  python3 -I benchmarks/hydraulic_rig/build_blocks.py <folder printed by this script> --check"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.data import DATA_DIR  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
ARCHIVE_NAME = "condition+monitoring+of+hydraulic+systems.zip"
EXPECTED_SHA256 = None  # placeholder until the first approved run; then the hash it records in src/envs/hydraulic_rig/data/CHECKSUMS.txt
SENSOR_FILES = tuple(f"{s}.txt" for s in (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"))
REQUIRED_FILES = SENSOR_FILES + ("profile.txt",)
DEFAULT_DEST = DATA_DIR / "uci_hydraulic"
CHECKSUMS = DATA_DIR / "CHECKSUMS.txt"
USER_AGENT = "Mozilla/5.0"  # a generic agent string, never a person's name or address


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def checksum_name(path):
    """The name CHECKSUMS.txt lists: relative to data/ when the file lies under it, so the check runs from data/."""
    path, base = Path(path).resolve(), CHECKSUMS.parent.resolve()
    return str(path.relative_to(base)) if path.is_relative_to(base) else str(path)


def record_checksum(checksums, name, digest):
    """Append '<sha256>  <name>' (the shasum format, so 'shasum -a 256 -c CHECKSUMS.txt' works from data/) unless the
    name is already listed; a listed name with a different hash means the upstream file changed, which is refused."""
    checksums = Path(checksums)
    lines = checksums.read_text(encoding="utf-8").splitlines() if checksums.exists() else []
    for line in lines:
        old, _, listed = line.partition("  ")
        if listed == name:
            if old != digest:
                raise ValueError(f"{name}: SHA-256 {digest} differs from {old} recorded in {checksums}")
            return False
    with open(checksums, "a", encoding="utf-8") as f:
        f.write(f"{digest}  {name}\n")
    return True


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="actually download; required, see the module docstring")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="a new or empty folder for the archive and its contents")
    a = ap.parse_args(argv)
    if not a.download:
        print(f"Refusing to download {URL} without --download: the archive is fetched only on an explicit "
              "request, so rerun with --download.", file=sys.stderr)
        return 2
    dest = Path(a.dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} is not empty; give a new folder with --dest so the download lands in a fresh directory.", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE_NAME
    download(URL, archive)
    digest = sha256_of(archive)
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        print(f"{archive}: SHA-256 {digest} differs from EXPECTED_SHA256 {EXPECTED_SHA256}; not extracted.", file=sys.stderr)
        return 1
    record_checksum(CHECKSUMS, checksum_name(archive), digest)
    safe_extract(archive, dest / "extracted")
    data_dir = find_data_dir(dest / "extracted")
    print(f"archive {archive} ({archive.stat().st_size} bytes), SHA-256 {digest}")
    if not EXPECTED_SHA256:
        print("first run: copy this SHA-256 into EXPECTED_SHA256 in benchmarks/hydraulic_rig/fetch_uci.py")
    print(f"raw files in {data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
