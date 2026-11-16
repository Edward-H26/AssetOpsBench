import hashlib
import json
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.episode_launchers import LAUNCH_PRIVATE
from CausalDiagnosis.experiments.run_verification.public_record import PRIVATE_FILE


# what no ledger records: the bytecode that Python writes beside a module it imports from a run folder, as the Plan-Execute
# executor does when it starts the launcher of an episode folder inside the checkout as python -m, and the folder file of Finder
IGNORED_DIRS = frozenset({"__pycache__"})
# and the private files of a run and of the launchers of an episode (public_record.PRIVATE_FILE, common.LAUNCH_PRIVATE), which
# lie outside the published set
IGNORED_NAMES = frozenset({".DS_Store", PRIVATE_FILE, LAUNCH_PRIVATE})


def sha256_of(path):
    """The full SHA-256 of a file (CausalDiagnosis.datasets.dataset_loader.recordings.file_sha256), or None when there is no such file."""
    from CausalDiagnosis.datasets.dataset_loader.recordings import file_sha256

    path = Path(path)
    return file_sha256(path) if path.is_file() else None


def ignored(relative):
    """Whether a path relative to a run folder is one that no ledger records, a name of IGNORED_NAMES or a path through a folder
    of IGNORED_DIRS, which the ledger check (verify_results_folder.py) and the file table of a walkthrough page leave out as well."""
    relative = Path(relative)
    return relative.name in IGNORED_NAMES or bool(IGNORED_DIRS & set(relative.parts))


def file_hashes(folder, exclude=()):
    """{path relative to the folder: {"sha256", "bytes"}} of every file below the folder, in path order, but the paths of exclude
    and those that ignored() leaves out."""
    folder = Path(folder)
    hashes = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder)
        if relative.as_posix() not in exclude and not ignored(relative):
            hashes[relative.as_posix()] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}
    return hashes


def read_manifest(path):
    """{path: SHA-256} of an upload manifest, whose lines are those of sha256sum: the hash, two spaces and the path."""
    return {line[66:]: line[:64] for line in Path(path).read_text(encoding="utf-8").splitlines() if line}


def canonical_sha256(value):
    """The SHA-256 of a JSON value written with sorted keys and no spaces, so two configurations compare by one value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def models_sha256(path):
    """The SHA-256 of the fitted models over their JSON without the top-level key generated, so that two builds on different
    days compare equal."""
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("generated", None)
    return canonical_sha256(value)


def prefix_hash(path, length=None):
    """(SHA-256, length in bytes) of a file as it stands now, or of its first length bytes, or (None, None) when it cannot be
    read or holds fewer bytes. A sampler that keeps appending holds the bytes hashed at the end of a run as the prefix of its
    file, which prefix_hash(path, length) checks later."""
    digest, read = hashlib.sha256(), 0
    try:
        with open(path, "rb") as f:
            while length is None or read < length:
                chunk = f.read(1 << 20 if length is None else min(1 << 20, length - read))
                if not chunk:
                    break
                digest.update(chunk)
                read += len(chunk)
    except OSError:
        return None, None
    if length is not None and read < length:
        return None, None
    return digest.hexdigest(), read
