"""Writing a JSON file through a temporary file and a rename, so that a server stopped mid-write never leaves a torn file
behind, and a lock around a read, a change and a write of a file that several processes add to, the digest of the code that
writes a generated table and the refusal to write one in a process that may only read them; shared by the hydraulic-rig
and the causal-diagnosis servers."""
import fcntl
import hashlib
import json
import os
import secrets
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


@contextmanager
def replaced_atomically(path):
    """A text file to write in place of path: a temporary file beside it, renamed over path when the block ends and removed
    when the block raises, so a failed write leaves the folder as it was. It is created as an ordinary new file, so its mode
    follows the umask of the process, as the run folders that other users read need. The folder of path is made first, as
    locked() makes it, since an output may lie under a folder that no earlier step made, as the runs root of a tree on a new root."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_atomically(path, obj, indent=None):
    with replaced_atomically(path) as f:
        json.dump(obj, f, indent=indent)


@contextmanager
def locked(path):
    """An exclusive lock on a file beside path, held while a process reads path, adds to it and writes it back, so that two
    processes that add at once keep each other's additions instead of the last writer dropping the first, as StateFileSessions
    of the metering layer locks the state file of an episode. The lock file stays behind, empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


READ_ONLY_TABLES = "CD_TABLES_READ_ONLY"  # 1 in a process that may read the generated tables and may not write them
TABLES_JOB = "precompute_tables.sbatch"  # the one process that writes the tables of a clone that cluster jobs share


class GeneratedTableMissing(RuntimeError):
    """A generated table that a process reads is missing or was written by other code, in a process that may not write it."""


def refuse_writing(what):
    """Stop before a process computes and writes a generated table (a label table, a grade, the models file) when it runs with
    CD_TABLES_READ_ONLY=1: on a clone that cluster jobs share one process writes the tables, the tables job, so that no table
    changes under a run in flight, and a registered run reads the tables as they are and computes none."""
    if os.environ.get(READ_ONLY_TABLES) == "1":
        raise GeneratedTableMissing(f"{what} is missing or was written by other code, and this process only reads the tables "
                                    f"({READ_ONLY_TABLES}=1): compute them first with the tables job, {TABLES_JOB}")


@lru_cache(maxsize=None)
def files_digest(root, asset_id):
    """The SHA-256 over the relative path and the SHA-256 of every .py file under root outside the folders named tests and
    the folders of other assets, in path order, computed once per process, root and asset: the code digest that a generated
    table records of the code that wrote it. The folder of an asset is a folder of root/assets that holds an asset.json, and a
    port adds its own, which writes no table of another asset. Root may also be a tuple of folders, read in its order, and
    the folder of an asset may also stand directly under a folder of root."""
    h = hashlib.sha256()
    for folder in map(Path, (root,) if isinstance(root, str) else root):
        assets = (*folder.glob("*/asset.json"), *(folder / "assets").glob("*/asset.json"))
        others = {p.parent for p in assets if p.parent.name != asset_id}
        for path in sorted(p for p in folder.rglob("*.py")
                           if "tests" not in p.relative_to(folder).parts and others.isdisjoint(p.parents)):
            h.update(path.relative_to(folder).as_posix().encode())
            h.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return h.hexdigest()
