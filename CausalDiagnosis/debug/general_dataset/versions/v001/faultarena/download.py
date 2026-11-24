"""Fetch an immutable release into generated local storage."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPOSITORY = "faultarena/FaultArena"
REVISION = "1d478b0d0950c4740dc5f05bae9c608e264159cc"
REQUIRED = ("README.md", "LICENSE", "data/all.jsonl", "data/train.jsonl", "data/test.jsonl",
            "tools/tools_schema.json", "metadata/dataset_split.json", "metadata/dataset_summary.json")


def relative_path(name):
    if not isinstance(name, str) or not name:
        raise ValueError("release filenames must be safe relative paths")
    path = PurePosixPath(name)
    if not path.parts or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError("release filenames must be safe relative paths")
    return path


def _fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-generalization"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_release(destination, *, include_images=False, fetch=None):
    """Download without overwriting; hashes identify bytes at the pinned revision.

    data/all.jsonl embeds every complete case, so individual cases/ copies are not
    needed. Images are optional for preparation; missing images are audited and
    multimodal execution must not be treated as text-only benchmark equivalence.
    """
    fetch = fetch or _fetch
    destination = Path(destination)
    if destination.exists():
        raise ValueError("release destination must be new")
    info = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}"))
    if info.get("sha") != REVISION:
        raise ValueError("the server did not resolve the pinned release")
    available = [entry["rfilename"] for entry in info["siblings"]]
    if not set(REQUIRED) <= set(available):
        raise ValueError("the release lacks required files")
    names = sorted(name for name in available if name in REQUIRED or name.startswith(("tools/", "metadata/"))
                   or include_images and name.startswith("images/"))
    if len(set(names)) != len(names):
        raise ValueError("duplicate release filenames")
    for name in names:
        relative_path(name)
    destination.mkdir(parents=True)
    manifest = {"repository": REPOSITORY, "revision": REVISION, "include_images": include_images,
                "status": "downloading", "files": {}}
    record = destination / "source.json"
    record.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def get(name):
        content = fetch(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}")
        path = destination.joinpath(*relative_path(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        return name, {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for name, identity in pool.map(get, names):
                manifest["files"][name] = identity
        manifest["status"] = "downloaded"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        record.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
