"""Fetch the publisher's raw files at one resolved source revision."""

import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path


REPOSITORY = "microsoft/sqlworkshops"
SOURCE_DIR = "SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data"
FILES = {
    "PdM_telemetry.csv": {"datetime", "machineID", "volt", "rotate", "pressure", "vibration"},
    "PdM_errors.csv": {"datetime", "machineID", "errorID"},
    "PdM_failures.csv": {"datetime", "machineID", "failure"},
    "PdM_maint.csv": {"datetime", "machineID", "comp"},
    "PdM_machines.csv": {"machineID", "model", "age"},
}


def download_raw(destination):
    """Download without overwriting, recording the commit and content hashes.

    The hashes identify the retrieved bytes, rather than an independently supplied
    reference release. The destination must contain none of the output files.
    """
    destination = Path(destination)
    outputs = (*FILES, "source.json", "CHECKSUMS.txt")
    if any((destination / name).exists() for name in outputs):
        raise ValueError("raw output files already exist; use a separate destination")
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}/commits/master",
                                     headers={"User-Agent": "maintenance-qa-data"})
    with urllib.request.urlopen(request) as response:
        revision = json.load(response)["sha"]
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("the publisher did not return a commit hash")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": f"https://github.com/{REPOSITORY}", "revision": revision, "files": {}}
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        temporary = Path(temporary)
        for name, required in FILES.items():
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{SOURCE_DIR}/{name}"
            path = temporary / name
            with urllib.request.urlopen(url) as response, path.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not required.issubset(reader.fieldnames or ()):
                    raise ValueError(f"{name}: missing CSV columns {sorted(required)}")
                machines, count = set(), 0
                for row in reader:
                    machines.add(int(row["machineID"]))
                    count += 1
            if not count:
                raise ValueError(f"{name}: empty CSV")
            manifest["files"][name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "rows": count, "machines": len(machines)}
        for name in FILES:
            shutil.move(str(temporary / name), destination / name)
    (destination / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "CHECKSUMS.txt").write_text(
        "".join(f"{entry['sha256']}  {name}\n" for name, entry in manifest["files"].items()), encoding="utf-8")
    return manifest
