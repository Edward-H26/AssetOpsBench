"""Writing a JSON file through a temporary file and a rename, so that a server stopped mid-write never leaves a torn file
behind; shared by the hydraulic-rig and the causal-diagnosis servers."""
import json
import os
import tempfile
from pathlib import Path


def write_json_atomically(path, obj, indent=None):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)
