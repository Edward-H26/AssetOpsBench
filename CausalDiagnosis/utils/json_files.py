import json
from pathlib import Path
from CausalDiagnosis.utils.atomic_writes_and_locks import write_json_atomically


def write_json(path, data):
    """An atomic JSON write; values json cannot encode (paths, tuples as keys) are written through str."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(path, json.loads(json.dumps(data, default=str)), indent=1)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
