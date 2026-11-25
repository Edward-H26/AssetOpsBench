"""Discover declared episode ports without importing a recorded-level engine."""

import importlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from CausalDiagnosis.utils.paths import DATASETS


@dataclass(frozen=True)
class EpisodeDataset:
    id: str
    kind: str
    adapter: str
    inputs: tuple
    folder: Path

    def source_identity(self):
        module = importlib.import_module(self.adapter.partition(":")[0])
        origin = Path(module.__file__).resolve()
        paths = {self.folder / "episode.json", origin}
        for folder, dirs, files in os.walk(self.folder):
            dirs[:] = [name for name in dirs if name not in {"data", "tests", "__pycache__"}]
            paths.update(Path(folder) / name for name in files if name.endswith(".py"))
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}

    def load(self, inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError(f"{self.id} requires exactly these inputs: {', '.join(self.inputs)}")
        for key, path in inputs.items():
            if not Path(path).exists():
                raise ValueError(f"missing input {key}: {path}")
        module, separator, name = self.adapter.partition(":")
        if not separator or not module or not name:
            raise ValueError(f"{self.id}: adapter must name module:factory")
        return getattr(importlib.import_module(module), name)({key: Path(value).resolve() for key, value in inputs.items()})


def discover(roots=None):
    """A port registers by adding episode.json in its dataset folder."""
    found = {}
    for root in roots if roots is not None else (DATASETS,):
        for path in sorted(Path(root).glob("*/episode.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if set(raw) != {"id", "kind", "adapter", "inputs"}:
                raise ValueError(f"{path}: invalid episode declaration keys")
            identifier = raw["id"]
            if not isinstance(identifier, str) or not identifier.isidentifier() or identifier in found:
                raise ValueError(f"{path}: invalid or duplicate episode dataset id")
            if not all(isinstance(raw[key], str) and raw[key] for key in ("kind", "adapter")):
                raise ValueError(f"{identifier}: kind and adapter must be nonempty strings")
            inputs = raw["inputs"]
            if not isinstance(inputs, list) or not inputs or not all(isinstance(key, str) and key.isidentifier() for key in inputs) or len(set(inputs)) != len(inputs):
                raise ValueError(f"{identifier}: inputs must be distinct names")
            found[identifier] = EpisodeDataset(identifier, raw["kind"], raw["adapter"], tuple(inputs), path.parent)
    return found
