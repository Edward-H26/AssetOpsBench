"""Separate published tool contracts and task messages from private replay data."""

import copy
from pathlib import Path

from .audit import _records, read_json


class FaultArenaRelease:
    """Read an already audited local release, preserving its official partitions."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self._cases = _records(self.folder / "data/all.jsonl")
        self._splits = {name: tuple(_records(self.folder / "data" / f"{name}.jsonl")) for name in ("train", "test")}
        self._tools = {}
        definitions = [read_json(path.read_text(encoding="utf-8"))
                       for path in (self.folder / "tools/toolslist").glob("*.json")]
        aggregate = read_json((self.folder / "tools/tools_schema.json").read_text(encoding="utf-8"))["tools"]
        for row in aggregate:
            name = row["name"]
            if name in self._tools:
                raise ValueError("duplicate tool name")
            # Individual files carry public state_change fields absent from the aggregate.
            matches = [tool for tool in definitions if tool.get("name") == name]
            if len(matches) != 1:
                raise ValueError(f"{name}: expected one individual tool contract")
            tool = matches[0]
            if tool["parameters"] != row["parameters"]:
                raise ValueError(f"{name}: aggregate and individual parameters differ")
            self._tools[name] = tool

    def case_ids(self, split):
        return self._splits[split]

    def task(self, case_id):
        record = self._cases[case_id]
        return copy.deepcopy({"case_id": case_id, "question": record["initial_query"],
                              "initial_states": record["case"]["initial_states"]})

    def modalities(self, case_id):
        record = self._cases[case_id]
        return ("text", "image") if record.get("is_multimodal") or record.get("image_paths") else ("text",)

    def image_paths(self, case_id):
        record = self._cases[case_id]
        paths = set(record.get("image_paths", ()))
        paths.update(item.get("observation", {}).get("image_path") for item in record["case"]["hidden_environment"])
        return sorted(paths - {None, ""})

    def tools(self):
        keys = ("name", "description", "parameters", "returns", "category", "cost", "cost_range",
                "is_destructive", "power_required", "state_change", "multimodal")
        return [{key: copy.deepcopy(tool[key]) for key in keys if key in tool} for tool in self._tools.values()]

    def replay_case(self, case_id):
        """Private simulator input; never an agent tool or planning-message payload."""
        return copy.deepcopy(self._cases[case_id]["case"])
