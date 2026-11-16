"""The evidence kind demo_history of the made-up port: a data file of JSON lines, each one record of the press from before
the start of an episode, with its level and one value per channel. A start shows every record of its combination from
before cycle 0, no cycle of the episode plays a record, and the faulty start has one recorded work order."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from CausalDiagnosis.datasets.splits import development_split
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader.ground_truth import LevelsTruth

HISTORY = ({"id": "hist/1", "component": "press", "closed_t": -4, "mode": "standard", "outcome": "closed"},)


@dataclass(frozen=True)
class Record:
    index: int
    combo: tuple  # the level of the press
    values: dict  # channel -> value


class Records(list):
    """The records of a data file in file order, with facts_sha256, the SHA-256 of the file."""

    def __init__(self, items=(), facts_sha256=None):
        super().__init__(items)
        self.facts_sha256 = facts_sha256


def record_problems(data):
    """The check of the kind: the records table names the column of the level."""
    key = data["records"].get("level_key")
    return [] if isinstance(key, str) and key else ["records.level_key must name the column of the level of the press"]


def load(path, profile):
    """The records of the data file and, for each combination, the indices of its records, after the checks of the data file
    of the asset (records.check_data)."""
    records.check_data(profile)
    raw = Path(path).read_bytes()
    key = profile.records["level_key"]
    items = Records((Record(rec["index"], (rec[key],), {c: rec[c] for c in profile.channels}) for rec in map(json.loads, raw.splitlines())),
                    hashlib.sha256(raw).hexdigest())
    blocks = {}
    for record in items:
        blocks.setdefault(record.combo, []).append(record.index)
    return items, blocks


class HistorySource:
    """The source of the kind: the records of a start before cycle 0, and no record for any cycle of the episode."""

    def __init__(self, profile, cycles, blocks):
        self.profile, self.cycles, self.blocks = profile, cycles, blocks
        self.window = None
        self.normal_ranges = self._normal_ranges()

    def start_candidates(self, config):
        if config.allowed is not None:
            return sorted(k for k in self.blocks if config.within_fault_rule(k) and k in {tuple(a) for a in config.allowed})
        moved = development_split.moved_to_development(self.profile, self.blocks)
        return sorted(k for k in self.blocks if config.within_fault_rule(k) and k not in moved)

    def starts(self, combination):
        return [(0, 0)] if combination in self.blocks else []

    def forget(self):
        self.window = None

    def open(self, state, t, rng, offset=None):
        self.window = {"start": 0, "offset": 0}

    def prehistory(self, state):
        return list(self.blocks[state])

    def play(self):
        return None

    def served_values(self, index, channels):
        values = self.cycles[index].values
        return {f"{self.profile.field_name(c)}_value": float(f"{values[c]:.6g}") for c in channels}

    def fix_history(self, state):
        return [] if tuple(state) == self.profile.healthy_state() else [dict(entry) for entry in HISTORY]

    def alarms(self, played, since, channels):
        first = {}
        for t, index in played:
            if t < since or index is None:
                continue
            for field, value in self.served_values(index, channels).items():
                band = self.normal_ranges[field]
                if field not in first and not band["low"] <= value <= band["high"]:
                    first[field] = t
        return [{"channel": field, "first_t": t} for field, t in sorted(first.items(), key=lambda kv: (kv[1], kv[0]))]

    def _normal_ranges(self):
        served = [self.served_values(i, tuple(self.profile.channels)) for i in self.blocks[self.profile.healthy_state()]]
        return {field: {"low": min(v[field] for v in served), "high": max(v[field] for v in served)} for field in served[0]}


class MarkedTruth(LevelsTruth):
    """The truth of the kind: the recorded levels truth, with a mark on every call of recovered, so that a test sees which
    truth model the verifier read."""

    calls = []

    def recovered(self, final):
        MarkedTruth.calls.append(dict(final))
        return super().recovered(final)
