"""Loading the recorded cycles and building the stable blocks per component-state combination.

v0 reads the IndustryAssetEQA facts file (labels plus per-cycle features); the release path recomputes the
same records from the CC BY 4.0 UCI raw data (docs/hydraulic_rig/environment.md).
"""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

COMPONENTS = ("cooler", "valve", "pump", "accumulator")
LABEL_KEYS = {"cooler": "cooler_condition", "valve": "valve_condition", "pump": "pump_leakage", "accumulator": "accumulator_pressure"}
LEVELS = {"cooler": (3, 20, 100), "valve": (73, 80, 90, 100), "pump": (0, 1, 2), "accumulator": (90, 100, 115, 130)}
HEALTHY = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
SENSORS = ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE")
TIER_SENSORS = {
    1: ("TS1", "TS2", "TS3", "TS4", "VS1", "CE"),
    2: ("TS1", "TS2", "TS3", "TS4", "VS1", "CE", "FS1", "FS2", "CP"),
    3: SENSORS,
}
SUMMARY_SUFFIXES = ("_mean", "_std", "_min", "_max")
STABLE_FLAG_STABLE = 0  # UCI convention: 0 = conditions were stable, 1 = static conditions might not have been reached
DATA_DIR = Path(__file__).resolve().parent / "data"  # what benchmarks/hydraulic_rig/fetch_uci.py downloads and build_blocks.py writes


@dataclass(frozen=True)
class Cycle:
    index: int
    combo: tuple  # (cooler, valve, pump, accumulator)
    stable: bool
    features: dict  # feature name -> value


def combo_of(labels):
    return tuple(labels[LABEL_KEYS[c]] for c in COMPONENTS)


def load_cycles(facts_path):
    cycles = []
    with open(facts_path) as f:
        for line in f:
            rec = json.loads(line)
            feats = {x["name"]: x["value"] for x in rec["features"]}
            cycles.append(Cycle(rec["cycle_index"], combo_of(rec["labels"]), rec["labels"]["stable_flag"] == STABLE_FLAG_STABLE, feats))
    cycles.sort(key=lambda c: c.index)
    return cycles


def build_blocks(cycles):
    """Return {combo: [cycle indices]} using stable cycles only; each block is checked to be contiguous."""
    blocks = {}
    for c in cycles:
        if c.stable:
            blocks.setdefault(c.combo, []).append(c.index)
    for combo, idx in blocks.items():
        if idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError(f"stable cycles of {combo} are not contiguous")
    return blocks


_SENSOR_RE = {s: re.compile(rf"_{s}(_|$)") for s in SENSORS}


def tier_features(cycle, tier):
    """The mean, std, min and max of every sensor in the tier, taken from the cycle's recorded features."""
    allowed = TIER_SENSORS[tier]
    out = {}
    for name, value in cycle.features.items():
        if not name.endswith(SUMMARY_SUFFIXES):
            continue
        for s in allowed:
            if _SENSOR_RE[s].search(name):
                out[name] = value
                break
    return out


def default_facts_path():
    """HYD_FACTS_PATH when set, else the release file that benchmarks/hydraulic_rig/build_blocks.py writes into DATA_DIR."""
    return Path(os.environ.get("HYD_FACTS_PATH", str(DATA_DIR / "hyd_cycles.jsonl")))
