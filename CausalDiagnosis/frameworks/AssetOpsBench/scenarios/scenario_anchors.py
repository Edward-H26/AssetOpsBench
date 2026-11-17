"""Scenario anchors (CausalDiagnosis/docs/experiments.md): a scenario date mapped to a recording
cycle and to the anchored start of an episode.

The mapping rests on two assumptions that are not confirmed yet, stated in ASSUMPTIONS and printed by every anchored
report: the asset hp_1 of the scenarios is the hyd_1 file with one recorded cycle per day from 2022-07-01, and the dates
were chosen for the faults the texts name, which the facts file does not show on some of those days. The hidden start of an
anchored episode is therefore the recorded condition of the day, read from the facts file at run time and never written
into code. The scenarios are the file that CausalDiagnosis/frameworks/AssetOpsBench/scenarios/download_scenarios.py fetches into the data folder of the asset, read
when a preset asks for them and never at import.
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios import source_block
from CausalDiagnosis.datasets.dataset_loader.evidence_source import BLOCK_CAP, SEGMENT
from CausalDiagnosis.datasets.hyd_1 import records as hyd_1

DATE0 = "2022-07-01"  # the first day of the hyd_1 file, cycle_index 0 of the facts file
SCENARIOS_FILE = "scenarios_17.json"
SCENARIOS_ENV = "HYD_SCENARIOS_PATH"  # the fetched file, where the data folder of the asset lies elsewhere
FETCH_COMMAND = "python -m CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios"
ASSET_ALIAS = "hp_1"  # the asset id the scenario texts name, which assumption A1 maps onto the asset of the recordings
PROFILE = hyd_1.tables()  # the asset of the recordings that assumption A1 maps the scenario dates onto
ASSUMPTIONS = [
    "A1 (mapping): the asset hp_1 of the scenarios is the hyd_1 file of the AssetOpsBench repository, which lays the "
    "recorded cycles out as one per day from 2022-07-01, so the recording cycle of a date is its number of days since "
    "2022-07-01 (the cycle_index of the facts file; the hyd_1 file numbers the same cycles from 1).",
    "A2 (dates): the dates were chosen for the faults the scenario texts name, and the facts file does not show that fault "
    "on some of those days, so the hidden condition of an anchored episode is the recorded condition of the day and not "
    "the fault the text names. Neither assumption is confirmed by the authors of AssetOpsBench yet.",
]


def scenarios_path():
    """HYD_SCENARIOS_PATH when set, else scenarios_17.json in the data folder of hyd_1, beside its recordings."""
    return Path(os.environ.get(SCENARIOS_ENV) or hyd_1.release_path().parent / SCENARIOS_FILE)


def load_scenarios(path=None):
    """The fetched scenario file, scenarios_path() by default. A missing file raises FileNotFoundError that names the command
    which fetches it, and a file whose source block is not the pin of download_scenarios.py raises ValueError."""
    path = Path(path or scenarios_path())
    if not path.is_file():
        raise FileNotFoundError(f"no scenario file at {path}: run {FETCH_COMMAND}, or set {SCENARIOS_ENV} to the file it wrote")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source") != source_block():
        raise ValueError(f"{path} records the source {data.get('source')}, and download_scenarios.py pins {source_block()}: run {FETCH_COMMAND}")
    return data


def cycle_of(day, date0=DATE0):
    """The recording cycle (facts cycle_index) of a date under assumption A1: one cycle per day from date0."""
    offset = (date.fromisoformat(day) - date.fromisoformat(date0)).days
    if offset < 0:
        raise ValueError(f"{day} lies before the first recorded day {date0}")
    return offset


def anchor(day, cycles, blocks, date0=DATE0):
    """The anchored start of a date: the recorded combination of that day's cycle and the position of the cycle in the
    combination's stable block. A day whose cycle is not stable is anchored at the nearest stable cycle of the same
    combination, and the result says so, since only stable cycles are served."""
    index = cycle_of(day, date0)
    by_index = {c.index: c for c in cycles}
    if index not in by_index:
        raise ValueError(f"{day} maps to cycle {index}, which the facts file does not hold")
    cycle = by_index[index]
    block = blocks[cycle.combo]
    anchored = index if cycle.stable else min(block, key=lambda i: (abs(i - index), i))
    return {
        "date": day,
        "date0": date0,
        "cycle_index": index,
        "stable": cycle.stable,
        "anchored_cycle_index": anchored,
        "combination": list(cycle.combo),
        "n_faulty": sum(1 for c, level in zip(PROFILE.components, cycle.combo) if level != PROFILE.healthy[c]),
        "block_position": block.index(anchored),
        "block_length": len(block),
    }


def start_of(anchored):
    """The start field of EpisodeConfig for an anchor: the entry offset whose live segment opens with the anchored recording,
    five positions before it, or a ValueError when the block serves no such recording."""
    position = anchored["block_position"]
    if position >= BLOCK_CAP:
        raise ValueError(f"{anchored['date']} is anchored at block position {position}, and only the first {BLOCK_CAP} recordings of a combination are served")
    return {"combination": list(anchored["combination"]), "entry_offset": (position + SEGMENT) % BLOCK_CAP}


# The scenario set "anchored" of the declaration of AssetOpsBench (SCENARIO_SETS): this module itself, whose DATE0, ASSET_ALIAS, ASSUMPTIONS,
# PROFILE, scenarios_path, load_scenarios, anchor and start_of the runner and the ledger read (run_setup.scenario_set). A test
# that replaces one of them on the module reaches the runner too.
ANCHORED = sys.modules[__name__]
