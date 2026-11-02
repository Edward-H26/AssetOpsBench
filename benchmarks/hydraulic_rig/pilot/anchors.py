"""Anchors of the pilot (design v2 section 9, docs/hydraulic_rig/benchmarks.md): a scenario date mapped to a recording
cycle and to the anchored start of an episode.

The mapping rests on two assumptions that the mentor has not confirmed, stated in ASSUMPTIONS and printed by every pilot
report: the asset hp_1 of the scenarios is the hyd_1 file with one recorded cycle per day from 2022-07-01, and the ten
dates were chosen for the faults the texts name, which four of the ten do not show in the facts file. The hidden start
of an anchored episode is therefore the recorded condition of the day, read from the facts file at run time and never
written into code.
"""
import json
from datetime import date
from pathlib import Path

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY

DATE0 = "2022-07-01"  # the first day of the hyd_1 file, cycle_index 0 of the facts file
SCENARIOS_PATH = Path(__file__).resolve().with_name("scenarios_17.json")


def load_scenarios(path=SCENARIOS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


ASSUMPTIONS = load_scenarios()["assumptions"]


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
        "n_faulty": sum(1 for c, level in zip(COMPONENTS, cycle.combo) if level != HEALTHY[c]),
        "block_offset": block.index(anchored),
        "block_length": len(block),
    }


def start_of(anchored):
    """The start field of EpisodeConfig for an anchor."""
    return {"combination": list(anchored["combination"]), "block_offset": anchored["block_offset"]}


def assumptions_text():
    """The two assumptions as Markdown lines, printed by every pilot report and run summary."""
    return "\n".join(f"- {a}" for a in ASSUMPTIONS)
