"""The counts and the statistics of the recordings that CausalDiagnosis/docs/dataset.md holds back until a CPU job of the cluster
has reproduced them from the pushed code, printed as one JSON object per value with its name and its group.

    python -m CausalDiagnosis.evaluation.result_tables.datasheet_statistics [--facts PATH]

Group C counts the recordings of the data file: the stable cycles of the smallest and of the largest block, the cycles of the
file and the stable cycles of the block of the fault-free combination. Group R holds statistics of the public recordings alone,
and no policy, model or agent is played: the fault-free cycles that fall outside the range of the other fault-free cycles when
each is left out, the drift inside a block against independent cycles simulated at its size, and how far that drift reaches
against the gaps between blocks. The asset is that of the preset of the first experiment. The test of the page imports these
functions, so the page and the job read one definition.
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world  # noqa: E402  (common puts this checkout's src first on sys.path)
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENTS_PRESET  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.evidence_source import RecordingSource, mean_field  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import starts_of_test_set  # noqa: E402

SIM_DRAWS, SIM_SEED = 2000, 0  # the simulated reference of independent cycles, per block size


def block_rows(profile, cycles, blocks):
    """{combination: the served statistics of every stable cycle of its block, in recording order}, each a mapping from the
    public field of a statistic of a channel to its value, every served statistic of every channel."""
    source = RecordingSource(profile, cycles, blocks)
    channels = tuple(profile.channels)
    return {key: [source.served_values(i, channels) for i in sorted(block)] for key, block in blocks.items()}


def half_difference(values):
    """The mean of the first half of a block's cycles minus the mean of the last half, halves of floor(n / 2) cycles, over the
    sample standard deviation of all n cycles."""
    h = len(values) // 2
    return (statistics.mean(values[:h]) - statistics.mean(values[-h:])) / statistics.stdev(values)


def block_sizes(blocks):
    """The stable cycles of the smallest and of the largest block."""
    lengths = [len(block) for block in blocks.values()]
    return {"smallest": min(lengths), "largest": max(lengths)}


def healthy_left_out(profile, cycles, blocks):
    """The stable cycles of the block of the fault-free combination, and how many of them fall outside the range of
    the others on at least one channel mean when each is left out, the normal range of the public profile as a leave one out."""
    source = RecordingSource(profile, cycles, blocks)
    fields = [mean_field(channel, profile) for channel in profile.channels]
    served = [source.served_values(i, tuple(profile.channels)) for i in blocks[profile.healthy_state()]]
    outside = 0
    for i, own in enumerate(served):
        rest = served[:i] + served[i + 1:]
        outside += any(not min(v[f] for v in rest) <= own[f] <= max(v[f] for v in rest) for f in fields)
    return {"cycles": len(served), "outside": outside}


def drift(rows):
    """|half_difference| per block and served field with a nonzero spread, the number of such pairs, their median, the
    largest with its field and its combination, and the median of the same quantity on independent standard normal cycles
    simulated at the size of each block."""
    found, sizes = [], {}
    for key, block in rows.items():
        for field in block[0]:
            values = [row[field] for row in block]
            if statistics.stdev(values) > 0:
                found.append((abs(half_difference(values)), field, key))
                sizes[len(values)] = sizes.get(len(values), 0) + 1
    rng, reference = random.Random(SIM_SEED), []
    for n in sorted(sizes):
        simulated = statistics.median(abs(half_difference([rng.gauss(0, 1) for _ in range(n)])) for _ in range(SIM_DRAWS))
        reference += [simulated] * sizes[n]
    largest = max(found)
    return {"pairs": len(found), "median": statistics.median(d for d, _field, _key in found), "largest": largest[0],
            "largest_field": largest[1], "largest_combination": list(largest[2]), "reference": statistics.median(reference)}


def crossing(rows, blocks, test_combinations):
    """Every served field standardized by its standard deviation within a block, pooled over the blocks, and the blocks
    whose late half mean lies nearer to the early half mean of another block than to their own early half, with how many of
    them are nearest to the block recorded right after them and how many are test combinations."""
    fields = list(next(iter(rows.values()))[0])
    halves, spread = {}, {f: [] for f in fields}
    for key, block in rows.items():
        h = len(block) // 2
        halves[key] = tuple({f: statistics.mean(row[f] for row in part) for f in fields} for part in (block[:h], block[-h:]))
        for f in fields:
            spread[f].append((statistics.variance([row[f] for row in block]), len(block) - 1))
    pooled = {f: math.sqrt(sum(v * d for v, d in parts) / sum(d for _v, d in parts)) for f, parts in spread.items()}
    pooled = {f: s for f, s in pooled.items() if s > 0}

    def distance(a, b):
        return math.sqrt(sum(((a[f] - b[f]) / s) ** 2 for f, s in pooled.items()))

    order = sorted(blocks, key=lambda k: min(blocks[k]))
    after = dict(zip(order, order[1:]))
    crossed = []
    for key, (early, late) in halves.items():
        nearest = min((distance(late, halves[other][0]), other) for other in halves if other != key)
        if nearest[0] < distance(late, early):
            crossed.append((key, nearest[1]))
    return {"crossed": len(crossed), "blocks": len(blocks), "next": sum(after.get(k) == other for k, other in crossed),
            "test": sum(tuple(k) in test_combinations for k, _other in crossed)}


def properties(facts=None):
    """Every held value as {"name", "group", "value"}, in the order of the held sentences of the page."""
    config = load_config(AGENTS_PRESET)[1]
    profile = get_profile(config.asset_id, engine=False)
    _path, cycles, blocks = load_world(facts, config.asset_id)
    rows, healthy = block_rows(profile, cycles, blocks), healthy_left_out(profile, cycles, blocks)
    test_combinations = {tuple(k) for k in starts_of_test_set(profile, list(blocks))}
    return [{"name": "block_sizes", "group": "C", "value": block_sizes(blocks)},
            {"name": "cycles", "group": "C", "value": len(cycles)},
            {"name": "healthy_cycles", "group": "C", "value": healthy["cycles"]},
            {"name": "healthy_left_out", "group": "R", "value": healthy["outside"]},
            {"name": "drift", "group": "R", "value": drift(rows)},
            {"name": "crossing", "group": "R", "value": crossing(rows, blocks, test_combinations)}]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the counts and the statistics of the recordings that the datasheet holds back.")
    ap.add_argument("--facts", default=None, help="the data file of the asset (default: its environment variable or its release file)")
    for item in properties(ap.parse_args(argv).facts):
        print(json.dumps(item))


if __name__ == "__main__":
    main()
