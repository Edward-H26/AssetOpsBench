"""Check the data facts of docs/hydraulic_rig/environment.md on the two local files. Apart from the
file hashes, every check is a relation that holds whatever the counts, so the script states no value of the data.
The checks describe the IndustryAssetEQA facts file: on another facts file, such as one that build_blocks.py rebuilt
from the UCI archive, only the hash check runs and fails, and build_blocks.py --check compares the two files instead.

Read-only: opens the IndustryAssetEQA facts file and the AssetOpsBench hyd_1 file, downloads nothing, writes nothing.
It does not import envs.hydraulic_rig, so benchmarks/hydraulic_rig/tests/test_data_facts.py can compare the environment's blocks with this recomputation.
usage: python3 -I benchmarks/hydraulic_rig/verify_data_facts.py [--facts PATH] [--hyd1 PATH]"""
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

SRC = Path(__file__).resolve().parents[2] / "src"
DEFAULT_FACTS = Path(os.environ.get("HYD_FACTS_PATH", str(SRC / "envs" / "hydraulic_rig" / "data" / "hyd_cycles.jsonl")))
DEFAULT_HYD1 = SRC / "couchdb" / "scenarios_data" / "shared" / "iot" / "hydraulic_pump_1.json"
FACTS_SHA256 = "7912b1b641254d54daffa40ea2fe84c64dedf50edb2d996951ad1f3d9a433e47"  # the IndustryAssetEQA facts file
HYD1_SHA256 = "622bcd1a668a939a2590b1aaf560ecce641731b04d802271b39ef454835918b5"  # the AssetOpsBench hyd_1 file
NOT_EQA = ("The facts file is not the IndustryAssetEQA file that these checks describe, so only the hash check runs. "
           "A file that build_blocks.py rebuilt from the UCI archive is compared with that file by build_blocks.py --check.")

COMPONENTS = ("cooler", "valve", "pump", "accumulator")
LABEL_KEYS = ("cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure")
HEALTHY = (100, 100, 0, 130)
CHANNELS = ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE")
HYD1_PREFIX = {**{c: c for c in CHANNELS}, "EPS1": "P1"}  # hyd_1 names the motor power channel P1
HYD1_META = ("asset_id", "cycle", "timestamp")
SUMMARY_STATS = ("mean", "std", "min", "max")

# Most means in the two files differ in the last digits, so agreement is tested to a relative and absolute 1e-9;
# the shifted-pairing control shows that this tolerance still separates neighbouring cycles.
MEAN_TOL = 1e-9


class Check(NamedTuple):
    fact: str
    expected: Any
    observed: Any
    note: str = ""

    @property
    def passed(self):
        return self.observed == self.expected


def load_facts(path):
    with open(path) as f:
        return sorted((json.loads(line) for line in f), key=lambda r: r["cycle_index"])


def load_hyd1(path):
    with open(path) as f:
        return sorted(json.load(f), key=lambda r: r["cycle"])


def combination(record):
    return tuple(record["labels"][k] for k in LABEL_KEYS)


def is_stable(record):
    return record["labels"]["stable_flag"] == 0


def stable_blocks(facts):
    """{combination: cycle indices of its stable cycles in cycle order}, the environment's blocks per docs/hydraulic_rig/environment.md."""
    blocks = {}
    for r in facts:
        if is_stable(r):
            blocks.setdefault(combination(r), []).append(r["cycle_index"])
    return blocks


def run_kinds(facts):
    """Counter of the runs of identical combination in cycle order: entirely stable, entirely not stable, or mixed."""
    kinds = Counter()
    for _, run in itertools.groupby(facts, key=combination):
        flags = {is_stable(r) for r in run}
        kinds["mixed" if len(flags) == 2 else "stable" if True in flags else "not stable"] += 1
    return kinds


def channel_stat(name):
    """(channel, statistic) of a per-channel feature name such as Pressure_PS1_mean; None for the other features."""
    parts = name.split("_")
    return (parts[1], "_".join(parts[2:])) if len(parts) >= 3 and parts[1] in CHANNELS else None


def feature_layout(record):
    """({channel: statistic names}, feature names that belong to no single channel) of one record."""
    stats, other = {}, []
    for f in record["features"]:
        cs = channel_stat(f["name"])
        if cs:
            stats.setdefault(cs[0], []).append(cs[1])
        else:
            other.append(f["name"])
    return stats, other


def channel_means(record):
    return {cs[0]: f["value"] for f in record["features"] if (cs := channel_stat(f["name"])) and cs[1] == "mean"}


def same_mean(a, b):
    return math.isclose(a, b, rel_tol=MEAN_TOL, abs_tol=MEAN_TOL)


def relative_difference(a, b):
    return abs(a - b) / max(abs(a), abs(b)) if a or b else 0.0


def paired_means(eqa_means, hyd1, field, channel, offset):
    """(hyd_1 value, EQA mean) for every hyd_1 cycle c with an EQA record at cycle_index c + offset."""
    return [(r[field], eqa_means[r["cycle"] + offset][channel]) for r in hyd1 if r["cycle"] + offset in eqa_means]


def whole_set_matches(eqa_means, hyd1, fields, offset):
    """hyd_1 cycles c whose 17 channel means all agree with EQA's at cycle_index c + offset."""
    return sum(1 for r in hyd1 if r["cycle"] + offset in eqa_means
               and all(same_mean(r[fields[c]], eqa_means[r["cycle"] + offset][c]) for c in CHANNELS))


def file_checks(facts_path, hyd1_path):
    """The two inputs are the files the facts were verified on (SHA-256 values recorded in docs/hydraulic_rig/environment.md)."""
    return [Check("SHA-256 of the EQA facts file and of the hyd_1 file", (FACTS_SHA256, HYD1_SHA256),
                  (sha256(facts_path), sha256(hyd1_path)))]


def source_checks(facts, hyd1):
    """Bullet "Source of record": cycles of 60 s, the 17 channels, one label row per cycle."""
    seconds = {p["samples_per_cycle"] / p["nominal_sample_rate_hz"] for r in facts for p in r["sensor_profiles"]}
    listed = sum(1 for r in facts if tuple(p["sensor"] for p in r["sensor_profiles"]) == CHANNELS)
    labelled = sum(1 for r in facts if set(r["labels"]) == {*LABEL_KEYS, "stable_flag"})
    return [
        Check("cycles in the hyd_1 file, against the EQA facts file", len(facts), len(hyd1)),
        Check("cycle length in s, EQA cycle_duration_sec", {60}, {r["cycle_duration_sec"] for r in facts}),
        Check("cycle length in s, samples per cycle over sample rate of every sensor", {60}, seconds),
        Check("EQA records whose sensor profiles are PS1 to PS6, EPS1, FS1, FS2, TS1 to TS4, VS1, CE, CP, SE",
              len(facts), listed),
        Check("EQA records with one label row: four conditions plus the stable flag", len(facts), labelled),
    ]


def eqa_checks(facts):
    """Bullet "IndustryAssetEQA's hyd_facts.jsonl": index range, one feature layout, provenance to the UCI files."""
    indices = [r["cycle_index"] for r in facts]
    stats, other = feature_layout(facts[0])
    (per_channel,) = {tuple(v) for v in stats.values()}  # every channel carries the same statistics
    summarised = sum(1 for c in CHANNELS if set(SUMMARY_STATS) <= set(stats.get(c, ())))
    sensor_files = {c: f"{c}.txt" for c in CHANNELS}
    traced = sum(1 for r in facts if r["provenance"]["sensor_files"] == sensor_files
                 and r["provenance"]["profile_file"] == "profile.txt" and r["provenance"]["cycle_row"] == r["cycle_index"])
    return [
        Check("EQA cycle_index runs from 0 without gaps", list(range(len(facts))), indices),
        Check("distinct feature counts across EQA records", 1, len({len(r["features"]) for r in facts})),
        Check("distinct feature name lists across EQA records", 1, len({tuple(f["name"] for f in r["features"]) for r in facts})),
        Check("channels with mean, std, min and max features", len(CHANNELS), summarised,
              f"{len(stats)} channels x {len(per_channel)} statistics ({', '.join(per_channel)}) "
              f"+ {len(other)} others ({', '.join(other)})"),
        Check("EQA records citing the 17 UCI sensor files, profile.txt and cycle_row = cycle_index", len(facts), traced),
    ]


def hyd1_checks(facts, hyd1):
    """Bullet "AssetOpsBench's hyd_1 asset": cycles, timestamps, no labels, means equal to EQA's at cycle_index c-1."""
    cycles = [r["cycle"] for r in hyd1]
    stamps = [r["timestamp"] for r in hyd1]
    keys = set().union(*hyd1)
    channel_keys = sorted(k for k in keys if k.split("_")[0] in HYD1_PREFIX.values())
    field = {k.split("_")[0]: k for k in channel_keys}
    eqa_means = {r["cycle_index"]: channel_means(r) for r in facts}
    fields = {c: field[HYD1_PREFIX[c]] for c in CHANNELS}
    pairs = {c: paired_means(eqa_means, hyd1, fields[c], c, -1) for c in CHANNELS}
    ps1 = pairs["PS1"]
    every = [p for c in CHANNELS for p in pairs[c]]
    shifted = max(sum(same_mean(a, b) for a, b in paired_means(eqa_means, hyd1, field["PS1"], "PS1", off)) for off in (-2, 0))
    worst = {c: max(relative_difference(a, b) for a, b in pairs[c]) for c in CHANNELS}
    worst_channel = max(worst, key=worst.get)
    shifted_sets = max(whole_set_matches(eqa_means, hyd1, fields, off) for off in (-2, 0))
    increasing = all(a < b for a, b in zip(stamps, stamps[1:]))
    day_gaps = {(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(stamps, stamps[1:])}
    return [
        Check("hyd_1 cycle runs from 1 without gaps", list(range(1, len(hyd1) + 1)), cycles),
        Check("hyd_1 timestamps distinct and increasing with cycle", (len(hyd1), True), (len(set(stamps)), increasing)),
        Check("hyd_1 timestamps one day apart from cycle to cycle", {1}, day_gaps),
        Check("hyd_1 channel fields, EPS1 named P1", sorted(HYD1_PREFIX.values()), sorted(k.split("_")[0] for k in channel_keys)),
        Check("hyd_1 fields other than the 17 channels (no labels)", list(HYD1_META), sorted(keys - set(channel_keys))),
        Check("hyd_1 asset_id", {"hyd_1"}, {r["asset_id"] for r in hyd1}),
        Check("PS1 means agreeing, hyd_1 cycle c against EQA cycle_index c-1", len(hyd1), sum(same_mean(a, b) for a, b in ps1)),
        Check("means agreeing over all 17 channels with the same pairing", len(CHANNELS) * len(hyd1),
              sum(same_mean(a, b) for a, b in every)),
        Check("control: PS1 means agreeing with the pairing shifted by one cycle either way", 0, shifted),
        Check("largest relative difference of the paired means within the agreement tolerance", True,
              worst[worst_channel] <= MEAN_TOL),
        Check("cycles whose 17 means all agree with the pairing shifted by one cycle either way", 0, shifted_sets),
    ]


def label_checks(facts):
    """Bullet "Labels and levels": the level sets, the meaning of the stable flag and the valve level of the cycles flagged not stable."""
    levels = {c: sorted({r["labels"][k] for r in facts}) for c, k in zip(COMPONENTS, LABEL_KEYS)}
    flags = Counter(r["labels"]["stable_flag"] for r in facts)
    unstable_valve = sorted({r["labels"]["valve_condition"] for r in facts if not is_stable(r)})
    regime = sorted({(fm["evidence"], fm["failure_code"]) for r in facts for fm in r["failure_modes"]
                     if fm["component"] == "system_regime"})
    return [
        Check("cooler condition levels", [3, 20, 100], levels["cooler"]),
        Check("valve condition levels", [73, 80, 90, 100], levels["valve"]),
        Check("internal pump leakage levels", [0, 1, 2], levels["pump"]),
        Check("accumulator pressure levels in bar", [90, 100, 115, 130], levels["accumulator"]),
        Check("stable flag values", [0, 1], sorted(flags)),
        Check("stable flag meaning in EQA's system_regime annotation",
              [("stable_flag=0", "steady"), ("stable_flag=1", "transient")], regime),
        Check("cycles flagged stable or not stable", len(facts), flags[0] + flags[1]),
        Check("valve levels of the cycles flagged not stable", [100], unstable_valve),
    ]


def ok_levels(facts, component, label_key):
    """Levels of one component that EQA labels <component>_ok."""
    return sorted({r["labels"][label_key] for r in facts for fm in r["failure_modes"]
                   if fm["component"] == component and fm["failure_code"] == f"{component}_ok"})


def healthy_checks(facts):
    """Bullet "Healthy state": EQA's *_ok levels and the stable cycles of the healthy combination."""
    checks = [Check(f"{c} levels labelled {c}_ok by EQA", [h], ok_levels(facts, c, k))
              for c, k, h in zip(COMPONENTS, LABEL_KEYS, HEALTHY)]
    return checks + [
        Check("the healthy combination (100, 100, 0, 130) has stable cycles", True, bool(stable_blocks(facts).get(HEALTHY))),
    ]


def combination_checks(facts):
    """Bullet "Combinations": every combination of the levels occurs, with one contiguous stable block each."""
    combos = [combination(r) for r in facts]
    counts = Counter(combos)
    blocks = stable_blocks(facts)
    contiguous = sum(1 for idx in blocks.values() if idx == list(range(idx[0], idx[0] + len(idx))))
    n_levels = [len({c[i] for c in combos}) for i in range(len(COMPONENTS))]
    kinds = run_kinds(facts)
    return [
        Check("combinations that occur, against the product of the level counts", math.prod(n_levels), len(counts)),
        Check("combinations whose stable cycles form one contiguous block", len(counts), contiguous),
        Check("runs entirely stable, one per combination, and mixed runs", (len(counts), 0), (kinds["stable"], kinds["mixed"])),
    ]


def collect_checks(facts, hyd1):
    return (source_checks(facts, hyd1) + eqa_checks(facts) + hyd1_checks(facts, hyd1) + label_checks(facts)
            + healthy_checks(facts) + combination_checks(facts))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(DEFAULT_FACTS))
    ap.add_argument("--hyd1", default=str(DEFAULT_HYD1))
    a = ap.parse_args(argv)
    for name, path in (("EQA facts", a.facts), ("hyd_1", a.hyd1)):
        print(f"{name}: {path} ({Path(path).stat().st_size} bytes)")
    checks = file_checks(a.facts, a.hyd1)
    if sha256(a.facts) == FACTS_SHA256:
        checks += collect_checks(load_facts(a.facts), load_hyd1(a.hyd1))
    else:
        print(NOT_EQA)
    for c in checks:
        note = f"  [{c.note}]" if c.note else ""
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.fact}: expected {c.expected}, observed {c.observed}{note}")
    failed = sum(not c.passed for c in checks)
    print(f"{len(checks) - failed} of {len(checks)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
