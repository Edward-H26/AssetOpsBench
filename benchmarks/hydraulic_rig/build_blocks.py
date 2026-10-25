"""Recompute the per-cycle features and the stable blocks of the hydraulic rig from the raw UCI files (docs/hydraulic_rig/environment.md,
release path), so that the released environment depends on CC BY 4.0 data only.

Input: the folder that holds the 17 sensor files (PS1.txt to SE.txt, one row of samples per 60 s cycle) and
profile.txt (one row of condition labels per cycle), as printed by benchmarks/hydraulic_rig/fetch_uci.py.
Output: <out-dir>/hyd_cycles.jsonl, one record per cycle in the IndustryAssetEQA facts layout that
envs.hydraulic_rig.data.load_cycles reads (cycle_index, labels, features named <Quantity>_<Sensor>_<stat>) with the
statistics the environment serves (mean, std, min, max), and <out-dir>/blocks.json, the stable block of every
component-state combination as envs.hydraulic_rig.data.build_blocks forms it from that file.
--check writes nothing: it compares the recomputed labels and statistics with the IndustryAssetEQA facts file for every
cycle and reports which standard-deviation convention (ddof 0 or 1) that file used; writing the outputs then needs
that convention as --std-ddof, so the recomputed features match the file the environment was developed on.
usage: python3 -I benchmarks/hydraulic_rig/build_blocks.py <raw folder> --check [--facts PATH]
       python3 -I benchmarks/hydraulic_rig/build_blocks.py <raw folder> --std-ddof {0,1} [--out-dir src/envs/hydraulic_rig/data]"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.data import (COMPONENTS, DATA_DIR, LABEL_KEYS, SENSORS, SUMMARY_SUFFIXES, build_blocks,  # noqa: E402
                             default_facts_path, load_cycles)

STATS = tuple(s.lstrip("_") for s in SUMMARY_SUFFIXES)  # mean, std, min, max: the statistics the environment serves
QUANTITY = {**{f"PS{i}": "Pressure" for i in range(1, 7)}, "EPS1": "MotorPower", "FS1": "VolumeFlow", "FS2": "VolumeFlow",
            **{f"TS{i}": "Temperature" for i in range(1, 5)}, "VS1": "Vibration", "CE": "CoolingEfficiency",
            "CP": "CoolingPower", "SE": "EfficiencyFactor"}  # feature-name prefixes of the IndustryAssetEQA facts file
PROFILE_COLUMNS = tuple(LABEL_KEYS[c] for c in COMPONENTS) + ("stable_flag",)  # column order of profile.txt
TOL = 1e-9  # relative tolerance with an absolute floor of the same size, as in benchmarks/hydraulic_rig/verify_data_facts.py


def feature_name(sensor, stat):
    return f"{QUANTITY[sensor]}_{sensor}_{stat}"


def moments(values):
    """(n, mean, sum of squared deviations, min, max) of one cycle's samples; math.fsum keeps the sums exactly rounded."""
    n = len(values)
    mean = math.fsum(values) / n
    return n, mean, math.fsum((v - mean) ** 2 for v in values), min(values), max(values)


def summary(m, ddof):
    n, mean, ss, lo, hi = m
    return {"mean": mean, "std": math.sqrt(ss / (n - ddof)) if n > ddof else 0.0, "min": lo, "max": hi}


def read_rows(path):
    """Yield one list of floats per line; every line of a file must hold the same, non-zero number of samples."""
    width = None
    with open(path, encoding="ascii") as f:
        for i, line in enumerate(f, 1):
            vals = [float(x) for x in line.split()]
            width = width or len(vals)
            if not vals or len(vals) != width:
                raise ValueError(f"{path}: line {i} has {len(vals)} samples, expected {width}")
            yield vals


def raw_moments(folder):
    """The label rows of profile.txt and {sensor: moments per cycle}, after checking one row per cycle in every file."""
    folder = Path(folder)
    labels = []
    for row in read_rows(folder / "profile.txt"):
        if len(row) != len(PROFILE_COLUMNS) or any(v != int(v) for v in row):
            raise ValueError(f"{folder / 'profile.txt'}: expected {len(PROFILE_COLUMNS)} integer columns, got {row}")
        labels.append(dict(zip(PROFILE_COLUMNS, (int(v) for v in row))))
    by_sensor = {}
    for s in SENSORS:
        by_sensor[s] = [moments(r) for r in read_rows(folder / f"{s}.txt")]
        if len(by_sensor[s]) != len(labels):
            raise ValueError(f"{s}.txt has {len(by_sensor[s])} rows, profile.txt has {len(labels)}")
    return labels, by_sensor


def records(labels, by_sensor, ddof):
    return [{"cycle_index": i, "labels": lab,
             "features": [{"name": feature_name(s, stat), "value": v} for s in SENSORS for stat, v in summary(by_sensor[s][i], ddof).items()]}
            for i, lab in enumerate(labels)]


def write_blocks(blocks, path):
    data = {"components": list(COMPONENTS), "blocks": [{"combo": list(c), "cycles": idx} for c, idx in sorted(blocks.items())]}
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def write_outputs(recs, out_dir):
    """Write the cycles file, then form the blocks from it with the environment's own loader and block builder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles_path = out_dir / "hyd_cycles.jsonl"
    with open(cycles_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    blocks = build_blocks(load_cycles(cycles_path))
    write_blocks(blocks, out_dir / "blocks.json")
    return cycles_path, blocks


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


def check(labels, by_sensor, facts_path):
    """Compare with the facts file. Returns (problems, ddof): the problems found, and the standard-deviation convention
    that agrees with the file on every value (None when neither does)."""
    facts = {c.index: c for c in load_cycles(facts_path)}
    problems = []
    if sorted(facts) != list(range(len(labels))):
        problems.append(f"the facts file has {len(facts)} cycles, the raw files {len(labels)}")
    std_agree, n_std = {0: 0, 1: 0}, 0
    for i, lab in enumerate(labels):
        c = facts.get(i)
        if c is None:
            continue
        if tuple(lab[k] for k in PROFILE_COLUMNS[:4]) != c.combo or (lab["stable_flag"] == 0) != c.stable:
            problems.append(f"cycle {i}: labels {lab} differ from the facts file")
        for s in SENSORS:
            ours = {d: summary(by_sensor[s][i], d) for d in (0, 1)}
            for stat in STATS:
                name = feature_name(s, stat)
                if name not in c.features:
                    problems.append(f"cycle {i}: {name} missing from the facts file")
                elif stat == "std":
                    n_std += 1
                    for d in (0, 1):
                        std_agree[d] += close(ours[d]["std"], c.features[name])
                elif not close(ours[0][stat], c.features[name]):
                    problems.append(f"cycle {i}: {name} {ours[0][stat]!r} against {c.features[name]!r}")
    ddof = next((d for d in (0, 1) if n_std and std_agree[d] == n_std), None)
    if ddof is None:
        problems.append(f"std agrees with ddof 0 on {std_agree[0]} and with ddof 1 on {std_agree[1]} of {n_std} values")
    return problems, ddof


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw", help="folder with the 17 sensor files and profile.txt")
    ap.add_argument("--check", action="store_true", help="compare with the facts file and write nothing")
    ap.add_argument("--facts", default=str(default_facts_path()), help="IndustryAssetEQA hyd_facts.jsonl")
    ap.add_argument("--std-ddof", type=int, choices=(0, 1), help="standard-deviation convention that --check reported")
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    a = ap.parse_args(argv)
    if not a.check and a.std_ddof is None:
        ap.error("run --check first and pass the ddof it reports as --std-ddof")
    labels, by_sensor = raw_moments(a.raw)
    if a.check:
        problems, ddof = check(labels, by_sensor, a.facts)
        for p in problems[:50]:
            print(p)
        print(f"cycles {len(labels)}; problems {len(problems)}; std convention of the facts file: "
              f"{'ddof ' + str(ddof) if ddof is not None else 'neither'}")
        return 0 if not problems else 1
    cycles_path, blocks = write_outputs(records(labels, by_sensor, a.std_ddof), a.out_dir)
    print(f"{cycles_path}: {len(labels)} cycles; {Path(a.out_dir) / 'blocks.json'}: {len(blocks)} stable blocks of "
          f"{min(map(len, blocks.values()))} to {max(map(len, blocks.values()))} cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
