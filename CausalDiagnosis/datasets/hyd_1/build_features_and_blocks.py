"""Recompute the per-cycle features and the stable blocks of the hydraulic rig from the raw UCI files (CausalDiagnosis/docs/environment.md,
release path), so that the released environment depends on CC BY 4.0 data only.

Input: the folder that holds the 17 sensor files (PS1.txt to SE.txt, one row of samples per 60 s cycle) and
profile.txt (one row of condition labels per cycle), as printed by CausalDiagnosis/datasets/hyd_1/download_and_build.py.
Output: <out-dir>/hyd_cycles.jsonl, one record per cycle in the IndustryAssetEQA facts layout that
CausalDiagnosis.datasets.dataset_loader.recordings.load_cycles reads (cycle_index, labels, features named <Quantity>_<Sensor>_<stat>) with the
statistics the environment serves (mean, std, min, max), and <out-dir>/blocks.json, the stable block of every
component-state combination as CausalDiagnosis.datasets.dataset_loader.recordings.build_blocks forms it from that file. The tables come from the asset
file of hyd_1, whose record builder this script is. CausalDiagnosis/datasets/hyd_1/download_and_build.py calls build() and installs the
two files when their SHA-256 equals its pin in CausalDiagnosis/datasets/hyd_1/data/CHECKSUMS.txt.
The build gives the same bytes on every machine: the sums of a cycle are exactly rounded, every deviation is squared by one
multiplication and the standard deviation divides by the number of samples less one, so no library function of the
platform enters a value.
usage: python3 -I -m CausalDiagnosis.datasets.hyd_1.build_features_and_blocks <raw folder> --check
           builds both files into a temporary folder and compares their SHA-256 with the pins of CHECKSUMS.txt
       python3 -I -m CausalDiagnosis.datasets.hyd_1.build_features_and_blocks <raw folder> --cross-check <IndustryAssetEQA facts file>
           compares the rebuilt labels and statistics with that file, cycle by cycle, and names its std convention
       python3 -I -m CausalDiagnosis.datasets.hyd_1.build_features_and_blocks <raw folder> --out-dir <folder>
           writes both files into the folder"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the fork root, which holds CausalDiagnosis/, also outside an editable install
from CausalDiagnosis.datasets.dataset_loader import recordings as asset_records  # noqa: E402
from CausalDiagnosis.datasets.hyd_1 import records as hyd_1  # noqa: E402

PROFILE = hyd_1.tables()
COMPONENTS, LABEL_KEYS, SENSORS = PROFILE.components, PROFILE.records["label_keys"], tuple(PROFILE.channels)
STATS = tuple(PROFILE.statistics["recorded"])  # mean, std, min, max: the statistics the environment serves
QUANTITY = {**{f"PS{i}": "Pressure" for i in range(1, 7)}, "EPS1": "MotorPower", "FS1": "VolumeFlow", "FS2": "VolumeFlow",
            **{f"TS{i}": "Temperature" for i in range(1, 5)}, "VS1": "Vibration", "CE": "CoolingEfficiency",
            "CP": "CoolingPower", "SE": "EfficiencyFactor"}  # feature-name prefixes of the IndustryAssetEQA facts file
PROFILE_COLUMNS = tuple(LABEL_KEYS[c] for c in COMPONENTS) + ("stable_flag",)  # column order of profile.txt
TOL = 1e-9  # relative tolerance with an absolute floor of the same size, as in CausalDiagnosis/datasets/hyd_1/verify_data_facts.py
STD_DDOF = 1  # the convention of the file the environment was developed on, which the cross-check reports
EQA_FACTS_URL = ("https://raw.githubusercontent.com/ChathurangiShyalika/EmbodiedQA/d7f42a703751cba077d99bae34e8c5fa4bcfa9fe/"
                 "data/outputs/hydraulic/hyd_facts.jsonl")
EQA_FACTS_SHA256 = "7912b1b641254d54daffa40ea2fe84c64dedf50edb2d996951ad1f3d9a433e47"  # the IndustryAssetEQA facts file at that commit
CYCLES_FILE = hyd_1.release_path().name
BLOCKS_FILE = "blocks.json"
BUILT = (CYCLES_FILE, BLOCKS_FILE)  # the files of a build, in the order they are written and checked


def feature_name(sensor, stat):
    return f"{QUANTITY[sensor]}_{sensor}_{stat}"


def moments(values):
    """(n, mean, sum of squared deviations, min, max) of one cycle's samples; math.fsum keeps the sums exactly rounded. A
    deviation is squared by one multiplication, which every machine rounds alike, where a library power need not."""
    n = len(values)
    mean = math.fsum(values) / n
    return n, mean, math.fsum((v - mean) * (v - mean) for v in values), min(values), max(values)


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
    cycles_path = out_dir / CYCLES_FILE
    with open(cycles_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    blocks = hyd_1.build_blocks(hyd_1.load_cycles(cycles_path))
    write_blocks(blocks, out_dir / BLOCKS_FILE)
    return cycles_path, blocks


def build(raw, out_dir):
    """Both files of the data of hyd_1 from the raw folder, with the standard deviation over n less one, into out_dir."""
    labels, by_sensor = raw_moments(raw)
    return write_outputs(records(labels, by_sensor, STD_DDOF), out_dir)


def pinned_lines(raw):
    """(one line per built file, whether any differs): a fresh build in a temporary folder against the pins of
    CHECKSUMS.txt of the release folder. A file without a pin differs."""
    pins = asset_records.checksums(hyd_1.profile())
    lines, differs = [], False
    with tempfile.TemporaryDirectory() as folder:
        build(raw, folder)
        for name in BUILT:
            got, pin = asset_records.file_sha256(Path(folder) / name), pins.get(name)
            differs = differs or got != pin
            lines.append(f"{name}: built {got}, pinned {pin}, {'equal' if got == pin else 'differs'}")
    return lines, differs


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


def cross_check(labels, by_sensor, facts_path):
    """Compare with the facts file. Returns (problems, ddof): the problems found, and the standard-deviation convention
    that agrees with the file on every value (None when neither does)."""
    facts = {c.index: c for c in hyd_1.load_cycles(facts_path)}
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
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="build into a temporary folder and compare with the pins of CHECKSUMS.txt")
    mode.add_argument("--cross-check", metavar="FILE", help="compare the rebuilt records with the IndustryAssetEQA facts file at FILE")
    mode.add_argument("--out-dir", metavar="DIR", help="write hyd_cycles.jsonl and blocks.json into DIR")
    a = ap.parse_args(argv)
    if a.check:
        lines, differs = pinned_lines(a.raw)
        for line in lines:
            print(line)
        return 1 if differs else 0
    if a.cross_check is not None:
        if asset_records.file_sha256(a.cross_check) != EQA_FACTS_SHA256:
            print(f"{a.cross_check} is not the IndustryAssetEQA facts file that this comparison is made with (SHA-256 "
                  f"{EQA_FACTS_SHA256}). Fetch it from {EQA_FACTS_URL}.", file=sys.stderr)
            return 1
        labels, by_sensor = raw_moments(a.raw)
        problems, ddof = cross_check(labels, by_sensor, a.cross_check)
        for p in problems[:50]:
            print(p)
        print(f"cycles {len(labels)}; problems {len(problems)}; std convention of the facts file: "
              f"{'ddof ' + str(ddof) if ddof is not None else 'neither'}")
        return 0 if not problems and ddof == STD_DDOF else 1
    cycles_path, _blocks = build(a.raw, a.out_dir)
    print(f"wrote {cycles_path} and {Path(a.out_dir) / BLOCKS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
