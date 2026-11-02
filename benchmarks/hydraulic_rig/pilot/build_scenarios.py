"""Write pilot/scenarios_17.json from the AssetOpsBench dataset clone (docs/hydraulic_rig/benchmarks.md).
usage: python3 -I benchmarks/hydraulic_rig/pilot/build_scenarios.py --clone <path to the hf clone> [--out <json>]

The 17 hydraulic scenarios are the hydrolic_pump configuration of the dataset (data/asset/hydrolicpump_utterance.jsonl,
spelled as in the file). Every record keeps its fields as released; the date of a scenario is the one date its text
names, which equals the date of its failure_event field. The output records the clone commit and the SHA-256 of the
source file, so a reader can check the copy against the dataset.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

SOURCE_FILE = Path("data/asset/hydrolicpump_utterance.jsonl")
DATASET = "ibm-research/AssetOpsBench"
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
ASSUMPTIONS = [
    "A1 (mapping): the asset hp_1 of the scenarios is the hyd_1 file of the AssetOpsBench repository, which lays the "
    "recorded cycles out as one per day from 2022-07-01, so the recording cycle of a date is its number of days since "
    "2022-07-01 (the cycle_index of the facts file; the hyd_1 file numbers the same cycles from 1).",
    "A2 (dates): the ten dates were chosen for the faults the scenario texts name, and four of the ten do not show that "
    "fault in the facts file on that day, so the hidden condition of an anchored episode is the recorded condition of "
    "the day and not the fault the text names. Neither assumption is confirmed by the mentor yet.",
]


def scenario_date(record):
    """The one date a scenario names; the text and the failure_event field agree in every record."""
    in_text = sorted(set(DATE.findall(record["text"])))
    in_event = sorted(set(DATE.findall(record.get("failure_event", ""))))
    if len(in_text) != 1 or in_event != in_text:
        raise ValueError(f"scenario {record.get('id')} names the dates {in_text} in its text and {in_event} in its failure event")
    date.fromisoformat(in_text[0])
    return in_text[0]


def build(clone):
    clone = Path(clone)
    source = clone / SOURCE_FILE
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    scenarios = [{"id": r["id"], "date": scenario_date(r), **{k: v for k, v in r.items() if k != "id"}} for r in records]
    scenarios.sort(key=lambda r: r["id"])
    if len({r["id"] for r in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario ids in the source file")
    try:
        commit = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "source": {"dataset": DATASET, "file": str(SOURCE_FILE), "commit": commit, "sha256": hashlib.sha256(raw).hexdigest(),
                   "records": len(scenarios)},
        "assumptions": ASSUMPTIONS,
        "dates": sorted({r["date"] for r in scenarios}),
        "scenarios": scenarios,
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True, help="the local clone of the Hugging Face dataset ibm-research/AssetOpsBench")
    ap.add_argument("--out", default=str(Path(__file__).resolve().with_name("scenarios_17.json")))
    a = ap.parse_args(argv)
    data = build(a.clone)
    Path(a.out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {data['source']['records']} scenarios on {len(data['dates'])} dates from commit {data['source']['commit']}")


if __name__ == "__main__":
    main(sys.argv[1:])
