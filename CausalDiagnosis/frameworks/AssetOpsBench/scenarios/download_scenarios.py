"""Fetch the hydraulic scenarios of the AssetOpsBench dataset into the data folder of the asset (CausalDiagnosis/docs/experiments.md).
usage: python -m CausalDiagnosis.frameworks.AssetOpsBench.scenarios.download_scenarios [--out PATH] [--from-file PATH]

The scenarios are the records of data/asset/hydrolicpump_utterance.jsonl (spelled as in the dataset) of the dataset
ibm-research/AssetOpsBench on Hugging Face, taken at the pinned revision REVISION. The SHA-256 of the bytes and the count of
records are checked against the pin before anything is written, and a file that differs is refused. --from-file reads the same
file of a local clone of the dataset under the same checks. The repository holds the pin and no record of the dataset.
"""
from CausalDiagnosis.utils.paths import SRC
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

DATASET = "ibm-research/AssetOpsBench"
REVISION = "5e25bb7f2cd37fb68b9a9e1f99d170ca5be7ce17"
SOURCE_FILE = "data/asset/hydrolicpump_utterance.jsonl"
SOURCE_SHA256 = "654ff0181dc3a8ef586585db45da16a6b36c618ac933279c90ec5aa923a0fa30"
RECORDS = 17
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
USER_AGENT = "Mozilla/5.0"  # a generic agent string, never a person's name or address


def source_url():
    """The resolve address of the source file at the pinned revision. The dataset is public, so no token is sent."""
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILE}"


def source_block():
    """The pin as the fetched file records it, which anchors.load_scenarios compares with this one."""
    return {"dataset": DATASET, "file": SOURCE_FILE, "revision": REVISION, "sha256": SOURCE_SHA256, "records": RECORDS}


def fetch(opener=urllib.request.urlopen):
    """The bytes of the source file. A test hands in its own opener, so that no test opens the network."""
    with opener(urllib.request.Request(source_url(), headers={"User-Agent": USER_AGENT}), timeout=120) as response:
        return response.read()


def verified(raw):
    """The records of the bytes when their SHA-256 and their count are those of the pin, else a ValueError."""
    found = hashlib.sha256(raw).hexdigest()
    if found != SOURCE_SHA256:
        raise ValueError(f"{SOURCE_FILE} has SHA-256 {found}, and the pin of {DATASET} at {REVISION} is {SOURCE_SHA256}")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(records) != RECORDS:
        raise ValueError(f"{SOURCE_FILE} holds {len(records)} records, and the pin of {DATASET} at {REVISION} holds {RECORDS}")
    return records


def scenario_date(record):
    """The one date a scenario names; the text and the failure_event field agree in every record."""
    in_text = sorted(set(DATE.findall(record["text"])))
    in_event = sorted(set(DATE.findall(record.get("failure_event", ""))))
    if len(in_text) != 1 or in_event != in_text:
        raise ValueError(f"scenario {record.get('id')} names the dates {in_text} in its text and {in_event} in its failure event")
    date.fromisoformat(in_text[0])
    return in_text[0]


def build(raw):
    """The scenario file of verified bytes: the pin, the dates and every record as released with the one date it names, in
    the order of the identifiers."""
    scenarios = [{"id": r["id"], "date": scenario_date(r), **{k: v for k, v in r.items() if k != "id"}} for r in verified(raw)]
    scenarios.sort(key=lambda r: r["id"])
    if len({r["id"] for r in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario ids in the source file")
    return {"source": source_block(), "dates": sorted({r["date"] for r in scenarios}), "scenarios": scenarios}


def default_out():
    """scenarios_path() of scenario_anchors.py, which imports the fork's src and the package of this folder."""
    for path in (str(SRC),):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from CausalDiagnosis.frameworks.AssetOpsBench.scenarios.scenario_anchors import scenarios_path

    return scenarios_path()


def main(argv=None, opener=urllib.request.urlopen):
    ap = argparse.ArgumentParser(description=f"fetch the hydraulic scenarios of {DATASET} at revision {REVISION}")
    ap.add_argument("--out", type=Path, default=None,
                    help="the file to write (default: HYD_SCENARIOS_PATH, else scenarios_17.json in the data folder of the asset)")
    ap.add_argument("--from-file", type=Path, default=None,
                    help=f"read {SOURCE_FILE} of a local clone of the dataset in place of the network, under the same checks")
    a = ap.parse_args(argv)
    data = build(a.from_file.read_bytes() if a.from_file else fetch(opener))
    out = a.out or default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['scenarios'])} scenarios of {DATASET} at revision {REVISION}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
