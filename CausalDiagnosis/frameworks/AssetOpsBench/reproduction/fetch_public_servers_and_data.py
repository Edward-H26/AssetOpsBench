"""Layer B of the AssetOpsBench reproduction (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md): the paper-era ports over the servers and
the data of IBM's public branches. Nothing of those branches is stored in the fork. prepare fetches them by commit into a
folder outside the repository, once, and the driver reads that folder.

usage, from the fork root with the fork's virtual environment first on PATH:
  python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.fetch_public_servers_and_data --scenarios FILE --out DIR [--recovered-rows FILE]

FILE is the scenario file of Hugging Face commit d925adb. DIR receives:
  servers/      src/servers, src/llm and src/couchdb of branch main-0.x, whose work order CSV files are the same bytes as those
                of archive/legacy-tmp and data/tsfm-data-couchdb
  iot.json      the Chiller 6 telemetry of main-0.x and the three Chiller 9 recordings of data/tsfm-data-couchdb, in one file,
                and the optional rows file over them
  datasets/     the Chiller 9 CSV files, the base folder the TSFM server resolves a relative dataset path against
  scenarios.jsonl  the scenario file with the two forms it lacks, the only rows of another commit
  LAYER.json    the commits, the hashes of what was built and the changes made to public data

The optional rows file is a JSON array of flat objects with asset_id, timestamp and sensor columns, kept outside the repository.
In every month its rows touch they replace all public rows of that asset, and only its SHA-256 and counts are recorded. Build a folder of
its own for it, since its rows change what every scenario reads.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import LAYER_SCENARIO_SHA256, SCENARIO_SHA256, sha256_of
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_sets import RECOVERED_UNLOCKS, SETS
from CausalDiagnosis.utils import download_retry

REPOSITORY = "https://github.com/IBM/AssetOpsBench.git"
SERVERS_COMMIT = "c6b96755b8a182aa090ee028852cbc83279b2982"  # main-0.x
DATA_COMMIT = "d6ab8039bbe176142aeb99fef0fdbf50edc04065"  # data/tsfm-data-couchdb
SERVER_PATHS = ("src/servers", "src/llm", "src/couchdb")
DATA_PATH = "src/couchdb/sample_data/tsfm"
FORMS_REVISION = "5e25bb7f2cd37fb68b9a9e1f99d170ca5be7ce17"
FORMS_SHA256 = "1e015de054b6e4f493bbceca2d5ebd7b464c91b7271cb3804679e2d40b56770a"
FORMS_URL = f"https://huggingface.co/datasets/ibm-research/AssetOpsBench/resolve/{FORMS_REVISION}/data/scenarios/all_utterance.jsonl"
FORMS_FROM_LATER_COMMIT = (621, 622)  # these two carry no form at the pinned commit
CHILLER_6 = "src/couchdb/sample_data/iot/chiller6_june2020_sensordata_couchdb.json"
CHILLER_9 = ("chiller9_tsad", "chiller9_finetuning_small", "chiller9_annotated_small_test")
# Scenario 223 names this file, which is on neither commit this layer fetches, so it is served from chiller9_tsad.csv.
ALIAS = ("data/tsfm_test_data/chiller9_tsad-small.csv", "chiller9_tsad.csv")
IOT_DB, WO_DB, VIBRATION_DB = "chiller", "workorder", "vibration"
LAYER_FILE = "LAYER.json"
# For the spans LAYER.json lists: rows of one asset belong to one span while no gap between them exceeds this.
SPAN_GAP = timedelta(days=1)


def fetch_tree(commit, paths, destination, repository=REPOSITORY):
    """The given paths of one commit under destination, fetched shallowly by the commit's own hash under the retry policy of
    every fetch (CausalDiagnosis/utils/download_retry.py), whose timeout ends a transfer that stalls. Returns its tree hash."""
    with tempfile.TemporaryDirectory(prefix="aob_fetch_") as clone:
        git = ["git", "-C", clone]
        subprocess.run([*git, "init", "-q"], check=True)

        def fetched(timeout):
            stall = ["-c", "http.lowSpeedLimit=1", "-c", f"http.lowSpeedTime={int(timeout)}"]
            return subprocess.run([*git, *stall, "fetch", "-q", "--depth", "1", repository, commit], check=True)

        download_retry.retried(fetched, f"git fetch {repository} {commit}")
        archive = Path(clone) / "tree.tar"
        with open(archive, "wb") as out:
            subprocess.run([*git, "archive", commit, "--", *paths], check=True, stdout=out)
        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter="data")
        return subprocess.run([*git, "rev-parse", f"{commit}^{{tree}}"], check=True, capture_output=True, text=True).stdout.strip()


def with_forms(base_text, later_rows, ids=FORMS_FROM_LATER_COMMIT):
    """The text of the scenario file, the given ids with the form of their later row and every other line byte for byte."""
    lines = base_text.split("\n")
    for i, line in enumerate(lines):
        row = json.loads(line) if line.strip() else None
        if row and int(row["id"]) in ids:
            lines[i] = json.dumps({**row, "characteristic_form": later_rows[int(row["id"])]["characteristic_form"]})
    return "\n".join(lines)


def iso_timestamp(stamp):
    """The form of the Chiller 6 records, 2020-06-01T00:00:00, in the recording's own local time. The IoT server compares
    these strings with such a value in CouchDB, and the Chiller 9 records write a space and a UTC offset instead."""
    return datetime.fromisoformat(stamp).replace(tzinfo=None).isoformat()


def merged_telemetry(chiller_6, chiller_9_files):
    records = json.loads(Path(chiller_6).read_text(encoding="utf-8"))
    for path in chiller_9_files:
        recording = json.loads(Path(path).read_text(encoding="utf-8"))
        records += [{**record, "timestamp": iso_timestamp(record["timestamp"])} for record in recording]
    return records


def recovered_rows(path):
    """The rows of the optional file with the timestamp form of merged_telemetry, without CouchDB's own fields."""
    rows = [{key: value for key, value in row.items() if key not in ("_id", "_rev")} | {"timestamp": iso_timestamp(row["timestamp"])}
            for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    keys = [(row["asset_id"], row["timestamp"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{path} holds an asset and timestamp twice")
    return rows


def unlocked_ids(records):
    """The scenarios of RECOVERED_UNLOCKS whose assets and days the records cover."""
    assets = {record["asset_id"] for record in records}
    days = {(record["asset_id"], record["timestamp"][:10]) for record in records}
    return sorted(i for i, needs in RECOVERED_UNLOCKS.items()
                  if all(asset in assets if day is None else (asset, day) in days for asset, day in needs))


def spans(records):
    """{asset: [[first timestamp, last timestamp, rows]]} of the continuous spans of the records."""
    found = {}
    for asset, stamp in sorted((record["asset_id"], record["timestamp"]) for record in records):
        runs = found.setdefault(asset, [])
        if runs and datetime.fromisoformat(stamp) - datetime.fromisoformat(runs[-1][1]) <= SPAN_GAP:
            runs[-1][1:] = [stamp, runs[-1][2] + 1]
        else:
            runs.append([stamp, stamp, 1])
    return found


def over_public(records, rows):
    """(public records with the rows over them, what the rows did). In a month the rows touch no public row of that asset stays,
    since the two series differ in value and in clock, and a reply must not mix them. A month is cut off at the rows then, which
    is the price of never mixing."""
    covered = spans(rows)
    months = {(row["asset_id"], row["timestamp"][:7]) for row in rows}
    kept, dropped = [], {}
    for record in records:
        asset = record["asset_id"]
        if (asset, record["timestamp"][:7]) in months:
            dropped[asset] = dropped.get(asset, 0) + 1
        else:
            kept.append(record)
    return kept + rows, {"rows": len(rows), "assets": sorted(covered), "spans": covered, "public_dropped": dropped,
                         "public_left": spans(kept)}


def write_telemetry(destination, records, rows_file=None):
    """iot.json, and what LAYER.json records of the optional rows: their hash, counts and the scenarios they unlock."""
    recovered = None
    if rows_file:
        rows = recovered_rows(rows_file)
        records, counts = over_public(records, rows)
        # The IoT server takes an asset's sensor list from the first document a find returns, without an index that means the
        # lowest _id, and real rows can hold a status alone. So the fullest row of every asset gets the lowest _id of that asset.
        records = [{**record, "_id": f"{i:08d}"} for i, record in enumerate(sorted(records, key=len, reverse=True))]
        recovered = {"sha256": sha256_of(rows_file), **counts, "unlocked_ids": unlocked_ids(records)}
    (Path(destination) / "iot.json").write_text(json.dumps(records), encoding="utf-8")
    return records, recovered


def build_datasets(csv_folder, destination):
    destination.mkdir(parents=True)
    for name in CHILLER_9:
        shutil.copy2(Path(csv_folder) / f"{name}.csv", destination / f"{name}.csv")
    alias, source = ALIAS
    (destination / alias).parent.mkdir(parents=True)
    shutil.copy2(destination / source, destination / alias)


def prepare(scenarios, out, fetch=fetch_tree, later_scenarios=None, rows_file=None):
    """Builds the folder of the module docstring, under a temporary name so that a failure or a parallel job leaves no half."""
    out = Path(out)
    if (out / LAYER_FILE).exists():
        built = (provenance(out).get("recovered_rows") or {}).get("sha256")
        if built != (sha256_of(rows_file) if rows_file else None):
            raise SystemExit(f"{out} was built with other rows, build another folder")
        return out
    if sha256_of(scenarios) != SCENARIO_SHA256:
        raise SystemExit(f"{scenarios} is not the scenario file with SHA-256 {SCENARIO_SHA256}")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{out.name}_", dir=out.parent))
    trees = {"servers": fetch(SERVERS_COMMIT, SERVER_PATHS, work / "servers"), "data": fetch(DATA_COMMIT, (DATA_PATH,), work / "data")}
    later = Path(later_scenarios) if later_scenarios else work / "later_scenarios.jsonl"
    if not later_scenarios:
        download_retry.download(FORMS_URL, later)
    if sha256_of(later) != FORMS_SHA256:
        raise SystemExit(f"{later} is not the scenario file of commit {FORMS_REVISION[:7]}")
    rows = {int(json.loads(line)["id"]): json.loads(line) for line in later.read_bytes().decode().split("\n") if line.strip()}
    (work / "scenarios.jsonl").write_bytes(with_forms(Path(scenarios).read_bytes().decode(), rows).encode())
    if sha256_of(work / "scenarios.jsonl") != LAYER_SCENARIO_SHA256:
        raise SystemExit(f"the merged scenario file has SHA-256 {sha256_of(work / 'scenarios.jsonl')}, not {LAYER_SCENARIO_SHA256}")
    csv_folder = work / "data" / DATA_PATH
    records = merged_telemetry(work / "servers" / CHILLER_6, [csv_folder / f"{name}.json" for name in CHILLER_9])
    records, recovered = write_telemetry(work, records, rows_file)
    build_datasets(csv_folder, work / "datasets")
    (work / LAYER_FILE).write_text(json.dumps({
        "format": 1, "repository": REPOSITORY, "servers_commit": SERVERS_COMMIT, "data_commit": DATA_COMMIT, "trees": trees,
        "scenarios": {"base": SCENARIO_SHA256, "forms_revision": FORMS_REVISION, "forms_of": FORMS_FROM_LATER_COMMIT,
                      "merged": LAYER_SCENARIO_SHA256},
        "changes_to_public_data": ["Chiller 9 timestamps of iot.json rewritten to the form of Chiller 6",
                                   f"{ALIAS[0]} is a copy of {ALIAS[1]}"],
        "iot_sha256": sha256_of(work / "iot.json"), "iot_records": len(records), "recovered_rows": recovered}, indent=1), encoding="utf-8")
    shutil.rmtree(work / "data")
    if not later_scenarios:
        later.unlink()
    work.chmod(0o700 if recovered else 0o755)  # the rows stay with their owner on a shared cluster
    try:
        work.rename(out)
    except OSError:
        if not (out / LAYER_FILE).exists():  # a parallel job may have finished first
            raise
    return out


def provenance(layer):
    return json.loads((Path(layer) / LAYER_FILE).read_text(encoding="utf-8"))


def recovered_ids(layer):
    """The set named recovered of this folder: the runnable ids and the ones its rows unlock."""
    recovered = provenance(layer).get("recovered_rows")
    if not recovered:
        raise SystemExit(f"--set recovered needs a folder built with --recovered-rows, and {layer} was not")
    return [*SETS["runnable"], *recovered["unlocked_ids"]]


def server_settings(layer, outputs):
    """What the servers of main-0.x read beyond the CouchDB address: IoT and work order databases, and the TSFM folders."""
    layer = Path(layer)
    return {"IOT_DBNAME": IOT_DB, "WO_DBNAME": WO_DB, "VIBRATION_DBNAME": VIBRATION_DB,
            "PATH_TO_MODELS_DIR": str(layer / "servers/src/servers/tsfm/artifacts/tsfm_models"),
            "PATH_TO_DATASETS_DIR": str(layer / "datasets"), "PATH_TO_OUTPUTS_DIR": str(outputs)}


def load_commands(layer):
    """The loaders of main-0.x with the files of this layer, each dropping its database first: IoT, work orders, vibration."""
    layer = Path(layer)
    couchdb = layer / "servers/src/couchdb"
    assets, orders = str(couchdb / "init_asset_data.py"), str(couchdb / "init_wo.py")
    vibration = str(couchdb / "sample_data/iot/bulk_docs_vibration.json")
    return [[sys.executable, assets, "--data-file", str(layer / "iot.json"), "--db", IOT_DB, "--drop"],
            [sys.executable, orders, "--data-dir", str(couchdb / "sample_data/work_order"), "--db", WO_DB, "--drop"],
            [sys.executable, assets, "--data-file", vibration, "--db", VIBRATION_DB, "--drop"]]


def main(argv):
    ap = argparse.ArgumentParser(description="fetch and build layer B of the AssetOpsBench reproduction")
    ap.add_argument("--scenarios", required=True, help="the scenario file of Hugging Face commit d925adb")
    ap.add_argument("--out", default=os.environ.get("AOB_LAYER_B"), required="AOB_LAYER_B" not in os.environ,
                    help="the folder to build (default AOB_LAYER_B)")
    ap.add_argument("--recovered-rows", default=os.environ.get("AOB_RECOVERED_ROWS"),
                    help="an optional private JSON file of telemetry rows outside the repository (default AOB_RECOVERED_ROWS)")
    a = ap.parse_args(argv)
    print(f"AOB_LAYER_B {prepare(a.scenarios, a.out, rows_file=a.recovered_rows)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
