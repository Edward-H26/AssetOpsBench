"""hyd_1 documents for the tests of CausalDiagnosis/frameworks/AssetOpsBench/tool_servers, from the document provider of the hyd_1 profile
(CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/profile_documents.py), normalised as AssetOpsBench's loader does: the registry entry with its four
children, one fmsr document per class, AssetOpsBench's failure code sample, and telemetry records with the mean under
the installed field name and the std under `<field>_std`. The values are synthetic, since these tests check the
storage and the layer, not the rig.
"""
import json
from datetime import datetime
from pathlib import Path

from couchdb import loader
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import ProfileDocuments
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.cycle_timestamps import Timeline
from CausalDiagnosis.datasets.hyd_1.former_field_names import HYD1_FIELDS

COUCHDB_DIR = Path(loader.__file__).resolve().parent
TIMELINE = Timeline(datetime(2026, 1, 1), 60)
FIELDS = list(HYD1_FIELDS.values())
DOCUMENTS = ProfileDocuments(get_profile("hyd_1"), TIMELINE)
SITE = DOCUMENTS.profile.text["fallback"]["site"]  # the site of the asset of the documents, which the producer writes


def normalise(collection, doc):
    """The loader's own normalisation, so the ids and the dataset field are those load_collection writes."""
    return loader._normalise(doc, collection, loader.collection_config(collection), loader._transform_for(collection))


def registry_documents():
    return [normalise("asset", doc) for doc in DOCUMENTS.registry()]


def failure_mode_documents():
    return [normalise("failure_mode", doc) for doc in DOCUMENTS.failure_modes()]


def failure_code_documents():
    path = COUCHDB_DIR / "scenarios_data" / "shared" / "failure_code" / "failure_code_sample.csv"
    return [normalise("failure_code", doc) for doc in loader.parse_csv(path, loader.collection_config("failure_code"))]


def timestamp(cycle):
    return TIMELINE.timestamp(cycle)


def engine_values(cycle):
    """The served values of a cycle as the engine names them, the same numbers telemetry_record stores."""
    values = {}
    for i, field in enumerate(FIELDS):
        values[f"{field}_mean"] = round(100 + i + 0.01 * cycle, 6)
        values[f"{field}_std"] = round(1 + 0.001 * i, 6)
    return values


def record_from_values(cycle, values):
    return normalise("iot", DOCUMENTS.telemetry(cycle, values))


def telemetry_record(cycle):
    return record_from_values(cycle, engine_values(cycle))



def load(server, names, cycles=range(-10, 0)):
    """Create the five databases with the loader's design document and indexes and insert the documents."""
    documents = {"asset": registry_documents(), "iot": [telemetry_record(c) for c in cycles],
                 "failure_mode": failure_mode_documents(), "workorder": [], "failure_code": failure_code_documents()}
    for collection, name in zip(("asset", "iot", "failure_mode", "workorder", "failure_code"), names.all()):
        config = loader.collection_config(collection)
        server.create_database(name, drop=True)
        if config.get("design_doc"):
            server.install_design(name, json.loads((COUCHDB_DIR / config["design_doc"]).read_text()))
        server.bulk_insert(name, documents[collection])
        server.create_indexes(name, config.get("indexes"))
