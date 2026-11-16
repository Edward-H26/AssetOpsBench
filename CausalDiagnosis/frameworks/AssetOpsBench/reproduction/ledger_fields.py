"""The kind of run that the AssetOpsBench reproduction writes (reproduction/run_scenarios/), as the run ledger reads it
(CausalDiagnosis/experiments/run_verification/run_ledger/): its settings, its data (the scenario file and the CouchDB manifest
with every data file it loads), its scorer code and the files its scorer writes after the run. The declaration names RUN_KIND
under RUN_KINDS, and the ledger resolves it by the kind its run.json records."""
import fnmatch
import os
import json
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.utils.paths import ROOT

KIND = "aob_repro"  # the kind of run.json that run_scenarios/ writes
SETTINGS = ("set", "scenario_ids", "repeats", "agents", "temperature", "episode_timeout_s",
            "fmsr_model", "reload_db", "servers", "server_env", "versions")
# the code that judges a reproduction run, hashed into the event of every scoring, since score_runs.py runs after the run and
# may differ from the code run.json recorded when the run started
SCORER_CODE = ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/score_runs.py", "CausalDiagnosis/frameworks/AssetOpsBench/reproduction/scenario_file.py", "CausalDiagnosis/frameworks/AssetOpsBench/reproduction/scenario_sets.py", "CausalDiagnosis/frameworks/AssetOpsBench/reproduction/run_folder_names.py", "CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/episode_judge.py",
               "CausalDiagnosis/utils/llm_backends.py", "CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/full_trajectory_scorer.py", "src/llm/openai_compat.py",
               "CausalDiagnosis/utils/model_request_retry.py", "CausalDiagnosis/utils/model_reply_room.py", "src/evaluation/scorers/llm_judge.py",
               "src/evaluation/evaluator.py")
SCORER_OUTPUTS = ("scores.csv", "judge_*.json")  # the names of the files score_runs.py writes into a run folder
FOUND_SCORES = "judge records and scores that score_runs.py wrote before the ledger existed"
NOT_REPLAYED = "not replayed, an AssetOpsBench reproduction run, which score_runs.py judges"


def couchdb_manifest():
    """The manifest src/couchdb/init_data.py loads when the reproduction reloads CouchDB before an episode: DEFAULT_MANIFEST,
    else default/manifest.json under SCENARIOS_DATA_DIR or under src/couchdb/scenarios_data, as init_data.py reads them."""
    data_dir = os.environ.get("SCENARIOS_DATA_DIR", str(ROOT / "src" / "couchdb" / "scenarios_data"))
    return Path(os.environ.get("DEFAULT_MANIFEST", os.path.join(data_dir, "default", "manifest.json")))


def couchdb_files(manifest=None):
    """{name: path} of the CouchDB manifest and of every data file it names, each resolved as src/couchdb/loader.py resolves
    it: against the manifest's folder, then that folder's parent, then the couchdb folder. A document written inline in the
    manifest is covered by the manifest's own hash."""
    manifest = Path(manifest or couchdb_manifest())
    roots = (manifest.parent, manifest.parent.parent, ROOT / "src" / "couchdb")
    files = {"couchdb_manifest": manifest}
    for value in json.loads(manifest.read_text(encoding="utf-8")).values():
        for relative in (r for r in (value if isinstance(value, list) else [value]) if isinstance(r, str)):
            files[f"couchdb:{relative}"] = next((root / relative for root in roots if (root / relative).exists()), roots[0] / relative)
    return files


def data_files(meta):
    """{name: {"path", "sha256"}} of the data a reproduction run reads: the scenario file, whose hash run.json keeps as well,
    and the CouchDB manifest with its data files."""
    from CausalDiagnosis.experiments.run_verification.run_ledger import sha256_of

    paths = {"scenarios": meta["scenarios"]["path"], **couchdb_files()}
    return {name: {"path": str(path), "sha256": sha256_of(path)} for name, path in paths.items()}


def fields(meta, serve):
    """The fields of a ledger that a reproduction run fills from its run.json: the labels, the model as serve() finds it, no
    judge (score_runs.py judges after the run, and every scoring enters as an event), the data and the settings."""
    from CausalDiagnosis.experiments.run_verification.run_ledger import canonical_sha256

    agents = "+".join(meta.get("agents") or [])
    model = serve(meta.get("model"), meta.get("base_url"), "HYD_ENDPOINT_FILE")
    if model and model["endpoint_models"] is None:
        model["endpoint_models"] = meta.get("endpoint_models")  # what the endpoint answered when the run started
    settings = {key: meta.get(key) for key in SETTINGS}
    return {"label": meta["kind"], "agent": agents, "agent_label": agents, "model": model, "judge": None,
            "data": data_files(meta),
            "configuration": {"settings": settings, "settings_sha256": canonical_sha256(settings)}, "seeds": None}


def scorer_output(path):
    """Whether a path of a run folder names a file that score_runs.py writes after a reproduction run."""
    return any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in SCORER_OUTPUTS)


# what the run ledger, the replay and the check of a results folder ask of this kind of run
RUN_KIND = SimpleNamespace(name=KIND, fields=fields, scorer_output=scorer_output, scorer_code=SCORER_CODE,
                           found_scores=FOUND_SCORES, not_replayed=NOT_REPLAYED, after_the_fact=True)
