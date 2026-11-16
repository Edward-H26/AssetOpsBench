"""CausalDiagnosis/experiments/run_verification/public_record.py: the public form of a record of a run, the command line, the paths and the published set,
and the launchers whose paths and store credentials stand in the private file beside them."""
import json

from CausalDiagnosis.experiments.episode_runners import episode_launchers, servers_of_a_run
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import public_record
from CausalDiagnosis.datasets.dataset_loader.recordings import REPO, public_path, resolve_recorded_path

SERVED = {"name": "m", "endpoint": "http://10.1.2.3:31000/v1",
          "endpoint_file": {"file": "/scratch/u/endpoints/4242.txt", "lines": ["http://10.1.2.3:31000/v1", "job 4242 node node082"],
                            "base_url": "http://10.1.2.3:31000/v1", "job": "4242", "node": "node082", "gpu_uuid": "GPU-1", "gpu_name": "H100",
                            "gpu_sampler": "/scratch/u/logs/gpu-4242.csv", "image": "/scratch/u/images/sglang.sif", "weights_commit": "a" * 40},
          "endpoint_models": [{"id": "m", "max_model_len": 32768, "root": "/scratch/u/models/m", "owned_by": "sglang"}]}


def test_a_record_in_public_form_names_no_endpoint_host_job_card_id_credential_or_machine_path():
    record = {"model": SERVED, "job": {"slurm_job_id": "4242", "host": "node082"}, "backend": {"kind": "couchdb", "url": "http://node082:5984",
              "username": "admin", "password": "secret"}, "facts_path": str(REPO / "CausalDiagnosis" / "datasets" / "hyd_1" / "data" / "hyd_cycles.jsonl"),
              "serving": {"agent": {"node": "node082", "endpoint_file": "/scratch/u/endpoints/4242.txt", "gpu_uuid": "GPU-1",
                                    "sampler": {"path": "/scratch/u/logs/gpu-4242.csv", "sha256": "s", "bytes": 3}}},
              "source": "https://huggingface.co/Qwen/x", "other": "http://node082:8000/v1"}
    public = public_record.public_value(record)
    text = json.dumps(public)
    assert all(word not in text for word in ("10.1.2.3", "node082", "4242", "GPU-1", "secret", "/scratch", "admin"))
    assert public["model"]["endpoint"] is None and public["model"]["endpoint_file"] == {"gpu_name": "H100", "image": "sglang.sif",
                                                                                         "weights_commit": "a" * 40, "base_url": None}
    assert public["model"]["endpoint_models"] == [{"id": "m", "max_model_len": 32768}] and public["backend"] == {"kind": "couchdb", "url": None}
    assert public["facts_path"] == "CausalDiagnosis/datasets/hyd_1/data/hyd_cycles.jsonl" and public["serving"]["agent"] == {"endpoint_file": None,
                                                                                                                       "sampler": {"sha256": "s", "bytes": 3}}
    assert public["source"] == "https://huggingface.co/Qwen/x" and public["other"] is None
    assert public_record.public_value(public) == public  # the form is stable, so a record read back and written again is equal


def test_a_command_line_keeps_its_options_and_names_no_machine_path_or_endpoint():
    argv = ["--config", "base", "--out", "/home/u/runs", "--base-url", "http://10.0.0.1:31000/v1", "--judge-model", "tokenrouter/judge",
            f"--facts={REPO / 'src/envs/x.jsonl'}", "--seeds", "1"]
    assert public_record.public_argv(argv) == ["--config", "base", "--out", "runs", "--base-url", "", "--judge-model", "tokenrouter/judge",
                                               "--facts=src/envs/x.jsonl", "--seeds", "1"]
    assert public_path("/home/u/a/b.json") == "b.json" and public_path("relative/c.json") == "relative/c.json" and public_path(None) is None
    assert resolve_recorded_path("CausalDiagnosis/datasets/dataset_loader/recordings.py") == REPO / "CausalDiagnosis" / "datasets" / "dataset_loader" / "recordings.py" and resolve_recorded_path("nowhere.json") is None


def test_the_published_set_leaves_out_the_private_files_of_the_run_and_of_the_launchers(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "seed000000").mkdir(parents=True)
    for rel in ("run.json", "ledger.json", "seed000000/episode.json", "seed000000/iot_server.py"):
        (run_dir / rel).write_text("{}", encoding="utf-8")
    public_record.add_private(run_dir, "starts", {"job": {"slurm_job_id": "4242", "host": "node082"}})
    (run_dir / "seed000000" / episode_launchers.LAUNCH_PRIVATE).write_text("{}", encoding="utf-8")
    assert public_record.published_files(run_dir) == ["ledger.json", "run.json", "seed000000/episode.json", "seed000000/iot_server.py"]
    assert (run_dir / public_record.PRIVATE_FILE).stat().st_mode & 0o777 == 0o600
    assert public_record.read_private(run_dir)["starts"][0]["job"]["host"] == "node082"
    assert ledger.ignored(public_record.PRIVATE_FILE) and ledger.ignored(f"seed000000/{episode_launchers.LAUNCH_PRIVATE}")


def test_a_launcher_of_a_store_with_credentials_names_neither_the_store_nor_a_path_of_the_machine(tmp_path):
    store = {"kind": "couchdb", "url": "http://node082:5984", "username": "admin", "password": "s3cret"}
    paths = episode_launchers.write_launchers(tmp_path, ("iot", "causal_diagnosis"), store=store, models_path=tmp_path / "models.json",
                                   data_file=("HYD_FACTS_PATH", tmp_path / "facts.jsonl"))
    texts = "".join(path.read_text(encoding="utf-8") for path in paths.values())
    assert all(word not in texts for word in ("node082", "s3cret", "admin", str(tmp_path), str(servers_of_a_run.SRC)))
    private = json.loads((tmp_path / episode_launchers.LAUNCH_PRIVATE).read_text(encoding="utf-8"))
    assert private["options"]["iot"]["store"] == store and private["options"]["causal_diagnosis"]["models_path"] == str(tmp_path / "models.json")
    assert private["src"] == [str(servers_of_a_run.ROOT), str(servers_of_a_run.SRC)] and private["environment"] == {"HYD_FACTS_PATH": str(tmp_path / "facts.jsonl")}
    assert (tmp_path / episode_launchers.LAUNCH_PRIVATE).stat().st_mode & 0o777 == 0o600
