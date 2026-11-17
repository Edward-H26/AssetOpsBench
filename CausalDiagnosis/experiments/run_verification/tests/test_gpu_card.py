"""The GPU card of record: the endpoint file of a serving job names the card of its GPU, the platform record carries it, the start
check holds a model agent and a judge in the run to the serving file of record and, once env.sh names it, to the card of record,
and names both cards when it differs, a development run plays and records the card it met with the hash of its GPU sampler at the end of the run, and
no table pools runs of two cards. Every endpoint file, card and sampler here is made up, and the model list of an endpoint is a stand-in."""
import json
import re

import pytest

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

MODEL = "made-up-model"
URL = "http://node-a:30000/v1"
JUDGE, JUDGE_URL = "made-up-judge", "http://node-b:30001/v1"
CARD = {"gpu_name": "MADE_UP_GPU_A", "gpu_mib": "81920"}
OTHER = {"gpu_name": "MADE_UP_GPU_B", "gpu_mib": "98304"}
ENGINE = {"weights_commit": "a" * 40, "weights_files_sha256": "d" * 64, "image_sha256": "b" * 64, "template_sha256": "c" * 64,
          "engine_version": "0.5.19", "dtype": "bfloat16", "dtype_launch": "bfloat16", "quantization": "none", "kv_cache_dtype": "auto",
          "launch_args_sha256": "e" * 64}


def _launch(served_name, dtype="bfloat16"):
    """The line launch_args of a made-up endpoint file: the flags in public form as JSON, as engine_record.py writes them."""
    return json.dumps(["--model-path", f"vendor/{served_name}@{ENGINE['weights_commit']}", "--served-model-name", served_name,
                       "--dtype", dtype], separators=(",", ":"))


REGISTERED = ("--config", "base", "--agent", "causal", "--starts", "sampled", "--model", MODEL, "--base-url", URL)


@pytest.mark.real_start_check
@pytest.mark.parametrize("field, value, declared", [("dtype_launch", "auto", "dtype"), ("sglang", "0.5.20", "engine_version"),
                                                    ("weights_commit", "f" * 40, "weights_commit"),
                                                    ("launch_args", _launch(MODEL, "float16"), "launch_args")])
def test_a_run_on_a_server_of_another_precision_is_not_of_record(tmp_path, monkeypatch, models, field, value, declared):
    sampler = _of_record(tmp_path, monkeypatch)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", sampler=sampler, engine={**ENGINE, field: value})))
    a, name, config = _parsed(*REGISTERED)
    reasons = run_episodes.start_check.registration(a, config, name, default_facts_path())
    assert len(reasons) == 1 and reasons[0].startswith(f"the model endpoint differs from the serving file of record: {declared}")


def _sampler(folder):
    """A GPU sampler file as a serving job appends to it."""
    path = folder / "sampler.csv"
    path.write_text("index, utilization.gpu, memory.used\n0, 35, 60000\n", encoding="utf-8")
    return path


def _endpoint(path, card=CARD, url=URL, sampler=None, served_name=MODEL, engine=None):
    """An endpoint file as a serving job writes it: the URL, the job and the engine, and with a card its GPU fields and sampler."""
    lines = [url, f"job 1 node node-a model vendor/{served_name} served_name {served_name}", "started 2026-12-01T00:00:00-05:00",
             *(f"{key} {value}" for key, value in {"launch_args": _launch(served_name), "sglang": "0.5.19",
                                                  **(engine or ENGINE)}.items()), "image /images/sglang.sif"]
    if card is not None:  # the one line of the card that the serving job writes after the line started
        lines.insert(3, f"gpu_name {card['gpu_name']} gpu_mib {card['gpu_mib']} gpu_uuid GPU-00000000-0000-0000-0000-000000000001 "
                        f"driver_version 550.00 cuda_version 12.9 gpu_sampler {sampler}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def models(monkeypatch):
    """The model list of every endpoint as SGLang answers it, without a request."""
    monkeypatch.setattr(ledger.serving_endpoints, "endpoint_models", lambda url: [{"id": MODEL, "max_model_len": 32768}])


def _of_record(tmp_path, monkeypatch, card=CARD, judge_card=None):
    """A tree whose rules pass but the serving and the card: the serving file of record of the made-up engine, the card of record
    in env.sh, the endpoint file of the card and, when given, of the judge card, and the sampler of those endpoints."""
    sampler = _sampler(tmp_path)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "of_record.txt", card=card, sampler=sampler)))
    if judge_card is not None:
        judge = _endpoint(tmp_path / "judge_of_record.txt", card=judge_card, url=JUDGE_URL, sampler=sampler, served_name=JUDGE)
        monkeypatch.setenv("HYD_JUDGE_ENDPOINT_FILE", str(judge))
    block = {"weights_commit": ENGINE["weights_commit"], "dtype": ENGINE["dtype_launch"], "engine_version": "0.5.19",
             "launch_args": {"alone": ["--dtype", "bfloat16"], "co_served": ["--dtype", "bfloat16"]}}
    serving = tmp_path / "serving_of_record.json"
    serving.write_text(json.dumps({"format": 1, "agent": {**block, "hub_id": f"vendor/{MODEL}", "served_name": MODEL},
                                   "judge": {**block, "hub_id": f"vendor/{JUDGE}", "served_name": JUDGE}}), encoding="utf-8")
    monkeypatch.setattr(ledger.serving_endpoints, "SERVING_FILE", serving)
    monkeypatch.setenv("GPU_OF_RECORD", card["gpu_name"].replace("_", " "))
    monkeypatch.setenv("GPU_MIB_OF_RECORD", card["gpu_mib"])
    for function in ("tree_record_reasons", "untracked_loaded", "data_reasons"):  # where the start check reads them
        monkeypatch.setattr(ledger.code_identity, function, lambda *args, **kwargs: [])
    monkeypatch.setattr(ledger.code_identity, "open_reasons", lambda: [])
    return sampler


def _parsed(*argv):
    a = run_episodes.build_parser().parse_args(list(argv))
    name, config, _prices = run_episodes.configure(a)
    return a, name, config


def test_the_endpoint_file_names_the_card_and_the_serving_identity_carries_it(tmp_path, monkeypatch, models):
    sampler = _sampler(tmp_path)
    path = _endpoint(tmp_path / "1.txt", sampler=sampler)
    fields = ledger.endpoint_file(path)
    assert ledger.card_of(fields) == CARD and fields["gpu_sampler"] == str(sampler) and fields["node"] == "node-a"
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(path))
    record = ledger.served(MODEL, URL, "HYD_ENDPOINT_FILE")
    identity = ledger.serving_identity(record)
    assert ledger.card_of(identity) == CARD and (identity["model"], identity["context_length"]) == (MODEL, 32768)
    kept = ledger.serving_record(record)
    assert {key: kept[key] for key in ("gpu_uuid", "driver_version", "cuda_version", "node", "endpoint_file", "matches_endpoint")} == {
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001", "driver_version": "550.00", "cuda_version": "12.9", "node": "node-a",
        "endpoint_file": str(path.resolve()), "matches_endpoint": True}
    assert kept["sampler"] == {"path": str(sampler), "sha256": None, "bytes": None} and ledger.serving_record(None) is None
    # the file of a serving job that wrote no card, and the card as a reason names it
    assert ledger.card_of(ledger.endpoint_file(_endpoint(tmp_path / "old.txt", card=None))) is None and ledger.card_text(None) == "no GPU card"
    assert ledger.card_text(CARD) == f"{CARD['gpu_name'].replace('_', ' ')} with {CARD['gpu_mib']} MiB"


def test_the_platform_record_takes_the_served_names_of_the_endpoint_files_and_carries_the_card(tmp_path, monkeypatch, models):
    sampler = _sampler(tmp_path)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "agent.txt", sampler=sampler)))
    judge = _endpoint(tmp_path / "judge.txt", card=OTHER, url=JUDGE_URL, sampler=sampler, served_name=JUDGE)
    monkeypatch.setenv("HYD_JUDGE_ENDPOINT_FILE", str(judge))
    record = ledger.platform_command(facts_path=default_facts_path())
    agent, judged = record["serving"]["agent"], record["serving"]["judge"]
    assert ledger.card_of(agent) == CARD and agent["model"] == MODEL and {key: agent[key] for key in ENGINE} == ENGINE
    assert ledger.card_of(judged) == OTHER
    assert record == ledger.platform_command(facts_path=default_facts_path(), model=MODEL, judge_model=JUDGE)


@pytest.mark.real_start_check
def test_a_run_on_the_card_of_record_passes_the_start_check(tmp_path, monkeypatch, models):
    sampler = _of_record(tmp_path, monkeypatch)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", sampler=sampler)))
    a, name, config = _parsed(*REGISTERED)
    assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == []


@pytest.mark.real_start_check
def test_a_registered_run_on_another_card_is_refused_before_its_first_episode_and_both_cards_are_named(tmp_path, monkeypatch, models):
    sampler = _of_record(tmp_path, monkeypatch)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", card=OTHER, sampler=sampler)))
    a, name, config = _parsed(*REGISTERED)
    named = f"the model endpoint: the endpoint serves on {ledger.card_text(OTHER)}, and the card of record is {ledger.card_text(CARD)}"
    assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == [named]
    every = run_episodes.start_states.replicate_seeds  # one episode of the test set is enough for the guard
    monkeypatch.setattr(run_episodes.start_states, "replicate_seeds", lambda *args, **kwargs: every(*args, **kwargs)[:1])
    out = tmp_path / "runs"
    with pytest.raises(SystemExit, match=f"--agent causal would play the test combination .*{re.escape(named)}"):
        run_episodes.main([*REGISTERED, "--out", str(out)])
    assert not out.exists()


@pytest.mark.real_start_check
def test_a_run_without_the_card_of_its_own_endpoint_is_not_of_record(tmp_path, monkeypatch, models):
    sampler = _of_record(tmp_path, monkeypatch)
    a, name, config = _parsed(*REGISTERED)
    old = _endpoint(tmp_path / "old.txt", card=None)
    elsewhere = _endpoint(tmp_path / "elsewhere.txt", url="http://node-b:30000/v1", sampler=sampler)
    cases = {
        old: f"the model endpoint: the endpoint serves on no GPU card, and the card of record is {ledger.card_text(CARD)}",
        elsewhere: f"the model endpoint: the endpoint file {elsewhere.resolve()} names http://node-b:30000/v1, not the endpoint of the run",
        tmp_path / "gone.txt": f"the model endpoint: the endpoint file {tmp_path / 'gone.txt'} does not exist",
    }
    _facts, _cycles, blocks = load_world()
    held = sorted(run_episodes.start_check.guarded_starts(config, blocks))[:1]
    for path, reason in cases.items():
        monkeypatch.setenv("HYD_ENDPOINT_FILE", str(path))
        reasons = run_episodes.start_check.registration(a, config, name, default_facts_path())
        assert reason in reasons, (path, reasons)
        with pytest.raises(SystemExit, match=re.escape(reason)):
            run_episodes.guard_test_set("causal", held, run_episodes.start_check.guarded_starts(config, blocks), reasons)
    monkeypatch.delenv("HYD_ENDPOINT_FILE")
    assert "the model endpoint: the endpoint has no endpoint file of its serving job, which would name its GPU card" in run_episodes.start_check.registration(
        a, config, name, default_facts_path())


@pytest.mark.real_start_check
def test_a_judge_in_the_run_is_held_to_the_serving_file_and_the_card_of_record(tmp_path, monkeypatch, models):
    """A run that judges in the run is of record only when the endpoint file of its judge equals the judge of the serving file and
    names the card of record, by the rule of the model endpoint, and a run that does not judge in the run never reads the judge's file."""
    sampler = _of_record(tmp_path, monkeypatch, judge_card=CARD)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", sampler=sampler)))
    judged = (*REGISTERED, "--judge", "local", "--judge-model", JUDGE, "--judge-base-url", JUDGE_URL)
    a, name, config = _parsed(*judged)
    other = f"the judge endpoint: the endpoint serves on {ledger.card_text(OTHER)}, and the card of record is {ledger.card_text(CARD)}"
    for card, reasons in ((CARD, []), (OTHER, [other])):
        judge = _endpoint(tmp_path / "judge_run.txt", card=card, url=JUDGE_URL, sampler=sampler, served_name=JUDGE)
        monkeypatch.setenv("HYD_JUDGE_ENDPOINT_FILE", str(judge))
        assert run_episodes.start_check.registration(a, config, name, default_facts_path()) == reasons
    monkeypatch.delenv("HYD_JUDGE_ENDPOINT_FILE")
    assert "the judge endpoint: the endpoint has no endpoint file of its serving job, which would name its GPU card" in run_episodes.start_check.registration(
        a, config, name, default_facts_path())
    without_judge, name, config = _parsed(*REGISTERED)
    assert run_episodes.start_check.registration(without_judge, config, name, default_facts_path()) == []


@pytest.mark.real_start_check
def test_a_development_run_on_another_card_passes_the_guard(tmp_path, monkeypatch, models):
    sampler = _of_record(tmp_path, monkeypatch)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", card=OTHER, sampler=sampler)))
    a, name, config = _parsed("--config", "base", "--agent", "causal", "--starts", "development", "--model", MODEL, "--base-url", URL)
    reasons = run_episodes.start_check.registration(a, config, name, default_facts_path())
    assert any(ledger.card_text(OTHER) in reason for reason in reasons) and any("--starts development" in reason for reason in reasons)
    _facts, _cycles, blocks = load_world()
    developed = run_episodes.development_config(config, blocks)
    run_episodes.guard_test_set("causal", {tuple(c) for c in developed.allowed}, run_episodes.start_check.guarded_starts(config, blocks), reasons)


def test_a_run_records_the_card_of_its_endpoint_and_the_sampler_at_its_end(tmp_path, monkeypatch, models):
    """A development run of the causal agent with a scripted planner behind a stand-in endpoint, in a tree that is no repository:
    run.json names the card it met and the hash of the sampler at the end, which the sampler keeps as a prefix as it appends."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ScriptedPlanner, _planner_endpoint, act_now

    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(ledger.code_identity, "git_state", lambda paths, root=None: (None, None))
    server, _requests = _planner_endpoint(ScriptedPlanner(act_now))
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    sampler = _sampler(tmp_path)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "run.txt", card=OTHER, url=base, sampler=sampler)))
    try:
        run_dir = run_episodes.main(["--config", "static", "--agent", "causal", "--model", MODEL, "--base-url", base, "--seeds", "1",
                                     "--no-grade", "--out", str(tmp_path / "runs")])
    finally:
        server.shutdown()
    meta = read_json(run_dir / "run.json")
    agent = meta["serving"]["agent"]
    assert meta["stamp"] == "development" and meta["serving"]["judge"] is None
    # run.json names the card and the sampler's hash, and neither the node nor a path of the machine (public_record)
    assert ledger.card_of(agent) == OTHER and (agent["cuda_version"], agent["matches_endpoint"]) == ("12.9", True) and "node" not in agent
    digest, length = ledger.prefix_hash(sampler)
    assert agent["sampler"] == {"sha256": digest, "bytes": length}
    with open(sampler, "a", encoding="utf-8") as f:
        f.write("1, 40, 61000\n")
    assert ledger.prefix_hash(sampler, length) == (digest, length) and ledger.prefix_hash(sampler)[1] > length
    assert ledger.prefix_hash(sampler, length + 10_000) == (None, None) and ledger.prefix_hash(tmp_path / "none.csv") == (None, None)


def _pooled_run(root, name, card, started, seeds, stamp=None):
    """A finished run folder of the causal agent on the static preset whose run.json and ledger name the card of its model endpoint."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    meta = {"label": name, "agent": "causal", "agent_label": "causal_m", "model": MODEL, "config_name": "static", "started": started,
            "ended": started, "seeds": seeds, "asset_id": "hyd_1", "code": {"commit": None, "identity": "a" * 64},
            "serving": {"agent": {**ENGINE, **card}, "judge": None}, **({"stamp": stamp} if stamp else {})}
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    fields = {"file": "/endpoints/1.txt", "lines": [URL], "base_url": URL, "matches_endpoint": True, **ENGINE, **card}
    (run_dir / "ledger.json").write_text(json.dumps({"status": "finished", "events": [], "model": {
        "name": MODEL, "endpoint": URL, "endpoint_file": fields, "endpoint_models": None}}), encoding="utf-8")
    return run_dir


def test_no_table_pools_runs_of_two_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(pooling_rules, "_SELECTED_COMMIT", None)
    registered = tmp_path / "registered"
    right = _pooled_run(registered, "right", CARD, "2026-12-01T00:00:00", [0])
    other = _pooled_run(registered, "other", OTHER, "2026-12-02T00:00:00", [1])
    with pytest.raises(SystemExit, match="fall into more than one group.*serving of the model endpoint"):
        pooling_rules.pooling_refusals([right, other])
    development = tmp_path / "development"
    _pooled_run(development, "earlier", OTHER, "2026-12-01T00:00:00", [0], stamp="development")
    _pooled_run(development, "later", CARD, "2026-12-02T00:00:00", [1], stamp="development")
    left = pooling_rules.pooling_left_out(development)
    assert list(left) == ["earlier"] and left["earlier"].startswith(f"it played on {ledger.card_text(OTHER)} and later, the latest start")
    assert left["earlier"].endswith("runs of two GPU cards are never pooled")
    assert [(entry["run"], entry["card"]) for entry in index_file.inputs(development)] == [("later", CARD)]


JUDGE_HUB = f"vendor/{JUDGE}"  # the line model of the made-up endpoint file of the judge
JUDGE_SETTING = {"HYD_JUDGE_MODEL": JUDGE, "HYD_JUDGE_HUB_ID": JUDGE_HUB, "HYD_JUDGE_REVISION": ENGINE["weights_commit"]}


@pytest.fixture
def judge_gone(monkeypatch):
    """The model list of the agent's endpoint as SGLang answers it, while the judge endpoint, whose job has ended, answers nothing."""
    monkeypatch.setattr(ledger.serving_endpoints, "endpoint_models", lambda url: None if url == JUDGE_URL else [{"id": MODEL, "max_model_len": 32768}])


def _judge_record(tmp_path, monkeypatch):
    """The endpoint file of the serving job of the agent and the judge endpoint record that the judge job left, where the
    platform record reads them."""
    sampler = _sampler(tmp_path)
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(_endpoint(tmp_path / "agent.txt", sampler=sampler)))
    judge = _endpoint(tmp_path / "judge.txt", card=OTHER, url=JUDGE_URL, sampler=sampler, served_name=JUDGE)
    monkeypatch.setenv("HYD_JUDGE_ENDPOINT_FILE", str(judge))
    return judge


def test_the_platform_record_names_the_judge_of_the_setting_after_the_judge_job_has_ended(tmp_path, monkeypatch, judge_gone):
    """The platform record is written in the serving job of the agent model, when the judge job of the registered order has
    ended and its endpoint answers nothing: without the setting the judge has no model, and with it the record names the served
    name, the Hugging Face id and the pinned revision of the setting beside the engine and the card of the judge endpoint record."""
    _judge_record(tmp_path, monkeypatch)
    bare = ledger.platform_command(facts_path=default_facts_path())["serving"]["judge"]
    assert (bare["model"], bare["hub_id"], bare["revision"]) == (None, None, None)
    for variable, value in JUDGE_SETTING.items():
        monkeypatch.setenv(variable, value)
    serving = ledger.platform_command(facts_path=default_facts_path())["serving"]
    judged, agent = serving["judge"], serving["agent"]
    assert (judged["model"], judged["hub_id"], judged["revision"], judged["context_length"]) == (JUDGE, JUDGE_HUB, ENGINE["weights_commit"], None)
    assert {key: judged[key] for key in ENGINE} == ENGINE and ledger.card_of(judged) == OTHER
    assert (agent["model"], agent["context_length"], agent["hub_id"], agent["revision"]) == (MODEL, 32768, None, None)


def test_a_setting_that_differs_from_its_endpoint_file_stops_the_platform_record(tmp_path, monkeypatch, judge_gone):
    """Each key of the setting must equal its line of the endpoint file, for the judge and for the agent alike, and a setting
    needs an endpoint file to check it against: otherwise run_ledger/ platform names the line, both values and writes nothing.
    A name that is empty checks nothing."""
    _judge_record(tmp_path, monkeypatch)
    out = tmp_path / "platform.json"
    facts = ["--facts", str(default_facts_path())]
    cases = (("HYD_JUDGE_MODEL", "another-judge", "judge", "served_name", JUDGE), ("HYD_JUDGE_HUB_ID", "vendor/another", "judge", "model", JUDGE_HUB),
             ("HYD_JUDGE_REVISION", "b" * 40, "judge", "weights_commit", ENGINE["weights_commit"]),
             ("HYD_REVISION", "b" * 40, "agent", "weights_commit", ENGINE["weights_commit"]))
    for variable, value, role, line, found in cases:
        monkeypatch.setenv(variable, value)
        with pytest.raises(SystemExit, match=re.escape(f"of the {role} has {line} {found} where {variable} names {value}")):
            ledger.main(["platform", "--out", str(out), *facts])
        assert not out.exists()
        monkeypatch.delenv(variable)
    monkeypatch.setenv("HYD_JUDGE_MODEL", "")
    assert ledger.main(["platform", "--out", str(out), *facts]) == 0 and read_json(out)["serving"]["judge"]["model"] is None
    out.unlink()
    monkeypatch.delenv("HYD_JUDGE_ENDPOINT_FILE")
    monkeypatch.setenv("HYD_JUDGE_MODEL", JUDGE)
    with pytest.raises(SystemExit, match=re.escape("the setting of the judge (HYD_JUDGE_MODEL) has no endpoint file of HYD_JUDGE_ENDPOINT_FILE")):
        ledger.main(["platform", "--out", str(out), *facts])
    assert not out.exists()

