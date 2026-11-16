"""judge_finished_runs.py fills only the empty judge cells of finished gym runs of model agents and of static runs and leaves the rest of the
row alone, and judges a run whose run.json names no judge model only under the overrides of the command line."""
import csv
import json
import threading
from types import SimpleNamespace

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import judge_finished_runs as rejudge
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
from CausalDiagnosis.utils.csv_rows import write_rows


def _run(tmp_path, agent, cells, judge_model="judge-x", mode="gym"):
    run = tmp_path / f"run_{agent}"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"agent": agent, "mode": mode, "judge_model": judge_model, "model": "granite", "ended": "x"}), encoding="utf-8")
    rows = []
    for i, cell in enumerate(cells):
        (run / f"seed{i:06d}").mkdir()
        rows.append({**{c: "" for c in COLUMNS}, "seed": str(i), "judge_pass": cell, "pass": "True"})
    write_rows(run / "scores.csv", rows, COLUMNS)
    return run


def test_only_empty_judge_cells_of_model_agents_are_filled(tmp_path):
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url):
        seen.append(folder.name)
        return {"pass": folder.name != "seed000002"}
    run = _run(tmp_path, "plan_execute", ["", "False", ""])
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 2
    with open(run / "scores.csv", newline="", encoding="utf-8") as f:
        cells = [r["judge_pass"] for r in csv.DictReader(f)]
    assert seen == ["seed000000", "seed000002"] and cells == ["True", "False", "False"]
    reference = _run(tmp_path, "oracle", [""])
    assert rejudge.rejudge_run(reference, None, "single", "http://judge", judged=fake) == 0


def test_a_run_without_a_judge_model_is_judged_only_under_the_overrides(tmp_path, monkeypatch):
    def fake(folder, backend, protocol, agent_model, base_url):
        return {"pass": True}
    run = _run(tmp_path, "causal", ["", ""], judge_model=None)  # started without the in-run judge
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 0
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake, unrecorded=True) == 2
    calls = []
    monkeypatch.setattr(rejudge.judge, "router_backend", lambda model, url: (model, url))
    monkeypatch.setattr(rejudge, "rejudge_run", lambda run_dir, backend, protocol, base_url, unrecorded=False, failed=False, workers=1, endpoint=None:
                        calls.append((backend, protocol, base_url, unrecorded, failed, workers)) or 0)
    rejudge.main([str(tmp_path), "--judge-model", "judge-y", "--judge-endpoint", "http://y"])
    rejudge.main([str(tmp_path), "--model", "judge-x", "--base-url", "http://x", "--protocol", "single"])
    rejudge.main([str(tmp_path), "--model", "judge-x", "--base-url", "http://x", "--rejudge-failed", "--workers", "3"])
    # the paper's five call protocol is the default, single the explicit development option
    assert calls == [(("judge-y", "http://y"), "paper", "http://y", True, False, 1), (("judge-x", "http://x"), "single", "http://x", False, False, 1),
                     (("judge-x", "http://x"), "paper", "http://x", False, True, 3)]
    for argv in (["--judge-model", "judge-y"], ["--model", "judge-x"], []):
        with pytest.raises(SystemExit):
            rejudge.main([str(tmp_path), *argv])


def test_a_rejudge_appends_one_ledger_event_with_the_old_and_new_hash_of_each_file_it_changed(tmp_path):
    import hashlib

    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
    from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _meta

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fake(folder, backend, protocol, agent_model, base_url):
        record = {"pass": True}
        (folder / "hidden").mkdir(exist_ok=True)
        (folder / "hidden" / "judge.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    run = _run(tmp_path, "plan_execute", ["", "False"])
    facts = tmp_path / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    ledger.start(run, _meta(facts), ["--agent", "plan_execute"])
    ledger.finish(run)
    old_scores = sha(run / "scores.csv")
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 1
    events = ledger.load(run)["events"]
    assert len(events) == 1 and events[0]["tool"] == "rejudge.py"
    assert events[0]["files"] == {"scores.csv": {"old": old_scores, "new": sha(run / "scores.csv")},
                                  "seed000000/hidden/judge.json": {"old": None, "new": sha(run / "seed000000" / "hidden" / "judge.json")}}
    # nothing left to judge, nothing changed, no second event
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 0
    assert len(ledger.load(run)["events"]) == 1


def test_rejudge_finds_the_folder_of_a_seed_an_anchored_scenario_and_a_gepa_call_by_the_driver_s_naming_rule(tmp_path):
    run = _run(tmp_path, "plan_execute", ["", "", ""])
    with open(run / "scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[1]["scenario_id"], rows[2]["gepa_call"] = "7", "3"
    write_rows(run / "scores.csv", rows, COLUMNS)
    names = ["seed000000", "scenario7_seed000001", "call00003_seed000002"]
    for name in names[1:]:
        (run / name).mkdir()  # beside the plain seed folders that _run made, which the rule must not take
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url):
        seen.append(folder.name)
        return {"pass": True}
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 3
    assert seen == names
    pytest.importorskip("mcp")
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
    from CausalDiagnosis.experiments.episode_runners.run_episodes import episode_plan

    assert [name for name, *_ in episode_plan(load_config("static")[1], [0], None, None)] == names[:1]


def test_an_episode_whose_judge_calls_all_failed_is_judged_again_although_its_cell_reads_false(tmp_path):
    """Five failed calls read false everywhere, so the cell of an episode judged while the endpoint was down is no longer
    empty, and judge_finished_runs.py judges such an episode again by its record, which holds no parsed review, while a record with a parsed
    review and a filled cell is left alone."""
    run = _run(tmp_path, "plan_execute", ["False", "False", ""])
    failed = {"calls_planned": 1, "calls": [{"passed": False, "error": "judge returned unparseable JSON", "details": {"raw": "x"}}], "pass": False}
    reviewed = {"calls_planned": 1, "calls": [{"passed": False, "error": None, "details": {"task_completion": False}}], "pass": False}
    for name, record in (("seed000000", failed), ("seed000001", reviewed)):
        (run / name / "hidden").mkdir()
        (run / name / "hidden" / "judge.json").write_text(json.dumps(record), encoding="utf-8")
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url):
        seen.append(folder.name)
        return {"pass": True}
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 2
    with open(run / "scores.csv", newline="", encoding="utf-8") as f:
        cells = [r["judge_pass"] for r in csv.DictReader(f)]
    assert seen == ["seed000000", "seed000002"] and cells == ["True", "False", "True"]


def _ledgered(tmp_path, agent, cells):
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
    from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _meta

    run = _run(tmp_path, agent, cells)
    facts = tmp_path / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    ledger.start(run, _meta(facts), ["--agent", agent])
    ledger.finish(run)
    return run


def _cells(run):
    with open(run / "scores.csv", newline="", encoding="utf-8") as f:
        return [r["judge_pass"] for r in csv.DictReader(f)]


def test_the_pass_refuses_a_run_whose_agent_model_is_the_judge(tmp_path, capsys):
    """For every agent the pass reads, the judge and the agent model are compared on the bare names, without the prefix of the
    router and the adapter after a colon, and a run whose agent model is the judge is refused before any episode is judged."""
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url, static=False):
        seen.append(folder.name)
        return {"pass": True}
    backend = SimpleNamespace(model_id="tokenrouter/Granite:lora")
    for agent, mode in (("plan_execute", "gym"), ("agent_as_tool", "gym"), ("causal", "gym"), ("static", "static")):
        (tmp_path / agent).mkdir()
        run = _run(tmp_path / agent, agent, [""], mode=mode)
        (run / "seed000000" / "answer.txt").write_text("The pump is worn.", encoding="utf-8")
        assert rejudge.rejudge_run(run, backend, "single", "http://judge", judged=fake) == 0
        assert f"run_{agent}: refused, the judge tokenrouter/Granite:lora is the agent model granite" in capsys.readouterr().out
    assert seen == []
    other = _run(tmp_path, "causal", [""])
    assert rejudge.rejudge_run(other, SimpleNamespace(model_id="tokenrouter/judge-x"), "single", "http://judge", judged=fake) == 1


def test_the_pass_goes_on_after_an_episode_it_cannot_judge_and_names_it(tmp_path, capsys):
    """An episode whose runner left no trajectory.json, and an episode whose judge raised, are named on the output and in the
    one ledger event, their cells stay empty, and the episodes after them are judged."""
    from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger

    def fake(folder, backend, protocol, agent_model, base_url):
        if folder.name == "seed000001":
            raise judge.NotJudged(f"{folder.name}: the runner left no trajectory.json, so the episode is not judged")
        if folder.name == "seed000002":
            raise RuntimeError("endpoint closed the connection")
        return {"pass": True}
    run = _ledgered(tmp_path, "plan_execute", ["", "", "", ""])
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 2
    out = capsys.readouterr().out
    assert "run_plan_execute/seed000001: not judged, NotJudged: seed000001: the runner left no trajectory.json" in out
    assert "run_plan_execute/seed000002: not judged, RuntimeError: endpoint closed the connection" in out
    assert _cells(run) == ["True", "", "", "True"]
    events = ledger.load(run)["events"]
    assert len(events) == 1 and "not judged: seed000001 (NotJudged" in events[0]["what"] and "seed000002 (RuntimeError" in events[0]["what"]


def test_a_judge_that_cannot_be_asked_ends_the_pass_with_its_reason_and_an_episode_it_cannot_judge_does_not(tmp_path, monkeypatch, capsys):
    """A judge endpoint that no longer answers its health probe (ServiceDown) or that gives no model listing
    (judge.ContextUnknown) is no failure of one episode: guarded raises it again, as run_episodes.run_judge does, so the pass
    ends at the first episode it can judge with the named reason and judges nothing after it, where it once named every episode
    not judged and returned 0. The control: an episode whose runner left no trajectory.json, before it, is still not judged,
    and the pass goes on past it."""
    import shutil

    from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
    from CausalDiagnosis.utils import model_request_retry as retry
    from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.tests.test_episode_judge import RUBRIC_PASS, FakeJudge, episode_folder, judge_endpoint

    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    for case, raised in (("dead", retry.ServiceDown), ("unlisted", judge.ContextUnknown)):
        if case == "unlisted":  # an endpoint that answers its model listing with 404
            judge_endpoint(monkeypatch, lambda method, url, body: (404, {}))
        (tmp_path / case).mkdir()
        run = _run(tmp_path / case, "plan_execute", ["", "", ""])
        for seed in range(3):
            shutil.rmtree(run / f"seed{seed:06d}")
            shutil.copytree(episode_folder(tmp_path / case / f"built{seed}"), run / f"seed{seed:06d}")
        (run / "seed000000" / judge.TRAJECTORY_FILE).unlink()
        with pytest.raises(raised):
            rejudge.rejudge_run(run, FakeJudge([RUBRIC_PASS] * 3), "single", "http://127.0.0.1:1/v1")
        out = capsys.readouterr().out
        assert "run_plan_execute/seed000000: not judged, NotJudged" in out and "seed000001" not in out and "seed000002" not in out, out
        assert not list(run.glob("*/hidden/judge.json")) and _cells(run) == ["", "", ""]


def test_scores_csv_is_written_after_every_judged_episode(tmp_path):
    """The pass stops after two judged episodes, and scores.csv already holds their two cells, with the ledger event of the two."""
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger

    class Stop(BaseException):
        pass

    def fake(folder, backend, protocol, agent_model, base_url):
        if folder.name == "seed000002":
            raise Stop
        return {"pass": folder.name == "seed000000"}
    run = _ledgered(tmp_path, "causal", ["", "", "", ""])
    with pytest.raises(Stop):
        rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake)
    assert _cells(run) == ["True", "False", "", ""]
    events = ledger.load(run)["events"]
    assert len(events) == 1 and events[0]["what"].startswith("judged 2 episodes") and "scores.csv" in events[0]["files"]


def test_rejudge_failed_judges_again_every_record_that_holds_a_failed_call(tmp_path):
    """A record with one parsed review and one failed call keeps its pass by default and is judged again under --rejudge-failed,
    and a record whose calls were all parsed is left alone in both."""
    run = _run(tmp_path, "agent_as_tool", ["False", "True", ""])
    calls = {"seed000000": [{"passed": False, "error": None, "details": {"task_completion": False}},
                            {"passed": False, "error": "judge backend error", "details": {}}],
             "seed000001": [{"passed": True, "error": None, "details": {"task_completion": True}}]}
    for name, record in calls.items():
        (run / name / "hidden").mkdir()
        (run / name / "hidden" / "judge.json").write_text(json.dumps({"calls_planned": 2, "calls": record}), encoding="utf-8")
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url):
        seen.append(folder.name)
        return {"pass": True}
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 1 and seen == ["seed000002"]
    seen.clear()
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake, failed=True) == 1 and seen == ["seed000000"]
    assert _cells(run) == ["True", "True", "True"]


def test_workers_give_the_same_records_as_one_worker(tmp_path):
    """With a stub backend that every thread shares, three workers write the cells and the one ledger event of one worker. Under
    three workers the episodes meet at a barrier of three, so three are judged at once in three threads."""
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger

    backend = SimpleNamespace(model_id="tokenrouter/judge-x")
    results, threads = {}, {1: set(), 3: set()}
    for workers in (1, 3):
        barrier = threading.Barrier(workers, timeout=30)

        def fake(folder, backend_seen, protocol, agent_model, base_url, workers=workers, barrier=barrier):
            assert backend_seen is backend
            threads[workers].add(threading.get_ident())
            barrier.wait()
            return {"pass": int(folder.name[-1]) % 2 == 0}
        (tmp_path / f"w{workers}").mkdir()
        run = _ledgered(tmp_path / f"w{workers}", "plan_execute", [""] * 6)
        assert rejudge.rejudge_run(run, backend, "single", "http://judge", judged=fake, workers=workers) == 6
        events = ledger.load(run)["events"]
        results[workers] = (_cells(run), len(events), sorted(events[0]["files"]))
    assert results[1] == results[3] == (["True", "False"] * 3, 1, ["scores.csv"])
    assert len(threads[1]) == 1 and len(threads[3]) == 3


def test_a_static_run_of_sc3_is_judged_under_the_overrides_in_the_judge_s_static_mode(tmp_path):
    """The grid plays the static agent without the in-run judge, so judge_finished_runs.py
    judges its answers under the overrides, in the static mode of judge_episode, and leaves an episode without an answer unjudged as
    the run does."""
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url, static=False):
        seen.append((folder.name, static))
        return {"pass": True}
    run = _run(tmp_path, "static", ["", "", ""], judge_model=None, mode="static")
    for name, answer in (("seed000000", "The pump is worn."), ("seed000001", ""), ("seed000002", "Replace the pump.")):
        (run / name / "answer.txt").write_text(answer, encoding="utf-8")
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake) == 0  # no judge model and no overrides
    assert rejudge.rejudge_run(run, None, "single", "http://judge", judged=fake, unrecorded=True) == 2
    with open(run / "scores.csv", newline="", encoding="utf-8") as f:
        cells = [r["judge_pass"] for r in csv.DictReader(f)]
    assert seen == [("seed000000", True), ("seed000002", True)] and cells == ["True", "", "True"]
    # a gym run is judged in the gym mode, and a static agent outside mode static is no run judge_finished_runs.py knows
    gym = _run(tmp_path, "causal", [""], judge_model=None)
    assert rejudge.rejudge_run(gym, None, "single", "http://judge", judged=fake, unrecorded=True) == 1 and seen[-1] == ("seed000000", False)
    (tmp_path / "odd").mkdir()
    odd = _run(tmp_path / "odd", "static", [""], judge_model=None)
    assert rejudge.rejudge_run(odd, None, "single", "http://judge", judged=fake, unrecorded=True) == 0


# ----- the judge held to the serving file of record -----

COMMIT = "c" * 40
DECLARED_JUDGE = {"served_name": "judge-x", "hub_id": "org/judge-x", "weights_commit": COMMIT}


def _judge_endpoint(tmp_path, **fields):
    """The judge endpoint as ledger.served meets it in a judge job, with the endpoint file of its serving job."""
    sampler = tmp_path / "gpu.csv"
    sampler.write_text("t,u\n", encoding="utf-8")
    url = "http://127.0.0.1:30009/v1"
    record = {"file": str(tmp_path / "judge.txt"), "lines": [url], "base_url": url, "matches_endpoint": True, "served_name": "judge-x",
              "model": "org/judge-x", "weights_commit": COMMIT, "image_sha256": "i", "template_sha256": "t", "sglang": "0.5.19",
              "gpu_name": "NVIDIA_H100", "gpu_mib": "81559", "gpu_uuid": "GPU-1", "driver_version": "1", "cuda_version": "12",
              "gpu_sampler": str(sampler), **fields}
    return {"name": "judge-x", "endpoint": url, "endpoint_file": record, "endpoint_models": [{"id": "judge-x", "max_model_len": 32768}]}


def _run_of_record(tmp_path, monkeypatch, declared=DECLARED_JUDGE, differences=(), cards=()):
    """A run of record (an identity and no stamp) under a serving file whose judge block is declared (None: the file is open), with
    ledger's comparisons of the endpoint against the file and against GPU_OF_RECORD giving differences and cards."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = _run(tmp_path, "causal", [""], judge_model=None)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    (run / "run.json").write_text(json.dumps({**meta, "asset_id": "hyd_1", "code": {"identity": "a" * 64}}), encoding="utf-8")

    def serving_of_record(path=None):
        if declared is None:
            raise ValueError("the field weights_commit of the judge is null")
        return {"format": 1, "agent": {}, "judge": declared}
    monkeypatch.setattr(rejudge.ledger.serving_endpoints, "serving_of_record", serving_of_record, raising=False)
    monkeypatch.setattr(rejudge.ledger.serving_endpoints, "serving_differences", lambda endpoint, role, mode=None: list(differences), raising=False)
    monkeypatch.setattr(rejudge.ledger.serving_endpoints, "card_differences", lambda endpoint: list(cards), raising=False)
    return run


def test_a_run_of_record_is_judged_only_by_the_judge_of_the_serving_file_and_every_record_names_its_endpoint(tmp_path, monkeypatch, capsys):
    """A run of record is judged when the served name and every declared field of the judge endpoint are those of the judge block of
    the serving file, and every record gets the serving identity of the endpoint; each difference refuses the run with both values,
    as do an endpoint without its endpoint file and a serving file that names no judge."""
    seen = []

    def fake(folder, backend, protocol, agent_model, base_url, serving=None):
        seen.append(serving)
        return {"pass": True}
    backend = SimpleNamespace(model_id="tokenrouter/judge-x")
    run = _run_of_record(tmp_path, monkeypatch)
    assert rejudge.rejudge_run(run, backend, "paper", "http://judge", judged=fake, unrecorded=True, endpoint=_judge_endpoint(tmp_path)) == 1
    assert seen[0]["weights_commit"] == COMMIT and seen[0]["model"] == "judge-x"
    no_file = {"name": "judge-x", "endpoint": "http://judge", "endpoint_file": None, "endpoint_models": None}
    for case, options, model, endpoint, reason in (
            ("name", {}, "tokenrouter/judge-y", _judge_endpoint(tmp_path), "the judge judge-y is not judge-x, the judge of the serving file of record"),
            ("field", {"differences": [("weights_commit", COMMIT, "d" * 40)]}, "tokenrouter/judge-x", _judge_endpoint(tmp_path),
             f"the weights_commit {'d' * 40} of the judge endpoint is not the weights_commit {COMMIT} of the serving file of record"),
            ("file", {}, "tokenrouter/judge-x", no_file, "the judge endpoint has no endpoint file"),
            ("open", {"declared": None}, "tokenrouter/judge-x", _judge_endpoint(tmp_path), "the serving file of record names no judge")):
        run = _run_of_record(tmp_path / case, monkeypatch, **options)
        assert rejudge.rejudge_run(run, SimpleNamespace(model_id=model), "paper", "http://judge", judged=fake, unrecorded=True, endpoint=endpoint) == 0, case
        assert reason in capsys.readouterr().out, case


def test_the_card_refuses_a_run_of_record_only_when_ledger_names_a_difference_from_gpu_of_record(tmp_path, monkeypatch, capsys):
    """The card of the judge endpoint is compared through ledger.card_differences, which names nothing when env.sh sets no
    GPU_OF_RECORD: its reasons refuse the run, and without one the run is judged."""
    def fake(folder, backend, protocol, agent_model, base_url, serving=None):
        return {"pass": True}
    backend = SimpleNamespace(model_id="tokenrouter/judge-x")
    card = "the judge endpoint serves on NVIDIA_A100, not on NVIDIA_H100 of GPU_OF_RECORD"
    run = _run_of_record(tmp_path / "other", monkeypatch, cards=[card])
    assert rejudge.rejudge_run(run, backend, "paper", "http://judge", judged=fake, unrecorded=True, endpoint=_judge_endpoint(tmp_path)) == 0
    assert card in capsys.readouterr().out
    run = _run_of_record(tmp_path / "unset", monkeypatch)
    assert rejudge.rejudge_run(run, backend, "paper", "http://judge", judged=fake, unrecorded=True, endpoint=_judge_endpoint(tmp_path)) == 1


def test_a_development_run_is_judged_without_the_serving_file_check(tmp_path):
    def fake(folder, backend, protocol, agent_model, base_url):
        return {"pass": True}
    run = _run(tmp_path, "causal", [""], judge_model=None)
    assert rejudge.rejudge_run(run, SimpleNamespace(model_id="tokenrouter/judge-y"), "paper", "http://judge", judged=fake, unrecorded=True,
                               endpoint={"name": "judge-y", "endpoint": "http://judge", "endpoint_file": None, "endpoint_models": None}) == 1


def test_the_count_selects_what_a_pass_judges(tmp_path, capsys, monkeypatch):
    """The count of the judge job reads the runs a pass reads and refuses the runs a pass refuses: an episode that a pass judges is
    left before the pass and judged after it, an episode no pass can judge is named apart, the episodes of a run of another mode, of a
    reference policy and of a run whose agent model is the judge are never left over, and a refused run is counted apart."""
    (tmp_path / "self").mkdir()
    gym = _run(tmp_path, "causal", ["", ""], judge_model=None)
    (gym / "seed000000" / "trajectory.json").write_text("[]", encoding="utf-8")  # seed000001 left no trajectory: no pass can judge it
    static = _run(tmp_path, "static", ["", ""], judge_model=None, mode="static")
    (static / "seed000000" / "answer.txt").write_text("The pump is worn.", encoding="utf-8")
    (static / "seed000001" / "answer.txt").write_text("", encoding="utf-8")
    _run(tmp_path, "plan_execute", [""], judge_model=None, mode="training")
    _run(tmp_path, "oracle", [""], judge_model=None)
    refused = _run(tmp_path / "self", "agent_as_tool", [""], judge_model=None)
    (refused / "seed000000" / "trajectory.json").write_text("[]", encoding="utf-8")
    refused_meta = json.loads((refused / "run.json").read_text(encoding="utf-8"))
    (refused / "run.json").write_text(json.dumps({**refused_meta, "model": "judge-x"}), encoding="utf-8")
    roots = [tmp_path, tmp_path / "self"]
    backend = SimpleNamespace(model_id="tokenrouter/judge-x")
    before = rejudge.tally(roots, backend, None, True)
    assert before["left"] == [gym / "seed000000", static / "seed000000"] and before["judged"] == []
    assert before["not_judged"] == [gym / "seed000001", static / "seed000001"]
    assert [(run, n) for run, _why, n in before["refused"]] == [(refused, 1)] and "is the agent model" in before["refused"][0][1]
    assert rejudge.tally(roots, backend, None, False)["left"] == []  # no run names a judge model, and no override is given

    def fake(folder, backend, protocol, agent_model, base_url, static=False, **mode):
        if not static and not (folder / "trajectory.json").exists():
            raise rejudge.judge.NotJudged(f"{folder.name}: no trajectory.json")
        (folder / "hidden").mkdir(exist_ok=True)
        (folder / "hidden" / "judge.json").write_text(json.dumps({"calls": [{"error": None, "details": {"x": 1}}], "pass": True}), encoding="utf-8")
        return {"pass": True}
    judged = sum(rejudge.rejudge_run(run, backend, "single", "http://judge", judged=fake, unrecorded=True) for run in (gym, static, refused))
    after = rejudge.tally(roots, backend, None, True)
    assert judged == 1 + 1 and after["left"] == [] and after["judged"] == [gym / "seed000000", static / "seed000000"]
    assert after["calls"] == 2 and after["failed_calls"] == 0 and after["failed"] == []
    capsys.readouterr()
    monkeypatch.setattr(rejudge.judge, "router_backend", lambda model, url: SimpleNamespace(model_id=f"tokenrouter/{model}"))
    rejudge.main([str(tmp_path), str(tmp_path / "self"), "--judge-model", "judge-x", "--judge-endpoint", "http://judge", "--count", "--names"])
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "0 2 0 2 0 2 1" and f"JUDGE_REFUSED_RUN {refused} the judge tokenrouter/judge-x is the agent model judge-x" in printed
    assert f"JUDGE_NOT_JUDGED_EPISODE {gym / 'seed000001'}" in printed
