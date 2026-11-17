"""The judge adapter with a fake backend: the fork's Evaluator and rubric scorer run on an episode folder, under both
protocols, on the trajectory and the closing answer of the runner, and the record lands under hidden/. The judge reads the whole
input and cuts only past the context of a served judge, which the tests give as a Budget of their own."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from llm import LLMBackend

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import judge_finished_runs as rejudge
from CausalDiagnosis.utils.llm_backends import EndpointChatBackend

RUBRIC_PASS = ('{"task_completion": true, "data_retrieval_accuracy": true, "generalized_result_verification": true, '
               '"agent_sequence_correct": true, "clarity_and_justification": true, "hallucinations": false, '
               '"suggestions": "none"}')
RUBRIC_FAIL = RUBRIC_PASS.replace('"hallucinations": false', '"hallucinations": true')
SUMMARY = "The cooler was replaced and the rig is back to normal."
TASK = "Find the condition of every component of hyd_1 and restore the rig."


class FakeJudge(LLMBackend):
    """Answers the rubric from a fixed list of replies and records every prompt with its temperature."""

    def __init__(self, replies, model_id="fake-judge"):
        self.replies = list(replies)
        self.calls = []
        self._model_id = model_id

    def generate(self, prompt, temperature=0.0):
        self.calls.append((prompt, temperature))
        return self.replies.pop(0)


TRAJECTORY = [{"step": 1, "server": "hydraulic_rig", "tool": "read_sensors", "tool_args": {"sensor_group": "tier1", "cycles": 1},
               "response": "TS1_Temperature_C_1Hz_mean 20.0", "error": None, "duration_ms": 3},
              {"step": 2, "server": "hydraulic_rig", "tool": "submit_diagnosis", "tool_args": {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130},
               "response": "diagnosis accepted", "error": None, "duration_ms": 2}]


def episode_folder(tmp_path, summary=SUMMARY):
    ep = tmp_path / "seed000007"
    (ep / "hidden").mkdir(parents=True)
    events = [{"event": "read", "step": 1, "t_start": 0, "t_end": 1, "class": "read", "cost": 1, "sensor_group": "tier1", "cycles": 1,
               "served": [{"n": 1, "t": 0, "repeated": False, "values": {"TS1_Temperature_C_1Hz_mean": 20.0}}]},
              {"event": "submit_diagnosis", "step": 2, "t_start": 1, "t_end": 1, "class": "write", "cost": 0,
               "diagnosis": {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130}, "evidence_ids": ["obs/1/TS1_Temperature_C_1Hz_mean"]},
              {"event": "finish", "step": 3, "t_start": 1, "t_end": 1, "class": "write", "cost": 0, "summary": summary}]
    (ep / "released.json").write_text(json.dumps({"format": 3, "asset_id": "hyd_1", "task_text": TASK, "guidance_level": 1, "rules": {},
                                                  "status": {}, "events": events, "observations": []}), encoding="utf-8")
    (ep / "hidden" / "oracle.json").write_text(json.dumps({"diagnosis": [3, 100, 0, 130], "work_orders": [{"component": "cooler", "window_end": 60}],
                                                            "finish": True}), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"model": "agent-model"}), encoding="utf-8")
    (ep / judge.TRAJECTORY_FILE).write_text(json.dumps(TRAJECTORY), encoding="utf-8")
    (ep / judge.ANSWER_FILE).write_text(summary, encoding="utf-8")
    return ep


def test_single_protocol_makes_one_call_at_temperature_zero_and_writes_the_record(tmp_path):
    ep = episode_folder(tmp_path)
    backend = FakeJudge([RUBRIC_PASS])
    record = judge.judge_episode(ep, backend, "single", agent_model="agent-model")
    assert len(backend.calls) == 1 and backend.calls[0][1] == 0.0
    prompt = backend.calls[0][0]
    assert TASK in prompt and SUMMARY in prompt and "cooler_efficiency_critical" in prompt and "submit_diagnosis" in prompt
    assert record["pass"] is True and record["pass_rate"] == 1.0 and record["score_mean"] == 1.0 and record["calls"][0]["passed"]
    assert record["calls"][0]["error"] is None
    assert record["judge_model"] == "fake-judge" and record["agent_model"] == "agent-model" and record["protocol"] == "single"
    written = json.loads((ep / "hidden" / "judge.json").read_text(encoding="utf-8"))
    assert written == record and written["reference_answer"] == judge.reference_answer({"diagnosis": [3, 100, 0, 130],
                                                                                        "work_orders": [{"component": "cooler", "window_end": 60}]})


def test_paper_protocol_makes_five_calls_at_temperature_0_3_and_passes_by_majority(tmp_path):
    ep = episode_folder(tmp_path)
    backend = FakeJudge([RUBRIC_PASS, RUBRIC_FAIL, RUBRIC_PASS, RUBRIC_PASS, RUBRIC_FAIL])
    record = judge.judge_episode(ep, backend, "paper", agent_model="agent-model")
    assert [t for _, t in backend.calls] == [0.3] * 5 and record["calls_planned"] == 5
    assert [c["passed"] for c in record["calls"]] == [True, False, True, True, False]
    assert record["pass_rate"] == 0.6 and record["pass"] is True
    minority = FakeJudge([RUBRIC_FAIL, RUBRIC_FAIL, RUBRIC_PASS, RUBRIC_FAIL, RUBRIC_PASS])
    assert judge.judge_episode(ep, minority, "paper", agent_model="agent-model")["pass"] is False


def test_the_reference_answer_names_the_faulty_set_and_every_code():
    text = judge.reference_answer({"diagnosis": {"cooler": 100, "valve": 73, "pump": 0, "accumulator": 90},
                                   "work_orders": [{"component": "valve", "window_end": 12}, {"component": "accumulator", "window_end": 60}]})
    assert "The faulty components are valve and accumulator." in text
    assert "valve_switching_critical" in text and "accumulator_pressure_low_critical" in text and "cooler_ok" in text
    assert "one work order on the valve closed by cycle 12" in text and "no work order on a healthy component" in text
    assert judge.reference_answer({"diagnosis": [100, 100, 0, 130], "work_orders": []}).count("No ") == 2


def test_self_judging_is_refused_by_the_evaluator(tmp_path):
    ep = episode_folder(tmp_path)
    with pytest.raises(ValueError, match="self-judging"):
        judge.judge_episode(ep, FakeJudge([RUBRIC_PASS], model_id="agent-model"), "single", agent_model="agent-model")


def test_an_unparseable_reply_is_a_judge_error_that_counts_as_false_over_the_planned_calls(tmp_path):
    """The pass rate is the share of passing calls over the planned calls with a failed call as false,
    the rule of the agents table (aggregate.judge_fields), and the pass is its majority, so a record whose calls all failed reads false
    everywhere, in the cell of scores.csv, in A1, on the leaderboard and in the scoring pairs alike."""
    ep = episode_folder(tmp_path)
    record = judge.judge_episode(ep, FakeJudge(["the judge rambles"]), "single", agent_model="agent-model")
    assert record["pass"] is False and record["pass_rate"] == 0.0 and record["score_mean"] is None and "unparseable" in record["calls"][0]["error"]
    # under the paper protocol a garbled call is false in the rate and the denominator stays the five planned calls
    mixed = FakeJudge([RUBRIC_PASS, "garbled", RUBRIC_FAIL, RUBRIC_PASS, RUBRIC_PASS])
    record = judge.judge_episode(ep, mixed, "paper", agent_model="agent-model")
    assert [c["error"] is None for c in record["calls"]] == [True, False, True, True, True]
    assert record["pass_rate"] == 0.6 and record["pass"] is True and record["score_mean"] == 0.95  # the mean score of the four parsed reviews
    # two passing reviews among three failed calls are no majority of the five planned calls
    record = judge.judge_episode(ep, FakeJudge(["bad", "bad", "bad", RUBRIC_PASS, RUBRIC_PASS]), "paper", agent_model="agent-model")
    assert record["pass_rate"] == 0.4 and record["pass"] is False
    record = judge.judge_episode(ep, FakeJudge(["bad"] * 5), "paper", agent_model="agent-model")
    assert record["pass_rate"] == 0.0 and record["pass"] is False and record["score_mean"] is None
    assert judge.summary_of(record["calls"], 5) == {"pass_rate": 0.0, "score_mean": None, "pass": False}


def test_main_reads_the_agent_model_from_the_run_folder_and_judges_under_the_paper_s_protocol_by_default(tmp_path, monkeypatch, capsys):
    """The five call protocol of the paper is the default of every judged run, and single stays an
    explicit development option."""
    ep = episode_folder(tmp_path)
    backend = FakeJudge([RUBRIC_PASS] * 5)
    monkeypatch.setattr(judge, "router_backend", lambda model, base_url: backend)
    monkeypatch.setattr(judge, "served_budget", lambda backend, base_url: None)  # an endpoint that names no context, as a router
    judge.main([str(ep), "--model", "judge-model", "--base-url", "http://127.0.0.1:1/v1"])
    record = json.loads((ep / "hidden" / "judge.json").read_text(encoding="utf-8"))
    assert record["agent_model"] == "agent-model" and record["base_url"] is None  # a judge record names no endpoint (public_record)
    assert record["protocol"] == "paper" and record["calls_planned"] == 5 and len(backend.calls) == 5
    assert "judge pass True (paper, pass rate 1.0)" in capsys.readouterr().out
    single = FakeJudge([RUBRIC_PASS])
    monkeypatch.setattr(judge, "router_backend", lambda model, base_url: single)
    judge.main([str(ep), "--model", "judge-model", "--protocol", "single"])
    assert json.loads((ep / "hidden" / "judge.json").read_text(encoding="utf-8"))["protocol"] == "single" and len(single.calls) == 1


def test_the_judge_backend_answers_without_thinking_and_the_record_says_so(monkeypatch, tmp_path):
    # a thinking judge that reaches the token cap returns cut off thinking, which the scorer cannot parse
    monkeypatch.setenv("API_KEY", "local")
    backend = judge.router_backend("Qwen3.8-27B-FP8", "http://judge.example/v1")
    assert backend.extra_body == {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": judge.reply_tokens()}
    fake = FakeJudge([RUBRIC_PASS])
    fake.extra_body = backend.extra_body
    record = judge.judge_episode(episode_folder(tmp_path), fake, "single")
    assert record["judge_request"] == backend.extra_body


def test_the_static_mode_judges_answer_txt_without_events_and_says_so_in_the_record(tmp_path):
    """A static answer has no finish event and no trajectory, so judge_episode's static mode reads answer.txt, shows the judge no
    events and writes mode static, the one function behind the driver's in-run judge and judge_finished_runs.py."""
    ep = episode_folder(tmp_path, summary="")
    (ep / "answer.txt").write_text("The cooler is degraded; replace the cooler.", encoding="utf-8")
    backend = FakeJudge([RUBRIC_PASS])
    record = judge.judge_episode(ep, backend, "single", agent_model="agent-model", static=True)
    prompt = backend.calls[0][0]
    assert "The cooler is degraded; replace the cooler." in prompt and "submit_diagnosis" not in prompt and TASK in prompt
    assert record["mode"] == "static" and record["answer"] == "The cooler is degraded; replace the cooler." and record["pass"] is True
    assert json.loads((ep / "hidden" / "judge.json").read_text(encoding="utf-8")) == record
    assert "mode" not in judge.judge_episode(ep, FakeJudge([RUBRIC_PASS]), "single", agent_model="agent-model")


def test_the_judge_reads_the_trajectory_of_the_runner_and_its_closing_answer_for_each_agent(tmp_path):
    """The records of Plan-and-Execute name the plan step, those of Agents-as-Tools the turn, the sub-agent and its request, those
    of the CausalDiagnosis agent nothing more; for each the prompt holds every call with its arguments and response and the
    closing answer, and a change of released.json changes no character of the prompt."""
    shapes = {"plan_execute": {"task": "read the cooler"}, "agent_as_tool": {"turn": 1, "sub_agent": "iot", "request": "read it"}, "causal": {}}
    for name, extra in shapes.items():
        ep = episode_folder(tmp_path / name)
        (ep / judge.TRAJECTORY_FILE).write_text(json.dumps([{**record, **extra} for record in TRAJECTORY]), encoding="utf-8")
        backend = FakeJudge([RUBRIC_PASS, RUBRIC_PASS])
        judge.judge_episode(ep, backend, "single", agent_model="agent-model")
        prompt = backend.calls[0][0]
        assert all(r["tool"] in prompt and r["response"] in prompt for r in TRAJECTORY) and SUMMARY in prompt, name
        assert all(str(value) in prompt for value in extra.values()), name
        released = json.loads((ep / "released.json").read_text(encoding="utf-8"))
        (ep / "released.json").write_text(json.dumps({**released, "events": []}), encoding="utf-8")
        judge.judge_episode(ep, backend, "single", agent_model="agent-model")
        assert backend.calls[1][0] == prompt, name


def test_an_episode_without_a_trajectory_is_not_judged_and_is_named(tmp_path):
    ep = episode_folder(tmp_path)
    (ep / judge.TRAJECTORY_FILE).unlink()
    with pytest.raises(judge.NotJudged, match="seed000007: the runner left no trajectory.json"):
        judge.judge_episode(ep, FakeJudge([RUBRIC_PASS]), "single", agent_model="agent-model")
    assert not (ep / "hidden" / "judge.json").exists()
    assert judge.judge_episode(ep, FakeJudge([RUBRIC_PASS]), "single", agent_model="agent-model", static=True)["mode"] == "static"


def characters(context, reply_tokens=0):
    """A Budget whose tokens are the characters of the prompt."""
    return judge.Budget(context, reply_tokens, len)


def test_a_prompt_under_the_context_of_the_served_judge_is_judged_whole_with_no_registered_number(tmp_path):
    """The judge reads the whole trajectory and the whole answer when the prompt fits the served context less the reply tokens:
    nothing is cut, no registered number is read, and the record names the context and the token counts."""
    long_answer = "w" * 5000
    ep = episode_folder(tmp_path, summary=long_answer)
    backend = FakeJudge([RUBRIC_PASS])
    record = judge.judge_episode(ep, backend, "single", agent_model="agent-model", budget=characters(10 ** 6, 2048))
    prompt = backend.calls[0][0]
    assert long_answer in prompt and json.dumps(TRAJECTORY, indent=2) in prompt
    assert record["cut"] == {"context": 10 ** 6, "reply_tokens": 2048, "prompt_tokens": len(prompt), "prompt_tokens_sent": len(prompt),
                             "records_left_out": 0, "texts_headed": 0, "head_chars": None, "answer_headed": False}
    assert record["answer"] == long_answer and record["format"] == 3 and "truncation" not in record


def test_past_the_context_records_leave_from_the_middle_first_then_texts_are_headed_then_the_answer_and_never_a_write():
    """On a made-up contract with tool names of its own, each step is taken only when the one before does not fit, and by the
    least that fits: records nearest the middle leave, one marker counting each run, the write in the middle stays; then the texts
    of the kept records are headed to one length; then the answer."""
    writes = frozenset({("plant", "close_valve")})
    records = [{"server": "plant", "tool": "probe", "tool_args": {"n": i}, "response": "x" * 50} for i in range(9)]
    records[4] = {"server": "plant", "tool": "close_valve", "tool_args": {"valve": 2}, "response": "c" * 400}
    prompt_of = lambda trajectory, answer: json.dumps(trajectory, indent=2) + answer  # noqa: E731
    whole_answer = "a" * 100
    whole = len(prompt_of(records, whole_answer))
    shown, answer, counts = judge.fitted(records, whole_answer, prompt_of, characters(whole), writes)
    assert (shown, answer, counts["records_left_out"]) == (records, whole_answer, 0)
    shown, answer, counts = judge.fitted(records, whole_answer, prompt_of, characters(whole - 1), writes)
    assert counts == {"records_left_out": 1, "texts_headed": 0, "head_chars": None, "answer_headed": False} and answer == whole_answer
    assert records[3] not in shown and records[4] in shown and {"records_left_out": 1} in shown
    only_write = [{"records_left_out": 4}, records[4], {"records_left_out": 4}]
    limit = len(prompt_of(only_write, whole_answer)) - 10
    shown, answer, counts = judge.fitted(records, whole_answer, prompt_of, characters(limit), writes)
    assert counts["records_left_out"] == 8 and counts["texts_headed"] == 1 and 0 < counts["head_chars"] < 400 and answer == whole_answer
    assert [r["tool"] for r in shown if "tool" in r] == ["close_valve"] and limit - 1 <= len(prompt_of(shown, answer)) <= limit
    headed_answer = judge.cut(whole_answer, 40)[0]
    limit = len(prompt_of(judge._headed(only_write, 0)[0], headed_answer))
    shown, answer, counts = judge.fitted(records, whole_answer, prompt_of, characters(limit), writes)
    assert counts["head_chars"] == 0 and counts["answer_headed"] and answer == headed_answer
    with pytest.raises(judge.NotJudged, match="context"):
        judge.fitted(records, whole_answer, prompt_of, characters(10), writes)


def judge_endpoint(monkeypatch, answer):
    """Routes the judge's requests (httpx.request) to answer(method, url, body), which returns (status, JSON) or an exception to
    raise; returns the list of (method, url, body, timeout) sent."""
    import httpx

    sent = []

    def request(method, url, headers=None, json=None, timeout=None):
        sent.append((method, url, json, timeout))
        out = answer(method, url, json)
        if isinstance(out, Exception):
            raise out
        status, payload = out
        return httpx.Response(status, json=payload, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)
    return sent


def test_an_endpoint_without_a_context_gets_no_cut_and_the_tokenize_request_carries_the_judge_request(monkeypatch):
    """served_budget reads max_model_len of the judge's entry in /models; a router that names none gets no Budget. The count of a
    prompt is the /tokenize answer over the one user message with the chat template switches of JUDGE_REQUEST."""
    from CausalDiagnosis.utils.model_request_retry import READ_TIMEOUT_S

    def answer(method, url, body):
        if url.endswith("/models"):
            return 200, {"data": [{"id": "judge-x", "max_model_len": models[0]}]}
        return 200, {"tokens": [1, 2, 3], "count": 3, "max_model_len": 99}

    sent = judge_endpoint(monkeypatch, answer)
    backend = EndpointChatBackend("tokenrouter/judge-x", base_url="http://127.0.0.1:1/v1", api_key="k", extra_body=judge.JUDGE_REQUEST)
    models = [None]
    assert judge.served_budget(backend, "http://127.0.0.1:1/v1") is None and judge.served_budget(backend, None) is None
    models = [32768]
    budget = judge.served_budget(backend, "http://127.0.0.1:1/v1")
    assert (budget.context, budget.reply_tokens) == (32768, judge.reply_tokens()) and budget.count("hello") == 3
    assert sent[-1] == ("POST", "http://127.0.0.1:1/v1/tokenize", {"model": "judge-x", "messages": [{"role": "user", "content": "hello"}],
                                                                  "chat_template_kwargs": {"enable_thinking": False}}, READ_TIMEOUT_S)


def test_a_judge_request_is_sent_again_after_the_registered_wait_with_the_read_timeout_and_no_number_of_its_own(monkeypatch):
    """Every request of the judge goes through CausalDiagnosis.utils.model_request_retry outside an episode: no connection, an attempt that the read timeout
    ends and a status of 503 are failed attempts, each followed by the registered wait while the endpoint's health probe answers,
    and every attempt has READ_TIMEOUT_S; the judge's code holds no urllib request and no timeout of its own. The control: with no
    probe named, the first failed attempt ends the request, which names its reason."""
    import httpx

    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
    from CausalDiagnosis.utils import model_request_retry as retry

    outcomes = [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), (503, {}), (200, {"data": [{"id": "judge-x", "max_model_len": 32768}]})]
    sent = judge_endpoint(monkeypatch, lambda method, url, body: outcomes.pop(0))
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    monkeypatch.setattr(failures, "endpoint_health", lambda base_url, key, timeout_s: {"ok": True, "status": 200})
    with judge.endpoint_probe("http://127.0.0.1:1/v1"):
        assert judge.served_context("http://127.0.0.1:1/v1", "judge-x") == 32768
    assert waits == [retry.RETRY_DELAY_S] * 3 and [timeout for *_rest, timeout in sent] == [retry.READ_TIMEOUT_S] * 4
    outcomes[:] = [httpx.ConnectError("refused"), (200, {"data": []})]
    with pytest.raises(judge.ContextUnknown, match="ConnectError: refused"):
        judge.served_context("http://127.0.0.1:1/v1", "judge-x")
    assert len(sent) == 5 and len(waits) == 3
    source = Path(judge.__file__).read_text(encoding="utf-8")
    assert "urllib" not in source and not re.search(r"timeout=\d", source)


def test_a_judge_that_cannot_read_its_served_context_stops_with_the_reason(monkeypatch):
    """A model listing that the endpoint refuses raises ContextUnknown with the status, so no prompt is sent without its budget;
    only a listing that names no max_model_len, as a router's, gives no budget."""
    listing = [(404, {"detail": "not found"})]
    judge_endpoint(monkeypatch, lambda method, url, body: listing[0])
    backend = EndpointChatBackend("tokenrouter/judge-x", base_url="http://127.0.0.1:1/v1", api_key="k", extra_body=judge.JUDGE_REQUEST)
    with pytest.raises(judge.ContextUnknown, match="gave no model listing .*404"):
        judge.served_budget(backend, "http://127.0.0.1:1/v1")
    listing[0] = (200, {"data": [{"id": "judge-x"}]})
    assert judge.served_budget(backend, "http://127.0.0.1:1/v1") is None


def test_a_dead_judge_endpoint_ends_the_judge_with_service_down(monkeypatch):
    """episode_judge.py names the endpoint's health probe (endpoint_probe): after a failed attempt on an endpoint whose /models answers no
    HTTP status at all the program ends with ServiceDown instead of attempting again for ever."""
    import httpx

    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
    from CausalDiagnosis.utils import model_request_retry as retry

    judge_endpoint(monkeypatch, lambda method, url, body: httpx.ConnectError("refused"))
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    probed = []
    monkeypatch.setattr(failures, "endpoint_health", lambda base_url, key, timeout_s: probed.append(timeout_s) or {"ok": False, "status": None})
    with pytest.raises(retry.ServiceDown):
        judge.main(["--thinking", "--model", "judge-x", "--base-url", "http://127.0.0.1:1/v1"])
    assert probed == [retry.READ_TIMEOUT_S]


def test_a_judge_on_a_dead_endpoint_ends_with_service_down_after_one_attempt_and_a_slow_live_one_is_judged(tmp_path, monkeypatch):
    """judge_episode names the health probe of its endpoint (endpoint_probe): on a port where nothing listens, the first failed
    attempt asks the probe, which gets no answer either, and the judging ends with ServiceDown after one attempt and one wait
    and writes no record, where it once sent again without end. The control: an endpoint whose model listing times out once
    while its probe answers is asked again, and the episode is judged."""
    import httpx

    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
    from CausalDiagnosis.utils import model_request_retry as retry

    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    dead = episode_folder(tmp_path / "dead")
    with pytest.raises(retry.ServiceDown):
        judge.judge_episode(dead, FakeJudge([RUBRIC_PASS]), "single", agent_model="agent-model", base_url="http://127.0.0.1:1/v1")
    assert waits == [retry.RETRY_DELAY_S] and not (dead / "hidden" / judge.JUDGE_NAME).exists()
    listings = [httpx.ReadTimeout("slow"), (200, {"data": [{"id": "fake-judge", "max_model_len": 32768}]})]

    def answer(method, url, body):
        return listings.pop(0) if url.endswith("/models") else (200, {"count": 3})

    sent = judge_endpoint(monkeypatch, answer)
    monkeypatch.setattr(failures, "endpoint_health", lambda base_url, key, timeout_s: {"ok": True, "status": 200})
    record = judge.judge_episode(episode_folder(tmp_path / "slow"), FakeJudge([RUBRIC_PASS]), "single", agent_model="agent-model",
                                 base_url="http://127.0.0.1:1/v1")
    assert record["pass"] is True and record["cut"]["context"] == 32768 and len(waits) == 2
    assert [url for _method, url, _body, _timeout in sent].count("http://127.0.0.1:1/v1/models") == 2


def test_a_service_down_inside_the_scorer_ends_the_judging_and_no_call_is_scored_as_failed(tmp_path):
    """The scorer keeps every error of a judge call as a failed review, so a ServiceDown of the backend would turn a dead endpoint
    into failed reviews: FixedTemperature keeps it and judge_once raises it after the evaluation. The control: any other error of
    the backend stays a failed review with its reason, as before."""
    from CausalDiagnosis.utils import model_request_retry as retry

    class Gone(FakeJudge):
        def generate(self, prompt, temperature=0.0):
            raise self.replies.pop(0)

    ep = episode_folder(tmp_path)
    with pytest.raises(retry.ServiceDown):
        judge.judge_episode(ep, Gone([retry.ServiceDown("the probe does not answer")]), "single", agent_model="agent-model")
    assert not (ep / "hidden" / judge.JUDGE_NAME).exists()
    record = judge.judge_episode(ep, Gone([RuntimeError("a broken reply")]), "single", agent_model="agent-model")
    assert record["pass"] is False and "a broken reply" in record["calls"][0]["error"]


def test_every_judge_request_sends_the_registered_reply_room_and_the_budget_leaves_it_free(monkeypatch):
    """judge.reply_tokens of the registered file, 2,048 from the client default of AssetOpsBench's evaluation, goes into the body of
    every request of the judge backend as max_tokens beside the switch that turns thinking off, and the budget reads the same value."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

    monkeypatch.setenv("API_KEY", "k")
    assert judge.reply_tokens() == registered.load()["judge"]["reply_tokens"] == 2048
    backend = judge.router_backend("judge-x", "http://127.0.0.1:1/v1")
    assert backend.extra_body == {**judge.JUDGE_REQUEST, "max_tokens": 2048}
    monkeypatch.setattr(judge, "served_context", lambda base_url, model: 32768)
    assert judge.served_budget(backend, "http://127.0.0.1:1/v1").reply_tokens == 2048


def test_the_cut_keeps_every_write_of_the_tool_contracts_and_the_record_counts_what_left(tmp_path):
    """Under the tool contracts of the rig and of the rows of AssetOpsBench the diagnosis and the work order are writes, so under a
    context that no read fits they stay, the answer stays whole, and the record counts the records that left and the tokens."""
    assert {("hydraulic_rig", "submit_diagnosis"), ("hydraulic_rig", "create_work_order"), ("hydraulic_rig", "finish")} <= judge.write_calls()
    assert ("hydraulic_rig", "read_sensors") not in judge.write_calls()
    ep = episode_folder(tmp_path)
    reads = [{**TRAJECTORY[0], "step": i} for i in range(30)]
    (ep / judge.TRAJECTORY_FILE).write_text(json.dumps([*reads[:15], TRAJECTORY[1], *reads[15:]]), encoding="utf-8")
    first = FakeJudge([RUBRIC_PASS])
    whole = judge.judge_episode(ep, first, "single", agent_model="agent-model", budget=characters(10 ** 6))["cut"]["prompt_tokens"]
    backend = FakeJudge([RUBRIC_PASS])
    record = judge.judge_episode(ep, backend, "single", agent_model="agent-model", budget=characters(whole - 2000))
    assert record["cut"]["records_left_out"] > 0 and record["cut"]["prompt_tokens"] == whole
    assert record["cut"]["prompt_tokens_sent"] <= whole - 2000 and not record["cut"]["answer_headed"]
    assert "submit_diagnosis" in backend.calls[0][0] and SUMMARY in backend.calls[0][0] and record["answer"] == SUMMARY
    assert aggregate.judge_fields({"episode_dir": ep})["judge_record_cut"]


def test_the_judge_record_names_the_judge_model_and_the_hashes_of_what_it_read(tmp_path):
    ep = episode_folder(tmp_path)
    serving = {"weights_commit": "w", "image_sha256": "i", "template_sha256": "t", "engine": "e", "model": "fake-judge", "context_length": 8}
    record = judge.judge_episode(ep, FakeJudge([RUBRIC_PASS]), "single", agent_model="agent-model", serving=serving)
    assert record["judge_model"] == "fake-judge" and record["endpoint_identity"] == serving and record["format"] == judge.JUDGE_FORMAT
    assert record["read"] == {"trajectory_sha256": hashlib.sha256((ep / judge.TRAJECTORY_FILE).read_bytes()).hexdigest(),
                              "answer_sha256": hashlib.sha256(SUMMARY.encode("utf-8")).hexdigest()}


def test_the_judge_record_names_the_judge_by_the_served_name_that_its_requests_send(tmp_path):
    """A record names its judge as the requests of the backend name it, the served name without the prefix that router_backend puts
    before it, the served name of the judge block of serving_of_record.json."""
    ep = episode_folder(tmp_path)
    record = judge.judge_episode(ep, FakeJudge([RUBRIC_PASS], model_id="tokenrouter/Judge-X"), "single", agent_model="agent-model")
    assert record["judge_model"] == "Judge-X" and judge.served_name(FakeJudge([], model_id="tokenrouter/Judge-X")) == "Judge-X"
    assert judge.served_name(FakeJudge([], model_id="judge-y")) == "judge-y"


def test_an_empty_review_object_is_a_failed_call_for_the_three_readers(tmp_path):
    empty = {"passed": False, "score": 0.0, "rationale": "", "details": {}, "error": None}
    parsed = {"passed": True, "score": 1.0, "rationale": "", "details": json.loads(RUBRIC_PASS), "error": None}
    assert not judge.reviewed(empty) and judge.reviewed(parsed)
    assert judge.summary_of([empty, parsed], 2) == {"pass_rate": 0.5, "score_mean": 1.0, "pass": False}
    ep = episode_folder(tmp_path)
    (ep / "hidden" / "judge.json").write_text(json.dumps({"protocol": "single", "judge_model": "j", "calls_planned": 1, "calls": [empty]}), encoding="utf-8")
    assert aggregate.judge_fields({"episode_dir": ep})["judge_failed_calls"] == 1 and rejudge.unreviewed(ep)


def test_self_judging_is_refused_on_the_bare_names_for_every_agent():
    assert judge.same_model("tokenrouter/Qwen3-8B", "qwen3-8b:ft_fold1") and judge.same_model("litellm_proxy/m", "m")
    assert not judge.same_model("tokenrouter/judge-model", "agent-model") and not judge.same_model(None, None)


def test_the_thinking_check_counts_the_replies_that_hold_thinking(monkeypatch):
    """judge.thinking_replies sends the request of the judge (JUDGE_REQUEST, the registered reply room) and counts the replies
    with a reasoning field or a <think> tag, which the judge job prints as JUDGE_THINKING."""
    replies = iter([{"content": "yes"}, {"content": "<think>hm</think> no", "reasoning_content": None}, {"content": "blue", "reasoning_content": "sky"}])
    sent = judge_endpoint(monkeypatch, lambda method, url, body: (200, {"choices": [{"message": next(replies)}]}))
    assert judge.thinking_replies("http://127.0.0.1:1/v1", "judge-model") == 2
    bodies = [body for _method, _url, body, _timeout in sent]
    assert all(body["chat_template_kwargs"] == {"enable_thinking": False} and body["model"] == "judge-model"
               and body["max_tokens"] == judge.reply_tokens() for body in bodies) and len(bodies) == 3


def test_the_thinking_check_counts_a_think_block_that_the_chat_template_opened_in_the_prompt(monkeypatch):
    """A chat template that opens the think block in the prompt leaves only its closing tag in the reply text, and the thinking
    check counts such a reply as the count of every judge record does (holds_thinking), beside a reply whose thinking is in the
    reasoning field. The control: a reply without a tag or a reasoning field counts nothing."""
    replies = iter([{"content": "the criteria first</think> yes"}, {"content": "yes", "reasoning": "the criteria first"}, {"content": "yes"}])
    judge_endpoint(monkeypatch, lambda method, url, body: (200, {"choices": [{"message": next(replies)}]}))
    assert judge.thinking_replies("http://127.0.0.1:1/v1", "judge-model") == 2


# ----- the code that wrote a judge record, and the replies that hold thinking -----

@pytest.fixture
def fresh_code(monkeypatch):
    """code_of_judge reads the tree once per process; each test of this block reads it again under its own patches."""
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger

    judge.code_of_judge.cache_clear()
    yield ledger
    judge.code_of_judge.cache_clear()


def test_the_judge_record_names_the_commit_of_the_clone_that_wrote_it(tmp_path, monkeypatch, fresh_code):
    monkeypatch.setattr(fresh_code.code_identity, "git_state", lambda paths, root=None: ("f" * 40, []))
    record = judge.judge_episode(episode_folder(tmp_path), FakeJudge([RUBRIC_PASS]), "single")
    assert record["code_commit"] == "f" * 40
    judge.code_of_judge.cache_clear()
    monkeypatch.setattr(fresh_code.code_identity, "git_state", lambda paths, root=None: (None, None))
    assert judge.judge_episode(episode_folder(tmp_path / "upload"), FakeJudge([RUBRIC_PASS]), "single")["code_commit"] is None


def test_the_judge_record_names_the_hash_of_the_code_list_that_holds_the_judge(tmp_path, monkeypatch, fresh_code):
    own = judge.Path(judge.__file__).resolve().relative_to(judge.ROOT.resolve()).as_posix()
    listed = {"scoring": {own: "1" * 64, "other/file.py": "2" * 64}, "harness": {"x.py": "3" * 64}}
    monkeypatch.setattr(fresh_code.code_identity, "tree_files", lambda root=None: listed)
    first = judge.judge_episode(episode_folder(tmp_path / "a"), FakeJudge([RUBRIC_PASS]), "single")["judge_code_sha256"]
    assert first == fresh_code.canonical_sha256(sorted(listed["scoring"].items()))
    judge.code_of_judge.cache_clear()
    listed["scoring"][own] = "4" * 64  # a change of the judge itself moves the hash, a change of another list does not
    second = judge.judge_episode(episode_folder(tmp_path / "b"), FakeJudge([RUBRIC_PASS]), "single")["judge_code_sha256"]
    assert second != first


def test_the_judge_record_counts_the_replies_that_hold_thinking(tmp_path):
    replies = ["<think>the rubric</think>" + RUBRIC_PASS, RUBRIC_PASS, RUBRIC_PASS, "<think>again", RUBRIC_PASS]
    record = judge.judge_episode(episode_folder(tmp_path), FakeJudge(replies), "paper")
    assert record["thinking_replies"] == 2 and [call["thinking_replies"] for call in record["calls"]] == [1, 0, 0, 1, 0]


def test_a_reply_whose_thinking_the_reasoning_parser_moved_out_of_the_text_counts_as_thinking(tmp_path):
    """A judge served with a reasoning parser returns its thinking in a field of the reply, outside the text, so the count of a
    record reads that field beside the think block of the text. The control: replies with an empty field or none, and no think
    block, count nothing."""
    class Reply:
        def __init__(self, text, reasoning):
            self.text, self.reasoning = text, reasoning

    class Parsed(FakeJudge):
        def generate_with_usage(self, prompt, temperature=0.0):
            self.calls.append((prompt, temperature))
            return Reply(*self.replies.pop(0))

        def generate(self, prompt, temperature=0.0):
            return self.generate_with_usage(prompt, temperature).text

    replies = [(RUBRIC_PASS, "the criteria first"), (RUBRIC_PASS, None), (RUBRIC_PASS, ""), ("<think>x</think>" + RUBRIC_PASS, None),
               (RUBRIC_PASS, None)]
    record = judge.judge_episode(episode_folder(tmp_path), Parsed(replies), "paper")
    assert record["thinking_replies"] == 2 and [call["thinking_replies"] for call in record["calls"]] == [1, 0, 0, 1, 0]
    assert record["pass"] is True


def test_a_judge_without_its_context_stops_the_run_and_leaves_no_empty_pass(monkeypatch, tmp_path):
    """run_episodes.run_judge raises ContextUnknown and ServiceDown again, so a run whose judge cannot learn its context, or whose
    judge endpoint is gone, stops with the reason; any other judge failure stays one episode's empty pass with its error."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.utils import model_request_retry as retry

    options = {"judge": "local", "backend": None, "protocol": "single", "agent_model": "agent-model", "base_url": "http://127.0.0.1:1/v1"}
    for failure in (judge.ContextUnknown("no model listing"), retry.ServiceDown("gone")):
        monkeypatch.setattr(judge, "judge_episode", lambda *args, failure=failure, **kwargs: (_ for _ in ()).throw(failure))
        with pytest.raises(type(failure)):
            run_episodes.run_judge(tmp_path, options)
    monkeypatch.setattr(judge, "judge_episode", lambda *args, **kwargs: (_ for _ in ()).throw(judge.NotJudged("no trajectory")))
    assert run_episodes.run_judge(tmp_path, options) == {"pass": None, "error": "NotJudged: no trajectory"}


def test_the_model_listing_and_the_tokenize_count_of_the_judge_are_no_model_reply(monkeypatch):
    """judge._request sends the model listing and the tokenize count as no model reply, as CausalDiagnosis.utils.model_reply_room does, so that
    inside an episode neither stands for the first reply, whose cached tokens the driver records. The control: a chat
    completion of the judge, a model reply, does."""
    import time

    from CausalDiagnosis.utils import model_request_retry as retry

    def answer(method, url, body):
        if url.endswith("/models"):
            return 200, {"data": [{"id": "judge-x", "max_model_len": 32768}]}
        if url.endswith("/tokenize"):
            return 200, {"count": 3}
        return 200, {"choices": [{"message": {"content": "yes"}}], "usage": {"prompt_tokens_details": {"cached_tokens": 4}}}

    judge_endpoint(monkeypatch, answer)
    with retry.episode_deadline(time.monotonic() + 60):
        assert judge.served_context("http://127.0.0.1:1/v1", "judge-x") == 32768
        assert judge.prompt_tokens("http://127.0.0.1:1/v1", "judge-x", "a prompt") == 3
        assert retry.first_cached_tokens() is None
        assert judge.thinking_replies("http://127.0.0.1:1/v1", "judge-x", prompts=["one"]) == 0
        assert retry.first_cached_tokens() == 4
