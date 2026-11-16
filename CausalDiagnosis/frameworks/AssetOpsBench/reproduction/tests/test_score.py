"""The judge and the efficiency counts on a hand-built run folder. The judge runs through the fork's
Evaluator with a scripted judge backend under both protocols, five calls at temperature 0.3 with a failed call counted as
false and one call at temperature 0. Also covered are an episode without a trajectory record, the skip of an episode judged
before, the verdicts of another judge kept, and the refusals."""
import json
import threading

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import score_runs as score
from llm import LLMBackend
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import sha256_of

QUESTION = "List the IoT sites."
FORM = "The answer names the site MAIN."


def review(task_completion):
    return json.dumps({"task_completion": task_completion, "data_retrieval_accuracy": True, "generalized_result_verification": False,
                       "agent_sequence_correct": True, "clarity_and_justification": True, "hallucinations": False,
                       "suggestions": "none"})


class ReviewBackend(LLMBackend):
    """Scripted judge replies handed out in order under a lock, since the scorer calls the judge from several threads.
    Every prompt and temperature is kept."""

    def __init__(self, replies, model="tokenrouter/judge-test"):
        self._model_id = model
        self._replies = iter(replies)
        self._lock = threading.Lock()
        self.prompts, self.temperatures = [], []

    def generate(self, prompt, temperature=0.0):
        with self._lock:
            self.prompts.append(prompt)
            self.temperatures.append(temperature)
            return next(self._replies)


@pytest.fixture
def run_dir(tmp_path):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(json.dumps({"id": 1, "type": "IoT", "text": QUESTION, "characteristic_form": FORM}) + "\n"
                         + json.dumps({"id": 601, "type": "", "text": "List the failure modes.", "characteristic_form": "modes"}) + "\n",
                         encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run": "run", "model": "granite-test", "scenario_ids": [1, 601], "repeats": 1,
                                              "scenarios": {"path": str(scenarios), "sha256": sha256_of(scenarios)}}), encoding="utf-8")
    pe = run / "plan_execute_s001_r1"
    pe.mkdir()
    (pe / "episode.json").write_text(json.dumps({
        "episode": pe.name, "agent": "plan_execute", "scenario_id": 1, "family": "IoT", "task_kind": "single", "repeat": 1,
        "error": None, "runtime_s": 12.5, "steps": 3, "tool_calls": 2, "plan_steps": 3, "model_calls": 4, "input_tokens": 100,
        "output_tokens": 20, "trajectory_record": f"{pe.name}.json"}),
        encoding="utf-8")
    (pe / f"{pe.name}.json").write_text(json.dumps({
        "run_id": pe.name, "scenario_id": "1", "runner": "plan-execute", "model": "tokenrouter/granite-test", "question": QUESTION,
        "answer": "The site is MAIN.", "trajectory": [{"step_number": 1, "tool": "sites", "response": "[\"MAIN\"]"}]}), encoding="utf-8")
    aat = run / "agent_as_tool_s601_r1"
    aat.mkdir()
    (aat / "episode.json").write_text(json.dumps({
        "episode": aat.name, "agent": "agent_as_tool", "scenario_id": 601, "family": "Multi-agent", "task_kind": "multi",
        "repeat": 1, "error": "MaxTurnsExceeded: 30", "runtime_s": 90.0, "steps": 29, "tool_calls": 29, "plan_steps": None,
        "model_calls": 30, "input_tokens": 9000, "output_tokens": 600,
        "trajectory_record": None}), encoding="utf-8")
    return run


def test_both_protocols_through_the_evaluator(run_dir, monkeypatch):
    # five paper calls (three true, one false, one without JSON) and then the single call at temperature 0
    backend = ReviewBackend([review(True), review(True), review(False), "I cannot judge this.", review(True), review(False)])
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: backend)
    rows = score.main([str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"])["run"]

    assert backend.temperatures == [0.3] * 5 + [0.0]
    assert all(QUESTION in p and FORM in p and "The site is MAIN." in p for p in backend.prompts)
    paper = json.loads((run_dir / "plan_execute_s001_r1" / "judge_paper.json").read_text())
    assert paper["protocol"] == "paper" and paper["calls_planned"] == 5 and len(paper["calls"]) == 5 and paper["failed"] == 1
    assert paper["mean"]["task_completion"] == pytest.approx(0.6) and paper["majority"]["task_completion"] is True
    assert paper["mean"]["data_retrieval_accuracy"] == pytest.approx(0.8) and paper["majority"]["data_retrieval_accuracy"] is True
    assert paper["mean"]["generalized_result_verification"] == 0 and paper["majority"]["generalized_result_verification"] is False
    assert paper["mean"]["hallucinations"] == 0 and paper["fork_pass"] == 0 and paper["judge_model"] == "tokenrouter/judge-test"
    single = json.loads((run_dir / "plan_execute_s001_r1" / "judge_single.json").read_text())
    assert single["temperature"] == 0.0 and single["calls_planned"] == 1 and single["mean"]["task_completion"] == 0
    for protocol in ("paper", "single"):
        unrecorded = json.loads((run_dir / "agent_as_tool_s601_r1" / f"judge_{protocol}.json").read_text())
        assert unrecorded["status"] == "no_record" and unrecorded["calls"] == [] and unrecorded["mean"]["task_completion"] == 0

    by_episode = {r["episode"]: r for r in rows}
    pe, aat = by_episode["plan_execute_s001_r1"], by_episode["agent_as_tool_s601_r1"]
    assert (pe["paper_task_completion"], pe["single_task_completion"]) == (pytest.approx(0.6), 0.0)
    assert pe["paper_judge_failed"] == 1 and pe["steps"] == 3
    # the parsed calls of the paper protocol answer true, true, false, true: three of their six pairs agree
    assert pe["paper_task_completion_agreement"] == 0.5 and pe["paper_data_retrieval_accuracy_agreement"] == 1.0
    assert pe["single_task_completion_agreement"] is None and aat["paper_task_completion_agreement"] is None
    assert aat["error"].startswith("MaxTurnsExceeded")
    assert aat["paper_judge_status"] == aat["single_judge_status"] == "no_record"

    # a second scoring by the same judge makes no call under either protocol
    again = ReviewBackend([])
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: again)
    score.main([str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"])
    assert again.prompts == []
    # another judge keeps the first verdicts beside its own
    other = ReviewBackend([review(False)] * 6, model="tokenrouter/judge-two")
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: other)
    score.main([str(run_dir), "--judge-model", "judge-two", "--judge-base-url", "http://127.0.0.1:9/v1"])
    episode = run_dir / "plan_execute_s001_r1"
    assert json.loads((episode / "judge_paper.tokenrouter_judge-test.json").read_text())["mean"]["task_completion"] == pytest.approx(0.6)
    assert (episode / "judge_single.tokenrouter_judge-test.json").exists() and len(other.prompts) == 6
    assert json.loads((episode / "judge_paper.json").read_text())["judge_model"] == "tokenrouter/judge-two"


def test_without_a_judge_only_efficiency_is_scored_and_the_refusals(run_dir, monkeypatch, tmp_path):
    rows = {r["episode"]: r for r in score.main([str(run_dir)])["run"]}
    assert all("paper_task_completion" not in r for r in rows.values())
    assert rows["plan_execute_s001_r1"]["steps"] == 3 and rows["agent_as_tool_s601_r1"]["runtime_s"] == 90.0
    same = ReviewBackend([], model="tokenrouter/granite-test")
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: same)
    with pytest.raises(SystemExit, match="self judging"):
        score.main([str(run_dir), "--judge-model", "granite-test", "--judge-base-url", "http://127.0.0.1:9/v1"])
    assert same.prompts == []
    with pytest.raises(SystemExit):
        score.main([str(run_dir), "--judge-model", "judge-test"])
    assert same.prompts == []
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256"):
        score.main([str(run_dir), "--scenarios", str(other)])


def test_every_verdict_and_scores_row_records_the_request_fields_of_the_judge(run_dir, monkeypatch):
    backend = ReviewBackend([review(True)] * 30)
    backend.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: backend)
    rows = score.main([str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"])["run"]
    row = {r["episode"]: r for r in rows}["plan_execute_s001_r1"]
    for protocol in ("paper", "single"):
        verdict = json.loads((run_dir / "plan_execute_s001_r1" / f"judge_{protocol}.json").read_text())
        assert verdict["judge_request"] == backend.extra_body
        assert json.loads(row[f"{protocol}_judge_request"]) == backend.extra_body


class OverflowBackend(ReviewBackend):
    def generate(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        raise ValueError("this model's maximum context length is exceeded")


def long_record(run_dir):
    """The fixture record of the first episode with a tool output past the released cut and a mark after it."""
    path = run_dir / "plan_execute_s001_r1" / "plan_execute_s001_r1.json"
    record = json.loads(path.read_text())
    record["trajectory"].append({"step_number": 2, "tool": "history", "response": "x" * 20_000 + "LAST STEP"})
    path.write_text(json.dumps(record))


def test_the_judge_reads_the_whole_trajectory(run_dir, monkeypatch):
    long_record(run_dir)
    backend = ReviewBackend([review(True)] * 12)
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: backend)
    score.main([str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"])
    assert backend.prompts and all("LAST STEP" in p for p in backend.prompts)


def test_a_call_that_overflows_the_context_of_the_judge_is_a_failed_call(run_dir, monkeypatch):
    long_record(run_dir)
    backend = OverflowBackend([])
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: backend)
    rows = score.main([str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"])["run"]
    row = {r["episode"]: r for r in rows}["plan_execute_s001_r1"]
    assert row["paper_judge_failed"] == 5 and row["single_judge_failed"] == 1 and len(backend.prompts) == 6
    assert row["paper_task_completion"] == 0 and row["paper_judge_status"] == "judged"


def judge_files(run_dir):
    return {p.relative_to(run_dir).as_posix(): p.read_bytes() for p in sorted(run_dir.rglob("judge_*.json"))}


def test_rejudge_failed_replaces_only_the_verdicts_with_failed_calls(run_dir, monkeypatch):
    args = [str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"]
    # the fixture verdict: one of the five paper calls has no JSON, the single call parses
    first = ReviewBackend([review(True), review(True), review(False), "I cannot judge this.", review(True), review(False)])
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: first)
    score.main(args)
    episode = run_dir / "plan_execute_s001_r1"
    assert json.loads((episode / "judge_paper.json").read_text())["failed"] == 1
    before = judge_files(run_dir)
    idle = ReviewBackend([])
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: idle)
    score.main(args)
    assert idle.prompts == [] and judge_files(run_dir) == before
    again = ReviewBackend([review(True)] * 5)
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: again)
    score.main([*args, "--rejudge-failed"])
    assert again.temperatures == [0.3] * 5
    paper = json.loads((episode / "judge_paper.json").read_text())
    assert paper["failed"] == 0 and paper["mean"]["task_completion"] == 1
    after = judge_files(run_dir)
    assert set(after) == set(before)  # replaced in place, no copy under the judge's name
    assert {path for path in after if after[path] != before[path]} == {"plan_execute_s001_r1/judge_paper.json"}


def test_the_verdicts_written_before_an_error_enter_the_ledger(run_dir, monkeypatch):
    """score_runs.py writes a verdict per episode and records the scoring once at the end, so an error after the first verdict, as
    when the Evaluator joins no scenario to a record, still records the verdicts written before it, and the ledger check of the
    run finds no file changed without an event."""
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
    from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    fields = {"label": "aob_repro", "agent": "plan_execute+agent_as_tool", "agent_label": "plan_execute+agent_as_tool", "model": None,
              "judge": None, "data": {}, "configuration": {}, "seeds": None}
    ledger.start(run_dir, {**meta, "started": "2026-11-13T00:00:00"}, ["--ids", "1", "601"], fields=fields)
    for episode in sorted(p.name for p in run_dir.iterdir() if p.is_dir()):
        ledger.ledger_file.add_episode(run_dir, episode, {"episode": episode})
    ledger.finish(run_dir)

    def no_scenario(record, scenarios, scenario_id, judge_model):
        raise SystemExit(f"the Evaluator joined no scenario to {record}")

    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: ReviewBackend([]))
    monkeypatch.setattr(score, "judge_call", no_scenario)
    with pytest.raises(SystemExit, match="joined no scenario"):
        score.main([str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:9/v1"])
    # the episode without a record is judged without a call before the episode whose calls fail
    assert (run_dir / "agent_as_tool_s601_r1" / "judge_paper.json").exists()
    event = ledger.load(run_dir)["events"][-1]
    assert event["tool"] == "score.py" and "agent_as_tool_s601_r1/judge_paper.json" in event["files"]
    assert verify_results.check_ledger(run_dir) == ("checked", [])


class PolicyBackend(LLMBackend):
    """A judge backend behind the one retry policy of the tree (CausalDiagnosis.utils.model_request_retry.post_with_retries), as the chat backend of the tree
    sends: each attempt takes the next of failures, an exception that fails it, and answers the review once they are used up."""

    def __init__(self, failures, model="tokenrouter/judge-test"):
        self._model_id = model
        self.failures, self.sent = list(failures), 0
        self._lock = threading.Lock()

    def generate(self, prompt, temperature=0.0):
        from CausalDiagnosis.utils import model_request_retry as retry

        def send(timeout):
            with self._lock:
                self.sent += 1
                if self.failures:
                    raise self.failures.pop(0)
            return review(True)

        return retry.post_with_retries(send)


class Unavailable(Exception):
    """An attempt that the service answers with 503, as the SDK raises it."""

    status_code = 503


def test_a_transient_failure_of_the_judge_is_sent_again_and_a_dead_judge_ends_the_scoring(run_dir, monkeypatch):
    """The scorer names the health probe of the judge's endpoint (judge.endpoint_probe). On an endpoint that no longer answers it,
    the first call ends the scoring with ServiceDown and no verdict is written, where every call once counted false on every
    criterion and the scoring ended with status 0. The control: a call whose first attempt the service answers with 503 while
    the probe answers is sent again and scores its review, where it once counted false."""
    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
    import httpx
    from CausalDiagnosis.utils import model_request_retry as retry

    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    health = {"status": None}
    monkeypatch.setattr(failures, "endpoint_health", lambda base_url, key, timeout_s: {"ok": health["status"] is not None, **health})
    argv = [str(run_dir), "--judge-model", "judge-test", "--judge-base-url", "http://127.0.0.1:1/v1", "--judge-protocols", "single"]
    dead = PolicyBackend([httpx.ConnectError("refused")] * 4)
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: dead)
    with pytest.raises(retry.ServiceDown, match="no longer answers its health probe"):
        score.main(argv)
    assert not (run_dir / "plan_execute_s001_r1" / "judge_single.json").exists()
    health["status"] = 200
    live = PolicyBackend([Unavailable()])
    monkeypatch.setattr(score, "judge_backend", lambda model, base_url: live)
    score.main(argv)
    verdict = json.loads((run_dir / "plan_execute_s001_r1" / "judge_single.json").read_text(encoding="utf-8"))
    assert verdict["failed"] == 0 and verdict["mean"]["task_completion"] == 1.0 and live.sent == 2
