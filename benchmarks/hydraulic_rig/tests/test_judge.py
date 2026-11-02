"""The judge adapter with a fake backend: the fork's Evaluator and rubric scorer run on an episode folder, under both
protocols, and the record lands under hidden/."""
import json

import pytest

from llm import LLMBackend

import judge

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


def episode_folder(tmp_path, summary=SUMMARY):
    ep = tmp_path / "seed000007"
    (ep / "hidden").mkdir(parents=True)
    events = [{"event": "read", "step": 1, "t_start": 0, "t_end": 1, "class": "read", "cost": 1, "sensor_group": "tier1", "cycles": 1,
               "served": [{"n": 1, "t": 0, "repeated": False, "values": {"TS1_Temperature_C_1Hz_mean": 20.0}}]},
              {"event": "submit_diagnosis", "step": 2, "t_start": 1, "t_end": 1, "class": "write", "cost": 0,
               "diagnosis": {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130}, "evidence_ids": ["obs/1/TS1_Temperature_C_1Hz_mean"]},
              {"event": "finish", "step": 3, "t_start": 1, "t_end": 1, "class": "write", "cost": 0, "summary": summary}]
    (ep / "released.json").write_text(json.dumps({"format": 2, "asset_id": "hyd_1", "task_text": TASK, "guidance_level": 1, "rules": {},
                                                  "status": {}, "events": events, "observations": []}), encoding="utf-8")
    (ep / "hidden" / "oracle.json").write_text(json.dumps({"diagnosis": [3, 100, 0, 130], "work_orders": [{"component": "cooler", "window_end": 60}],
                                                            "finish": True}), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"model": "agent-model"}), encoding="utf-8")
    return ep


def test_single_protocol_makes_one_call_at_temperature_zero_and_writes_the_record(tmp_path):
    ep = episode_folder(tmp_path)
    backend = FakeJudge([RUBRIC_PASS])
    record = judge.judge_episode(ep, backend, "single", agent_model="agent-model", base_url="http://127.0.0.1:1/v1")
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


def test_an_unparseable_reply_is_a_judge_error_not_an_agent_failure(tmp_path):
    ep = episode_folder(tmp_path)
    record = judge.judge_episode(ep, FakeJudge(["the judge rambles"]), "single", agent_model="agent-model")
    assert record["pass"] is None and record["pass_rate"] is None and "unparseable" in record["calls"][0]["error"]
    # under the paper protocol the reviewed calls decide and the garbled one is left out of the rate
    mixed = FakeJudge([RUBRIC_PASS, "garbled", RUBRIC_FAIL, RUBRIC_PASS, RUBRIC_PASS])
    record = judge.judge_episode(ep, mixed, "paper", agent_model="agent-model")
    assert [c["error"] is None for c in record["calls"]] == [True, False, True, True, True]
    assert record["pass_rate"] == 0.75 and record["pass"] is True


def test_main_reads_the_agent_model_from_the_run_folder(tmp_path, monkeypatch, capsys):
    ep = episode_folder(tmp_path)
    backend = FakeJudge([RUBRIC_PASS])
    monkeypatch.setattr(judge, "router_backend", lambda model, base_url: backend)
    judge.main([str(ep), "--model", "judge-model", "--base-url", "http://127.0.0.1:1/v1"])
    record = json.loads((ep / "hidden" / "judge.json").read_text(encoding="utf-8"))
    assert record["agent_model"] == "agent-model" and record["base_url"] == "http://127.0.0.1:1/v1"
    assert "judge pass True" in capsys.readouterr().out
