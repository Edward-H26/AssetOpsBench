"""The GEPA adapter with a fake episode player: the prompt override, the scores and traces of evaluate, the reflective
dataset, the placeholder guard and the feedback text; gepa itself is not imported."""
import json

import gepa_optimize as g  # registers the agent package by path first, so src/agent/__init__ and its SDK imports never run
from agent.plan_execute import planner as planner_module  # noqa: E402


def test_seed_candidate_is_the_runner_prompt_with_its_placeholders():
    candidate = g.seed_candidate()
    assert candidate[g.PROMPT_COMPONENT] == planner_module._PLAN_PROMPT
    assert all(p in candidate[g.PROMPT_COMPONENT] for p in g.PLACEHOLDERS)


def test_planner_prompt_override_is_restored_even_after_an_error():
    original = planner_module._PLAN_PROMPT
    with g.planner_prompt("custom {servers} {question}"):
        assert planner_module._PLAN_PROMPT == "custom {servers} {question}"
    assert planner_module._PLAN_PROMPT == original
    try:
        with g.planner_prompt("broken"):
            raise RuntimeError("episode failed")
    except RuntimeError:
        pass
    assert planner_module._PLAN_PROMPT == original


def _fake_play(seed, prompt_text):
    row = {"seed": seed, "reward": 0.5 if "better" in prompt_text else 0.1, "error": "", "end_reason": "finish", "cost": 40, "cost_oracle": 40}
    trace = {"question": f"q{seed}", "plan": "#Task1: read", "answer": f"answer {seed}", "feedback": "check hard: ok", "episode": f"call_{seed}"}
    return row, trace


def test_evaluate_plays_one_episode_per_item_and_scores_the_reward(tmp_path):
    adapter = g.PlannerPromptAdapter(None, tmp_path, {}, play=_fake_play)
    batch = adapter.evaluate([{"seed": 1}, {"seed": 2}], {g.PROMPT_COMPONENT: "better {servers} {question}"}, capture_traces=True)
    assert batch.scores == [0.5, 0.5] and batch.outputs == ["answer 1", "answer 2"]
    assert [t["question"] for t in batch.trajectories] == ["q1", "q2"]
    assert adapter.metric_calls == 2 and [r["gepa_call"] for r in adapter.rows] == [1, 2]
    without_traces = adapter.evaluate([{"seed": 3}], {g.PROMPT_COMPONENT: "{servers} {question}"})
    assert without_traces.scores == [0.1] and without_traces.trajectories is None and adapter.metric_calls == 3


def test_a_candidate_without_the_placeholders_scores_zero_and_plays_nothing(tmp_path):
    adapter = g.PlannerPromptAdapter(None, tmp_path, {}, play=_fake_play)
    batch = adapter.evaluate([{"seed": 1}], {g.PROMPT_COMPONENT: "no placeholders here"}, capture_traces=True)
    assert batch.scores == [0.0] and adapter.rows == [] and adapter.metric_calls == 1
    assert batch.trajectories[0]["feedback"] == g.MISSING_PLACEHOLDERS


def test_reflective_dataset_carries_question_outputs_and_feedback(tmp_path):
    adapter = g.PlannerPromptAdapter(None, tmp_path, {}, play=_fake_play)
    batch = adapter.evaluate([{"seed": 7}], {g.PROMPT_COMPONENT: "{servers} {question}"}, capture_traces=True)
    dataset = adapter.make_reflective_dataset({g.PROMPT_COMPONENT: "{servers} {question}"}, batch, [g.PROMPT_COMPONENT])
    assert list(dataset) == [g.PROMPT_COMPONENT]
    item = dataset[g.PROMPT_COMPONENT][0]
    assert item["Inputs"] == {"question": "q7"} and item["Generated Outputs"] == {"plan": "#Task1: read", "answer": "answer 7"}
    assert item["Feedback"] == "check hard: ok"


def test_feedback_text_reads_the_checks_the_failure_class_and_the_diagnostics_messages(tmp_path):
    ep = tmp_path / "call00001_seed000003"
    (ep / "hidden").mkdir(parents=True)
    (ep / "hidden" / "verdict.json").write_text(json.dumps({"checks": {"count_gate": {"ok": True}, "hard": {"ok": False, "diagnosis_all_correct": False}},
                                                             "failure_class": "premature_diagnosis"}), encoding="utf-8")
    (ep / "trajectory.json").write_text(json.dumps([
        {"server": "causal_diagnosis", "tool": "rank_root_causes", "response": json.dumps({"message": "two candidates remain"}), "error": None},
        {"server": "hydraulic_rig", "tool": "finish", "response": "{}", "error": None},
        {"server": "causal_diagnosis", "tool": "verify_outcome", "response": "", "error": "Unknown tool"}]), encoding="utf-8")
    text = g.feedback_text(ep, {"error": "", "end_reason": "finish", "cost": 70, "cost_oracle": 40, "reward": 0.0})
    assert "check count_gate: ok" in text and 'check hard: failed {"diagnosis_all_correct": false}' in text
    assert "failure class: premature_diagnosis; end reason: finish; cost 70 against the oracle's 40; reward 0.0" in text
    assert "- rank_root_causes: two candidates remain" in text and "verify_outcome" not in text


def test_result_record_reads_a_result_object_defensively():
    class Result:
        candidates = [{g.PROMPT_COMPONENT: "seed {servers} {question}"}, {g.PROMPT_COMPONENT: "best {servers} {question}"}]
        best_idx = 1
        best_candidate = candidates[1]
        val_subscores = [[0.1, 0.2], [0.5, 0.7]]
        val_aggregate_scores = [0.15, 0.6]
        total_metric_calls = 9

    class Args:
        max_metric_calls, train_seeds, val_seeds, first_seed, reflection_minibatch, model, base_url = 20, 4, 2, 0, 3, "m", "http://x/v1"

    adapter = g.PlannerPromptAdapter(None, "/nonexistent", {}, play=_fake_play)
    adapter.metric_calls = 9
    record = g.result_record(Result(), adapter, Args())
    assert record["best_candidate"].startswith("best") and record["seed_val_scores"] == [0.1, 0.2] and record["best_val_scores"] == [0.5, 0.7]
    assert record["episodes_played"] == 9 and record["total_metric_calls"] == 9 and len(record["candidates"]) == 2
    empty = g.result_record(object(), adapter, Args())
    assert empty["best_candidate"] is None and empty["candidates"] == [] and empty["seed_val_scores"] == []
