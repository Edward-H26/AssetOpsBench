"""The audit report of an episode built in process, the one-line tool summaries, the heuristic reading of a closing answer
and the run summary."""
import csv
import json
from pathlib import Path

import pytest

from common import ROOT, answer_claims_recovery, scenario_env_kwargs, scenario_task_text, start_episode
from envs.hydraulic_rig.data import default_facts_path
from envs.hydraulic_rig.scoring import score
from report import episode_report, run_summary, summarise_response
from servers.causal_diagnosis.adapter import CausalTools, load_models
from servers.hydraulic_rig import episode

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

SEED = 268  # the S1 start state (100, 100, 1, 130): the pump at level 1


def _call(path, tool, arguments, action):
    return json.loads(episode.call(path, tool, arguments, action))


def _episode_folder(tmp_path, answer):
    """An S1 episode played in process with both servers' logic: three readings, a ranking, a prediction, the diagnosis,
    the pump replaced (which recovers the rig) and the verification, plus the files a Plan-Execute run would add."""
    ep_dir = tmp_path / "seed000268"
    path = start_episode(ep_dir, SEED, scenario_env_kwargs("S1"), default_facts_path(), task_text=scenario_task_text("S1"))
    tools = CausalTools(load_models(), episode.released_path(path))
    for tier in (1, 2, 3):
        _call(path, "measure", {"tier": tier}, {"type": "measure", "tier": tier})
    ranked = tools.rank_root_causes(2)
    effect = tools.estimate_intervention_effect("replace:pump")
    diagnosis = {"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130}
    _call(path, "submit", diagnosis, {"type": "submit", "diagnosis": diagnosis})
    _call(path, "replace", {"component": "pump"}, {"type": "replace", "component": "pump"})
    verification = tools.verify_outcome(effect.prediction_id)
    env = episode.replay(episode.load_state(path))
    (ep_dir / "log.json").write_text(json.dumps(env.log, default=str), encoding="utf-8")
    (ep_dir / "score.json").write_text(json.dumps(score(env.log), default=str), encoding="utf-8")
    (ep_dir / "trajectory.json").write_text(json.dumps([
        {"step": 1, "task": "rank", "server": "causal_diagnosis", "tool": "rank_root_causes", "tool_args": {"top_k": 2}, "response": ranked.model_dump_json(), "error": None},
        {"step": 2, "task": "replace", "server": "hydraulic_rig", "tool": "replace", "tool_args": {"component": "pump"},
         "response": json.dumps({"event": env.observation()["history"][-1], "cost": 50, "budget_left": 86, "repairs_left": 1, "steps_left": 195, "episode_over": True}), "error": None},
        {"step": 3, "task": "verify", "server": "causal_diagnosis", "tool": "verify_outcome", "tool_args": {"prediction_id": "pred/1"}, "response": verification.model_dump_json(), "error": None},
        {"step": 4, "task": "broken", "server": "hydraulic_rig", "tool": "finish", "tool_args": {}, "response": "", "error": "Unknown tool"},
    ]), encoding="utf-8")
    (ep_dir / "plan.txt").write_text("#Task1: rank\n#Server1: causal_diagnosis\n#Tool1: rank_root_causes\n", encoding="utf-8")
    (ep_dir / "answer.txt").write_text(answer, encoding="utf-8")
    (ep_dir / "llm_calls.json").write_text(json.dumps([{"prompt": "p", "reply": "r", "input_tokens": 10, "output_tokens": 2, "seconds": 0.5}]), encoding="utf-8")
    return ep_dir


def test_episode_report_shows_truth_tool_calls_predictions_outcome_and_answer(tmp_path):
    ep_dir = _episode_folder(tmp_path, "The pump was replaced and the rig is restored.")
    report = episode_report(ep_dir, {"scenario": "S1", "agent": "test", "servers": ["hydraulic_rig", "causal_diagnosis"]})
    assert "Hidden start combination, revealed by the scorer after the episode: (cooler 100, valve 100, pump 1, accumulator 130)." in report
    assert "Task text: Hydraulic unit hyd_1 was flagged" in report
    assert "| 1 | causal_diagnosis.rank_root_causes | `{\"top_k\":2}` | 10 candidate combination(s); top (cooler 100, valve 100, pump 1, accumulator 130)" in report
    assert "| 2 | hydraulic_rig.replace | `{\"component\":\"pump\"}` | replaced pump for 50; budget left 86, repairs left 1, over True |" in report
    assert "| 3 | causal_diagnosis.verify_outcome" in report and "| 4 | hydraulic_rig.finish | `{}` | error: Unknown tool |" in report
    assert "| pred/1 | replace:pump | 3 | 1.000 | 1.000 |" in report
    assert "| pump | 1 | 1 | yes |" in report and "Whole combination right: yes." in report
    assert "| 1 | pump | yes | (cooler 100, valve 100, pump 0, accumulator 130) | 50 |" in report
    assert "Recovered: yes; end reason 'recovered'; cost 64 (before the submission 14); steps 5" in report
    assert "> The pump was replaced and the rig is restored." in report
    assert "claims a recovery; the log says recovered: agreement." in report
    assert "1 calls, 10 input tokens, 2 output tokens, 0.5 s in the model." in report
    assert (ROOT / "src").exists()


def test_episode_report_flags_an_answer_that_contradicts_the_log(tmp_path):
    ep_dir = _episode_folder(tmp_path, "The pump could not be repaired and the rig is not restored.")
    report = episode_report(ep_dir, {})
    assert "claims no recovery; the log says recovered: DISAGREEMENT, read the answer." in report


@pytest.mark.parametrize("answer, claims", [
    ("The rig has been restored to healthy operation.", True),
    ("After replacing the valve the unit is now healthy.", True),
    ("The rig was not restored; the valve remains faulty.", False),
    ("Replacements were made; the valve remains faulty.", False),
    ("scripted run complete", False),
    ("", False),
])
def test_answer_heuristic(answer, claims):
    assert answer_claims_recovery(answer) is claims


def test_summarise_response_reads_rig_events_causal_messages_and_errors():
    measured = json.dumps({"event": {"event": "measure", "tier": 1, "cost": 1, "result": {"a": 1.0, "b": 2.0}, "repeated": False},
                           "cost": 1, "budget_left": 149, "repairs_left": 2, "steps_left": 199, "episode_over": False})
    assert summarise_response("hydraulic_rig", "measure", measured) == "measured tier 1 for 1 (2 values); budget left 149, repairs left 2, over False"
    refused = json.dumps({"event": {"event": "refused", "action": {"type": "replace"}, "reason": "diagnosis must be submitted before the first repair"},
                          "cost": 0, "budget_left": 149, "repairs_left": 2, "steps_left": 198, "episode_over": False})
    assert summarise_response("hydraulic_rig", "replace", refused).startswith("refused: diagnosis must be submitted")
    assert summarise_response("hydraulic_rig", "finish", json.dumps({"note": "over", "episode_over": True})) == "the episode was over; the call changed nothing"
    assert summarise_response("causal_diagnosis", "rank_root_causes", json.dumps({"message": "10 candidates", "candidates": []})) == "10 candidates"
    assert summarise_response("causal_diagnosis", "verify_outcome", json.dumps({"error": "unknown prediction_id"})) == "error: unknown prediction_id"
    assert summarise_response("causal_diagnosis", "verify_outcome", "", "boom") == "error: boom"
    assert summarise_response("hydraulic_rig", "measure", "not json") == "not json"


def test_run_summary_aggregates_per_agent_and_links_reports(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"scenario": "S1", "agent": "scripted", "servers": ["hydraulic_rig"], "started": "t0", "ended": "t1"}), encoding="utf-8")
    rows = [
        {"seed": 1, "scenario": "S1", "record": "[3, 100, 0, 130]", "agent": "scripted", "error": "", "submitted": True, "diagnosis_all_correct": True,
         "recovered": True, "cost": 64, "unnecessary_replacements": 0, "insufficient_evidence": False, "steps": 5, "end_reason": "recovered",
         "answer_claims_recovery": True, "causal_calls": 5, "llm_calls": 14, "input_tokens": 0},
        {"seed": 2, "scenario": "S1", "record": "[20, 100, 0, 130]", "agent": "scripted", "error": "", "submitted": True, "diagnosis_all_correct": False,
         "recovered": False, "cost": 65, "unnecessary_replacements": 1, "insufficient_evidence": False, "steps": 7, "end_reason": "finish",
         "answer_claims_recovery": True, "causal_calls": 5, "llm_calls": 14, "input_tokens": 0},
    ]
    with open(tmp_path / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = run_summary(tmp_path)
    assert "| scripted | 2 | 1.00 | 0.50 | 0.50 | 64.5 | 1 | 0.00 | 6.0 | 0 |" in summary
    assert "2 of 2 answers claim a recovery; 1 of those episodes did not recover; 0 recovered episodes have an answer that claims none." in summary
    assert "[seed000002](seed000002/report.md)" in summary
