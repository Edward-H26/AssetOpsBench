"""The feedback card of gepa_prompt_optimization/feedback_card.py, which the online loop of GEPA hands the reflection model, and its leak guard. The card
keeps the check flags, the outcome, the public trajectory and the diagnostics messages of an episode, and no hidden field
and no seed reaches the card or its text. The online loop is the only GEPA driver of the folder."""
import json
from pathlib import Path

from CausalDiagnosis.training.gepa_prompt_optimization import feedback_card


def test_feedback_card_keeps_the_check_flags_the_outcome_and_the_diagnostics_messages(tmp_path):
    ep = tmp_path / "call00001_seed000003"
    (ep / "hidden").mkdir(parents=True)
    (ep / "hidden" / "verdict.json").write_text(json.dumps({
        "checks": {"count_gate": {"ok": True}, "hard": {"ok": False, "diagnosis_all_correct": False}}, "failure_class": "premature_diagnosis",
        "pass": False, "success": False, "abstained": False, "unnecessary": 0, "cost": 70, "end_reason": "finish", "steps": 3, "refusals": 1}),
        encoding="utf-8")
    (ep / "trajectory.json").write_text(json.dumps([
        {"server": "causal_diagnosis", "tool": "rank_root_causes", "tool_args": {}, "response": json.dumps({"message": "two candidates remain"}), "error": None},
        {"server": "hydraulic_rig", "tool": "read_sensors", "tool_args": {"group": "tier1"}, "response": "x" * 1000, "error": None},
        {"server": "causal_diagnosis", "tool": "verify_outcome", "tool_args": {}, "response": "", "error": "Unknown tool"}]), encoding="utf-8")
    card = feedback_card.episode_card(ep, 0.0)
    text = feedback_card.feedback_string(card)
    assert "check count_gate: ok" in text and "check hard: failed" in text and "diagnosis_all_correct" not in text
    assert "pass False, success False, failure class premature_diagnosis, abstained False, unnecessary orders 0" in text
    assert "reward 0.0, cost 70, end reason finish, steps 3, refusals 1" in text
    assert card["diagnostics_messages"] == ["rank_root_causes: two candidates remain"] and "- verify_outcome {} error Unknown tool" in text
    assert card["trajectory"][1]["response"] == "x" * 1000 and "RESPONSE_SHOWN" not in vars(feedback_card)  # shown whole


# Every field that tells which components of a past episode were faulty, with a value no allowed field can produce.
HIDDEN_KEYS = ("work_orders_oracle", "diagnosis_per_component", "hidden_start", "truth_at_write", "cost_oracle", "cost_normalized", "seed")
HIDDEN_VALUES = ("SENTINEL", "987654.25", "0.123457", "424242")
TRAJECTORY = [{"step": 1, "task": "read", "server": "hydraulic_rig", "tool": "read_sensors", "tool_args": {"group": "tier1"},
               "response": json.dumps({"t": 3}), "error": None, "duration_ms": 12.0},
              {"step": 2, "task": "rank", "server": "causal_diagnosis", "tool": "rank_root_causes", "tool_args": {},
               "response": json.dumps({"message": "two candidates remain"}), "error": None, "duration_ms": 30.0}]


def _poisoned_verdict():
    return {"checks": {"count_gate": {"ok": True},
                       "hard": {"ok": False, "diagnosis_all_correct": False, "diagnosis_per_component": {"cooler": "SENTINEL_PER_COMPONENT"},
                                "work_orders_oracle": ["SENTINEL_ORACLE_SET"]}},
            "diagnosis_per_component": {"cooler": "SENTINEL_PER_COMPONENT"}, "hidden_start": {"cooler": "SENTINEL_HIDDEN_START"},
            "truth_at_write": {"valve": "SENTINEL_TRUTH_AT_WRITE"}, "cost_oracle": 987654.25, "cost_normalized": 0.123457, "seed": 424242,
            "pass": False, "success": False, "failure_class": "wrong_diagnosis", "abstained": False, "unnecessary": 1, "cost": 70,
            "end_reason": "finish", "steps": 5, "refusals": 0}


def _write_poisoned_episode(ep_dir):
    (ep_dir / "hidden").mkdir(parents=True)
    (ep_dir / "hidden" / "verdict.json").write_text(json.dumps(_poisoned_verdict()), encoding="utf-8")
    (ep_dir / "trajectory.json").write_text(json.dumps(TRAJECTORY), encoding="utf-8")


def _leaks(text):
    return [k for k in HIDDEN_KEYS if k in text] + [v for v in HIDDEN_VALUES if v in text]


def test_no_hidden_field_reaches_the_card_or_the_feedback_string(tmp_path):
    ep = tmp_path / "call00001_seed424242"
    _write_poisoned_episode(ep)
    card = feedback_card.episode_card(ep, 0.3)
    assert _leaks(json.dumps(card)) == [] and _leaks(feedback_card.feedback_string(card)) == []
    assert set(card) == {"assets", "reward", "cost", "end_reason", "steps", "refusals", "trajectory", "diagnostics_messages"}
    assert set(card["assets"][0]) == {"checks", "pass", "success", "failure_class", "abstained", "unnecessary_orders"}
    assert card["assets"][0]["checks"] == {"count_gate": True, "hard": False} and card["reward"] == 0.3 and card["cost"] == 70


def test_the_online_loop_is_the_only_gepa_driver_beside_its_feedback_card():
    """The offline optimizer over the planner prompt of Plan-Execute left the fork, so the GEPA modules of the folder are the
    online loop, gepa_prompt_optimization/, and the feedback card it hands the reflection model."""
    assert Path(feedback_card.__file__).parent.name == "gepa_prompt_optimization" and not sorted(Path(feedback_card.__file__).parents[1].glob("gepa_*.py"))
