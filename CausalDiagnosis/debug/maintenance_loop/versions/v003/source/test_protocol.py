"""Run directly: avoids the repository's Linux-only episode-runner imports on Windows."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import Plan, Step, check_plan
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
from CausalDiagnosis.debug.maintenance_loop.protocol import first_step, instructions


def board():
    b = Board(server=False)
    b.set_profile({
        "asset_id": "test_asset", "components": ["pump"], "levels": {"pump": [0, 1, 2]},
        "healthy": {"pump": 0}, "sensor_groups": {"all": ["pressure"]},
        "prices": {"read": {"all": 1}, "parts": {"pump": 30}, "repair_charge": 20, "visit": 15},
        "durations": {"read_per_cycle": 1, "advance_per_cycle": 1,
                      "lead_time": {"pump": 3}, "repair_downtime": {"pump": 2}},
        "downtime_rate": {}, "budget": 300, "horizon": 60, "max_steps": 60,
        "clock_mode": "declared", "statistics": ["mean"],
        "normal_ranges": {"pressure_mean": {"low": 0, "high": 10}},
    })
    b.status = {"time": 0, "budget_left": 300, "steps_left": 60}
    b.opening_done = ["profile", "failure_modes", "fix_history"]
    b.model_candidates = [{"rank": 1, "combination": {"pump": 2}, "own": True}]
    b.manual_channels = {"pump": {"pressure"}}
    return b


class ProtocolTest(unittest.TestCase):
    def test_opening_valid_prefix_survives_invalid_tail(self):
        b = Board()
        plan = Plan(None, "test_first", "", [Step("profile"), Step("diagnose")])
        self.assertTrue(check_plan(plan, b))
        self.assertEqual(check_plan(first_step(plan), b), [])
        self.assertEqual(len(plan.steps), 2)  # no mutation of model proposal

    def test_illegal_first_action_is_still_refused(self):
        self.assertTrue(check_plan(first_step(Plan(None, "act_now", "", [Step("order", {"component": "pump"})])), Board()))

    def test_duplicate_order_is_still_refused(self):
        b = board()
        b.diagnosis = {"diagnosis": {"pump": 2}}
        b.orders = {"wo-1": {"id": "wo-1", "component": "pump", "created_t": 0,
                              "expected_close_t": 5, "status": "open", "visit": 1, "prediction_id": "plan 1"}}
        p = Plan(None, "act_now", "", [Step("order", {"component": "pump"})])
        self.assertTrue(any("still open" in x for x in check_plan(first_step(p), b)))

    def test_wait_progress_and_finish_guard_use_native_checker(self):
        b = board()
        b.diagnosis = {"diagnosis": {"pump": 2}}
        b.orders = {"wo-1": {"id": "wo-1", "component": "pump", "created_t": 0,
                              "expected_close_t": 5, "status": "open", "visit": 1, "prediction_id": "plan 1"}}
        self.assertEqual(check_plan(Plan(None, "act_now", "", [Step("wait", {"cycles": 5})]), b), [])
        self.assertTrue(check_plan(Plan(None, "act_now", "", [Step("finish")]), b))

    def test_no_wait_does_not_get_invented(self):
        b = board()
        b.unserved = frozenset({"wait"})
        self.assertIn('"wait_available": false', instructions(b))
        self.assertTrue(check_plan(Plan(None, "act_now", "", [Step("wait", {"cycles": 5})]), b))

    def test_overlay_installs_without_editing_official_module(self):
        from CausalDiagnosis.debug.maintenance_loop.run import install
        install()
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent import controller
        text = controller.planning_turn(Board(), 6)
        self.assertIn("maintenance-loop-v3", text)
        self.assertIn("exactly ONE", text)

    def test_controller_uses_checked_prefix_and_records_discarded_tail(self):
        import json
        from types import SimpleNamespace
        from CausalDiagnosis.debug.maintenance_loop.run import install
        install()
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import CausalLoopRunner
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.episode_result import PlanRecord

        class Backend:
            def generate_with_usage(self, prompt, temperature):
                return SimpleNamespace(text=json.dumps({"branch": None, "mode": "test_first",
                    "steps": [{"kind": "profile"}, {"kind": "diagnose"}]}), finish_reason="stop")

        runner = CausalLoopRunner(Backend(), {})
        runner.board, runner._retry, runner.model_calls, runner.fallbacks = Board(), None, 0, 0
        record = PlanRecord(1, "", {}, [])
        plan, source = runner._model_plan(record)
        self.assertEqual([s.kind for s in plan.steps], ["profile"])
        self.assertEqual(source, "debug_model_first_step")
        self.assertIn("diagnose", runner.board.notes[-1])
        self.assertEqual(record.check_problems, [])

    def test_second_diagnosis_is_not_enabled_by_debug(self):
        b = board()
        b.diagnosis = {"diagnosis": {"pump": 2}}
        problems = check_plan(Plan(1, "act_now", "", [Step("diagnose", {"rank": 1})]), b)
        self.assertTrue(any("already made" in x for x in problems))


if __name__ == "__main__":
    unittest.main()
