"""Offline regressions for the v5 prompt/trace layer; never calls a model API."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CausalDiagnosis/__main__.py").is_file())
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v5_protocol_tests_target", Path(__file__).with_name("protocol.py"))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import PlanError, parse_plan
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board


def board():
    b = Board(server=False, reads_take_time=False)
    b.profile = {"components": ["pump"], "healthy": {"pump": 0}, "levels": {"pump": [0, 1, 2]},
                 "sensor_groups": {"group_a": ["pressure"]}, "clock_mode": "declared",
                 "durations": {"read_per_cycle": 1}}
    b.status = {"time": 0, "recorded_from": 0, "recorded_until": -1}
    return b


class V5Test(unittest.TestCase):
    def test_opening_progress_works_without_profile_or_wait(self):
        b = Board(reads_take_time=False, unserved=frozenset({"wait"}))
        self.assertIsNone(protocol.public_progress(b)["forward_read_cycles_per_cycle"])

    def test_contract_says_one_group_string_and_requires_mode_and_steps(self):
        c = protocol.contract(board())
        self.assertEqual(c["argument_types"]["read"]["sensor_group"]["enum"], ["group_a"])
        self.assertIn("never an array", c["argument_types"]["read"]["sensor_group"]["type"])
        self.assertIn("mode", c["required_root_fields"])
        self.assertIn("steps", c["required_root_fields"])

    def test_native_parser_still_rejects_channel_array(self):
        bad = {"branch": None, "mode": "test_first", "steps": [
            {"kind": "read", "args": {"sensor_group": ["pressure"], "cycles": 1}}]}
        with self.assertRaises(PlanError):
            parse_plan(json.dumps(bad), 6, ["pump"])
        bad["steps"][0]["args"]["sensor_group"] = "group_a"
        self.assertEqual(parse_plan(json.dumps(bad), 6, ["pump"]).steps[0].args["sensor_group"], "group_a")

    def test_exhausted_history_omitted_but_original_guard_not_removed(self):
        b = board()
        self.assertNotIn("read", protocol.offered_kinds(b))
        self.assertIn("wait", protocol.offered_kinds(b))
        self.assertIn("diagnose", protocol.offered_kinds(b))
        self.assertIn("time", protocol.offered_kinds(b))
        self.assertIsNotNone(protocol.v4.unavailable_read(protocol._read_step("group_a"), b))

    def test_forward_and_thinking_reads_remain_offered(self):
        b = board()
        b.unserved = frozenset({"wait"})
        self.assertIn("read", protocol.offered_kinds(b))
        b.unserved = frozenset()
        b.profile["clock_mode"] = "generation_time"
        self.assertIn("read", protocol.offered_kinds(b))

    def test_prompt_does_not_offer_wait_when_native_clock_disallows_it(self):
        b = board()
        b.clock_runs = False
        self.assertNotIn("wait", protocol.offered_kinds(b))

    def test_finish_advice_uses_public_verification_without_auto_selecting_finish(self):
        b = board()
        b.diagnosis = {"diagnosis": {"pump": 2}}
        b.verifications = {"pred/1": {"consistent": True, "predicted_combination": {"pump": 0}}}
        before = dict(b.verifications)
        self.assertTrue(protocol.public_progress(b)["post_repair_verifications"]["pred/1"]["consistent"])
        self.assertIn("finish", protocol.offered_kinds(b))
        self.assertIn("wait", protocol.offered_kinds(b))  # an offered choice, not an automatic action
        self.assertEqual(b.verifications, before)

    def test_live_trace_writes_each_event_without_model_or_world_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            with patch.dict(os.environ, {"HPML_DEBUG_TRACE": str(path)}):
                protocol.trace("model_reply", reply="raw invalid reply")
                protocol.trace("plan_decision", check_problems=["wrong argument"])
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["event"] for x in events], ["model_reply", "plan_decision"])
            self.assertEqual(events[0]["reply"], "raw invalid reply")

    def test_post_opening_prompt_is_renderable_and_preserves_exact_retry(self):
        from CausalDiagnosis.debug.maintenance_loop.test_protocol import board as fixture
        b = fixture()
        b.model_candidates = []  # the checker fixture's minimal candidates are not renderable observations
        b.diagnosis = {"diagnosis": {"pump": 2}}
        b.verifications = {"pred/1": {"consistent": True, "predicted_combination": {"pump": 0}}}
        text = protocol.planning_prompt(b, "exact refusal: sensor_group must be a string")
        self.assertIn("exact refusal: sensor_group must be a string", text)
        self.assertIn("post_repair_verifications", text)
        self.assertIn("choose finish", text)
        self.assertNotIn('"sensor_group": "a listed group"', text)  # no fixed read action to copy

    def test_installed_controller_records_parse_failure_and_does_not_execute_it(self):
        spec = importlib.util.spec_from_file_location("v5_test_runner", Path(__file__).with_name("run.py"))
        runner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner_module)
        runner_module.install()
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import CausalLoopRunner
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.episode_result import PlanRecord

        class Backend:
            def generate_with_usage(self, prompt, temperature):
                return SimpleNamespace(text="not JSON", finish_reason="stop")

        r = CausalLoopRunner(Backend(), {})
        r.board, r._retry, r.model_calls, r.fallbacks = Board(), None, 0, 0
        record = PlanRecord(1, "", {}, [])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            with patch.dict(os.environ, {"HPML_DEBUG_TRACE": str(path)}):
                plan, source = r._model_plan(record)
            events = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(plan.steps, [])
            self.assertEqual(source, "no_plan_after_parse_failure")
            self.assertTrue(record.parse_errors)
            self.assertEqual([e["event"] for e in events], ["model_request", "model_reply", "plan_decision"])


if __name__ == "__main__":
    unittest.main()
