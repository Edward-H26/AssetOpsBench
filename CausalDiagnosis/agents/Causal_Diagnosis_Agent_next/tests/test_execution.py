"""Generic execution, public-state checks and normalized answer boundaries."""

import copy
import json
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import patch

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller import AgentRunner
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_check import inspect_plan
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_schema import Plan, Step
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.working_memory.board import Board
from CausalDiagnosis.agents.utils.answer_forms import observations
from CausalDiagnosis.agents.utils.tool_answers import read
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.contract import declaration


def unrelated():
    base = declaration()
    names = {"get_task": "read_manifest", "read_fact": "inspect_package", "submit_answer": "file_report"}
    rows = {names[name]: replace(row, name=names[name], server="packages", after=tuple(names[n] for n in row.after),
                                candidate=("report",) if row.commits else ()) for name, row in base.rows.items()}
    return replace(base, rows=rows, asset_id="packages", vocabulary=replace(base.vocabulary, noun="package", short_noun="package"))


class Sessions:
    def __init__(self):
        self.calls = []
        self.tools = {"packages": [{"name": name, "description": "A package tool", "inputSchema": {
            "type": "object", "properties": {"report": {"type": "object"}} if name == "file_report" else {},
            "required": ["report"] if name == "file_report" else []}} for name in unrelated().rows]}

    async def call(self, server, tool, args):
        self.calls.append(tool)
        if tool == "inspect_package":
            result = {"result": {"weight": 4}, "observations": [{"id": "package/a", "text": "weight 4"}]}
        else:
            result = {"result": {}, "status": {"over": tool == "file_report"}}
        return json.dumps(result), False


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_answer_guard_is_recorded_and_no_later_tool_runs(self):
        sessions = Sessions()

        @asynccontextmanager
        async def opened(paths):
            yield sessions

        class Model:
            max_calls = 1
            takes_response_format = True

            async def generate_with_usage(self, prompt, **kwargs):
                return {"text": json.dumps({"why": "Inspect.", "steps": [
                    {"tool": "read_manifest"}, {"tool": "inspect_package"}]})}

        guarded = {"refused": "the answer could not be read", "status": {},
                   "notifications": [], "observations": [], "guard": "RuntimeError"}
        with patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.open_sessions", opened), \
             patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.tool_answers.read", return_value=guarded):
            result = await AgentRunner(Model(), {}, unrelated()).run("Inspect the package.")
        self.assertEqual(sessions.calls, ["read_manifest"])
        self.assertEqual(result.trajectory[0].guard, "RuntimeError")
        self.assertEqual(result.plans[0].steps[0].status, "surprise")
        self.assertIn("the answer of read_manifest could not be read", result.plans[0].steps[0].note)

    async def test_unrelated_tools_execute_without_qa_tool_names_or_payload_fields(self):
        sessions = Sessions()

        @asynccontextmanager
        async def opened(paths):
            yield sessions

        class Model:
            max_calls = 1
            takes_response_format = True

            async def generate_with_usage(self, prompt, **kwargs):
                return {"text": json.dumps({"why": "Inspect and report.", "steps": [
                    {"tool": "read_manifest", "args": {}}, {"tool": "inspect_package", "args": {}},
                    {"tool": "file_report", "args": {"report": {"weight": 4}}}]})}

        with patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.open_sessions", opened):
            result = await AgentRunner(Model(), {}, unrelated()).run("Inspect the package.")
        self.assertEqual(result.end, "finish", result.error)
        self.assertEqual(sessions.calls, ["read_manifest", "inspect_package", "file_report"])
        self.assertEqual(result.board["commitment"]["args"], {"report": {"weight": 4}})

    def test_check_projects_observation_but_changes_no_live_state(self):
        sessions, dec = Sessions(), unrelated()
        table = dec.table(sessions.tools)
        ledger = dec.rules.ledger(None, {}, (), "Inspect")
        board = Board(table, ledger, "Inspect", Options())
        before = copy.deepcopy(board.to_json(()))
        plan = Plan(None, "Inspect", [Step(name, args, {}, []) for name, args in [
            ("read_manifest", {}), ("inspect_package", {}), ("file_report", {"report": {}})]], [], "")
        checked = inspect_plan(plan, board, ledger, table, ())
        self.assertFalse(checked.problems)
        self.assertEqual(board.to_json(()), before)
        self.assertFalse(ledger.observed)
        self.assertEqual(len(checked.executable()), 3)

    def test_answer_reader_rejects_ambiguous_and_nonfinite_envelopes(self):
        for text in ('{"result":{},"result":1}', '{"result":NaN}', '{"result":{},"refused":"no"}',
                     '{"result":{},"observations":[3]}'):
            self.assertIn("refused", read(text, False))
        response = read('{"result":{"value":4}}\n{"steps_left":2,"cycle":7}', False)
        self.assertEqual(response["status"], {"steps_left": 2, "time": 7})
        self.assertEqual(response["result"], {"value": 4})

    def test_forms_keep_natural_text_and_numeric_range_evidence_distinct(self):
        rows, note = observations("records", [{"n": 1, "t": 0, "values": {"mass": 4}}], {"mass": (1, 3)})
        self.assertIsNone(note)
        self.assertEqual(rows[0]["id"], "obs/1/mass")
        self.assertFalse(rows[0]["inside_band"])
        self.assertIsNotNone(observations("records", [{"n": True, "t": 0, "values": {}}], {})[1])
        self.assertEqual(observations("text", "package ready", {})[0], [{"text": "package ready"}])
