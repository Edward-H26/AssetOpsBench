"""Complete shared-controller episodes over real isolated QA MCP sessions."""

import json
import tempfile
import unittest
from pathlib import Path

from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.runner import run_questions
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.tests.test_qa import records


def plan(*steps):
    return {"text": json.dumps({"why": "Use the released evidence.", "steps": steps})}


def step(name, **args):
    return {"tool": name, "args": args}


class GeneralTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, replies, limit):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fact, question, _answer = records()
            facts, qa = root / "facts.jsonl", root / "qa.jsonl"
            facts.write_text(json.dumps(fact), encoding="utf-8")
            qa.write_text(json.dumps(question), encoding="utf-8")
            prompts, schemas = [], []
            iterator = iter(replies)

            async def ask(messages, *, response_format=None):
                prompts.append(messages)
                schemas.append(response_format)
                reply = next(iterator)
                if isinstance(reply, Exception):
                    raise reply
                return reply

            out = root / "out"
            meta = await run_questions(facts, qa, ["question_a"], out, sqlite=False, ask=ask,
                                       settings={"backend": "artificial_model"}, max_turns=limit, agent="general")
            result = json.loads((out / "episode_00000/result.json").read_text())
            self.assertTrue((out / "episode_00000/hidden").is_dir())
            self.assertTrue(all(schema["type"] == "json_schema" for schema in schemas))
            self.assertTrue(all(schema["json_schema"]["strict"] is True for schema in schemas))
            self.assertNotIn("private_reference", json.dumps(prompts))
            self.assertNotIn("private_label", json.dumps(prompts))
            self.assertNotIn("private_component", json.dumps(prompts))
            return meta, result, prompts

    async def test_complete_episode_records_common_plans_and_dataset_prediction(self):
        answer = records()[2]
        meta, result, _ = await self.run_case([
            plan(step("get_task"), step("read_fact")),
            plan(step("submit_answer", answer=answer))], 2)
        self.assertEqual(meta["status"], "completed", result)
        self.assertEqual(result["end"], "submitted", result)
        self.assertEqual(result["answer"], answer)
        self.assertEqual([call["tool"] for call in result["trajectory"]], ["get_task", "read_fact", "submit_answer"])
        self.assertEqual(result["episode"]["end"], "finish")
        self.assertFalse(result["episode"]["server"])
        self.assertEqual(result["episode"]["board"]["observations"][0]["id"], "fact/fact_a")
        self.assertTrue(result["checks"]["provenance_ok"])
        self.assertEqual(result["episode"]["board"]["reply_schema"]["bound"], 2)
        self.assertTrue(result["episode"]["board"]["reply_schema"]["carried"])

    async def test_refused_submission_does_not_commit_and_can_be_corrected(self):
        answer = records()[2]
        meta, result, prompts = await self.run_case([
            plan(step("get_task"), step("read_fact")),
            plan(step("submit_answer", answer={})),
            plan(step("submit_answer", answer=answer))], 3)
        self.assertEqual(meta["status"], "completed", result)
        self.assertEqual(result["episode"]["plans"][1]["steps"][0]["status"], "surprise")
        self.assertIn("submission failed", json.dumps(prompts[2]))
        self.assertEqual(result["answer"], answer)

    async def test_submission_without_observation_and_bad_reply_make_no_write(self):
        meta, result, _ = await self.run_case([
            {"text": "invalid JSON"}, plan(step("get_task")),
            plan(step("submit_answer", answer=records()[2]))], 3)
        self.assertEqual(meta["status"], "incomplete")
        self.assertEqual(result["end"], "request_budget")
        self.assertIsNone(result["answer"])
        self.assertEqual([call["tool"] for call in result["trajectory"]], ["get_task"])
        self.assertIsNone(result["episode"]["board"]["commitment"])

    async def test_model_failure_saves_answered_reads_and_failed_request(self):
        _meta, result, _ = await self.run_case([
            plan(step("get_task"), step("read_fact")), RuntimeError("fixture endpoint failure")], 2)
        self.assertEqual(result["end"], "error")
        self.assertIn("fixture endpoint failure", result["error"])
        self.assertEqual(len(result["trajectory"]), 2)
        self.assertIn("fixture endpoint failure", result["model_calls"][-1]["error"])
