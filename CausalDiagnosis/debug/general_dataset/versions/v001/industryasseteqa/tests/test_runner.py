"""Controls for the reusable loop and real isolated MCP transport."""

import copy
import importlib.util
import json
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.tool_loop import run_tool_loop


class FakeSessions:
    def __init__(self):
        self.tools = {"example": [SimpleNamespace(name=name, description="Example tool", inputSchema={"type": "object"})
                                  for name in ("sample_sensor", "file_answer")]}
        self.calls = []

    async def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return json.dumps({"accepted": tool == "file_answer"}), False


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_tool_and_bad_json_make_no_call_then_a_valid_submission_ends(self):
        replies = iter(['not JSON', '{"tool":"invented","args":{}}',
                        '{"tool":"sample_sensor","args":{"value":NaN}}',
                        '{"tool":"sample_sensor","args":{}}', '{"tool":"file_answer","args":{}}'])
        sessions, checkpoints = FakeSessions(), []

        async def ask(messages):
            return {"text": next(replies)}

        def finish(tool, args, text, is_error):
            return {"answer": "example"} if tool == "file_answer" else None

        result = await run_tool_loop(ask, sessions, {"question": "example"}, instructions="Use observations.",
                                     max_turns=5, finish=finish, checkpoint=checkpoints.append)
        self.assertEqual(result["end"], "submitted")
        self.assertEqual([call[1] for call in sessions.calls], ["sample_sensor", "file_answer"])
        self.assertEqual(len(result["model_calls"]), 5)
        self.assertEqual(checkpoints[0]["model_calls"][0].get("response"), None)
        self.assertEqual(checkpoints[-1]["end"], "submitted")

    async def test_model_failure_retains_the_observation_and_failed_request(self):
        sessions = FakeSessions()

        async def ask(messages):
            if len(messages) > 2:
                raise RuntimeError("fixture failure")
            return {"text": '{"tool":"sample_sensor","args":{}}'}

        result = await run_tool_loop(ask, sessions, {}, instructions="Use observations.", max_turns=3,
                                     finish=lambda *args: None, checkpoint=lambda result: None)
        self.assertEqual(result["end"], "error")
        self.assertEqual(len(result["model_calls"]), 2)
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertIn("response", result["trajectory"][0])

    async def test_exhausted_budget_is_incomplete_without_a_submission(self):
        async def ask(messages):
            return {"text": '{"tool":"sample_sensor","args":{}}'}

        result = await run_tool_loop(ask, FakeSessions(), {}, instructions="Use observations.", max_turns=1,
                                     finish=lambda *args: None, checkpoint=lambda result: None)
        self.assertEqual(result["end"], "turn_limit")
        self.assertIsNone(result["answer"])

    async def test_duplicate_tool_names_are_refused_before_model_access(self):
        sessions = FakeSessions()
        sessions.tools["other"] = sessions.tools["example"]
        with self.assertRaisesRegex(ValueError, "duplicate listed tool"):
            await run_tool_loop(None, sessions, {}, instructions="", max_turns=1,
                                finish=lambda *args: None, checkpoint=lambda result: None)


@unittest.skipUnless(importlib.util.find_spec("mcp"), "the MCP dependency is needed for process integration")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_mcp_submission_and_partial_run_records(self):
        from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.runner import run_questions

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            facts, questions = directory / "facts.jsonl", directory / "qa.jsonl"
            fact = {"fact_id": "fact_a", "features": [{"name": "reading_mean", "value": 12.5}],
                    "label": "private_label", "failure_component": "private_component"}
            question = {"qa_id": "question_a", "fact_id": "fact_a", "task_type": "descriptive", "question": "What is the mean?",
                        "direct_answer": "private_reference", "reasoning_answer": "private_reasoning"}
            facts.write_text(json.dumps(fact), encoding="utf-8")
            questions.write_text(json.dumps(question), encoding="utf-8")
            prompts = []

            async def ask(messages):
                prompts.append(copy.deepcopy(messages))
                if len(messages) == 2:
                    step = {"tool": "read_fact", "args": {}}
                else:
                    observed = json.loads(json.loads(messages[-1]["content"])["result"])
                    answer = {"direct_answer": str(observed["features"][0]["value"]),
                              "reasoning_answer": "The released measurement gives the mean.", "confidence": 1.0,
                              "provenance": {"fact_id": observed["fact_id"], "features": [observed["features"][0]["name"]]}}
                    step = {"tool": "submit_answer", "args": {"answer": answer}}
                return {"text": json.dumps(step), "usage": None}

            out = directory / "run"
            meta = await run_questions(facts, questions, ["question_a"], out, sqlite=False, ask=ask,
                                       settings={"backend": "artificial_model"}, max_turns=2)
            self.assertEqual(meta["status"], "completed")
            result = json.loads((out / "episode_00000/result.json").read_text())
            self.assertEqual(result["end"], "submitted", result)
            self.assertEqual(result["answer"]["direct_answer"], "12.5")
            self.assertTrue(result["checks"]["provenance_ok"])
            self.assertTrue((out / "episode_00000/transcript.jsonl").is_file())
            self.assertNotIn("private", json.dumps(prompts))
            transcript = [json.loads(line) for line in (out / "episode_00000/transcript.jsonl").read_text().splitlines()]
            self.assertTrue(any(row.get("kind") == "list" for row in transcript))
            self.assertEqual([row.get("tool") for row in transcript if row.get("tool")], ["read_fact", "submit_answer"])
            with self.assertRaisesRegex(ValueError, "must be new"):
                await run_questions(facts, questions, ["question_a"], out, sqlite=False, ask=ask, settings={}, max_turns=2)

            async def failing(messages):
                if len(messages) > 2:
                    raise RuntimeError("fixture failure")
                return {"text": '{"tool":"read_fact","args":{}}'}

            failed = directory / "failed"
            meta = await run_questions(facts, questions, ["question_a"], failed, sqlite=False, ask=failing,
                                       settings={"backend": "artificial_model"}, max_turns=2)
            self.assertEqual(meta["status"], "incomplete")
            partial = json.loads((failed / "episode_00000/result.json").read_text())
            self.assertEqual(len(partial["trajectory"]), 1)
            self.assertEqual(partial["error"]["type"], "RuntimeError")


@unittest.skipUnless(importlib.util.find_spec("openai"), "the chat client dependency is needed")
class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_client_sends_json_steps_without_a_native_tool_parser(self):
        from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.model import ChatModel

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "body": body})
                response = {"id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture",
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": '{"tool":"read_fact","args":{}}'}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                content = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        model = ChatModel("fixture", f"http://127.0.0.1:{server.server_port}/v1", "fixture-key", 0.0)
        try:
            response = await model([{"role": "user", "content": "Return a JSON step."}])
            schema = {"type": "json_schema", "json_schema": {"name": "plan", "strict": False,
                      "schema": {"type": "object", "properties": {"steps": {"type": "array"}}}}}
            await model([{"role": "user", "content": "Return a plan."}], response_format=schema)
        finally:
            await model.close()
            server.shutdown()
            server.server_close()
            worker.join()
        self.assertEqual(json.loads(response["text"])["tool"], "read_fact")
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertNotIn("tools", requests[0]["body"])
        self.assertEqual(requests[0]["body"]["model"], "fixture")
        self.assertNotIn("response_format", requests[0]["body"])
        self.assertEqual(requests[1]["body"]["response_format"], schema)


if __name__ == "__main__":
    unittest.main()
