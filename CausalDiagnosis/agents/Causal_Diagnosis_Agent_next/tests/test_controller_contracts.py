"""The loop respects standing expectations, fills and summary request budgets."""

import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.contract import QARules, QALedger, QAWalk
from ..controller import AgentRunner


@pytest.fixture
def anyio_backend():
    return "asyncio"


class SampleWalk(QAWalk):
    def step(self, row, args):
        result = super().step(row, args)
        if row.answer == "text":
            self.ledger.observed = True
        return result


class SampleLedger(QALedger):
    def record(self, row, args, answer):
        super().record(row, args, answer)
        self.observed = self.observed or bool(answer.get("observations"))

    def walk(self):
        import copy
        return SampleWalk(copy.deepcopy(self))


class SampleRules(QARules):
    def ledger(self, facts, flags, cards, task):
        return SampleLedger()


class Sessions:
    def __init__(self, summary=False):
        self.calls = []
        properties = {"claim": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}, "mode": {"type": "string"}}
        if summary:
            properties["summary"] = {"type": "string"}
        self.tools = {"sample": [
            {"name": name, "description": "A public tool.", "inputSchema": {"type": "object", "properties": props,
             "required": list(props), "additionalProperties": False}}
            for name, props in (("get_task", {}), ("inspect", {}), ("report", properties))]}

    async def call(self, server, name, args):
        self.calls.append((name, args))
        result = {"result": "low", "notifications": [{"event": "alarm"}]} if name == "inspect" else {
            "result": {}, "status": {"over": name == "report"}}
        return json.dumps(result), False


class Model:
    takes_response_format = True

    def __init__(self, replies):
        self.replies, self.requests, self.max_calls = iter(replies), [], len(replies)

    async def generate_with_usage(self, prompt, **kwargs):
        self.requests.append(kwargs)
        return {"text": next(self.replies)}


def plan(*steps):
    return json.dumps({"why": "Read and report.", "steps": list(steps)})


def declaration(summary=False):
    fills = {"evidence": "evidence", "mode": "fixed:checked"}
    if summary:
        fills["summary"] = "closing_summary"
    rows = [ToolRow("get_task", "sample", "knowledge", facts=True),
            ToolRow("inspect", "sample", "knowledge", answer="text", words=("low", "normal")),
            ToolRow("report", "sample", "task", write="answer", ends=True, commits=True,
                    requires_observation=True, fills=fills, candidate=("claim",))]
    return Declaration("invented", "maintenance_qa", None, Vocabulary("sample", "sample"),
                       {row.name: row for row in rows}, {}, {}, (), {}, ("alarm",), SampleRules(), None)


async def run(replies, summary=False):
    sessions, model = Sessions(summary), Model(replies)

    @asynccontextmanager
    async def opened(paths):
        yield sessions

    with patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.open_sessions", opened):
        result = await AgentRunner(model, {}, declaration(summary)).run("Inspect this sample.")
    return result, sessions, model


@pytest.mark.anyio
async def test_failed_expectation_on_a_standing_answer_stops_the_later_write():
    result, sessions, _ = await run([
        plan({"tool": "get_task"}, {"tool": "inspect"}),
        plan({"tool": "inspect", "expect": {"word": "normal"}}, {"tool": "report", "args": {"claim": "ready"}})])
    assert [name for name, _ in sessions.calls] == ["get_task", "inspect"]
    assert result.plans[1].steps[0].status == "surprise"
    assert "no call of this step can change" in result.plans[1].steps[0].note


@pytest.mark.anyio
async def test_standing_no_event_ignores_old_notifications_and_fills_drop_forged_values():
    result, sessions, _ = await run([
        plan({"tool": "get_task"}, {"tool": "inspect"}),
        plan({"tool": "inspect", "expect": {"no_event": ["alarm"]}},
             {"tool": "report", "args": {"claim": "ready", "evidence": ["forged"], "mode": "forged"},
              "evidence": ["obs/1", "forged"]})])
    assert result.end == "finish", result.error
    assert len(sessions.calls) == 3
    assert sessions.calls[-1][1] == {"claim": "ready", "evidence": ["obs/1"], "mode": "checked"}
    assert any("written value was dropped" in note["text"] for note in result.board["notes"])


@pytest.mark.anyio
async def test_closing_summary_has_no_schema_and_is_paid_from_the_request_budget():
    first = plan({"tool": "get_task"}, {"tool": "inspect"}, {"tool": "report", "args": {"claim": "ready"}})
    result, sessions, model = await run([first, "A summary."], summary=True)
    assert result.end == "finish", result.error
    assert sessions.calls[-1][1]["summary"] == "A summary."
    assert "response_format" in model.requests[0] and "response_format" not in model.requests[1]
    assert result.board["reply_schema"]["bound"] == 1
    result, sessions, _ = await run([first], summary=True)
    assert result.end == "request_budget" and len(sessions.calls) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("room_for_plan", [True, False])
async def test_context_compaction_uses_the_backend_test_and_counts_the_summary(room_for_plan):
    sessions = Sessions()
    first = plan({"tool": "get_task"}, {"tool": "inspect"})
    last = plan({"tool": "report", "args": {"claim": "ready"}})

    class ContextModel(Model):
        def fits_context(self, prompt):
            return not ("# Planning" in prompt and "retire this note" in prompt)

    replies = [first, "Earlier work summarized."] + ([last] if room_for_plan else [])
    model = ContextModel(replies)

    @asynccontextmanager
    async def opened(paths):
        yield sessions

    with patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.open_sessions", opened):
        runner = AgentRunner(model, {}, declaration())
        seeded = False

        def checkpoint(episode):
            nonlocal seeded
            if not seeded and len(episode.trajectory) == 2:
                runner._board.add_note("retire this note")
                seeded = True

        runner.on_update = checkpoint
        result = await runner.run("Inspect this sample.")
    assert result.end == ("finish" if room_for_plan else "request_budget"), result.error
    assert result.board["summary_text"] == "Earlier work summarized."
    assert "response_format" not in model.requests[1]
    assert result.model_calls == len(replies)
    assert result.board["observations"][0]["id"] == "obs/1"


def test_outside_expectations_judge_served_prefix_fields_without_requiring_unserved_fields():
    from ..execution import expectation_failures
    from ..plan.plan_schema import Step
    step = Step("read", {}, {"outside": ["pressure", "not_served"]}, [])
    answer = {"observations": [{"field": "pressure_mean", "inside_band": False}]}
    assert not expectation_failures(step, answer)
    answer["observations"][0]["inside_band"] = True
    assert expectation_failures(step, answer) == ["pressure was served inside its normal range"]


@pytest.mark.anyio
async def test_an_initially_cut_step_does_not_resume_after_the_opening_answers():
    from dataclasses import replace
    sessions = Sessions()
    sessions.tools["sample"].append({"name": "operate", "description": "Operate a listed component.",
        "inputSchema": {"type": "object", "properties": {"component": {"type": "string"}}, "required": ["component"]}})
    original = sessions.call

    async def call(server, name, args):
        if name == "get_task":
            sessions.calls.append((name, args))
            return json.dumps({"result": {"components": ["sample"]}}), False
        return await original(server, name, args)

    sessions.call = call
    dec = declaration()
    dec = replace(dec, rows={**dec.rows, "operate": ToolRow("operate", "sample", "act", write="probe", values={"component": "components"})})
    model = Model([plan({"tool": "get_task"}, {"tool": "operate", "args": {"component": "sample"}})])

    @asynccontextmanager
    async def opened(paths):
        yield sessions

    with patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.open_sessions", opened):
        result = await AgentRunner(model, {}, dec).run("Inspect this sample.")
    assert [name for name, _ in sessions.calls] == ["get_task"]
    assert result.plans[0].cut["step"] == 2
    assert result.plans[0].cut["kinds"] == ["operate"]
    assert [item["tool"] for item in result.board["plan"]["steps"]] == ["get_task"]


@pytest.mark.anyio
async def test_a_token_capped_reply_runs_nothing_even_when_its_json_is_complete():
    sessions = Sessions()
    complete = plan({"tool": "get_task"}, {"tool": "inspect"}, {"tool": "report", "args": {"claim": "ready"}})

    class CutModel(Model):
        async def generate_with_usage(self, prompt, **kwargs):
            response = await super().generate_with_usage(prompt, **kwargs)
            response["finish_reason"] = "length" if len(self.requests) == 1 else "stop"
            if len(self.requests) == 2:
                assert "token cap" in prompt
            return response

    model = CutModel([complete, complete])

    @asynccontextmanager
    async def opened(paths):
        yield sessions

    with patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller.open_sessions", opened):
        result = await AgentRunner(model, {}, declaration()).run("Inspect this sample.")
    assert result.end == "finish", result.error
    assert result.plans[0].steps == [] and result.plans[0].parse_errors
    assert [name for name, _ in sessions.calls] == ["get_task", "inspect", "report"]
