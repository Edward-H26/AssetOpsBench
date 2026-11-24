"""The loop of the general agent (CausalDiagnosis/docs/general-agent-plan.md, section The loop). The controller calls a tool of
the asset only when a step names it; the calls it makes of its own are those of a module on an advice tool, which change nothing
on the asset and count no step. Stubs of the freeze; the loop builds them."""
import asyncio
import inspect
import json
import time
from dataclasses import asdict
from typing import Protocol

from CausalDiagnosis.agents.utils import tool_answers
from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions
from .advice import modules_for, expectation_forms
from .agent_options import Options
from .episode_result import EpisodeResult, PlanRecord, StepRecord, ToolRecord
from .plan.plan_check import inspect_plan
from .plan.plan_schema import Plan
from .plan.reply_parser import parse_usable_plan, parse_differential
from .plan.reply_schema import plan_schema_of, differential_schema, response_format, digest
from .prompt.assembler import planning_turn, orientation_turn
from .working_memory.board import Board


class BindingView(Protocol):
    """The read only view of the board and the ledger that a binding reads when it writes the arguments of a listed tool or reads
    its answer: the facts (None until the facts tool is answered), the status of the ledger, and paid(series), the units of a
    series that the episode has paid for."""

    facts: dict | None
    status: dict

    def paid(self, series):
        ...


class AgentRunner:
    """The general agent over the servers of its run: llm the model backend, server_paths {server name: launcher}, declaration
    the Declaration of the run, options the Options (their defaults where None). It builds the table of each episode from the
    listing of its sessions (Declaration.table, or Declaration.listed with the switch binding off)."""

    def __init__(self, llm, server_paths, declaration, options=None):
        self.llm, self.server_paths, self.declaration = llm, server_paths, declaration
        self.options = options or Options()
        self.model_history = []
        self.on_update = None
        self._board = None
        limit = getattr(llm, "max_calls", None)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("the backend must supply an explicit positive max_calls request budget")
        if declaration.binding is not None:
            raise NotImplementedError("surface bindings are not implemented by this runner")

    async def _request(self, prompt, schema):
        kwargs = {"temperature": 0.0}
        if self.options.structured_output:
            if not getattr(self.llm, "takes_response_format", False):
                raise ValueError("structured output requires a backend with response_format support")
            kwargs["response_format"] = response_format("agent_reply", schema)
        if self._board is not None:
            if self._board.reply_schema is None:
                self._board.reply_schema = {"plan": digest(self._schemas["plan"]),
                                            "differential": digest(self._schemas["differential"]),
                                            "bound": 0, "carried": bool(getattr(self.llm, "takes_response_format", False))}
            if "response_format" in kwargs:
                self._board.reply_schema["bound"] += 1
        record = {"prompt": prompt, "settings": kwargs, "response": None, "error": None}
        self.model_history.append(record)
        try:
            method = self.llm.generate_with_usage
            response = await method(prompt, **kwargs) if inspect.iscoroutinefunction(method) else await asyncio.to_thread(method, prompt, **kwargs)
            if inspect.isawaitable(response):
                response = await response
            record["response"] = response if isinstance(response, dict) else {"text": response.text}
            return record["response"]["text"]
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise

    async def run(self, question):
        """One episode on the task message question: the EpisodeResult."""
        self.model_history = []
        plans, trajectory, surprises, differential = [], [], [], []
        self._board = None
        board, end, error, reason = None, "request_budget", None, "model request budget exhausted"
        has_advice = False
        retry, not_run, stood = None, None, []
        mode, oriented = None, False

        def result():
            committed = board.commitment if board else None
            answer = json.dumps(committed["args"], ensure_ascii=False) if committed and end == "finish" else ""
            return EpisodeResult(question, answer, plans, trajectory, len(self.model_history), surprises,
                                 max(0, len(plans) - 1), 0, mode, differential,
                                 board.to_json(()) if board else {}, reason, error,
                                 server=has_advice, end=end)

        async def checkpoint():
            if self.on_update:
                update = self.on_update(result())
                if inspect.isawaitable(update):
                    await update

        try:
            async with open_sessions(self.server_paths) as sessions:
                table = self.declaration.table(sessions.tools) if self.options.binding else self.declaration.listed(sessions.tools)
                has_advice = bool(table.of_role("advice"))
                modules = modules_for(table, self.options)
                if any(row.fills for row in table.tools.values()):
                    raise NotImplementedError("controller argument fills are not implemented by this runner")
                ledger = self.declaration.rules.ledger(None, {}, self.declaration.cards, question)
                board = Board(table, ledger, question, self.options)
                self._board = board
                board.set_modules(modules)
                forms = expectation_forms(table, modules)
                self._schemas = {"plan": plan_schema_of(table, forms, self.options), "differential": differential_schema()}
                while len(self.model_history) < self.llm.max_calls:
                    if ledger.status().get("over"):
                        end, reason = "episode_over", "public status ended the episode"
                        break
                    board.mark_turn(len(self.model_history) + 1)
                    orienting = [row.name for row in table.tools.values() if row.orients]
                    if orienting and not oriented and all(board.holds_answer(name, None) for name in orienting):
                        text = await self._request(orientation_turn(board, table, self.options, modules, retry), differential_schema())
                        differential, refusal = parse_differential(text)
                        board.differential = differential
                        retry = refusal.reason if refusal else None
                        oriented = refusal is None
                        await checkpoint()
                        continue
                    prompt = planning_turn(board, table, self.options, modules, retry, not_run, stood)
                    text = await self._request(prompt, plan_schema_of(table, forms, self.options))
                    board.mark_absorbed(board.observation_count)
                    parsed = parse_usable_plan(text, table, forms, self.options.plan_length)
                    record = PlanRecord(len(plans) + 1, "model", parsed.plan.to_json() if parsed.plan else {}, [])
                    plans.append(record)
                    retry, not_run, stood = None, None, []
                    if parsed.refusal:
                        retry = parsed.refusal.reason
                        record.parse_errors.append(retry)
                        record.guard = parsed.refusal.fault
                        board.add_note(retry)
                        await checkpoint()
                        continue
                    plan = parsed.plan
                    record.candidates = plan.candidates
                    board.model_candidates, board.plan = plan.candidates, plan.to_json()
                    if parsed.unusable:
                        record.cut = asdict(parsed.unusable)
                        not_run = record.cut
                    pending = list(plan.steps)
                    while pending:
                        live = Plan(plan.follow, plan.why, pending, plan.candidates, plan.raw)
                        checked = inspect_plan(live, board, ledger, table, modules)
                        if checked.problems:
                            record.check_problems.extend(checked.problems)
                        available = checked.executable()
                        if not available:
                            retry = checked.problems
                            if retry:
                                board.add_note(str(retry))
                            break
                        entry = available[0]
                        step = entry.step
                        # Expansion is live: consume the prepared call, retaining the remaining calls.
                        expanded_prefix = [item.step for item in checked.steps if item.index == 1]
                        if len(expanded_prefix) > 1:
                            pending = expanded_prefix + pending[1:]
                        pending.pop(0)
                        row = table.tools.get(step.tool)
                        for note in entry.notes:
                            board.add_note(note)
                        if entry.stood:
                            stood.append((len(record.steps) + 1, step.tool, entry.stood.reason))
                            record.steps.append(StepRecord(step.tool, step.args, step.expect, "standing", {"stands": entry.stood.clause}))
                            continue
                        if step.tool == "recall":
                            board.recall(step.args.get("series"), *step.args["observations"])
                            record.steps.append(StepRecord(step.tool, step.args, step.expect, "done"))
                            continue
                        if mode is None and not row.facts and row.role != "knowledge":
                            mode = "test_first" if row.role == "observe" else step.tool
                        started = time.monotonic()
                        call = ToolRecord(len(trajectory) + 1, row.server, step.tool, step.args, "")
                        trajectory.append(call)
                        try:
                            call.response, is_error = await sessions.call(row.server, step.tool, step.args)
                            normalized = tool_answers.read(call.response, is_error, self.declaration.rules.refusal_marks)
                            call.guard = normalized.pop("guard", None)
                            if call.guard:
                                normalized["refused"] = f"the answer of {step.tool} could not be read"
                        except Exception as failure:
                            call.error = f"{type(failure).__name__}: {failure}"
                            raise
                        finally:
                            call.duration_ms = (time.monotonic() - started) * 1000
                        board.record_call(step, call, normalized, None)
                        answer = board.calls[-1]["answer"]
                        failures = []
                        if "refused" in answer:
                            failures.append(answer["refused"])
                        for key, expected in step.expect.items():
                            if key == "no_event" and any(item.get("kind", item.get("event")) in expected for item in answer.get("notifications", [])):
                                failures.append("an excluded event occurred")
                            elif key == "word" and not any(expected in item.get("words", []) for item in answer.get("observations", [])):
                                failures.append("the expected word was absent")
                            elif key == "outside":
                                outside = {item.get("field") for item in answer.get("observations", []) if item.get("inside_band") is False}
                                if set(expected) != outside:
                                    failures.append("the outside fields differ from the expectation")
                        status = "surprise" if failures and self.options.expectations else "failed" if failures else "done"
                        record.steps.append(StepRecord(step.tool, step.args, step.expect, status, answer, "; ".join(failures)))
                        if failures:
                            retry = failures
                            surprises.append({"tool": step.tool, "reasons": failures})
                            board.surprises.append(surprises[-1])
                        await checkpoint()
                        if "refused" in answer or failures and self.options.expectations:
                            not_run = [item.to_json() for item in pending]
                            break
                        if row.ends and board.calls[-1]["answered"]:
                            end, reason = "finish", "the ending write was accepted"
                            break
                    await checkpoint()
                    if end == "finish":
                        break
        except asyncio.CancelledError:
            end, reason = "cancelled", "the caller cancelled the episode"
            await checkpoint()
            raise
        except Exception as failure:
            end, reason = "error", "the episode raised an error"
            error = f"{type(failure).__name__}: {failure}"
        await checkpoint()
        return result()


def request_settings(llm, temperature=None):
    """What a run folder records of the controller's requests on a backend: the blocks decoding and transport, at the
    controller's temperature unless a sampling run sends another."""
    return {"decoding": {"temperature": 0.0 if temperature is None else temperature},
            "transport": {"structured_output": bool(getattr(llm, "takes_response_format", False)),
                          "max_calls": getattr(llm, "max_calls", None)}}
