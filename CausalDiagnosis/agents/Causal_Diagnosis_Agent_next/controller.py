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
from .execution import expectation_failures, filled_arguments, held_answer
from .plan.messages import STANDS_EXPECTATION, REPLY_TOKEN_CAP
from .prompt.assembler import summary_turn
from .prompt.assembler import earlier_turns_turn
from .media import served_images


class _RequestBudgetExhausted(Exception):
    """A summary and a planning request share the caller's model-call budget."""


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
        if len(self.model_history) >= self.llm.max_calls:
            raise _RequestBudgetExhausted("model request budget exhausted")
        kwargs = {"temperature": 0.0}
        if self.options.structured_output and schema is not None:
            if not getattr(self.llm, "takes_response_format", False):
                raise ValueError("structured output requires a backend with response_format support")
            kwargs["response_format"] = response_format("agent_reply", schema)
        images = served_images(self._board)
        if images:
            if not getattr(self.llm, "supports_images", False):
                raise ValueError("the tool returned images but this backend declares no image transport")
            kwargs["images"] = images
        if self._board is not None:
            if self._board.reply_schema is None:
                self._board.reply_schema = {"plan": digest(self._schemas["plan"]),
                                            "differential": digest(self._schemas["differential"]),
                                            "bound": 0, "carried": bool(getattr(self.llm, "takes_response_format", False))}
            if "response_format" in kwargs:
                self._board.reply_schema["bound"] += 1
        settings = {key: value for key, value in kwargs.items() if key != "images"}
        if images:
            settings["images"] = [{key: value for key, value in item.items() if key != "data"} for item in images]
        record = {"prompt": prompt, "settings": settings, "response": None, "error": None}
        self.model_history.append(record)
        try:
            method = self.llm.generate_with_usage
            response = await method(prompt, **kwargs) if inspect.iscoroutinefunction(method) else await asyncio.to_thread(method, prompt, **kwargs)
            if inspect.isawaitable(response):
                response = await response
            record["response"] = response if isinstance(response, dict) else {"text": response.text}
            self._last_finish_reason = record["response"].get("finish_reason")
            if self._last_finish_reason == "length" and schema is None:
                raise ValueError("the summary reply was cut at the token cap")
            return record["response"]["text"]
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise

    async def _planning_prompt(self, board, table, modules, retry, not_run, stood):
        """Compact only when the backend supplies an actual context-fit test."""
        checker = getattr(self.llm, "fits_context", None)

        def prompt(memory):
            return planning_turn(memory, table, self.options, modules, retry, not_run, stood)

        if checker is None:
            return prompt(board)

        def fits_text(text):
            images = served_images(board)
            result = checker(text, images=images) if images else checker(text)
            if not isinstance(result, bool):
                raise ValueError("fits_context must synchronously return a boolean")
            return result

        def fits(memory):
            return fits_text(prompt(memory))

        board.drop_unfit_recall(fits)
        board.compact(fits=fits, after_call=len(board.calls))
        if not fits(board) and self.options.summary_turn:
            entries = board.replaceable()
            if entries:
                for left_out in range(len(entries) + 1):
                    view = board.without(entries[:left_out])
                    request = earlier_turns_turn(view, table, self.options, modules, left_out)
                    if fits_text(request):
                        text = await self._request(request, None)
                        board.apply_summary(text, entries, left_out)
                        break
        if not fits(board):
            raise ValueError("the protected board cannot fit the backend's declared context")
        return prompt(board)

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
            if board is not None and plans:
                last = plans[-1]
                board.plan = {"proposal": last.plan, "steps": [asdict(item) for item in last.steps],
                              "cut": last.cut, "surprise": last.surprise}
            if self.on_update:
                update = self.on_update(result())
                if inspect.isawaitable(update):
                    await update

        try:
            async with open_sessions(self.server_paths) as sessions:
                table = self.declaration.table(sessions.tools) if self.options.binding else self.declaration.listed(sessions.tools)
                has_advice = bool(table.of_role("advice"))
                modules = modules_for(table, self.options)
                unsupported = [(row.name, source) for row in table.tools.values() for source in row.fills.values()
                               if source not in ("evidence", "closing_summary") and not source.startswith("fixed:")]
                if unsupported:
                    raise NotImplementedError(f"controller fill providers are unavailable: {unsupported}")
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
                        if self._last_finish_reason == "length":
                            retry = REPLY_TOKEN_CAP
                            board.add_note(retry, tie="differential")
                            await checkpoint()
                            continue
                        differential, refusal = parse_differential(text)
                        board.differential = differential
                        retry = refusal.reason if refusal else None
                        oriented = refusal is None
                        await checkpoint()
                        continue
                    prompt = await self._planning_prompt(board, table, modules, retry, not_run, stood)
                    text = await self._request(prompt, plan_schema_of(table, forms, self.options))
                    if self._last_finish_reason == "length":
                        record = PlanRecord(len(plans) + 1, "model", {}, [])
                        record.parse_errors.append(REPLY_TOKEN_CAP)
                        plans.append(record)
                        retry = REPLY_TOKEN_CAP
                        board.add_note(retry, turn=record.index)
                        await checkpoint()
                        continue
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
                    initial = inspect_plan(plan, board, ledger, table, modules)
                    record.check_problems.extend(initial.problems)
                    if any(index is None for index, _ in initial.problems):
                        retry = initial.problems
                        board.add_note(str(retry), turn=record.index)
                        await checkpoint()
                        continue
                    stop = min((index for index, _ in initial.problems), default=len(plan.steps) + 1)
                    if stop <= len(plan.steps):
                        record.cut = {"step": stop, "reasons": [reason for index, reason in initial.problems if index == stop],
                                      "kinds": [item.tool for item in plan.steps[stop - 1:]] +
                                               (list(parsed.unusable.kinds) if parsed.unusable else [])}
                        retry, not_run = initial.problems, record.cut
                    pending = list(enumerate(plan.steps[:stop - 1], 1))
                    while pending:
                        live = Plan(plan.follow, plan.why, [item for _, item in pending], plan.candidates, plan.raw)
                        checked = inspect_plan(live, board, ledger, table, modules)
                        if checked.problems:
                            scoped = [(None if index is None else pending[index - 1][0], reason) for index, reason in checked.problems]
                            record.check_problems.extend(item for item in scoped if item not in record.check_problems)
                        available = checked.executable()
                        if not available:
                            retry = scoped if checked.problems else []
                            not_run = [item.to_json() for _, item in pending]
                            if retry:
                                board.add_note(str(retry))
                            break
                        entry = available[0]
                        step = entry.step
                        if checked.problems:
                            cut = min((index for index, _ in checked.problems if index is not None), default=len(pending) + 1)
                            record.cut = {"step": pending[cut - 1][0], "reasons": [reason for index, reason in checked.problems if index == cut],
                                          "kinds": [item.tool for _, item in pending[cut - 1:]]} if cut <= len(pending) else record.cut
                            not_run = record.cut
                            pending = pending[:cut - 1]
                        # Expansion is live: consume the prepared call, retaining the remaining calls.
                        expanded_prefix = [item.step for item in checked.steps if item.index == 1]
                        if len(expanded_prefix) > 1:
                            pending = [(pending[0][0], item) for item in expanded_prefix] + pending[1:]
                        pending.pop(0)
                        row = table.tools.get(step.tool)
                        for note in entry.notes:
                            board.add_note(note)
                        if entry.stood:
                            stood.append((len(record.steps) + 1, step.tool, entry.stood.reason))
                            failures = expectation_failures(step, held_answer(board, step, row), standing=True) if self.options.expectations else []
                            note = STANDS_EXPECTATION.format(stands=entry.stood.reason, detail="; ".join(failures)) if failures else entry.stood.reason
                            record.steps.append(StepRecord(step.tool, step.args, step.expect, "surprise" if failures else "standing",
                                                           {"stands": entry.stood.clause}, note))
                            if failures:
                                retry = note
                                record.surprise = {"tool": step.tool, "reasons": [note]}
                                surprises.append(record.surprise)
                                board.surprises.append(record.surprise)
                                not_run = [item.to_json() for _, item in pending]
                                break
                            continue
                        if step.tool == "recall":
                            board.recall(step.args.get("series"), *step.args["observations"])
                            record.steps.append(StepRecord(step.tool, step.args, step.expect, "done"))
                            continue
                        if mode is None and not row.facts and row.role != "knowledge":
                            mode = "test_first" if row.role == "observe" else step.tool
                        started = time.monotonic()
                        summary = None
                        if "closing_summary" in row.fills.values():
                            if len(self.model_history) >= self.llm.max_calls:
                                pending.clear()
                                reason = "model request budget leaves no closing summary"
                                break
                            summary = await self._request(summary_turn(board, table, self.options, modules), None)
                        tool_args = filled_arguments(step, row, board, summary)
                        call = ToolRecord(len(trajectory) + 1, row.server, step.tool, tool_args, "")
                        trajectory.append(call)
                        try:
                            call.response, is_error = await sessions.call(row.server, step.tool, tool_args)
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
                        failures = expectation_failures(step, answer) if self.options.expectations else ([answer["refused"]] if "refused" in answer else [])
                        status = "surprise" if "refused" in answer or failures and self.options.expectations else "done"
                        record.steps.append(StepRecord(step.tool, step.args, step.expect, status, answer, "; ".join(failures)))
                        if failures:
                            retry = failures
                            record.surprise = {"tool": step.tool, "reasons": failures}
                            surprises.append(record.surprise)
                            board.surprises.append(record.surprise)
                        await checkpoint()
                        if "refused" in answer or failures and self.options.expectations:
                            not_run = [item.to_json() for _, item in pending]
                            break
                        if row.ends and board.calls[-1]["answered"]:
                            end, reason = "finish", "the ending write was accepted"
                            break
                    await checkpoint()
                    if end == "finish":
                        break
        except _RequestBudgetExhausted as failure:
            end, reason = "request_budget", str(failure)
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
