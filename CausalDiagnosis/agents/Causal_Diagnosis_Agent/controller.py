import asyncio
import time
from dataclasses import asdict, replace
from CausalDiagnosis.agents.utils import core_texts
from CausalDiagnosis.agents.utils.mcp_sessions import SESSIONS, ToolTransportError, open_sessions
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board, band_verdict, channels_outside, combination_text, fmt
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import (ESTIMATE, OPENING_KINDS, Plan, Refusal, Step, batch_reads, check_plan, exploration, parse_differential, parse_usable_plan, ranking_detail,
                   read_reply, reply_schema, rule_evidence, rule_plan, standing, step_problems, submitted_levels)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.plan_check import NOT_COLLAPSED, second_commitment
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.standing import standing_commitment
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import earlier_turns_turn, orientation_turn, planning_turn, summary_turn
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import CAUSAL_SERVER, RigSurface, not_json
from CausalDiagnosis.agents.utils.idle_hook import aiteration_end
from CausalDiagnosis.agents.utils.tool_answers import json_object
from . import agent_options, episode_result


TEMPERATURE = 0.0
# the one bound of every model request and tool call of an episode, under the key and in the words the ports record it
DEADLINE_RULE = "the wall limit of the row (CausalDiagnosis.utils.model_request_retry.deadline_left)"
# the reason a reply that the endpoint cut at the token cap is not used, which the next planning turn quotes
CUT_REPLY = ("the reply was cut at the token cap before the JSON object was complete, so nothing of it was used. Return the object "
             "alone and shorter.")
UNJUDGED = "its expectation was not judged, since another read step of its batch stopped the plan"
# the reasons of the guards of a diagnosis and of a work order, each naming the change
GUARD_RANK = "rank {rank} is not among the candidates that the plan was made from; name a rank among the {count} candidates shown"
GUARD_COMMIT_FIRST = "put a commitment before this step, since none stands on the board"
SURROGATES = range(0xD800, 0xE000)  # the code points that no UTF-8 text holds, which a string can still hold one by one
REPLACEMENT_CHARACTER = chr(0xFFFD)  # U+FFFD REPLACEMENT CHARACTER, in place of a character that a text cannot hold
CUT = "RunStopped: cancelled"  # the error of the call that was in flight when the caller cut the run: given up with the run, not by the transport


def named_steps(first, kinds):
    """The steps from first on by number and kind, as step 4 (read) or as steps 4 (read), 5 (diagnose) and 6 (finish)."""
    names = [f"{n} ({kind})" for n, kind in enumerate(kinds, first)]
    return f"step {names[0]}" if len(names) == 1 else "steps " + ", ".join(names[:-1]) + f" and {names[-1]}"


def observation_order(evidence_id):
    """The order of evidence identifiers by their observation, the part between the first two slashes, then by the identifier: a
    number (obs/<n>/<field>), compared as a number, or a timestamp (obs/<timestamp>/<field>), compared as text."""
    part = evidence_id.split("/")[1] if evidence_id.count("/") >= 2 else ""
    return (0, int(part), evidence_id) if part.isdigit() else (1, part, evidence_id)


def request_settings(llm, temperature=None):
    """What a run folder records of the controller's requests on a backend: the blocks decoding and transport of the backend,
    at the controller's temperature unless a sampling run sends another, with how the controller reaches its tool servers and
    the deadline that bounds every request and call of an episode, so that its transport holds the keys of every agent's."""
    settings = llm.request_settings(TEMPERATURE if temperature is None else temperature)
    return {"decoding": dict(settings["decoding"]),
            "transport": {**settings["transport"], "sessions": SESSIONS, "tool_timeout_s": None, "deadline": DEADLINE_RULE}}


# the note of a finish on a clocked row that serves no wait and whose reads play no cycles, where no step of the agent reaches the close
NO_WAY_TO_THE_CLOSE = ("No wait is served and no read plays cycles on this row, so the controller finished before the last work order "
                       "closed and without its verification.")


class CausalLoopRunner:
    """The CausalDiagnosis agent over the action servers of its row and, when server_paths registers it, the diagnostics server."""

    def __init__(self, llm, server_paths, options=None, surface=None):
        """surface binds the agent's action kinds to the tools of the layer it runs on and reads their answers
        (tools/); the hydraulic rig's action server is the default. Without the diagnostics server among server_paths the
        agent runs on its own candidates and the normal ranges, and a switch that changes the server's advice has
        nothing to change, so it is refused."""
        self._llm = llm
        self._server_paths = server_paths  # {server name: launcher}, which every caller passes
        self.options = options or agent_options.CausalLoopOptions()
        self.surface = surface or RigSurface()
        self.server = CAUSAL_SERVER in self._server_paths
        self._cut = False  # whether the endpoint cut the latest reply at the token cap (_generate)
        if not self.server:
            defaults = agent_options.CausalLoopOptions()
            changed = [name for name in agent_options.SERVER_SWITCHES if getattr(self.options, name) != getattr(defaults, name)]
            if changed:
                raise ValueError(f"{', '.join(changed)} change what the CausalDiagnosis MCP advises, and the run registers no CausalDiagnosis MCP")

    async def run(self, question):
        self.question = question
        options = self.options
        self.board = Board(show_branches=options.branches, single_hypothesis=options.single_hypothesis, evidence_form=self.surface.evidence_form,
                           reads_take_time=self.surface.reads_take_time, noun=self.surface.noun, server=self.server, question=question,
                           read_advice=options.reads, backtracking=options.backtracking,
                           parallel=options.parallel, summary_call=self.surface.summary_is_call, recall_offered=options.recall)
        self.trajectory, self.plans, self.surprises = [], [], []
        self.model_calls = self.revisions = self.fallbacks = 0
        self.reader_faults = []  # the replies that only the last guard of the reader refused, a fault of the reader (_json_reply)
        self.replaced_characters = []  # the characters of a summary that UTF-8 cannot encode, replaced (_kept_summary)
        self.mode_of_first_plan = None
        self.answer = ""
        self._retry = None  # the reasons a turn without a usable plan hands to the next planning turn
        self._not_run = None  # the steps of a cut plan that did not run, with the reasons, which the next planning turn names
        self._stood = None  # the standing steps of the last plan as (number, kind, reason), which the next planning turn names
        self._schemas = {}  # the response format of each reply that a schema binds (_reply_schemas)
        self._ended_idle = False  # the environment ended the episode at the no_call step of a turn without a call
        self._oriented = False
        self._orient_turn = None  # the planning turn in which the differential was last asked for
        self._turn_index = 0  # the planning turns begun, counted by the loop
        self._orient_retry = None  # the parse error the next request for the differential quotes
        self._served = {}  # (plan, step) -> the cycles a read step served with their channel values, which the normal range verification judges
        self._trigger_reader()  # a backend that counts the context but cannot read the trigger fails here, before any call
        error = end = None
        try:
            async with open_sessions(self._server_paths) as sessions:
                self._sessions = sessions  # every tool call of the episode goes through this one block
                # the served list of the row's level, from the opening listing; the episode folder of the launchers names the row
                self.surface.listed(sessions.tools, where=str(next(iter(self._server_paths.values()), "")).rsplit("/", 1)[0] or None)
                self.board.summary_call, self.board.unserved = self.surface.summary_is_call, self.surface.unserved_kinds()
                self._schemas = self._reply_schemas()
                while not self.board.over and not self._ended_idle:
                    self._turn_index += 1
                    self.board.mark_turn(self._turn_index)
                    if self.board.profile is not None and "failure_modes" in self.board.opening_done and not self._oriented:
                        self._orient()
                    # the belief moves with the clock, so every planning turn with the server sees a fresh ranking, the first before any
                    # reading, which the board shows as advice
                    if self.server and self.board.profile is not None and not self.board.ranking_is_current():
                        await self._rank()
                    self.board.drop_unfit_recall(self._request_fits)  # after the orientation and the fresh ranking
                    record = self._plan()
                    self.board.recalled = []  # shown on this turn's request only
                    await self._execute(record)
                end = "finish" if self.board.end_reason == "finish" else "episode_over"
                if not self.answer:
                    self.answer = self._summarize(f"The episode ended by {self.board.end_reason or 'the environment'} before the controller could finish.")
        except asyncio.CancelledError:
            # the caller cut the run: what was recorded so far is the result, and the cancellation ends here, so the task takes it back
            asyncio.current_task().uncancel()
            end = "cancelled"
        except Exception as exc:  # noqa: BLE001 - a failed run is a result of the benchmark; what was recorded so far is kept
            error, end = f"{type(exc).__name__}: {exc}", "error"
        return episode_result.CausalLoopResult(
            question=question, answer=self.answer, plans=[asdict(p) for p in self.plans], trajectory=[asdict(t) for t in self.trajectory],
            model_calls=self.model_calls, surprises=list(self.surprises), revisions=self.revisions, fallbacks=self.fallbacks,
            mode_of_first_plan=self.mode_of_first_plan, differential=list(self.board.differential), board=self.board.to_json(),
            end_reason=self.board.end_reason, error=error, server=self.server, end=end, reader_faults=list(self.reader_faults),
            replaced_characters=list(self.replaced_characters))

    # ----- the model -----
    def _reply_schemas(self):
        """The schemas of the plan and of the differential of this row, built once after the listing from the kinds and the modes
        that the row serves and the options of the run, with their digests and whether the backend takes a response format on the
        board; the response format of each with the switch on, else none."""
        board = self.board
        kinds = reply_schema.served_kinds(board.unserved, self.surface.opening_kinds(board), self.options.recall)
        schemas = {reply_schema.PLAN: reply_schema.plan_schema_of(kinds, reply_schema.served_modes(board.unserved), self.options.plan_length),
                   reply_schema.DIFFERENTIAL: reply_schema.differential_schema()}
        board.reply_schema = {**{name: reply_schema.digest(schema) for name, schema in schemas.items()}, "bound": 0,
                              "carried": self._takes_schema()}
        if not self.options.structured_output:
            return {}
        return {name: reply_schema.response_format(name, schema) for name, schema in schemas.items()}

    def _takes_schema(self):
        """Whether the backend binds a reply to a JSON schema; one that does not is sent no field."""
        return bool(getattr(self._llm, "takes_response_format", False))

    def _generate(self, prompt, schema=None):
        """One model call, its reply bound to the response format schema where one is given, which the board counts; whether the
        endpoint cut the reply at the token cap is kept for the parser of the reply (_cut)."""
        self.model_calls += 1
        bound = {"response_format": schema} if schema is not None and self._takes_schema() else {}
        if bound:
            self.board.reply_schema["bound"] += 1
        result = self._llm.generate_with_usage(prompt, TEMPERATURE, **bound)
        self._cut = getattr(result, "finish_reason", None) == "length"
        return result.text or ""

    def _json_reply(self, text, parse, *args):
        """What parse reads from a reply through the boundary of the reader (read_reply), or a Refusal; a reply cut at the token cap
        is not read, whatever it holds. A refusal that only the last guard of the reader gave is kept in reader_faults with the class
        of its exception and the number of the model call, a fault of the reader that the records of the episode show."""
        if self._cut:
            return Refusal(reason=CUT_REPLY)
        read = read_reply(parse, text, *args)
        if isinstance(read, Refusal) and read.fault is not None:
            self.reader_faults.append({"model_call": self.model_calls, "reader": parse.__name__, "fault": read.fault})
        return read

    def _kept_summary(self, text, summary):
        """A summary of the model as the controller keeps it, the closing summary or the summary of the earlier turns, which pass no
        reader: every character that UTF-8 cannot encode, a lone surrogate that the client of an endpoint hands over when the
        endpoint's JSON escapes one, is replaced by U+FFFD, since the board, the requests, the finish call and answer.txt encode the
        summary. The count goes to replaced_characters with the number of the model call, while the record of the call keeps the
        endpoint's text."""
        count = sum(1 for character in text if ord(character) in SURROGATES)
        if not count:
            return text
        self.replaced_characters.append({"model_call": self.model_calls, "summary": summary, "characters": count})
        return "".join(REPLACEMENT_CHARACTER if ord(character) in SURROGATES else character for character in text)

    def _orient(self):
        """The orientation once the profile is on the board: the differential from the profile and the failure modes the opening read,
        parsed from the last JSON object of the reply. A reply that does not parse leaves the differential empty for now and counts
        as a fallback; the parse error goes quoted into the next request for the differential, in the next planning turn, the carried
        form of a refused plan. There is no count: a turn without a counted call takes its step through the idle hook. At
        most one request for the differential is made per planning turn. The rule only test tool skips it."""
        if self.options.rule_only:
            self._oriented = True
            return
        if self._orient_turn == self._turn_index:  # asked already in this planning turn
            return
        self._orient_turn = self._turn_index
        board, length = self.board, self.options.plan_length
        retry, self._orient_retry = self._orient_retry, None
        text = self._generate(orientation_turn(self.question, board.profile, board.failure_modes, length, retry=retry,
                                               surface=self.surface, server=self.server, reads=self.options.reads,
                                               parallel=self.options.parallel, recall=self.options.recall),
                              self._schemas.get(reply_schema.DIFFERENTIAL))
        read = self._json_reply(text, parse_differential)
        if isinstance(read, Refusal):
            self.fallbacks += 1
            self._orient_retry = f"The reply could not be parsed as a differential: {read.reason}"
            board.add_note(f"The orientation reply could not be parsed ({read.reason}); you are asked again for the differential.", tie="differential")
        else:
            board.set_differential(read)
            self._oriented = True

    def _summarize(self, note=None):
        text = self._kept_summary(self._generate(summary_turn(self.board, self.options.plan_length, note,
                                                             surface=self.surface)), "closing").strip()
        return text or (note or "The episode ended.")

    # ----- planning -----
    def _plan(self):
        after_surprise = bool(self.plans) and self.plans[-1].surprise is not None
        record = episode_result.PlanRecord(index=len(self.plans) + 1, source="", plan={}, candidates=self._snapshot(), after_surprise=after_surprise)
        scheduled = exploration(self.board, self.options.reads)  # the schedule the board shows at this turn, recorded with the plan
        record.schedule = None if scheduled is None else scheduled.to_json()
        if after_surprise:
            self.revisions += 1
        if self.options.rule_only:
            plan, source = self._rule_or_finish(record), "rule_only"
        else:
            plan, source = self._model_plan(record)
        if plan.candidates:
            # the candidates the plan lists become the board's, after the ranked ones with the server, and its diagnosis names them by rank
            self.board.set_candidates(plan.candidates)
            record.candidates = self._snapshot()
        self.board.plan_basis = self.board.observations
        record.plan, record.source = plan.to_json(), source
        self.plans.append(record)
        if self.mode_of_first_plan is None and any(s.kind not in OPENING_KINDS for s in plan.steps):
            self.mode_of_first_plan = plan.mode
        self.board.plan = {**record.plan, "steps": [dict(s, status="pending") for s in record.plan["steps"]]}
        return record

    def _model_plan(self, record):
        """One planning call. A plan runs up to its first step that the controller cannot use, a step that does not parse, a finish
        with steps after it or a step that the check refuses on its own: the steps before it run as the model wrote them, and that
        step and every step after it do not run and go to the next planning turn with the reasons (_cut_plan). A reply that does not
        parse as a plan, a plan whose first step cannot be used, and a plan with a reason on the whole of it, its candidates,
        branch or mode or the fit to the budget, the horizon and the step cap, run nothing: the turn makes no counted call, so it
        takes the no_call step of a turn without a call, and its reasons go to the next planning turn. The candidates of a plan made
        before the profile is read are set aside, since no profile names their levels yet, and a note of the board says so. There
        is no count of retries and no second request inside the turn."""
        board, length = self.board, self.options.plan_length
        retry, self._retry = self._retry, None
        not_run, self._not_run = self._not_run, None
        stood, self._stood = self._stood, None
        parsed = self._parse(self._generate(planning_turn(board, length, retry=retry, surface=self.surface, not_run=not_run, stood=stood),
                                            self._schemas.get(reply_schema.PLAN)), record)
        if parsed is None:
            return self._no_plan(record, "no_plan_after_parse_failure", f"could not be parsed: {record.parse_errors[-1]}",
                                 f"The reply could not be parsed as a plan: {record.parse_errors[-1]}")
        plan, unusable = parsed
        if board.profile is None and plan.candidates:  # no profile names their components and levels yet, so they are set aside
            plan = replace(plan, candidates=[])
            board.add_note(f"The candidates of your plan of turn {record.index} were not kept, since no profile was on the board; list "
                           "them again in your next plan.", tie="candidates", turn=record.index)
        refused = step_problems(plan, board)
        if refused:
            record.check_problems.append([reason for _step, reason in refused])
        if unusable is not None:
            record.parse_errors.extend(unusable.reasons)
        reasons = [reason for _step, reason in refused] + (unusable.reasons if unusable is not None else [])
        cut = min([step for step, _reason in refused if step is not None] + ([unusable.step] if unusable is not None else []), default=None)
        if not refused and cut == 1:  # the first step of the reply does not parse
            return self._no_plan(record, "no_plan_after_parse_failure", f"could not be parsed: {reasons[0]}",
                                 f"The reply could not be parsed as a plan: {reasons[0]}")
        if any(step is None for step, _reason in refused) or cut == 1:
            return self._no_plan(record, "no_plan_after_check_failure", "was refused: " + "; ".join(reasons),
                                 "The plan was refused for these reasons:\n- " + "\n- ".join(reasons))
        if cut is not None:
            return self._cut_plan(record, plan, cut, reasons, unusable)
        return plan, "model"

    def _cut_plan(self, record, plan, cut, reasons, unusable):
        """A plan cut at its first step that the controller cannot use: the steps before it run as the model wrote them, while that
        step and every step after it do not, and the record (PlanRecord.cut), the board's notes and the next planning turn name
        them with the reasons. A cut plan is a plan of the model and no fallback."""
        kinds = [step.kind for step in plan.steps[cut - 1:]] + (unusable.kinds if unusable is not None else [])
        steps = named_steps(cut, kinds)
        record.cut = {"step": cut, "steps": kinds, "reasons": list(reasons)}
        self._not_run = f"{steps[0].upper()}{steps[1:]} did not run, for these reasons:\n- " + "\n- ".join(reasons)
        # written before the steps run, so it says where the plan was cut and not how far it ran, which a surprise can shorten
        self.board.add_note(f"Your plan of turn {record.index} was cut before step {cut}, so {steps} did not run: " + "; ".join(reasons) + ".",
                            turn=record.index)
        return Plan(branch=plan.branch, mode=plan.mode, why=plan.why, steps=plan.steps[:cut - 1], raw=plan.raw, candidates=plan.candidates), "model_cut"

    def _no_plan(self, record, source, why, retry):
        """A turn without a usable plan: nothing runs, the reasons go to the board's notes and, quoted, to the next planning turn,
        and the turn counts as a fallback."""
        self.fallbacks += 1
        self._retry = retry
        self.board.add_note(f"Your plan of turn {record.index} {why}.", turn=record.index)
        return Plan(branch=None, mode=None, why=f"controller: no usable plan, the reply {why}", steps=[]), source

    def _parse(self, text, record):
        """(the plan of the reply up to its first step that the parser cannot use, that step or None) (parse_usable_plan), or None
        for a reply that does not parse as a plan, whose error the record keeps."""
        components = self.board.profile["components"] if self.board.profile is not None else ()
        read = self._json_reply(text, parse_usable_plan, self.options.plan_length, components)
        if isinstance(read, Refusal):
            record.parse_errors.append(read.reason)
            return None
        return read

    def _rule_or_finish(self, record):
        """The rule's plan for the turn, the internal test tool; when not even that fits what is left, a bare finish, whose guard
        decides what the remaining budget, cycles and steps still allow."""
        plan = rule_plan(self.board, self.options.plan_length, self.options.reads)
        problems = check_plan(plan, self.board)
        if not problems:
            return plan
        record.check_problems.append(problems)
        return Plan(branch=None, mode="act_now", why="rule: nothing else fits the budget, the horizon or the step cap, so finish", steps=[Step("finish")])

    def _snapshot(self):
        return [{"rank": c["rank"], "combination": dict(c["combination"]), "current_combination": dict(c.get("current_combination") or c["combination"]),
                 "probability": c.get("probability"), "faults": [f.get("code") for f in c.get("faults", [])], **({"own": True} if c.get("own") else {})}
                for c in self.board.candidates()]

    # ----- execution -----
    async def _execute(self, record):
        """The steps of a plan in order. A run of consecutive read steps is one batch (_read_batch), and one step with the switch
        off; a read step that stands when the run is formed is handled alone (_stand_read), so a batch is a run of steps that each
        made a call. The board is compacted after every other step and every batch, and the first surprise stops the plan. The
        standing steps are kept for the next planning turn."""
        steps = [Step(**s) for s in record.plan["steps"]]
        calls_before = sum(1 for t in self.trajectory if t.server in self.surface.action_servers)
        k = 1
        while k <= len(steps) and not self.board.over:
            step = steps[k - 1]
            if step.kind == "read" and standing(step, self.board, record=record) is not None:
                outcomes = [await self._read_step(step, record, k)]
            elif step.kind == "read":
                run = 1
                while (self.options.parallel and k + run <= len(steps) and steps[k + run - 1].kind == "read"
                       and standing(steps[k + run - 1], self.board, record=record) is None):
                    run += 1
                outcomes = await self._read_batch(steps[k - 1:k - 1 + run], record, k)
            else:
                outcomes = [await self._step(step, record, k)]
            for offset, outcome in enumerate(outcomes):
                record.steps.append(asdict(outcome))
                self.board.plan["steps"][k - 1 + offset]["status"] = outcome.status
            self._compact()
            at = next((offset for offset, outcome in enumerate(outcomes) if outcome.status == "surprise"), None)
            if at is not None:
                stopped = steps[k - 1 + at]
                surprise = {"plan": record.index, "step": k + at, "kind": stopped.kind, "reason": outcomes[at].note,
                            "expected": dict(stopped.expect), "observed": outcomes[at].observed}
                record.surprise = surprise
                self.surprises.append(surprise)
                self.board.surprises.append(surprise)
                break
            k += len(outcomes)
        self._stood = [(number, s["kind"], s["note"]) for number, s in enumerate(record.steps, 1) if s["status"] == "standing"] or None
        made_calls = sum(1 for t in self.trajectory if t.server in self.surface.action_servers) > calls_before
        # the driver's idle hook (CausalDiagnosis/agents/utils/idle_hook.py) at the end of every planning turn: with acted False it makes the no_call
        # step of a turn without a call; it answers whether the episode is over, and outside an episode
        # it does nothing
        self._ended_idle = bool(await aiteration_end(made_calls))

    async def _step(self, step, record, k):
        if step.kind not in OPENING_KINDS and self.board.profile is not None and not self._oriented:
            self._orient()  # a plan past the opening without the failure modes: the orientation runs on what the board holds
        handler = {"read": self._read_step, "diagnose": self._diagnose, "declare": self._declare, "order": self._order,
                   "wait": self._wait, "finish": self._finish, "recall": self._recall,
                   **{kind: self._opening_step for kind in OPENING_KINDS}}[step.kind]
        return await handler(step, record, k)

    def _fail(self, step, reason, observed=None):
        """A check that failed: a surprise that stops the plan, or with the expectations off a failed step the plan runs past."""
        status = "surprise" if self.options.expectations else "failed"
        return episode_result.StepRecord(kind=step.kind, args=dict(step.args), expect=dict(step.expect), status=status, observed=observed or {}, note=reason)

    def _stand(self, step, stands, observed=None):
        """A step whose effect already stands (plan.standing): no call, its clause under observed["stands"] and its reason as the note."""
        return episode_result.StepRecord(kind=step.kind, args=dict(step.args), expect=dict(step.expect), status="standing",
                                         observed={**(observed or {}), "stands": stands.clause}, note=stands.reason)

    async def _recall(self, step, record, k):
        """A recall step: the full rows of a group's collapsed observations come back on the next board. It calls no tool and
        costs nothing; a turn of recalls alone makes no counted call and takes the no_call step."""
        group, (first, last) = step.args["sensor_group"], step.args["observations"]
        rows = self.board.recall(group, first, last)
        if not rows:
            return self._fail(step, NOT_COLLAPSED.format(**KIND_WORDS, name=group, first=first, last=last))
        return self._done(step, {"rows": rows})

    def _done(self, step, observed=None, note=""):
        return episode_result.StepRecord(kind=step.kind, args=dict(step.args), expect=dict(step.expect), status="done", observed=observed or {}, note=note)

    def _planned_candidate(self, record, rank):
        if rank is None or rank < 1 or rank > len(record.candidates):
            return None
        return record.candidates[rank - 1]

    async def _opening_step(self, step, record, k):
        """A free read of the opening, which the model plans: the profile, the failure modes of every component or of
        the one named, the fix history, and the clock and the extent of the record that a surface needs before a reading. A read
        whose answer stands on the board makes no call (plan.standing); a failure_modes step for every component calls only for
        the components whose answer the board does not hold. Every answered request is noted on the board."""
        board, kind = self.board, step.kind
        stands = standing(step, board, record=record)
        if stands is not None:
            return self._stand(step, stands)
        if kind == "failure_modes":
            component = step.args.get("component")
            modes = []
            for request in ([{"component": component}] if component else self.surface.failure_mode_requests(board)):
                if board.holds_answer(kind, request):
                    continue
                env = await self._act("failure_modes", **request)
                if "refused" in env:
                    return self._fail(step, f"the failure modes read was refused: {env['refused']}")
                modes += (env.get("result") or {}).get("failure_modes", [])
                board.note_answered(kind, request)
            board.add_failure_modes(modes)
        else:
            env = await self._act(kind)
            if "result" not in env:
                return self._fail(step, f"the {kind} read was refused: {env.get('refused')}")
            board.note_answered(kind, {})
            if kind == "profile":
                board.set_profile(env["result"])
                board.required_opening = list(self.surface.opening_kinds(board))
                board.clock_runs = self.surface.can_observe_after(board, board.time)
            elif kind == "fix_history":
                board.set_fix_history(env["result"])
        if kind not in board.opening_done:
            board.opening_done.append(kind)
        return self._done(step)

    async def _read_step(self, step, record, k):
        """One read step alone: one that stands (_stand_read), else a batch of one."""
        stands = standing(step, self.board, record=record)
        if stands is not None:
            return await self._stand_read(step, record, stands)
        return (await self._read_batch([step], record, k))[-1]

    async def _read_batch(self, steps, record, k):
        """A run of consecutive read steps that the plan named, the k-th step of the plan first: the readings one after the
        other, each logged with what the ranking before the batch expected of its group, then one ranking with the server, the
        record of the batch on the board, the verification of every repair the readings can verify, and the expectations of the
        steps against that ranking: with the server the top candidate, its mass or the effective hypotheses, and with or without
        it the channels expected outside their normal ranges. A read that the environment refuses or that serves no cycle the board
        does not hold ends the batch, and its record comes last; a read that stands by the time it is reached, since an earlier
        reading of the batch took its last unread cycle, ends the batch before it and is handled alone. A refuted plan is the
        surprise of the last read judged, else the first expectation that fails is the surprise, and the reads made before or after
        it are done with their expectation not judged. Returns one step record per read handled; for a run of one read the calls
        and the checks are those of one read step."""
        board, made, ended = self.board, [], None
        advised = self._advised(record)
        for offset, step in enumerate(steps):
            if offset and standing(step, board, record=record) is not None:
                break
            observed, failure = await self._take_reading(step, record, k + offset, advised)
            if failure is not None:
                ended = failure
                break
            made.append((step, observed))
            if board.over or not self._served[(record.index, k + offset)]:
                break
        if not made:
            return [ended]
        rankings = 0
        if self.server:
            await self._rank()
            rankings = 1
        board.record_batch(record.index, k, len(made), rankings)
        top = board.ranked_top() if self.server else None  # an expectation on the top or its mass reads the ranking alone
        if self.server:
            for _step, observed in made:
                observed.update(top=None if top is None else combination_text(top["combination"]), top_mass=None if top is None else top["probability"])
        judged = made
        if not self._served[(record.index, k + len(made) - 1)]:  # the read was made, so the turn is not idle, and it showed no cycle the board does not hold
            step, observed = made[-1]
            ended = self._fail(step, core_texts.NOTHING_NEW_AFTER_CALL.format(**KIND_WORDS, last=board.status.get("recorded_until")),
                               {**observed, "served": 0})
            judged = made[:-1]
            if not judged:
                return [ended]
        verified, refuted = await self._verify_ready()
        surprise = self._refutation(judged[-1][0], verified, refuted, judged[-1][1])
        if surprise is not None:
            outcomes = [self._done(step, observed, note=UNJUDGED) for step, observed in judged[:-1]] + [surprise]
        else:
            outcomes = []
            for step, observed in judged:
                if any(o.status == "surprise" for o in outcomes):
                    outcomes.append(self._done(step, observed, note=UNJUDGED))
                else:
                    outcomes.append(self._judge_reading(step, record, observed, top))
        return outcomes + ([ended] if ended is not None else [])

    async def _stand_read(self, step, record, stands):
        """A read step that stands, alone: the ranking brought up to date where it is stale, as before a diagnosis, and its expectation
        judged on the board as it stands, which no call of the step could change; one that fails is a surprise whose reason names the
        change, and the step keeps its clause."""
        if self.server and not self.board.ranking_is_current():
            await self._rank()
        judged = self._judge_reading(step, record, {}, self.board.ranked_top() if self.server else None)
        if judged.status == "done":
            return self._stand(step, stands, judged.observed)
        return self._fail(step, core_texts.STANDS_EXPECTATION.format(stands=stands.reason, detail=judged.note),
                          {**judged.observed, "stands": stands.clause})

    def _advised(self, record):
        """The sensor groups the board advises when a batch starts: before a commitment with the server the reads it advises, and
        the reads of the schedule recorded with the plan."""
        board = self.board
        groups = {g for g, _c in batch_reads(board, board.read_advice)} if self.server and not board.committed else set()
        return groups | {read["sensor_group"] for read in (record.schedule or {}).get("reads") or []}

    async def _take_reading(self, step, record, k, advised):
        """One reading of a batch: the call, the rows on the board and the entry of the read log, with what the ranking before the
        batch expected of its group. Returns (observed, None) for a read made and (None, the record of the step) for one that the
        environment refused."""
        board = self.board
        group, cycles = step.args["sensor_group"], step.args["cycles"]
        expected = dict(board.elimination_of(group) or {}) if self.server else {}
        env = await self._read(group, cycles)
        if "refused" in env:
            return None, self._fail(step, f"read_sensors refused: {env['refused']}", {"refused": env["refused"]})
        served = env.get("served_cycles") or []
        self._served[(record.index, k)] = served
        bands = board.normal_ranges()
        board.record_read({"plan": record.index, "step": k, "sensor_group": group, "cycles": cycles, "advised": group in advised,
                           "effective_before": expected.get("effective_before"), "expected_after": expected.get("expected_effective_after"),
                           "expected_elimination": expected.get("expected_elimination"), "expected_information_gain": expected.get("information_gain"),
                           "ig_per_cost": expected.get("ig_per_cost"), "realized_after": None,
                           "t_first": min((c[1] for c in served), default=None), "t_last": max((c[1] for c in served), default=None),
                           "cycles_outside": [[n, t, len(channels_outside(bands, values)), sum(1 for name in values if name in bands)]
                                              for n, t, values in served]})
        return {"time": board.time}, None

    def _judge_reading(self, step, record, observed, top):
        """The expectations of one read step against the ranking after its batch, top its top candidate with the server."""
        board = self.board
        group = step.args["sensor_group"]
        if self.options.expectations and self.server and top is not None:
            # judged at the level the board predicts: the faulty set of the planned candidate against the
            # set on top after the reading, and the set's mass as the server sums it over every combination of the set
            want = self._planned_candidate(record, step.expect.get("top_rank"))
            top_set, top_mass = board.top_set()
            observed["top_set"], observed["top_set_mass"] = top_set, top_mass
            if want is not None:
                want_set = sorted(board.faulty_set(want.get("current_combination") or want["combination"]))
                if want_set != top_set:
                    return self._fail(step, f"after the reading the faulty set on top is ({', '.join(top_set) or 'none'}), not that of the planned rank "
                                            f"{step.expect['top_rank']} ({', '.join(want_set) or 'none'})", observed)
            floor = step.expect.get("top_mass_at_least")
            if floor is not None and top_mass < floor - self.options.expectation_tolerance:
                return self._fail(step, f"after the reading the mass of the set on top is {fmt(top_mass)}, below the expected {fmt(floor)}", observed)
            ceiling, effective = step.expect.get("effective_at_most"), board.effective_hypotheses()
            observed["effective_hypotheses"] = effective
            if ceiling is not None and effective is not None and effective > ceiling * (1 + self.options.expectation_tolerance):
                return self._fail(step, f"after the reading the belief holds {fmt(effective)} effective hypotheses, more than the {fmt(ceiling)} "
                                        "before it, so the reading eliminated nothing", observed)
        inside = self._expected_outside_but_inside(step.expect.get("outside"), group)
        if self.options.expectations and inside:
            return self._fail(step, f"after the reading {', '.join(inside)} lies inside its normal range, against the expectation that it lies outside",
                              observed)
        return self._done(step, observed)

    def _expected_outside_but_inside(self, names, group):
        """The channels a read step expected outside their normal ranges that its latest cycle served inside them; a name matches a field by
        itself or as its tag, PS1 for PS1_Pressure_bar_100Hz_mean, and a channel the group did not serve is not judged."""
        latest = self.board.latest_cycles.get(group) or {}
        outside = {name for name, _side in latest.get("outside") or []}
        missing = []
        for name in names or []:
            matched = [f for f in (latest.get("inside") or []) + sorted(outside) if f == name or f.startswith(f"{name}_")]
            if matched and not any(f in outside for f in matched):
                missing.append(name)
        return missing

    async def _diagnose(self, step, record, k):
        board = self.board
        if self.server and not board.ranking_is_current():  # the advice is refreshed on the current evidence before the write
            await self._rank()
        want = self._planned_candidate(record, step.args["rank"])
        if want is None:
            return self._fail(step, GUARD_RANK.format(rank=step.args["rank"], count=len(record.candidates)))
        observed = {}
        if self.server and not want.get("own"):
            # without an expectation the step expects the candidate on top, which the advice of the server names
            expected = self._planned_candidate(record, step.expect.get("top_rank", 1))
            if expected is None:
                return self._fail(step, GUARD_RANK.format(rank=step.expect.get("top_rank", 1), count=len(record.candidates)))
            top = board.ranked_top()
            observed = {"top": None if top is None else combination_text(top["combination"]), "top_mass": None if top is None else top["probability"]}
            if self.options.expectations and top is not None and top["combination"] != expected["combination"]:
                return self._fail(step, "the ranking's top is " + ("unknown" if top is None else f"({combination_text(top['combination'])})")
                                  + f", not the planned candidate ({combination_text(expected['combination'])})", observed)
            current = next((c for c in board.candidates() if c["combination"] == want["combination"]), None)
            extra = (current or {}).get("evidence_ids") or []
        else:
            # a candidate of the model's own, the only kind without the server, with the channel means outside their normal ranges as
            # the evidence behind it
            extra = rule_evidence(board, None)
        held = self._held_commitment(step, record, observed)
        if held is not None:
            return held
        # the diagnosis states the condition now, scored at the cycle of the write (version 7), so it submits the current combination
        env = await self._act("diagnose", levels=submitted_levels(board, want), evidence=self._evidence(step.evidence, extra))
        if "refused" in env:
            return self._fail(step, f"the diagnosis was refused: {env['refused']}", observed)
        board.diagnosis = env["result"]
        return self._done(step, observed)

    async def _declare(self, step, record, k):
        board = self.board
        # a declaration counts only when it cites a reading, and a top candidate with no fault bears on none
        if self.server:
            if not board.ranking_is_current():
                await self._rank()
            top = board.top()
            readings = [i for i in (board.ranking or {}).get("evidence_ids") or [] if i.startswith("obs/")]
            extra = [*((top or {}).get("evidence_ids") or []), *readings[-1:]]
        else:
            extra = [*rule_evidence(board, None), *self._last_reading()]
        held = self._held_commitment(step, record)
        if held is not None:
            return held
        evidence = self._evidence(step.evidence, extra)
        env = await self._act("declare", sensor_group=step.args["sensor_group"], cycles=step.args["cycles"], evidence=evidence)
        if "refused" in env:
            return self._fail(step, f"the declaration was refused: {env['refused']}")
        board.declaration = env["result"]
        return self._done(step)

    def _held_commitment(self, step, record, observed=None):
        """Right before the call of a commit step: the step stands where the board holds the commitment it would make, and where
        another commitment stands it makes no call and is a surprise, since the environment takes one; None where the call is made."""
        stands = standing(step, self.board, record=record)
        if stands is not None:
            return self._stand(step, stands, observed)
        if self.board.committed:
            return self._fail(step, second_commitment(standing_commitment(self.board)), observed)
        return None

    def _last_reading(self):
        """The identifier of a value of the latest reading on the board, which a declaration without the server cites: the newest
        evidence row, else the absorbed identifier of the newest observation (observation_order)."""
        latest = next((row["evidence_id"] for row in reversed(self.board.evidence)), None)
        return [latest] if latest else sorted(self.board.absorbed_ids, key=observation_order)[-1:]

    async def _order(self, step, record, k):
        board = self.board
        if not board.committed:  # guard: no order without a prior diagnosis or declaration
            return self._fail(step, GUARD_COMMIT_FIRST)
        if not self.server:
            # no prediction exists without the server: the orders of one plan form one repair plan, keyed by the plan, which the read
            # after its last close verifies on the normal ranges
            evidence = self._evidence(step.evidence, (board.diagnosis or {}).get("evidence_ids") or [])
            env = await self._act("order", component=step.args["component"], prediction_id="", evidence=evidence)
            if "refused" in env:
                return self._fail(step, f"the work order was refused: {env['refused']}")
            board.add_order(env["result"], f"plan {record.index}")
            return self._done(step, {"work_order": env["result"]["work_order"]["id"]})
        component, prediction_id = step.args["component"], step.args["prediction_id"]
        if prediction_id == ESTIMATE or prediction_id not in board.predictions:  # guard: no order without a prediction
            estimate = await self._diag("estimate", action=f"replace:{component}")
            if "error" in estimate:
                return self._fail(step, f"the estimate failed: {estimate['error']}")
            board.add_estimate(estimate)
            if prediction_id != ESTIMATE:
                board.add_note(f"The prediction {prediction_id} was unknown, so an estimate ({estimate['prediction_id']}) replaced it.")
            prediction_id = estimate["prediction_id"]
        prediction = board.predictions[prediction_id]
        observed = {"prediction_id": prediction_id, "recovery": prediction.get("recovery")}
        floor = step.expect.get("recovery_at_least")
        if (self.options.expectations and floor is not None and prediction.get("recovery") is not None
                and prediction["recovery"] < floor - self.options.expectation_tolerance):
            return self._fail(step, f"the prediction {prediction_id} gives a recovery of {fmt(prediction['recovery'])}, below the expected {fmt(floor)}", observed)
        evidence = self._evidence(step.evidence, [prediction_id] + ((board.diagnosis or {}).get("evidence_ids") or []))
        env = await self._act("order", component=component, prediction_id=prediction_id, evidence=evidence)
        if "refused" in env:
            return self._fail(step, f"the work order was refused: {env['refused']}", observed)
        board.add_order(env["result"], prediction_id)
        observed["work_order"] = env["result"]["work_order"]["id"]
        return self._done(step, observed)

    async def _wait(self, step, record, k):
        """The wait step. The controller waits event by event, wait_for_notification with the cycles of the step as the
        timeout, so that it learns of every close by its notification and never waits past an event it should react to.
        The change summary then lists the alarms, the closed plans are verified and a refuted plan or a new event is a
        surprise."""
        board = self.board
        since = board.time
        target = since + board.wait_cycles(step.args["cycles"])
        while not board.over and board.time < target:
            before = board.time
            env = await self._act("wait", timeout=target - board.time)
            if "refused" in env:
                return self._fail(step, f"the wait was refused: {env['refused']}")
            if board.time <= before:  # the clock did not move, so waiting again would loop
                break
        new_alarms = []
        if not board.over:
            summary = await self._act("changes", since=since)
            if "result" in summary:
                # the summary lists every channel outside its normal range since the cycle, the long known faults included; a new
                # alarm is a channel never seen outside its normal range that left it after the advance began
                known = {a["channel"] for a in board.alarms} | {r["field"] for r in board.evidence if r["inside_band"] is False}
                board.add_changes(summary["result"])
                new_alarms = [a for a in summary["result"].get("alarms", []) if a["channel"] not in known and a["first_t"] > since]
        verified, refuted = await self._verify_ready()
        observed = {"time": board.time, "new_alarms": [a["channel"] for a in new_alarms]}
        if self.options.expectations:
            if step.expect.get("no_new_alarm", True) and new_alarms:
                return self._fail(step, "a new alarm appeared: " + ", ".join(f"alarm/{a['first_t']}/{a['channel']}" for a in new_alarms),
                                  {**observed, "verified": {key: v.get("consistent") for key, v in verified}, "refuted": [r["components"] for r in refuted]})
        surprise = self._refutation(step, verified, refuted, observed)
        return surprise if surprise is not None else self._done(step, observed)

    def _refutation(self, step, verified, refuted, observed):
        """The surprise a verification raises, a refuted repair plan, or with backtracking off a verdict that found the reading after
        the repair inconsistent; None when every verdict was consistent or the expectations are off."""
        observed.update(verified={key: v.get("consistent") for key, v in verified}, refuted=[r["components"] for r in refuted])
        if not self.options.expectations:
            return None
        if refuted:
            return self._fail(step, "; ".join(f"the repair plan on {', '.join(r['components'])} was refuted: {r['reason']}" for r in refuted), observed)
        inconsistent = [key for key, v in verified if v.get("consistent") is False]
        if inconsistent:
            judge = "verify_outcome" if self.server else "the normal ranges"
            return self._fail(step, f"{judge} found the reading after the repair inconsistent with " + ", ".join(inconsistent), observed)
        return None

    async def _refute(self, verified):
        """Backtracking (version 7, as advice): a repair is refuted only when its verdict finds the reading after its last
        close inconsistent, verify_outcome's E17 at the declared alpha with the server and the normal ranges without it. A consistent
        verdict stands whatever the belief's top after that reading shows, since a channel mean just outside its normal range can put a near
        healthy level on top of a belief that the verification has already answered. The refuted component set is recorded and marked
        on the board, the server ranks again, and the advice names the cheapest other previewed plan."""
        if not self.options.backtracking:
            return []
        board, refuted = self.board, []
        for key, verdict in verified:
            if verdict.get("consistent") is not False or not str(verdict.get("status", "")).startswith("verified"):
                continue
            components = sorted({o["component"] for o in board.orders.values() if o["prediction_id"] == key})
            reason = ("the reading after the last close is inconsistent with the plan's prediction" if self.server else
                      f"the reading after the last close shows a channel {board.statistic} outside its normal range")
            board.record_refuted(components, reason)
            refuted.append({"components": components, "reason": reason})
        if refuted and self.server:
            await self._rank()
        return refuted

    async def _finish(self, step, record, k):
        """The finish step. The verdicts a reading allows are given first, and a refuted plan stops it. Its guard holds a finish the plan
        reached before the last order closed or before the reading that verifies it, while the budget, the horizon and the step cap
        leave room for the wait and the read: the step is a surprise that names what is missing and the model plans again, since the
        controller never waits or reads for it."""
        board = self.board
        note = None
        verified, refuted = await self._verify_ready()
        surprise = self._refutation(step, verified, refuted, {})
        if surprise is not None:
            return surprise
        missing = self._unverified_reason()
        if missing:
            return self._fail(step, missing)
        if board.open_orders() or (board.clock_runs and board.pending_verifications()):
            note = (NO_WAY_TO_THE_CLOSE if board.clock_runs and "wait" in board.unserved and not board.forward_read else
                    "The budget, the horizon or the step cap ended the episode before the last work order could be closed and verified; "
                    "the controller finished without that verification.")
        summary = self._summarize(note)
        env = await self._act("finish", summary=summary)
        if "refused" in env:
            return self._fail(step, f"the finish was refused: {env['refused']}")
        self.answer = summary
        return self._done(step, note=note or "")

    def _unverified_reason(self):
        """Why a finish comes too early: an order still open, or a closed repair without the reading that verifies it, while a wait and
        a read fit the budget, the horizon and the step cap; None otherwise, with a note when no cycle runs after a close on this row."""
        board = self.board
        pending, unverified = board.open_orders(), board.pending_verifications()
        if not pending and not unverified:
            return None
        if not board.clock_runs:
            for key, _closed_t in unverified:
                board.add_note(f"No cycle runs after the close of the orders citing {key} on this row, so it stays unverified.")
            return None
        tail = board.verifying_tail()  # the waits, the verifying read and finish, as the plan check counts them, when they fit
        if tail is None:
            return None
        last, first, group = board.time + tail.to_close, tail.first, board.verification_group(board.repaired())
        if pending and "wait" in board.unserved:
            return (f"finish before the close: the order on the {', '.join(o['component'] for o in pending)} closes at cycle {last}, so read "
                    f"{group} for {max(1, tail.cycles // board.forward_read)} cycles, whose window plays to the close and verifies the repair, then finish")
        if pending:
            return (f"finish before the close: the order on the {', '.join(o['component'] for o in pending)} closes at cycle {last}, so wait to the "
                    f"close and read {group} at or after cycle {first}, which verifies the repair, then finish")
        return f"finish before the verification: read {group} at or after cycle {first}, after the last close, which verifies the repair, then finish"

    # ----- the controller's own calls, those of the diagnostics server that refresh its advice -----
    async def _read(self, group, cycles):
        """A reading and its channel means against their normal ranges per cycle. The ranking follows the batch of read steps
        (_read_batch), so the board never shows evidence the advice has not seen when the model plans again."""
        env = await self._act("read", sensor_group=group, cycles=cycles)
        if "result" in env:
            env["served_cycles"] = self.board.add_reading(env["result"])
        return env

    async def _rank(self):
        data = await self._diag("rank", branch_detail=ranking_detail(self.board))  # compact, or the long form under an arm
        if "error" in data:
            self.board.add_note(f"the ranking failed: {data['error']}", tie="ranking")
            return None
        self.board.set_ranking(data)
        return data

    async def _verify_ready(self):
        """The verdict of every repair whose orders have all closed and for which a reading at or after the last close exists, which only a
        plan's read step takes: with the server verify_outcome on its prediction, after the ranking of the belief after
        the repair, and without it the normal ranges of the first reading after the close (board.band_verdict). Returns the verdicts and
        the refuted plans that backtracking recorded."""
        board, verified = self.board, []
        for key, closed_t in board.pending_verifications():
            if board.over:
                break
            if self._verifying_reading(key, closed_t) is None:
                continue  # no verifying reading after the close yet: the model plans it, and the plan check asks for it before finish
            if not self.server:
                if board.open_orders():
                    continue  # the normal ranges judge the whole asset, so the verdict waits until no other order of the agent is open
                verdict = self._band_verdict(key, closed_t)
                board.verifications[key] = verdict
                verified.append((key, verdict))
                continue
            if not board.ranking_is_current():  # the belief after the repair, which the refutation test reads
                await self._rank()
            verdict = await self._diag("verify", prediction_id=key)
            if "error" in verdict:
                board.notes.append(f"the verification of {key} failed: {verdict['error']}")
                board.verifications[key] = {"status": f"error: {verdict['error']}", "consistent": None, "recovery_predicted": None,
                                            "recovery_posterior": None, "recovery_status": "", "message": verdict["error"]}
                continue
            board.add_verification(verdict)
            verified.append((key, verdict))
        return verified, await self._refute(verified)

    def _verifying_reading(self, key, closed_t):
        """The first reading at or after the last close of the repair whose orders cite key, of a group that verifies the repaired
        components (Board.verifies), or None: the one reading that verifies a repair, with the server and without it."""
        board = self.board
        components = sorted({o["component"] for o in board.orders.values() if o["prediction_id"] == key})
        return next((e for e in board.reads if e.get("t_last") is not None and e["t_last"] >= closed_t and board.verifies(e["sensor_group"], components)),
                    None)

    def _band_verdict(self, key, closed_t):
        """The verdict of a repair plan without the CausalDiagnosis MCP: its verifying reading (_verifying_reading), the cycles of it
        served from the close on, judged by board.band_verdict on the normal ranges the profile serves."""
        entry = self._verifying_reading(key, closed_t)
        cycles = [(n, values) for n, t, values in self._served[(entry["plan"], entry["step"])] if t >= closed_t]
        bands = self.board.normal_ranges()
        outside = [[n, name, side] for n, values in cycles for name, side in channels_outside(bands, values)]
        where = f"the {entry['sensor_group']} reading after the close"
        message = (f"outside their normal ranges in {where}: " + ", ".join(f"{name} {side} at observation {n}" for n, name, side in outside)
                   if outside else f"every channel {self.board.statistic} of {where} inside its normal range")
        return {"status": "verified by the normal ranges", "consistent": band_verdict(bands, [values for _n, values in cycles]),
                "reading": [entry["plan"], entry["step"]], "outside": outside, "message": message}

    def _evidence(self, cited, extra):
        """The identifiers a write cites: the model's, kept only when the board holds them (an identifier the episode never
        issued would fail the verifier's provenance gate), plus the ones the controller adds."""
        known = self.board.known_ids()
        kept, dropped = [], []
        for item in list(cited) + list(extra):
            if item in known:
                if item not in kept:
                    kept.append(item)
            elif item in cited:
                dropped.append(item)
        if dropped:
            self.board.add_note(f"{len(dropped)} cited identifier(s) not on the board were dropped: {', '.join(dropped)}.")
        return kept

    def _compact(self):
        """Idle compaction: after an executed step, while no model call is pending, the board collapses what a ranking has absorbed,
        and more only while the next planning request is past the trigger, 90 percent of the served context, the last step one
        summary of the earlier turns; the compaction log records each change with the index of the last call."""
        self.board.compact(self._request_under_trigger, after_call=len(self.trajectory), summarize=self._summary_of_earlier_turns if self.options.summary_turn else None,
                           fits_context=self._request_fits)

    def _request_under_trigger(self, board):
        """Whether the planning request built on the board is under the compaction trigger: the backend's request_under_trigger,
        the prompt's tokens alone within 90 percent of the served context, as in Codex.
        A backend that counts the context (request_fits) and lacks it is refused loudly, since a fallback to the whole context
        would move the trigger to 100 percent without a word; one that can tell neither, such as a
        scripted planner, takes every request as under it."""
        under = self._trigger_reader()
        return True if under is None else bool(under(planning_turn(board, self.options.plan_length, surface=self.surface)))

    def _trigger_reader(self):
        """The backend's request_under_trigger, or None for a backend that counts no context. A backend with request_fits and
        without it is refused; run() asks once before the first model call, so such a run spends no call."""
        under = getattr(self._llm, "request_under_trigger", None)
        if under is None and getattr(self._llm, "request_fits", None) is not None:
            raise TypeError(f"{type(self._llm).__name__} has request_fits but no request_under_trigger, so the compaction "
                            "trigger at 90 percent of the context cannot be read")
        return under

    def _summary_of_earlier_turns(self, board, entries):
        """Step 3: one request for a summary of the earlier turns, (text, k) with k the oldest entries left out of it. k is the
        smallest that makes the request fit the context, found by a bisection since leaving more out never lengthens it;
        the summary of an earlier event stays in every request. None when the request does not fit with every entry left
        out, so the run ends as context where the next request is sent. A reply cut by length becomes the empty summary, and the
        reply never enters the reply term of request_fits, which reserves room for a plan reply."""
        fits = getattr(self._llm, "request_fits", None)

        def request(k):
            return earlier_turns_turn(board.without(entries[:k]) if k else board, self.options.plan_length, k, surface=self.surface)

        def fitting(k):
            return fits is None or bool(fits(request(k)))
        low, high = 0, len(entries)
        if not fitting(high):
            return None
        while low < high:
            middle = (low + high) // 2
            if fitting(middle):
                high = middle
            else:
                low = middle + 1
        longest = getattr(self._llm, "longest_reply", None)
        text = self._generate(request(low))
        if longest is not None:
            self._llm.longest_reply = longest
        return ("" if self._cut else self._kept_summary(text, "earlier turns")), low

    def _request_fits(self, board):
        """Whether the planning request built on the board fits the context of the pinned model service: the backend's
        request_fits (the prompt's tokens plus the longest reply of the episode so far within the context). A backend that cannot
        tell, such as a scripted planner, takes every request as fitting."""
        fits = getattr(self._llm, "request_fits", None)
        return True if fits is None else bool(fits(planning_turn(board, self.options.plan_length, surface=self.surface)))

    # ----- the tool servers -----
    async def _call(self, server, tool, args):
        """One tool call on the session of its server, recorded in the trajectory; returns the answer text and the error of a
        call without an answer, if any. An answer that the server marks as an error is an answer: the record keeps its text as the
        response and as the error, and the surface reads it."""
        started = time.perf_counter()
        try:
            text, is_error = await self._sessions.call(server, tool, args)
        except asyncio.CancelledError:
            self._record(server, tool, args, "", CUT, started)
            raise
        except ToolTransportError as exc:
            error = f"ToolTransportError: {exc}"
            self._record(server, tool, args, "", error, started)
            return "", error
        self._record(server, tool, args, text, text if is_error else None, started)
        return text, None

    def _record(self, server, tool, args, text, error, started):
        self.trajectory.append(episode_result.ToolRecord(step=len(self.trajectory) + 1, server=server, tool=tool, tool_args=dict(args), response=text, error=error,
                                          duration_ms=(time.perf_counter() - started) * 1000))

    async def _act(self, kind, **fields):
        """An action kind on the action servers, bound to its tool and read back by the surface, or answered by the surface
        itself for a kind without a tool on it; a transport failure reads as a refusal, and every envelope updates the
        board and records its notifications. A call of the environment opens, continues or ends the run of orders that form one
        visit (Board.note_step)."""
        binding = self.surface.bind(kind, self.board, **fields)
        if binding is None:
            data = self.surface.local(kind, self.board, **fields)
        else:
            server, tool, args = binding
            if kind != "changes":  # the change summary is the controller's own call, not a step of a plan
                self.board.note_call(kind, fields)
            text, error = await self._call(server, tool, args)
            data = {"refused": error} if error else self.surface.parse(kind, tool, text, self.board, **fields)
            self.board.note_step(kind, data)
        self.board.update_status(data)
        self.board.add_notifications(data.get("notifications"))
        if data.get("episode_over") and self.board.end_reason is None:
            self.board.end_reason = self._end_reason(data, kind)
        return data

    async def _diag(self, kind, **fields):
        """A diagnostics kind, bound to its tool by the surface; the answer is the server's JSON object or an error."""
        server, tool, args = self.surface.bind(kind, self.board, **fields)
        text, error = await self._call(server, tool, args)
        data = json_object(text)
        return data if data is not None else {"error": error or not_json(tool, text)}

    def _end_reason(self, data, kind):
        """The reason the end event of the environment carries, where a notification of the answer holds it, and otherwise the
        reason read from the answer and the board."""
        reason = next((n.get("reason") for n in data.get("notifications") or [] if isinstance(n, dict) and n.get("event") == "end" and n.get("reason")), None)
        if reason:
            return reason
        if data.get("refused") == "budget":
            return "budget"
        if kind == "finish" and "result" in data:
            return "finish"
        if self.board.steps_left <= 0:
            return "max_steps"
        if self.board.time >= self.board.horizon:
            return "horizon"
        return "ended"
