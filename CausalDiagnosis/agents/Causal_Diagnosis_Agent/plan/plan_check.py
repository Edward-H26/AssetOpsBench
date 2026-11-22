from CausalDiagnosis.agents.utils import core_texts
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import ReadWalk, profile_level, visit_closes
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS, commitment_text
from . import plan_schema
from .standing import Walked, commitment_of, planned_candidate, requests_of, standing, standing_commitment


# the reasons for a step or a mode that the row or the run does not offer, each naming the change first; a reply that a schema bound
# holds no such kind and no such mode, since the schema of a row leaves them out
UNSERVED_STEP = "leave this step out, since this row serves no {step_name} step"
UNSERVED_MODE = "choose a mode among {modes}, since this row serves no {step_name} step"
UNNEEDED_STEP = "leave this step out, since this row needs no {step_name} step"
NO_TIME_PASSES = "leave this step out, since no {time_unit} passes on this row"
NOT_OFFERED = "leave this step out, since this run offers no {step_name} step"
NOT_COLLAPSED = "leave this step out, since no row of {series} {name} in observations {first} to {last} was collapsed, so nothing would come back"


def _joined_names(names, last="and"):
    """Names joined by commas, the last two by last."""
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {last} {names[-1]}"


def second_commitment(stands):
    """The reason for a commit step that differs from the commitment that stands, (kind, value) as standing_commitment gives it."""
    return core_texts.SECOND_COMMITMENT.format(commitment=commitment_text(*stands))


def candidate_problems(candidates, profile):
    """The reasons to refuse the model's own candidates: a component the asset lacks or a level it does not have."""
    problems = []
    for i, candidate in enumerate(candidates, 1):
        for component, level in candidate["levels"].items():
            if component not in profile["components"]:
                problems.append(f"candidate {i}: unknown component {component!r}, and the components are {', '.join(profile['components'])}")
            elif profile_level(profile, component, level) is None:
                problems.append(f"candidate {i}: {component} has the levels {', '.join(map(str, profile['levels'][component]))}, not {level}")
    return problems


def _opening_problems(plan):
    """The reasons to refuse a plan made before the profile is read: the board holds nothing about the asset yet, so the plan holds
    only the reads of the opening and at most a finish after them, the profile before what needs its names (the failure modes of a
    component), and the clock and the extent of the record, which a surface needs or not as its profile says, only in a plan after
    it, and its own candidates, which name levels of the profile, not at all. Each reason names the change first, in the words that
    the board and the prompts state the rule in (plan_schema), and belongs to its step, as (step, reason), or to the plan as a
    whole, as (None, reason)."""
    problems, profile_read = [], False
    if plan.candidates:
        problems.append((None, f"leave candidates out of a plan made before the profile is read, and list them {plan_schema.NEXT_REQUEST}"))
    later = f"leave this step out and plan it {plan_schema.NEXT_REQUEST}"
    for i, step in enumerate(plan.steps, 1):
        name = f"step {i} ({step.kind})"
        if step.kind not in (*plan_schema.OPENING_KINDS, "finish"):
            problems.append((i, f"{name}: {later}, since {plan_schema.OPENING_RULE}"))
        elif step.kind in ("time", "extent"):
            problems.append((i, f"{name}: {later}, since the profile says whether this row needs it"))
        elif step.kind == "failure_modes" and not profile_read:
            problems.append((i, f"{name}: put the profile step before this step, since the profile names the components"))
        profile_read = profile_read or step.kind == "profile"
    return problems


def _tail_fits(board, tail, time, money, calls):
    """Whether the completion of a repair from a state of the walk at time, which spent money and calls, fits the budget, the
    step cap and the horizon."""
    return money + tail.price <= board.budget_left and calls + tail.calls <= board.steps_left and time + tail.cycles < board.horizon


def _verification_problem(name, board, tail, repaired, verified_by, fits):
    """The reason to refuse a finish that no verifying read at or after the last close precedes, or None. The controller
    verifies a repair only on a reading that a plan named of a group that verifies it (Board.verifies), so the plan buys one
    wherever a cycle runs after a close and the completion from the finish's state fits (fits); a row that records ahead of
    its reads shows a cycle at or after the close only from the next cycle on (tail.first)."""
    if not tail.reads or (verified_by is not None and verified_by >= tail.first) or not fits:
        return None
    return (f"{name}: finish comes before the reading that verifies the repair, so read {board.verification_group(repaired)} for 1 cycle "
            f"at or after cycle {tail.first}, after the last close, then finish")


def check_plan(plan, board):
    """The reasons the controller refuses a plan, or an empty list (step_problems)."""
    return [reason for _step, reason in step_problems(plan, board)]


def step_problems(plan, board):
    """The reasons the controller refuses a plan, each as (step, reason): the step that the reason belongs to, or None for a
    reason on the plan as a whole, its candidates, its branch and its mode and the fit to the budget, the horizon and the step
    cap, so that a plan runs up to its first refused step (controller._model_plan). The check keeps the rules that do not come
    from the diagnostics server: the plan schema and what the environment would refuse, the fit to the budget, the horizon and
    the step cap, the declaration that cites a reading and the read that verifies a repair before finish.
    The plan is walked on a copy of the clock. Before the profile is read it holds only the reads of the opening, the profile
    before the failure modes and no time or extent step, and at most a finish after them, and lists no candidates. A reading waits
    for the reads its surface needs first, a diagnosis names a candidate of the ranking or, without the server, of the model's
    own list, whose components and levels the profile holds, a diagnosis or a declaration precedes every order and a second one
    is refused, an order names a known component without an open order, finish comes after the close of every order and,
    wherever a cycle runs after a close and a read fits, after a read at or after the last close, and the prices, the cycles and
    the tool calls fit the budget, the horizon and the step cap, a reason of those limits naming the first step after which the
    walk passes it. Downtime is not simulated, since it depends on the hidden state; the environment refuses an action the budget
    cannot pay."""
    if board.profile is None:
        return _opening_problems(plan)
    problems, owners = [], []  # the reasons in their order and the step each belongs to, None for the plan as a whole

    def own(step):
        """The reasons given since the last call belong to step."""
        owners.extend([step] * (len(problems) - len(owners)))
    server = board.server
    profile = board.profile
    components = profile["components"]
    groups = profile["sensor_groups"]
    # the ranking's candidates followed by the plan's own, or by the board's kept list when the plan lists none
    problems += candidate_problems(plan.candidates, profile)
    candidates = board.candidates(board.as_candidates(plan.candidates) if plan.candidates and not problems else None)
    of = ("candidates of yours" if not server else "ranked candidates" if len(candidates) == len(board.ranked())
          else "ranked candidates and candidates of yours")
    read_before = any(i.startswith("obs/") for i in board.known_ids())
    opened = set(board.opening_done)
    time, money, calls = board.time, 0, 0
    walk = ReadWalk(board)  # the read steps of the plan one after the other, every paid cycle priced once
    run = board.run  # the visit of the open run of orders: consecutive orders form one visit and pay its charge once
    committed = board.committed
    visits = board.open_visits()  # visit -> its orders (component, created cycle, close of a repair begun), the plan's added

    def closes():
        """The closes the walk knows: every visit, the board's and the plan's, as the environment schedules it."""
        return [close for members in visits.values() for close in visit_closes(profile, members).values()]
    unverified = [closed_t for _key, closed_t in board.pending_verifications()]
    verified_by = None  # the cycle the plan's last verifying read starts at, which verifies every repair closed by then

    def tail_from(time):
        """The completion of the repair from the walk's state at time: the board's open visits and the plan's, the pending closes."""
        return board.completion(time=time, visits=visits, pending=unverified, repaired=board.repaired(planned), commit=False)
    # component -> the cycle its open or planned order closes at (None where the board knows no close), for a second order on it
    open_until = {o["component"]: o.get("expected_close_t") for o in board.open_orders()}
    planned = []
    totals = []  # the prices, the cycle and the tool calls of the walk before each step, then after the last
    if plan.branch is not None and candidates and plan.branch > len(candidates):
        problems.append(f"branch {plan.branch} is not among the {len(candidates)} {of}")
    if plan.mode in board.unserved:
        served = [mode for mode in plan_schema.MODES if mode not in board.unserved]
        problems.append(f"mode {plan.mode}: {UNSERVED_MODE.format(modes=_joined_names(served), step_name=plan.mode)}")
    own(None)
    # what the walk has done before each step, so that a step stands in the walk exactly when it will stand when it is reached
    walked = Walked(candidates=candidates)
    for i, step in enumerate(plan.steps, 1):
        own(i - 1)  # a step may end early (continue), so its reasons are claimed when the next one begins
        totals.append((money, time, calls))
        name = f"step {i} ({step.kind})"
        given = len(problems)  # a step with a reason on its own form never stands
        joined, run = run, None  # every step but an order ends the run of orders
        if step.kind in board.unserved:
            problems.append(f"{name}: {UNSERVED_STEP.format(step_name=step.kind)}")
        elif step.kind in plan_schema.OPENING_KINDS:
            if step.kind in plan_schema.SURFACE_KINDS and step.kind not in board.required_opening:
                problems.append(f"{name}: {UNNEEDED_STEP.format(step_name=step.kind)}")
            component = step.args.get("component")
            if component and component not in components:
                problems.append(f"{name}: unknown component {component!r}, and the components are {', '.join(components)}")
            opened.add(step.kind)
            requests = requests_of(step, board)
            if len(problems) > given or standing(step, board, walked=walked) is None:
                # a read whose answer holds for the episode calls only for the requests that nothing answered yet
                fresh = [r for r in requests if not (board.holds_answer(step.kind, r) or [step.kind, r] in walked.answered)]
                calls += len(fresh) if step.kind in plan_schema.EPISODE_KINDS else len(requests)
            walked.answered += [[step.kind, r] for r in requests]
        elif step.kind == "read":
            group, cycles = step.args["sensor_group"], step.args["cycles"]
            if group not in groups:
                problems.append(f"{name}: unknown sensor group {group!r}; the groups are {', '.join(groups)}")
                continue
            missing = [kind for kind in board.required_opening if kind not in opened]
            if missing:
                problems.append(f"{name}: a reading on this row needs the steps {' and '.join(missing)} first, which give its window of the record")
            rank = step.expect.get("top_rank")
            if server and rank is not None and candidates and rank > len(candidates):
                problems.append(f"{name}: expect.top_rank {rank} is not among the {len(candidates)} {of}")
            verifying = board.verifies(group, board.repaired(planned))
            # where no wait is served, a verifying read while a planned or open order has not closed in the record reads forward
            forward = bool(board.forward_read and verifying and (planned or board.open_orders() or any(c > time - 1 for c in closes() + unverified)))
            price, duration = walk.take(group, cycles, forward)
            walked.new, walked.time = walk.new, time
            stands = None if len(problems) > given else standing(step, board, walked=walked)
            walked.new = walked.time = None
            if stands is None:  # a standing read is no call, verifies no repair and gives a declaration nothing to cite
                if verifying:  # a forward read verifies at its end, where its window has played the cycles to the close
                    verified_by = time + duration if board.forward_read else time
                calls += 1
                read_before = True
            money += price
            time += duration
        elif step.kind == "diagnose":
            rank = step.args["rank"]
            if not candidates and server:
                problems.append(f"{name}: no ranking exists yet; read first, or list candidates of your own under candidates")
            elif not candidates:
                problems.append(f"{name}: no candidate exists yet, so list your candidates under candidates before the diagnosis")
            elif rank > len(candidates):
                problems.append(f"{name}: rank {rank} is not among the {len(candidates)} {of}")
            if len(problems) > given or standing(step, board, walked=walked) is None:
                _commit(step, board, walked, committed, planned_candidate(step, candidates), problems, name)
                committed = True
                calls += 1
        elif step.kind == "declare":
            group = step.args["sensor_group"]
            if group not in groups:
                problems.append(f"{name}: unknown sensor group {group!r}; the groups are {', '.join(groups)}")
            if len(problems) > given or standing(step, board, walked=walked) is None:
                if not read_before:
                    problems.append(f"{name}: a declaration counts only when it cites a reading, and none exists yet; read first")
                _commit(step, board, walked, committed, None, problems, name)
                committed = True
                calls += 1
        elif step.kind == "order":
            component = step.args["component"]
            if component not in components:
                problems.append(f"{name}: unknown component {component!r}, and the components are {', '.join(components)}")
                continue
            if not committed:
                problems.append(f"{name}: a diagnosis or a declaration must precede every work order; none exists and no earlier step makes one")
            close = open_until.get(component, -1)
            if close is None or close > time:
                until = "" if close is None else core_texts.OPEN_REPAIR_UNTIL.format(**KIND_WORDS, close=close)
                problems.append(f"{name}: {core_texts.OPEN_REPAIR.format(**KIND_WORDS, name=component)}{until}")
            planned.append(component)
            money += board.order_price(component)  # the reading that verifies the repair is a read step of its own
            if joined is None:
                money += board.visit_charge
                joined = ("plan", i)
            run = joined
            visits.setdefault(run, []).append((component, time, None))
            open_until.update(visit_closes(profile, visits[run]))
            calls += 1
        elif step.kind == "wait":
            if not board.clock_runs:
                problems.append(f"{name}: {NO_TIME_PASSES.format(**KIND_WORDS)}")
                continue
            start = time
            time += board.wait_cycles(step.args["cycles"])
            calls += board.wait_calls(start, time, visits)
        elif step.kind == "recall":  # free: no call, no price, no cycle
            group, (first, last) = step.args["sensor_group"], step.args["observations"]
            if not board.recall_offered:
                problems.append(f"{name}: {NOT_OFFERED.format(step_name=step.kind)}")
            elif not board.recallable(group, first, last):
                problems.append(f"{name}: {NOT_COLLAPSED.format(**KIND_WORDS, name=group, first=first, last=last)}")
        elif step.kind in board.state_writes:  # a write of a public state, which no surface of this agent declares (standing)
            if len(problems) > given or standing(step, board, walked=walked) is None:
                calls += 1
                state, value = board.state_writes[step.kind]
                walked.states[state] = value
        elif step.kind == "finish":
            known = closes()
            late = [c for c in known if c > time]
            tail = tail_from(time)
            fits = _tail_fits(board, tail, time, money, calls)
            # where the budget, the horizon or the step cap leaves no room for the wait and the read, or no wait and no read reaches the
            # close, finish may come first
            if late and fits and (tail.reads or "wait" not in board.unserved):
                way = ("wait to the close" if "wait" not in board.unserved
                       else f"read {board.verification_group(board.repaired(planned))} over the cycles to the close")
                problems.append(f"{name}: finish precedes the close of a work order (the last closes at cycle {max(late)}, the plan reaches cycle "
                                f"{time}); {way} so that the controller can verify it first")
            elif not late and (known or unverified):
                problem = _verification_problem(name, board, tail, board.repaired(planned), verified_by, fits)
                if problem:
                    problems.append(problem)
            calls += 1
    own(len(plan.steps))
    totals.append((money, time, calls))
    return list(zip(owners, problems)) + [(None, reason) for reason in _limit_problems(plan, board, totals)]


def _commit(step, board, walked, committed, want, problems, name):
    """A commit step of the walk that does not stand: where a commitment stands it differs, so it is refused as the environment would
    refuse it, else it is the plan's commitment, which a later commit step of the plan is held against."""
    if committed:
        stands = standing_commitment(board, walked)
        if stands is not None:  # an earlier commit step whose rank names no candidate states no commitment, and it cuts the plan already
            problems.append(f"{name}: {second_commitment(stands)}")
    else:
        walked.commitment = commitment_of(step, board, want)


def _limit_problems(plan, board, totals):
    """The reasons of the budget left, the horizon and the steps left, which the plan as a whole passes or not: the walk only
    grows, so it passes a limit exactly when it has passed it after its last step, and the reason names the first step after which
    it has, totals[i] holding the prices, the cycle and the tool calls after step i. A plan without a step passes no limit by a step
    of its own, so it gets no such reason, even on a board that has reached a limit."""
    if not plan.steps:
        return []

    def first_past(index, passes):
        """The name of the first step after which the walk's total at index, the prices, the cycle or the tool calls, passes a limit."""
        at = next(i for i in range(1, len(totals)) if passes(totals[i][index]))
        return f"step {at} ({plan.steps[at - 1].kind})"
    money, time, calls = totals[-1]
    problems, horizon = [], board.horizon
    if money > board.budget_left:
        problems.append(f"{first_past(0, lambda spent: spent > board.budget_left)}: end the plan before this step or make the steps up to it "
                        f"cheaper, since the plan's prices ({money:g}) exceed the budget left ({board.budget_left:g}) and pass it at this step")
    if time >= horizon:
        problems.append(f"{first_past(1, lambda cycle: cycle >= horizon)}: end the plan before this step or shorten the waits and reads up to it, "
                        f"since the plan reaches cycle {time} but the episode ends at the horizon of {horizon} cycles before finish can run, and "
                        "this step reaches it")
    if calls > board.steps_left:
        problems.append(f"{first_past(2, lambda used: used > board.steps_left)}: end the plan before this step, since the plan needs {calls} tool "
                        f"calls on the action server but {board.steps_left} steps are left, and this step passes them")
    return problems
