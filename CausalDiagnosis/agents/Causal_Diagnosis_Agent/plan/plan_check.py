from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import ReadWalk, profile_level, visit_closes
from . import plan_schema


UNSERVED_NOUNS = {"declare": "declaration", "wait": "wait", "time": "time step"}  # a kind the row does not serve, as the plan check names it


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
    """The reasons to refuse a plan made before the profile is read: the board holds nothing about the asset yet, so the plan reads
    the opening, the profile before what needs its names (the failure modes of a component), and the clock and the extent of
    the record, which a surface needs or not as its profile says, only in a plan after it."""
    problems, profile_read = [], False
    for i, step in enumerate(plan.steps, 1):
        name = f"step {i} ({step.kind})"
        if step.kind not in (*plan_schema.OPENING_KINDS, "finish"):
            problems.append(f"{name}: the board holds no profile yet, so the plan reads the opening first: profile, then failure_modes and fix_history")
        elif step.kind in ("time", "extent") or (step.kind == "failure_modes" and not profile_read):
            problems.append(f"{name}: read the profile first, which names the components and the record")
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
    """The reasons the controller refuses a plan, or an empty list. The check keeps the rules that do not come
    from the diagnostics server: the plan schema and what the environment would refuse, the fit to the budget, the horizon and
    the step cap, the declaration that cites a reading and the read that verifies a repair before finish.
    The plan is walked on a copy of the clock. Before the profile is read it reads the opening. A reading waits
    for the reads its surface needs first, a diagnosis names a candidate of the ranking or, without the server, of the model's
    own list, whose components and levels the profile holds, a diagnosis or a declaration precedes every order and a second one
    is refused, an order names a known component without an open order, finish comes after the close of every order and,
    wherever a cycle runs after a close and a read fits, after a read at or after the last close, and the prices, the cycles and
    the tool calls fit the budget, the horizon and the step cap. Downtime is not simulated, since it depends on the hidden
    state; the environment refuses an action the budget cannot pay."""
    if board.profile is None:
        return _opening_problems(plan)
    problems = []
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
    if plan.branch is not None and candidates and plan.branch > len(candidates):
        problems.append(f"branch {plan.branch} is not among the {len(candidates)} {of}")
    if plan.mode in board.unserved:
        problems.append(f"mode {plan.mode}: this row serves no {UNSERVED_NOUNS[plan.mode]}")
    for i, step in enumerate(plan.steps, 1):
        name = f"step {i} ({step.kind})"
        joined, run = run, None  # every step but an order ends the run of orders
        if step.kind in board.unserved:
            problems.append(f"{name}: this row serves no {UNSERVED_NOUNS[step.kind]}")
        elif step.kind in plan_schema.OPENING_KINDS:
            if step.kind in ("time", "extent") and step.kind not in board.required_opening:
                problems.append(f"{name}: this row needs no {step.kind} step, its readings take their window without it")
            component = step.args.get("component")
            if component and component not in components:
                problems.append(f"{name}: unknown component {component!r}, and the components are {', '.join(components)}")
            opened.add(step.kind)
            calls += len(components) if step.kind == "failure_modes" and not component else 1
        elif step.kind == "read":
            group, cycles = step.args["sensor_group"], step.args["cycles"]
            if group not in groups:
                problems.append(f"{name}: unknown sensor group {group!r}; the groups are {', '.join(groups)}")
                continue
            missing = [kind for kind in board.required_opening if kind not in opened]
            if missing:
                problems.append(f"{name}: a reading on this row needs the steps {' and '.join(missing)} first, which give its window of the record")
            verifying = board.verifies(group, board.repaired(planned))
            # where no wait is served, a verifying read while a planned or open order has not closed in the record reads forward
            forward = bool(board.forward_read and verifying and (planned or board.open_orders() or any(c > time - 1 for c in closes() + unverified)))
            price, duration = walk.take(group, cycles, forward)
            if verifying:  # a forward read verifies at its end, where its window has played the cycles to the close
                verified_by = time + duration if board.forward_read else time
            money += price
            time += duration
            calls += 1
            read_before = True
            rank = step.expect.get("top_rank")
            if server and rank is not None and candidates and rank > len(candidates):
                problems.append(f"{name}: expect.top_rank {rank} is not among the {len(candidates)} {of}")
        elif step.kind == "diagnose":
            rank = step.args["rank"]
            if not candidates and server:
                problems.append(f"{name}: no ranking exists yet; read first, or list candidates of your own under candidates")
            elif not candidates:
                problems.append(f"{name}: no candidate exists yet, so list your candidates under candidates before the diagnosis")
            elif rank > len(candidates):
                problems.append(f"{name}: rank {rank} is not among the {len(candidates)} {of}")
            if committed:
                problems.append(f"{name}: a diagnosis or a declaration was already made; the environment refuses a second one")
            committed = True
            calls += 1
        elif step.kind == "declare":
            group = step.args["sensor_group"]
            if group not in groups:
                problems.append(f"{name}: unknown sensor group {group!r}; the groups are {', '.join(groups)}")
            if not read_before:
                problems.append(f"{name}: a declaration counts only when it cites a reading, and none exists yet; read first")
            if committed:
                problems.append(f"{name}: a diagnosis or a declaration was already made; the environment refuses a second one")
            committed = True
            calls += 1
        elif step.kind == "order":
            component = step.args["component"]
            if component not in components:
                problems.append(f"{name}: unknown component {component!r}")
                continue
            if not committed:
                problems.append(f"{name}: a diagnosis or a declaration must precede every work order; none exists and no earlier step makes one")
            close = open_until.get(component, -1)
            if close is None or close > time:
                problems.append(f"{name}: a work order on the {component} is still open at this step; the environment refuses an order on a "
                                "component with an open order")
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
                problems.append(f"{name}: no clock runs on this row, so no wait is possible")
                continue
            start = time
            time += board.wait_cycles(step.args["cycles"])
            calls += board.wait_calls(start, time, visits)
        elif step.kind == "recall":  # free: no call, no price, no cycle
            group, (first, last) = step.args["sensor_group"], step.args["observations"]
            if not board.recall_offered:
                problems.append(f"{name}: this run offers no recall step")
            elif not board.recallable(group, first, last):
                problems.append(f"{name}: no row of {group!r} in observations {first} to {last} was collapsed, so nothing comes back; "
                                "the collapsed rows stand as summary rows under the evidence")
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
    if money > board.budget_left:
        problems.append(f"the plan's prices ({money:g}) exceed the budget left ({board.budget_left:g})")
    horizon = board.horizon
    if time >= horizon:
        problems.append(f"the plan reaches cycle {time} but the episode ends at the horizon of {horizon} cycles before finish can run")
    if calls > board.steps_left:
        problems.append(f"the plan needs {calls} tool calls on the action server but {board.steps_left} steps are left")
    return problems
