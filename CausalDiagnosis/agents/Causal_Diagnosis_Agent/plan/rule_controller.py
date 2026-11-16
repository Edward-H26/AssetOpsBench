from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import branch_prediction_id
from . import plan_schema, repair_budget, read_advice


WHY_SCHEDULE = "rule: the schedule advises these reads as one batch"


def read_step(board, group, cycles, evidence=None):
    """A read step of the controller's own plans with the expectation the batch rule checks: the effective number of
    hypotheses after the reading at most the number before it, when the server reports it, else the top candidate kept on
    top at any mass, which only a refusal can fail."""
    entry = board.elimination_of(group)
    if entry and entry.get("effective_before") is not None:
        expect = {"effective_at_most": float(entry["effective_before"])}
    else:
        expect = {"top_rank": 1, "top_mass_at_least": 0.0}
    return plan_schema.Step("read", {"sensor_group": group, "cycles": cycles}, expect, list(evidence or []))


def rule_evidence(board, candidate):
    """What the rule cites: the channel means seen outside their normal range and the released values the ranking names for the
    candidate, every one of them an identifier the board holds."""
    outside = [r["evidence_id"] for r in board.evidence if r["inside_band"] is False]
    bearing = [e for e in (candidate or {}).get("evidence_ids") or [] if e not in outside]
    return outside + bearing


def _current_faulty(board, candidate):
    """The components a plan on the candidate repairs: its faulty set now, which a closed order may have shrunk."""
    return board.faulty_set(candidate.get("current_combination") or candidate["combination"])


WHY_READS = {"batch": "rule: the ranking says read first, so read the batch whose choice no outcome can change",
             "ec2": "rule: the ranking says read first, so read the group whose reading cuts the most expected weight between repair plan classes per unit cost (one step edge cutting)",
             "separating": "rule: the ranking says read first and the separating test is affordable"}


def rule_plan(board, plan_length, reads="gain"):
    """The rule's own choice for the turn. While nothing is committed and
    the ranking says read first: read the affordable prefix of the read batch (or the separating test),
    one step per read, and when no read is affordable declare insufficient evidence with the separating test and repair
    the cheapest previewed branch. Otherwise diagnose the candidate whose branch has the cheapest act now cost (rank 1
    without previews), order its current faulty set citing the branch's prediction
    (else an estimate), advance to the last close and finish. After a commitment, order the cheapest other plan when the
    last one was refuted, else what is still faulty and unordered, then advance and finish. Trimmed to plan_length steps when a
    count is set;
    the controller plans again when the steps run out."""
    opening = opening_plan(board)
    if opening is not None:
        return opening
    scheduled = read_advice.exploration(board, reads)
    if scheduled is not None and scheduled.differs and scheduled.reads:
        evidence = rule_evidence(board, board.top())
        steps = [read_step(board, read.group, read.cycles, evidence) for read in scheduled.reads]
        return plan_schema.Plan(branch=1, mode="test_first", why=WHY_SCHEDULE, steps=steps[:plan_length])
    steps = []
    top = board.top()
    if not board.committed:
        if board.ranking is None:
            steps.append(plan_schema.Step("read", {"sensor_group": board.cheapest_group(), "cycles": 1}, {}, []))
            return plan_schema.Plan(branch=None, mode="test_first", why="rule: no ranking yet, read the cheapest group", steps=steps[:plan_length])
        advised = board.server_advice() or {}
        follow_read, follow_repair = read_advice.follows_advice(board, "read"), read_advice.follows_advice(board, "repair")
        if follow_read and advised.get("action") == "read":
            steps = [read_step(board, advised["sensor_group"], int(advised.get("cycles") or 1), rule_evidence(board, top))]
            return plan_schema.Plan(branch=1, mode="test_first", why="rule: the server advises this read before a diagnosis", steps=steps[:plan_length])
        source, _ = read_advice.read_candidates(board, reads)
        planned = [] if follow_read else read_advice.batch_reads(board, reads)
        if planned:
            steps = [read_step(board, group, cycles, rule_evidence(board, top)) for group, cycles in planned]
            return plan_schema.Plan(branch=1, mode="test_first", why=WHY_READS[source], steps=steps[:plan_length])
        test = board.separating_test()
        options = repair_budget.act_options(board)
        declares = "declare" not in board.unserved
        if follow_repair and declares and advised.get("action") == "declare":
            steps.append(plan_schema.Step("declare", {"sensor_group": advised["sensor_group"], "cycles": int(advised.get("cycles") or 1)}, {}, rule_evidence(board, top)))
            rank, candidate = 1, top
            faulty = _current_faulty(board, candidate)
            mode, why = "declare", "rule: the server advises a declaration, the top mass below its fixed cut with no reading left"
        elif follow_repair:
            rank = int(advised.get("rank") or 1)
            candidate = board.candidate(rank) or top
            steps.append(plan_schema.Step("diagnose", {"rank": rank}, {"top_rank": 1}, rule_evidence(board, candidate)))
            faulty = _current_faulty(board, candidate)
            mode, why = "act_now", f"rule: the server advises the diagnosis of rank {rank}"
        elif declares and board.read_first() and test:
            # no separating read is affordable, so the rule abstains and repairs the cheapest plan after the declaration
            steps.append(plan_schema.Step("declare", {"sensor_group": test["sensor_group"], "cycles": int(test.get("cycles") or 1)}, {}, rule_evidence(board, top)))
            rank = options[0][1] if options else 1
            candidate = board.candidate(rank) or top
            faulty = _current_faulty(board, candidate)
            mode, why = "declare", f"rule: the ranking says read first and no separating read is affordable, so declare and repair rank {rank}"
        else:
            rank = options[0][1] if options else 1
            candidate = board.candidate(rank) or top
            # expect top_rank 1 whatever the rank acted on, since no reading comes between the plan and the diagnosis
            steps.append(plan_schema.Step("diagnose", {"rank": rank}, {"top_rank": 1}, rule_evidence(board, candidate)))
            faulty = _current_faulty(board, candidate)
            mode = "act_now"
            why = (f"rule: the ranking does not say read first or no test is affordable, so act on rank {rank}, the branch with the cheapest act now cost"
                   if options else "rule: the ranking does not say read first or no test is affordable, so act on the top candidate")
    else:
        options = repair_budget.repair_options(board) if board.refuted_last else []
        if options:
            # backtracking: the last plan was refuted, so the cheapest other previewed plan is repaired next
            rank = options[0][1]
            faulty = _current_faulty(board, board.candidate(rank) or top)
            mode, why = "act_now", f"rule: the last repair plan was refuted, so repair rank {rank}, the cheapest other plan"
        else:
            levels = board.diagnosis["diagnosis"] if board.diagnosis is not None else (top["combination"] if top else board.healthy_set())
            faulty = board.faulty_set(levels)
            # the branch of the diagnosed candidate, which the cost choice may have taken below rank 1
            rank = next((c["rank"] for c in board.candidates() if c["combination"] == levels), 1)
            mode, why = "act_now", "rule: order what is still faulty, wait for the close and finish"
    branch = board.branch(rank) or {}
    # backtracking repairs what the belief still holds faulty, which a closed order may cover again; the fallback on the
    # diagnosis never reorders a component that had an order
    ordered = {o["component"] for o in board.orders.values() if o["status"] != "closed" or not board.refuted_last}
    chosen = []  # the orders whose completion, with the reading that verifies them, the budget pays
    preceded = bool(steps)  # a step before the orders ends the board's run of orders, so they open a visit of their own
    for component in faulty:
        if component in ordered or board.completion(chosen + [component], commit=False, after_step=preceded).price > board.budget_left:
            continue
        chosen.append(component)
        prediction = branch_prediction_id(branch, component) or plan_schema.ESTIMATE
        steps.append(plan_schema.Step("order", {"component": component, "prediction_id": prediction}, {"prediction_id": prediction, "recovery_at_least": 0.0},
                          [prediction] if prediction != plan_schema.ESTIMATE else []))
    # the controller verifies a repair only on a reading that a plan names, so the rule reads after the last close of
    # its orders and of every closed repair still waiting for its reading
    tail = board.completion(chosen, commit=False, after_step=preceded)
    if tail.first is not None and tail.first > board.time and "wait" not in board.unserved:
        steps.append(plan_schema.Step("wait", {"cycles": tail.first - board.time}, {"no_new_alarm": True}, []))
    if tail.reads:  # where no wait is served, one forward read plays the cycles to the close
        cycles = max(1, tail.cycles // board.forward_read) if board.forward_read else 1
        steps.append(plan_schema.Step("read", {"sensor_group": board.verification_group(board.repaired(chosen)), "cycles": cycles}, {}, []))
    steps.append(plan_schema.Step("finish", {}, {}, []))
    return plan_schema.Plan(branch=rank, mode=mode, why=why, steps=steps[:plan_length])


def opening_plan(board):
    """The rule's plan of the opening, or None once it is read: the profile, the failure modes of every component and the fix
    history, then the reads the surface needs before a reading, the clock and the extent of the record on the AssetOpsBench rows."""
    if board.profile is None:
        steps = [plan_schema.Step("profile"), plan_schema.Step("failure_modes", {"component": ""}), plan_schema.Step("fix_history")]
        return plan_schema.Plan(branch=None, mode="test_first", why="rule: read the opening, the profile first", steps=steps)
    missing = [kind for kind in board.required_opening if kind not in board.opening_done]
    if missing:
        return plan_schema.Plan(branch=None, mode="test_first", why="rule: read what this row needs before a reading", steps=[plan_schema.Step(kind) for kind in missing])
    return None
