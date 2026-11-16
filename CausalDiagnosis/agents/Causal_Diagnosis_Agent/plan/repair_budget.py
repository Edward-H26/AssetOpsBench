from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import repair_steps


# ----- the advice of the diagnostics server -----
def read_affordable(board, group, cycles=1):
    """Whether a reading fits the budget, the step cap and the horizon beside the completion of the reserve plan
    (reserve_completion)."""
    if group not in board.profile["sensor_groups"]:
        return False
    if board.new_cycles(group, cycles) == 0:  # every recorded cycle of the group is paid, so the read would show nothing new
        return False
    return _fits_beside_reserve(board, board.read_price(group, cycles), board.read_duration(cycles))


def _fits_beside_reserve(board, price, duration):
    """Whether one read of this price and duration, one call, leaves the completion of the reserve plan: its price, its calls
    and its cycles before the horizon (the step cap is the budget of the free base row)."""
    reserve = reserve_completion(board)
    return (price <= board.budget_left - reserve.price and 1 + reserve.calls <= board.steps_left
            and duration + reserve.cycles < board.cycles_left)


def plan_components(branch):
    """The components the repair plan of a previewed branch orders, those of its steps or its faulty set; none for a branch under
    which no component is faulty."""
    return [s["component"] for s in repair_steps(branch)] or list(branch.get("faulty_set") or [])


def _branch_costs(board):
    """[(act now cost to a restored asset, rank)] of the previewed branches whose completion (Board.completion: the orders with
    their visit charge and the reading that verifies them) fits the budget left, cheapest first and ties to the higher posterior rank. A branch
    under which no component is faulty is an option of the cost choice with its own cost, a diagnosis and a finish without a
    work order. The set of the last refuted plan is left out until a later ranking names it first (backtracking, version 7)."""
    options = []
    for branch in board.branches():
        cost = (branch.get("cost_to_restored") or {}).get("act_now")
        components = plan_components(branch)
        if board.refuted_last and sorted(components) == board.refuted_last:
            continue
        price = board.completion(components).price if components else 0
        if cost is not None and price <= board.budget_left:
            # rounded, so that branches with one faulty set, whose costs may differ in the last bits of a sum, tie
            options.append((round(cost, 6), branch["rank"]))
    return sorted(options)


def act_options(board):
    """The branch costs a diagnosis chooses among; empty after a commitment and without troubleshooting plans. Whether to
    test is the evidence guard's decision, so the branches compare by acting now only."""
    return [] if board.committed else _branch_costs(board)


def repair_options(board):
    """The branch costs a repair after the commitment chooses among, which backtracking uses after a refuted plan. A branch
    under which no component is faulty is no option there: the refuted verdict says that the repair did not restore the asset,
    so a finish without a work order cannot follow it."""
    return [(cost, rank) for cost, rank in _branch_costs(board) if plan_components(board.branch(rank) or {})]


# ----- the rule's plan -----
def _reserve_components(board):
    """The components the reserve repairs: the repair plan of the top branch, or the top candidate's faulty set without
    previews (which took the fault rule out of everything the agents read)."""
    branch = board.branch(1)
    if branch is not None:
        return plan_components(branch)
    top = board.top()
    return board.faulty_set(top.get("current_combination") or top["combination"]) if top else []


def reserve_completion(board):
    """What a reading leaves for the repair: the completion (Board.completion) of the repair plan of the first candidate,
    without the components that have an open order."""
    opened = {o["component"] for o in board.open_orders()}
    return board.completion([c for c in _reserve_components(board) if c not in opened], after_step=True)


def order_reserve(board):
    """The budget kept for the repair before a reading is bought (reserve_completion)."""
    return reserve_completion(board).price


def order_reserve_calls(board):
    """The tool calls kept for the repair before a reading is bought (reserve_completion)."""
    return reserve_completion(board).calls
