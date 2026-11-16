"""The technician visits of the work orders, as one rule for the environment, the agent and the diagnostics server.

A visit is a run of consecutive accepted standard orders: an order joins the visit of the order before it when the step of the
environment just before it was that accepted standard order, and any other step ends the run, a refused call included. The
agent's thinking time and the calls of the diagnostics server are no steps of the environment, so they end no run, and a
no_call step (a turn of the agent that calls no tool of the rig) is a step of the clock and no call
of the environment, so it ends no run either (keeps_run). An emergency order opens a visit of its own. One visit charge per visit.

The schedule of a visit: its repairs that have begun keep their cycles; the technician starts the others at the latest of the
largest creation cycle plus lead time among them and the close of the last repair begun, and repairs them back to back, the
smaller priority number first and ties in creation order. With every order created in one cycle and none begun this is the
schedule of the orders created in one cycle. The module imports nothing, so that every reader holds the same rule."""


NO_CALL_ACTION = "no_call"  # the step of an agent's iteration that called no tool, sent by the driver's entry (CausalDiagnosis.agents.utils.idle_hook)
RUN_KEEPING_STEPS = (NO_CALL_ACTION,)  # the step kinds that are no call of the environment, which end no run of orders


def is_no_call(entry):
    """Whether an entry of the log (its event) or of the state's tool_calls (its tool) is a no_call step, which the step limit counts
    and no count of tool calls does: not the cost on the calls unit, the tool calls of a row or its refusals."""
    return entry.get("event") == NO_CALL_ACTION or entry.get("tool") == NO_CALL_ACTION


def keeps_run(kind):
    """Whether a step of this kind leaves the run of orders open: the no_call step alone, which calls no tool of the rig."""
    return kind in RUN_KEEPING_STEPS


def joins(previous_was_accepted_standard_order, mode="standard"):
    """Whether an order in this mode joins the visit of the order before it: only a standard order, and only when the step of the
    environment just before it was an accepted standard order."""
    return mode == "standard" and bool(previous_was_accepted_standard_order)


def visit_schedule(members, after=None):
    """[(index into members, execute cycle, close cycle)] in the order of the repairs, for the orders of one visit that have not
    begun. members are (created cycle, lead time, repair downtime, priority) in creation order; after is the close of the last
    repair of the visit that has begun, or None."""
    members = list(members)
    if not members:
        return []
    start = max(created + lead for created, lead, _downtime, _priority in members)
    if after is not None:
        start = max(start, after)
    out = []
    for i in sorted(range(len(members)), key=lambda i: members[i][3]):  # a stable sort keeps the creation order on a tie
        close = start + members[i][2]
        out.append((i, start, close))
        start = close
    return out
