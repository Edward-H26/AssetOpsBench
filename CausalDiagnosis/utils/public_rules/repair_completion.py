"""What finishing a repair needs, as one rule for every reader of it: the agent's plan check, advice, schedule and finish guard
and the diagnostics server's reserve. The rule is the environment's own, read from what the profile publishes: the technician
of a visit arrives at the latest creation cycle plus lead time of its orders and repairs one component after the other, the
smaller priority number first and ties in creation order (visits.visit_schedule); a wait for notification returns at every
public event of a visit, the start of a repair with downtime one cycle after it begins and every close (a repair without
downtime one cycle after its close, none for a repair that closes inside its own order call); the read that verifies a repair
serves a cycle at or after the last close. Every value here is public: prices, durations, priorities and the flags of the
surface. The visits themselves follow the rule of visits.py. The module imports the standard library and visits.py alone, so
that the agent and the diagnostics server read one rule (a test holds that). What the rule cannot know is said where it matters: an alarm of a read group or a failure during the wait ends a
wait early, and thinking time moves the clock on the rows whose clock counts it."""
from dataclasses import dataclass, field

from .visits import visit_schedule


@dataclass(frozen=True)
class Rules:
    """The public values the rule reads. order_price is the part plus the repair charge per component; first_offset is 1 on a
    row whose read serves the recorded cycles before the clock, else 0; summary_call says whether the change summary after a
    wait is a call of its own; clock_runs whether a cycle runs after a close, so that a read can verify a repair. wait_served says
    whether the row serves wait_for_notification, and forward_read the cycles one cycle of a read takes where no wait is served and
    the history plays the cycles a window reaches past the present (0 elsewhere); both are required and keyword-only, so no caller can leave them out."""

    lead_time: dict
    repair_downtime: dict
    wait_served: bool = field(kw_only=True)
    forward_read: int = field(kw_only=True)
    priority: dict = field(default_factory=dict)
    order_price: dict = field(default_factory=dict)
    visit_charge: float = 0
    read_duration: int = 0
    first_offset: int = 0
    summary_call: bool = True
    clock_runs: bool = True


@dataclass(frozen=True)
class Completion:
    """What finishing a repair needs from one state: the price (of the orders and of the verifying read, read_price, the part a
    caller that prices the repair alone names), the tool calls (the commitment where none is made, the orders, the waits and the
    change summary, the verifying read, finish), the cycles to the end of that read, the cycles to the last close, and the first
    cycle at which the read verifies (None where no read is due), and whether a verifying read is in the tail (reads)."""

    price: float
    calls: int
    cycles: int
    to_close: int
    first: object
    read_price: float = 0
    reads: int = 0


def schedule(members, lead_time, repair_downtime, priority=None):
    """[(component, created cycle, execute cycle, close cycle)] of the orders of one visit, members (component, created cycle,
    close cycle of a repair that has begun or None) in creation order: the repairs begun keep their close, the others follow by
    visits.visit_schedule."""
    priority = priority or {}
    begun = [(c, created, close - repair_downtime[c], close) for c, created, close in members if close is not None]
    waiting = [(c, created) for c, created, close in members if close is None]
    after = max((close for *_rest, close in begun), default=None)
    plan = visit_schedule([(created, lead_time[c], repair_downtime[c], priority.get(c, 0)) for c, created in waiting], after)
    return begun + [(waiting[i][0], waiting[i][1], execute_t, close_t) for i, execute_t, close_t in plan]


def visit_moments(scheduled):
    """The clock values at which a wait over a visit (schedule) returns."""
    moments = set()
    for _c, created_t, execute_t, close_t in scheduled:
        if close_t > execute_t:
            moments.update((execute_t + 1, close_t))
        elif close_t > created_t:
            moments.add(close_t + 1)
    return moments


def wait_calls(start, end, moments, summary_call):
    """The tool calls of a wait from start to end: one per moment strictly inside, the call that reaches the end, and the
    change summary where it is a call."""
    if end <= start:
        return 0
    return sum(1 for m in moments if start < m < end) + 1 + (1 if summary_call else 0)


def completion(rules, time, new=(), visits=None, pending=(), read_price=0, commit=False, run=None, verifiable=True):
    """What finishing the repair needs from the state at `time`: `new` are orders created at `time`, which join the visit `run`
    when the step just before them was an accepted standard order of that visit (no visit charge) and open a visit otherwise,
    `visits` {visit: [(component, created cycle, close of a repair begun or None)]} the open orders of the state, `pending` the
    closes of repairs closed and not yet verified, read_price the price of the verifying read, commit whether the diagnosis or
    the declaration is still to be made, verifiable whether some read can verify the repair (where none can, the tail still waits
    for the close, since finish cannot come before it, and buys no read)."""
    visits = {key: list(members) for key, members in (visits or {}).items()}
    new = list(new)
    if new:
        visits.setdefault(run if run is not None else ("new", time), []).extend((c, time, None) for c in new)
    price = sum(rules.order_price.get(c, 0) for c in new) + (rules.visit_charge if new and run is None else 0)
    calls = (1 if commit else 0) + len(new) + 1
    closes, moments = list(pending), set()
    for members in visits.values():
        scheduled = schedule(members, rules.lead_time, rules.repair_downtime, rules.priority)
        closes += [close for *_rest, close in scheduled]
        moments |= visit_moments(scheduled)
    if not closes:
        return Completion(price, calls, 0, 0, None)
    last = max(closes)
    if not rules.clock_runs:
        return Completion(price, calls, 0, max(0, last - time), None)
    first = last + rules.first_offset
    if not rules.wait_served:
        # no wait: the close comes with the thinking, or with one read of the verifying group whose window plays the cycles to it
        if not verifiable or not rules.forward_read:
            return Completion(price, calls, 0, max(0, last - time), None)
        played = max(0, last - time + 1)
        paid = read_price * max(1, played)
        return Completion(price + paid, calls + 1, played * rules.forward_read, max(0, last - time), first, paid, 1)
    wait = max(0, first - time)
    calls += wait_calls(time, time + wait, moments, rules.summary_call)
    if not verifiable:
        return Completion(price, calls, wait, max(0, last - time), first)
    return Completion(price + read_price, calls + 1, wait + rules.read_duration, max(0, last - time), first, read_price, 1)


def verifies(channels, components, indicating, groups):
    """Whether a reading that serves these channels can verify a repair of the components: it holds an indicating channel of
    every repaired component whose indicating channels the reader knows (indicating: {component: channels}, from the failure
    modes or the repair manual) and some sensor group (groups: {group: channels}) holds; for a repaired component of which the
    reader knows no such channel, only a reading of every channel of the asset verifies."""
    channels = set(channels)
    every = set().union(*(set(g) for g in groups.values())) if groups else set()
    for component in components:
        known = set(indicating.get(component) or ())
        if not known or not any(set(g) & known for g in groups.values()):
            known = None
        if not (channels & known if known is not None else channels >= every):
            return False
    return True
