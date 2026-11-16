"""The schedule of the candidate plans on the board of the CausalDiagnosis agent (CausalDiagnosis/docs/experiments.md, section The
driver, agent causal).

The candidate plans of one board share one asset and one body of evidence, so one reading serves every plan whose test its
sensor group covers. At every planning turn the controller reads the plans as jobs: with the diagnostics server the
troubleshooting plans of the ranking, each with its confirm test and its repair plan, and without it the model's own
candidates, each tested by the cheapest group whose channels indicate its faulty components. The schedule holds the reads that
the server names and the reads whose test several plans share, each once with the plans it serves, as one batch of consecutive
read steps. The batch is cut where its reads together would no longer leave the budget, the tool calls or the cycles for a
repair, and a plan whose repair no longer fits after them is named as left out. After a refuted repair plan the schedule lists
what each remaining plan still needs, and no executed step comes back: an open order is not ordered again and a paid cycle is
not priced again.

The schedule is advice on the board. The controller makes no call for it, and the model plans the reads it takes, other reads,
fewer reads or none. Every rule here reads the plans, the sensor groups, the prices, the durations and the limits from the
board, so it names no asset.
"""
from dataclasses import dataclass, field

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import ReadWalk, n_cycles, repair_steps

EVERY_COMPONENT = "repair of every component"  # the last condition of a read that the server does not name
NOTHING_NEW = "nothing new"  # a named read that would pay for no cycle the board does not hold
REASONS = {"budget": "the reads together would not leave the budget for the repair",
           "step cap": "the reads together would not leave the tool calls for the repair",
           "horizon": "the reads together would not leave the cycles for the repair",
           EVERY_COMPONENT: "the budget would not pay the read beside one visit for every component not yet repaired",
           NOTHING_NEW: "it would show no cycle that the board does not hold"}


@dataclass(frozen=True)
class Needs:
    """What must be left after the reads for a completion: a price, tool calls and cycles to the end of the verifying read, and
    the cycles to the last close (Board.completion)."""

    price: float = 0
    calls: float = 0
    cycles: float = 0
    to_close: float = 0


@dataclass(frozen=True)
class Job:
    """One candidate plan: its rank, its mass (None without the diagnostics server), the components its repair still orders, its
    test as (sensor group, cycles) or None, and what its completion needs."""

    rank: int
    mass: object
    components: tuple
    test: object
    needs: Needs


@dataclass
class FusedRead:
    """One read of the batch: its sensor group and cycles, the ranks whose tests it serves and whether the server names it."""

    group: str
    cycles: int
    serves: list = field(default_factory=list)
    named: bool = False


@dataclass
class Schedule:
    """The schedule of one planning turn. differs is true when it says more than the board without it says."""

    reads: list = field(default_factory=list)
    cut: list = field(default_factory=list)  # [sensor group, cycles, limit] of the reads the batch leaves out
    left_out: list = field(default_factory=list)  # [rank, limit] of the plans whose repair no longer fits after the reads
    remaining: list = field(default_factory=list)  # after a refuted plan, what each remaining plan still needs
    differs: bool = False

    def to_json(self):
        return {"reads": [{"sensor_group": r.group, "cycles": r.cycles, "serves": list(r.serves), "named": r.named} for r in self.reads],
                "cut": [list(c) for c in self.cut], "left_out": [list(j) for j in self.left_out],
                "remaining": [dict(r) for r in self.remaining], "differs": self.differs}


def needs_of(board, components):
    """What the completion of a plan on the components needs (Board.completion), as the order reserve counts it."""
    tail = board.completion(list(components), after_step=True)
    return Needs(tail.price, tail.calls, tail.cycles, tail.to_close)


def residual(board, needs):
    """What the episode has left after a plan alone: the budget, the cycles and the steps, compared in this order."""
    return (board.budget_left - needs.price, board.cycles_left - needs.cycles, board.steps_left - needs.calls)


def model_test(board, components):
    """The test of a candidate of the model's own: the cheapest group whose channels indicate every component, for one cycle, or
    None when the failure modes on the board name no channel for any of them."""
    indicating = {c for mode in board.failure_modes if mode.get("component") in components for c in mode.get("channels") or []}
    return (board.verification_group(list(components)), 1) if indicating else None


def jobs(board):
    """The candidate plans of the board as jobs, in the order of the queue. With the diagnostics server one job per branch that
    the board shows, the larger mass first, then the least that the episode has left after the plan alone, then the rank: the
    set of the last refuted repair plan is no job until a later ranking names it first, a component with an open order of the
    agent is not ordered again, and a component whose order has closed stays while the branch still lists it. Without the
    server the model's candidates in its order, without every component that an order of the agent names, since no belief
    follows a repair there."""
    queue = []
    groups = board.profile.get("sensor_groups") or {}
    if board.server:
        opened = {o["component"] for o in board.open_orders()}
        for branch in board.branches():
            planned = [s["component"] for s in repair_steps(branch)] or list(branch.get("faulty_set") or [])
            if board.refuted_last and sorted(planned) == board.refuted_last:
                continue
            components = tuple(c for c in planned if c not in opened)
            test = branch.get("confirm_test") or {}
            pair = (test["sensor_group"], int(test.get("cycles") or 1)) if test.get("sensor_group") in groups else None
            queue.append(Job(branch.get("rank"), branch.get("probability"), components, pair, needs_of(board, components)))
        return sorted(queue, key=lambda j: (-(j.mass or 0), residual(board, j.needs), j.rank))
    ordered = {o["component"] for o in board.orders.values()}
    for candidate in board.model_candidates:
        faulty = board.faulty_set(candidate["combination"])
        if board.refuted_last and sorted(faulty) == board.refuted_last:
            continue
        components = tuple(c for c in faulty if c not in ordered)
        queue.append(Job(candidate["rank"], None, components, model_test(board, components), needs_of(board, components)))
    return queue


def limit(board, walk, needs, every_component=False):
    """The limit that the reads of the walk break beside these needs, or None, in the form of the check of one read: their price
    and the price of the needs at most the budget left, their calls and the calls of the needs at most the steps left, their
    duration and the cycles of the needs, which end with the verifying read, below the cycles left. With every_component also
    their price beside the completion of one visit that repairs every component that no order of the agent names, since an open
    order is paid already."""
    if walk.price + needs.price > board.budget_left:
        return "budget"
    if walk.calls + needs.calls > board.steps_left:
        return "step cap"
    if not walk.duration + needs.cycles < board.cycles_left:
        return "horizon"
    if every_component:
        ordered = {o["component"] for o in board.orders.values()}
        if walk.price + board.completion([c for c in board.profile["components"] if c not in ordered], after_step=True).price > board.budget_left:
            return EVERY_COMPONENT
    return None


def walk_of(board, reads):
    """The walk of fused reads taken in their order."""
    walk = ReadWalk(board)
    for read in reads:
        walk.take(read.group, read.cycles)
    return walk


def served(board, walk, test):
    """Whether the test of a plan is served: by the board, when it would pay for no cycle anew; by the walk, when its reads
    delivered to the test's group at least the cycles that the test alone would pay for; and without the diagnostics server
    also by a reading of the group or of a group that covers it, made after the model listed its candidates."""
    group, cycles = test
    count = board.new_cycles(group, cycles)
    if count == 0 or walk.delivered(group) >= count:
        return True
    if board.server:
        return False
    return any(board.covers(g, group) and latest["n"] > board.candidates_seen for g, latest in board.latest_cycles.items())


def pack(board, queue, named, reserve, shared=True):
    """The batch. The reads that the server names come first, in its order and as it names them, as long as their sum leaves the
    reserve of the repair and each pays for a cycle that the board does not hold; the first one that does not ends them. Then
    every plan whose completion still fits after them is served and every other plan is left out with the limit it breaks.
    Then, with shared, the served plans are taken in the order of the queue, and a test that is not served joins when one read
    of its group serves the tests of two served plans or more, those in its group or in a group it covers: the read asks for
    the most cycles that those tests would pay for, raises a read of the same group that the batch holds or comes after the
    others, and joins only when with it the reserve, every served plan and one visit for every component not yet ordered still
    fit. Returns the fused reads, the reads not taken with their limit and the plans left out with theirs."""
    groups = board.profile.get("sensor_groups") or {}
    reads, cut = [], []
    walk = walk_of(board, reads)
    for group, cycles in named:
        trial = walk_of(board, reads + [FusedRead(group, cycles, named=True)]) if group in groups else None
        why = NOTHING_NEW if trial is None or trial.new == 0 else limit(board, trial, reserve)
        if why:
            cut.append([group, cycles, why])
            break
        reads.append(FusedRead(group, cycles, named=True))
        walk = trial
    chosen, left_out = [], []
    for job in queue:
        why = limit(board, walk, job.needs)
        if why:
            left_out.append([job.rank, why])
        else:
            chosen.append(job)
    tried = set()  # the groups whose read was cut: adding reads only raises the sums, so a cut read stays cut
    for job in chosen if shared else []:
        if job.test is None or job.test[0] in tried or served(board, walk, job.test):
            continue
        group = job.test[0]
        sharing = [j for j in chosen if j.test and board.covers(group, j.test[0]) and not served(board, walk, j.test)]
        if len(sharing) < 2:
            continue
        more = max(board.new_cycles(*j.test) for j in sharing) - walk.delivered(group)
        held = next((r for r in reads if r.group == group), None)
        trial_reads = [FusedRead(r.group, r.cycles + (more if r is held else 0), named=r.named) for r in reads]
        if held is None:
            trial_reads.append(FusedRead(group, more))
        trial = walk_of(board, trial_reads)
        why = (limit(board, trial, reserve) or next((w for w in (limit(board, trial, j.needs) for j in chosen) if w), None)
               or limit(board, trial, Needs(), every_component=True))
        if why:
            cut.append([group, more, why])
            tried.add(group)
            continue
        reads, walk = trial_reads, trial
    for read in reads:
        alone = walk_of(board, [read])
        read.serves = [j.rank for j in chosen if j.test and board.covers(read.group, j.test[0])
                       and alone.delivered(j.test[0]) >= board.new_cycles(*j.test) > 0]
    return reads, cut, left_out


def remaining(board, queue):
    """After a refuted plan, what each remaining plan still needs: the components to order, the cycles to the close of that
    visit and the group of the reading that verifies it."""
    return [{"rank": j.rank, "components": list(j.components), "cycles_to_close": j.needs.to_close,
             "verification_group": board.verification_group(list(j.components)) if j.components else None} for j in queue]


def schedule(board, named, reads_open, reserve, refuted):
    """The schedule of the turn: the batch when reads are open, else no read and still the plans that cannot be completed, and
    after a refuted plan what each remaining plan still needs. named are the reads that the server names, reserve the needs of
    the repair that every read leaves, as the check of one read keeps them."""
    queue = jobs(board)
    reads, cut, left_out = pack(board, queue, named if reads_open else [], reserve, shared=reads_open)
    return Schedule(reads=reads, cut=cut, left_out=left_out, remaining=remaining(board, queue) if refuted else [])


def lines(scheduled, server):
    """The section of the board, its heading first; a line is printed only when it has content."""
    whom = "ranks" if server else "your candidates"
    plan = "plan" if server else "candidate"
    out = ["## Schedule of the candidate plans (computed by the controller from the plans on this board, advice like the rest)" if server
           else "## Schedule of your candidates (computed by the controller from your candidates, advice)"]
    if scheduled.reads:
        items = []
        for read in scheduled.reads:
            notes = (["the read the server names"] if read.named else []) + ([f"serves {whom} {', '.join(map(str, read.serves))}"] if read.serves else [])
            items.append(f"{read.group} x{read.cycles}" + (f" ({', '.join(notes)})" if notes else ""))
        out.append("Reads to plan as consecutive read steps, one batch: " + ", ".join(items) + ".")
    for group, cycles, why in scheduled.cut:
        out.append(f"Not in the batch: {group} x{cycles}, since {REASONS[why]}.")
    after = " after these reads" if scheduled.reads else ""
    for rank, why in scheduled.left_out:
        out.append(f"Left out: rank {rank}, whose repair no longer fits the {why}{after}.")
    if scheduled.remaining:
        needs = [f"rank {r['rank']}: order {', '.join(r['components'])}, {n_cycles(r['cycles_to_close'])} to the close of the visit, "
                 f"then read {r['verification_group']} for 1 cycle" if r["components"] else f"rank {r['rank']}: no order left"
                 for r in scheduled.remaining]
        out.append(f"After the refuted plan, what each remaining {plan} still needs: " + "; ".join(needs) + ".")
    return out
