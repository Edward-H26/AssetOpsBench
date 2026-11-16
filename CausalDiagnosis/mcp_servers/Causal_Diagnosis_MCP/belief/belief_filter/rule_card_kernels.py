import numpy as np


# ----- the kernels of the rule cards -----
def close_kernel(states, j, mode, t):
    """A work order on component j closes at cycle t: the component becomes healthy when every component it depends on is
    healthy, else the order is ineffective and nothing changes; an emergency close holds the level it replaced until
    t + hold, a standard close ends a hold; replacing a healthy component without a hold damages it with probability
    collateral, as RigClock._close does."""
    rules = states.rules
    deps_ok = np.ones(len(states), dtype=bool)
    for d in rules.deps[j]:
        deps_ok &= states.pos[:, d] == 0
    held = states.hold_pos[:, j] >= 0
    replaced_healthy = (states.pos[:, j] == 0) & ~held
    rows = np.nonzero(deps_ok)[0]
    if mode == "emergency":
        replaced = np.where(held, states.hold_pos[:, j], states.pos[:, j])
        holding = rows[replaced[rows] != 0]
        states.hold_pos[holding, j] = replaced[holding]
        states.hold_exp[holding, j] = t + rules.hold
    else:
        states.clear_hold(rows, j)
    states.set_position(rows, j, 0)
    if rules.collateral > 0:
        damaged = np.nonzero(replaced_healthy & (states.pos[:, j] < rules.worst[j]))[0]
        if len(damaged):
            copy = states.branch(damaged, rules.collateral)
            copy.set_position(np.arange(len(copy)), j, copy.pos[:, j] + 1)
            copy.clear_hold(np.arange(len(copy)), j)
            states.extend(copy)


def expiry_kernel(states, t):
    """An emergency hold that expires at or before cycle t returns its component to the level it replaced."""
    for j in range(states.rules.J):
        rows = np.nonzero((states.hold_exp[:, j] >= 0) & (states.hold_exp[:, j] <= t))[0]
        if len(rows):
            states.set_position(rows, j, states.hold_pos[rows, j])
            states.clear_hold(rows, j)


def schedule_kernel(states, t):
    """The scheduled drops of cycle t: a scheduled component drops to the k-th level below its start level at its first
    drop cycle plus k - 1 intervals, an absolute level, and the drop ends its hold."""
    rules = states.rules
    for s in range(states.sched.shape[1]):
        c, tau, l0 = states.sched[:, s, 0], states.sched[:, s, 1], states.sched[:, s, 2]
        k = t - tau
        target = l0 + 1 + np.where(k >= 0, k, 0) // rules.interval
        fire = (k >= 0) & (k % rules.interval == 0) & (target <= rules.worst[c])
        for j in range(rules.J):
            rows = np.nonzero(fire & (c == j))[0]
            if len(rows):
                states.clear_hold(rows, j)
                states.set_position(rows, j, target[rows])


def hazard_kernel(states):
    """The stochastic hazard of one cycle: every component below its worst level drops one level with the rate of its
    depth, independently across components, and a drop ends its hold. Equal rows are merged after each component, which
    keeps the rows of one cycle near the number of reachable states."""
    rules = states.rules
    for j in range(rules.J):
        q = rules.hazard_rate[j][states.pos[:, j]]
        rows = np.nonzero(q > 0)[0]
        if len(rows):
            copy = states.branch(rows, q[rows])
            copy.set_position(np.arange(len(copy)), j, copy.pos[:, j] + 1)
            copy.clear_hold(np.arange(len(copy)), j)
            states.extend(copy)
            states.merge()


def cycle_slots(closures, rules):
    """{cycle: {"start": [...], "in_cycle": [...]}}: a closure of an agent's order whose repair had a downtime closed at the
    end of the cycle before its stamp, so it comes first; a repair without a downtime closes inside its cycle after the drops. The closures of one slot close in priority order, ties
    in the order given, as RigClock closes them, so that a dependency closing in the same cycle takes first."""
    slots = {}
    for closure in sorted(closures, key=lambda c: (c.t, rules.priority[rules.components.index(c.component)])):
        j = rules.components.index(closure.component)
        slot = "start" if rules.downtime[j] > 0 else "in_cycle"
        slots.setdefault(closure.t, {"start": [], "in_cycle": []})[slot].append(closure)
    return slots


def run_cycle(states, t, slot, on_change):
    """The kernels of cycle t that follow the closures of its start, in the order of RigClock.advance_cycle, with the
    closures inside the cycle of the slot."""
    rules = states.rules
    expiry_kernel(states, t)
    if rules.scheduled:
        schedule_kernel(states, t)
    if rules.stochastic:
        hazard_kernel(states)
    for closure in slot.get("in_cycle", ()):
        close_kernel(states, rules.components.index(closure.component), closure.mode, t)
    on_change(states)


def enter(states, label, covariance, block):
    """Rows whose combination changed enter a fresh block, labelled by the number of recorded cycles observed before it:
    the offset restarts at the model mean with the block covariance of each hypothesis's fit, which the rows of one hypothesis
    entered between the same two observations share."""
    rows = np.nonzero(states.changed)[0]
    if len(rows):
        states.entry[rows] = label
        states.payload["offset"][:, rows, :] = 0.0
        covariance.setdefault(label, block.copy())
    states.changed[:] = False
