import math


ABSTENTION_CREDIT = 0.5  # the outcome term of a correct declaration, between a wrong answer and a repair, so that a repair earns more
# c_max, the largest cost ratio the reward charges: with w_c * COST_CAP + w_t below the smallest gap between two outcome levels,
# the outcome term orders any two episodes whatever they cost; the default weights give 0.2 * 1.5 + 0.1 = 0.4
COST_CAP = 1.5


def outcome_gap(declaration_credit=ABSTENTION_CREDIT):
    """The smallest gap between two outcome levels of the reward, a pass at 1, a correct declaration at declaration_credit
    and anything else at 0, levels that coincide counted once."""
    levels = sorted({0.0, float(declaration_credit), 1.0})
    return min(b - a for a, b in zip(levels, levels[1:]))


def good(verdict):
    """Whether an episode is good: the verifier's pass with the provenance check holding, so that an answer that cites an
    identifier the released evidence does not hold never makes a good episode. The one definition that the outcome term of the
    reward and the filter of the training data read; pass itself leaves provenance out."""
    return bool(verdict.get("pass")) and bool(((verdict.get("checks") or {}).get("provenance") or {}).get("ok"))


def reward(verdict, horizon, w_c=0.2, w_t=0.1, declaration_credit=ABSTENTION_CREDIT, cost_cap=COST_CAP):
    """The terminal reward of policy learning: the outcome term - w_c * min(cost / cost_oracle, cost_cap)
    - w_t * min(max_delay / horizon, 1), where
    the two costs share the unit of the episode's row (calls on the free base row), the outcome term is 1 for a good episode (good),
    declaration_credit for a correct declaration, ABSTENTION_CREDIT unless the caller switches it, as the online loop of GEPA does
    with zero to score its stream by the pass alone, and 0 otherwise, and max_delay is the largest detection or repair
    delay of the episode, or the horizon when a fault occurrence of the episode was never restored by an agent close (the verdict's
    unrestored), so an episode that leaves a fault in place takes the full delay term and doing nothing is never cheaper in delay
    than a slow repair. A cited identifier that the released evidence does not hold scores the episode as a failure,
    the outcome term 0 with the cost and delay terms kept, so that a made up identifier never earns more
    than an honest failure at the same cost. None when the verifier could not run or the cost of the oracle is unknown.
    The cap orders the outcomes for any asset: each penalty lies between 0 and w_c * cost_cap + w_t, so
    two episodes whose outcome terms differ by at least outcome_gap(declaration_credit) keep that order whenever
    w_c * cost_cap + w_t is below the gap and both weights are nonnegative, and weights that break either condition raise
    ValueError, since a negative weight turns a penalty into a bonus that the bound does not see."""
    if w_c < 0 or w_t < 0:
        raise ValueError(f"the weights of the reward must be nonnegative, w_c = {w_c} and w_t = {w_t}")
    if w_c * cost_cap + w_t >= outcome_gap(declaration_credit):
        raise ValueError(f"w_c * cost_cap + w_t = {w_c * cost_cap + w_t} must lie below the smallest gap between outcome "
                         f"levels, {outcome_gap(declaration_credit)}, or a cheap failure could outscore a costly success")
    if not verdict.get("verified") or verdict.get("cost_normalized") is None:
        return None
    provenance_ok = verdict["checks"]["provenance"]["ok"]
    outcome = 1.0 if good(verdict) else declaration_credit if provenance_ok and verdict["abstention_correct"] else 0.0
    delays = [d for d in (*verdict["detection_delay"].values(), *verdict["repair_delay"].values()) if d is not None]
    max_delay = horizon if verdict["unrestored"] else max(delays, default=0)
    return outcome - w_c * min(verdict["cost_normalized"], cost_cap) - w_t * min(max_delay / horizon, 1.0)


def pass_power_k(successes, n, k):
    """The unbiased estimator of pass^k of tau bench: for a task solved c times in n trials, C(c, k) / C(n, k); for several
    tasks, the mean of that quantity. successes is one count or a sequence of counts, one per task."""
    counts = [successes] if isinstance(successes, int) else list(successes)
    if not counts or not 1 <= k <= n:
        raise ValueError(f"pass^k needs at least one task and 1 <= k <= n, got k={k}, n={n}")
    if any(not 0 <= c <= n for c in counts):
        raise ValueError(f"every success count must lie between 0 and n={n}, got {counts}")
    return sum(math.comb(c, k) for c in counts) / (len(counts) * math.comb(n, k))
