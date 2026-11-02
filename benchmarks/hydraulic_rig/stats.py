"""Statistics behind the result tables of the hydraulic rig benchmark (docs/hydraulic_rig/benchmarks.md).

Every function takes plain sequences and returns plain numbers, so aggregate.py and the tests call them on arrays they
build themselves. Episodes that share a recorded block share most of their cycles, so an interval of a difference
resamples whole blocks rather than episodes; a family of primary tests is corrected by the Holm step down; a
proportion gets a Wilson interval; turns to a correct diagnosis are summarized as a restricted mean survival time with
censoring at the step cap; the agreement of two verdicts is Cohen's kappa; the ranking of agents under two verifier
scopes is compared by Spearman's correlation.
"""
import math
import random

import numpy as np

Z_95 = 1.959963984540054  # two sided 95 percent normal quantile


def paired_cluster_bootstrap(a, b, clusters, resamples=10_000, seed=0, level=0.95):
    """Mean of the paired differences a[i] - b[i] with a percentile interval from a bootstrap that resamples whole
    clusters (the recorded blocks) with replacement. The two sided p value is the add one share of resampled means on
    the far side of zero, doubled and capped at one, so it never reaches zero on a finite number of resamples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or len(a) != len(clusters):
        raise ValueError("a, b and clusters need one entry per episode")
    if len(a) == 0:
        return {"difference": None, "low": None, "high": None, "p_value": None, "clusters": 0, "episodes": 0}
    d = a - b
    members = {}
    for i, label in enumerate(clusters):
        members.setdefault(label, []).append(i)
    labels = sorted(members, key=str)
    rng = random.Random(f"paired-cluster-bootstrap:{seed}")
    means = np.empty(resamples)
    for r in range(resamples):
        drawn = rng.choices(labels, k=len(labels))
        idx = np.concatenate([members[label] for label in drawn])
        means[r] = d[idx].mean()
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    share_low = (np.sum(means <= 0) + 1) / (resamples + 1)
    share_high = (np.sum(means >= 0) + 1) / (resamples + 1)
    return {"difference": float(d.mean()), "low": float(low), "high": float(high),
            "p_value": float(min(1.0, 2 * min(share_low, share_high))), "clusters": len(labels), "episodes": int(len(d))}


def cluster_bootstrap(statistic, items, clusters, resamples=10_000, seed=0, level=0.95):
    """A statistic of a list of items with a percentile interval from a bootstrap that resamples whole clusters with
    replacement; statistic(items) returns a number or None, and resamples where it is None are dropped."""
    if len(items) != len(clusters):
        raise ValueError("items and clusters need one entry per episode")
    estimate = statistic(list(items)) if items else None
    if estimate is None:
        return {"estimate": None, "low": None, "high": None, "clusters": len(set(map(str, clusters))), "items": len(items)}
    members = {}
    for i, label in enumerate(clusters):
        members.setdefault(label, []).append(i)
    labels = sorted(members, key=str)
    rng = random.Random(f"cluster-bootstrap:{seed}")
    values = []
    for _ in range(resamples):
        drawn = rng.choices(labels, k=len(labels))
        value = statistic([items[i] for label in drawn for i in members[label]])
        if value is not None:
            values.append(value)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail]) if values else (None, None)
    return {"estimate": float(estimate), "low": None if low is None else float(low), "high": None if high is None else float(high),
            "clusters": len(labels), "items": len(items)}


def holm(p_values, alpha=0.05):
    """Holm step down over one family: adjusted p values (monotone in the sorted order, capped at one) and the
    rejections at alpha, both in the order of the input."""
    p = list(p_values)
    order = sorted(range(len(p)), key=lambda i: p[i])
    adjusted = [None] * len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return {"adjusted": adjusted, "reject": [q <= alpha for q in adjusted]}


def wilson(successes, n, z=Z_95):
    """Wilson score interval of a proportion; every field is None when n is zero."""
    if n == 0:
        return {"proportion": None, "low": None, "high": None}
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"proportion": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def rmst(turns, observed, cap):
    """Restricted mean survival time over turns: the area under the Kaplan Meier survival curve of the turn of the
    event (a correct diagnosis) from turn zero to the cap. turns[i] is the event turn when observed[i] is true and the
    censoring turn otherwise; an episode that ends without the event is censored at the step cap by the caller. With
    integer turns the area is the expected number of turns spent before the event, capped."""
    pairs = sorted(zip(turns, observed))
    if not pairs:
        return None
    at_risk = len(pairs)
    survival, area, previous = 1.0, 0.0, 0.0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        if t >= cap:
            break
        events = sum(1 for turn, seen in pairs[i:] if turn == t and seen)
        count = sum(1 for turn, _ in pairs[i:] if turn == t)
        area += survival * (t - previous)
        survival *= 1 - events / at_risk
        at_risk -= count
        previous = t
        i += count
    area += survival * (cap - previous)
    return area


def cohens_kappa(a, b):
    """Cohen's kappa of two label sequences; None when both raters use one label and agree everywhere, since the
    chance agreement is then one and kappa is undefined."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("kappa needs two sequences of the same length")
    if not a:
        return None
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    chance = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)


def spearman(x, y):
    """Spearman's rank correlation with average ranks for ties; None with fewer than two pairs or a constant input."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks
