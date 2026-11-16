"""Statistics behind the result tables of the hydraulic rig benchmark (CausalDiagnosis/docs/experiments.md).

Every function takes plain sequences and returns plain numbers, so tables_from_runs/ and the tests call them on arrays they
build themselves. Episodes that share a recorded block share most of their cycles, so an interval of a difference
resamples whole blocks rather than episodes, and so does the interval of a proportion, since the episodes of one start
combination share its block (cluster_proportion). Every interval comes from one bootstrap (cluster_bootstrap): the
statistic is a function of sums over items, so the sums of every resample are one matrix product of how often each
cluster is drawn (resample_counts) with the sums of each cluster, and every result carries the standard error, the
spread of the resampled values, and the number of clusters. The p value of a paired difference comes from a sign flip
test over the blocks (sign_flip_p): blocks whose differences sum to zero are dropped, the test of a zero or one outcome
is exact by convolution whatever the number of blocks, and any other outcome is enumerated when the blocks are few and
drawn otherwise. A family of primary tests is corrected by the Holm step down; the agreement of two verdicts is Cohen's
kappa; the ranking of agents under two verifier scopes is compared by Spearman's correlation.
"""
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

Z_95 = 1.959963984540054  # two sided 95 percent normal quantile


# six digits, since the pages print a cell of the tables again with fewer, and four move the last printed digit of some shares of a row's episodes
def rounded(x, digits=6):
    """A cell of a table: the number rounded to digits, or empty for None."""
    return "" if x is None else round(float(x), digits)


@lru_cache(maxsize=64)
def resample_counts(n_clusters, resamples=10_000, seed=0):
    """How often each cluster is drawn in each bootstrap resample, as a matrix of shape (resamples, n_clusters): a resample
    draws n_clusters clusters with replacement, so every row sums to n_clusters. The matrix is drawn once per
    (n_clusters, resamples, seed) from bounded integers of numpy's PCG64 generator and kept for the process, so every
    statistic over the same number of clusters reads the same resamples. It is read only."""
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, n_clusters, size=(resamples, n_clusters))
    cells = (np.arange(resamples)[:, None] * n_clusters + draws).ravel()
    # a count never exceeds n_clusters, so the smallest unsigned type that holds n_clusters holds every count, for any asset
    counts = np.bincount(cells, minlength=resamples * n_clusters).reshape(resamples, n_clusters).astype(np.min_scalar_type(n_clusters))
    counts.setflags(write=False)
    return counts


def ratio(sums):
    """Column 0 over column 1 of every row of sums: a mean when column 1 counts the items, a share when it counts the
    counted items, a ratio of two means when both columns are totals. A row whose column 1 is zero has no value."""
    sums = np.asarray(sums, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sums[:, 1] != 0, sums[:, 0] / sums[:, 1], np.nan)


def kappa_of_counts(sums):
    """Cohen's kappa of two yes or no ratings from the counts of every row of sums, in the columns both yes, the first
    alone, the second alone, neither. A row in which a rating gives one answer throughout has no value, since a margin is
    then empty and the formula reads 0 whatever the other rating says. On the counts of two sequences in which both ratings
    give both answers it equals cohens_kappa of them."""
    both, first, second, neither = (np.asarray(sums, dtype=float)[:, k] for k in range(4))
    n = both + first + second + neither
    varies = (both + first > 0) & (second + neither > 0) & (both + second > 0) & (first + neither > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = (both + neither) / n
        chance = ((second + neither) / n) * ((first + neither) / n) + ((both + first) / n) * ((both + second) / n)
        return np.where(varies, (observed - chance) / (1 - chance), np.nan)


def resampled_values(columns, clusters, statistic, resamples=10_000, seed=0):
    """The value of a statistic of sums on every bootstrap resample of whole clusters, one value per row of resample_counts.
    columns holds one row per item and one column per summed quantity. The columns are summed per cluster, the clusters
    in the sorted order of their labels, the count matrix times the cluster sums gives the sums of every resample, and
    statistic(sums) returns one value per row of sums by array arithmetic (ratio, kappa_of_counts), not a number where
    the resample has no value. Row r is the value on the resample that draws the k-th cluster resample_counts[r, k] times."""
    columns = np.asarray(columns, dtype=float)
    labels = sorted(set(clusters), key=str)
    position = {label: k for k, label in enumerate(labels)}
    sums = np.zeros((len(labels), columns.shape[1]))
    np.add.at(sums, [position[label] for label in clusters], columns)
    return np.asarray(statistic(resample_counts(len(labels), resamples, seed) @ sums), dtype=float)


def cluster_bootstrap(columns, clusters, statistic, resamples=10_000, seed=0, level=0.95):
    """A statistic of sums with a percentile interval and a standard error from a bootstrap that resamples whole clusters
    with replacement (resampled_values). A value that is no number, as for a resample without a counted item, is dropped.
    Returns the estimate on the sums of all items, low and high, se, the standard deviation of the resampled values (the
    bootstrap estimate of the standard error, None below two values), the number of clusters, of items and of resamples
    used. The estimate, the interval and the standard error are None when the items give no estimate."""
    columns = np.asarray(columns, dtype=float)
    if len(columns) != len(clusters):
        raise ValueError("columns and clusters need one entry per item")
    nothing = {"estimate": None, "low": None, "high": None, "se": None, "clusters": len(set(clusters)), "items": len(columns), "resamples_used": 0}
    if not len(columns):
        return nothing
    if columns.ndim != 2:
        raise ValueError("columns needs one row per item and one column per summed quantity")
    estimate = float(np.asarray(statistic(columns.sum(axis=0, keepdims=True)), dtype=float)[0])
    if not math.isfinite(estimate):
        return nothing
    values = resampled_values(columns, clusters, statistic, resamples, seed)
    values = values[np.isfinite(values)]
    if not len(values):
        return {**nothing, "estimate": estimate}
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"estimate": estimate, "low": float(low), "high": float(high), "se": float(values.std(ddof=1)) if len(values) > 1 else None,
            "clusters": len(set(clusters)), "items": len(columns), "resamples_used": int(len(values))}


def paired_cluster_bootstrap(a, b, clusters, resamples=10_000, seed=0, level=0.95):
    """Mean of the paired differences a[i] - b[i] with a percentile interval and a standard error from a bootstrap that
    resamples whole clusters (the recorded blocks) with replacement (cluster_bootstrap). The p value of the difference
    comes from sign_flip_p."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or len(a) != len(clusters):
        raise ValueError("a, b and clusters need one entry per episode")
    if len(a) == 0:
        return {"difference": None, "low": None, "high": None, "se": None, "clusters": 0, "episodes": 0}
    boot = cluster_bootstrap(np.column_stack([a - b, np.ones(len(a))]), clusters, ratio, resamples, seed, level)
    return {"difference": boot["estimate"], "low": boot["low"], "high": boot["high"], "se": boot["se"], "clusters": boot["clusters"],
            "episodes": int(len(a))}


def sign_flip_p(differences, clusters, flips=10_000, seed=0, whole=False):
    """The two sided p value of the mean of paired differences from a sign flip test over whole clusters. Under the null of no
    difference the sum of each cluster's differences is symmetric about zero, so every assignment of signs to the clusters is
    equally likely, and the p value is the share of assignments whose total is at least as far from zero as the observed one.
    A cluster whose sum is zero changes no assignment and is dropped first, and the p value is one when none is left. With
    whole set every remaining sum must be a whole number, as the differences of a zero or one outcome give, and the share is
    exact by convolution over the clusters (_exact_sign_flip), with no random draw, whatever their number. Without whole, all
    2**G assignments of the G remaining clusters are enumerated when 2**G is at most flips, which is exact; otherwise flips random
    assignments are drawn and the share is add one, so it never reaches zero. The caller sets whole by the kind of the outcome,
    never by the data. None without a difference."""
    d = np.asarray(differences, dtype=float)
    if len(d) != len(clusters):
        raise ValueError("differences and clusters need one entry per episode")
    if len(d) == 0:
        return None
    totals = {}
    for value, label in zip(d, clusters):
        totals[label] = totals.get(label, 0.0) + value
    sums = np.array([totals[label] for label in sorted(totals, key=str) if totals[label] != 0])
    if not len(sums):
        return 1.0
    if whole:
        if np.any(sums != np.round(sums)):
            raise ValueError("the exact sign flip test needs whole number sums per cluster")
        return _exact_sign_flip([int(s) for s in sums])
    observed = abs(sums.sum())
    tolerance = 1e-9 * max(1.0, observed)  # an assignment that ties the observed sum counts, whatever the rounding of the sums
    count = len(sums)
    if 2 ** count <= flips:
        signs = 1 - 2 * ((np.arange(2 ** count)[:, None] >> np.arange(count)) & 1)
        return float(np.mean(np.abs(signs @ sums) >= observed - tolerance))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(flips, count))
    extreme = int(np.sum(np.abs(signs @ sums) >= observed - tolerance))
    return (extreme + 1) / (flips + 1)


def _exact_sign_flip(sums):
    """The exact two sided p value of the sign flip test for whole number sums, none of them zero: the number of the 2**G sign
    assignments whose total is at least as far from zero as the observed total, over 2**G. ways[t + S] counts the assignments
    with total t, S the sum of the absolute sums, built by one shift and add per cluster in whole number arithmetic."""
    reach = sum(abs(s) for s in sums)
    ways = [0] * (2 * reach + 1)
    ways[reach] = 1
    for s in sums:
        step = abs(s)
        ways = [(ways[t - step] if t >= step else 0) + (ways[t + step] if t + step <= 2 * reach else 0) for t in range(2 * reach + 1)]
    observed = abs(sum(sums))
    extreme = sum(count for t, count in enumerate(ways) if abs(t - reach) >= observed)
    return extreme / 2 ** len(sums)


def majority_share(labels):
    """(k, n) of the constant answer: k the count of the most frequent label among n labels, (0, 0) for none."""
    labels = list(labels)
    if not labels:
        return 0, 0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()), len(labels)


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


def binomial_bar(n, right, alpha=0.05):
    """The smallest count k of n episodes that an exact one-sided binomial test at level alpha separates from a constant answer
    right in right of the same n: the upper tail P(X >= k) of X ~ Binomial(n, right / n) at most alpha. The tail is summed in
    whole numbers, n ** n times the probability, so that no rounding decides a count at the edge. None when no count of n
    separates, as when the constant answer is right in every episode."""
    if n < 1:
        return None
    limit, tail = Fraction(str(alpha)) * n ** n, 0
    for k in range(n, -1, -1):
        tail += math.comb(n, k) * right ** k * (n - right) ** (n - k)
        if tail > limit:
            return k + 1 if k < n else None
    return 0


def wilson(successes, n, z=Z_95):
    """Wilson score interval of a proportion; every field is None when n is zero."""
    if n == 0:
        return {"proportion": None, "low": None, "high": None}
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"proportion": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def cluster_proportion(hits, clusters, counted=None, resamples=10_000, seed=0, level=0.95):
    """The share of hits among the counted items, every item when counted is None, with a percentile interval and a standard
    error from a bootstrap that resamples whole clusters with replacement (cluster_bootstrap). The episodes of one start
    combination share its recorded block and are not independent, so a Wilson interval, which takes them for independent,
    comes out too narrow. A resample whose clusters count no item is dropped, and the share, the interval and
    the standard error are None when no item is counted."""
    hits = np.asarray(hits, dtype=float)
    counted = np.ones(len(hits)) if counted is None else np.asarray(counted, dtype=float)
    if len(hits) != len(clusters) or len(counted) != len(hits):
        raise ValueError("hits, counted and clusters need one entry per item")
    boot = cluster_bootstrap(np.column_stack([hits * counted, counted]), clusters, ratio, resamples, seed, level)
    return {"proportion": boot["estimate"], "low": boot["low"], "high": boot["high"], "se": boot["se"], "clusters": boot["clusters"]}


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
