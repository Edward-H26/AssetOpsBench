"""Per-tier classifiers for the greedy reference policy (docs/hydraulic_rig/environment.md). Standard library only.

One linear discriminant analysis per tier and component predicts the component's level from the channel summaries
that a measurement returns. The classifiers are cross-fitted by combination: the 144 combinations go to N_FOLDS
folds by a seeded draw stratified by level, and the stable cycles of each fold are predicted by classifiers fitted on
the other folds. The held-out predictions form a table keyed by the measured values, and their counts give the
observation model of the greedy policy. The evaluation scripts repeat everything over FOLD_SEEDS.
"""
import itertools
import math
import random
from operator import mul

from .data import COMPONENTS, LEVELS, tier_features

COMBOS = tuple(itertools.product(*(LEVELS[c] for c in COMPONENTS)))
COMBO_INDEX = {combo: h for h, combo in enumerate(COMBOS)}
TIERS = (1, 2, 3)
N_FOLDS = 8
FOLD_SEEDS = tuple(range(10))
SPLIT_RULE = ("folds of the 144 combinations, stratified by level: the combinations are shuffled with the fold seed and each "
              "joins the fold, among those not yet full, where its levels are least represented; the stable cycles of each "
              "fold are predicted by classifiers fitted on the stable cycles of the other folds")
SHRINKAGE = 0.01
ALPHA_GRID = (0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 1000)


def signature(features):
    return tuple(sorted(features.items()))


def dot(a, b):
    return sum(map(mul, a, b))


def scatter(rows):
    """Sum of x x^T over rows, as a list of lists."""
    cols = list(zip(*rows))
    d = len(cols)
    out = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            out[i][j] = out[j][i] = dot(cols[i], cols[j])
    return out


def solve_spd(a, rhs):
    """Solve a x = b for each b in rhs, where a is symmetric positive definite (Cholesky factorisation)."""
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - dot(L[i][:j], L[j][:j])
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    LT = [list(col) for col in zip(*L)]
    out = []
    for b in rhs:
        y = [0.0] * n
        for i in range(n):
            y[i] = (b[i] - dot(L[i][:i], y[:i])) / L[i][i]
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - dot(LT[i][i + 1:], x[i + 1:])) / L[i][i]
        out.append(x)
    return out


def lda_fit(total_scatter, class_sums, class_counts, shrinkage=SHRINKAGE):
    """LDA with equal class priors from sufficient statistics of the training rows.

    total_scatter is the sum of x x^T, class_sums[k] the sum of x and class_counts[k] the number of rows of class k.
    Features are standardised on the training rows and the pooled within-class covariance is shrunk towards a
    scaled identity. Returns (mean, scale, coef, bias) for lda_predict."""
    n, k = sum(class_counts), len(class_counts)
    d = len(total_scatter)
    mean = [sum(s[j] for s in class_sums) / n for j in range(d)]
    scale = [math.sqrt(max(total_scatter[j][j] / n - mean[j] ** 2, 0.0)) or 1.0 for j in range(d)]
    centroids = [[s[j] / c for j in range(d)] for s, c in zip(class_sums, class_counts)]
    cov = [[(total_scatter[i][j] - sum(c * m[i] * m[j] for c, m in zip(class_counts, centroids))) / (scale[i] * scale[j] * (n - k))
            for j in range(d)] for i in range(d)]
    ridge = shrinkage * sum(cov[j][j] for j in range(d)) / d
    cov = [[(1 - shrinkage) * cov[i][j] + (ridge if i == j else 0.0) for j in range(d)] for i in range(d)]
    z_centroids = [[(m[j] - mean[j]) / scale[j] for j in range(d)] for m in centroids]
    coef = solve_spd(cov, z_centroids)
    bias = [-0.5 * dot(m, w) for m, w in zip(z_centroids, coef)]
    return mean, scale, coef, bias


def lda_predict(model, x):
    mean, scale, coef, bias = model
    z = [(v - m) / s for v, m, s in zip(x, mean, scale)]
    scores = [dot(z, w) + b for w, b in zip(coef, bias)]
    return max(range(len(scores)), key=scores.__getitem__)


def stratified_folds(n_folds=N_FOLDS, seed=0):
    """Fold of each combination under fold seed `seed`. The combinations are shuffled with the seed, and each in turn
    joins the fold, among those not yet full, in which its four levels are least represented. A level's count is
    weighted by its component's number of levels, so that it measures the level against its share of a full fold;
    ties go to a draw from the same stream. Stratifying matters because the levels form a grid: a split by sorted
    index modulo 8 would give every fold a single accumulator level."""
    rng = random.Random(seed)
    order = list(COMBOS)
    rng.shuffle(order)
    room = [len(COMBOS) // n_folds + (f < len(COMBOS) % n_folds) for f in range(n_folds)]
    counts = [[[0] * len(LEVELS[c]) for c in COMPONENTS] for _ in range(n_folds)]
    fold = {}
    for combo in order:
        level = [LEVELS[c].index(v) for c, v in zip(COMPONENTS, combo)]
        load = {f: sum(counts[f][j][k] * len(LEVELS[c]) for j, (c, k) in enumerate(zip(COMPONENTS, level)))
                for f in range(n_folds) if room[f]}
        least = min(load.values())
        f = rng.choice([g for g, v in load.items() if v == least])
        fold[combo] = f
        room[f] -= 1
        for j, k in enumerate(level):
            counts[f][j][k] += 1
    return fold


def cross_fitted_predictions(x, cycle_combo, fold, shrinkage=SHRINKAGE):
    """Held-out predicted level indices per cycle: the cycles of each fold are predicted by LDA models fitted on the
    cycles of the other folds. x maps a cycle index to its feature values, cycle_combo maps it to its combination,
    and fold maps a combination to its fold."""
    folds = sorted({fold[cycle_combo[i]] for i in x})
    by_fold = {f: [i for i in x if fold[cycle_combo[i]] == f] for f in folds}
    fold_scatter = {f: scatter([x[i] for i in idx]) for f, idx in by_fold.items()}
    d = len(next(iter(x.values())))
    total = [[sum(fold_scatter[f][r][c] for f in by_fold) for c in range(d)] for r in range(d)]
    pred = {}
    for f, test in by_fold.items():
        train = [i for g, idx in by_fold.items() if g != f for i in idx]
        train_scatter = [[total[r][c] - fold_scatter[f][r][c] for c in range(d)] for r in range(d)]
        models = []
        for j, comp in enumerate(COMPONENTS):
            k = len(LEVELS[comp])
            sums, counts = [[0.0] * d for _ in range(k)], [0] * k
            for i in train:
                level = LEVELS[comp].index(cycle_combo[i][j])
                counts[level] += 1
                sums[level] = [a + b for a, b in zip(sums[level], x[i])]
            if 0 in counts:
                raise ValueError(f"the training set of fold {f} has no cycle with {comp} at {LEVELS[comp][counts.index(0)]}")
            models.append(lda_fit(train_scatter, sums, counts, shrinkage))
        for i in test:
            pred[i] = tuple(lda_predict(m, x[i]) for m in models)
    return pred


def held_out_accuracy(pred, cycle_combo, combos=None):
    """Share of held-out predictions equal to the true level, per component, over the cycles of `combos` (default all)."""
    keep = set(combos) if combos is not None else set(COMBOS)
    cycles = [i for i in pred if cycle_combo[i] in keep]
    return {c: sum(pred[i][j] == LEVELS[c].index(cycle_combo[i][j]) for i in cycles) / len(cycles)
            for j, c in enumerate(COMPONENTS)}


def dm_log_likelihood(alpha, rows, counts):
    """Dirichlet-multinomial log likelihood of per-block prediction counts around per-block base rows."""
    total = 0.0
    for base, cnt in zip(rows, counts):
        total += math.lgamma(alpha) - math.lgamma(alpha + sum(cnt))
        total += sum(math.lgamma(alpha * p + c) - math.lgamma(alpha * p) for p, c in zip(base, cnt))
    return total


def observation_model(pred, cycle_combo):
    """The greedy policy's observation model of one tier from held-out predictions (cycle index -> level indices).

    base[j][h]: distribution of the predicted level of component j under hypothesis COMBOS[h], from the predictions
    of every block except h's own (one added to each count). alpha[j]: Dirichlet-multinomial concentration of
    repeated predictions within one block, the maximum-likelihood value on ALPHA_GRID."""
    base, alpha = [], []
    for j, comp in enumerate(COMPONENTS):
        k = len(LEVELS[comp])
        block_counts = [[0] * k for _ in COMBOS]
        for i, p in pred.items():
            block_counts[COMBO_INDEX[cycle_combo[i]]][p[j]] += 1
        level = [LEVELS[comp].index(h[j]) for h in COMBOS]
        pooled = [[0] * k for _ in range(k)]
        for h, cnt in enumerate(block_counts):
            pooled[level[h]] = [a + b for a, b in zip(pooled[level[h]], cnt)]
        rows = []
        for h, cnt in enumerate(block_counts):
            own_out = [a - b + 1.0 for a, b in zip(pooled[level[h]], cnt)]
            rows.append([v / sum(own_out) for v in own_out])
        base.append(rows)
        alpha.append(max(ALPHA_GRID, key=lambda a: dm_log_likelihood(a, rows, block_counts)))
    return base, alpha


class TierClassifiers:
    """Cross-fitted per-tier, per-component LDA predictions and the observation model built from them.

    fold[combo]: the fold of each combination under fold seed `seed`. features[tier]: cycle index -> the tier's
    feature values in sorted-name order. lookup[tier]: measured values -> predicted level indices. base[tier] and
    alpha[tier]: the observation model of that tier (observation_model).
    """

    def __init__(self, cycles, blocks, shrinkage=SHRINKAGE, n_folds=N_FOLDS, seed=0):
        if tuple(sorted(blocks)) != COMBOS:
            raise ValueError("blocks must cover exactly the 144 combinations")
        by_index = {c.index: c for c in cycles}
        self.seed = seed
        self.fold = stratified_folds(n_folds, seed)
        self.split = f"{n_folds} {SPLIT_RULE}; fold seed {seed}"
        self.cycle_combo = {i: combo for combo in COMBOS for i in blocks[combo]}
        self.features, self.lookup, self.predicted, self.base, self.alpha = {}, {}, {}, {}, {}
        for tier in TIERS:
            feats = {i: tier_features(by_index[i], tier) for i in self.cycle_combo}
            names = sorted(next(iter(feats.values())))
            x = {i: [f[n] for n in names] for i, f in feats.items()}
            self.features[tier] = x
            pred = cross_fitted_predictions(x, self.cycle_combo, self.fold, shrinkage)
            self.predicted[tier] = pred
            self.lookup[tier] = {signature(feats[i]): p for i, p in pred.items()}
            self.base[tier], self.alpha[tier] = observation_model(pred, self.cycle_combo)

    def predict(self, tier, features):
        """Held-out predicted level indices (one per component) for a measurement the environment returned."""
        try:
            return self.lookup[tier][signature(features)]
        except KeyError:
            raise KeyError("measurement not in the cross-fitted table; build TierClassifiers from the environment's cycles") from None

    def accuracy(self, tier, combos=None):
        """Share of held-out predictions equal to the true level, per component, over the stable cycles of `combos` (default all)."""
        return held_out_accuracy(self.predicted[tier], self.cycle_combo, combos)
