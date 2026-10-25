"""The additive effect model behind estimate_intervention_effect, shared by fit.py, which fits it, and adapter.py, which
predicts with it (docs/hydraulic_rig/causal-diagnosis-server.md).

A channel mean is modelled as an intercept plus one effect per non-healthy level of each component. For every pool
combination the model is fitted on the other pool combinations, so a combination's own recordings never enter its own
prediction, and the residuals of that fit are kept as quantiles on EFFECT_GRID; a combination outside the pool, which no
fit has seen, takes the fit on the whole pool. A prediction under a posterior over combinations is the mixture of the
per-combination predictions, and its band the 1st to 99th percentile of the mixture.
"""
import bisect

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS

# every half percent in the tails, where the band lies, and every five percent in the body: 39 points
EFFECT_GRID = tuple(round(q, 3) for q in [i * 0.005 for i in range(11)] + [0.05 + i * 0.05 for i in range(1, 18)] + [0.95 + i * 0.005 for i in range(11)])
BAND = (0.01, 0.99)
NEGLIGIBLE = 1e-6  # hypotheses below this posterior weight are left out of a mixture


def combo_key(combo):
    return ",".join(str(v) for v in combo)


def parse_combo(key):
    return tuple(int(v) for v in key.split(","))


def design(combo):
    """Intercept plus one indicator per non-healthy level of each component (healthy levels are the baseline)."""
    row = [1.0]
    for c, v in zip(COMPONENTS, combo):
        row += [1.0 if v == lv else 0.0 for lv in LEVELS[c] if lv != HEALTHY[c]]
    return row


def percentile(sorted_vals, q):
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def cdf(grid, quantiles, x):
    """F(x) of the distribution whose quantile function is known at the grid points, interpolated linearly between them."""
    if x < quantiles[0]:
        return 0.0
    if x >= quantiles[-1]:
        return 1.0
    k = bisect.bisect_right(quantiles, x) - 1
    return grid[k] + (grid[k + 1] - grid[k]) * (x - quantiles[k]) / (quantiles[k + 1] - quantiles[k])


def mixture(parts, grid=EFFECT_GRID, band=BAND):
    """Mean and band of the posterior-weighted mixture of per-combination predictions; parts = [(weight, mean, quantiles)]."""
    parts = [(w, m, q) for w, m, q in parts if w > NEGLIGIBLE]
    total = sum(w for w, _, _ in parts)
    parts = [(w / total, m, q) for w, m, q in parts]
    mean = sum(w * m for w, m, _ in parts)

    def mixture_cdf(x):
        return sum(w * cdf(grid, q, x - m) for w, m, q in parts)

    def quantile(p):
        lo = min(m + q[0] for _, m, q in parts)
        hi = max(m + q[-1] for _, m, q in parts)
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mixture_cdf(mid) < p else (lo, mid)
        return (lo + hi) / 2

    return mean, quantile(band[0]), quantile(band[1])
