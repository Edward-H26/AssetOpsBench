"""E14, E16 and E17 of design v2: split conformal intervals calibrated by block, conformal p values and the family wise
verdict (docs/hydraulic_rig/causal-diagnosis-server.md).

The calibration set holds the held out pool cycles of the cross fit with one unit of weight per block shared by the
block's cycles, so that the exchangeability behind the coverage guarantee is across blocks rather than across the
correlated cycles of one block. The nonconformity score of a cycle is the absolute residual of a summary over its
within standard deviation. The only constant is the declared error level alpha, printed in every result.
"""
import numpy as np

DEFAULT_ALPHA = 0.05  # the declared error level of the intervals and of the verdict; an error level, not a fitted threshold


class Calibration:
    """Per summary, the sorted held out scores with their block weights and cumulative weights."""

    def __init__(self, scores, weights, n_blocks):
        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(scores, axis=0, kind="stable")
        self.scores = np.take_along_axis(scores, order, axis=0)
        self.weights = weights[order]
        self.cumulative = np.cumsum(self.weights, axis=0)
        self.n_blocks = n_blocks

    def quantile(self, column, alpha):
        """E14: the smallest score with at least (1 - alpha)(n + 1) units of calibration weight at or below it, n the number
        of blocks. Returns the score and whether the target exceeded the calibration set, in which case the largest score
        stands in and the interval carries no guarantee."""
        target = (1 - alpha) * (self.n_blocks + 1)
        i = int(np.searchsorted(self.cumulative[:, column], target - 1e-9, side="left"))
        capped = i >= len(self.scores)
        return float(self.scores[min(i, len(self.scores) - 1), column]), capped

    def p_value(self, column, score):
        """E16: (1 + weight of calibration scores at or above the score) / (n + 1)."""
        mass = float(self.weights[self.scores[:, column] >= score, column].sum())
        return (1 + mass) / (self.n_blocks + 1)


def interval(mean, sigma, quantile):
    return mean - quantile * sigma, mean + quantile * sigma


def verdict(p_values, alpha):
    """E17: Bonferroni rejects a channel when p <= alpha / K, which bounds the family wise error under any dependence;
    Holm steps down from the smallest p value with alpha / (K - i) and rejects a superset of Bonferroni's channels. The
    outcome is inconsistent when the smallest p value is at or below alpha / K."""
    k = len(p_values)
    if k == 0:
        return {"consistent": None, "bonferroni": {}, "holm": {}, "channels": k}
    bonferroni = {f: p <= alpha / k for f, p in p_values.items()}
    holm = {f: False for f in p_values}
    for i, f in enumerate(sorted(p_values, key=lambda f: (p_values[f], f))):
        if p_values[f] <= alpha / (k - i):
            holm[f] = True
        else:
            break
    return {"consistent": not any(bonferroni.values()), "bonferroni": bonferroni, "holm": holm, "channels": k}
