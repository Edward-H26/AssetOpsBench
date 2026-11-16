"""E14, E16 and E17: the intervals and the verdict of the post repair check by a leave one block out conformal calibration
(CausalDiagnosis/docs/causal-diagnosis-mcp.md).

The calibration set holds the cycles of every recorded block but the episode's start's, each scored against the mean of
the fit that left that block out as well, with one unit of weight per block shared by the block's cycles, so
that the exchangeability behind the guarantee is across blocks rather than across the correlated cycles of one block. No split is
held out besides the scored block, so the calibration leaves one block out at a time. The nonconformity score of a summary
is the absolute residual over its within standard deviation, which gives the interval of E14 per summary. The post repair
check of E16 and E17 uses one combined score per reading, the largest standardized deviation over the summaries the reading
holds, and one conformal p value against the same combined score of the calibration cycles, since a per
summary p value cannot fall below 1 / (n + 1) and a Bonferroni cutoff of alpha / K with K summaries lies below that floor on
this rig. Both score a reading against the most probable combination of the belief after an action, from its fit.

The promise is a rate over the calibration blocks. For a block drawn at random among them and a cycle drawn at random in
it, the verdict finds the reading inconsistent at a rate of at most alpha, and each summary lies in its interval at a rate
of at least 1 - alpha, when the level is reachable (Calibration.level_reachable). Nothing holds for one named combination,
and a reading of the block the episode started in, which a monitor prediction scores, is scored by a fit that holds one
block more than the fits of the calibration. The only constant is the declared error level alpha, printed in every result.
"""
import math

import numpy as np

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import PredictedSummary

DEFAULT_ALPHA = 0.05  # the declared error level of the intervals and of the verdict; an error level, not a fitted threshold


class Calibration:
    """The held out scores of the calibration cycles (cycles, summaries) with their block weights: per summary the sorted
    scores with cumulative weights for the quantile of E14, and the rows as scored for the combined score of E16."""

    def __init__(self, scores, weights, n_blocks):
        self.rows = np.asarray(scores, dtype=float)
        self.row_weights = np.asarray(weights, dtype=float)
        order = np.argsort(self.rows, axis=0, kind="stable")
        self.scores = np.take_along_axis(self.rows, order, axis=0)
        self.weights = self.row_weights[order]
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

    def level_reachable(self, alpha):
        """Whether the declared level can be carried by the n calibration blocks: (1 - alpha)(n + 1) is at most n, within the
        rounding that quantile allows. Below that the quantile is the largest score and the smallest p value lies above alpha,
        so no interval carries its level and no reading can be found inconsistent."""
        return (1 - alpha) * (self.n_blocks + 1) - 1e-9 <= self.n_blocks

    def combined_p_value(self, columns, scores, keep=None):
        """E16: the combined score of a reading, the largest standardized deviation over the columns it holds, and its conformal
        p value, (1 + weight of the calibration cycles whose combined score over the same columns is at or above it) / (n + 1).
        keep, a boolean per calibration cycle, restricts the calibration set, which the level test uses to leave a block out."""
        columns = np.asarray(columns, dtype=int)
        combined = float(np.max(np.asarray(scores, dtype=float)))
        reference = self.rows[:, columns].max(axis=1)
        weights = self.row_weights if keep is None else np.where(keep, self.row_weights, 0.0)
        blocks = self.n_blocks if keep is None else self.n_blocks - 1
        return combined, (1 + float(weights[reference >= combined].sum())) / (blocks + 1)


def interval(mean, sigma, quantile):
    return mean - quantile * sigma, mean + quantile * sigma


def most_probable(belief, space):
    """The most probable combination of a belief {combination: mass} and its mass, a tie going to the combination that comes first
    in the candidate order of the space."""
    combination = min(belief, key=lambda h: (-belief[h], space.combo_index[tuple(h)]))
    return tuple(combination), float(belief[combination])


def predict_summaries(models, after, alpha):
    """E14 for every summary under a belief over combinations: the mean of its most probable combination, from the
    fit that scores that combination, and the conformal interval at alpha, shared by the intervention tool, the troubleshooting
    plans and the offline table. Where the level cannot be reached the quantile is the largest score, which the results say
    through Calibration.level_reachable, the same fact for every summary."""
    means = models.effect.mu(most_probable(after, models.effect.space)[0])
    out = []
    for k, field in enumerate(models.effect.fields):
        q = models.calibration.quantile(k, alpha)[0]
        low, high = interval(float(means[k]), float(models.sigma[k]), q)
        out.append(PredictedSummary(field=field, mean=float(means[k]), low=low, high=high, quantile=q))
    return out


def fits_for_level(alpha):
    """The fewest resampled fits whose range stands for the declared level: the range of R fits holds a further fit with
    probability (R - 1) / (R + 1), which reaches 1 - alpha when R alpha is at least 2 - alpha, within the rounding of the
    division, so 39 fits at alpha 0.05."""
    return math.ceil((2 - alpha) / alpha - 1e-9)


def verdict(p_value, alpha):
    """E17: the outcome is inconsistent when the conformal p value of the combined score is at or below alpha; one test per
    reading, so no correction over summaries is needed."""
    return p_value > alpha
