from dataclasses import dataclass
import numpy as np
from . import model_terms, covariance_estimators, recorded_blocks


class EffectModel:
    """E1 as the filter and the belief read it, one row per hypothesis: the mean table (combinations, d), the within and block
    covariances (combinations, d, d), or one (d, d) matrix standing for every hypothesis as the synthetic models of the tests
    give it, the resampled mean tables of E6, the coefficients each mean comes from (E4) and the healthy reference of E3.
    mu_rows gives the means of several combinations at once for the filter."""

    def __init__(self, space, pairs, mu_table, sigma_within, sigma_block, healthy_reference, healthy_scale, coefficients=None, resampler=None):
        self.space = space
        self.pairs = [tuple(p) for p in pairs]
        self.mu_table = np.asarray(mu_table, dtype=float)
        self.sigma_within = np.asarray(sigma_within, dtype=float)
        self.sigma_block = np.asarray(sigma_block, dtype=float)
        self.healthy_reference = np.asarray(healthy_reference, dtype=float)
        self.healthy_scale = np.asarray(healthy_scale, dtype=float)
        self._coefficients = coefficients
        self._resampler = resampler
        self._resampled = []

    @property
    def fields(self):
        return self.space.fields

    @property
    def resample_count(self):
        return 0 if self._resampler is None else self._resampler.count

    def tables(self, n):
        """(n + 1, combinations, d): the mean table and the first n resampled tables of E6, drawn once in a fixed order."""
        for r in range(len(self._resampled), n):
            self._resampled.append(self._resampler.table(r))
        return np.concatenate([self.mu_table[None], np.array(self._resampled[:n])]) if n else self.mu_table[None]

    def mu(self, combo):
        return self.mu_table[self.space.combo_index[tuple(combo)]]

    def mu_rows(self, combos):
        """(n, d) means of the given combinations."""
        return self.mu_table[[self.space.combo_index[tuple(h)] for h in combos]]

    def component_terms(self, combo):
        """E4: per component, its contribution to the predicted deviation of every summary under combo, from the coefficients
        of the fit that scores the combination: its main effect plus half of every interaction it takes part in. Components at
        their healthy level contribute nothing."""
        combo = tuple(combo)
        beta = self._coefficients(combo)
        terms = np.zeros((len(self.space.components), len(self.space.fields)))
        column = 1
        for j, level in self.space.main_terms:
            if combo[j] == level:
                terms[j] += beta[column]
            column += 1
        for (j, k), lj, lk in model_terms.terms_of(self.space, self.pairs):
            if combo[j] == lj and combo[k] == lk:
                terms[j] += beta[column] / 2
                terms[k] += beta[column] / 2
            column += 1
        return terms

    def anomaly_scores(self, values):
        """E3: |x_k - reference_k| / scale_k for the summaries present in values (a dict field -> value), the reference being the
        healthy combination's mean and the scale its within standard deviation, both from the fit that scores that combination."""
        index = self.space.field_index
        return {f: float(abs(v - self.healthy_reference[index[f]]) / self.healthy_scale[index[f]]) for f, v in values.items() if f in index}


class Resampler:
    """The resampled mean tables of E6 under the rule of the means: resample r draws the pool's blocks with replacement, and
    every row is the prediction of its combination by the fit on the drawn blocks other than that combination's own copies, or,
    for a combination without a block in the pool, other than the copies of the second block (position second in the pool).
    grid_rows gives the place of every block of the pool in the grid. Drawn again until the drawn blocks identify the model
    without any one of them, with no count of attempts, in one order from the seed. The redraws end with probability 1 because
    the draw that takes every block once has probability n!/n^n > 0 among the n^n draws of n blocks, and the constructor
    refuses a pool for which that draw fails (a design short of full rank, or a block whose leverage reaches 1 - LEVERAGE_GUARD,
    so that it alone supports a term): every accepted pool has at least one accepting draw, and the draws are independent.
    redraws counts them, fixed per seed."""

    def __init__(self, design, grid, means, counts, rows, grid_rows, second, count, seed):
        self.design, self.grid, self.means, self.counts, self.rows = design, grid, means, counts, rows
        self.grid_rows, self.second = grid_rows, second
        self.count, self._rng, self._draws = count, np.random.default_rng(seed), []
        self.redraws = 0  # the draws that did not identify the model and were drawn again
        if self._accepted(np.ones(len(rows))) is None:
            raise ValueError("the pool's blocks taken once each do not identify the model without any one of them, so no "
                             "resample would ever be accepted")

    def _accepted(self, multiplicity):
        """(w, gram_inverse, own) of a draw whose blocks identify the model without any one of them, else None."""
        x = self.design[self.rows]
        if not model_terms.full_rank(x[multiplicity > 0]):
            return None
        w = self.counts[self.rows] * multiplicity
        gram_inverse = np.linalg.inv((x.T * w) @ x)
        own = w * np.einsum("ij,jk,ik->i", x, gram_inverse, x)
        return (w, gram_inverse, own) if np.all(own < 1 - recorded_blocks.LEVERAGE_GUARD) else None

    def _draw(self, r):
        while len(self._draws) <= r:
            while True:
                multiplicity = np.bincount(self._rng.integers(0, len(self.rows), len(self.rows)), minlength=len(self.rows)).astype(float)
                accepted = self._accepted(multiplicity)
                if accepted is not None:
                    self._draws.append((multiplicity, *accepted))
                    break
                self.redraws += 1
        return self._draws[r]

    def table(self, r):
        multiplicity, w, gram_inverse, own = self._draw(r)
        x, means = self.design[self.rows], self.means[self.rows]
        beta = gram_inverse @ ((x.T * w) @ means)
        fitted = x @ beta
        b = self.second
        without_second = beta - np.outer(gram_inverse @ (x[b] * w[b]), means[b] - fitted[b]) / (1 - own[b])
        table = self.grid @ without_second
        table[self.grid_rows] = fitted - (own / (1 - own))[:, None] * (means - fitted)
        return table


@dataclass(eq=False)
class EpisodeFit:
    """The fits of one episode whose start's block is left out: the effect model whose hypothesis h
    is scored by the fit without the start's block and h's own, or without the second block for a hypothesis without a block
    of its own, the within standard deviation of the pool, which scales the conformal scores, the calibration scores of the
    pool's cycles against the means of the fits without their own block, with one unit of weight per block, the pool, and
    second, the combination of the second block."""
    effect: EffectModel
    sigma: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    n_blocks: int
    pool: tuple
    second: tuple


def fit_episode(recorded, pairs, left_out, resamples, seed, covariance=covariance_estimators.DEFAULT_COVARIANCE, also_left_out=()):
    """The fits of an episode that started in the combination left_out, whose block every fit leaves out, with the blocks of
    also_left_out, none by default. The pool is every other recorded block. Every hypothesis with a block in the pool is scored
    by the pool's fit without its own block, and every other hypothesis, the start and any combination without a recording, by
    the pool's fit without the second block (second_block), so each is scored by a fit that leaves out the same number of
    blocks. The within covariance of a hypothesis pools the same blocks as its mean, by the estimator covariance of the
    start's form, and its block covariance is the covariance of the leave one block out misfits among them, by
    block_covariance_estimator of it."""
    space = recorded.space
    design = model_terms.design_matrix(space, recorded.combos, pairs)
    grid = model_terms.design_matrix(space, space.combinations, pairs)
    rows = recorded.pool([left_out, *also_left_out])
    fit = recorded_blocks.fit_pool(design, recorded.means, recorded.counts, rows)
    n, blocks = len(space.combinations), len(rows)
    pool = tuple(recorded.combos[i] for i in rows)
    pool_sums = recorded.sums(rows)
    # the fit of the pool and the fit of every hypothesis, which leaves one more block of the pool out
    covariance_estimators.refuse_constant_summaries(space, pool_sums.varying[None, :] - recorded.varies[rows])
    pool_within = covariance_estimators.one_cycle_covariance(pool_sums, covariance, recorded.window)
    second = recorded_blocks.second_block(left_out, pool)
    b = pool.index(second)

    def without(j):
        """The coefficients of the pool's fit less the block at position j of the pool."""
        z = fit.x[j] * recorded.counts[rows][j]
        return fit.beta - np.outer(fit.gram_inverse @ z, fit.residuals[j]) / (1 - fit.leverage[j])

    def within_without(j):
        return covariance_estimators.one_cycle_covariance(pool_sums.minus(recorded.sums(rows[j])), covariance, recorded.window)

    def block_without(j):
        return covariance_estimators.covariance_of_rows(recorded_blocks.leave_two_out_misfits(fit, recorded.counts, j), blocks - 1, covariance_estimators.block_covariance_estimator(covariance))

    mu_table = grid @ without(b)
    sigma_within = np.repeat(within_without(b)[None], n, axis=0)
    sigma_block = np.repeat(block_without(b)[None], n, axis=0)
    loo = recorded_blocks.leave_one_out(fit, recorded.means)
    grid_rows = np.array([space.combo_index[h] for h in pool], dtype=int)
    for j, h in enumerate(grid_rows):
        mu_table[h] = loo[j]
        sigma_within[h] = within_without(j)
        sigma_block[h] = block_without(j)
    position = {h: j for j, h in enumerate(pool)}

    def coefficients(combo):
        """The coefficients of the fit that scores the combination: the pool's fit less the combination's own block, or less
        the second block for a combination without a block in the pool."""
        return without(position.get(tuple(combo), b))

    healthy = space.combo_index[space.healthy_combination]
    effect = EffectModel(space, pairs, mu_table, sigma_within, sigma_block, mu_table[healthy], np.sqrt(np.diag(sigma_within[healthy])),
                         coefficients, Resampler(design, grid, recorded.means, recorded.counts, rows, grid_rows, b, resamples, seed))
    sd = np.sqrt(np.diag(pool_within))
    scores = np.concatenate([np.abs(recorded.arrays[i] - loo[j]) / sd for j, i in enumerate(rows)])
    weights = np.concatenate([np.full(len(recorded.arrays[i]), 1.0 / len(recorded.arrays[i])) for i in rows])
    return EpisodeFit(effect, sd, scores, weights, blocks, pool, second)
