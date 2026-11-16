import hashlib
from dataclasses import dataclass
import numpy as np
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import combo_key  # the one key of a combination, which the label table and models.json share
from . import model_terms, covariance_estimators


LEAVE_OUT_SEED = 0  # keys the second block that the fit of a hypothesis without a block of its own leaves out
LEVERAGE_GUARD = 1e-9  # a block whose leverage, or a pair of blocks whose joint deletion, comes within this of one alone supports some term


class PoolData:
    """The fitted blocks as arrays: the combinations, their block arrays of the summaries, block means and lengths."""

    def __init__(self, space, blocks, combos):
        self.pool = [tuple(h) for h in combos]
        self.arrays = [np.asarray(blocks[h], dtype=float) for h in self.pool]
        self.means = np.array([x.mean(axis=0) for x in self.arrays])
        self.counts = np.array([len(x) for x in self.arrays], dtype=float)
        self.combos = np.array(self.pool)
        self.varies = covariance_estimators.varies_within(self.arrays)


def second_block(start, pool):
    """The block that the fit of a hypothesis without a block of its own in the pool leaves out besides the start's: among the
    combinations of pool that differ from the start in the level of exactly one component, or among all of pool when it holds
    none of them, the one with the smallest SHA-256 of LEAVE_OUT_SEED, the start's levels and its own levels.

    A near rival of the start, one component away, is scored by a fit without the start's block, which is one of its own
    neighbors, so the fit of the start's own hypothesis leaves out a neighbor of the start as well. The choice depends on the
    start and on the recorded combinations alone, not on their order in a grid or a file, nor on the process."""
    def key(h):
        return hashlib.sha256(f"{LEAVE_OUT_SEED}|{combo_key(start)}|{combo_key(h)}".encode()).digest()
    pool = [tuple(h) for h in pool]
    neighbors = [h for h in pool if sum(a != b for a, b in zip(h, start)) == 1]
    return min(neighbors or pool, key=key)


# ----- the recorded blocks and the fits of one episode -----
class RecordedBlocks:
    """The recorded blocks as the fits read them, in grid order: per combination its stable cycles, their mean and their number,
    and the sums over its cycles of the centered cross products and of their squares, so that the within covariance of any set
    of blocks is a sum over the blocks and a block leaves it by a subtraction. With a window width it keeps the same sums for
    the window means of every block, which the estimator window_means reads. It also keeps, per block, whether each summary
    takes more than one value there."""

    def __init__(self, space, blocks, window=None):
        self.space, self.window = space, window
        self.combos = tuple(h for h in space.combinations if h in blocks)
        self.index = {h: i for i, h in enumerate(self.combos)}
        self.arrays = [np.asarray(blocks[h], dtype=float) for h in self.combos]
        self.means = np.array([x.mean(axis=0) for x in self.arrays])
        self.counts = np.array([len(x) for x in self.arrays], dtype=float)
        deviations = [x - x.mean(axis=0) for x in self.arrays]
        self.cross = np.array([dv.T @ dv for dv in deviations])
        self.squares = np.array([(dv ** 2).T @ (dv ** 2) for dv in deviations])
        of_windows = [covariance_estimators.window_means(x, window) if window else np.zeros((0, x.shape[1])) for x in self.arrays]
        window_deviations = [w - w.mean(axis=0) if len(w) else w for w in of_windows]
        self.window_cross = np.array([dv.T @ dv for dv in window_deviations])
        self.window_squares = np.array([(dv ** 2).T @ (dv ** 2) for dv in window_deviations])
        self.windows = np.array([len(dv) for dv in window_deviations], dtype=float)
        self.varies = covariance_estimators.varies_within(self.arrays)

    def sums(self, rows):
        """The BlockSums of the blocks at the given indices, one index or several."""
        rows = np.atleast_1d(rows)
        return covariance_estimators.BlockSums(self.cross[rows].sum(axis=0), self.squares[rows].sum(axis=0), float(self.counts[rows].sum()), len(rows),
                         self.window_cross[rows].sum(axis=0), self.window_squares[rows].sum(axis=0), float(self.windows[rows].sum()),
                         float(np.maximum(self.windows[rows] - 1, 0).sum()), self.varies[rows].sum(axis=0))

    def pool(self, left_out):
        """The indices of the recorded combinations outside the left out set."""
        out = {tuple(h) for h in left_out}
        return np.array([i for i, h in enumerate(self.combos) if h not in out], dtype=int)


@dataclass(frozen=True, eq=False)
class PoolFit:
    """The weighted least squares fit on the blocks of a pool, with what the deletion formula needs: the coefficients, the
    inverse of the weighted Gram matrix, the residuals and the leverages of the pool's blocks."""
    rows: np.ndarray
    x: np.ndarray
    gram_inverse: np.ndarray
    beta: np.ndarray
    residuals: np.ndarray
    leverage: np.ndarray


def fit_pool(design, means, counts, rows):
    """The fit on the blocks of rows. A design of the blocks that is not of full column rank holds a term that no block of
    the pool identifies, and a block whose leverage reaches one alone identifies a term, so the fit without it, which every
    hypothesis needs, does not exist. The model is refused in both cases, before any inverse."""
    x, w = design[rows], counts[rows]
    if not model_terms.full_rank(x):
        raise ValueError("the recorded blocks of the fit do not identify every term of the model")
    gram_inverse = np.linalg.inv((x.T * w) @ x)
    beta = gram_inverse @ ((x.T * w) @ means[rows])
    leverage = w * np.einsum("ij,jk,ik->i", x, gram_inverse, x)
    if np.any(leverage >= 1 - LEVERAGE_GUARD):
        raise ValueError("a recorded block alone identifies a term of the model, so no fit without that block identifies it")
    return PoolFit(rows, x, gram_inverse, beta, means[rows] - x @ beta, leverage)


def leave_one_out(fit, means):
    """The prediction of every block of the pool by the fit on the other blocks, from the one fit: the block's mean less its
    residual over one minus its leverage, which equals refitting without the block."""
    return means[fit.rows] - fit.residuals / (1 - fit.leverage)[:, None]


def leave_two_out_misfits(fit, counts, j):
    """The misfit of every other block of the pool against the fit without that block and block j (position j in the pool),
    from the one fit by the deletion of two rows: the deleted residuals of a pair of rows are (I - H_pair)^-1 times their
    weighted residuals, H the hat matrix of the weighted fit. Rows of the result follow the pool without j."""
    w = counts[fit.rows]
    root = np.sqrt(w)
    z = fit.x * root[:, None]
    hat_j = z @ (fit.gram_inverse @ z[j])
    e = fit.residuals * root[:, None]
    determinant = (1 - fit.leverage) * (1 - fit.leverage[j]) - hat_j ** 2
    keep = np.arange(len(fit.rows)) != j
    if np.any(determinant[keep] <= LEVERAGE_GUARD):
        raise ValueError("two recorded blocks alone identify a term of the model, so the fit without both, which the block covariance "
                         "of a hypothesis needs, does not exist")
    deleted = ((1 - fit.leverage[j]) * e[keep] + hat_j[keep, None] * e[j]) / determinant[keep, None]
    return deleted / root[keep, None]
