from dataclasses import dataclass
import numpy as np


# the estimators of the covariance of one cycle, registered before any fit, the estimator that the fits used before the other
# two first, so that it wins a tie and stands as the fallback, and the block effects whose held out density every form reports
COVARIANCE_ESTIMATORS = ("ledoit_wolf_diagonal", "diagonal", "window_means")
DEFAULT_COVARIANCE = COVARIANCE_ESTIMATORS[0]


def shrunk_toward_diagonal(cross, squares, n, dof):
    """The sample covariance cross / dof shrunk toward its diagonal with the shrinkage weight of Ledoit and Wolf as Schaefer and
    Strimmer 2005 write it for a diagonal target: the summed sampling variance of the off diagonal entries over their summed
    squares, clipped to [0, 1]. cross is the sum over n centered rows of their outer products and squares the sum of the squares
    of those products, so that the covariance of a set of blocks is a sum over the blocks. The sampling variance is estimated as if
    the n rows were independent. Deviations pooled within blocks hold fewer degrees of freedom than rows, so for them the weight
    is smaller than a weight of that form whose variance is taken over blocks. Returns the matrix and the weight."""
    d = cross.shape[0]
    sample = cross / dof
    variance = (n / (n - 1) ** 3) * (squares - n * (cross / n) ** 2)
    off = ~np.eye(d, dtype=bool)
    denominator = (sample[off] ** 2).sum()
    weight = float(min(1.0, max(0.0, variance[off].sum() / denominator))) if denominator > 0 else 1.0
    return (1 - weight) * sample + weight * np.diag(np.diag(sample)), weight


def covariance_from_sums(cross, squares, n, dof, estimator):
    """The covariance of n centered rows from the sum of their outer products (cross) and the sum of the squares of those
    products (squares), under ledoit_wolf_diagonal, the sample covariance shrunk toward its diagonal, or diagonal, the variances
    alone. Rows that leave no degree of freedom are refused."""
    if n < 2 or dof <= 0:
        raise ValueError("a covariance needs at least two rows and one degree of freedom")
    if estimator == "ledoit_wolf_diagonal":
        return shrunk_toward_diagonal(cross, squares, n, dof)[0]
    if estimator == "diagonal":
        return np.diag(np.diag(cross) / dof)
    raise ValueError(f"the covariance of rows is ledoit_wolf_diagonal or diagonal, got {estimator!r}")


def covariance_of_rows(deviations, dof, estimator):
    """covariance_from_sums of centered rows."""
    products = deviations[:, :, None] * deviations[:, None, :]
    return covariance_from_sums(products.sum(axis=0), (products ** 2).sum(axis=0), len(deviations), dof, estimator)


def window_means(x, window):
    """The means of the consecutive windows of `window` cycles of a block in recording order, which do not overlap. A
    remainder shorter than a window is left out."""
    x = np.asarray(x, dtype=float)
    m = len(x) // window
    return x[: m * window].reshape(m, window, x.shape[1]).mean(axis=1)


@dataclass(frozen=True, eq=False)
class BlockSums:
    """What the covariance of one cycle of a set of blocks is computed from. Every term is a sum over the blocks, so a block
    leaves the set by a subtraction: the cross products of the cycles around their block's mean and the squares of those
    products, the cycles and the blocks, and the same two sums for the window means around their block's mean, with their
    number and their degrees of freedom, the windows of a block less one, and per summary the count of the blocks in which it
    takes more than one value."""
    cross: np.ndarray
    squares: np.ndarray
    cycles: float
    blocks: int
    window_cross: np.ndarray
    window_squares: np.ndarray
    windows: float
    window_dof: float
    varying: np.ndarray

    def minus(self, other):
        return BlockSums(*(getattr(self, name) - getattr(other, name) for name in self.__dataclass_fields__))


def one_cycle_covariance(sums, estimator, window=None):
    """The covariance of one cycle of a set of blocks under a registered estimator. ledoit_wolf_diagonal
    and diagonal estimate it from the cycles around their block's mean. window_means estimates it as the window width times
    the within block covariance of the window means, shrunk toward its diagonal as the first, which equals the covariance of
    one cycle when the cycles of a block are independent and grows with any drift within a block that a window does not
    average out."""
    if estimator == "window_means":
        if window is None:
            raise ValueError("window_means needs the width of the window")
        if sums.windows < 2 or sums.window_dof <= 0:
            raise ValueError("window_means needs a block of at least two windows")
        return window * shrunk_toward_diagonal(sums.window_cross, sums.window_squares, sums.windows, sums.window_dof)[0]
    if estimator in COVARIANCE_ESTIMATORS:
        return covariance_from_sums(sums.cross, sums.squares, sums.cycles, sums.cycles - sums.blocks, estimator)
    raise ValueError(f"the covariance estimator must be one of {COVARIANCE_ESTIMATORS}, got {estimator!r}")


def block_covariance_estimator(estimator):
    """The estimator of the block covariance under an estimator of the covariance of one cycle: the same one, or
    ledoit_wolf_diagonal under window_means, since the mean of a block has no windows."""
    return "ledoit_wolf_diagonal" if estimator == "window_means" else estimator


def ledoit_wolf_diagonal(deviations, dof):
    """The sample covariance of centered rows shrunk toward its diagonal by the Ledoit and Wolf weight (shrunk_toward_diagonal).
    Returns the shrunk matrix and the weight."""
    products = deviations[:, :, None] * deviations[:, None, :]
    return shrunk_toward_diagonal(products.sum(axis=0), (products ** 2).sum(axis=0), len(deviations), dof)


def varies_within(arrays):
    """(blocks, summaries): whether each summary takes more than one value within each block, read from the recorded values
    exactly, so that no tolerance decides it."""
    return np.array([(np.asarray(x) != np.asarray(x)[0]).any(axis=0) for x in arrays])


def refuse_constant_summaries(space, varying):
    """Refuse, by name, every summary that takes one value within every block of a fit: varying counts per summary the blocks
    of the fit in which it takes more than one value, one row of counts per fit or one count vector. Such a summary has no
    within variance, so the fit holds no density of a reading of it, whichever estimator would build the covariance."""
    constant = np.nonzero((np.atleast_2d(varying) == 0).any(axis=0))[0]
    if len(constant):
        raise ValueError(f"the summaries {', '.join(space.fields[i] for i in constant)} take one value within every recorded block of a "
                         "fit, so the fit has no within variance for them")


def within_covariance(block_arrays):
    """Pooled within combination covariance: every block centered on its own mean, degrees of freedom cycles minus blocks."""
    deviations = np.concatenate([x - x.mean(axis=0) for x in block_arrays])
    return ledoit_wolf_diagonal(deviations, deviations.shape[0] - len(block_arrays))
