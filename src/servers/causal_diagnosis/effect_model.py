"""The interventional effect model E1 of design v2 (docs/hydraulic_rig/causal-diagnosis-server.md).

The experimenters set every component level, so the recordings under a combination are draws from the interventional
distribution of the channel summaries under that combination, and the effect of a component condition on a channel is
estimated from the recordings without causal discovery. The mean of each of the 34 summaries (the mean and the standard
deviation of the 17 channels over one recorded cycle) is an intercept plus one main effect per non healthy level of each
component plus the pairwise interactions that lower the cross fitted log likelihood over the fitting pool. The residual
covariance is the pooled within combination covariance over the pool, shrunk toward its diagonal by the Ledoit and Wolf
rule. The model is refit on resamples of the pool drawn by recorded block for the stability of E6, and the cross fitted
block means give the misfit covariance of the hierarchical variant and the calibration scores of the conformal intervals.
"""
import math

import numpy as np

from envs.hydraulic_rig.classifiers import COMBOS, stratified_folds
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS, TIER_SENSORS, tier_features
from servers.hydraulic_rig.names import HYD1_FIELDS, public_name

STATISTICS = ("mean", "std")
SUMMARY_FIELDS = tuple(sorted(f"{HYD1_FIELDS[s]}_{stat}" for s in SENSORS for stat in STATISTICS))
FIELD_INDEX = {field: i for i, field in enumerate(SUMMARY_FIELDS)}
HEALTHY_COMBO = tuple(HEALTHY[c] for c in COMPONENTS)
COMBO_INDEX = {combo: i for i, combo in enumerate(COMBOS)}
MAIN_TERMS = tuple((j, level) for j, c in enumerate(COMPONENTS) for level in LEVELS[c] if level != HEALTHY[c])
PAIRS = tuple((j, k) for j in range(len(COMPONENTS)) for k in range(j + 1, len(COMPONENTS)))
FOLDS, FOLD_SEED = 8, 0
RESAMPLE_ATTEMPTS = 100  # a block resample that drops a level cannot identify its effect and is drawn again


def channel_field(summary_field):
    """PS1_Pressure_bar_100Hz_mean -> PS1_Pressure_bar_100Hz."""
    return summary_field.rsplit("_", 1)[0]


def statistic_of(summary_field):
    return summary_field.rsplit("_", 1)[1]


GROUP_FIELDS = {f"tier{t}": tuple(f for f in SUMMARY_FIELDS if channel_field(f) in {HYD1_FIELDS[s] for s in sensors})
                for t, sensors in TIER_SENSORS.items()}


def n_faulty(combo):
    return sum(1 for j, c in enumerate(COMPONENTS) if combo[j] != HEALTHY[c])


def field_indices(fields):
    return np.array([FIELD_INDEX[f] for f in fields], dtype=int)


def interaction_terms(pair):
    """One indicator per pair of non healthy levels of the two components."""
    j, k = pair
    return tuple((pair, lj, lk) for lj in LEVELS[COMPONENTS[j]] if lj != HEALTHY[COMPONENTS[j]]
                 for lk in LEVELS[COMPONENTS[k]] if lk != HEALTHY[COMPONENTS[k]])


def terms_of(pairs):
    return tuple(term for pair in pairs for term in interaction_terms(pair))


def design_matrix(combos, pairs):
    """Intercept, main effects of the non healthy levels, then the interaction indicators of the selected pairs."""
    combos = np.asarray(list(combos))
    columns = [np.ones(len(combos))]
    columns += [(combos[:, j] == level).astype(float) for j, level in MAIN_TERMS]
    columns += [((combos[:, j] == lj) & (combos[:, k] == lk)).astype(float) for (j, k), lj, lk in terms_of(pairs)]
    return np.column_stack(columns)


def full_rank(x):
    return np.linalg.matrix_rank(x) == x.shape[1]


def weighted_fit(x, y, weights):
    """Least squares of the block means with the block lengths as weights, which equals the least squares over cycles."""
    xtw = x.T * weights
    return np.linalg.solve(xtw @ x, xtw @ y)


def ledoit_wolf_diagonal(deviations, dof):
    """The sample covariance of centered rows shrunk toward its diagonal, with the shrinkage weight of Ledoit and Wolf as
    Schaefer and Strimmer 2005 write it for a diagonal target: the summed sampling variance of the off diagonal entries over
    their summed squares, clipped to [0, 1]. Returns the shrunk matrix and the weight."""
    n, d = deviations.shape
    sample = deviations.T @ deviations / dof
    products = deviations[:, :, None] * deviations[:, None, :]
    variance = (n / (n - 1) ** 3) * ((products - products.mean(axis=0)) ** 2).sum(axis=0)
    off = ~np.eye(d, dtype=bool)
    denominator = (sample[off] ** 2).sum()
    weight = float(min(1.0, max(0.0, variance[off].sum() / denominator))) if denominator > 0 else 1.0
    return (1 - weight) * sample + weight * np.diag(np.diag(sample)), weight


def within_covariance(block_arrays):
    """Pooled within combination covariance: every block centered on its own mean, degrees of freedom cycles minus blocks."""
    deviations = np.concatenate([x - x.mean(axis=0) for x in block_arrays])
    return ledoit_wolf_diagonal(deviations, deviations.shape[0] - len(block_arrays))


def gaussian_log_density(residuals, cholesky):
    """Log density of rows of residuals under N(0, L L^T); residuals (..., k), cholesky (..., k, k) broadcastable."""
    k = residuals.shape[-1]
    z = np.linalg.solve(cholesky, np.swapaxes(residuals, -1, -2) if residuals.ndim > 1 else residuals[:, None])
    quad = (z ** 2).sum(axis=-2) if residuals.ndim > 1 else float((z ** 2).sum())
    logdet = 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
    return -0.5 * (k * math.log(2 * math.pi) + logdet[..., None] + quad) if residuals.ndim > 1 else -0.5 * (k * math.log(2 * math.pi) + logdet + quad)


class PoolData:
    """The pool as arrays: the combinations, their block arrays of the 34 summaries, block means and lengths."""

    def __init__(self, cycles, blocks, pool):
        self.pool = [tuple(h) for h in pool]
        by_index = {c.index: c for c in cycles}
        self.facts_name = facts_names(next(iter(by_index.values())))
        self.arrays = [self.block_array(by_index, blocks[h]) for h in self.pool]
        self.means = np.array([x.mean(axis=0) for x in self.arrays])
        self.counts = np.array([len(x) for x in self.arrays], dtype=float)
        self.combos = np.array(self.pool)

    def block_array(self, by_index, indices):
        return np.array([[by_index[i].features[self.facts_name[f]] for f in SUMMARY_FIELDS] for i in indices])


def facts_names(cycle):
    """Public summary field -> the feature name of the facts file (Pressure_PS1_mean for PS1_Pressure_bar_100Hz_mean), taken
    from the summaries of the full sensor group as the environment serves them."""
    names = {}
    for name in tier_features(cycle, max(TIER_SENSORS)):
        if name.endswith(tuple(f"_{s}" for s in STATISTICS)):
            names[public_name(name)] = name
    missing = [f for f in SUMMARY_FIELDS if f not in names]
    if missing:
        raise ValueError(f"the facts file lacks the summaries {missing}")
    return names


def cross_fitted_log_likelihood(data, pairs, fold_of):
    """Sum over folds of the log density of the held out blocks' cycles under the fit on the other blocks, with the within
    covariance of the training blocks. Minus infinity when a training design cannot identify the model."""
    total = 0.0
    folds = sorted(set(fold_of.values()))
    for f in folds:
        train = [i for i, h in enumerate(data.pool) if fold_of[h] != f]
        test = [i for i, h in enumerate(data.pool) if fold_of[h] == f]
        if not test:
            continue
        x_train = design_matrix(data.combos[train], pairs)
        if not full_rank(x_train):
            return -math.inf
        beta = weighted_fit(x_train, data.means[train], data.counts[train])
        sigma, _ = within_covariance([data.arrays[i] for i in train])
        cholesky = np.linalg.cholesky(sigma)
        x_test = design_matrix(data.combos[test], pairs)
        for row, i in enumerate(test):
            total += float(gaussian_log_density(data.arrays[i] - x_test[row] @ beta, cholesky).sum())
    return total


def select_interactions(data, fold_of):
    """Forward selection of interaction pairs by the cross fitted log likelihood. A pair is admissible only when the pool
    design stays of full column rank with it, since an unidentified interaction would make the predicted mean of an
    evaluation combination arbitrary. Returns the selected pairs and the selection log."""
    selected, log = [], []
    current = cross_fitted_log_likelihood(data, selected, fold_of)
    log.append({"step": "main effects", "cross_fitted_log_likelihood": current})
    while True:
        best, best_value = None, current
        for pair in PAIRS:
            if pair in selected:
                continue
            candidate = selected + [pair]
            if not full_rank(design_matrix(data.combos, candidate)):
                log.append({"step": f"pair {COMPONENTS[pair[0]]} x {COMPONENTS[pair[1]]}", "admissible": False})
                continue
            value = cross_fitted_log_likelihood(data, candidate, fold_of)
            # minus infinity means a training fold could not identify the candidate, so it is not selectable either
            log.append({"step": f"pair {COMPONENTS[pair[0]]} x {COMPONENTS[pair[1]]}", "admissible": True,
                        "identified_in_every_fold": math.isfinite(value),
                        "cross_fitted_log_likelihood": value if math.isfinite(value) else None})
            if value > best_value:
                best, best_value = pair, value
        if best is None:
            return selected, log
        selected.append(best)
        current = best_value
        log.append({"step": f"selected {COMPONENTS[best[0]]} x {COMPONENTS[best[1]]}", "cross_fitted_log_likelihood": current})


def held_out_block_predictions(data, pairs, fold_of):
    """For every pool block, the predicted means under the fit of the folds that do not hold it."""
    predicted = np.zeros_like(data.means)
    for f in sorted(set(fold_of.values())):
        train = [i for i, h in enumerate(data.pool) if fold_of[h] != f]
        test = [i for i, h in enumerate(data.pool) if fold_of[h] == f]
        beta = weighted_fit(design_matrix(data.combos[train], pairs), data.means[train], data.counts[train])
        predicted[test] = design_matrix(data.combos[test], pairs) @ beta
    return predicted


def fit_effect_model(cycles, blocks, pool, interactions="select", resamples=200, seed=0, folds=FOLDS, fold_seed=FOLD_SEED):
    """E1 on the pool: the selected interactions, the full pool fit, the within covariance, the block misfit covariance
    from the cross fitted block means, the resampled fits drawn by block, the healthy reference of E3 and the conformal
    calibration scores of E14. interactions is "select" or "none"."""
    data = PoolData(cycles, blocks, pool)
    fold_all = stratified_folds(folds, fold_seed)
    fold_of = {h: fold_all[h] for h in data.pool}
    if interactions == "select":
        pairs, selection_log = select_interactions(data, fold_of)
    elif interactions == "none":
        pairs, selection_log = [], [{"step": "main effects", "cross_fitted_log_likelihood": cross_fitted_log_likelihood(data, [], fold_of)}]
    else:
        raise ValueError("interactions must be select or none")
    x_pool = design_matrix(data.combos, pairs)
    if not full_rank(x_pool):
        raise ValueError("the pool design does not identify the main effects")
    beta = weighted_fit(x_pool, data.means, data.counts)
    sigma_within, weight_within = within_covariance(data.arrays)
    held_out = held_out_block_predictions(data, pairs, fold_of)
    block_residuals = data.means - held_out
    sigma_block, weight_block = ledoit_wolf_diagonal(block_residuals, len(data.pool))
    sigma = np.sqrt(np.diag(sigma_within))
    scores = [np.abs(data.arrays[i] - held_out[i]) / sigma for i in range(len(data.pool))]
    calibration = {
        "n_blocks": len(data.pool),
        "scores": np.concatenate(scores),
        # one unit of calibration weight per block, shared by its cycles, so that exchangeability holds across blocks
        "weights": np.concatenate([np.full(len(s), 1.0 / len(s)) for s in scores]),
    }
    healthy = data.arrays[data.pool.index(HEALTHY_COMBO)] if HEALTHY_COMBO in data.pool else None
    if healthy is None:
        raise ValueError("the pool must hold the healthy combination, the reference of the anomaly score")
    median = np.median(healthy, axis=0)
    mad = np.median(np.abs(healthy - median), axis=0)
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(resamples):
        for _attempt in range(RESAMPLE_ATTEMPTS):
            idx = rng.integers(0, len(data.pool), len(data.pool))
            x_b = design_matrix(data.combos[idx], pairs)
            if full_rank(x_b):
                break
        else:
            raise ValueError("no block resample identified the model")
        beta_b = weighted_fit(x_b, data.means[idx], data.counts[idx])
        sigma_b, _ = within_covariance([data.arrays[i] for i in idx])
        drawn.append((beta_b, sigma_b))
    return EffectModel(pairs, beta, sigma_within, sigma_block, drawn, median, mad, calibration,
                       {"pool": [list(h) for h in data.pool], "fold_seed": fold_seed, "folds": folds, "fold_of": {combo_key(h): f for h, f in fold_of.items()},
                        "selection_log": selection_log, "shrinkage_within": weight_within, "shrinkage_block": weight_block,
                        "resample_seed": seed, "resamples": resamples, "sigma_for_scores": sigma.tolist()})


def combo_key(combo):
    return ",".join(str(v) for v in combo)


def parse_combo(key):
    return tuple(int(v) for v in key.split(","))


class EffectModel:
    """The fitted E1: means of the 34 summaries for every combination, the covariances, the resampled fits and the
    healthy reference. mu_rows gives the means of several combinations at once for the posterior."""

    def __init__(self, pairs, beta, sigma_within, sigma_block, resampled, healthy_median, healthy_mad, calibration, provenance):
        self.pairs = [tuple(p) for p in pairs]
        self.beta = np.asarray(beta, dtype=float)
        self.sigma_within = np.asarray(sigma_within, dtype=float)
        self.sigma_block = np.asarray(sigma_block, dtype=float)
        self.resampled = [(np.asarray(b, dtype=float), np.asarray(s, dtype=float)) for b, s in resampled]
        self.healthy_median = np.asarray(healthy_median, dtype=float)
        self.healthy_mad = np.asarray(healthy_mad, dtype=float)
        self.calibration = calibration
        self.provenance = provenance
        self.design_all = design_matrix(COMBOS, self.pairs)
        self.mu_table = self.design_all @ self.beta
        self.resampled_mu = np.array([self.design_all @ b for b, _ in self.resampled]) if self.resampled else np.zeros((0,) + self.mu_table.shape)
        self.resampled_sigma = np.array([s for _, s in self.resampled]) if self.resampled else np.zeros((0,) + self.sigma_within.shape)

    @property
    def fields(self):
        return SUMMARY_FIELDS

    def mu(self, combo):
        return self.mu_table[COMBO_INDEX[tuple(combo)]]

    def mu_rows(self, combos):
        """(n, 34) means of the given combinations under the full fit."""
        return self.mu_table[[COMBO_INDEX[tuple(h)] for h in combos]]

    def resampled_mu_rows(self, combos):
        """(resamples, n, 34) means of the given combinations under every resampled fit."""
        return self.resampled_mu[:, [COMBO_INDEX[tuple(h)] for h in combos], :]

    def sigma_for(self, fields):
        """The block of the within covariance for the given summaries, the Sigma_s of E2."""
        idx = field_indices(fields)
        return self.sigma_within[np.ix_(idx, idx)]

    def component_terms(self, combo):
        """E4: per component, its contribution to the predicted deviation of every summary under combo: its main effect
        plus half of every interaction it takes part in. Components at their healthy level contribute nothing."""
        combo = tuple(combo)
        terms = np.zeros((len(COMPONENTS), len(SUMMARY_FIELDS)))
        column = 1
        for j, level in MAIN_TERMS:
            if combo[j] == level:
                terms[j] += self.beta[column]
            column += 1
        for (j, k), lj, lk in terms_of(self.pairs):
            if combo[j] == lj and combo[k] == lk:
                terms[j] += self.beta[column] / 2
                terms[k] += self.beta[column] / 2
            column += 1
        return terms

    def anomaly_scores(self, values):
        """E3: |x_k - median_k(h0)| / MAD_k(h0) for the summaries present in values (a dict field -> value); a summary
        whose MAD is zero takes the within standard deviation as its scale so that the score stays finite."""
        scale = np.where(self.healthy_mad > 0, self.healthy_mad, np.sqrt(np.diag(self.sigma_within)))
        return {f: float(abs(v - self.healthy_median[FIELD_INDEX[f]]) / scale[FIELD_INDEX[f]]) for f, v in values.items() if f in FIELD_INDEX}

    def to_json(self):
        return {
            "summaries": list(SUMMARY_FIELDS),
            "groups": {g: list(fields) for g, fields in GROUP_FIELDS.items()},
            "main_terms": [[COMPONENTS[j], level] for j, level in MAIN_TERMS],
            "interactions": [[COMPONENTS[j], COMPONENTS[k]] for j, k in self.pairs],
            "interaction_terms": [[COMPONENTS[j], COMPONENTS[k], lj, lk] for (j, k), lj, lk in terms_of(self.pairs)],
            "beta": self.beta.tolist(),
            "sigma_within": self.sigma_within.tolist(),
            "sigma_block": self.sigma_block.tolist(),
            "healthy_median": self.healthy_median.tolist(),
            "healthy_mad": self.healthy_mad.tolist(),
            "calibration": {"n_blocks": self.calibration["n_blocks"], "scores": rounded(self.calibration["scores"]),
                            "weights": rounded(self.calibration["weights"])},
            "resampled": [{"beta": rounded(b), "sigma_within": rounded(s)} for b, s in self.resampled],
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, obj):
        pairs = [(COMPONENTS.index(a), COMPONENTS.index(b)) for a, b in obj["interactions"]]
        resampled = [(r["beta"], r["sigma_within"]) for r in obj["resampled"]]
        calibration = {"n_blocks": obj["calibration"]["n_blocks"], "scores": np.asarray(obj["calibration"]["scores"], dtype=float),
                       "weights": np.asarray(obj["calibration"]["weights"], dtype=float)}
        return cls(pairs, obj["beta"], obj["sigma_within"], obj["sigma_block"], resampled, obj["healthy_median"], obj["healthy_mad"],
                   calibration, obj["provenance"])


def rounded(array):
    """Six significant digits, the precision at which values are served, for the resampled fits and the calibration
    scores, which enter intervals and verdicts but not the posterior of the main fit."""
    array = np.asarray(array)
    if array.ndim == 1:
        return [float(f"{v:.6g}") for v in array]
    return [[float(f"{v:.6g}") for v in row] for row in array]
