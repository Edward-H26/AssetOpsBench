import math
import numpy as np
from . import model_terms, covariance_estimators, recorded_blocks


FOLDS, FOLD_SEED = 8, 0
BLOCK_EFFECTS = ("joint_block", "joint")


def gaussian_log_density(residuals, cholesky):
    """Log density of the rows of residuals (n, k) under N(0, L L^T)."""
    k = residuals.shape[-1]
    z = np.linalg.solve(cholesky, residuals.T)
    logdet = 2 * np.log(np.diagonal(cholesky)).sum()
    return -0.5 * (k * math.log(2 * math.pi) + logdet + (z ** 2).sum(axis=0))


def block_log_density(residuals, sigma_within, sigma_block):
    """Log density of the n rows of one block, residuals (n, k) from their mean, when they share one offset drawn from
    N(0, sigma_block) and vary around it by N(0, sigma_within): the stacked rows are normal with covariance
    I (x) sigma_within + 1 1' (x) sigma_block, whose density splits into the n - 1 contrasts under sigma_within and the mean
    under sigma_within + n sigma_block. A zero sigma_block gives independent rows. The rows are cycles, or window means with
    sigma_within the covariance of one cycle over the window width."""
    n, k = residuals.shape
    mean = residuals.mean(axis=0)
    within = np.linalg.cholesky(sigma_within)
    total = np.linalg.cholesky(sigma_within + n * sigma_block)
    contrasts = np.linalg.solve(within, (residuals - mean).T)
    offset = np.linalg.solve(total, mean)
    logdet = (n - 1) * 2 * np.log(np.diagonal(within)).sum() + 2 * np.log(np.diagonal(total)).sum()
    return float(-0.5 * (n * k * math.log(2 * math.pi) + logdet + (contrasts ** 2).sum() + n * (offset ** 2).sum()))


def window_scatter_log_density(scatter, m, scale):
    """The log density, up to a term that no scale changes, of the scatter of m window means around their own mean when each
    window mean varies by N(0, scale): -((m - 1) log det(scale) + tr(scale^-1 scatter)) / 2."""
    cholesky = np.linalg.cholesky(scale)
    logdet = 2 * np.log(np.diagonal(cholesky)).sum()
    return float(-0.5 * ((m - 1) * logdet + np.trace(np.linalg.solve(scale, scatter))))


def fold_fit(space, data, pairs, train):
    """The coefficients fitted on the blocks of the given indices, or None when they cannot identify the model."""
    x = model_terms.design_matrix(space, data.combos[train], pairs)
    if not model_terms.full_rank(x):
        return None
    return model_terms.weighted_fit(x, data.means[train], data.counts[train])


def cross_fitted_log_likelihood(space, data, pairs, fold_of):
    """Sum over folds of the log density of the held out blocks' cycles under the fit on the other blocks, with the within
    covariance of the training blocks. Minus infinity when a training design cannot identify the model."""
    total = 0.0
    for f in sorted(set(fold_of[h] for h in data.pool)):
        train = [i for i, h in enumerate(data.pool) if fold_of[h] != f]
        test = [i for i, h in enumerate(data.pool) if fold_of[h] == f]
        beta = fold_fit(space, data, pairs, train)
        if beta is None:
            return -math.inf
        covariance_estimators.refuse_constant_summaries(space, data.varies[train].sum(axis=0))
        sigma, _ = covariance_estimators.within_covariance([data.arrays[i] for i in train])
        cholesky = np.linalg.cholesky(sigma)
        x_test = model_terms.design_matrix(space, data.combos[test], pairs)
        for row, i in enumerate(test):
            total += float(gaussian_log_density(data.arrays[i] - x_test[row] @ beta, cholesky).sum())
    return total


def select_interactions(space, data, fold_of):
    """Forward selection of interaction pairs by the cross fitted log likelihood. Returns the selected pairs and the log."""
    return forward_selection(space, data.combos, lambda pairs: cross_fitted_log_likelihood(space, data, pairs, fold_of))


def forward_selection(space, combos, log_likelihood):
    """Forward selection of interaction pairs by a held out log likelihood of the pairs. A pair is admissible only when the
    design of the fitted blocks stays of full column rank with it, since an unidentified interaction would make the predicted
    mean of a combination arbitrary. Returns the selected pairs and the selection log."""
    selected, log = [], []
    current = log_likelihood(selected)
    log.append({"step": "main effects", "held_out_log_likelihood": current})
    while True:
        best, best_value = None, current
        for pair in space.pairs:
            if pair in selected:
                continue
            candidate = selected + [pair]
            name = f"pair {space.components[pair[0]]} x {space.components[pair[1]]}"
            if not model_terms.full_rank(model_terms.design_matrix(space, combos, candidate)):
                log.append({"step": name, "admissible": False})
                continue
            value = log_likelihood(candidate)
            # minus infinity means a training fold could not identify the candidate, so it is not selectable either
            log.append({"step": name, "admissible": True, "identified_in_every_fold": math.isfinite(value),
                        "held_out_log_likelihood": value if math.isfinite(value) else None})
            if value > best_value:
                best, best_value = pair, value
        if best is None:
            return selected, log
        selected.append(best)
        current = best_value
        log.append({"step": f"selected {space.components[best[0]]} x {space.components[best[1]]}", "held_out_log_likelihood": current})


def select_pool_pairs(space, blocks, left_out, fold_of, interactions="select"):
    """The interaction pairs selected on the recorded blocks outside the left out set, in the stratified folds restricted to
    them, and the selection log; interactions none keeps the main effects."""
    out = {tuple(h) for h in left_out}
    pool = [h for h in space.combinations if h in blocks and h not in out]
    data = recorded_blocks.PoolData(space, blocks, pool)
    folds = {h: fold_of[h] for h in pool}
    if interactions == "none":
        return [], [{"step": "main effects", "held_out_log_likelihood": cross_fitted_log_likelihood(space, data, [], folds)}]
    if interactions != "select":
        raise ValueError("interactions must be select or none")
    return select_interactions(space, data, folds)


# ----- the form of the model, chosen before any episode -----
def folds_of_pool(pool, fold_of):
    """[(training indices, held out indices)] of every fold among the combinations of pool, by their place in pool."""
    return [(np.array([i for i, h in enumerate(pool) if fold_of[h] != f], dtype=int), [i for i, h in enumerate(pool) if fold_of[h] == f])
            for f in sorted(set(fold_of[h] for h in pool))]


def form_window_likelihood(recorded, folds, estimator):
    """The criterion that chooses the covariance estimator: over the folds and over the held out blocks
    with at least two windows, the log density, up to a term that no estimator changes, of the scatter of a block's window
    means around their own mean, under the estimator's covariance of one cycle of the training blocks over the window width.
    It reads neither the mean model nor the block offset. None when the covariance of a fold cannot be estimated or factored,
    so that such an estimator is not chosen."""
    total = 0.0
    for train, test in folds:
        try:
            scale = covariance_estimators.one_cycle_covariance(recorded.sums(train), estimator, recorded.window) / recorded.window
            total += sum(window_scatter_log_density(recorded.window_cross[i], recorded.windows[i], scale) for i in test if recorded.windows[i] >= 2)
        except (ValueError, np.linalg.LinAlgError):
            return None
    return total


def window_block_likelihood(space, data, recorded, pairs, folds, estimator, block):
    """The report of a block effect: over the folds, the log density of the held out blocks' window means
    around the fit on the training blocks, the within covariance of a window mean the estimator's covariance of one cycle over
    the window width and, when block, one offset per block with the block covariance of the training blocks' leave one block
    out misfits. None when a training fit does not exist. It is reported and chooses nothing."""
    x = model_terms.design_matrix(space, data.combos, pairs)
    total = 0.0
    for train, test in folds:
        try:
            if not len(train) or not model_terms.full_rank(x[train]):
                return None
            fit = recorded_blocks.fit_pool(x, data.means, data.counts, train)
            scale = covariance_estimators.one_cycle_covariance(recorded.sums(train), estimator, recorded.window) / recorded.window
            offset = (covariance_estimators.covariance_of_rows(fit.residuals / (1 - fit.leverage)[:, None], len(train), covariance_estimators.block_covariance_estimator(estimator))
                      if block else np.zeros_like(scale))
            for i in test:
                means = covariance_estimators.window_means(data.arrays[i], recorded.window)
                if len(means):
                    total += block_log_density(means - x[i] @ fit.beta, scale, offset)
        except (ValueError, np.linalg.LinAlgError):
            return None
    return total


def identifies_every_pair(space, combos, fold_of):
    """Whether the blocks of the combinations identify the model with every interaction pair on every training fold, so that a
    forward selection on them turns no pair away for want of recordings. It reads the grid and the folds and no recording."""
    combos = [tuple(h) for h in combos]
    if not combos:
        return False
    design = model_terms.design_matrix(space, combos, list(space.pairs))
    return all(model_terms.full_rank(design[[i for i, h in enumerate(combos) if fold_of[h] != f]]) for f in sorted(set(fold_of[h] for h in combos)))


def development_identifies(space, development, fold_of):
    """Whether the development blocks identify every pair in every pool of them that the rule of the forms uses: all of them,
    for a start outside them, and all of them less one, for each start among them. False without a development block."""
    development = [tuple(h) for h in development]
    return bool(development) and all(identifies_every_pair(space, pool, fold_of)
                                     for pool in (development, *([h for h in development if h != s] for s in development)))


def held_out_window_dof(recorded):
    """The degrees of freedom of the criterion that chooses the covariance estimator: over the blocks with at least two
    windows, the windows of a block less one. Every block is held out once, so the count is a property of the blocks and of
    the window width, whatever an estimator could evaluate."""
    return int(np.maximum(recorded.windows - 1, 0).sum())


def select_form(space, blocks, pool, fold_of, window):
    """The form that the recorded blocks of pool choose, and no other block: the registered covariance estimator of the
    largest form_window_likelihood, ties to the earlier registered one, or DEFAULT_COVARIANCE with fallback set when the held
    out windows give fewer degrees of freedom than there are summary fields, too few to tell the estimators apart, or when no
    estimator can be evaluated. Beside it, the interaction pairs that the forward selection picks on those blocks with their
    log, and under those pairs and the chosen estimator the window level density of every registered block effect, which is
    reported and chooses nothing. Blocks that lie in fewer than two folds, no block among them, hold nothing out against a fit
    on the others: the form is then the fallback, with no pair and no report."""
    members = {tuple(h) for h in pool}
    pool = [h for h in space.combinations if h in members and h in blocks]
    forms = [{"covariance": estimator, "window_log_likelihood": None} for estimator in covariance_estimators.COVARIANCE_ESTIMATORS]
    effects = [{"block_effect": effect, "window_log_likelihood": None} for effect in BLOCK_EFFECTS]
    folds_of_blocks = {h: fold_of[h] for h in pool}
    folds = folds_of_pool(pool, folds_of_blocks)
    if len(folds) < 2:
        return {"pairs": [], "log": [], "covariance": covariance_estimators.DEFAULT_COVARIANCE, "fallback": True, "held_out_window_dof": 0, "forms": forms,
                "block_effects": effects}
    data = recorded_blocks.PoolData(space, blocks, pool)
    recorded = recorded_blocks.RecordedBlocks(space, {h: blocks[h] for h in pool}, window)
    for form in forms:
        form["window_log_likelihood"] = form_window_likelihood(recorded, folds, form["covariance"])
    dof = held_out_window_dof(recorded)
    eligible = [form for form in forms if form["window_log_likelihood"] is not None]
    fallback = dof < len(space.fields) or not eligible
    covariance = covariance_estimators.DEFAULT_COVARIANCE if fallback else max(eligible, key=lambda form: form["window_log_likelihood"])["covariance"]
    pairs, log = select_interactions(space, data, folds_of_blocks)
    for effect in effects:
        effect["window_log_likelihood"] = window_block_likelihood(space, data, recorded, pairs, folds, covariance,
                                                                  effect["block_effect"] == "joint_block")
    return {"pairs": pairs, "log": log, "covariance": covariance, "fallback": fallback, "held_out_window_dof": dof, "forms": forms,
            "block_effects": effects}


def model_forms(space, blocks, development, fold_of, window):
    """The form of the model for every recorded start, and how each part of it was chosen. development names the development
    blocks, the recorded combinations outside the test set of the asset.

    A part of the form that the development blocks identify is chosen on them, for a start among them on the others. A part
    that they do not identify is chosen once per start on every recorded block but the two that the fit of the start's own
    hypothesis leaves out, the start's and its second block. The covariance estimator is always evaluated on the development
    blocks (select_form), which keeps the first registered one when they are too few. The interaction pairs are chosen on the
    development blocks when every pool of them that the rule uses identifies every pair (development_identifies), and per start
    otherwise. The block effect is the registered ranking model and is only reported.

    Returns ({start: selection}, parts). A selection holds the pairs with their log and the blocks their selection left out
    (left_out), the estimator with its fallback flag, its degrees of freedom, the criterion of every registered estimator and
    the block it left out (covariance_left_out), and the report of the block effects with the pairs it was computed under
    (report_pairs). parts says where each part of the form was chosen."""
    recorded = [h for h in space.combinations if h in blocks]
    named = {tuple(h) for h in development}
    development = [h for h in recorded if h in named]
    members = set(development)
    pools = {s: [h for h in development if h != s] for s in development}
    shared = select_form(space, blocks, development, fold_of, window)
    on_development = development_identifies(space, development, fold_of)
    selections = {}
    for s in recorded:
        form = select_form(space, blocks, pools[s], fold_of, window) if s in members else shared
        own = [s] if s in members else []
        if on_development:
            pairs, log, left_out = form["pairs"], form["log"], own
        else:
            left_out = [s, recorded_blocks.second_block(s, [h for h in recorded if h != s])]
            pairs, log = select_pool_pairs(space, blocks, left_out, fold_of)
        selections[s] = {"pairs": pairs, "log": log, "left_out": left_out, "covariance": form["covariance"], "covariance_left_out": own,
                         "fallback": form["fallback"], "held_out_window_dof": form["held_out_window_dof"], "forms": form["forms"],
                         "block_effects": form["block_effects"], "report_pairs": form["pairs"]}
    parts = {"pairs": {"chosen": "development" if on_development else "per_start", "development_identifies": on_development},
             "covariance": {"chosen": "development"}, "block_effect": {"chosen": "registered"}}
    return selections, parts
