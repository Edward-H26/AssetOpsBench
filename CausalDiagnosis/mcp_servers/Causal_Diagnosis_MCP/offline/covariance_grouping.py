"""The grouping of the covariances of the effect model, part of the form of every start, chosen before any episode
(CausalDiagnosis/docs/causal-diagnosis-mcp.md, section The models file).

A candidate is the pooled form, the covariances of the fits of an episode as fit_episode builds them, or a component and a
weight: every hypothesis mixes into its within and block covariance, at one minus the weight, the centred sample covariances of
the recorded blocks at its level of the component without the block its fit leaves out (episode_fit._grouped). A candidate is
scored on a development start by the held out log score of the true start: the mean over the served entry offsets of the log
posterior probability of the start after its record of recordings before cycle 0, under the fits of an episode of that start,
whose block none of them reads, with a uniform prior, which is the belief the filter gives from a record. The start of every
recorded combination takes the candidate of the largest summed score over the development starts other than itself, and the
pooled form unless that candidate beats it by more than one standard error of the paired total (the one standard error rule);
a tie goes to the pooled form. The nested gain, each development start scored under the candidate chosen without it against
the pooled form, is reported with a bootstrap interval over the starts and chooses nothing."""
import math
import numpy as np
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import fit_episode, group_tables

GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # the weights of the pooled covariances; 1 is the pooled form
NESTED_RESAMPLES, NESTED_SEED = 10000, 0  # the bootstrap of the reported nested gain


def candidates(space):
    """None for the pooled form first, so that it wins a tie, then (component index, weight) for every component and weight."""
    return [None, *((k, w) for k in range(len(space.components)) for w in GRID)]


def record_log_likelihoods(effect, record):
    """The log likelihood of every hypothesis for the cycles of one record (n, d) under the joint_block likelihood of the
    effect model: the cycles share one block offset with the block covariance and vary around it by the within covariance
    (model_form.block_log_density), for every hypothesis at once."""
    n, d = record.shape
    resid = record[None] - effect.mu_table[:, None, :]
    mean = resid.mean(axis=1)
    within = np.linalg.cholesky(effect.sigma_within)
    total = np.linalg.cholesky(effect.sigma_within + n * effect.sigma_block)
    contrasts = np.linalg.solve(within, (resid - mean[:, None, :]).transpose(0, 2, 1))
    offset = np.linalg.solve(total, mean[:, :, None])[..., 0]
    logdet = (n - 1) * 2 * np.log(np.diagonal(within, axis1=1, axis2=2)).sum(axis=1) + 2 * np.log(np.diagonal(total, axis1=1, axis2=2)).sum(axis=1)
    return -0.5 * (n * d * math.log(2 * math.pi) + logdet + (contrasts ** 2).sum(axis=(1, 2)) + n * (offset ** 2).sum(axis=1))


def start_scores(recorded, pairs, covariance, start, records):
    """[mean held out log score of the start over its records] for every candidate (candidates), records the (n, d) cycles of
    each served entry offset, the infinite minimum for a candidate whose covariance of some hypothesis is not positive definite."""
    space = recorded.space
    base = fit_episode(recorded, pairs, start, 0, 0, covariance=covariance).effect
    truth = space.combo_index[tuple(start)]
    groups = {}
    out = []
    for candidate in candidates(space):
        effect = base
        if candidate is not None:
            k, w = candidate
            if k not in groups:
                groups[k] = group_tables(recorded, pairs, start, k)
            effect = type(base).__new__(type(base))
            effect.__dict__.update(base.__dict__, sigma_within=(1 - w) * groups[k][0] + w * base.sigma_within,
                                   sigma_block=(1 - w) * groups[k][1] + w * base.sigma_block)
        try:
            scores = []
            for record in records:
                ll = record_log_likelihoods(effect, record)
                top = ll.max()
                scores.append(float(ll[truth] - top - math.log(np.exp(ll - top).sum())))
            out.append(float(np.mean(scores)))
        except np.linalg.LinAlgError:
            out.append(-math.inf)
    return out


def choose(scores, rows):
    """The index of the candidate (candidates) that the scores of the rows choose: the largest summed score, the pooled form
    (index 0) on a tie or unless the gain over it exceeds one standard error of the paired total, the standard deviation of the
    paired differences times the square root of their number. Returns (index, gain, standard error)."""
    matrix = np.asarray([scores[r] for r in rows], dtype=float)
    totals = matrix.sum(axis=0)
    best = int(np.argmax(totals))  # argmax takes the first of equal totals, the pooled form
    if best == 0 or not math.isfinite(totals[best]):
        return 0, 0.0, 0.0
    paired = matrix[:, best] - matrix[:, 0]
    gain, se = float(paired.sum()), float(paired.std(ddof=1) * math.sqrt(len(paired))) if len(paired) > 1 else math.inf
    return (best, gain, se) if gain > se else (0, gain, se)


def select_groupings(recorded, forms, development, view, segments):
    """({start: candidate} for every recorded start, the report). forms holds per recorded start its pairs and its covariance
    estimator, development the development starts, segments the block positions of the record of every served entry offset."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.offline_table import released

    space = recorded.space
    index = space.field_index
    scores = {}
    for start in development:
        block = recorded.arrays[recorded.index[start]]
        records = [np.array([[released(index, block[p], view.summary_fields)[f] for f in space.fields] for p in positions]) for positions in segments]
        scores[start] = start_scores(recorded, forms[start]["pairs"], forms[start]["covariance"], start, records)
    listed = candidates(space)
    chosen, nested = {}, []
    for start in recorded.combos:
        index_, gain, se = choose(scores, [s for s in development if s != start])
        chosen[start] = {"candidate": listed[index_], "gain": gain, "se": se}
        if start in scores:
            nested.append(scores[start][index_] - scores[start][0])
    report = {}
    if nested:
        nested = np.array(nested)
        draws = np.random.default_rng(NESTED_SEED).integers(0, len(nested), (NESTED_RESAMPLES, len(nested)))
        means = nested[draws].mean(axis=1)
        report = {"starts": len(nested), "mean_gain": float(nested.mean()), "interval_95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                  "resamples": NESTED_RESAMPLES, "seed": NESTED_SEED}
    return chosen, report, scores
