"""The belief over the combinations of the profile at one cycle, collapsed from the augmented states of the filter
(version 7: E2, E4, E5, E6 and the predictive of E9, E20 and E21, CausalDiagnosis/docs/causal-diagnosis-mcp.md, section
The filter).

Every hypothesis is a combination of the profile and probabilities[i] is its mass now, the state that the diagnosis names
and the verifier scores. The block offset of combination i has the mean mean[i] and the covariance covariance[i], and the
fit that scores combination i has its own within covariance, so the next reading of the block served now has,
under combination i, the Gaussian predictive N(mu(i) + mean[i], covariance[i] + within[i]) on the summaries read. The
predictive of a reading is factored once per combination with mass, in one batch. The server keeps no start combination.
"""
import math
from collections import namedtuple

import numpy as np

RANKING_MODELS = ("joint", "joint_block", "discriminant_urn")
DEFAULT_RANKING_MODEL = "joint_block"  # chosen from the calibration report of calibration_report.py; joint and discriminant_urn are the En9 alternatives
LOG_2PI = math.log(2 * math.pi)

Gaussians = namedtuple("Gaussians", "index mean cholesky inverse logdet")


def normalized(log_weights):
    """Probabilities from log weights along the last axis."""
    shifted = log_weights - np.max(log_weights, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def proportions(masses):
    """The masses over their sum, refused when the sum is not positive, nan included, since a belief needs a combination of
    positive mass."""
    total = masses.sum()
    if not total > 0:
        raise ValueError("the masses of the belief sum to zero, so no combination has a probability")
    return masses / total


def unit(value):
    """A probability summed from floating point terms, kept inside [0, 1]."""
    return float(min(1.0, max(0.0, value)))


def entropy(p):
    """Shannon entropy in nats along the last axis."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -terms.sum(axis=-1)


def effective_number(p):
    """N(b) = exp(H(b)), the Hill number of order one: m for a belief uniform over m hypotheses."""
    return np.exp(entropy(p))


def per_hypothesis(matrix, n):
    """(n, d, d) from one covariance per hypothesis or from one (d, d) covariance shared by all of them."""
    matrix = np.asarray(matrix, dtype=float)
    return np.broadcast_to(matrix, (n,) + matrix.shape) if matrix.ndim == 2 else matrix


def gaussians(index, mean, covariance):
    """The terms of the Gaussian densities of several hypotheses, each with its own covariance, in one batch."""
    cholesky = np.linalg.cholesky(covariance)
    return Gaussians(index, mean, cholesky, np.linalg.inv(cholesky), 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1))


def sequence_of(fields):
    """The summaries of each cycle of a reading: one tuple of field names, or a list of such tuples for several cycles."""
    fields = tuple(fields)
    if fields and not isinstance(fields[0], str):
        return tuple(tuple(f) for f in fields)
    return (fields,)


def batch_log_density(terms, x):
    """(S, n) log densities of the rows of x (S, k) under the n hypotheses of terms."""
    z = terms.inverse @ (x[None, :, :] - terms.mean[:, None, :]).transpose(0, 2, 1)
    return (-0.5 * (x.shape[1] * LOG_2PI + terms.logdet[:, None] + (z ** 2).sum(axis=1))).T


class Posterior:
    """The collapsed belief. candidates lists every combination of the profile in grid order; probabilities is the main
    fit's mass, resampled_probabilities (resamples, H) the masses under the resampled fits of E6 when the filter carried
    them; mean (H, d) is the offset mean of every combination and covariance (H, d, d) its covariance."""

    def __init__(self, effect, probabilities, mean, covariance, resampled=None, time=0, n_states=None, entry_of_top=0, observations=()):
        self.effect = effect
        self.space = effect.space
        self.candidates = list(self.space.combinations)
        self.probabilities = proportions(np.asarray(probabilities, dtype=float))
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = per_hypothesis(covariance, len(self.candidates))
        self.within = per_hypothesis(effect.sigma_within, len(self.candidates))
        self.resampled_probabilities = None if resampled is None or len(resampled) == 0 else np.asarray(resampled, dtype=float)
        self.time = time
        self.n_states = len(self.candidates) if n_states is None else n_states
        self.entry_of_top = entry_of_top
        self.observations = list(observations)
        self._terms = {}

    @classmethod
    def uniform(cls, effect, block=True, support=None):
        """The belief over every combination, or over the given support, before any reading, with a fresh block offset."""
        space = effect.space
        keep = None if support is None else {tuple(h) for h in support}
        p = np.array([1.0 if keep is None or h in keep else 0.0 for h in space.combinations])
        sigma = per_hypothesis(effect.sigma_block, len(p))
        return cls(effect, p, np.zeros((len(p), len(space.fields))), sigma if block else np.zeros_like(sigma))

    def override(self, probabilities):
        """Replace the masses by another ranking model's over the same combinations; the predictive of the joint model is
        kept for E9 and the predictions."""
        self.probabilities = proportions(np.clip(np.asarray(probabilities, dtype=float), 0.0, None))
        self.resampled_probabilities = None
        self._terms = {}

    # ----- E5 -----
    def order(self):
        """Combination indices by decreasing mass, ties by grid order."""
        return sorted(range(len(self.candidates)), key=lambda i: (-self.probabilities[i], self.candidates[i]))

    def current_marginals(self):
        """E5: P(component j faulty now | readings)."""
        grid = np.array(self.candidates)
        return {c: unit(self.probabilities[grid[:, j] != self.space.healthy[j]].sum()) for j, c in enumerate(self.space.components)}

    def level_probability(self):
        """P(level of component j now) for every component and level."""
        grid = np.array(self.candidates)
        return {c: {level: unit(self.probabilities[grid[:, j] == level].sum()) for level in self.space.levels[j]} for j, c in enumerate(self.space.components)}

    def current_posterior(self):
        """{combination: mass} over the combinations with positive mass."""
        return {h: unit(p) for h, p in zip(self.candidates, self.probabilities) if p > 0}

    def effective_hypotheses(self):
        return float(effective_number(self.probabilities))

    # ----- E6 -----
    def stability(self):
        """E6: over the resampled fits, the interval of the top component's marginal and the share of resamples in which
        that component stays on top. None without resampled fits."""
        if self.resampled_probabilities is None:
            return None
        marginals = self.current_marginals()
        components = self.space.components
        top = max(components, key=lambda c: (marginals[c], -components.index(c)))
        faulty = np.array([[h[j] != self.space.healthy[j] for j in range(len(components))] for h in self.candidates], dtype=float)
        per_resample = np.clip(self.resampled_probabilities @ faulty, 0.0, 1.0)
        j = components.index(top)
        # a component shares the top when its marginal equals the largest one, as after readings that settle two faults
        on_top = per_resample[:, j] >= per_resample.max(axis=1) - 1e-12
        return {"top_component": top, "top_marginal": marginals[top], "interval_low": float(per_resample[:, j].min()),
                "interval_high": float(per_resample[:, j].max()), "top_share": float(on_top.mean()), "resamples": int(len(per_resample))}

    # ----- E3 and E4 -----
    def latest_values(self):
        """The mean of every summary over the observations of the block the most probable state has been in since it was
        entered, those after the first entry_of_top observations in the order of the replay, or over the latest observation
        when none falls in it, empty before the first reading."""
        own = self.observations[self.entry_of_top:] or self.observations[-1:]
        sums, counts = {}, {}
        for observation in own:
            for f, v in zip(observation.fields, observation.x):
                sums[f] = sums.get(f, 0.0) + float(v)
                counts[f] = counts.get(f, 0) + 1
        return {f: sums[f] / counts[f] for f in sums}

    def attribution(self, top_index, effect=None):
        """E4: for the top combination at the latest readings, the share of every summary's predicted deviation that each
        component explains, and the explained anomaly mass E_j = sum_k a_k s_kj, from the coefficients and the healthy
        reference of effect, the belief's own effect model when None."""
        effect = self.effect if effect is None else effect
        values = self.latest_values()
        combo = self.candidates[top_index]
        terms = np.abs(effect.component_terms(combo))
        totals = terms.sum(axis=0)
        shares = np.divide(terms, totals, out=np.zeros_like(terms), where=totals > 0)
        anomaly = effect.anomaly_scores(values)
        index = self.space.field_index
        out = {}
        for j, c in enumerate(self.space.components):
            channels = [{"field": f, "anomaly_score": a, "share": float(shares[j, index[f]])} for f, a in anomaly.items() if shares[j, index[f]] > 0]
            channels.sort(key=lambda item: -item["anomaly_score"] * item["share"])
            out[c] = {"explained_anomaly": float(sum(item["anomaly_score"] * item["share"] for item in channels)), "channels": channels}
        return out, anomaly, combo

    # ----- the predictive of the next readings -----
    def live(self):
        return np.nonzero(self.probabilities > 0)[0]

    def predictive_mean(self, fields):
        idx = self.space.field_indices(fields)
        return self.effect.mu_table[:, idx] + self.mean[:, idx]

    def predictive(self, fields):
        """The Gaussian terms of the next readings under every combination with positive mass, each with its own covariance.
        fields names the summaries of one cycle, or is a list of such tuples for readings of several cycles in a row, which
        share the block offset and draw their own within noise, so that the stacked density is the exact sequential update."""
        sequence = sequence_of(fields)
        if sequence not in self._terms:
            indices = [self.space.field_indices(f) for f in sequence]
            stacked = np.concatenate(indices)
            live = self.live()
            covariance = self.covariance[live][:, stacked][:, :, stacked]
            start = 0
            for idx in indices:
                covariance[:, start:start + len(idx), start:start + len(idx)] += self.within[live][:, idx][:, :, idx]
                start += len(idx)
            mean = np.concatenate([self.predictive_mean(f)[live] for f in sequence], axis=1)
            self._terms[sequence] = (gaussians(live, mean, covariance), len(stacked))
        return self._terms[sequence]

    def log_density(self, fields, x):
        """(S, H) log density of every row of x (S, k) under every combination with positive mass, minus infinity
        elsewhere, for the readings that fields names."""
        terms, _k = self.predictive(fields)
        out = np.full((len(x), len(self.candidates)), -np.inf)
        out[:, terms.index] = batch_log_density(terms, np.asarray(x, dtype=float))
        return out

    def sample_next(self, fields, n_samples, rng, weights=None):
        """Draws from the predictive mixture of the readings that fields names: a combination by its mass, or by the given
        weights over combinations with positive mass, then its Gaussian."""
        terms, k = self.predictive(fields)
        p = self.probabilities if weights is None else np.asarray(weights, dtype=float)
        which = rng.choice(len(self.candidates), size=n_samples, p=p)
        if not np.all(self.probabilities[which] > 0):
            raise ValueError("draws are taken only under combinations with positive mass")
        noise = rng.standard_normal((n_samples, k))
        position = np.searchsorted(terms.index, which)
        return terms.mean[position] + np.einsum("sij,sj->si", terms.cholesky[position], noise)

    def update_with(self, fields, x):
        """Masses (S, H) after each of the S hypothetical readings x (S, k) that fields names."""
        with np.errstate(divide="ignore"):
            return normalized(np.log(self.probabilities)[None, :] + self.log_density(fields, x))

    def updated(self, fields, y):
        """The belief after one reading y of the given summaries: masses by the predictive density, offsets and their
        covariances by one Kalman step per combination."""
        idx = self.space.field_indices(fields)
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore"):
            probabilities = normalized(np.log(self.probabilities) + self.log_density(fields, y[None])[0])
        residual = y[None] - self.predictive_mean(fields)
        rows = self.covariance[:, idx, :]
        gain = np.linalg.solve(rows[:, :, idx] + self.within[:, idx][:, :, idx], rows).transpose(0, 2, 1)
        mean = self.mean + np.einsum("hdk,hk->hd", gain, residual)
        covariance = self.covariance - gain @ rows
        return Posterior(self.effect, probabilities, mean, covariance, time=self.time, n_states=self.n_states,
                         entry_of_top=self.entry_of_top, observations=self.observations)
