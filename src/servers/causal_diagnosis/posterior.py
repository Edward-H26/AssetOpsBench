"""E2 to E6 of design v2: the posterior over start combinations under the joint Gaussian likelihood of the effect model
(docs/hydraulic_rig/causal-diagnosis-server.md).

The combination current at a reading is the start combination transformed by the work orders closed before the
reading, so readings are grouped into segments between closures and each segment is read under its own combination. A
cycle served twice within a segment (the window of the recording wrapped, or two groups read the same cycle) is one
observation with the union of its summaries. The literal E2 (ranking model joint) treats the observations of a segment as
independent draws around the model mean. The default, joint_block, adds a latent block mean: the cycles of one recorded
block share an offset from the model mean, because the block was recorded in one run of the rig under one setting and the
effect model predicts the block from the other blocks, so the misfit of a block is common to all of its cycles. The offset
has the covariance of the cross fitted block mean residuals, is integrated out by a Kalman step per observation, and makes
repeated readings of one block add less than independent draws would; it is the same reason E6 resamples the pool by block
rather than by cycle, and the covariance revision that section 5.6 allows, chosen as the default after the calibration
report of 2026-11-01. Scheduled drops are not modeled.
"""
import math
from collections import namedtuple

import numpy as np

from envs.hydraulic_rig.classifiers import COMBOS
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS

from .effect_model import FIELD_INDEX, HEALTHY_COMBO, SUMMARY_FIELDS, field_indices, n_faulty

Reading = namedtuple("Reading", "n t group values")
ClosedOrder = namedtuple("ClosedOrder", "t component source id")
Observation = namedtuple("Observation", "fields index x numbers t")
Segment = namedtuple("Segment", "orders observations")
RANKING_MODELS = ("joint", "joint_block", "discriminant_urn")
DEFAULT_RANKING_MODEL = "joint_block"  # chosen from the calibration report of section 5.6; joint and discriminant_urn are the A12 arms


def candidates(rules):
    """The start combinations the public fault rule allows, in grid order."""
    low, high = rules["fault_rule"]["min_faulty"], rules["fault_rule"]["max_faulty"]
    return [h for h in COMBOS if low <= n_faulty(h) <= high]


def apply_orders(combo, orders):
    """The combination after the closed work orders: each replaced component is at its healthy level."""
    combo = list(combo)
    for order in orders:
        combo[COMPONENTS.index(order.component)] = HEALTHY[order.component]
    return tuple(combo)


def orders_before(closed_orders, t):
    """The closed orders whose closing cycle is at or before t, in closing order; the component is healthy from the
    closing cycle on, so a reading at that cycle is served from the new block."""
    return tuple(sorted((o for o in closed_orders if o.t <= t), key=lambda o: (o.t, o.id)))


def modeled(values):
    return {f: v for f, v in values.items() if f in FIELD_INDEX}


def merge_observations(readings):
    """The distinct cycles among the readings of one segment: a reading whose values agree with an earlier observation on
    every shared summary is the same recorded cycle and adds its other summaries to that observation."""
    merged = []
    for reading in readings:
        values = modeled(reading.values)
        if not values:
            continue
        for item in merged:
            shared = [f for f in values if f in item["values"]]
            if shared and all(item["values"][f] == values[f] for f in shared):
                item["values"].update(values)
                item["numbers"].append(reading.n)
                break
        else:
            merged.append({"values": dict(values), "numbers": [reading.n], "t": reading.t})
    out = []
    for item in merged:
        fields = tuple(f for f in SUMMARY_FIELDS if f in item["values"])
        out.append(Observation(fields, field_indices(fields), np.array([item["values"][f] for f in fields]), tuple(item["numbers"]), item["t"]))
    return out


def segments(readings, closed_orders):
    """Readings grouped by the closed orders applied before them, in time order."""
    out = []
    for reading in sorted(readings, key=lambda r: (r.t, r.n)):
        applied = orders_before(closed_orders, reading.t)
        if out and out[-1][0] == applied:
            out[-1][1].append(reading)
        else:
            out.append((applied, [reading]))
    return [Segment(applied, merge_observations(group)) for applied, group in out]


def segment_log_likelihood(mu, sigma_within, sigma_block, observations):
    """Log likelihood of a segment's observations for every candidate, and the state of the latent block mean after them.

    mu is (..., H, d): the segment combination's mean per candidate, with optional leading batch axes (resampled fits);
    sigma_within is (..., d, d); sigma_block is (d, d) or None for the literal E2. Returns (log likelihood (..., H),
    mean (..., H, d), covariance (..., d, d)) where the mean and covariance describe the next observation's predictive
    distribution: N(mean[I], (covariance + sigma_within)[I, I]) for the summaries I."""
    mean = np.array(mu, dtype=float)
    batch = mean.shape[:-2]
    d = mean.shape[-1]
    sigma_within = np.broadcast_to(np.asarray(sigma_within, dtype=float), batch + (d, d))
    covariance = np.broadcast_to(sigma_block if sigma_block is not None else np.zeros((d, d)), batch + (d, d)).copy()
    log_likelihood = np.zeros(mean.shape[:-1])
    for observation in observations:
        idx = observation.index
        predictive = (covariance + sigma_within)[..., idx, :][..., :, idx]
        cholesky = np.linalg.cholesky(predictive)
        residual = observation.x - mean[..., :, idx]
        log_likelihood += log_density(residual, cholesky)
        if sigma_block is not None:
            # Kalman step on the latent block mean, shared by every candidate since the covariance does not depend on it
            cross = covariance[..., :, idx]
            gain = np.swapaxes(np.linalg.solve(predictive, np.swapaxes(cross, -1, -2)), -1, -2)
            mean = mean + np.swapaxes(gain @ np.swapaxes(residual, -1, -2), -1, -2)
            covariance = covariance - gain @ np.swapaxes(cross, -1, -2)
    return log_likelihood, mean, covariance


def log_density(residual, cholesky):
    """Log density of N(0, L L^T) at the rows of residual (..., H, k) with cholesky (..., k, k)."""
    k = residual.shape[-1]
    z = np.linalg.solve(cholesky, np.swapaxes(residual, -1, -2))
    quad = (z ** 2).sum(axis=-2)
    logdet = 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
    return -0.5 * (k * math.log(2 * math.pi) + logdet[..., None] + quad)


def normalized(log_weights):
    """Probabilities from log weights along the last axis."""
    shifted = log_weights - log_weights.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def unit(value):
    """A probability summed from floating point terms, kept inside [0, 1]."""
    return float(min(1.0, max(0.0, value)))


class Posterior:
    """The posterior over the start combinations of the public rule after the released readings and closed orders.

    probabilities is the posterior over start combinations, current[i] the combination candidate i is in now, and the
    predictive state (mean, covariance) describes the next reading of the current block for E9 and E13. With
    block_variance the latent block mean of the block variant is used, else the literal E2."""

    def __init__(self, effect, rules, readings, closed_orders, time, block_variance=False, resampled=False, starts=None):
        self.effect = effect
        self.candidates = candidates(rules) if starts is None else [tuple(h) for h in starts]
        self.start = np.array(self.candidates)
        self.closed_orders = tuple(sorted(closed_orders, key=lambda o: (o.t, o.id)))
        self.segments = segments(readings, closed_orders)
        self.time = time
        self.block_variance = block_variance
        self.sigma_block = effect.sigma_block if block_variance else None
        self.current_orders = orders_before(self.closed_orders, time)
        self.current = [apply_orders(h, self.current_orders) for h in self.candidates]
        self.log_prior = np.full(len(self.candidates), -math.log(len(self.candidates)))
        self.log_weights, self.mean, self.covariance = self._run(effect.mu_rows, effect.sigma_within)
        self.probabilities = normalized(self.log_weights)
        self.resampled_probabilities = None
        if resampled and len(effect.resampled):
            log_weights, _mean, _cov = self._run(effect.resampled_mu_rows, effect.resampled_sigma)
            self.resampled_probabilities = normalized(log_weights)

    def override(self, probabilities):
        """Replace the posterior by another ranking model's probabilities over the same candidates; the predictive state of
        the joint model is kept for the lookahead and the predictions."""
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-300, None)
        self.log_weights = np.log(p / p.sum())
        self.probabilities = normalized(self.log_weights)
        self.resampled_probabilities = None

    def _run(self, mu_rows, sigma_within):
        """Log weights after every segment and the predictive state of the current block."""
        log_weights = np.broadcast_to(self.log_prior, sigma_within.shape[:-2] + self.log_prior.shape).copy()
        state = None
        for segment in self.segments:
            combos = [apply_orders(h, segment.orders) for h in self.candidates]
            ll, mean, covariance = segment_log_likelihood(mu_rows(combos), sigma_within, self.sigma_block, segment.observations)
            log_weights = log_weights + ll
            state = (segment.orders, mean, covariance)
        if state is not None and state[0] == self.current_orders:
            return log_weights, state[1], state[2]
        # no reading of the current block yet: the latent block mean sits at the model mean of the current combination
        mean = mu_rows(self.current)
        d = mean.shape[-1]
        covariance = np.broadcast_to(self.sigma_block if self.sigma_block is not None else np.zeros((d, d)), sigma_within.shape[:-2] + (d, d)).copy()
        return log_weights, mean, covariance

    # ----- E5 -----
    def level_probability(self):
        """P(level of component j at t0) for every component and level."""
        return {c: {level: unit(self.probabilities[self.start[:, j] == level].sum()) for level in LEVELS[c]} for j, c in enumerate(COMPONENTS)}

    def component_marginals(self):
        """E5: P(component j faulty at t0 | readings)."""
        return {c: unit(self.probabilities[self.start[:, j] != HEALTHY[c]].sum()) for j, c in enumerate(COMPONENTS)}

    def current_marginals(self):
        """P(component j faulty now | readings): the t0 marginal of a component no closed order has replaced, else 0."""
        current = np.array(self.current)
        return {c: unit(self.probabilities[current[:, j] != HEALTHY[c]].sum()) for j, c in enumerate(COMPONENTS)}

    def current_posterior(self):
        """Posterior over the combination current now, summed over the start candidates that lead to it."""
        out = {}
        for h, p in zip(self.current, self.probabilities):
            out[h] = out.get(h, 0.0) + float(p)
        return {h: unit(p) for h, p in out.items()}

    def order(self):
        """Candidate indices by decreasing posterior, ties by grid order."""
        return sorted(range(len(self.candidates)), key=lambda i: (-self.probabilities[i], self.candidates[i]))

    # ----- E6 -----
    def stability(self):
        """E6: over the resampled fits, the interval of the top component's marginal and the share of resamples in which
        that component stays on top. None without resampled fits."""
        if self.resampled_probabilities is None:
            return None
        marginals = self.component_marginals()
        top = max(COMPONENTS, key=lambda c: (marginals[c], -COMPONENTS.index(c)))
        faulty = np.array([[h[j] != HEALTHY[c] for j, c in enumerate(COMPONENTS)] for h in self.candidates], dtype=float)
        per_resample = np.clip(self.resampled_probabilities @ faulty, 0.0, 1.0)  # (B, components)
        j = COMPONENTS.index(top)
        # a component shares the top when its marginal equals the largest one, as after readings that settle two faults
        on_top = per_resample[:, j] >= per_resample.max(axis=1) - 1e-12
        return {"top_component": top, "top_marginal": marginals[top], "interval_low": float(per_resample[:, j].min()),
                "interval_high": float(per_resample[:, j].max()), "top_share": float(on_top.mean()), "resamples": int(len(per_resample))}

    def resampled_recovery(self, component):
        """E15: the recovery probability after replacing the component (None for no repair) under every resampled fit, for
        the dead band of the direction label."""
        if self.resampled_probabilities is None:
            return None
        j = None if component is None else COMPONENTS.index(component)
        healthy_after = np.array([all(v == HEALTHY[c] for k, (v, c) in enumerate(zip(h, COMPONENTS)) if k != j) for h in self.current], dtype=float)
        return np.clip(self.resampled_probabilities @ healthy_after, 0.0, 1.0)

    # ----- E3 and E4 -----
    def latest_values(self):
        """The mean of every summary over the observations of the latest segment that has readings, the evidence the
        anomaly scores and the attribution describe; empty before the first reading."""
        for segment in reversed(self.segments):
            if segment.observations:
                sums, counts = {}, {}
                for observation in segment.observations:
                    for f, v in zip(observation.fields, observation.x):
                        sums[f] = sums.get(f, 0.0) + float(v)
                        counts[f] = counts.get(f, 0) + 1
                return {f: sums[f] / counts[f] for f in sums}, segment.orders
        return {}, self.current_orders

    def attribution(self, top_index):
        """E4: for the top candidate's combination at the latest readings, the share of every summary's predicted
        deviation that each component explains, and the explained anomaly mass E_j = sum_k a_k s_kj."""
        values, orders = self.latest_values()
        combo = apply_orders(self.candidates[top_index], orders)
        terms = np.abs(self.effect.component_terms(combo))
        totals = terms.sum(axis=0)
        shares = np.divide(terms, totals, out=np.zeros_like(terms), where=totals > 0)
        anomaly = self.effect.anomaly_scores(values)
        out = {}
        for j, c in enumerate(COMPONENTS):
            channels = [{"field": f, "anomaly_score": a, "share": float(shares[j, FIELD_INDEX[f]])} for f, a in anomaly.items() if shares[j, FIELD_INDEX[f]] > 0]
            channels.sort(key=lambda item: -item["anomaly_score"] * item["share"])
            out[c] = {"explained_anomaly": float(sum(item["anomaly_score"] * item["share"] for item in channels)), "channels": channels}
        return out, anomaly, combo

    # ----- predictive sampling for E9 and E18 -----
    def predictive(self, fields):
        """Mean (H, k) and Cholesky factor (k, k) of the next reading of the given summaries under every candidate."""
        idx = field_indices(fields)
        predictive = (self.covariance + self.effect.sigma_within)[np.ix_(idx, idx)]
        return self.mean[:, idx], np.linalg.cholesky(predictive)

    def sample_next(self, fields, n_samples, rng):
        """Draws from the predictive mixture of the next reading: a candidate by its posterior, then its Gaussian."""
        mean, cholesky = self.predictive(fields)
        which = rng.choice(len(self.candidates), size=n_samples, p=self.probabilities)
        noise = rng.standard_normal((n_samples, len(cholesky)))
        return mean[which] + noise @ cholesky.T

    def update_with(self, fields, x):
        """Posterior probabilities (S, H) after each of the S hypothetical readings x (S, k) of the given summaries."""
        mean, cholesky = self.predictive(fields)
        residual = x[:, None, :] - mean[None, :, :]
        return normalized(self.log_weights[None, :] + log_density(residual, cholesky))

    def post_repair(self, component):
        """E13: the posterior over combinations after replacing the component, the probability that the replacement
        addresses a fault and the recovery probability."""
        j = COMPONENTS.index(component)
        after, addresses = {}, 0.0
        for h, p in zip(self.current, self.probabilities):
            if h[j] != HEALTHY[component]:
                addresses += float(p)
            moved = h[:j] + (HEALTHY[component],) + h[j + 1:]
            after[moved] = after.get(moved, 0.0) + float(p)
        after = {h: unit(p) for h, p in after.items()}
        return after, unit(addresses), after.get(HEALTHY_COMBO, 0.0)
