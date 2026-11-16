import math
import numpy as np
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import per_hypothesis
from . import released_episode, rule_cards, augmented_states, rule_card_kernels


def log_density(residual, cholesky):
    """Log density of N(0, L L^T) at the rows of residual (..., n, k) with cholesky (k, k) or (..., n, k, k)."""
    k = residual.shape[-1]
    if cholesky.ndim == 2:
        z = np.linalg.solve(cholesky, np.swapaxes(residual, -1, -2))
        quad = (z ** 2).sum(axis=-2)
        logdet = 2 * np.log(np.diagonal(cholesky)).sum()
    else:
        z = np.linalg.solve(cholesky, residual[..., None])[..., 0]
        quad = (z ** 2).sum(axis=-1)
        logdet = 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
    return -0.5 * (k * math.log(2 * math.pi) + logdet + quad)


def logsumexp(values, axis=-1):
    top = np.max(values, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.squeeze(top, axis=axis) + np.log(np.exp(values - top).sum(axis=axis))


# ----- the filter -----
class Filter:
    """The replay of the released evidence into the belief at the current cycle, under the main fit and the resampled
    mean tables of E6, every registered one."""

    def __init__(self, view, effect, episode, resampled=True, prior="history", block=True, support=None, replay=True):
        """prior "history" weighs the start by the released fix history and None leaves it uniform; block False is the
        literal likelihood without the block offset (ranking model joint); support restricts the hypotheses at the start;
        replay False stops at the belief of cycle 0, with the resampled fits sized for the episode's time. effect scores every
        reading, before cycle 0 and from it, with one within and one block covariance per hypothesis."""
        self.view, self.effect, self.space = view, effect, effect.space
        self.rules = rule_cards.Rules(view, self.space)
        self.episode = episode
        n = len(self.space.combinations)
        self.within = per_hypothesis(effect.sigma_within, n)
        self.block = per_hypothesis(effect.sigma_block, n).copy() if block else np.zeros((n,) + effect.sigma_block.shape[-2:])
        self.support = None if support is None else {tuple(h) for h in support}
        self.observations = released_episode.merge_observations(episode.readings, self.space)
        self.slots = rule_card_kernels.cycle_slots(episode.closures, self.rules)
        init = self._initial_states(prior)
        self.peak_states = len(init[0])
        n_resampled = effect.resample_count if resampled else 0
        self.tables = effect.tables(n_resampled)
        self.n_resampled = n_resampled
        pos, sched, logp = init
        n = len(pos)
        J = self.rules.J
        R = len(self.tables)
        d = len(self.space.fields)
        self.states = augmented_states.States(self.rules, pos, np.zeros(n, dtype=int), sched, np.full((n, J), -1), np.full((n, J), -1),
                             np.repeat(logp[None], R, axis=0), {"offset": np.zeros((R, n, d))})
        self.covariance = {0: self.block.copy()}
        self.time = 0
        self.seen = 0
        if replay:
            self._replay()

    def _initial_states(self, prior):
        """Positions, schedule draws and log prior of the states at cycle 0."""
        rules = self.rules
        history = self.episode.history if prior == "history" else None
        rows, scheds, logs = [], [], []
        for combo in self.space.combinations:
            if self.support is not None and combo not in self.support:
                continue
            base = rules.history_log_likelihood(combo, history) if history is not None else 0.0
            if not math.isfinite(base):
                continue
            for log_draw, draw in rules.schedule_draws(combo):
                rows.append(rules.positions(combo))
                scheds.append([list(slot) for slot in draw])
                logs.append(base + log_draw)
        if not rows and prior is None:  # no combination of the support, so a second pass without the history finds none either
            raise ValueError("the support of the filter holds no combination of the asset, so no state starts the belief")
        if not rows:  # a history the rule cannot produce under any combination says nothing
            return self._initial_states(prior=None)
        S = rules.schedule_count
        sched = np.array(scheds, dtype=int).reshape(len(rows), S, 3)
        logp = np.array(logs)
        return np.array(rows, dtype=int), sched, logp - logsumexp(logp)

    def _entered(self, states):
        """Rows whose combination changed enter a fresh block after the recorded cycles observed so far."""
        rule_card_kernels.enter(states, self.seen, self.covariance, self.block)

    def _settled(self, closure):
        """Whether the environment closed an agent's order while the clock stood at the closure's cycle rather than in a
        cycle's kernels: an order of no repair downtime on a row without a clock or at the cycle a finish
        reached."""
        no_downtime = self.rules.downtime[self.rules.components.index(closure.component)] == 0
        return no_downtime and (not self.episode.clock or closure.t >= self.episode.time)

    def _replay(self):
        """Every released reading and closure enters once, in the order the rig served it: the
        readings of negative cycles, recorded before the episode began, before any cycle runs, then for every cycle up to the
        current one the closures that ended the cycle before, what the rig served while the clock stood at the cycle in the
        order it was released, and for a cycle the clock ran its other kernels and the readings recorded during it."""
        time, standing, recorded = self.episode.time, {}, {}
        kernels = [c for c in self.episode.closures if not self._settled(c)]
        for closure in self.episode.closures:
            if self._settled(closure):
                standing.setdefault(closure.t, []).append(closure)
        for observation in self.observations:
            stood = observation.t >= 0 and (observation.standing or observation.t >= time)
            (standing if stood else recorded).setdefault(observation.t, []).append(observation)
        for t in sorted(t for t in recorded if t < 0):
            for observation in recorded[t]:
                self.observe(observation)
        for t in range(time):
            self.predict(t, kernels, released_episode.release_order(standing.get(t, ())))
            for observation in recorded.get(t, ()):
                self.observe(observation)
        self.settle(kernels, released_episode.release_order(standing.get(time, ())))

    # ----- the predict and update steps, in place -----
    def copy(self):
        """A filter with its own copy of the states and covariances, for a step that must leave this one unchanged."""
        twin = object.__new__(Filter)
        twin.__dict__.update(self.__dict__)
        twin.states = self.states.take(np.arange(len(self.states)))
        twin.covariance = dict(self.covariance)
        return twin

    def predict(self, t, closures=(), standing=()):
        """Cycle t in the clock's order, with the closures among `closures` stamped t: the closures that ended the cycle
        before, then what the rig served while the clock stood at t (standing, readings and settled closures in the order
        released), then the other kernels of the cycle, which give the belief that a reading recorded during cycle t meets."""
        slot = rule_card_kernels.cycle_slots([c for c in closures if c.t == t], self.rules).get(t, {})
        self._stand(t, slot.get("start", ()), standing)
        rule_card_kernels.run_cycle(self.states, t, slot, self._entered)
        self.states.merge()
        self.time = t + 1
        self.peak_states = max(self.peak_states, len(self.states))
        return self

    def observe(self, observation):
        """The update step of one recorded cycle."""
        self._update(observation)
        self.seen += 1
        return self

    def settle(self, closures=(), standing=()):
        """The cycle at which the clock stands: the closures stamped with it that ended the last cycle that ran, then what
        the rig served while the clock stood there (standing, in the order released), which give the belief now."""
        ended = rule_card_kernels.cycle_slots([c for c in closures if c.t == self.time], self.rules).get(self.time, {}).get("start", ())
        self._stand(self.time, ended, standing)
        self.states.merge()
        self._drop_unused_covariances()
        return self

    def _stand(self, t, ended, standing):
        """The closures that ended the cycle before t, then the readings and the settled closures of standing in their
        order, a closure entering the rows it changes into a fresh block before the next reading."""
        for closure in ended:
            rule_card_kernels.close_kernel(self.states, self.rules.components.index(closure.component), closure.mode, t)
        self._entered(self.states)
        for item in standing:
            if isinstance(item, released_episode.Closure):
                rule_card_kernels.close_kernel(self.states, self.rules.components.index(item.component), item.mode, t)
                self._entered(self.states)
            else:
                self.observe(item)

    def _drop_unused_covariances(self):
        used = set(self.states.entry.tolist())
        self.covariance = {e: p for e, p in self.covariance.items() if e in used}

    def _update(self, observation):
        """The joint_block likelihood of one recorded cycle: per entry label and hypothesis the offset covariance, per state
        its offset mean, a Kalman step after the density, each hypothesis under the covariances of its own fit."""
        states, idx = self.states, observation.index
        hidx = self.rules.combo_index(states.pos)
        offset = states.payload["offset"]
        within = self.within[:, idx][:, :, idx]
        for e in np.unique(states.entry):
            rows = np.nonzero(states.entry == e)[0]
            present = np.unique(hidx[rows])
            where = np.searchsorted(present, hidx[rows])
            P = self.covariance[e]
            gains = P[present][:, idx, :]
            S = gains[:, :, idx] + within[present]
            cholesky = np.linalg.cholesky(S)
            inverse = np.linalg.inv(cholesky)
            residual = observation.x - (self.tables[:, hidx[rows]][:, :, idx] + offset[:, rows][:, :, idx])
            z = inverse[where] @ residual.transpose(1, 2, 0)
            logdet = 2 * np.log(np.diagonal(cholesky, axis1=1, axis2=2)).sum(axis=1)
            states.logw[:, rows] += -0.5 * (len(idx) * math.log(2 * math.pi) + logdet[where][None, :] + (z ** 2).sum(axis=1).T)
            gain = np.linalg.solve(S, gains).transpose(0, 2, 1)
            offset[:, rows, :] += (gain[where] @ residual.transpose(1, 2, 0)).transpose(2, 0, 1)
            updated = P.copy()
            updated[present] = P[present] - gain @ gains
            self.covariance[e] = updated
        if not np.isfinite(states.logw).any(axis=1).all():  # a fit under which no state has a positive likelihood
            raise ValueError("the readings have zero likelihood under every state of a fit, so no belief follows them")
        states.logw -= np.max(states.logw, axis=1, keepdims=True)

    # ----- the belief at the current cycle -----
    def weights(self):
        """(R, n) normalized state weights per fit, row 0 the main fit."""
        w = np.exp(self.states.logw - logsumexp(self.states.logw, axis=1)[:, None])
        return w / w.sum(axis=1, keepdims=True)

    def posterior(self):
        """The belief collapsed to the combinations: probabilities per combination under the main fit and the resampled
        fits, and per combination the offset of its block moment matched over its states."""
        return collapse(self, self.states, self.weights(), self.covariance)

    def ahead(self, cycles, plan=()):
        """A copy of the filter after the predict step of the cycles now to now + cycles - 1 without readings, with the
        closures of the open work orders on their released close cycles and the hypothetical closures of a plan
        [(component, mode, close cycle)], so that ahead(1) holds the belief that a reading recorded during the current cycle
        meets, after its kernels and before the closures at its end. The resampled fits move through the same kernels."""
        future = list(self.episode.pending) + [released_episode.Closure(close_t, component, "agent", "plan", mode) for component, mode, close_t in plan]
        twin = self.copy()
        for t in range(self.time, self.time + cycles):
            twin.predict(t, future)
        return twin

    def next_reading(self):
        """A copy of the filter at the belief that the next reading of one cycle meets. A read that plays its
        cycle records it during the current cycle, so the belief is ahead(1). A read of zero duration serves it at once
        while the clock stands, and so does a read of a recorded cycle on a row that reads a record, which moves no clock,
        so the belief is the one now, after the open work orders of a row without a clock, whose visit closes in priority
        order before the next call that is not an order. The belief now is the belief at the cycle of
        that reading wherever no rule card moved the state since, and an approximation elsewhere."""
        if self.view.rules["durations"]["read_per_cycle"] > 0 and not self.view.reads_record:
            return self.ahead(1)
        twin = self.copy()
        if not self.episode.clock:
            twin.settle(standing=rule_card_kernels.cycle_slots(self.episode.pending, self.rules).get(self.time, {}).get("in_cycle", ()))
        return twin


def collapse(flt, states, weights, covariance):
    """The Posterior over the combinations from state rows and their weights (R, n). A combination whose states all
    entered their block under one label carries one offset with its covariance under that label, and a combination whose
    states differ carries the moment matched mixture of its states, each state weighed by its share of the combination's own
    mass. The shares come from the log weights of the main fit and not from the normalized weights, since a mass below the
    smallest normal float holds its weights with a few bits only, which would leave the mixed covariance quantized,
    asymmetric and without a positive eigenvalue. So the moments of a combination do not depend on its mass, and its
    covariance is a convex mix of the covariances of its states plus a Gram matrix, symmetric and positive semidefinite at
    any mass a float holds."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior

    space = flt.space
    H = len(space.combinations)
    d = len(space.fields)
    hidx = flt.rules.combo_index(states.pos)
    probabilities = np.zeros((weights.shape[0], H))
    np.add.at(probabilities.T, hidx, weights.T)
    main = weights[0]
    mass = probabilities[0]
    offset = states.payload["offset"][0]
    logw = states.logw[0]
    mean = np.zeros((H, d))
    collapsed = flt.block.copy()
    for h in np.nonzero(mass > 0)[0]:
        rows = np.nonzero((hidx == h) & (main > 0))[0]
        entries = np.unique(states.entry[rows])
        if len(entries) == 1 and np.all(offset[rows] == offset[rows[0]]):
            mean[h] = offset[rows[0]]
            collapsed[h] = covariance[int(entries[0])][h]
            continue
        share = np.exp(logw[rows] - logw[rows].max())
        share /= share.sum()
        mean[h] = share @ offset[rows]
        deviation = np.sqrt(share)[:, None] * (offset[rows] - mean[h])
        collapsed[h] = sum(s * covariance[states.entry[r]][h] for s, r in zip(share, rows)) + deviation.T @ deviation
    posterior = Posterior(flt.effect, probabilities[0], mean, collapsed, resampled=probabilities[1:] if len(probabilities) > 1 else None,
                          time=flt.time, n_states=len(states), observations=flt.observations)
    # E4 reads the block of the first combination, so its entry is that of the heaviest state among that combination's states
    rows = np.nonzero(hidx == posterior.order()[0])[0]
    posterior.entry_of_top = int(states.entry[rows[np.argmax(main[rows])]])
    return posterior
