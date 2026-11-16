"""Synthetic effect models for the unit tests: a small grid of two components, or the rig's grid, with a mean table set
by hand, so that a test controls which readings separate which combinations."""
import itertools

import numpy as np

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import Costs
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import EffectModel, EffectSpace, design_matrix

TINY = EffectSpace(components=("A", "B"), levels=((0, 1, 2), (0, 1, 2, 3)), healthy=(0, 0), fields=("f1", "f2", "f3", "f4"),
                   groups={"g1": ("f1",), "g2": ("f2",), "g3": ("f3",), "g4": ("f4",)})


class FixedTables:
    """Resampled mean tables of E6 given by hand, served in the order the filter asks for them."""

    def __init__(self, tables):
        self._tables = [np.asarray(table, dtype=float) for table in tables]
        self.count = len(self._tables)

    def table(self, r):
        return self._tables[r].copy()


def effect_with_means(space, mu_table, noise=0.05, block=0.0, resamples=0, jitter=0.0, seed=0):
    """An EffectModel of the space whose mean table is mu_table (combinations, fields), with isotropic within noise and an
    isotropic block offset shared by every hypothesis, and resampled tables jittered around the mean table."""
    d = len(space.fields)
    terms = design_matrix(space, space.combinations[:1], []).shape[1]
    rng = np.random.default_rng(seed)
    mu_table = np.asarray(mu_table, dtype=float)
    resampled = [mu_table + jitter * rng.normal(size=mu_table.shape) for _ in range(resamples)]
    return EffectModel(space, [], mu_table, noise ** 2 * np.eye(d), block ** 2 * np.eye(d), np.zeros(d), np.ones(d),
                       coefficients=lambda combo: np.zeros((terms, d)), resampler=FixedTables(resampled) if resamples else None)


def table_of(space, mean_of):
    """The mean table whose row for combination h is mean_of(h), a vector over the fields."""
    return np.array([mean_of(dict(zip(space.components, h))) for h in space.combinations], dtype=float)


def tiny_costs(space=TINY, prices=None, horizon=60):
    """E7 costs on the tiny grid: every read costs its price and one cycle, every faulty level loses one unit per cycle."""
    prices = prices or {g: 1.0 for g in space.groups}
    healthy = dict(zip(space.components, space.healthy))
    rates = {c: {level: 0 if level == healthy[c] else 1 for level in levels} for c, levels in zip(space.components, space.levels)}
    missed = {c: {level: rate * horizon for level, rate in rates[c].items()} for c in space.components}
    return Costs(tuple(space.components), healthy, {c: 40.0 for c in space.components}, missed, dict(prices), {g: 1 for g in prices}, horizon, rates)


def grid(space):
    return list(itertools.product(*space.levels))
