import itertools
from dataclasses import dataclass
from functools import cached_property
import numpy as np


@dataclass(frozen=True)
class EffectSpace:
    """The grid and the fields an effect model is fitted on: the components in combination order, their levels in the order
    of the asset tables, their healthy levels, the summary fields and the summaries of every sensor group."""
    components: tuple
    levels: tuple
    healthy: tuple
    fields: tuple
    groups: dict

    @classmethod
    def from_view(cls, view):
        return cls(tuple(view.components), tuple(tuple(view.levels[c]) for c in view.components),
                   tuple(view.healthy[c] for c in view.components), tuple(view.summary_fields),
                   {g: tuple(view.group_fields(g)) for g in view.sensor_groups})

    @cached_property
    def combinations(self):
        return tuple(itertools.product(*self.levels))

    @cached_property
    def combo_index(self):
        return {combo: i for i, combo in enumerate(self.combinations)}

    @cached_property
    def field_index(self):
        return {field: i for i, field in enumerate(self.fields)}

    @cached_property
    def main_terms(self):
        return tuple((j, level) for j, levels in enumerate(self.levels) for level in levels if level != self.healthy[j])

    @cached_property
    def pairs(self):
        return tuple((j, k) for j in range(len(self.components)) for k in range(j + 1, len(self.components)))

    @property
    def healthy_combination(self):
        return tuple(self.healthy)

    def n_faulty(self, combo):
        return sum(1 for j, v in enumerate(combo) if v != self.healthy[j])

    def faulty(self, combo):
        return [c for j, c in enumerate(self.components) if combo[j] != self.healthy[j]]

    def field_indices(self, fields):
        return np.array([self.field_index[f] for f in fields], dtype=int)

    def to_json(self):
        return {"components": list(self.components), "levels": [list(levels) for levels in self.levels], "healthy": list(self.healthy),
                "fields": list(self.fields), "groups": {g: list(f) for g, f in self.groups.items()}}

    @classmethod
    def from_json(cls, obj):
        return cls(tuple(obj["components"]), tuple(tuple(levels) for levels in obj["levels"]), tuple(obj["healthy"]), tuple(obj["fields"]),
                   {g: tuple(f) for g, f in obj["groups"].items()})


def interaction_terms(space, pair):
    """One indicator per pair of non healthy levels of the two components."""
    j, k = pair
    return tuple((pair, lj, lk) for lj in space.levels[j] if lj != space.healthy[j] for lk in space.levels[k] if lk != space.healthy[k])


def terms_of(space, pairs):
    return tuple(term for pair in pairs for term in interaction_terms(space, pair))


def design_matrix(space, combos, pairs):
    """Intercept, main effects of the non healthy levels, then the interaction indicators of the selected pairs."""
    combos = np.asarray(list(combos))
    columns = [np.ones(len(combos))]
    columns += [(combos[:, j] == level).astype(float) for j, level in space.main_terms]
    columns += [((combos[:, j] == lj) & (combos[:, k] == lk)).astype(float) for (j, k), lj, lk in terms_of(space, pairs)]
    return np.column_stack(columns)


def full_rank(x):
    return np.linalg.matrix_rank(x) == x.shape[1]


def weighted_fit(x, y, weights):
    """Least squares of the block means with the block lengths as weights, which equals the least squares over cycles."""
    xtw = x.T * weights
    return np.linalg.solve(xtw @ x, xtw @ y)


def parse_combo(key):
    return tuple(int(v) for v in key.split(","))
