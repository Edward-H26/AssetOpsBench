"""The v1 ranking model, kept for ablation A12 (docs/hydraulic_rig/causal-diagnosis-server.md): one linear discriminant
per sensor group and component reads a level from the summaries, and repeated readings of one block combine by a Polya
urn around the confusion row of a cross fit within the pool. It is fitted on the same pool and the same 34 summaries as
the joint model, so the ablation compares the observation models and nothing else. Standard library arithmetic of
envs.hydraulic_rig.classifiers, as in v1.
"""
from envs.hydraulic_rig.classifiers import ALPHA_GRID, cross_fitted_predictions, dm_log_likelihood, lda_fit, lda_predict, scatter
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS

from .effect_model import GROUP_FIELDS, SUMMARY_FIELDS, n_faulty
from .posterior import candidates, segments


def fit_discriminant_urn(data, fold_of):
    """Per sensor group: the discriminants fitted on the pool, the confusion rows of the cross fit counted on the pool's
    faulty combinations, the urn concentrations on ALPHA_GRID and the held out accuracy per component."""
    cycle_combo, x_all = {}, {}
    for i, h in enumerate(data.pool):
        for r, row in enumerate(data.arrays[i]):
            cycle_combo[(i, r)] = h
            x_all[(i, r)] = row
    level_index = {h: tuple(LEVELS[c].index(v) for c, v in zip(COMPONENTS, h)) for h in data.pool}
    out = {}
    for group, fields in GROUP_FIELDS.items():
        columns = [SUMMARY_FIELDS.index(f) for f in fields]
        x = {key: [float(row[c]) for c in columns] for key, row in x_all.items()}
        keys = list(x)
        total = scatter([x[k] for k in keys])
        models = []
        for j, comp in enumerate(COMPONENTS):
            sums = [[0.0] * len(fields) for _ in LEVELS[comp]]
            counts = [0] * len(LEVELS[comp])
            for key in keys:
                k = level_index[cycle_combo[key]][j]
                counts[k] += 1
                sums[k] = [a + b for a, b in zip(sums[k], x[key])]
            mean, scale, coef, bias = lda_fit(total, sums, counts)
            models.append({"mean": mean, "scale": scale, "coef": coef, "bias": bias})
        pred = cross_fitted_predictions(x, cycle_combo, fold_of)
        source = [key for key in keys if n_faulty(cycle_combo[key]) > 0]
        rows, alpha, accuracy = [], [], []
        for j, comp in enumerate(COMPONENTS):
            n = len(LEVELS[comp])
            counts = [[0] * n for _ in range(n)]
            per_block = {}
            for key in source:
                counts[level_index[cycle_combo[key]][j]][pred[key][j]] += 1
                per_block.setdefault(cycle_combo[key], [0] * n)[pred[key][j]] += 1
            smoothed = [[(c + 1.0) / (sum(r) + n) for c in r] for r in counts]
            base = [smoothed[level_index[h][j]] for h in per_block]
            rows.append(smoothed)
            alpha.append(max(ALPHA_GRID, key=lambda a: dm_log_likelihood(a, base, list(per_block.values()))))
            accuracy.append(sum(counts[k][k] for k in range(n)) / len(source))
        out[group] = {"fields": list(fields), "lda": models, "rows": rows, "alpha": alpha, "accuracy": accuracy, "row_cycles": len(source)}
    return out


def classify(group_model, values):
    x = [values[f] for f in group_model["fields"]]
    return tuple(lda_predict((m["mean"], m["scale"], m["coef"], m["bias"]), x) for m in group_model["lda"])


def richest_group(models, fields):
    """The largest sensor group whose summaries the observation all carries, which reads a cycle once."""
    present = set(fields)
    groups = [g for g, m in models.items() if set(m["fields"]) <= present]
    return max(groups, key=lambda g: len(models[g]["fields"])) if groups else None


class UrnBelief:
    """The v1 belief: hypotheses per start combination with their current combination and urn counts per group."""

    def __init__(self, models, starts):
        self.models = models
        self.hypotheses = {h: self._fresh(h, 1.0 / len(starts)) for h in starts}

    def _fresh(self, current, weight):
        return {"current": current, "weight": weight, "counts": {g: [[0] * len(LEVELS[c]) for c in COMPONENTS] for g in self.models}}

    def observe(self, group, values):
        model = self.models[group]
        read = classify(model, values)
        rows, alpha = model["rows"], model["alpha"]
        for s in self.hypotheses.values():
            idx = tuple(LEVELS[c].index(v) for c, v in zip(COMPONENTS, s["current"]))
            like = 1.0
            for j in range(len(COMPONENTS)):
                n = s["counts"][group][j]
                like *= (alpha[j] * rows[j][idx[j]][read[j]] + n[read[j]]) / (alpha[j] + sum(n))
            s["weight"] *= like
            for j in range(len(COMPONENTS)):
                s["counts"][group][j][read[j]] += 1
        total = sum(s["weight"] for s in self.hypotheses.values())
        for s in self.hypotheses.values():
            s["weight"] /= total

    def replace(self, component):
        """Where the component was faulty the environment serves a new block, so the urn starts again; where it was
        healthy the block and its counts stay. Recovery does not end a v2 episode, so no hypothesis is dropped."""
        j = COMPONENTS.index(component)
        for s in self.hypotheses.values():
            if s["current"][j] != HEALTHY[component]:
                after = s["current"][:j] + (HEALTHY[component],) + s["current"][j + 1:]
                s.update(self._fresh(after, s["weight"]))


def urn_probabilities(models, rules, readings, closed_orders):
    """Posterior over the start combinations of the public rule by the v1 model, in the order of posterior.candidates:
    the readings are replayed in time order, each distinct cycle once at its richest group, with the closed orders applied
    at their closing cycle."""
    starts = candidates(rules)
    belief = UrnBelief(models, starts)
    applied = ()
    for segment in segments(readings, closed_orders):
        for order in segment.orders[len(applied):]:
            belief.replace(order.component)
        applied = segment.orders
        for observation in segment.observations:
            group = richest_group(models, observation.fields)
            if group is not None:
                belief.observe(group, dict(zip(observation.fields, observation.x)))
    return [belief.hypotheses[h]["weight"] for h in starts]
