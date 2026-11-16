"""The v1 ranking model, kept for the ranking model ablation (CausalDiagnosis/docs/causal-diagnosis-mcp.md, section
Ranking models): one linear discriminant per sensor group and component reads a level from the summaries, and repeated
readings of one block combine by a Polya urn around the confusion row of a cross fit. It is fitted once per episode on the
recorded blocks without the start's block, with the folds and the summaries of the effect model, so no part of it reads the
block the readings come from and the ablation compares the observation models. Its hypotheses are every combination of the profile with a uniform prior, a closed work order restores its
component as in v1, and it has no transition model, so the ablation runs it on static episodes. Standard library
arithmetic of CausalDiagnosis.environment.reference_policies.tier_classifiers, as in v1.
"""
from CausalDiagnosis.environment.reference_policies.tier_classifiers import ALPHA_GRID, dm_log_likelihood, lda_fit, lda_predict, scatter

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import level_predictions
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import merge_observations


def fit_discriminant_urn(space, data, fold_of):
    """Per sensor group: the discriminants fitted on the blocks, the confusion rows of the cross fit counted on the faulty
    combinations, the urn concentrations on ALPHA_GRID and the held out accuracy per component."""
    cycle_combo, x_all = {}, {}
    for i, h in enumerate(data.pool):
        for r, row in enumerate(data.arrays[i]):
            cycle_combo[(i, r)] = h
            x_all[(i, r)] = row
    level_index = {h: tuple(space.levels[j].index(v) for j, v in enumerate(h)) for h in data.pool}
    out = {}
    for group, fields in space.groups.items():
        columns = [space.field_index[f] for f in fields]
        x = {key: [float(row[c]) for c in columns] for key, row in x_all.items()}
        keys = list(x)
        total = scatter([x[k] for k in keys])
        models = []
        for j, levels in enumerate(space.levels):
            sums = [[0.0] * len(fields) for _ in levels]
            counts = [0] * len(levels)
            for key in keys:
                k = level_index[cycle_combo[key]][j]
                counts[k] += 1
                sums[k] = [a + b for a, b in zip(sums[k], x[key])]
            mean, scale, coef, bias = lda_fit(total, sums, counts)
            models.append({"mean": mean, "scale": scale, "coef": coef, "bias": bias})
        pred = level_predictions(x, cycle_combo, fold_of, space)
        source = [key for key in keys if space.n_faulty(cycle_combo[key]) > 0]
        rows, alpha, accuracy = [], [], []
        for j, levels in enumerate(space.levels):
            n = len(levels)
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
    """The v1 belief: one hypothesis per combination of the profile at the first reading, with its current combination
    after the released closures and its urn counts per group."""

    def __init__(self, models, space):
        self.models = models
        self.space = space
        self.hypotheses = {h: self._fresh(h, 1.0 / len(space.combinations)) for h in space.combinations}

    def _fresh(self, current, weight):
        return {"current": current, "weight": weight, "counts": {g: [[0] * len(levels) for levels in self.space.levels] for g in self.models}}

    def observe(self, group, values):
        model = self.models[group]
        read = classify(model, values)
        rows, alpha = model["rows"], model["alpha"]
        for s in self.hypotheses.values():
            idx = tuple(self.space.levels[j].index(v) for j, v in enumerate(s["current"]))
            like = 1.0
            for j in range(len(self.space.components)):
                n = s["counts"][group][j]
                like *= (alpha[j] * rows[j][idx[j]][read[j]] + n[read[j]]) / (alpha[j] + sum(n))
            s["weight"] *= like
            for j in range(len(self.space.components)):
                s["counts"][group][j][read[j]] += 1
        total = sum(s["weight"] for s in self.hypotheses.values())
        for s in self.hypotheses.values():
            s["weight"] /= total

    def replace(self, component):
        """Where the component was faulty the environment serves a new block, so the urn starts again; where it was
        healthy the block and its counts stay."""
        j = self.space.components.index(component)
        healthy = self.space.healthy[j]
        for s in self.hypotheses.values():
            if s["current"][j] != healthy:
                s.update(self._fresh(s["current"][:j] + (healthy,) + s["current"][j + 1:], s["weight"]))


def urn_probabilities(models, space, readings, closures):
    """The posterior over the current combinations by the v1 model: the readings are replayed in time order, each distinct
    cycle once at its richest group, with every released closure applied at its cycle. Returns {combination: mass}."""
    belief = UrnBelief(models, space)
    pending = sorted(closures, key=lambda c: (c.t, c.id))
    for observation in merge_observations(readings, space):
        while pending and pending[0].t <= observation.t:
            belief.replace(pending.pop(0).component)
        group = richest_group(models, observation.fields)
        if group is not None:
            belief.observe(group, dict(zip(observation.fields, observation.x)))
    for closure in pending:
        belief.replace(closure.component)
    out = {}
    for s in belief.hypotheses.values():
        out[s["current"]] = out.get(s["current"], 0.0) + s["weight"]
    return out
