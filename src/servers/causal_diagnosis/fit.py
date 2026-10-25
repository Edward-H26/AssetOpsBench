"""Fit the fixed-pool models of the causal-diagnosis server once and write them to models.json beside this file, a file
generated from the public rig data and never committed (ensure_models fits it on first use in the drivers and tests)
(docs/hydraulic_rig/causal-diagnosis-server.md: models fitted on a fixed pool, never per episode).

The pool is the stable cycles of every combination outside the evaluation class S2, that is the combinations with one,
three or four faulty components and the healthy one, so that no model has seen an S2 combination. Per measurement tier
the file holds the per-component LDA classifiers fitted on the pool, the confusion rows counted on the pool's one- and
three-fault combinations from a cross-fit within the pool, the urn concentrations and the held-out accuracies, and the
healthy band of every served summary; per pool combination it holds the additive effect model of the 17 channel means
fitted on the other pool combinations, and for combinations outside the pool the fit on the whole pool. Standard
library only. The LDA parameters are kept at full precision, since the classifications must not change with the file;
bands and quantiles are rounded to the six significant digits at which values are served.

usage: HYD_FACTS_PATH=<facts file> python -m servers.causal_diagnosis.fit     (src/ on sys.path, as under uv run)
"""
import datetime
import hashlib
from pathlib import Path

from envs.hydraulic_rig.classifiers import ALPHA_GRID, COMBOS, TIERS, cross_fitted_predictions, dm_log_likelihood, lda_fit, scatter, solve_spd, stratified_folds
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, build_blocks, default_facts_path, load_cycles, tier_features
from envs.hydraulic_rig.env import n_faulty
from servers.hydraulic_rig.names import public_name
from servers.json_files import write_json_atomically

from .effect import BAND, EFFECT_GRID, combo_key, design, percentile
from .knowledge import INDICATES

MODELS_PATH = Path(__file__).resolve().with_name("models.json")
EVALUATION_FAULTS = 2  # the evaluation class S2 holds every two-fault combination; the pool is everything else
ROW_FAULTS = (1, 3)  # confusion rows are counted on the pool's one- and three-fault combinations, the neighbours of S2
FOLDS, FOLD_SEED = 8, 0


def rounded(value):
    return float(f"{value:.6g}")


def fit(facts_path):
    cycles = load_cycles(facts_path)
    blocks = build_blocks(cycles)
    by_index = {c.index: c for c in cycles}
    cycle_combo = {i: h for h in COMBOS for i in blocks[h]}
    pool = [h for h in COMBOS if n_faulty(h) != EVALUATION_FAULTS]
    pool_cycles = [i for h in pool for i in blocks[h]]
    level_index = {h: tuple(LEVELS[c].index(v) for c, v in zip(COMPONENTS, h)) for h in COMBOS}
    out = {
        "format": 1,
        "generated": datetime.date.today().isoformat(),
        "facts_sha256": hashlib.sha256(Path(facts_path).read_bytes()).hexdigest(),
        "records": len(cycles),
        "pool": {"rule": f"stable cycles of every combination whose number of faulty components is not {EVALUATION_FAULTS}",
                 "combinations": [combo_key(h) for h in pool], "cycles": len(pool_cycles), "row_faults": list(ROW_FAULTS),
                 "folds": FOLDS, "fold_seed": FOLD_SEED},
        "components": list(COMPONENTS), "levels": {c: list(LEVELS[c]) for c in COMPONENTS}, "healthy": dict(HEALTHY),
        "tiers": {}, "bands": {}, "band_quantiles": list(BAND),
    }
    fold_all = stratified_folds(FOLDS, FOLD_SEED)
    fold_pool = {h: fold_all[h] for h in pool}
    for tier in TIERS:
        feats = {i: tier_features(by_index[i], tier) for i in pool_cycles}
        names = sorted(next(iter(feats.values())))
        x = {i: [feats[i][n] for n in names] for i in pool_cycles}
        total = scatter([x[i] for i in pool_cycles])
        models = []
        for j, comp in enumerate(COMPONENTS):
            sums = [[0.0] * len(names) for _ in LEVELS[comp]]
            counts = [0] * len(LEVELS[comp])
            for i in pool_cycles:
                k = level_index[cycle_combo[i]][j]
                counts[k] += 1
                sums[k] = [a + b for a, b in zip(sums[k], x[i])]
            mean, scale, coef, bias = lda_fit(total, sums, counts)
            models.append({"mean": mean, "scale": scale, "coef": coef, "bias": bias})
        pred = cross_fitted_predictions(x, {i: cycle_combo[i] for i in pool_cycles}, fold_pool)
        source = [i for i in pool_cycles if n_faulty(cycle_combo[i]) in ROW_FAULTS]
        rows, alpha, accuracy = [], [], []
        for j, comp in enumerate(COMPONENTS):
            n = len(LEVELS[comp])
            counts = [[0] * n for _ in range(n)]
            per_block = {}
            for i in source:
                counts[level_index[cycle_combo[i]][j]][pred[i][j]] += 1
                per_block.setdefault(cycle_combo[i], [0] * n)[pred[i][j]] += 1
            smoothed = [[(c + 1.0) / (sum(r) + n) for c in r] for r in counts]
            base = [smoothed[level_index[h][j]] for h in per_block]
            rows.append(smoothed)
            alpha.append(max(ALPHA_GRID, key=lambda a: dm_log_likelihood(a, base, list(per_block.values()))))
            accuracy.append(sum(counts[k][k] for k in range(n)) / len(source))
        out["tiers"][str(tier)] = {"names": names, "public_names": [public_name(n) for n in names], "lda": models,
                                   "rows": rows, "alpha": alpha, "accuracy": accuracy, "row_cycles": len(source)}
        for name in names:
            pub = public_name(name)
            if pub in out["bands"]:
                continue
            channel = name.split("_")[1]
            comps = INDICATES.get(channel, ())
            vals = sorted(feats[i][name] for i in pool_cycles if all(cycle_combo[i][COMPONENTS.index(c)] == HEALTHY[c] for c in comps))
            out["bands"][pub] = {"low": rounded(percentile(vals, BAND[0])), "high": rounded(percentile(vals, BAND[1])), "cycles": len(vals),
                                 "healthy": list(comps)}
    # the additive effect model of the channel means: one fit per pool combination on the other pool combinations, and
    # one on the whole pool for the combinations no fit has seen
    feats3 = {i: tier_features(by_index[i], 3) for i in pool_cycles}
    mean_names = sorted(n for n in next(iter(feats3.values())) if n.endswith("_mean"))
    d_of = {h: design(h) for h in pool}
    dim = len(d_of[pool[0]])
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = {n: [0.0] * dim for n in mean_names}
    ysum = {h: {n: sum(feats3[i][n] for i in blocks[h]) for n in mean_names} for h in pool}
    for h in pool:
        d, n_h = d_of[h], len(blocks[h])
        for a in range(dim):
            for b in range(dim):
                xtx[a][b] += n_h * d[a] * d[b]
            for n in mean_names:
                xty[n][a] += d[a] * ysum[h][n]

    def betas(exclude):
        """Coefficients per channel of the fit on the pool without `exclude` (None for the whole pool)."""
        if exclude is None:
            a, rhs = xtx, [xty[n] for n in mean_names]
        else:
            d_s, n_s = d_of[exclude], len(blocks[exclude])
            a = [[xtx[r][c] - n_s * d_s[r] * d_s[c] for c in range(dim)] for r in range(dim)]
            rhs = [[xty[n][r] - d_s[r] * ysum[exclude][n] for r in range(dim)] for n in mean_names]
        return dict(zip(mean_names, solve_spd(a, rhs)))

    def residual_quantiles(beta, exclude):
        quantiles = {}
        for n in mean_names:
            fitted = {h: sum(b * v for b, v in zip(beta[n], d_of[h])) for h in pool if h != exclude}
            resid = sorted(feats3[i][n] - fitted[h] for h in fitted for i in blocks[h])
            quantiles[n] = [rounded(percentile(resid, q)) for q in EFFECT_GRID]
        return quantiles

    out["effect_grid"] = list(EFFECT_GRID)
    out["effect"] = {}
    for h_star in pool:
        beta = betas(h_star)
        quantiles = residual_quantiles(beta, h_star)
        out["effect"][combo_key(h_star)] = {public_name(n): [rounded(sum(b * v for b, v in zip(beta[n], d_of[h_star])))] + quantiles[n]
                                            for n in mean_names}
    beta = betas(None)
    quantiles = residual_quantiles(beta, None)
    out["effect_pool"] = {public_name(n): {"beta": beta[n], "q": quantiles[n]} for n in mean_names}
    return out


def write_models(path=MODELS_PATH):
    out = fit(default_facts_path())
    write_json_atomically(path, out)
    return out


def ensure_models(path=MODELS_PATH):
    if not path.exists():
        write_models(path)
    return path


def main():
    out = write_models()
    print(f"wrote {MODELS_PATH.name}: {out['records']} records, pool {len(out['pool']['combinations'])} combinations and {out['pool']['cycles']} cycles, "
          f"{len(out['bands'])} bands, {len(out['effect'])} effect models, {MODELS_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
