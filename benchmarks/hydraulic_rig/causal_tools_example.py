"""The worked episode of docs/hydraulic_rig/causal-diagnosis-server.md, played with models fitted here from the recorded
cycles (the server under src/servers/causal_diagnosis/ fits the same models and takes its bands from stored residual
quantiles), and, printed last, the folds behind the page's case against a cross-fit per episode.

The episode is an S2 episode as docs/hydraulic_rig/environment.md defines one: start combination (100, 80, 0, 90), the
public rule "exactly two components off healthy" (min_faulty = max_faulty = 2), seed 0. Every model the design names
is fitted once on the fixed pool, the stable cycles of every combination outside the evaluation class S2, so that no
model depends on the episode: per-tier, per-component LDA classifiers (src/envs/hydraulic_rig/classifiers.py) with confusion rows
from a cross-fit within the pool, counted on its one- and three-fault combinations, the healthy bands, and an additive
model for predicted summaries. Repeated readings of one recorded block are combined by the Polya urn of
src/envs/hydraulic_rig/greedy.py, and a cycle read again at the same tier adds nothing. The tools read a
measurement as the agent sees it (names.public_result, six significant digits). Standard library only.

usage, from the repository root: python3 benchmarks/hydraulic_rig/causal_tools_example.py
"""
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))  # the fork's src/, also outside an editable install
from servers.hydraulic_rig import episode  # noqa: E402
from servers.hydraulic_rig.names import public_name, public_result  # noqa: E402
from envs.hydraulic_rig.classifiers import (ALPHA_GRID, COMBOS, TIERS, cross_fitted_predictions, dm_log_likelihood, lda_fit,  # noqa: E402
                                    lda_predict, scatter, signature, solve_spd, stratified_folds)
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, default_facts_path, tier_features  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv, n_faulty  # noqa: E402

START, SEED = (100, 80, 0, 90), 0
S2 = [h for h in COMBOS if n_faulty(h) == 2]
POOL = [h for h in COMBOS if n_faulty(h) != 2]
CONFIDENCE = 0.95  # the threshold of src/envs/hydraulic_rig/greedy.py below which the ranking reports insufficient evidence
# The pool holds no two-fault combination, so the confusion rows are counted on its one- and three-fault combinations,
# the neighbours of S2; the classifiers themselves are fitted on the whole pool.
ROW_FAULTS = (1, 3)
# A channel indicates a component when it, or the virtual sensor served as CE, CP or SE, is among the component's most
# correlated features in Helwig et al. 2015, Table II (p. 3); the temperatures follow the cooler (pp. 3 to 4).
INDICATES = {
    "CE": ("cooler",), "CP": ("cooler",), "TS1": ("cooler",), "TS2": ("cooler",), "TS3": ("cooler",), "TS4": ("cooler",),
    "PS1": ("valve", "accumulator"), "PS2": ("valve", "accumulator"), "PS3": ("valve", "pump", "accumulator"),
    "FS1": ("valve", "pump", "accumulator"), "EPS1": ("valve", "accumulator"), "SE": ("valve", "pump", "accumulator"),
}

cycles, blocks = episode.load_world(str(default_facts_path()))
by_index = {c.index: c for c in cycles}
cycle_combo = {i: h for h in COMBOS for i in blocks[h]}
pool_cycles = [i for h in POOL for i in blocks[h]]
feats = {t: {i: tier_features(by_index[i], t) for i in cycle_combo} for t in TIERS}
NAMES = {t: sorted(next(iter(feats[t].values()))) for t in TIERS}
level_index = {h: tuple(LEVELS[c].index(v) for c, v in zip(COMPONENTS, h)) for h in COMBOS}


def vector(tier, values):
    return [values[n] for n in NAMES[tier]]


def released_vector(tier, released):
    return [released[public_name(n)] for n in NAMES[tier]]


def fit_models(tier):
    x = [vector(tier, feats[tier][i]) for i in pool_cycles]
    total = scatter(x)
    models = []
    for j, comp in enumerate(COMPONENTS):
        sums = [[0.0] * len(NAMES[tier]) for _ in LEVELS[comp]]
        counts = [0] * len(LEVELS[comp])
        for i, row in zip(pool_cycles, x):
            k = level_index[cycle_combo[i]][j]
            counts[k] += 1
            sums[k] = [a + b for a, b in zip(sums[k], row)]
        models.append(lda_fit(total, sums, counts))
    return models


def confusion_rows(tier):
    """rows[j][k]: distribution of the predicted level of component j when its true level index is k, from a cross-fit
    within the pool (8 folds of the stratified fold rule at fold seed 0, restricted to the pool), counted on the cycles
    of the ROW_FAULTS combinations and add-one smoothed; alpha[j]: the Dirichlet-multinomial concentration of repeated
    readings within one block (maximum likelihood on ALPHA_GRID, as classifiers.observation_model); accuracy[j]: the
    share of those held-out readings that equal the true level."""
    fold_all = stratified_folds(8, 0)
    pred = cross_fitted_predictions({i: vector(tier, feats[tier][i]) for i in pool_cycles},
                                    {i: cycle_combo[i] for i in pool_cycles}, {h: fold_all[h] for h in POOL})
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
    return rows, alpha, accuracy


MODELS = {t: fit_models(t) for t in TIERS}
ROWS, ALPHA, ACCURACY = {}, {}, {}
for _t in TIERS:
    ROWS[_t], ALPHA[_t], ACCURACY[_t] = confusion_rows(_t)


def read(tier, result):
    """The fixed-pool classifiers applied to the released values, checked against the full-precision values."""
    released = public_result(result)
    p = tuple(lda_predict(m, released_vector(tier, released)) for m in MODELS[tier])
    assert p == tuple(lda_predict(m, vector(tier, result)) for m in MODELS[tier])
    return p, released


def fresh_block():
    """Readings of one served block: urn counts and the cycles already read, per tier."""
    return {"counts": {t: [[0] * len(LEVELS[c]) for c in COMPONENTS] for t in TIERS}, "seen": {t: set() for t in TIERS}}


def update(belief, blocks_read, tier, p, released):
    """Bayes update with the urn of greedy.py: the next reading of a block follows its confusion row pulled towards the
    block's earlier readings at that tier; a cycle read again at the same tier adds nothing."""
    key = signature(released)
    new = {}
    for h, b in belief.items():
        block = blocks_read[h]
        if key in block["seen"][tier]:
            new[h] = b
            continue
        like = 1.0
        for j in range(len(COMPONENTS)):
            n = block["counts"][tier][j]
            like *= (ALPHA[tier][j] * ROWS[tier][j][level_index[h][j]][p[j]] + n[p[j]]) / (ALPHA[tier][j] + sum(n))
        new[h] = b * like
        block["seen"][tier].add(key)
        for j in range(len(COMPONENTS)):
            block["counts"][tier][j][p[j]] += 1
    z = sum(new.values())
    return {h: v / z for h, v in new.items()}


def replace(belief, blocks_read, comp):
    """Move every hypothesis through the replacement. Where the component was faulty the environment serves a new block
    (env.py lines 175 to 179), so the urn starts again; where it was healthy the block, and its readings, stay."""
    j = COMPONENTS.index(comp)
    new_belief, new_blocks, restarted = {}, {}, set()
    for h, b in belief.items():
        after = tuple(HEALTHY[c] if c == comp else v for c, v in zip(COMPONENTS, h))
        reset = h[j] != HEALTHY[comp]
        if after in new_belief:
            # several faulty levels of the replaced component lead to the same combination, all with a new block; a
            # merge of a new block with a kept one cannot occur under a fixed fault count such as the S2 rule, since
            # the fault counts differ, but can under a range of counts, where hypotheses must stay per start combination
            assert reset and after in restarted
            new_belief[after] += b
            continue
        new_belief[after] = b
        new_blocks[after] = fresh_block() if reset else blocks_read[h]
        if reset:
            restarted.add(after)
    return new_belief, new_blocks


def report(label, belief, truth):
    print(f"\n{label}")
    for j, c in enumerate(COMPONENTS):
        pf = sum(b for h, b in belief.items() if h[j] != HEALTHY[c])
        levels = ", ".join(f"{lv}: {sum(b for h, b in belief.items() if h[j] == lv):.3f}" for lv in LEVELS[c])
        print(f"  {c:12s} P(faulty) {pf:.3f}  levels {levels}")
    top = sorted(belief, key=belief.get, reverse=True)
    for h in top[:3]:
        print(f"  {h}  {belief[h]:.3f}")
    print(f"  true {truth}  {belief.get(truth, 0):.3f}; insufficient evidence: {belief[top[0]] < CONFIDENCE}")
    return top[0]


def percentile(sorted_vals, q):
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def band(name, tier, cycle_ids=None):
    """1st to 99th percentile of a summary over the given cycles, by default the pool cycles, in which every component
    the channel indicates is healthy."""
    channel = name.split("_")[1]
    comps = INDICATES.get(channel, ())
    vals = sorted(feats[tier][i][name] for i in (pool_cycles if cycle_ids is None else cycle_ids)
                  if all(cycle_combo[i][COMPONENTS.index(c)] == HEALTHY[c] for c in comps))
    return percentile(vals, 0.01), percentile(vals, 0.99), len(vals), comps


def design(h):
    """Intercept plus one indicator per non-healthy level of each component (healthy levels are the baseline)."""
    row = [1.0]
    for c, v in zip(COMPONENTS, h):
        row += [1.0 if v == lv else 0.0 for lv in LEVELS[c] if lv != HEALTHY[c]]
    return row


def additive_prediction(name, h_after):
    """Predicted summary and residual quantiles for combination h_after, from an additive model of the component levels
    fitted on the pool cycles of every other combination (h_after's own recordings never enter its prediction)."""
    rows, ys = [], []
    for h in POOL:
        if h == h_after:
            continue
        for i in blocks[h]:
            rows.append(design(h))
            ys.append(feats[3][i][name])
    d = len(rows[0])
    xtx = [[sum(r[a] * r[b] for r in rows) for b in range(d)] for a in range(d)]
    xty = [sum(r[a] * y for r, y in zip(rows, ys)) for a in range(d)]
    beta = solve_spd(xtx, [xty])[0]
    resid = sorted(y - sum(b * v for b, v in zip(beta, r)) for r, y in zip(rows, ys))
    return sum(b * v for b, v in zip(beta, design(h_after))), resid


def mixture(name, weights):
    """Posterior-weighted mixture of the additive predictions: mean and 1st to 99th percentile."""
    parts = [(w, *additive_prediction(name, h)) for h, w in weights.items() if w > 1e-6]
    total = sum(w for w, _, _ in parts)
    parts = [(w / total, m, r) for w, m, r in parts]

    def cdf(x):
        out = 0.0
        for w, m, r in parts:
            lo, hi = 0, len(r)
            while lo < hi:
                mid = (lo + hi) // 2
                if r[mid] <= x - m:
                    lo = mid + 1
                else:
                    hi = mid
            out += w * lo / len(r)
        return out

    def quantile(q):
        lo = min(m + r[0] for _, m, r in parts)
        hi = max(m + r[-1] for _, m, r in parts)
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cdf(mid) < q else (lo, mid)
        return (lo + hi) / 2

    return sum(w * m for w, m, _ in parts), quantile(0.01), quantile(0.99)


env = TinyHydraulicEnv(cycles, blocks, **{**episode.default_env_kwargs(), "min_faulty": 2, "max_faulty": 2, "allowed": [START]})
obs = env.reset(SEED)
assert tuple(env.initial) == START
print(f"pool: {len(POOL)} combinations, {len(pool_cycles)} stable cycles; S2 candidates {len(S2)}")
print("observation rule:", obs["min_faulty"], obs["max_faulty"], "budget", obs["budget_left"], "steps", obs["steps_left"], "repairs", obs["repairs_left"])

belief = {h: 1 / len(S2) for h in S2}
blocks_read = {h: fresh_block() for h in S2}
print("urn concentrations (alpha) per tier and component:", {t: ALPHA[t] for t in TIERS})
print("accumulator held-out accuracy on the row cycles, tiers 1 to 3:", [round(ACCURACY[t][3], 3) for t in TIERS])
obs, cost, done, _ = env.step({"type": "measure", "tier": 1})
p1, released1 = read(1, obs["history"][-1]["result"])
print(f"\nmeasure(1): {len(released1)} items; CE_CoolingEfficiency_percent_1Hz_mean = {released1['CE_CoolingEfficiency_percent_1Hz_mean']}")
print("  fixed-pool reading (levels):", {c: LEVELS[c][k] for c, k in zip(COMPONENTS, p1)})
lo, hi, n, comps = band("CoolingEfficiency_CE_mean", 1)
print(f"  CE mean pool band ({comps}): {lo:.6g} to {hi:.6g} over {n} cycles")
belief = update(belief, blocks_read, 1, p1, released1)
report("rank_root_causes after measure(1)", belief, START)

healthy = tuple(HEALTHY[c] for c in COMPONENTS)
ref = [feats[1][i] for i in blocks[healthy]]
print("\nTS means of measure(1) against the block of the healthy combination and against the pool bands")
for ch in ("TS1", "TS2", "TS3"):
    name = f"Temperature_{ch}_mean"
    v = obs["history"][-1]["result"][name]
    mu = statistics.fmean(r[name] for r in ref)
    sd = statistics.pstdev([r[name] for r in ref])
    lo, hi, n, comps = band(name, 1)
    print(f"  {ch}: value {v:.4f}, z against the {len(ref)} cycles of the healthy combination {(v - mu) / sd:+.1f}; pool band ({comps}) {lo:.4f} to {hi:.4f} over {n} cycles, inside {lo <= v <= hi}")

obs, cost, done, _ = env.step({"type": "measure", "tier": 3})
p3, released3 = read(3, obs["history"][-1]["result"])
print("\nmeasure(3): fixed-pool reading (levels):", {c: LEVELS[c][k] for c, k in zip(COMPONENTS, p3)})
belief = update(belief, blocks_read, 3, p3, released3)
best = report("rank_root_causes after measure(3)", belief, START)

acc = COMPONENTS.index("accumulator")
addresses = sum(b for h, b in belief.items() if h[acc] != HEALTHY["accumulator"])
recovery = sum(b for h, b in belief.items() if all(v == HEALTHY[c] for j, (c, v) in enumerate(zip(COMPONENTS, h)) if j != acc))
price = obs["prices"]["parts"]["accumulator"] + obs["prices"]["downtime"]
after, blocks_after = replace(belief, blocks_read, "accumulator")
print(f"\nestimate_intervention_effect(replace:accumulator): addresses {addresses:.3f}, recovery {recovery:.3f}, price {price}")
predicted = {}
for name in ("Pressure_PS1_mean", "VolumeFlow_FS1_mean"):
    predicted[name] = mixture(name, after)
    m, lo, hi = predicted[name]
    print(f"  predicted {public_name(name)}: {m:.6g}, band {lo:.6g} to {hi:.6g}")

diagnosis = dict(zip(COMPONENTS, best))
obs, cost, done, _ = env.step({"type": "submit", "diagnosis": diagnosis})
print("\nsubmit", diagnosis, "phase", obs["phase"])
obs, cost, done, _ = env.step({"type": "replace", "component": "accumulator"})
print("replace(accumulator): cost", cost, "episode over", done, "repairs_left", obs["repairs_left"], "budget_left", obs["budget_left"])
obs, cost, done, _ = env.step({"type": "measure", "tier": 3})
p4, released4 = read(3, obs["history"][-1]["result"])
print("measure(3) after the repair: fixed-pool reading (levels):", {c: LEVELS[c][k] for c, k in zip(COMPONENTS, p4)})
for name, (m, lo, hi) in predicted.items():
    v = released4[public_name(name)]
    print(f"  verify {public_name(name)}: observed {v}, residual {v - m:+.4g}, inside band {lo <= v <= hi}")
belief = update(after, blocks_after, 3, p4, released4)
report("rank_root_causes after the repair and measure(3)", belief, tuple(env.hidden))
valve = COMPONENTS.index("valve")
odds_before = {lv: sum(b for h, b in after.items() if h[valve] == lv) for lv in (80, 90)}
odds_after = {lv: sum(b for h, b in belief.items() if h[valve] == lv) for lv in (80, 90)}
row = {lv: ROWS[3][valve][LEVELS["valve"].index(lv)][p4[valve]] for lv in (80, 90)}
print(f"  valve read as {LEVELS['valve'][p4[valve]]}: tier-3 confusion rows {row[80]:.3f} under level 80, {row[90]:.3f} under "
      f"level 90; odds of 90 against 80 {odds_before[90] / odds_before[80]:.2f} before, {odds_after[90] / odds_after[80]:.2f} after")
addresses = sum(b for h, b in belief.items() if h[valve] != HEALTHY["valve"])
recovery = sum(b for h, b in belief.items() if all(v == HEALTHY[c] for j, (c, v) in enumerate(zip(COMPONENTS, h)) if j != valve))
print(f"estimate_intervention_effect(replace:valve): addresses {addresses:.3f}, recovery {recovery:.3f}, price {obs['prices']['parts']['valve'] + obs['prices']['downtime']}")
obs, cost, done, _ = env.step({"type": "replace", "component": "valve"})
print("\nreplace(valve): the answer shows episode_over", done, "and repairs_left", obs["repairs_left"], "| hidden end reason (log only):", env.log[-1]["reason"] if env.log[-1]["event"] == "end" else env.log[-1])
print("hidden after the repairs (not shown to the agent)", tuple(env.hidden))

# A cross-fit per episode would leave out the fold of the hidden start combination (the 8
# folds of classifiers.stratified_folds at fold seed 0, over all 144 combinations), so the bands it serves name the fold.
fold_of = stratified_folds(8, 0)
default_candidates = [h for h in COMBOS if 1 <= n_faulty(h) <= 2]
s1_combos = [h for h in COMBOS if n_faulty(h) == 1]
print(f"\nfolds of a cross-fit per episode: {len(default_candidates)} candidates of the default rule, {len(s1_combos)} S1 "
      "combinations")
print("  candidates per fold:", {f: sum(fold_of[h] == f for h in default_candidates) for f in range(8)})
print("  S1 combinations per fold:", {f: sum(fold_of[h] == f for h in s1_combos) for f in range(8)})
band_sets = set()
for f in range(8):
    kept = [i for h in COMBOS if fold_of[h] != f for i in blocks[h]]
    band_sets.add(tuple(float(f"{x:.6g}") for name in NAMES[1] for x in band(name, 1, kept)[:2]))
print(f"  distinct sets of the {len(NAMES[1])} tier-1 bands over the 8 folds, at six significant digits: {len(band_sets)}")
