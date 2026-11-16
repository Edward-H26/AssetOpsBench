"""Start states of an environment (CausalDiagnosis/docs/environment.md): the (combination, window start, entry offset)
triples that env.reset can draw. Every combination gives one window start and BLOCK_CAP entry offsets, whatever the length
of its block.
A seed table maps every start state to the smallest seed whose reset reaches it, so an episode from any start state
replays through the public reset(seed); for a scenario record, an environment whose config allows its one combination,
the table lists the record's own seeds. An anchored start (config.start) fixes the state, so it needs no table.
Every scan reads the start state of a seed through env.start_of, which learns it without a reset where the source can, so a
scan leaves the environment as it found it, and the label seeds, the reserved seeds and the test states of an environment
are kept per environment and arguments, since its configuration and world never change."""
import math
import random
import weakref
from fractions import Fraction

from CausalDiagnosis.datasets.splits.development_split import coverage_order, faulty_levels
from CausalDiagnosis.datasets.splits.train_test_folds import fault_folds


def combo_key(combination):
    """The key of a combination of condition levels in the tables of the starts, the label table and models.json: its levels
    joined by commas."""
    return ",".join(str(level) for level in combination)

MAX_SEED = 1_000_000  # far beyond the seeds that the start states of a scenario class need
EVALUATION_POOL = 8  # the smallest seeds of a start from which the evaluation set draws, the reserved seed among them
REPLICATES = 5  # the start states of every start combination that the test set plays, one run of a grid line each
# the share of the test set that a grid plays unless told otherwise (pending_grid_lines.py, tables_from_runs/): one whole replicate
DEFAULT_FRACTION = "0.2"
# the agents of the wiring check: one run per preset proves the wiring, so on the test set such an agent plays the first replicate alone
WIRING_AGENTS = ("scripted",)
# the rule of the replicates, which run.json records, so that a run of another rule never stands for a replicate of a grid line
_KEPT = weakref.WeakKeyDictionary()  # environment -> {(table, arguments): the table}, for the life of the environment
REPLICATE_RULE = ("replicate 1 is the seed that the master seed draws from the evaluation pool of the start less its reserved seed, "
                  "or the first seed of that pool whose start state differs from that of the episode of the reserved seed when the draw replays it, "
                  "and every further replicate is the smallest seed of the start after the earlier ones that is no rollout seed of its "
                  "separability label and enters at an entry offset that neither the episode of the reserved seed nor an earlier replicate plays")
# the rule of a cut look, which run.json records beside the fraction, so that a run of another rule never stands for its cells
FRACTION_RULE = ("a cut look plays replicate 1 on the first ceil(f x replicates x combinations) start combinations of an order that "
                 "the master seed shuffles, the combinations that add a faulty level no earlier one holds first")


def start_state(env):
    """(combination, window start, entry offset) of the current episode, read right after reset."""
    return env.initial, env.window_start, env.entry_offset


def _kept(env, name, args, compute):
    """compute(), kept for the environment under the name of the table and its arguments."""
    tables = _KEPT.setdefault(env, {})
    if (name, args) not in tables:
        tables[(name, args)] = compute()
    return tables[(name, args)]


def start_states(env):
    """Every (combination, window start, entry offset) that env.reset can draw."""
    return [(combo, w, e) for combo in env.candidates for w, e in env.source.starts(combo)]


def combination_seeds(env, per_combination=1, after=0):
    """The per_combination smallest seeds whose reset draws each start combination of the environment, after the `after`
    smallest of each, flat and in candidate order, so that every line of a grid plays every combination its config allows
    under the same seeds. The separability label plays the seeds after EVALUATION_POOL, so no rollout of a label replays an
    evaluation episode or the episode of a reserved seed."""
    found = {combo: [] for combo in env.candidates}
    drawn = dict.fromkeys(env.candidates, 0)
    for seed in range(MAX_SEED):
        combo = env.start_of(seed)[0]
        drawn[combo] += 1
        seeds = found[combo]
        if after < drawn[combo] and len(seeds) < per_combination:
            seeds.append(seed)
            if all(len(s) == per_combination for s in found.values()):
                return [seed for combo in env.candidates for seed in found[combo]]
    raise RuntimeError(f"seeds 0 to {MAX_SEED - 1} draw {sum(len(s) for s in found.values())} of the "
                       f"{per_combination * len(found)} wanted episodes")


def sampled_combination_seeds(env, master_seed=42, pool=EVALUATION_POOL, replicates=1):
    """The evaluation set of a grid, the test set of the paper, flat in replicate order: one seed per start combination in
    candidate order for the first replicate, then for the second, and so on, which every line plays under the same seeds and
    which replays from the master seed alone. The first replicate draws for every combination, by a generator seeded with
    master_seed, one of the pool smallest seeds whose reset draws it, the smallest left out, the reserved seed that
    combination_seeds plays, and where the drawn seed replays the start state of the episode of the reserved seed it takes
    the first seed of the pool that does not (_off_reserved), since on a row without hidden processes the start state is the
    whole episode. Every further replicate enters at a new start state of the combination (_further_replicates), so the
    replicates of a start are distinct episodes, none replays the episode of the reserved seed and none plays a rollout seed of
    the separability label."""
    if pool < 2:
        raise ValueError(f"the pool needs at least two seeds per combination, since the reserved seed is left out, got {pool}")
    if not 1 <= replicates <= REPLICATES:
        raise ValueError(f"the test set holds 1 to {REPLICATES} replicates of every start, got {replicates}")
    found = {combo: [] for combo in env.candidates}
    for seed in range(MAX_SEED):
        seeds = found[env.start_of(seed)[0]]
        if len(seeds) < pool:
            seeds.append(seed)
            if all(len(s) == pool for s in found.values()):
                break
    else:
        raise RuntimeError(f"seeds 0 to {MAX_SEED - 1} draw {sum(len(s) for s in found.values())} of the {pool * len(found)} pool entries")
    rng = random.Random(master_seed)
    # the draw consumes the generator as it did before the replicates, so a moved start leaves every other draw as it stood
    first = [_off_reserved(env, found[combo], rng.choice(found[combo][1:])) for combo in env.candidates]
    if replicates == 1:
        return first
    further = _further_replicates(env, first, replicates - 1)
    return first + [further[combo][k] for k in range(replicates - 1) for combo in env.candidates]


def replicate_seeds(env, replicate, master_seed=42, pool=EVALUATION_POOL):
    """The seeds of one replicate of the evaluation set, one per start combination in candidate order, which one run of a grid
    line plays."""
    return sampled_combination_seeds(env, master_seed, pool, replicate)[(replicate - 1) * len(env.candidates):]


def fraction_plan(text, replicates=REPLICATES):
    """("whole", r) for a share r/replicates, which plays the replicates 1 to r as a grid plays them without a share, or ("cut", f)
    for a share below one replicate, a cut look of replicate 1. Any other share is refused with the two nearest whole shares, since
    a cut inside a later replicate gives the combinations of its prefix one episode more than the others, so every mean would
    weight the start combinations unequally. The test is exact, on a Fraction of the decimal text, and a share whose one written
    form (fraction_text) is not that value exactly is refused, so the units and the driver cut the same prefix."""
    if "/" in str(text):
        raise ValueError(f"the share {text} is written as a ratio: write it as a decimal, such as 0.05 or 0.4")
    try:
        f = Fraction(str(text))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"the share {text!r} is no number") from None
    if not 0 < f <= 1:
        raise ValueError(f"the share {text} lies outside (0, 1]")
    if Fraction(fraction_text(f)) != f:
        raise ValueError(f"the share {text} has no exact written form of six digits: write a shorter decimal")
    share = f * replicates
    if share.denominator == 1:
        return "whole", int(share)
    if share < 1:
        return "cut", f
    low, high = math.floor(share), math.ceil(share)
    raise ValueError(f"the share {text} cuts inside replicate {high} of {replicates}, which would weight its start combinations "
                     f"unequally: play {fraction_text(Fraction(low, replicates))} or {fraction_text(Fraction(high, replicates))}")


def fraction_text(f):
    """The one written form of a share, as the units of a grid carry it after --fraction."""
    return f"{float(f):g}"


def fraction_value(text):
    """The one stored form of a share, the float that run.json, the condition and the tables hold."""
    return float(Fraction(str(text)))


def cut_combinations(env, fraction, master_seed=42):
    """The start combinations of a cut look: the first ceil(f x REPLICATES x C) of the coverage order of the C candidates under
    the master seed, so a smaller cut lies inside a larger one and every row with the same candidates plays the same cells."""
    healthy = [env.config.healthy[c] for c in env.config.components]
    covering, rest = coverage_order(env.candidates, lambda combination: faulty_levels(combination, healthy), master_seed)
    return frozenset((covering + rest)[:math.ceil(Fraction(str(fraction)) * REPLICATES * len(env.candidates))])


def cut_seeds(env, seeds, fraction, master_seed=42):
    """The seeds of replicate 1 that a cut look plays, those whose start combination cut_combinations keeps, in their order. A fold
    line passes the seeds of its fold, so the prefix is cut over every combination before the fold filter."""
    kept = cut_combinations(env, fraction, master_seed)
    return [seed for seed in seeds if env.start_of(seed)[0] in kept]


def fold_line_empty(env, split, fold, fraction, master_seed=42):
    """Whether the line of the model of `fold` plays no episode at the share: a cut look whose prefix holds no combination of
    that fold. Such a line is not pending (pending_grid_lines.py) and no table expects its run (tables_from_runs/); a whole share gives
    every fold the combinations it holds out."""
    if fraction_plan(fraction)[0] == "whole":
        return False
    return not any(split[c][0] == fold for c in cut_combinations(env, fraction, master_seed) if c in split)


def training_start_states(env, split, fold, master_seed=42, pool=EVALUATION_POOL):
    """The start states that the model of fold `fold` trains on, the one exclusion that the stream of
    the online loop of GEPA and the fine tuning share. split is the asset's registered split, {start combination: (fold, models
    that train on it)}, and the model of a fold is the model that holds that fold out: every start state of a combination whose
    models name the fold that no replicate of the test set plays, so no test episode enters training. The list includes every
    start state that only label rollouts or the episode of the reserved seed play, since no test number scores those episodes."""
    _check_fold(split, fold)
    tested = _kept(env, "tested", (master_seed, pool), lambda: frozenset(
        _state_of(env, seed) for seed in sampled_combination_seeds(env, master_seed, pool, REPLICATES)))
    trains = {combination for combination, (_f, models) in split.items() if fold in models}
    return [state for state in sorted(start_states(env)) if state[0] in trains and state not in tested]


def _off_reserved(env, pool, seed):
    """seed, or the first seed of the pool after the reserved seed pool[0] whose start state differs from that of the episode
    of the reserved seed when seed replays it."""
    reserved = _state_of(env, pool[0])
    if _state_of(env, seed) != reserved:
        return seed
    for other in pool[1:]:
        if _state_of(env, other) != reserved:
            return other
    raise ValueError(f"every seed of the pool of {reserved[0]} replays the start state of the episode of its reserved seed")


def _further_replicates(env, first, count):
    """{start combination: the seeds of its replicates after the first}, count of each: the smallest seeds of the combination in
    seed order that are no rollout seed of its separability label (label_seeds) and enter at an entry offset that neither the
    episode of its reserved seed, its smallest seed, nor an earlier replicate plays, the first replicate being the seeds of
    first. So no test seed is a label rollout seed and no test episode replays the episode of a reserved seed, while the label
    rollouts and the test episodes share start states on many starts, since a block has ten entry offsets, and all of them
    read recordings of the one block of the start."""
    labels = label_seeds(env)
    played = {}
    for seed in (*combination_seeds(env, 1), *first):
        state = _state_of(env, seed)
        played.setdefault(state[0], set()).add(state)
    out = {combo: [] for combo in env.candidates}
    for seed in range(MAX_SEED):
        if all(len(seeds) == count for seeds in out.values()):
            return out
        state = _state_of(env, seed)
        combo = state[0]
        if len(out[combo]) < count and seed not in labels[combo] and state not in played[combo]:
            out[combo].append(seed)
            played[combo].add(state)
    raise RuntimeError(f"seeds 0 to {MAX_SEED - 1} hold {sum(len(s) for s in out.values())} of the {count * len(out)} further replicates")


def label_seeds(env):
    """{start combination: the seeds that the rollouts of its separability label play}, the ROLLOUTS smallest seeds after
    EVALUATION_POOL of every start of the label's static configuration (separability.compute_table), whose fault rule may
    allow starts that the configuration of env leaves out, so the seeds come from an environment of that configuration."""
    return {combo: set(seeds) for combo, seeds in _kept(env, "label_seeds", (), lambda: _label_seeds(env)).items()}


def _label_seeds(env):
    # imported here, since separability imports this module
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.reference_policies.diagnosability_label import ROLLOUTS, static_config
    label_env = HydraulicRigEnv(sorted(env.cycles.values(), key=lambda c: c.index), env.blocks, static_config(env.config))
    seeds = combination_seeds(label_env, ROLLOUTS, after=EVALUATION_POOL)
    return {combo: frozenset(seeds[i * ROLLOUTS:(i + 1) * ROLLOUTS]) for i, combo in enumerate(label_env.candidates)}


def _state_of(env, seed):
    return env.start_of(seed)


def seed_table(env):
    """[(seed, combination, window start, entry offset)] in start-state order, each with the smallest seed reaching it."""
    wanted = set(start_states(env))
    first = {}
    for seed in range(MAX_SEED):
        first.setdefault(env.start_of(seed), seed)
        if len(first) == len(wanted):
            return [(first[s], *s) for s in sorted(first)]
    raise RuntimeError(f"seeds 0 to {MAX_SEED - 1} reach {len(first)} of the {len(wanted)} start states")


def split_folds(split):
    """The folds of a registered split, 1 to K, each holding the test starts of the model that holds it out."""
    return tuple(sorted({fold for fold, _models in split.values()}))


def _check_fold(split, fold):
    if fold not in split_folds(split):
        raise ValueError(f"the registered split names the folds {split_folds(split)}, got {fold!r}")


def combination_atoms(env, combination):
    """The fault atoms of a start combination, (component, level) for every component off its healthy level, in the
    component order of the config."""
    return tuple((c, level) for c, level in zip(env.config.components, combination) if level != env.config.healthy[c])


def combination_folds(env, k, seed):
    """{start combination: (fold, models that train on it)}, the general rule of CausalDiagnosis/datasets/splits/train_test_folds.py
    on the start combinations of env with K folds, which write_training_split.py registers per asset before any run. A repair of the
    rig only returns a component to healthy, so the folds are closed under repair."""
    starts = {combination: combination_atoms(env, combination) for combination in sorted(env.candidates)}
    return fault_folds(starts, list(env.config.components), k, seed)[0]


def fold_replicate_seeds(env, split, replicate, fold, master_seed=42, pool=EVALUATION_POOL):
    """The seeds of one replicate of the test set that the model of `fold` plays: those of the
    combinations of that fold, the fold the model holds out, in candidate order, so the K models together play each replicate
    once and no model plays a combination it trained on. A configuration with a start that the registered split does not hold
    is refused, since no model of the split is judged on that start."""
    _check_fold(split, fold)
    outside = [combination for combination in env.candidates if combination not in split]
    if outside:
        raise ValueError(f"{outside[0]} is no start of the registered split, so no model of the split plays this configuration")
    seeds = replicate_seeds(env, replicate, master_seed, pool)
    return [seed for seed, combination in zip(seeds, env.candidates) if split[combination][0] == fold]


def reserved_seeds(env):
    """The seeds that no training episode plays, the reserved seed of every start and every rollout seed of a separability
    label, so that training replays neither the episode of a reserved seed nor a label rollout, while it may share their
    start states."""
    return set(_kept(env, "reserved_seeds", (), lambda: frozenset(
        set(combination_seeds(env, 1)) | {seed for seeds in label_seeds(env).values() for seed in seeds})))


def training_seeds(env, split, sample, fold, master_seed=42, pool=EVALUATION_POOL):
    """The seeds of training sample `sample` of one fold, counted from 1, one per training start state of the fold in the order
    of training_start_states: the sample-th smallest seed that enters at the state and is no reserved seed, so the samples of a
    state are distinct episodes."""
    if sample < 1:
        raise ValueError(f"the training samples count from 1, got {sample}")
    states = training_start_states(env, split, fold, master_seed, pool)
    reserved = reserved_seeds(env)
    found = {state: [] for state in states}
    for seed in range(MAX_SEED):
        if all(len(seeds) == sample for seeds in found.values()):
            return [found[state][-1] for state in states]
        if seed in reserved:
            continue
        state = _state_of(env, seed)
        if state in found and len(found[state]) < sample:
            found[state].append(seed)
    raise RuntimeError(f"seeds 0 to {MAX_SEED - 1} hold {sum(len(s) for s in found.values())} of the {sample * len(found)} training seeds")


def training_stream(env, split, count, fold, first_seed=0, master_seed=42, pool=EVALUATION_POOL, skip_per_state=0):
    """The stream of the online loop of GEPA on one fold: at most count seeds from first_seed on in seed order, each entering
    at a training start state of the fold and none reserved, the pool that the fine tuning of the fold samples from as well. Per
    training start state the first skip_per_state such seeds are passed over, counted from seed 0 whatever first_seed is: they
    are the seeds that training_seeds gives to the samples 1 to skip_per_state of that state, so a stream that passes over every
    sample of the fine tuning shares no seed with it."""
    states = set(training_start_states(env, split, fold, master_seed, pool))
    reserved = reserved_seeds(env)
    seeds, passed_over = [], dict.fromkeys(states, 0)
    for seed in range(0 if skip_per_state else first_seed, MAX_SEED):
        if len(seeds) == count:
            break
        if seed in reserved:
            continue
        state = _state_of(env, seed)
        if state not in passed_over:
            continue
        if passed_over[state] < skip_per_state:
            passed_over[state] += 1
        elif seed >= first_seed:
            seeds.append(seed)
    return seeds
