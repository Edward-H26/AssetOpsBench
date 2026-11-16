from fractions import Fraction
from statistics import stdev
from CausalDiagnosis.utils.csv_rows import as_bool, as_float  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402  the fractions of the test set (fraction_plan, fraction_text, fraction_value)
from CausalDiagnosis.experiments.episode_runners import run_configuration  # noqa: E402  the empty line of a fold at a cut look (fold_line_empty)
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402  the folds of a split (fold_ids)
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import REGISTERED  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_proportion, paired_cluster_bootstrap, sign_flip_p  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares


JUDGE_CRITERIA = ("task_completion", "data_retrieval_accuracy", "generalized_result_verification")
PAPER_PROTOCOL = "paper"  # five judge calls at temperature 0.3, the only protocol the tables of the judge print
PROPORTIONS = frozenset({"pass", "success", "separable", "commitment_on_inseparable", "diagnosis_all_correct", "recovered", "abstained",
                         "abstention_correct", "abstention_cites_reading", "judge_pass", "faulty_set_right", "restored_clean", "provenance_ok", "write_count_ok", "timing_ok", "ordering_ok", "static_faulty_set_correct", "static_repairs_match_oracle",
                         "test_first", "level_accuracy", "scope_pass", "diagnosis_wrong", "unmatched_evidence", "judge_record_cut", "judge_passed_unrecovered",
                         *(f"{criterion}_majority" for criterion in JUDGE_CRITERIA)})
FACTORS = ("config", "agent", "model", "clock_mode", "surface", "guidance_level", "diagnostics_server", "prices", "verifier_scope", "ranking_model")
IDENTITY = ("seed", "scenario_id")
# how an episode that the verifier did not score counts on each outcome, the one home of the registered counting.unscored and
# counting.reward: "failure" on a share that is better when higher, "zero" on the reward and a graded outcome, "left out" of a share
# that is better when lower or is no outcome of the agent (the row prints its _n), and "derived" for a field computed from an outcome
# that counts already; the judge outcomes keep counting.unjudged (JUDGE_OUTCOMES), and a missing episode is never counted, its row is
# incomplete and its tests are not run. A test holds every share, graded outcome and the reward to one entry.
OUTCOME_COUNTING = {
    **dict.fromkeys(("pass", "scope_pass", "success", "diagnosis_all_correct", "faulty_set_right", "recovered", "restored_clean", "provenance_ok", "write_count_ok", "timing_ok", "ordering_ok",
                     "abstention_correct", "abstention_cites_reading"), "failure"),
    **dict.fromkeys(("graded_pass", "graded_top1", "reward"), "zero"),
    **dict.fromkeys(("diagnosis_wrong", "unmatched_evidence", "commitment_on_inseparable", "abstained", "separable"), "left out"),
    **dict.fromkeys(("level_accuracy", "test_first", "judge_passed_unrecovered", "judge_record_cut"), "derived"),
    **dict.fromkeys(("static_faulty_set_correct", "static_repairs_match_oracle"), "failure"),
}
STATIC_OUTCOMES = frozenset({"static_faulty_set_correct", "static_repairs_match_oracle"})  # counted on a played static answer
# the outcomes that an unscored played gym episode reads as 0
VERIFIED_OUTCOMES = frozenset(m for m, rule in OUTCOME_COUNTING.items() if rule in ("failure", "zero") and m not in STATIC_OUTCOMES)
JUDGE_OUTCOMES = frozenset({"judge_pass", *JUDGE_CRITERIA, *(f"{criterion}_majority" for criterion in JUDGE_CRITERIA)})
# the modes whose episodes the judge scores: a played gym episode and a static answer, which table A3 reads
JUDGED_MODES = ("gym", "static")
# the outcomes that are whole numbers by their kind beside the shares, whose sign flip p value is exact by convolution
WHOLE_OUTCOMES = frozenset({"repair_declared"})


def value(episode, metric):
    """The metric of an episode as a number: a flag as 1 or 0, an empty cell as None, except an outcome of the counting rule on an
    episode that it counts (counts_as_failure), which reads 0."""
    cell = episode.get(metric, "")
    if metric in PROPORTIONS:
        flag = as_bool(cell)
        number = None if flag is None else float(flag)
    else:
        number = as_float(cell)
    return 0.0 if number is None and counts_as_failure(episode, metric) else number


def counts_as_failure(episode, metric):
    """Whether an empty cell counts as a failure (OUTCOME_COUNTING): a verified outcome of a played gym episode, an outcome of a played
    static answer, and a judge outcome of a played gym
    episode or of a static answer (JUDGED_MODES) whose run names a judge, the registered counting.unjudged. A static answer has no
    verified outcome, since no verifier scores it, so those cells stay out. The means of cost, time and calls keep leaving such an
    episode out and print their own number of episodes."""
    mode = episode.get("mode", "gym")
    if metric in JUDGE_OUTCOMES:
        return mode in JUDGED_MODES and bool(episode.get("run_judge_model"))
    if metric in STATIC_OUTCOMES:
        return mode == "static"
    # the reward of an episode that the verifier scored but whose oracle cost is unknown is undefined and stays out; every other
    # empty cell of these outcomes on a played gym episode counts
    return mode == "gym" and metric in VERIFIED_OUTCOMES and (metric != "reward" or as_bool(episode.get("pass")) is None)


def cluster_of(episode):
    """The cluster of an episode, the unit that every bootstrap and sign flip test resamples: its column cluster when the row has
    one, else its block, the start combination, so that a port whose episodes share another unit of independence needs no copy of
    the tables."""
    cluster = episode.get("cluster")
    return episode.get("block", "") if cluster in (None, "") else cluster


def registered_of(ctx):
    """The registered analysis file that a table reads, the one of aggregate() or the tree's."""
    return ctx.get("registered") or registered.load()


def read_text(ctx):
    """The read cell of a row: "partial f" for a read below the fraction of record, which every row of a table carries, else empty."""
    read = ctx.get("read") or {}
    return f"partial {read['text']}" if read.get("partial") else ""


def wiring_agent(episode):
    """Whether an episode is of an agent of the wiring check, which plays the first replicate of the test set alone."""
    return str(episode.get("agent", "")).split("_")[0] in start_states.WIRING_AGENTS


def expected_units(episodes, ctx):
    """The (replicate, fold) units that a row of the test set holds at the read of ctx, one finished run each: the replicates 1 to r
    of the read, the first alone for a cut look or an agent of the wiring check, times, for a trained row, the folds of its split
    whose line is not empty at the share (common.fold_line_empty, the predicate pending_grid_lines.py shares). A replicate whose run
    left no scored row is expected all the same and reads as missing. None outside the test set or without a read, which keeps
    the rule of the planned episodes alone."""
    read = ctx.get("read")
    sampled = [e for e in episodes if e.get("replicate") not in (None, "")]
    if not read or not sampled:
        return None
    replicates = [1] if any(wiring_agent(e) for e in sampled) else list(range(1, read["replicates"] + 1))
    first = sampled[0]
    if all(e.get("training_fold") in (None, "") for e in sampled):
        folds = [None]
    else:
        split = first.get("training_split") or REGISTERED
        folds = [fold for fold in training.held_out_folds.fold_ids(first.get("asset_id") or DEFAULT_ASSET, split)
                 if not run_configuration.fold_line_empty(first.get("config"), fold, read["text"], training_split=split)]
    return [(replicate, fold) for replicate in replicates for fold in folds]


def replicates_text(replicates):
    """The replicates a row holds, as "1-3" for a run of numbers and "1, 3" otherwise."""
    ordered = sorted(set(replicates))
    if not ordered:
        return "none"
    return f"{ordered[0]}-{ordered[-1]}" if len(ordered) > 1 and ordered == list(range(ordered[0], ordered[-1] + 1)) else ", ".join(map(str, ordered))


def unit_counts(episodes):
    """{(replicate, fold): {run: [played, planned]}} over the episodes of the test set of one row."""
    units = {}
    for e in episodes:
        if e.get("replicate") in (None, ""):
            continue
        fold = None if e.get("training_fold") in (None, "") else e["training_fold"]
        run = units.setdefault((int(e["replicate"]), fold), {}).setdefault(e.get("run_dir") or e.get("run"), [0, e.get("episodes_planned")])
        run[0] += 1
    return units


def replicate_state(episodes, ctx):
    """(missing units, replicates held, planned episodes of the missing units or None when one has no run of its fold to count from)
    of a row of the test set, or None outside it (expected_units)."""
    expected = expected_units(episodes, ctx)
    if expected is None:
        return None
    units = unit_counts(episodes)
    finished = {unit for unit, runs in units.items() if any(planned not in (None, "") and played == int(planned) for played, planned in runs.values())}
    missing = [unit for unit in expected if unit not in finished]
    per_fold = {unit[1]: int(planned) for unit, runs in units.items() for _, planned in runs.values() if planned not in (None, "")}
    absent = [unit for unit in missing if unit not in units]
    extra = sum(per_fold[fold] for _, fold in absent) if all(fold in per_fold for _, fold in absent) else None
    return missing, replicates_text(replicate for replicate, _ in units), extra


def counting_text(read="", played=None, planned=None, replicates=""):
    """The one form of the state of a row under the counting rule, which row_status writes and main_tables prints after a label:
    "partial f" for a read below the fraction of record, "incomplete, k of n" or "incomplete, k played" when the planned episodes
    of a missing unit are not known, with ", replicates H" on the test set, the two joined by "; "; empty for neither."""
    parts = [read] if read else []
    if played is not None:
        counted = f"{played} of {planned}" if planned is not None else f"{played} played"
        parts.append(f"incomplete, {counted}" + (f", replicates {replicates}" if replicates else ""))
    return "; ".join(parts)


def completeness(episodes, ctx):
    """The counts of the counting rule over the episodes of one row: played, the rows of
    scores.csv, planned, the sum of episodes_planned of its runs or empty when a run names none, unscored and unjudged, and whether
    the row is incomplete, fewer played than planned, or refused, an unscored share above the registered limit (none while the limit
    is empty). The runs are keyed by their folder, since two replicate runs started in the same second share the run cell, and by
    the run cell for an episode without a folder."""
    runs = {}
    for e in episodes:
        runs.setdefault(e.get("run_dir") or e.get("run"), e.get("episodes_planned"))
    known = [n for n in runs.values() if n not in (None, "")]
    planned = sum(int(n) for n in known) if runs and len(known) == len(runs) else None
    played = len(episodes)
    unscored = sum(1 for e in episodes if e.get("mode", "gym") == "gym" and as_bool(e.get("pass")) is None)
    unjudged = sum(1 for e in episodes if e.get("run_judge_model") and as_bool(e.get("judge_pass")) is None)
    limit = registered_of(ctx)["counting"].get("max_unscored_share")
    # a row of the test set is complete only when every unit of the read holds a finished run
    state = replicate_state(episodes, ctx)
    missing, held = (state[0], state[1]) if state else ([], "")
    if state and missing and planned is not None:
        planned = None if state[2] is None else planned + state[2]
    return {"played": played, "planned": "" if planned is None else planned, "unscored": unscored, "unjudged": unjudged,
            "incomplete": bool(missing) or (planned is not None and played < planned),
            "refused": limit is not None and played > 0 and unscored / played > limit,
            "replicates": held, "read": read_text(ctx)}


def row_status(episodes, ctx):
    """Why the tests that read a row are not run, or empty: refused; incomplete, with the replicates a row of the test set holds; or
    partial with its fraction, a read below the fraction of record, which runs no registered test."""
    counts = completeness(episodes, ctx)
    if counts["refused"]:
        return "refused"
    if not counts["incomplete"]:
        return counting_text(counts["read"])
    return counting_text(counts["read"], counts["played"], None if counts["planned"] == "" else counts["planned"], counts["replicates"])


def with_metric(episodes, metric):
    return [e for e in episodes if value(e, metric) is not None]


def groups(episodes, *fields):
    """{(field values): [episodes]} in sorted key order."""
    out = {}
    for e in episodes:
        out.setdefault(tuple(e.get(f, "") for f in fields), []).append(e)
    return dict(sorted(out.items(), key=lambda kv: tuple(str(k) for k in kv[0])))


def proportion_cells(hits, clusters, ctx, counted=None):
    """The share of the hits among the counted items with its interval, from a bootstrap that resamples recorded blocks
    (stats.cluster_proportion), the interval of every proportion of the tables, as {proportion, low, high}."""
    return cluster_proportion(hits, clusters, counted, ctx["resamples"], ctx["seed"])


def summary(episodes, metric, ctx):
    """Mean of a metric with its interval and standard error from a bootstrap that resamples recorded clusters: the share of a
    proportion through proportion_cells, the mean of any other metric through the paired bootstrap against zero. Below the
    registered smallest number of clusters the estimate prints alone (too_few_clusters)."""
    eps = with_metric(episodes, metric)
    if not eps:
        return {metric: "", f"{metric}_low": "", f"{metric}_high": "", f"{metric}_se": "", f"{metric}_n": 0}
    values = [value(e, metric) for e in eps]
    clusters = [cluster_of(e) for e in eps]
    if metric in PROPORTIONS:
        w = proportion_cells(values, clusters, ctx)
        cells = {metric: _r(w["proportion"]), f"{metric}_low": _r(w["low"]), f"{metric}_high": _r(w["high"]), f"{metric}_se": _r(w.get("se")),
                 f"{metric}_n": len(values)}
    else:
        boot = paired_cluster_bootstrap(values, [0.0] * len(values), clusters, ctx["resamples"], ctx["seed"])
        cells = {metric: _r(boot["difference"]), f"{metric}_low": _r(boot["low"]), f"{metric}_high": _r(boot["high"]), f"{metric}_se": _r(boot.get("se")),
                 f"{metric}_n": len(values)}
    return blank_spread(cells, (f"{metric}_low", f"{metric}_high", f"{metric}_se"), len(set(clusters)), ctx)


def too_few_clusters(clusters, ctx):
    """Whether a cell over this many clusters prints its estimate alone, below the registered smallest number (none while it is empty)."""
    least = registered_of(ctx)["intervals"].get("min_clusters")
    return (ctx.get("read") or {}).get("kind") == "cut" or (least is not None and clusters < least)


def cluster_floor(ctx):
    """The smallest number of clusters below which a cell prints its estimate alone, for a reader with an argument of its own:
    above every count in a cut look (too_few_clusters)."""
    return 10 ** 9 if (ctx.get("read") or {}).get("kind") == "cut" else registered_of(ctx)["intervals"].get("min_clusters")


def blank_spread(cells, keys, clusters, ctx):
    """The cells with their interval and standard error emptied below the registered smallest number of clusters."""
    return {**cells, **{key: "" for key in keys}} if too_few_clusters(clusters, ctx) else cells


def interval_cells(name, boot, ctx, estimate="estimate"):
    """The cells of one statistic from a bootstrap of table_statistics.py (cluster_bootstrap, cluster_proportion or paired_cluster_bootstrap):
    the estimate, its interval and its standard error, the last three empty below the registered smallest number of clusters."""
    cells = {name: _r(boot[estimate]), f"{name}_low": _r(boot["low"]), f"{name}_high": _r(boot["high"]), f"{name}_se": _r(boot.get("se"))}
    return blank_spread(cells, (f"{name}_low", f"{name}_high", f"{name}_se"), boot["clusters"], ctx)


def spread(episodes, metric):
    """The sample standard deviation of a metric over the episodes that have it, empty below two values."""
    values = [value(e, metric) for e in with_metric(episodes, metric)]
    return {f"{metric}_std": _r(stdev(values)) if len(values) > 1 else ""}


def matched(level_eps, reference_eps, factor, metric):
    """The (episode, reference episode) pairs of a level against its reference: episodes match on their identity and on
    every factor other than the one compared (or the several that a comparison between agents leaves out), and both
    carry the metric."""
    excluded = {factor} if isinstance(factor, str) else set(factor)
    key_fields = [f for f in (*IDENTITY, *FACTORS) if f not in excluded]

    def key(e):
        return tuple(e.get(f, "") for f in key_fields)

    reference = {key(e): e for e in with_metric(reference_eps, metric)}
    return [(e, reference[key(e)]) for e in with_metric(level_eps, metric) if key(e) in reference]


def paired(level_eps, reference_eps, factor, metric, ctx):
    """The paired difference of a metric between a level and the reference level of a factor over the matched episodes."""
    return paired_difference(matched(level_eps, reference_eps, factor, metric), metric, ctx)


def paired_difference(pairs, metric, ctx):
    """The paired difference of a metric over (episode, reference episode) pairs, which may pool several rows, clustered by the
    recorded block of the start so that the pairs of one start move together: the mean with its bootstrap interval, and the p value
    of the sign flip test over the blocks, exact for a zero or one outcome (PROPORTIONS), whose differences are whole numbers, and
    with as many flips as the bootstrap has resamples for any other metric."""
    if not pairs:
        return {f"{metric}_diff": "", f"{metric}_diff_low": "", f"{metric}_diff_high": "", f"{metric}_diff_se": "", f"{metric}_p": "", "pairs": 0, "clusters": 0}
    mine, theirs, blocks = [value(a, metric) for a, _ in pairs], [value(b, metric) for _, b in pairs], [cluster_of(a) for a, _ in pairs]
    boot = paired_cluster_bootstrap(mine, theirs, blocks, ctx["resamples"], ctx["seed"])
    p = sign_flip_p([x - y for x, y in zip(mine, theirs)], blocks, ctx["resamples"], ctx["seed"], whole=metric in PROPORTIONS or metric in WHOLE_OUTCOMES)
    cells = {f"{metric}_diff": _r(boot["difference"]), f"{metric}_diff_low": _r(boot["low"]), f"{metric}_diff_high": _r(boot["high"]),
             f"{metric}_diff_se": _r(boot.get("se")), f"{metric}_p": _r(p, 6), "pairs": boot["episodes"], "clusters": boot["clusters"]}
    return blank_spread(cells, (f"{metric}_diff_low", f"{metric}_diff_high", f"{metric}_diff_se"), boot["clusters"], ctx)


def trace(ctx, row, episodes, reference=None):
    """Records the episodes behind a row, and those of the level it is paired against, for the results index of
    index_file.py and pooling_rules.py. A builder called outside aggregate() records nothing. Returns the row."""
    if "trace" in ctx:
        ctx["trace"][id(row)] = (list(episodes), None if reference is None else list(reference))
    return row


def _no_pairing(metrics):
    """The empty paired cells of a row that has no reference: the reference row itself, or a level without one."""
    cells = {"pairs": "", "clusters": ""}
    for m in metrics:
        cells.update({f"{m}_diff": "", f"{m}_diff_low": "", f"{m}_diff_high": "", f"{m}_p": ""})
    return cells


def _paired_cells(level, reference, factor, metrics, ctx):
    """The paired cells of several metrics against one reference; pairs and clusters are those of the first metric, the
    primary test of the table."""
    cells = {}
    for m in metrics[1:]:
        cells.update({k: v for k, v in paired(level, reference, factor, m, ctx).items() if k not in ("pairs", "clusters")})
    cells.update(paired(level, reference, factor, metrics[0], ctx))
    return cells


def _share(k, n):
    return _r(Fraction(k, n)) if n else ""


SPREAD = ("", "_low", "_high", "_se")
INTERVAL = SPREAD[:3]  # the columns of a cell in a table declaration, after whose _high column tables() adds the _se column


def _part_cells(pairs, metric, part, ctx):
    """The difference, the interval and the standard error of paired_difference over the pairs, under the column names of a part
    of a descriptive table, with no p value."""
    cells = paired_difference(pairs, metric, ctx)
    return {f"{metric}_{part}{suffix}": cells[f"{metric}_diff{suffix}"] for suffix in SPREAD}


def _prefixed(cells, prefix):
    return {f"{prefix}{key}": cell for key, cell in cells.items()}
