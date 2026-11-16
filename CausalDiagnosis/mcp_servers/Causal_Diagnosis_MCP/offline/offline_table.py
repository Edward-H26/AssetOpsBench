"""The offline table of the CausalDiagnosis MCP: its ranking and its post repair check on the starts of the
test set with no model and no agent, the rows of CausalDiagnosis/results/offline_mcp.csv that the paper's offline
table prints (CausalDiagnosis/docs/causal-diagnosis-mcp.md, section The offline table).

The starts are a start set of the asset by the one split of the environment (asset_model.start_sets), which no tool of the
server reads: the starts of the test set by default, or the development starts with --starts development, for a report that
must read no start of the test set. An episode is one start entered at one offset of its block, scored by the fits of an
episode of that start
(Models.without), which leave its block out of every fit, through the filter of the server under the default ranking model and
the default rules of the asset. Its readings are those the environment serves. With prehistory 5 they open with the five
recordings of the start's block that the AssetOpsBench rows release before cycle 0, every summary of each, and k live readings
follow, the reading of cycle t serving the cycle t mod 5 of the live window through the sensor groups in turn from the one of
fewest summaries, so that a wrapped cycle adds the summaries of a richer group. With prehistory 0 the live readings come alone.
The row of prehistory 5 and no live reading is the evidence of the two rows without a clock, and the pair of no prehistory and no
live reading holds no evidence and is not built. Every reading carries the mark of a record, as on the AssetOpsBench rows, and
each has a cycle of its own. Values carry six significant digits, as the environment releases them.

A row holds, for one kind of row and one number of live readings over every start and entry offset, the mean probability of the
first candidate, the share of episodes whose first candidate is the start at every level and the count of the others, the share
of the released live summaries inside the E14 interval of the true combination, an empty cell where a row released none, and
the post repair check of the repair of the
faults the first candidate names. That repair is wrong when it leaves the rig, under the repair kernel of the rules, in another
combination than the one the MCP predicts, the most probable combination of its belief after the repair, and E16
and E17 then score the first reading after it, through the next sensor group in turn, against that combination's mean, as
verify_outcome does. Since that reading can fall on any of the five cycles of the live window
of the combination the repair leaves, the share of those cycles found inconsistent, averaged over the wrong repairs, is the power
of the check.

The calibration check of the selected pairs compares the served fits with a main effects fit built by the same rule, on the
count of wrong first candidates and on the gap between the mean probability of the first candidate and the share of right ones.
The row prints it against two references. The like for like reference takes the covariance estimator of the start's form, so
it differs from the served fits in the pairs alone. The second reference takes the registered default estimator, the reference
of the unit test over the starts of the test set, so against it the pairs and the estimator differ at once. The two are one fit
for a start whose form holds the default estimator. Both tests of the gap are printed, the two sided one as first registered
and the one sided one that replaced it (comparison).

The run order computes the table on the cluster, and the file in the repository holds its header alone until the test set runs.

usage: python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.offline_table [--starts test|development] [--out PATH]
"""
import argparse
import csv
from collections import namedtuple
from pathlib import Path

import numpy as np

from CausalDiagnosis.utils.atomic_writes_and_locks import replaced_atomically

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import EffectDiagnostics
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import asset_view, served_segments, start_sets
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.calibration_report import RESULTS_DIR
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import DEFAULT_ALPHA, most_probable, predict_summaries
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import DEFAULT_COVARIANCE, fit_episode
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Episode, Filter, Reading
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import ensure_models

RESULTS_PATH = RESULTS_DIR / "offline_mcp.csv"
READINGS = (0, 1, 2, 3, 5, 10)  # live readings per row; with no prehistory the count 0 holds no evidence and is not built
COLUMNS = ("prehistory", "readings", "episodes", "top_mass", "top1_rate", "exact_wrong", "e14_covers_truth", "e17_rejects_wrong_repair",
           "wrong_repairs", "main_effects_top_mass", "main_effects_top1_rate",
           "calibration_gap", "main_effects_exact_wrong", "main_effects_calibration_gap", "fewer_wrong", "gap_two_sided", "gap_one_sided",
           "guard_reference_top_mass", "guard_reference_top1_rate", "guard_reference_exact_wrong", "guard_reference_calibration_gap",
           "guard_fewer_wrong", "guard_gap_one_sided")
START_SETS = ("test", "development")

StartFits = namedtuple("StartFits", "episode model main_effects guard_reference mu width")
Scored = namedtuple("Scored", "top_mass right covered summaries wrong_repair rejected main_mass main_right guard_mass guard_right")


def groups_in_turn(view):
    """The sensor groups from the one of fewest summaries to the one of most, the order in which the live readings take them."""
    return sorted(view.sensor_groups, key=lambda g: (len(view.group_fields(g)), g))


def released(index, cycle, fields):
    """The summaries of a recorded cycle as the environment releases them, at six significant digits."""
    return {f: float(f"{cycle[index[f]]:.6g}") for f in fields}


def first_candidate(post):
    return tuple(post.candidates[post.order()[0]])


def start_fits(models, start, alpha):
    """The fits of an episode of the start, the E14 interval of the start, and the two main effects fits built by the same rule:
    the like for like reference under the covariance estimator of the start's form, and the reference under the registered
    default estimator, which the unit test over the starts of the test set builds. They are one fit when the form holds the
    default estimator."""
    episode = models.without(start)
    width = np.array([episode.calibration.quantile(k, alpha)[0] for k in range(len(models.space.fields))]) * episode.sigma
    estimator = models.covariance_without(start)
    main_effects = fit_episode(models.recorded, [], start, 0, 0, covariance=estimator).effect
    guard_reference = main_effects if estimator == DEFAULT_COVARIANCE else fit_episode(models.recorded, [], start, 0, 0).effect
    return StartFits(episode, EffectDiagnostics(episode, block=True, resampled=False), main_effects, guard_reference, episode.effect.mu(start), width)


def after_visit(model, view, combination, components):
    """The combination that a visit on the components leaves the rig in when it holds the given one, under the repair kernel of
    the rules, in which a replacement takes only when the components it depends on are healthy at its close."""
    held = Filter(view, model.models.effect, Episode(0, [], [], None), resampled=False, prior=None, support=[combination])
    return first_candidate(model.after_repair(held, components, view).posterior())


def score_episode(models, view, start, fits, history, live, k, alpha):
    """One episode of the start, with the recordings of the history positions before cycle 0 and k live readings from the live
    window, whose positions are those of the episode's entry offset in every block."""
    index, groups, block = models.space.field_index, groups_in_turn(view), models.blocks[start]
    reads = [(live[t % len(live)], groups[t % len(groups)]) for t in range(k)]
    readings = [Reading(n + 1, n - len(history), "history", released(index, block[p], view.summary_fields), recorded=True)
                for n, p in enumerate(history)]
    readings += [Reading(len(history) + t + 1, t, g, released(index, block[p], view.group_fields(g)), recorded=True) for t, (p, g) in enumerate(reads)]
    episode = Episode(k, readings, [], None)
    flt = fits.model.replay(view, episode)
    post = flt.posterior()
    top = first_candidate(post)
    # a wrapped cycle releases only the summaries no earlier reading of it released, so each summary counts once
    shown = {(p, f): v for p, g in reads for f, v in released(index, block[p], view.group_fields(g)).items()}
    covered = sum(abs(v - fits.mu[index[f]]) <= fits.width[index[f]] for (_p, f), v in shown.items())
    components = view.faulty(top)
    after = fits.model.after_repair(flt, components, view).posterior().current_posterior()
    actual = after_visit(fits.model, view, start, components)
    wrong_repair = actual != most_probable(after, models.space)[0]
    rejected = None
    if wrong_repair:
        prediction = {"alpha": alpha, "predicted": {s.field: [s.mean, s.low, s.high, s.quantile] for s in predict_summaries(fits.episode, after, alpha)}}
        fields = view.group_fields(groups[k % len(groups)])
        rejected = float(np.mean([not fits.model.outcome_check(prediction, released(index, cycle, fields))[3]
                                  for cycle in models.blocks[actual][live]]))
    main = Filter(view, fits.main_effects, episode, resampled=False, block=True).posterior()
    guard = main if fits.guard_reference is fits.main_effects else Filter(view, fits.guard_reference, episode, resampled=False, block=True).posterior()
    return Scored(float(post.probabilities[post.order()[0]]), top == start, int(covered), len(shown), wrong_repair, rejected,
                  float(main.probabilities[main.order()[0]]), first_candidate(main) == start,
                  float(guard.probabilities[guard.order()[0]]), first_candidate(guard) == start)


def ranked(masses, rights):
    """(the mean probability of the first candidate, the share of first candidates that are the start, the count of the others)."""
    return float(np.mean(masses)), float(np.mean(rights)), int(len(rights) - sum(rights))


def comparison(wrong, gap, reference_wrong, reference_gap):
    """The calibration check of the served fits against a main effects reference, each verdict 1 when it holds and 0 when not.
    wrong counts the first candidates that are not the start, and gap is the mean probability of the first candidate minus the
    share of first candidates that are the start, positive for a belief that is too sure. fewer_wrong: strictly fewer wrong than
    the reference. gap_two_sided: the size of the gap at or below the size of the reference's gap, the test as first registered.
    gap_one_sided: the gap itself at or below that size, the test that replaced it, which lets a belief that is not sure enough
    pass, since only a belief that is too sure misleads."""
    return {"fewer_wrong": int(wrong < reference_wrong), "gap_two_sided": int(abs(gap) <= abs(reference_gap)),
            "gap_one_sided": int(gap <= abs(reference_gap))}


def row_of(prehistory, k, scored):
    repairs = [s.rejected for s in scored if s.wrong_repair]
    shown = sum(s.summaries for s in scored)  # zero on a row that released no live summary
    mass, rate, wrong = ranked([s.top_mass for s in scored], [s.right for s in scored])
    main_mass, main_rate, main_wrong = ranked([s.main_mass for s in scored], [s.main_right for s in scored])
    guard_mass, guard_rate, guard_wrong = ranked([s.guard_mass for s in scored], [s.guard_right for s in scored])
    guard = comparison(wrong, mass - rate, guard_wrong, guard_mass - guard_rate)
    return {"prehistory": prehistory, "readings": k, "episodes": len(scored), "top_mass": mass, "top1_rate": rate, "exact_wrong": wrong,
            "e14_covers_truth": sum(s.covered for s in scored) / shown if shown else None,
            "e17_rejects_wrong_repair": float(np.mean(repairs)) if repairs else None, "wrong_repairs": len(repairs),
            "main_effects_top_mass": main_mass, "main_effects_top1_rate": main_rate,
            "calibration_gap": mass - rate, "main_effects_exact_wrong": main_wrong, "main_effects_calibration_gap": main_mass - main_rate,
            **comparison(wrong, mass - rate, main_wrong, main_mass - main_rate),
            "guard_reference_top_mass": guard_mass, "guard_reference_top1_rate": guard_rate, "guard_reference_exact_wrong": guard_wrong,
            "guard_reference_calibration_gap": guard_mass - guard_rate,
            "guard_fewer_wrong": guard["fewer_wrong"], "guard_gap_one_sided": guard["gap_one_sided"]}


def table(models, view, starts=None, readings=READINGS, alpha=DEFAULT_ALPHA):
    """The rows of the table, one per kind of row and number of live readings, over every recorded start of the given ones at
    every entry offset, the starts of the test set of the asset when none are given. An entry offset serves one segment as the
    prehistory and the segment after it as the live window, as the environment does, so the rows with a prehistory release all
    of its recordings and the others none. A start set without a recorded start is refused, since a row divides by the
    count of the summaries it scored."""
    segments = served_segments()
    size = len(segments[0])
    starts = [h for h in (start_sets(view, models.blocks)[1] if starts is None else starts) if h in models.blocks]
    if not starts:
        raise ValueError("the start set holds no recorded start of the asset, so the table has no episode to score")
    scored = {(p, k): [] for p in (size, 0) for k in readings if p or k}
    for start in starts:
        fits = start_fits(models, start, alpha)
        for entry, history in enumerate(segments):
            live = segments[(entry + size) % len(segments)]
            for p, k in scored:
                scored[(p, k)].append(score_episode(models, view, start, fits, history[size - p:], live, k, alpha))
    return [row_of(p, k, rows) for (p, k), rows in scored.items()]


def write(rows, path):
    """The table as CSV, the floats at six significant digits and an empty cell where a row has no wrong repair or released no
    live summary. The file is replaced through a temporary file, so a failed write leaves the earlier table."""
    with replaced_atomically(path) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(["" if row[c] is None else f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in COLUMNS])


def main(argv=None):
    parser = argparse.ArgumentParser(description="the offline table of the CausalDiagnosis MCP on a start set of the asset, no model and no agent")
    parser.add_argument("--starts", choices=START_SETS, default=START_SETS[0],
                        help="the start set: the starts of the test set, or the development starts, which need their own --out")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)
    if args.starts != START_SETS[0] and args.out == RESULTS_PATH:
        parser.error(f"--starts {args.starts} needs its own --out, since {RESULTS_PATH.name} holds the table of the starts of the test set")
    ensure_models()
    models, view = load_models(), asset_view()
    development, test = start_sets(view, models.blocks)
    rows = table(models, view, starts=development if args.starts == "development" else test)
    write(rows, args.out)
    print(f"wrote {args.out}: {len(rows)} rows of {rows[0]['episodes']} episodes each, on the {args.starts} starts")


if __name__ == "__main__":
    main()
