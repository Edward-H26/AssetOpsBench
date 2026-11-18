"""The summary of a cell, the episodes of the run folders that share one condition and one serving of the agent model, as ratios of
sums over the episodes (CausalDiagnosis/performance/README.md).

A cell is the asset, the agent, the preset, the clock mode, the model, the runs in flight and the serving identity of the agent's
endpoint, its weights, precision, engine, launch flags, context and card, so that the cells of one comparison differ in what the
comparison varies and no cell pools two assets, two cards or two launches. The share of a stage over a cell is the sum of its seconds over the episodes
over the sum of their totals, so that every episode weighs as its seconds do, and the spread of a stage is its median with the
10th and the 90th percentile over the episodes."""
import collections

import numpy as np

from CausalDiagnosis.experiments.run_verification.run_ledger import serving_endpoints
from CausalDiagnosis.utils.csv_rows import as_bool
from . import engine_export, stage_profile

# the serving identity of the agent's endpoint that a cell holds fixed: what run.json records under serving and the card
SERVING = (*serving_endpoints.SERVING_KEYS, "engine", "context_length", *serving_endpoints.CARD_KEYS)
QUANTITIES = ("total_s", *stage_profile.SECONDS)


def cell_key(meta):
    """The condition and the serving of a run as its cell: the asset that the run records, the agent label, the preset, the clock
    mode, the model, the runs in flight and the serving identity of the agent's endpoint, all None for a run without a served
    model."""
    serving = (meta.get("serving") or {}).get("agent") or {}
    return {"asset": meta.get("asset_id"), "agent": meta.get("agent_label"), "config": meta.get("config_name"), "clock_mode": meta.get("clock_mode"),
            "model": meta.get("model"), "runs_in_flight": meta.get("runs_in_flight"),
            "serving": {name: serving.get(name) for name in SERVING}}


def spread(values):
    """The median, the 10th and the 90th percentile of values and their count, each None without a value."""
    if not values:
        return {"median": None, "p10": None, "p90": None, "n": 0}
    p10, median, p90 = np.percentile(np.asarray(values, dtype=float), [10, 50, 90])
    return {"median": float(median), "p10": float(p10), "p90": float(p90), "n": len(values)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def seconds_summary(kept):
    """Per quantity of the profile over the kept episodes that hold it: its sum, its share as the ratio of its sum to the sum of the
    totals of the same episodes, and its spread."""
    out = {}
    for name in QUANTITIES:
        holding = [row for row in kept if row[name] is not None]
        total = sum(row[name] for row in holding) if holding else None
        out[name] = {"sum": total, "share": ratio(total, sum(row["total_s"] for row in holding)) if holding else None,
                     **spread([row[name] for row in holding])}
    return out


def held_sum(rows, name):
    """The sum of a column over the rows that hold it, None when none does, so that a column that no episode holds never reads
    as zero."""
    held = [row[name] for row in rows if row.get(name) is not None]
    return sum(held) if held else None


def clock_summary(kept):
    """The charged thinking of a cell against its horizon, the cycles charged over the cycles of the horizons of the same
    episodes, and the sums of the clock columns over the episodes that hold each."""
    sums = {name: held_sum(kept, name) for name in stage_profile.CLOCK if name not in ("think_seconds_per_cycle", "thinking_share_of_horizon")}
    with_horizon = [row for row in kept if row["horizon_cycles"] is not None]
    return {**sums, "thinking_share_of_horizon": ratio(sum(row["thinking_cycles"] for row in with_horizon),
                                                       sum(row["horizon_cycles"] for row in with_horizon))}


def call_summary(kept, call_seconds):
    """The model calls of a cell over the kept episodes that hold llm_calls.json: their count, the spread of the seconds of one call
    over every call, the tokens, and the output tokens over the seconds of the episodes whose calls report both."""
    holding = [row for row in kept if row["model_calls"] is not None]
    timed = [row for row in holding if row["output_tokens_per_call_second"] is not None]
    return {"episodes": len(holding), **{name: held_sum(holding, name) for name in ("model_calls", "failed_model_calls")},
            "call_seconds": spread(call_seconds), **{name: held_sum(holding, name) for name in ("input_tokens", "output_tokens")},
            "output_tokens_per_call_second": ratio(sum(row["output_tokens"] for row in timed), sum(row["call_seconds_sum"] for row in timed))}


def engine_summary(kept, call_figures):
    """The engine's split of the model calls of a cell over the kept episodes that the join reached, those without a joined call
    among them: the counts of the join, the sums of the parts of the seconds of their calls with the shares of those seconds, which
    add up to one, the cached and the decode tokens, the decode rate as a ratio of sums, and the spread of the prefill and of the
    decode of one call that agrees with its lines; every figure None where the join reached no episode."""
    reached = [row for row in kept if row.get("engine_calls_joined") is not None]
    sums = {name: held_sum(reached, name) for name in (*engine_export.COUNTS, *engine_export.PARTS, "cached_prompt_tokens", "decode_tokens")}
    seconds = held_sum(reached, "call_seconds_sum")
    agreeing = [figure for figure in call_figures if figure["input_match"] and figure["output_match"]]
    return {"episodes": len(reached), **sums, "call_seconds": seconds,
            **{share: ratio(sums[name], seconds) for name, share in zip(engine_export.PARTS, engine_export.CALL_SHARES)},
            "decode_tokens_per_s": ratio(sums["decode_tokens"], sums["decode_s"]),
            "prefill_s_per_call": spread([figure["prefill_s"] for figure in agreeing]),
            "decode_s_per_call": spread([figure["decode_s"] for figure in agreeing])}


def gpu_summary(kept):
    """The GPU figures of a cell from the sampler rows of its episodes: the rows of the episodes that the sampler was read for, the
    mean busy share over every row, and the energy per episode and per passed episode over the episodes that have it."""
    busy = [row for row in kept if row.get("gpu_busy_share") is not None]
    energy = [row for row in kept if row.get("gpu_energy_j") is not None]
    joules = sum(row["gpu_energy_j"] for row in energy)
    return {"sampler_rows": held_sum(kept, "sampler_rows"),
            "gpu_busy_share": ratio(sum(row["gpu_busy_share"] * row["sampler_rows"] for row in busy), sum(row["sampler_rows"] for row in busy)),
            "gpu_energy_j_per_episode": ratio(joules, len(energy)),
            "gpu_energy_j_per_passed_episode": ratio(joules, sum(1 for row in energy if as_bool(row["pass"])))}


def cost_summary(kept):
    """The card-seconds that the episodes of a cell held the card alone, at one run in flight, per episode and per passed episode,
    and the pass rate of the kept episodes."""
    held = [row for row in kept if row.get("held_card_s") is not None]
    seconds = sum(row["held_card_s"] for row in held)
    scored = [row for row in kept if as_bool(row["pass"]) is not None]
    return {"pass_rate": ratio(sum(1 for row in scored if as_bool(row["pass"])), len(scored)),
            "held_card_s_per_episode": ratio(seconds, len(held)),
            "held_card_s_per_passed_episode": ratio(seconds, sum(1 for row in held if as_bool(row["pass"])))}


def job_summary(index, jobs):
    """(summary, notes): the cost and the rate of a cell over the jobs that played it alone, the card-seconds of their Elapsed over
    their model episodes and over the passed ones, not divided when one of those jobs leaves its own undivided, and both rates as
    the sums of the counts over the sum of the windows' lengths; the notes name each job that leaves its card-seconds undivided,
    and each whose Elapsed, which the card-seconds of the cell divide, holds the time of runs without the served model."""
    alone = [job for job in jobs if job.get("cell") == index and job.get("elapsed_s") is not None]
    rated = [job for job in alone if job.get("window_s") is not None]
    window = sum(job["window_s"] for job in rated) if rated else None
    undivided = [job["grid"] for job in alone if job["card_s_undivided"]]
    elapsed = sum(job["elapsed_s"] for job in alone)
    notes = [f"the job {grid} leaves its card-seconds undivided, so those of the cell are not divided" for grid in undivided]
    notes += [] if undivided else [f"the card-seconds of the cell divide the Elapsed of the job {job['grid']}, which holds the time of "
                                   f"the runs without the served model ({', '.join(job['runs_without_model'])})"
                                   for job in alone if job["runs_without_model"]]
    return {"jobs_alone": [job["grid"] for job in alone],
            "card_s_per_episode": None if undivided else ratio(elapsed, sum(job["model_episodes"] for job in alone)),
            "card_s_per_passed_episode": None if undivided else ratio(elapsed, sum(job["passed"] for job in alone)),
            "window_s": window,
            "ending_per_card_hour": ratio(3600 * sum(job["episodes_ending_inside"] for job in rated), window),
            "shares_per_card_hour": ratio(3600 * sum(job["episode_shares_inside"] for job in rated), window)}, notes


def cell_summary(index, key, runs, rows, call_seconds, call_figures, jobs, notes):
    """The summary of one cell: its key, its runs and jobs, the episodes kept and left out with their reasons, per check the
    episodes that fail it among those whose files it reads, the seconds and shares of every stage, the clock, the calls, the
    engine's split of the calls, the GPU, the cost, the rate, and the notes of its runs and of the jobs that played it alone."""
    kept = [row for row in rows if not row["left_out"]]
    jobs_view, jobs_notes = job_summary(index, jobs)
    return {"cell": index, "key": key, "runs": runs, "jobs": sorted({row["grid"] for row in rows if row["grid"]}),
            "episodes": len(rows), "kept": len(kept),
            "left_out": dict(collections.Counter(row["left_out"] for row in rows if row["left_out"])),
            "checks": {name: {"failing": sum(1 for row in kept if row[name]), "checked": sum(1 for row in kept if row[name] is not None)}
                       for name in stage_profile.CHECKS},
            "seconds": seconds_summary(kept), "clock": clock_summary(kept), "calls": call_summary(kept, call_seconds),
            "engine": engine_summary(kept, call_figures), "gpu": gpu_summary(kept), "cost": cost_summary(kept),
            "jobs_view": jobs_view, "notes": notes + jobs_notes}
