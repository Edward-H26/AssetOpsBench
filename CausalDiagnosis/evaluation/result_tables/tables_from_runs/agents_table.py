import json
from fractions import Fraction
from CausalDiagnosis.utils.csv_rows import as_float  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402  the verifier's reading of the log, which the time to repair in declared cycles shares
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import REPLICATES  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_bootstrap, majority_share, ratio  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import cell_statistics, family_correction, episode_selection, episode_fields, judge_tables


def turns_summary(episodes, ctx):
    """The turns to a correct diagnosis of the scores table of the paper: the restricted mean survival time over turns,
    restricted at the smallest step limit among the runs of the episodes, with a percentile interval from a bootstrap that
    resamples recorded blocks. An episode is censored only at the step limit of its run, so none is censored before the
    restriction, and the restricted mean is the mean of the turns capped at it."""
    kept = [(e, r) for e in episodes if (r := episode_fields.turn_record(e)) is not None]
    if not kept:
        return {"turns_rmst": "", "turns_rmst_low": "", "turns_rmst_high": "", "turns_rmst_n": 0}
    limit = min(r[2] for _, r in kept)
    boot = cluster_bootstrap([[min(r[0], limit), 1.0] for _, r in kept], [cell_statistics.cluster_of(e) for e, _ in kept], ratio, ctx["resamples"], ctx["seed"])
    return {**cell_statistics.interval_cells("turns_rmst", boot, ctx), "turns_rmst_n": len(kept)}


# the cells of a row of AGENTS: the verified outcome (top-1, pass, success, the wrong diagnoses, the declarations and the
# model calls) and the reward, the cost and the delays beside it
AGENT_METRICS = ("pass", "diagnosis_all_correct", "faulty_set_right", "restored_clean", "graded_top1", "graded_pass", "success", "reward",
                 "diagnosis_wrong", "abstained",
                 "unmatched_evidence", "llm_calls", "cost", "cost_normalized", "downtime_total", "detection_delay", "repair_delay")
COUNTS = ("played", "planned", "unscored", "unjudged", "incomplete", "refused", "replicates", "read")  # the counting rule of every row
REPLAY_RATES = ("pass", "diagnosis_all_correct", "faulty_set_right", "restored_clean", "success")  # the verified rates read again with replays failed
REPLAY_COLUMNS = ("replayed", "replays", *(f"{m}_replays_failed" for m in REPLAY_RATES))
PASS_KS = tuple(range(1, REPLICATES + 1))  # the k of pass^k, up to the start states that the test set plays of each start combination
PASS_K_COLUMNS = tuple(f"pass_k{k}{part}" for k in PASS_KS for part in ("", "_low", "_high", "_n"))
DO_NOTHING = "do_nothing"  # the reference policy whose cost is the zero of the normalized score


def start_key(episode):
    """The start of an episode on its preset: the config, the seed and the scenario, the key on which a reference policy that played
    the same starts is read beside a row."""
    return episode.get("config"), episode.get("seed"), episode.get("scenario_id") or ""


def normalized_score(episodes, nothing):
    """The normalized score of a row on cost (Fu et al., 2020): the mean cost of the do-nothing policy less the mean cost of the row,
    over the mean cost of the do-nothing policy less the mean cost of the oracle, the means over the episodes of the row whose start
    the do-nothing policy played on the same preset (nothing, its episodes by start_key), the oracle cost the one each episode records.
    It reads 0 for doing nothing and 1 for the oracle, and not run without such an episode or when the two references are equal."""
    kept = []
    for e in episodes:
        zero = nothing.get(start_key(e))
        costs = (as_float(e.get("cost")), as_float(e.get("cost_oracle")), None if zero is None else as_float(zero.get("cost")))
        if None not in costs:
            kept.append(costs)
    if not kept:
        return {"normalized_score": "not run", "normalized_score_n": 0}
    row, oracle, zero = (sum(c[i] for c in kept) / len(kept) for i in range(3))
    if zero == oracle:
        return {"normalized_score": "not run", "normalized_score_n": len(kept)}
    return {"normalized_score": _r((zero - row) / (zero - oracle)), "normalized_score_n": len(kept)}


def replay_cells(episodes):
    """The replays of a row, the episodes played once more after the infrastructure broke an attempt (replays of scores.csv above zero)
    and the replays in all, and each verified rate of REPLAY_RATES again with every replayed episode counted as a failure, the reading
    of the appendix beside the rate of the row."""
    replayed = [(as_float(e.get("replays")) or 0) > 0 for e in episodes]
    cells = {"replayed": sum(replayed), "replays": int(sum(as_float(e.get("replays")) or 0 for e in episodes))}
    for m in REPLAY_RATES:
        kept = int(sum(0 if again else (cell_statistics.value(e, m) or 0) for e, again in zip(episodes, replayed)))
        cells[f"{m}_replays_failed"] = _r(Fraction(kept, len(episodes))) if episodes else ""
    return cells


def pass_hat_k(episodes, ctx):
    """pass^k of tau-bench for every k of PASS_KS over the episodes of one row: per start combination, the recorded block, with c
    passing episodes among its n, the share C(c, k) / C(n, k) of its k-subsets in which every episode passes (verifier.pass_power_k),
    averaged over the combinations with at least k episodes, so that a combination with fewer start states enters every k it can.
    The interval and the standard error come from a bootstrap that resamples the clusters of the combinations, and the count of the
    combinations that enter stands beside each mean."""
    combinations = {}
    for e in cell_statistics.with_metric(episodes, "pass"):
        combinations.setdefault(e.get("block", ""), (cell_statistics.cluster_of(e), []))[1].append(int(cell_statistics.value(e, "pass")))
    cells = {}
    for k in PASS_KS:
        kept = [(cluster, verifier.pass_power_k(sum(passes), len(passes), k)) for cluster, passes in combinations.values() if len(passes) >= k]
        boot = cluster_bootstrap([[share, 1.0] for _, share in kept], [cluster for cluster, _ in kept], ratio, ctx["resamples"], ctx["seed"])
        cells.update({**cell_statistics.interval_cells(f"pass_k{k}", boot, ctx), f"pass_k{k}_n": len(kept)})
    return cells


def table_agents(episodes, ctx):
    """One row per CONDITION of the six rows (agent label, model, config, clock mode, tool surface, diagnostics_server level and
    SETTINGS), so every clock mode of a preset and every training variant is a row of its own, with the summary intervals of
    AGENT_METRICS: pass, top-1 accuracy (diagnosis_all_correct), success, which credits a correct declaration of insufficient
    evidence, the verifier's reward, the share of episodes whose submitted diagnosis is wrong (diagnosis_wrong), the share of
    episodes that declared insufficient evidence (abstained), the model calls per episode, cost, normalized cost, the production loss
    within the cost (downtime_total), the time to work order and the time to repair (detection_delay and repair_delay of the
    verifier, over the episodes that have one), the restricted mean survival time over turns to a correct diagnosis (turns_summary)
    and pass^k for every k of PASS_KS (pass_hat_k), with the hypotheses in AGENTS_pairs."""
    rows = []
    nothing = {start_key(e): e for e in episodes if e.get("agent") == DO_NOTHING and e.get("mode") == "gym"}
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        row = {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), **cell_statistics.completeness(eps, ctx), **normalized_score(eps, nothing)}
        for m in AGENT_METRICS:
            row.update(cell_statistics.summary(eps, m, ctx))
        row.update(turns_summary(eps, ctx))
        row.update(pass_hat_k(eps, ctx))
        row.update(judge_tables.judge_kappa(eps, ctx))
        row.update(exact_match_floor(eps, row["faulty_set_right"], ctx))
        row.update(replay_cells(eps))
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def faulty_set_of(episode):
    """The faulty set of the start of an episode, the components of its start combination (the block) off their healthy level, for
    any asset; None without a block."""
    block = episode.get("block")
    if not block:
        return None
    asset = get_profile(episode.get("asset_id") or DEFAULT_ASSET)
    return tuple(c for c, level in zip(asset.components, json.loads(block)) if level != asset.healthy[c])


def exact_match_floor(episodes, rate, ctx=None):
    """The constant-answer baseline of the exact match of the faulty set over the episodes of a row, the share of them whose faulty
    set is the most frequent one (stats.majority_share), its floor, the larger of the registered floor of pass and that share, and whether the
    rate of the row lies at or below the floor."""
    k, n = majority_share(s for s in (faulty_set_of(e) for e in episodes) if s is not None)
    if not n:
        return {"faulty_set_right_baseline_k": "", "faulty_set_right_baseline_n": 0, "faulty_set_right_baseline": "", "faulty_set_right_floor": "",
                "faulty_set_right_at_floor": ""}
    floor = max(family_correction.registered_floor(ctx), Fraction(k, n))
    measured = as_float(rate)
    return {"faulty_set_right_baseline_k": k, "faulty_set_right_baseline_n": n, "faulty_set_right_baseline": _r(Fraction(k, n)),
            "faulty_set_right_floor": _r(floor), "faulty_set_right_at_floor": "" if measured is None else Fraction(measured).limit_denominator(10 ** 6) <= floor}
