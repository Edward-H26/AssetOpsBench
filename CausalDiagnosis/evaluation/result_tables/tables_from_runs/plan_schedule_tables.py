from itertools import takewhile
from pathlib import Path
from CausalDiagnosis.utils.csv_rows import as_bool, as_float  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import PARALLEL_CONTRAST_ROWS, PARALLEL_PRESET, PARALLEL_ROW  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import cell_statistics, family_correction, episode_selection


PARALLEL_METRICS = ("repair_declared", "repair_delay", "thinking_cycles", "work_orders", "downtime_total", "pass", "cost_normalized", "llm_calls",
                    "ranking_calls")


def parallel_arms(episodes):
    """{(model, surface): (on, off)}: the row of the pair of the schedule of candidate plans on its preset as released, and the same agent with
    the switch off (ablation Ag9, the label suffix of the switch), each at the main level of every setting. The rows with the switch
    off enter no other table."""
    on = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == PARALLEL_PRESET and episode_selection.at_main_levels(e) and episode_selection.in_rows(e, (PARALLEL_ROW,))]
    off = [e for e in episode_selection.causal_episodes(episodes) if e.get("config") == PARALLEL_PRESET and episode_selection.at_main_levels(e) and episode_selection.switch_of(e) == "parallel_off"
           and as_bool(e.get("diagnostics_server")) is PARALLEL_ROW[1]]
    arms = {}
    for key, eps in cell_statistics.groups(on, "model", "surface").items():
        arms.setdefault(key, ([], []))[0].extend(eps)
    for key, eps in cell_statistics.groups(off, "model", "surface").items():
        arms.setdefault(key, ([], []))[1].extend(eps)
    return arms


def parallel_designs(episodes):
    """{(model, surface): [episodes, ...]}: per row of design.PARALLEL_CONTRAST_ROWS, in that order, its episodes on the preset of the pair
    at the main level of every setting, which the table of the pair sets beside it."""
    played = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == PARALLEL_PRESET and episode_selection.at_main_levels(e)]
    designs = {}
    for row in PARALLEL_CONTRAST_ROWS:
        for key, eps in cell_statistics.groups([e for e in played if episode_selection.in_rows(e, (row,))], "model", "surface").items():
            designs.setdefault(key, []).append(eps)
    return designs


MODEL_PLAN = "model"  # the source of every plan the model wrote begins so in plans.json (model, model_retry, model_repaired)


def advised_batches(episode):
    """For each batch of reads that the board advised to a plan the model wrote, in the order of plans.json, whether the model kept it: the
    schedule recorded with the plan holds reads, and the read steps that begin the plan, as (sensor group, cycles) in order, are those
    reads cut to the plan length of the run, which no plan can pass. A plan of the rule and a turn without a usable plan have no batch of
    the model. Empty for an episode without plans.json."""
    folder = episode.get("episode_dir")
    plans = read_json(Path(folder) / "plans.json", None) if folder is not None else None
    cut = int(episode["plan_length"]) if episode.get("plan_length") not in (None, "") else None
    kept = []
    for record in (plans or {}).get("plans") or []:
        advised = [(read["sensor_group"], int(read["cycles"])) for read in (record.get("schedule") or {}).get("reads") or []][:cut]
        if not advised or not str(record.get("source", "")).startswith(MODEL_PLAN):
            continue
        reads = takewhile(lambda step: step.get("kind") == "read", (record.get("plan") or {}).get("steps") or [])
        kept.append([(step["args"]["sensor_group"], int(step["args"]["cycles"])) for step in reads] == advised)
    return kept


PARALLEL_BATCH_COLUMNS = ("advised_batches", "kept_batches", "changed_batches", "kept_share", "kept_share_low", "kept_share_high", "kept_share_n")


def batch_cells(episodes, ctx):
    """The batches of advised_batches over the episodes of a row, how many the model kept and changed, and the share it kept with its
    interval and standard error from a bootstrap clustered by recorded block, empty for a row without one."""
    batches = [(cell_statistics.cluster_of(e), kept) for e in episodes for kept in advised_batches(e)]
    cells = {"advised_batches": len(batches), "kept_batches": sum(1 for _, kept in batches if kept),
             "changed_batches": sum(1 for _, kept in batches if not kept), "kept_share_n": len(batches)}
    if not batches:
        return {**cells, **dict.fromkeys(("kept_share", "kept_share_low", "kept_share_high", "kept_share_se"), "")}
    boot = cell_statistics.proportion_cells([1.0 if kept else 0.0 for _, kept in batches], [cluster for cluster, _ in batches], ctx)
    return {**cells, **cell_statistics.interval_cells("kept_share", boot, ctx, estimate="proportion")}


def gap_explained(on, off, other, direction):
    """The share of the gap between another design and the row with the switch on that the switch explains: off less on over the design
    less on, both read in the registered direction of the family (below, the lower value is the better), so that a positive gap is the
    design behind the row with the switch on. Empty when a mean is missing or the gap is not positive, since no gap is then left to explain."""
    if None in (on, off, other):
        return ""
    sign = -1 if direction == "below" else 1
    gap, effect = sign * (on - other), sign * (on - off)
    return _r(effect / gap) if gap > 0 else ""


def table_parallel(episodes, ctx):
    """PARALLEL: the two rows of the pair of the schedule of candidate plans, the switch on and off, then the rows of parallel_designs beside them, each
    over its registered episodes with the counting rule: the mean and standard error of PARALLEL_METRICS, the batches that the board
    advised to a plan of the model with how many it kept (batch_cells), and on the row of another design the share of its gap to the row
    with the switch on that the switch explains, on the measure of the family parallel_time (gap_explained). None of them is a hypothesis
    test, which PARALLEL_pairs holds."""
    family = cell_statistics.registered_of(ctx)["families"]["parallel_time"]
    measure, direction = family["measures"][0], family["direction"]
    designs, rows = parallel_designs(episodes), []

    def cells(eps, model, surface, switch):
        row = {"agent": eps[0].get("agent"), "model": model, "surface": surface, "config": PARALLEL_PRESET, "switch": switch, "n": len(eps),
               "status": cell_statistics.row_status(eps, ctx)}
        for m in PARALLEL_METRICS:
            row.update(cell_statistics.summary(eps, m, ctx))
        return {**row, **batch_cells(eps, ctx), "gap_explained": ""}

    for (model, surface), (on, off) in parallel_arms(episodes).items():
        means = {}
        for switch, eps in (("on", on), ("off", off)):
            if eps:
                row = cells(eps, model, surface, switch)
                means[switch] = as_float(row[measure])
                rows.append(cell_statistics.trace(ctx, row, eps))
        for eps in designs.get((model, surface), []):
            row = cells(eps, model, surface, "")
            row["gap_explained"] = gap_explained(means.get("on"), means.get("off"), as_float(row[measure]), direction)
            rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def _mean_difference(pairs, metric):
    differences = [cell_statistics.value(a, metric) - cell_statistics.value(b, metric) for a, b in pairs]
    return _r(sum(differences) / len(differences)) if differences else ""


def table_parallel_pairs(episodes, ctx):
    """PARALLEL_pairs, the family parallel_time: the paired difference of the time to repair in declared cycles, the switch on less
    off, matched by seed, exact by convolution, Holm over the registered size, the registered direction below zero, the two margins
    on the paired differences of pass and of the cost over the oracle, and the split by pairs with and without a refuted repair,
    which no test reads. Not run while either row is incomplete."""
    family_spec = cell_statistics.registered_of(ctx)["families"]["parallel_time"]
    margin_pass, margin_cost = family_spec.get("margin_pass"), family_spec.get("margin_cost")
    rows = []
    for (model, surface), (on, off) in parallel_arms(episodes).items():
        if not (on and off):
            continue
        row = {"agent": on[0].get("agent"), "reference": off[0].get("agent"), "model": model, "surface": surface, "config": PARALLEL_PRESET,
               "status": cell_statistics.row_status(on, ctx) or cell_statistics.row_status(off, ctx), "margin_pass": "" if margin_pass is None else margin_pass,
               "margin_cost": "" if margin_cost is None else margin_cost}
        if not row["status"]:
            pairs = cell_statistics.matched(on, off, ("agent",), "repair_declared")
            row.update(cell_statistics.paired_difference(pairs, "repair_declared", ctx))
            passes, costs = cell_statistics.paired_difference(cell_statistics.matched(on, off, ("agent",), "pass"), "pass", ctx), cell_statistics.matched(on, off, ("agent",), "cost_normalized")
            # the two margins are read on the intervals of the differences, so neither difference brings a p value into the table
            row.update({k: v for k, v in passes.items() if k not in ("pairs", "clusters", "pass_p")})
            row.update({k: v for k, v in cell_statistics.paired_difference(costs, "cost_normalized", ctx).items() if k not in ("pairs", "clusters", "cost_normalized_p")})
            row["cost_pairs_without_cost"] = passes["pairs"] - len(costs)
            refuted = [(a, b) for a, b in pairs if a.get("refuted_repair") is True or b.get("refuted_repair") is True]
            plain = [(a, b) for a, b in pairs if (a, b) not in refuted]
            row.update({"refuted_pairs": len(refuted), "refuted_diff": _mean_difference(refuted, "repair_declared"),
                        "unrefuted_pairs": len(plain), "unrefuted_diff": _mean_difference(plain, "repair_declared")})
        rows.append(cell_statistics.trace(ctx, row, on, off))
        family_correction.holm_over([row], ("repair_declared",), ctx["alpha"], family_correction.registered_size(ctx, "parallel_time"))  # one family per model and tool surface
    for row in rows:
        low, high = as_float(row.get("pass_diff_low")), as_float(row.get("cost_normalized_diff_high"))
        row["margin_pass_met"] = "" if margin_pass is None or low is None else low > -margin_pass
        row["margin_cost_met"] = "" if margin_cost is None or high is None else high < margin_cost
        row["reading"] = family_correction.reading(row, ("repair_declared",), {"repair_declared": -1})
        met = (row["margin_pass_met"], row["margin_cost_met"])
        row["holds"] = "" if "" in met else row["reading"] == "holds" and all(met)
    return rows
