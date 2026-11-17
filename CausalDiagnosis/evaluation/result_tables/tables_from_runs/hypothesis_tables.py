from CausalDiagnosis.utils.csv_rows import as_float  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import H1_TOP, H1_TOP_PRESET, H2_PRIME, H2_PRIME_HELPS, H2_PRIME_MEASURES, HYPOTHESES, HYPOTHESIS_MEASURES, LADDER, ROBUSTNESS_GUIDANCE_LEVEL, AGENTS_PRESET, AGENT_ROWS, row_label  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_bootstrap, ratio  # noqa: E402
from . import cell_statistics, family_correction, episode_selection


def turns_pair(level, reference, ctx):
    """The paired difference of the capped turns between two rows on matched episodes, the difference of their restricted
    mean survival times over turns, with its bootstrap interval clustered by block and its p value."""
    cells = cell_statistics.paired(level, reference, episode_selection.AGENT_KEY, "turns_capped", ctx)
    return {key.replace("turns_capped", "turns_rmst"): v for key, v in cells.items() if key not in ("pairs", "clusters")}


def cost_ratio(level, reference, ctx):
    """The mean cost of a level over the mean cost of its reference on the matched episodes, the ratio of mean cost that the
    registered analysis names, with a bootstrap interval clustered by block over the pairs (cluster_bootstrap over the sum of
    the level's costs and the sum of the reference's, empty when the reference sums to zero). The p value is the one of the
    paired cost difference, since the ratio of the means is one exactly when the mean difference is zero."""
    pairs = cell_statistics.matched(level, reference, episode_selection.AGENT_KEY, "cost")
    boot = cluster_bootstrap([[cell_statistics.value(a, "cost"), cell_statistics.value(b, "cost")] for a, b in pairs], [cell_statistics.cluster_of(a) for a, _ in pairs], ratio,
                             ctx["resamples"], ctx["seed"])
    return {**cell_statistics.interval_cells("cost_ratio", boot, ctx), "cost_ratio_p": cell_statistics.paired(level, reference, episode_selection.AGENT_KEY, "cost", ctx)["cost_p"]}


# the paired cells of a hypothesis of the six rows: top-1 and pass, which decide, then success and the verifier's reward,
# which are printed, the first giving the pairs and clusters
PAIR_MEASURES = (*HYPOTHESIS_MEASURES, "success", "reward")
SCORING_MEASURES = ("pass", "judge_pass")  # the verified pass and the judge's pass, the two scoring rules of the scoring pairs
# the two families of AGENTS_pairs, each with a Holm of its own: the 14 tests of the seven hypotheses, and the scoring pairs
FAMILY = {"hypotheses": HYPOTHESIS_MEASURES, "scoring": SCORING_MEASURES}


def agent_rows(model):
    """The six rows of design.AGENT_ROWS as (agent label, diagnostics_server) in the order of the tables."""
    return [(row_label(row, model), row[1]) for row in AGENT_ROWS]


def hypothesis_comparisons(model):
    """The comparisons of design.HYPOTHESES as (name, (agent, server), (reference, server))."""
    return [(name, (row_label(row, model), row[1]), (row_label(reference, model), reference[1])) for name, row, reference in HYPOTHESES]


def family_of(row):
    return "scoring" if row["hypothesis"] == "scoring" else "hypotheses"


def holm_family(rows, alpha, sizes=None):
    """Holm within each family of FAMILY over the rows of one model, config and tool surface: the top-1 and pass tests of the seven
    hypotheses over the registered size of the agents family, a test not run entering at one, and apart from them the verified
    and judge pass tests of the scoring pairs. Every tested measure gets <measure>_holm_p and <measure>_reject, and every row the
    number of tests its family counted (family_tests)."""
    for family, measures in FAMILY.items():
        family_correction.holm_over([row for row in rows if family_of(row) == family], measures, alpha, (sizes or {}).get(family))
    return rows


def holds(row):
    """Whether a hypothesis holds on its comparison: the paired differences of top-1 and of pass both lie above zero and both
    Holm decisions reject. Empty while one of the four cells is missing."""
    differences = [as_float(row.get(f"{m}_diff")) for m in HYPOTHESIS_MEASURES]
    confirmed = [row.get(f"{m}_reject") for m in HYPOTHESIS_MEASURES]
    if None in differences or None in confirmed:
        return ""
    return all(d > 0 for d in differences) and all(confirmed)


def reversed_order(row):
    """Whether the judge orders the two rows of a scoring pair against the verified pass, the two differences having
    opposite signs. Empty when either difference is missing."""
    verified, judged = as_float(row.get("pass_diff")), as_float(row.get("judge_pass_diff"))
    return "" if verified is None or judged is None else verified * judged < 0


def _comparison_row(hypothesis, agent, server, reference, reference_server, model, config, setting):
    return {"hypothesis": hypothesis, "agent": agent, "agent_server": server, "reference": reference, "reference_server": reference_server,
            "model": model, "config": config, **dict(zip(episode_selection.PAIR_SETTINGS, setting))}


def comparison_arms(eps, comparisons, model, config, setting):
    """(row, level, reference) of every comparison whose two arms have episodes in the group, the row naming the
    hypothesis, the two agents with their server levels, the model, the config and the settings of PAIR_SETTINGS."""
    for hypothesis, (agent, server), (reference, reference_server) in comparisons:
        level, against = episode_selection._arm(eps, agent, server), episode_selection._arm(eps, reference, reference_server)
        if level and against:
            yield _comparison_row(hypothesis, agent, server, reference, reference_server, model, config, setting), level, against


def _effect_cells(level, against, ctx):
    """The paired cells of a hypothesis: PAIR_MEASURES, the ratio of mean cost and the paired difference of the turns."""
    return {**cell_statistics._paired_cells(level, against, episode_selection.AGENT_KEY, PAIR_MEASURES, ctx), **cost_ratio(level, against, ctx), **turns_pair(level, against, ctx)}


def table_agents_pairs(episodes, ctx):
    """The hypotheses of the six rows and the scoring pairs per model and tool surface on the preset of the agents table (layer 0),
    over the episodes at the main level of every setting (at_main_levels), so that no ablation run of the grid and no trained
    agent pools into a pair. The seven hypotheses of design.HYPOTHESES pair the CausalDiagnosis agent against each AssetOpsBench
    agent, both without and both with the CausalDiagnosis MCP (H1a to H1d), and every agent with the MCP against itself without it
    (H2a to H2c), on the paired differences of top-1 and pass, which decide, of success and of the verifier's reward, with the ratio
    of mean cost and the paired difference of the restricted mean survival time over turns (turns_pair), which no Holm decision
    covers. The scoring pairs pair every two of the six rows, the later row of the tables' order against the earlier, under the
    verified pass and under the judge's pass. Episodes match on seed, scenario and every factor other than agent and
    diagnostics_server, and a comparison with no episode on one side is left out. Holm runs within each family (holm_family),
    and holds says whether a hypothesis holds on its comparison."""
    rows = []
    for (model, config, *setting), eps in episode_selection.main_level_groups(episodes).items():
        if config != AGENTS_PRESET:
            continue
        six = agent_rows(model)
        comparisons = hypothesis_comparisons(model) + [("scoring", six[j], six[i]) for i in range(len(six)) for j in range(i + 1, len(six))]
        family = []
        for row, level, against in comparison_arms(eps, comparisons, model, config, setting):
            if row["hypothesis"] == "scoring":
                row["status"] = cell_statistics.row_status(level, ctx) or cell_statistics.row_status(against, ctx)
                row.update(cell_statistics._paired_cells([] if row["status"] else level, [] if row["status"] else against, episode_selection.AGENT_KEY, SCORING_MEASURES, ctx))
                row["reversed"] = reversed_order(row)
            else:
                row["status"] = cell_statistics.row_status(level, ctx) or cell_statistics.row_status(against, ctx)
                if not row["status"]:
                    row.update(_effect_cells(level, against, ctx))
                if row["hypothesis"].startswith("H1"):
                    row.update(family_correction.floor_cells(against, ctx))
            family.append(cell_statistics.trace(ctx, row, level, against))
        # the scoring family counts its tests that ran and both tests of every scoring pair that was not run (row_status), at p = 1
        scoring = sum(1 for row in family if row["hypothesis"] == "scoring" for m in SCORING_MEASURES if as_float(row.get(f"{m}_p")) is not None) + len(
            SCORING_MEASURES) * sum(1 for row in family if row["hypothesis"] == "scoring" and row.get("status"))
        for row in holm_family(family, ctx["alpha"], {"hypotheses": family_correction.registered_size(ctx, "agents"), "scoring": scoring}):
            if family_of(row) == "hypotheses":
                row["holds"] = holds(row)
                row["reading"] = family_correction.reading(row, HYPOTHESIS_MEASURES)
            rows.append(row)
    return rows


def table_h1_guidance(episodes, ctx):
    """H1 under guidance level ROBUSTNESS_GUIDANCE_LEVEL, a robustness check: the comparisons of H1 in design.HYPOTHESES on
    the En7 runs at that level, per model and tool surface, the paired differences of top-1 and pass with sign flip p values and a Holm of
    their own over these tests, which decides nothing of H1, with the floor of each baseline beside it. A comparison of an incomplete or
    refused row is not run (row_status) and enters the Holm at p = 1."""
    runs = [e for e in episode_selection.placed_episodes(episodes, "En7", "guidance_level") if str(e.get("guidance_level")) == str(ROBUSTNESS_GUIDANCE_LEVEL)]
    rows = []
    for (model, surface), eps in cell_statistics.groups(runs, "model", "surface").items():
        family = []
        comparisons = [comparison for comparison in hypothesis_comparisons(model) if comparison[0].startswith("H1")]
        for row, level, against in comparison_arms(eps, comparisons, model, AGENTS_PRESET, ()):
            status = cell_statistics.row_status(level, ctx) or cell_statistics.row_status(against, ctx)
            row.update({"surface": surface, "guidance_level": ROBUSTNESS_GUIDANCE_LEVEL, "status": status,
                        **cell_statistics._paired_cells([] if status else level, [] if status else against, episode_selection.AGENT_KEY, HYPOTHESIS_MEASURES, ctx), **family_correction.floor_cells(against, ctx)})
            family.append(cell_statistics.trace(ctx, row, level, against))
        for row in family_correction.holm_over(family, HYPOTHESIS_MEASURES, ctx["alpha"], len(family) * len(HYPOTHESIS_MEASURES)):
            row["holds"] = holds(row)
            rows.append(row)
    return rows


def h2_prime_cells(episodes, ctx):
    """The comparisons of H2' before their correction, per model and tool surface: on the top rung of the ladder (the last preset of
    design.LADDER), every agent with the CausalDiagnosis MCP against itself without it, both as released and at the main level of
    every setting, on the paired differences of H2_PRIME_MEASURES matched on seed, scenario and every factor other than the agent and
    the level of the MCP. A comparison of an incomplete or refused row is not run (row_status) and enters the Holm at p = 1."""
    top = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == LADDER[-1] and episode_selection.at_main_levels(e)]
    cells = {}
    for (model, surface), eps in cell_statistics.groups(top, "model", "surface").items():
        family = []
        for name, row, reference in H2_PRIME:
            level, against = episode_selection._arm(eps, row_label(row, model), row[1]), episode_selection._arm(eps, row_label(reference, model), reference[1])
            if level and against:
                keys = {"hypothesis": name, "agent": row_label(row, model), "agent_server": row[1], "reference": row_label(reference, model),
                        "reference_server": reference[1], "model": model, "config": LADDER[-1], "surface": surface,
                        "status": cell_statistics.row_status(level, ctx) or cell_statistics.row_status(against, ctx)}
                paired = cell_statistics._paired_cells([] if keys["status"] else level, [] if keys["status"] else against, episode_selection.AGENT_KEY, H2_PRIME_MEASURES, ctx)
                family.append(cell_statistics.trace(ctx, {**keys, **paired}, level, against))
        cells[(model, surface)] = family
    return cells


def table_h2_prime(episodes, ctx):
    """H2': the comparisons of h2_prime_cells with Holm over the correction group of the family (corrected_family), and holds when
    both differences lie on the side where the MCP helps and both Holm decisions reject."""
    rows = []
    for family in corrected_family("h2_prime", episodes, ctx).values():
        for row in family:
            row["holds"] = helps_on_every_measure(row)
            row["reading"] = family_correction.reading(row, H2_PRIME_MEASURES, H2_PRIME_HELPS)
            rows.append(row)
    return rows


def helps_on_every_measure(row):
    """Whether an H2' comparison holds: every difference of H2_PRIME_MEASURES lies on the side of H2_PRIME_HELPS and every Holm decision
    rejects. Empty while a cell is missing."""
    differences = [as_float(row.get(f"{m}_diff")) for m in H2_PRIME_MEASURES]
    confirmed = [row.get(f"{m}_reject") for m in H2_PRIME_MEASURES]
    if None in differences or None in confirmed:
        return ""
    return all(d * H2_PRIME_HELPS[m] > 0 for d, m in zip(differences, H2_PRIME_MEASURES)) and all(confirmed)


# ----- the families of the full environment: H1 on the top layer, the full environment against the base layer, the schedule of candidate plans -----
def h1_top_cells(episodes, ctx):
    """The comparisons of H1 asked again on the preset of training, where every big layer is on, before their correction, per model
    and tool surface, from cells that exist: the rows without the MCP are the reference lines without it on all six big layers and the rows with it
    the part environment, both on the server of grid_agents.txt. Top-1 and pass, and the floor of each reference."""
    top = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == H1_TOP_PRESET and episode_selection.at_main_levels(e)]
    cells = {}
    for (model, surface), eps in cell_statistics.groups(top, "model", "surface").items():
        comparisons = [(name, (row_label(row, model), row[1]), (row_label(ref, model), ref[1])) for name, row, ref in H1_TOP]
        family = []
        for row, level, against in comparison_arms(eps, comparisons, model, H1_TOP_PRESET, ()):
            row["surface"], row["status"] = surface, cell_statistics.row_status(level, ctx) or cell_statistics.row_status(against, ctx)
            if not row["status"]:
                row.update(cell_statistics._paired_cells(level, against, episode_selection.AGENT_KEY, HYPOTHESIS_MEASURES, ctx))
            row.update(family_correction.floor_cells(against, ctx))
            family.append(cell_statistics.trace(ctx, row, level, against))
        cells[(model, surface)] = family
    return cells


def table_h1_top(episodes, ctx):
    """The family h1_top: the comparisons of h1_top_cells with Holm over the correction group of the family (corrected_family), and
    the reading."""
    rows = []
    for family in corrected_family("h1_top", episodes, ctx).values():
        for row in family:
            row["holds"], row["reading"] = holds(row), family_correction.reading(row, HYPOTHESIS_MEASURES)
            rows.append(row)
    return rows


# the families that a group of primary may join, each with its builder of the rows before the correction and its tested measures
CORRECTED = {"h1_top": (h1_top_cells, HYPOTHESIS_MEASURES), "h2_prime": (h2_prime_cells, H2_PRIME_MEASURES)}


def correction_group(ctx, family):
    """The families that the registered file corrects together with a family: its group of primary, or the family alone."""
    for group in cell_statistics.registered_of(ctx)["primary"]["groups"]:
        if family in group["families"]:
            return tuple(group["families"])
    return (family,)


def corrected_family(name, episodes, ctx):
    """The rows of a family of CORRECTED per model and tool surface, Holm run once over every family of its correction group on the
    same model and tool surface, its size the sum of their registered sizes. A group that joins a family outside CORRECTED is
    refused, since the table of that family corrects apart."""
    group = correction_group(ctx, name)
    outside = [family for family in group if family not in CORRECTED]
    if outside:
        raise ValueError(f"the registered group of {name} joins {', '.join(outside)}, whose table corrects apart")
    cache = ctx.setdefault("corrected_families", {})
    key = (group, id(episodes))
    if key not in cache:
        built = {family: CORRECTED[family][0](episodes, ctx) for family in group}
        size = sum(family_correction.registered_size(ctx, family) for family in group)
        for cell in sorted(set().union(*(set(found) for found in built.values()))):
            family_correction.holm_joint([(built[family].get(cell, []), CORRECTED[family][1]) for family in group], ctx["alpha"], size)
        cache[key] = (episodes, built)  # the episodes stay referenced, so that their id names them while the entry lives
    return cache[key][1][name]
