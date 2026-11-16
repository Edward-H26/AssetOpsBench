import json
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import HYPOTHESIS_MEASURES, OURS, AGENTS_PRESET, AGENT_ROWS, row_label  # noqa: E402
from . import cell_statistics, episode_selection, hypothesis_tables


CONTRAST_MEASURES = (*HYPOTHESIS_MEASURES, "faulty_set_right")  # Top-1, pass and the exact match of the faulty set


def _triples(ours, theirs_mcp, theirs, metric):
    """(the CausalDiagnosis agent, the common design with the MCP, the common design without it) matched on identity and every factor but the agent and the
    diagnostics server, all three carrying the metric, so the difference against the design without the MCP is the sum of its two
    parts on the same episodes."""
    fields = [f for f in (*cell_statistics.IDENTITY, *cell_statistics.FACTORS) if f not in episode_selection.AGENT_KEY]

    def key(e):
        return tuple(e.get(f, "") for f in fields)

    with_mcp, without = {key(e): e for e in cell_statistics.with_metric(theirs_mcp, metric)}, {key(e): e for e in cell_statistics.with_metric(theirs, metric)}
    return [(e, with_mcp[key(e)], without[key(e)]) for e in cell_statistics.with_metric(ours, metric) if key(e) in with_mcp and key(e) in without]


def table_system_contrast(episodes, ctx):
    """SYSTEM_CONTRAST: per model, tool surface and common design of the main table, on its preset, the paired difference of the
    CausalDiagnosis agent with the MCP against the common design without it for Top-1, pass and the exact match of the faulty set,
    beside its two parts on the same matched episodes: the design part, the CausalDiagnosis agent against the design both with the MCP, and the MCP
    part, the design with the MCP against itself without it. The total is the sum of its parts. No p value and no Holm cell, since
    the table is no test."""
    ours = next(row for row in AGENT_ROWS if row[0] == OURS and row[1])
    main = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == AGENTS_PRESET and episode_selection.at_main_levels(e)]
    rows = []
    for (model, surface), eps in cell_statistics.groups(main, "model", "surface").items():
        mine = episode_selection._arm(eps, row_label(ours, model), True)
        for common in (row for row in AGENT_ROWS if row[0] != OURS and not row[1]):
            with_mcp = next(row for row in AGENT_ROWS if row[0] == common[0] and row[1])
            theirs_mcp, theirs = episode_selection._arm(eps, row_label(with_mcp, model), True), episode_selection._arm(eps, row_label(common, model), False)
            if not (mine and theirs_mcp and theirs):
                continue
            row = {"agent": row_label(ours, model), "design": row_label(common, model), "model": model, "surface": surface, "config": AGENTS_PRESET}
            for metric in CONTRAST_MEASURES:
                triples = _triples(mine, theirs_mcp, theirs, metric)
                row[f"{metric}_pairs"], row[f"{metric}_clusters"] = len(triples), len({cell_statistics.cluster_of(a) for a, _b, _c in triples})
                for part, pairs in (("total", [(a, c) for a, _b, c in triples]), ("design", [(a, b) for a, b, _c in triples]),
                                    ("mcp", [(b, c) for _a, b, c in triples])):
                    row.update(cell_statistics._part_cells(pairs, metric, part, ctx))
            rows.append(cell_statistics.trace(ctx, row, mine, theirs))
    return rows


def fault_subsets(profile):
    """The subsets of the starts that PAIRS_BY_FAULT reads, from the profile of the asset: for each component the starts in which it is
    healthy, then for each component and each of its faulty levels the starts that hold that fault, as (subset, component, level)."""
    healthy = [("healthy", c, profile.healthy[c]) for c in profile.components]
    return healthy + [("fault", c, level) for c in profile.components for level in profile.levels[c] if level != profile.healthy[c]]


def _in_subset(episode, profile, component, level):
    block = episode.get("block")
    return bool(block) and dict(zip(profile.components, json.loads(block))).get(component) == level


def table_pairs_by_fault(episodes, ctx):
    """PAIRS_BY_FAULT: for every comparison of the agents family and both of its measures, per model and tool surface on the main
    table's preset, the paired difference over the starts in which a component is healthy, for each component in turn, and over the
    starts that hold a fault, for each component and faulty level of the asset file. The test starts are read as an exchangeable
    sample of the fault situations of this class of asset, so a difference is the mean paired difference over them. Each row prints
    its pairs and clusters, and below the registered smallest number of clusters the difference alone. No test."""
    rows = []
    for (model, config, *setting), eps in episode_selection.main_level_groups(episodes).items():
        if config != AGENTS_PRESET:
            continue
        for row, level, against in hypothesis_tables.comparison_arms(eps, hypothesis_tables.hypothesis_comparisons(model), model, config, setting):
            profile = get_profile(level[0].get("asset_id") or DEFAULT_ASSET)
            for subset, component, value_ in fault_subsets(profile):
                cells = {**row, "subset": subset, "component": component, "level": value_}
                for metric in HYPOTHESIS_MEASURES:
                    pairs = [(a, b) for a, b in cell_statistics.matched(level, against, episode_selection.AGENT_KEY, metric) if _in_subset(a, profile, component, value_)]
                    cells.update(cell_statistics._part_cells(pairs, metric, "diff", ctx))
                    cells[f"{metric}_pairs"], cells[f"{metric}_clusters"] = len(pairs), len({cell_statistics.cluster_of(a) for a, _b in pairs})
                rows.append(cell_statistics.trace(ctx, cells, level, against))
    return rows
