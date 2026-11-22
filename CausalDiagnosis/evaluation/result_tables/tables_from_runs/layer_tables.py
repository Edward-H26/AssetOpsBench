from CausalDiagnosis.experiments.table_definitions.experiment_tables import FULL_ENVIRONMENT, LADDER, LADDER_ROWS, AGENTS_PRESET, TRAINING_PRESET, row_label  # noqa: E402
from . import cell_statistics, family_correction, episode_selection


def table_policy_layers(episodes, ctx):
    """POLICY_LAYERS: the effect of each single layer of the ladder read on the reference policies, which need no model: per policy of
    design.REFERENCE_POLICIES, tool surface and rung of design.LADDER after the first, the paired difference of pass, success and
    reward between the rung and the rung below, matched on seed and every factor but the preset and the clock mode, as two rungs of
    the ladder are matched, with the 95 percent interval and the standard error. No test, no family and no Holm cell."""
    kept = [e for e in episodes if e.get("mode") == "gym" and e.get("agent") in episode_selection.REFERENCE_LABELS and episode_selection.at_main_levels(e)]
    rows = []
    for (agent, surface), eps in cell_statistics.groups(kept, "agent", "surface").items():
        for below, preset in zip(LADDER, LADDER[1:]):
            level, reference = [e for e in eps if e.get("config") == preset], [e for e in eps if e.get("config") == below]
            if not (level and reference):
                continue
            row = {"agent": agent, "surface": surface, "config": preset, "reference_config": below}
            for metric in POLICY_METRICS:
                pairs = cell_statistics.matched(level, reference, LADDER_KEY, metric)
                row.update(cell_statistics._part_cells(pairs, metric, "diff", ctx))
                row[f"{metric}_pairs"] = len(pairs)
            rows.append(cell_statistics.trace(ctx, row, level, reference))
    return rows


# ----- the reference policies on the ladder (POLICIES) and the effect of each big layer (LAYERS_pairs) -----
POLICY_METRICS = ("pass", "success", "reward")


def table_policies(episodes, ctx):
    """One row per policy of design.POLICY_TABLE_POLICIES, the reference policies of design.REFERENCE_POLICIES and the MCP follower,
    preset, clock mode and tool surface at the main level of every setting, the policies on the rows of the environment, which need
    no model: the pass, the verifier's success and the reward over the gym episodes, with intervals from a bootstrap clustered by block."""
    rows = []
    kept = [e for e in episodes if e.get("mode") == "gym" and e.get("agent") in episode_selection.POLICY_TABLE_LABELS and episode_selection.at_main_levels(e)]
    for (agent, config, clock, surface), eps in cell_statistics.groups(kept, "agent", "config", "clock_mode", "surface").items():
        row = {"agent": agent, "config": config, "clock_mode": clock, "surface": surface, "n": len(eps)}
        for m in POLICY_METRICS:
            row.update(cell_statistics.summary(eps, m, ctx))
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


LADDER_KEY = ("config", "clock_mode")  # what two rows of the ladder differ in besides the big layer, left out of the pairing key


def table_layers_pairs(episodes, ctx):
    """The effect of each big layer of the environment: per model, tool surface and layer of design.LADDER after the
    first, the paired difference of pass between the preset of the layer and the preset below it for each of the three rows with
    the MCP (design.LADDER_ROWS), each at the main level of every setting in its preset's own clock mode, matched on seed, scenario
    and every factor but the preset and the clock mode, and the difference pooled over the pairs of the three (agent pooled), with
    intervals from a bootstrap clustered by block. Holm runs over the pooled tests of the five layers of a model and tool surface. A
    difference over an incomplete or refused row is not computed (row_status), and the pooled test of a layer is not run when any row
    it pools is, entering the Holm at p = 1."""
    rows = []
    main = [e for e in episode_selection.agent_episodes(episodes) if episode_selection.at_main_levels(e)]
    for (model, surface), eps in cell_statistics.groups(main, "model", "surface").items():
        family = []
        for below, preset in zip(LADDER, LADDER[1:]):
            pooled, used, against, statuses = [], [], [], []
            for row in LADDER_ROWS:
                arm = episode_selection._arm(eps, row_label(row, model), row[1])
                level, reference = [e for e in arm if e.get("config") == preset], [e for e in arm if e.get("config") == below]
                pairs = cell_statistics.matched(level, reference, LADDER_KEY, "pass")
                if not pairs:
                    continue
                status = cell_statistics.row_status(level, ctx) or cell_statistics.row_status(reference, ctx)
                statuses.append(status)
                pooled += pairs
                used += level
                against += reference
                cells = {"model": model, "surface": surface, "config": preset, "reference_config": below, "agent": row_label(row, model), "status": status}
                rows.append(cell_statistics.trace(ctx, {**cells, **cell_statistics.paired_difference([] if status else pairs, "pass", ctx)}, level, reference))
            if pooled:
                status = next((s for s in statuses if s), "")
                family.append(cell_statistics.trace(ctx, {"model": model, "surface": surface, "config": preset, "reference_config": below, "agent": "pooled", "status": status,
                                          **cell_statistics.paired_difference([] if status else pooled, "pass", ctx), **family_correction.floor_cells(against, ctx)}, used, against))
        family_correction.holm_over(family, ("pass",), ctx["alpha"], family_correction.registered_size(ctx, "layers"))
        for row in family:  # the keys of the layer effects that main_tables reads
            row["holm_p"], row["reject"] = row.get("pass_holm_p", ""), row.get("pass_reject", "")
            row["reading"] = family_correction.two_sided_reading(row, "pass")
        rows += family
    return rows


def table_full_environment(episodes, ctx):
    """The family full_environment: for each row with the MCP the paired difference of pass between the full environment, its cell
    without training in the training table, and the base layer, its cell of the main table, matched on seed and every factor but the
    preset and the clock mode, two-sided, Holm over the registered size, the base cell as the reference of the floor."""
    main = [e for e in episode_selection.agent_episodes(episodes) if episode_selection.at_main_levels(e)]
    rows = []
    for (model, surface), eps in cell_statistics.groups(main, "model", "surface").items():
        family = []
        for name, row in FULL_ENVIRONMENT:
            arm = episode_selection._arm(eps, row_label(row, model), row[1])
            level, reference = [e for e in arm if e.get("config") == TRAINING_PRESET], [e for e in arm if e.get("config") == AGENTS_PRESET]
            if not (level and reference):
                continue
            cells = {"test": name, "agent": row_label(row, model), "agent_server": row[1], "model": model, "surface": surface,
                     "config": TRAINING_PRESET, "reference_config": AGENTS_PRESET, "status": cell_statistics.row_status(level, ctx) or cell_statistics.row_status(reference, ctx)}
            if not cells["status"]:
                cells.update(cell_statistics.paired_difference(cell_statistics.matched(level, reference, LADDER_KEY, "pass"), "pass", ctx))
            family.append(cell_statistics.trace(ctx, {**cells, **family_correction.floor_cells(reference, ctx)}, level, reference))
        for row in family_correction.holm_over(family, ("pass",), ctx["alpha"], family_correction.registered_size(ctx, "full_environment")):
            row["reading"] = family_correction.two_sided_reading(row, "pass")
            rows.append(row)
    return rows
