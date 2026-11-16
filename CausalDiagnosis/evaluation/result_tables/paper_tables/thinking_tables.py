from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.utils.csv_rows import as_float
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ABLATION_PLACES, AS_RELEASED, LADDER, LADDER_NAMES, LADDER_ROWS, TRAINING_PRESET, TRAINING_ROWS, TRAINING_VARIANTS, VARIANT_NAMES, row_label
from . import table_cells, appendix_layer_tables


# the offline table of the MCP as the paper prints it: the rows of offline_mcp.csv with the start's five recordings served before the live
# readings, one line per readings count, the row with no live reading among them, the evidence of the rows without a clock, and its
# columns as (head, field of the file), the count of exact wrong episodes among them.
# The file name, the readings counts and the five recordings, the segment of CausalDiagnosis.datasets.dataset_loader.evidence_source, copy those of the writer,
# CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.offline_table, so that this module stays clear of the server package, and a test holds them equal
# the columns of tab:thinking as (head lines, column of THINKING): the cycles charged and the seconds behind them with their split
THINKING_CELLS = ((("Cycles", "charged"), "thinking_cycles"), (("Seconds", "charged"), "seconds_charged"), (("Generation", "(s)"), "generation_seconds"),
                  (("Wall minus", "monotonic (s)"), "wall_minus_monotonic_seconds"), (("Rest", "(s)"), "rest_seconds"))


def plays_generation_time(preset):
    """Whether a preset plays in the mode that charges the wall seconds between tool calls as thinking time, the mode of every
    episode that tab:thinking reads; charges_thinking below is the engine's wider rule, which a token-clock twin also meets."""
    return aggregate.config_clock_mode(preset) == aggregate.GENERATION_TIME


def thinking_groups(sheet, cells_of):
    """The groups of tab:thinking from the design: the rungs of LADDER that charge thinking time with the rows of LADDER_ROWS, the
    preset of training with the rows of TRAINING_ROWS under each training variant, and the arm presets that charge it on the first layer
    that does, with the rows of their ablation place, each line from a function of (line labels, agent label, server, preset, variant).
    A group none of whose presets charges thinking time is left out."""
    def lines(rows, entries, variant=AS_RELEASED):
        return [cells_of([name, setting], row_label((prefix, server), sheet.model), server, preset, variant)
                for prefix, server, name in rows for preset, setting in entries]

    rungs = [(preset, setting) for preset, setting in zip(LADDER, LADDER_NAMES) if plays_generation_time(preset)]
    groups = [lines(LADDER_ROWS, rungs)]
    if plays_generation_time(TRAINING_PRESET):
        groups.append([line for prefix, server, name in TRAINING_ROWS for variant in TRAINING_VARIANTS
                       for line in lines([(prefix, server, name)], [(TRAINING_PRESET, VARIANT_NAMES[variant])], variant)])
    if rungs:
        place = f"L{LADDER.index(rungs[0][0])}"
        arms = [(arm, setting) for arm, (layer, setting) in appendix_layer_tables.ARM_PRESETS.items() if layer == place and plays_generation_time(arm)]
        groups.append(lines(ABLATION_PLACES.get(place, (None, ()))[1], arms))
    return [group for group in groups if group]


def thinking_outside(sheet):
    """The computed sentence that the body of the paper cites for tab:thinking: over the lines of the table with a split, the
    largest share of the seconds charged that falls outside the model's generation, and the line it comes from, as
    {"line", "share", "sentence"}, or None before any line has a split. The share is the mean seconds charged minus the mean
    generation over the mean seconds charged of one line."""
    best = None
    for line, row in sheet.thinking_rows:
        if not row:
            continue
        charged, generation = as_float(row.get("seconds_charged")), as_float(row.get("generation_seconds"))
        if charged and generation is not None and (best is None or (charged - generation) / charged > best[0]):
            best = ((charged - generation) / charged, line)
    if best is None:
        return None
    share, line = best
    return {"line": line, "share": round(share, 4),
            "sentence": f"Over the lines of `tab:thinking`, at most {100 * share:.1f}% of the seconds charged for thinking fall outside the "
                        f"model's generation, on the line {line}."}


def thinking_table(sheet):
    sheet.thinking_rows = []

    def cells_of(labels, label, server, preset, variant):
        line = ", ".join(labels)
        if variant == AS_RELEASED and preset != TRAINING_PRESET:
            row = sheet.read("thinking", line, "THINKING", label, server, preset)
        else:
            row = sheet.cite("thinking", line, "THINKING", sheet.first("THINKING", model=sheet.model, agent=label, diagnostics_server=server,
                                                                        config=preset, clock_mode=aggregate.GENERATION_TIME, training_variant=variant))
        sheet.thinking_rows.append((line, row))
        return sheet.counted("thinking", line, labels, [table_cells.marked(table_cells.estimate(row, metric, ".1f"), row) for _, metric in THINKING_CELLS] if row
                             else [table_cells.plain(table_cells.NO_RUN)] * len(THINKING_CELLS))

    return table_cells.MainTable("thinking", "Thinking time charged and where its seconds went",
                     "Every row played in the mode that charges the wall seconds between tool calls as thinking time, one line per agent and setting: "
                     "the cycles charged per episode, the seconds charged behind them, and of these the seconds of the model's generation, the sleep of "
                     "the machine or a step of the wall clock, measured as the wall gap minus the monotonic gap, which cannot see a stopped process since "
                     "both clocks run while a process is stopped, and the rest, serving, tools and the harness, each a mean over the episodes. A dagger "
                     "marks a line of a trained variant whose fold trained no adapter and reads the runs of no training there. The setting of a line with the CausalDiagnosis MCP is its training variant on the preset of training, and "
                     "a line of the last group plays an arm of the layer that charges thinking time.",
                     ["Agent", "Setting", *(" ".join(head) for head, _ in THINKING_CELLS)], "ll" + "c" * len(THINKING_CELLS),
                     thinking_groups(sheet, cells_of),
                     tex_header=["Agent & Setting & " + " & ".join(table_cells.stacked(*head) for head, _ in THINKING_CELLS) + r" \\"])


def charges_thinking(preset):
    """Whether the clock of a preset charges thinking time, by the engine's own rule for one cycle of thinking."""
    return real_time.thinking(aggregate.preset_config(preset), 1) > 0
