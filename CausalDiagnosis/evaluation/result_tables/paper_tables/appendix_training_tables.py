from CausalDiagnosis.experiments.table_definitions.experiment_tables import BUDGET_FIELDS, TRAINING_COMPARISONS, TRAINING_ROWS, VISIT_FIELDS, fold_ids, row_label
from . import table_cells, body_tables


# the head of the budget table in the order of design.BUDGET_FIELDS, each name in two parts, which the LaTeX form sets on two lines
BUDGET_NAMES = (("rounds", "sampled"), ("episodes", "sampled"), ("kept by", "the filter"), ("distinct training", "initial states"), ("model calls", "as examples"),
                ("GEPA", "candidates"))
# the heads of the table of the training episodes through a test combination, the names of design.VARIANT_NAMES split where the LaTeX
# form breaks them onto two lines, as the paper prints them
VISIT_HEADS = {"gepa": ("GEPA",), "gepa_score_only": ("GEPA on the", "score alone"), "ft": ("fine-tuning",), "ft_gepa": ("fine-tuning", "+ GEPA"),
               "ft_unfiltered": ("fine-tuning on", "unfiltered trajectories")}


def training_pairs_table(sheet):
    decided = []

    def cells(line, label, variant, reference):
        keys = {"model": sheet.model, "agent": label, "training_variant": variant, "reference_variant": reference}
        found = {subset: sheet.cite("trainhyp", line, "TRAINING_pairs", sheet.first("TRAINING_pairs", subset=subset, **keys)) for subset, _ in body_tables.SUBSETS}
        decided.append(found["all"])
        return [*(cell for _, metric in body_tables.MEASURES for cell in (table_cells.difference(found["all"], metric), table_cells.holm_cell(found["all"], f"{metric}_holm_p"))),
                *(table_cells.difference(found[subset], metric) for _, metric in body_tables.MEASURES for subset, _ in body_tables.SUBSETS[1:])]

    groups = [[sheet.counted("trainhyp", f"{name}, {comparison}", [name, comparison], cells(f"{name}, {comparison}", row_label((prefix, server), sheet.model), variant, reference))
               for variant, reference, comparison in TRAINING_COMPARISONS] for prefix, server, name in TRAINING_ROWS]
    # the head names the pairs the decision holds, all 235 episodes when the test set has run
    decision = " ".join(part for part in ("all", table_cells.counted(decided, "pairs"), "episodes") if part)
    tested = [part for name, _ in body_tables.MEASURES for part in (f"{table_cells.DELTA}{name}", "Holm p")]
    strata = [f"{table_cells.DELTA}{name} {head}" for name, _ in body_tables.MEASURES for _, head in body_tables.SUBSETS[1:]]
    return table_cells.MainTable("trainhyp", "The training hypotheses",
                     "Paired by test combination and entry offset over every test episode, each played by the model that holds out the fold of its fault "
                     "combination, with Holm over the 30 tests of this table, and the paired differences on the test combinations with a single fault and "
                     "with two faults beside "
                     "each test, without a test. The second comparison of each agent asks whether a gain of GEPA comes from the feedback of the verifier or only "
                     "from the reflection model, and the last whether a gain of fine-tuning comes from the signal of the environment or only from more exposure "
                     "to the task.",
                     ["Agent", "Comparison", *(f"{head}, {decision}" if head != "Holm p" else head for head in tested), *strata], "llcccccccc", groups,
                     tex_header=[rf" & & \multicolumn{{4}}{{c}}{{{decision}, the decision}} & \multicolumn{{4}}{{c}}{{strata, no test}} \\",
                                 r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
                                 "Agent & Comparison & " + " & ".join([*tested, *(table_cells.stacked(f"{table_cells.DELTA}{name}", head) for name, _ in body_tables.MEASURES for _, head in body_tables.SUBSETS[1:])]) + r" \\"],
                     first_only=True, stacked_first="t")


def folds_of(sheet):
    """The folds of the registered split of the asset of a sheet's training rows, as the rows of TRAINING_pairs and TRAINING_RECORD
    name them, or those of the default asset when no training row exists."""
    named = {int(str(r["subset"]).split("_", 1)[1]) for r in sheet.tables.get("TRAINING_pairs", []) if r.get("model") == sheet.model
             and str(r.get("subset", "")).startswith("fold_")}
    named |= {int(r["fold"]) for r in sheet.tables.get("TRAINING_RECORD", []) if r.get("model") == sheet.model and str(r.get("fold", "")).isdigit()}
    return sorted(named) or list(fold_ids())


def fold_groups(first, k):
    """The \\cmidrule of one column group per measure, each as wide as the k folds of the registered split, from column `first` on."""
    spans = [(first + i * k, first + (i + 1) * k - 1) for i in range(len(body_tables.MEASURES))]
    return "".join(rf"\cmidrule({'l' if i == len(spans) - 1 else 'lr'}){{{a}-{b}}}" for i, (a, b) in enumerate(spans))


def training_folds_table(sheet):
    folds = folds_of(sheet)

    def cells(line, label, variant, reference):
        keys = {"model": sheet.model, "agent": label, "training_variant": variant, "reference_variant": reference}
        found = {fold: sheet.cite("trainfolds", line, "TRAINING_pairs", sheet.first("TRAINING_pairs", subset=f"fold_{fold}", **keys)) for fold in folds}
        return [table_cells.difference(found[fold], metric) for _, metric in body_tables.MEASURES for fold in folds]

    groups = [[sheet.counted("trainfolds", f"{name}, {comparison}", [name, comparison], cells(f"{name}, {comparison}", row_label((prefix, server), sheet.model), variant, reference))
               for variant, reference, comparison in TRAINING_COMPARISONS] for prefix, server, name in TRAINING_ROWS]
    heads = [f"{table_cells.DELTA}{name} fold {fold}" for name, _ in body_tables.MEASURES for fold in folds]
    return table_cells.MainTable("trainfolds", "The training hypotheses by fold",
                     "The paired differences of the training hypotheses on each fold of the cross-fitting, without a test. Fold k holds the test episodes "
                     "whose fault combination lies in fold k, each played by the model that holds fold k out.",
                     ["Agent", "Comparison", *heads], "ll" + "c" * len(heads), groups,
                     tex_header=[" & & " + " & ".join(rf"\multicolumn{{{len(folds)}}}{{c}}{{{table_cells.DELTA}{name}}}" for name, _ in body_tables.MEASURES) + r" \\",
                                 fold_groups(3, len(folds)), "Agent & Comparison & " + " & ".join(f"fold {fold}" for _ in body_tables.MEASURES for fold in folds) + r" \\"],
                     first_only=True)


def record_groups(sheet, table_label, fields):
    """One group per row of design.TRAINING_ROWS with a line per fold of the cross-fitting, each cell a field of the line's row of
    TRAINING_RECORD, which tables_from_runs/ reads from run.json of the trained evaluation runs, so a line without a trained run says no
    record and a field that none of its trained runs carries says none."""
    def cells(line, label, fold):
        row = sheet.cite(table_label, line, "TRAINING_RECORD", sheet.first("TRAINING_RECORD", model=sheet.model, agent=label, fold=fold))
        return [table_cells.plain(str(row[field]) if row[field] != "" else table_cells.NO_VALUE) if row else table_cells.plain(table_cells.NO_RECORD) for field in fields]

    return [[sheet.counted(table_label, f"{name}, fold {fold}", [name, str(fold)], cells(f"{name}, fold {fold}", row_label((prefix, server), sheet.model), fold))
             for fold in folds_of(sheet)] for prefix, server, name in TRAINING_ROWS]


def budget_table(sheet):
    """What each agent trained on, one line per agent and fold of the cross-fitting."""
    return table_cells.MainTable("budget", "What each agent trained on",
                     "The same budget for all three, one line per fold of the cross-fitting, whose model holds that fold out and trained on the initial "
                     "states of the combinations whose faults all lie outside it, at entry offsets that no test episode plays, and a loop of GEPA plays "
                     "those initial states alone, past the seeds of the sampling. The rounds sampled, the episodes sampled and kept by the filter and the distinct training initial states "
                     "among the kept come from the gate of fine-tuning, which samples a second round when the first kept fewer episodes than the model has "
                     "training initial states and trains no adapter when both rounds together keep fewer.",
                     ["Agent", "Fold", *(" ".join(name) for name in BUDGET_NAMES)], "ll" + "c" * len(BUDGET_FIELDS), record_groups(sheet, "budget", BUDGET_FIELDS),
                     tex_header=["Agent & Fold & " + " & ".join(table_cells.stacked(*name) for name in BUDGET_NAMES) + r" \\"], first_only=True)


def hazard_table(sheet):
    """The training episodes of each trained variant whose hidden log passes through a combination that holds a single fault of the fold
    its model holds out, the one path from training to a fault on which its model is tested under repair-closed training, which the
    data and every loop leave out, per agent and fold."""
    return table_cells.MainTable("hazard", "Training episodes through a test combination",
                     "The training episodes of each trained variant whose hidden log passes through a combination that holds a single fault of the fold "
                     "its model holds out, a path that only the random hazard of deterioration opens, counted from the hidden records and left out of the "
                     "training data before the filter, the control and every loop of GEPA, one line per agent and fold: the sampled episodes for "
                     "fine-tuning and its control, fine-tuning on unfiltered trajectories, and the stream of its loop for each variant of GEPA.",
                     ["Agent", "Fold", *(" ".join(VISIT_HEADS[variant]) for variant in VISIT_FIELDS)], "ll" + "c" * len(VISIT_FIELDS),
                     record_groups(sheet, "hazard", list(VISIT_FIELDS.values())),
                     tex_header=["Agent & Fold & " + " & ".join(table_cells.stacked(*VISIT_HEADS[variant]) if len(VISIT_HEADS[variant]) > 1 else VISIT_HEADS[variant][0]
                                                                for variant in VISIT_FIELDS) + r" \\"], first_only=True)
