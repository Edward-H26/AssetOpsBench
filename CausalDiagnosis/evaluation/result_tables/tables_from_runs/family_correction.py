from fractions import Fraction
from CausalDiagnosis.utils.csv_rows import as_float  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import holm  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import cell_statistics


# the readings of a registered comparison, reversed when it holds in the other direction
READINGS = ("holds", "reversed", "does not hold", "not run", "uninformative")


def add_holm(rows, metric, alpha):
    """Holm over the p values of the primary metric of one table. A test that was not run, a row without a p value whose status says
    why (row_status), enters at p = 1, the registered counting.not_played; the other rows without a p value stay unmarked."""
    indexed = [(i, as_float(r.get(f"{metric}_p"))) for i, r in enumerate(rows)]
    tested = [(i, p) for i, p in indexed if p is not None]
    unrun = sum(1 for (_, p), r in zip(indexed, rows) if p is None and r.get("status"))
    corrected = holm([p for _, p in tested] + [1.0] * unrun, alpha) if tested or unrun else {"adjusted": [], "reject": []}
    for (i, _), adjusted, reject in zip(tested, corrected["adjusted"], corrected["reject"]):
        rows[i]["holm_p"] = _r(adjusted, 6)
        rows[i]["reject"] = reject
    return rows


def registered_size(ctx, family):
    """The registered number of tests of a family of the registered file."""
    return cell_statistics.registered_of(ctx)["families"][family]["size"]


def two_sided_reading(row, metric):
    """The reading of a two-sided registered test (READINGS without reversed): uninformative at the floor of its reference, not run
    without a decision, holds when Holm rejects, whose direction the sign of the difference gives (<metric>_sign), and does not
    hold otherwise."""
    difference = as_float(row.get(f"{metric}_diff"))
    row[f"{metric}_sign"] = "" if difference is None else "below zero" if difference < 0 else "above zero" if difference > 0 else "zero"
    if row.get("at_floor") is True:
        return "uninformative"
    decision = row.get(f"{metric}_reject")
    if decision in (None, "") or difference is None:
        return "not run"
    return "holds" if decision else "does not hold"


def holm_joint(parts, alpha, size=None):
    """Holm over the tests of several (rows, measures) parts as one family: every tested measure gets <measure>_holm_p and
    <measure>_reject, and every row the number of tests the family counted (family_tests). With a registered size the family has
    that many tests whatever is present, a test that was not run entering with a p value of one, so a missing row never makes the
    correction milder; a family with more tests than its size is refused. Parts let one family hold rows of different measures, as
    a group of primary that joins two families does (corrected_family)."""
    tested = [(row, m, as_float(row.get(f"{m}_p"))) for rows, measures in parts for row in rows for m in measures]
    tested = [(row, m, p) for row, m, p in tested if p is not None]
    if size is not None and len(tested) > size:
        raise ValueError(f"{len(tested)} tests in a family registered with {size}")
    padding = [1.0] * ((size or 0) - len(tested))
    corrected = holm([p for _, _, p in tested] + padding, alpha) if tested or padding else {"adjusted": [], "reject": []}
    for (row, m, _), adjusted, reject in zip(tested, corrected["adjusted"], corrected["reject"]):
        row[f"{m}_holm_p"], row[f"{m}_reject"] = _r(adjusted, 6), reject
    for rows, _ in parts:
        for row in rows:
            row["family_tests"] = len(tested) if size is None else size
    return parts


def holm_over(rows, measures, alpha, size=None):
    """Holm over the tests of measures on rows as one family, holm_joint with one part."""
    holm_joint([(rows, measures)], alpha, size)
    return rows


def reading(row, measures, signs=None):
    """The reading of a registered comparison (READINGS): uninformative when its reference is at the floor, not run while a test is
    missing, holds when every difference lies on the registered side of its measure (signs, +1 above zero by default, -1 below) and
    every Holm decision rejects, reversed when every difference lies on the other side and every decision rejects, and does not hold
    otherwise."""
    if row.get("at_floor") is True:
        return "uninformative"
    differences = [as_float(row.get(f"{m}_diff")) for m in measures]
    confirmed = [row.get(f"{m}_reject") for m in measures]
    if None in differences or any(c in (None, "") for c in confirmed):
        return "not run"
    sides = [d * (signs or {}).get(m, 1) for d, m in zip(differences, measures)]
    if all(confirmed) and all(s > 0 for s in sides):
        return "holds"
    if all(confirmed) and all(s < 0 for s in sides):
        return "reversed"
    return "does not hold"


def untested(row):
    """A row of a read below the fraction of record: no registered test and no Holm decision, so its p values, decisions and the
    counts of its family stay empty and its reading says not run."""
    for key in list(row):
        if key.endswith(("_p", "_holm_p", "_reject")) or key in ("family_tests", "holds", "reversed"):
            row[key] = ""
    if "reading" in row:
        row["reading"] = "not run"
    return row


def registered_floor(ctx):
    """The floor of pass of the registered file (descriptive.floor_of_pass) as an exact fraction; an open floor stops the caller."""
    floor = (cell_statistics.registered_of(ctx or {}).get("descriptive") or {}).get("floor_of_pass")
    if floor is None:
        raise registered.OpenValue("the registered value descriptive.floor_of_pass is not set yet, and the floor of every table reads it")
    return Fraction(str(floor))


def floor_cells(reference, ctx=None):
    """The floor of a comparison: the reference's verified passes k of its n episodes and whether k / n lies below the registered
    floor of pass, as exact fractions, where the comparison is reported as uninformative. Empty without an episode."""
    passes = [cell_statistics.value(e, "pass") for e in cell_statistics.with_metric(reference, "pass")]
    k, n = int(sum(passes)), len(passes)
    return {"reference_pass_k": k, "reference_pass_n": n, "at_floor": Fraction(k, n) < registered_floor(ctx) if n else ""}
