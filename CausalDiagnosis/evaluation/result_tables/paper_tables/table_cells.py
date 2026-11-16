from typing import NamedTuple
from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.utils.csv_rows import as_bool, as_float


NO_RUN, NO_JUDGE, MIXED, NO_VALUE, NO_RECORD = "no run", "no judge", "mixed protocols", "none", "no record"
MIXED_JUDGES = "mixed judge models"  # the judge cells of a line that two judge models scored, which only a development root holds
PLUS_MINUS, DELTA, DAGGER, KAPPA = "\N{PLUS-MINUS SIGN}", "\N{GREEK CAPITAL LETTER DELTA}", "\N{DAGGER}", "\N{GREEK SMALL LETTER KAPPA}"
SHORT = "\N{DOUBLE DAGGER}"  # the LaTeX mark of a line that misses planned episodes (counting.not_played)
REFUSED = "refused"  # every cell of a refused line (counting.max_unscored_share_rule)
INCOMPLETE, PARTIAL = "incomplete", "partial"  # a line that misses planned episodes, a read below the fraction of record
DASH = "\N{MINUS SIGN}"  # the cell of a measure that its line cannot have, a measure of the layers on the base layer
JUDGED = (("TC", "task_completion"), ("DRA", "data_retrieval_accuracy"), ("RV", "generalized_result_verification"))
PERCENT_UNIT = " (%)"  # the page heads each column of percentages with its unit, the LaTeX form heads the judge columns together under JUDGE_GROUP
JUDGE_GROUP = r"Judge metrics (\%)"
# the registered counting rule of an unjudged episode (counting.unjudged), which every caption with judge cells states
UNJUDGED_RULE = "An episode without a judge record counts as false on every criterion of the judge."
# the two outcome shares that every table prints as a percentage wherever a level of them stands, the form of the tables of the agents
SHARE_METRICS = ("diagnosis_all_correct", "pass")


class Cell(NamedTuple):
    short: str  # what the LaTeX form prints
    full: str  # what the page prints


class Marked(str):
    """The last label of a line under a state of the counting rule: the label itself, with the fraction of a partial read and, for a
    line that misses planned episodes, the episodes its shortest row played and planned and the replicates it holds, which the page
    prints after it in the form of aggregate.counting_text. The LaTeX form marks a short line with SHORT and states a partial read
    in the caption, since every line of such a table reads it."""

    def __new__(cls, label, played=None, planned=None, replicates="", read=""):
        marked = super().__new__(cls, label)
        marked.played, marked.planned, marked.replicates, marked.read = played, planned, replicates, read
        return marked

    @property
    def short(self):
        return self.played is not None

    def page(self):
        return f"{self} ({aggregate.counting_text(f'{PARTIAL} {self.read}' if self.read else '', self.played, self.planned, self.replicates)})"


def page_label(label):
    return label.page() if isinstance(label, Marked) else label


class MainTable(NamedTuple):
    label: str  # the label of the table in the paper, tab:<label>
    title: str
    caption: str
    header: list  # the header cells of the page, and of the LaTeX form unless tex_header names its own lines
    tex_format: str
    groups: list  # [[(line labels, [Cell])]], a rule between groups
    tex_header: list = None
    first_only: bool = False  # the first label of a group names it on its first line alone, as the paper prints the ladder
    stacked_first: str = ""  # that name splits before its MCP onto two lines aligned on its top line (t), as the paper prints the
    # training hypotheses, or on its centre (c), beside the two lines of the cells of the main table
    indented: bool = False  # the lines of a group named "+ ..." are indented under the agent, the variants of the training table
    spaced: bool = False  # the groups part by a small space instead of a rule, as the paper prints the designs of the agents tables
    counting: tuple = ()  # (sentence of the page, sentence of the LaTeX form) on its refused and incomplete lines, empty without one


class Column(NamedTuple):
    head: tuple  # the head, one line or the two lines that the LaTeX form stacks
    metric: str  # its column of AGENTS
    spec: str
    meaning: str  # what the caption says the column holds
    scale: int = 1  # 100 for a share printed as a percentage
    layers: bool = False  # a measure that exists only with the layers on, so a line on the base layer prints DASH
    group: str = "Verified outcome"  # the head over neighbouring columns of the same group in the LaTeX form


def plain(text):
    return Cell(text, text)


def estimate(row, metric, spec, scale=1):
    """The estimate of a metric alone for the LaTeX form and with its interval for the page, or the word for no value."""
    if as_float(row.get(metric)) is None:
        return plain(NO_VALUE)

    def shown(column):
        return f"{as_float(row[column]) * scale:{spec}}"

    # below the registered smallest number of clusters the interval is empty and the estimate prints alone
    full = shown(metric) if as_float(row.get(f"{metric}_low")) is None else f"{shown(metric)} [{shown(f'{metric}_low')}, {shown(f'{metric}_high')}]"
    return Cell(shown(metric), full)


def with_interval(row, metric, spec):
    """The estimate with its interval in both forms."""
    return plain(estimate(row, metric, spec).full)


def mean_std(row, metric, spec, scale=1):
    """The mean and the standard deviation over episodes, as Tables 11 and 12 of AssetOpsBench print them, in both forms."""
    if as_float(row.get(metric)) is None:
        return plain(NO_VALUE)
    mean = f"{as_float(row[metric]) * scale:{spec}}"
    spread = as_float(row.get(f"{metric}_std"))
    return plain(mean if spread is None else f"{mean} {PLUS_MINUS} {spread * scale:{spec}}")


def mean_and_error(row, metric, spec, scale=1):
    """The mean of a metric of a row and its standard error from the bootstrap that resamples test combinations, both as printed, the
    error None where it is empty, below the registered smallest number of clusters, or None for a row without the metric."""
    if as_float(row.get(metric)) is None:
        return None
    error = as_float(row.get(f"{metric}_se"))
    return f"{as_float(row[metric]) * scale:{spec}}", None if error is None else f"{error * scale:{spec.lstrip('+')}}"


def over_se(row, metric, spec, scale=1):
    """The cell of the result tables but the two tables of the agents: the mean over the episodes and, after the plus-minus sign, its
    standard error, which the LaTeX form sets under the mean in smaller type. The mean prints alone where the standard error is empty,
    and a row without the metric prints the word for no value."""
    printed = mean_and_error(row, metric, spec, scale)
    if printed is None:
        return plain(NO_VALUE)
    mean, error = printed
    return plain(mean) if error is None else Cell(rf"\st{{{mean}}}{{{PLUS_MINUS}\,{error}}}", f"{mean} {PLUS_MINUS} {error}")


def one_line_se(row, metric, spec, scale=1):
    """The cell of the two tables of the agents: the mean and, on the same line after the plus-minus sign in smaller type, its standard
    error, the mean alone where the standard error is empty, and the word for no value for a row without the metric."""
    printed = mean_and_error(row, metric, spec, scale)
    if printed is None:
        return plain(NO_VALUE)
    mean, error = printed
    return plain(mean) if error is None else Cell(rf"{mean}\,{{\scriptsize{PLUS_MINUS}\,{error}}}", f"{mean} {PLUS_MINUS} {error}")


def judge_over_se(row, metric):
    return over_se(row, metric, ".1f", 100)


def level_form(metric, spec=".2f"):
    """(format, scale) of a level: a share of SHARE_METRICS as a percentage to one decimal, any other measure in spec."""
    return (".1f", 100) if metric in SHARE_METRICS else (spec, 1)


def level_over_se(row, metric, spec=".2f"):
    """A level with its standard error in the form of over_se, a share of SHARE_METRICS as a percentage."""
    return over_se(row, metric, *level_form(metric, spec))


def level_head(name, metric):
    """The head of a level on the page, with the unit of a share of SHARE_METRICS."""
    return f"{name}{PERCENT_UNIT}" if metric in SHARE_METRICS else name


def percent_one_line(row, metric):
    return one_line_se(row, metric, ".1f", 100)


def kappa_cell(row, form=estimate):
    """Cohen's kappa of the judge's majority pass against the verified pass in the form of its table, the word for no judge when no
    episode of the row was judged under the five call protocol, and the word for no value when one rule puts every judged episode on
    one side."""
    if row is None:
        return plain(NO_RUN)
    if not as_float(row.get("judge_kappa_n")):
        return plain(NO_JUDGE)
    return form(row, "judge_kappa", ".2f")


def difference(row, metric):
    """A paired difference with its interval in both forms, signed, or the word for no run or no value."""
    if row is None:
        return plain(NO_RUN)
    return with_interval(row, f"{metric}_diff", "+.2f") if as_float(row.get(f"{metric}_diff")) is not None else plain(NO_VALUE)


def holm_cell(row, column):
    """A p value, Holm adjusted or not, to three digits, with a bound below the smallest it prints, or the word for no run or no value."""
    if row is None:
        return plain(NO_RUN)
    p = as_float(row.get(column))
    return plain(NO_VALUE if p is None else "< 0.001" if p < 0.001 else f"{p:.3f}")


def signed(row, metric):
    """A signed estimate with its interval in both forms, or the word for no run or no value."""
    if row is None:
        return plain(NO_RUN)
    return with_interval(row, metric, "+.2f") if as_float(row.get(metric)) is not None else plain(NO_VALUE)


def number(row, column, spec=".2f"):
    """A plain number of a row, or the word for no run or no value."""
    if row is None:
        return plain(NO_RUN)
    return plain(NO_VALUE if as_float(row.get(column)) is None else f"{as_float(row[column]):{spec}}")


def flag(row, column):
    """A yes or no cell of a row, or the word for no run or no value."""
    if row is None:
        return plain(NO_RUN)
    value = as_bool(row.get(column))
    return plain(NO_VALUE if value is None else "yes" if value else "no")


def word(row, column):
    """A text cell of a row, or the word for no run or no value."""
    if row is None:
        return plain(NO_RUN)
    return plain(str(row.get(column, "")) or NO_VALUE)


def pairs_clause(rows):
    """The pairs and the start combinations behind the paired differences of a table, for its caption, empty before any run."""
    pairs, clusters = counted(rows, "pairs"), counted(rows, "clusters")
    return f", over {pairs} pairs in {clusters} test combinations" if pairs else ""


def counted(rows, column="n"):
    """The episodes or pairs that the rows hold as one number, or as a range when they differ, and empty without a row."""
    counts = sorted({int(row[column]) for row in rows if row and str(row.get(column, "")) != ""})
    return "" if not counts else str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"


def stacked(top, bottom, align="c", vertical=None):
    """A cell of the LaTeX form on two lines, as the paper sets the heads of its narrow columns, aligned on its centre, its top line or
    its bottom line (vertical c, t or b), by default the centre for a centred cell and the top line for a left one."""
    vertical = vertical or ("c" if align == "c" else "t")
    return rf"\begin{{tabular}}[{vertical}]{{@{{}}{align}@{{}}}}{top}\\{bottom}\end{{tabular}}"


def judge_heads():
    """The heads of the three judge columns on the page, each with its unit."""
    return [f"{name}{PERCENT_UNIT}" for name, _ in JUDGED]


def column_groups(columns, first=2):
    """The runs of neighbouring columns of one group, as [group, first column, last column] from column `first` of the tabular on."""
    runs = []
    for position, column in enumerate(columns, first):
        if runs and runs[-1][0] == column.group:
            runs[-1][2] = position
        else:
            runs.append([column.group, position, position])
    return runs


def listed(items):
    """Names joined as a sentence lists them, with and before the last."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def in_words(n):
    """A count as the prose of a caption writes it, in words below ten and in digits from ten on."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def marked(cell, row):
    """A cell of the training table whose episodes include a fold that trained no adapter and reads the runs of no training."""
    return Cell(f"{cell.short}{DAGGER}", f"{cell.full}{DAGGER}") if as_float(row.get("aliased")) else cell
