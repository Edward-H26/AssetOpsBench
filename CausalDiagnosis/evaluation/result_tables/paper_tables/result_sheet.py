import re
from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.utils.csv_rows import as_bool, as_float
from CausalDiagnosis.experiments.table_definitions.experiment_tables import LADDER, LADDER_NAMES, LADDER_ROWS, AGENTS_PRESET, AGENT_LINES, AGENT_ROWS, row_label
from . import table_cells


PAPER_PROTOCOL = aggregate.PAPER_PROTOCOL
STATUS_PARTS = (re.compile(r"partial (?P<fraction>\d+(?:\.\d+)?)"),
                re.compile(r"incomplete, (?P<played>\d+) (?:of (?P<planned>\d+)|played)(?:, replicates (?P<replicates>[^;]+))?"))


def status_parts(status):
    """{"fraction": f} and {"played", "planned", "replicates"} of a status that aggregate.row_status writes in the form of
    aggregate.counting_text, its parts joined by "; ", or None for any other status, such as an H0 test unrun for want of a judge
    record, which is no state of the counting rule."""
    found = {}
    for part in str(status).split("; ") if status else []:
        match = next((m for m in (pattern.fullmatch(part) for pattern in STATUS_PARTS) if m), None)
        if match is None:
            return None
        found.update({key: value for key, value in match.groupdict().items() if value is not None})
    return found or None


def row_read(row):
    """The fraction of a read below the fraction of record that a row holds, from its read cell (aggregate.COUNTS) or its status,
    or empty for a row at the read of record."""
    read = str(row.get("read", ""))
    if read:
        return read.removeprefix(f"{table_cells.PARTIAL} ")
    return (status_parts(row.get("status", "")) or {}).get("fraction", "")


def row_state(row):
    """("refused", None), ("incomplete", (played, planned or None, replicates held)), ("partial", fraction) or None: the state of a
    row under the registered counting rule, refused before incomplete before partial, from its counts where its table carries
    them (aggregate.COUNTS) and else from its status as aggregate.row_status writes it. planned is None when a missing unit has
    no run to count its planned episodes from."""
    if "refused" in row and "incomplete" in row:
        if as_bool(row["refused"]):
            return (table_cells.REFUSED, None)
        if as_bool(row["incomplete"]):
            planned = str(row.get("planned", ""))
            return (table_cells.INCOMPLETE, (int(row["played"]), int(planned) if planned else None, str(row.get("replicates", ""))))
        return (table_cells.PARTIAL, row_read(row)) if row_read(row) else None
    status = str(row.get("status", ""))
    if status == table_cells.REFUSED:
        return (table_cells.REFUSED, None)
    parts = status_parts(status)
    if not parts:
        return None
    if "played" in parts:
        return (table_cells.INCOMPLETE, (int(parts["played"]), int(parts["planned"]) if "planned" in parts else None, parts.get("replicates", "")))
    return (table_cells.PARTIAL, parts["fraction"])


def shortness(counts):
    """The order of the counts of incomplete rows, the shortest first: a row whose planned episodes are not known, then the smallest
    share of its planned episodes played."""
    played, planned, _ = counts
    return (planned is not None, played / planned if planned else 0.0)


def episode_counts(row):
    """(run, planned) of a row that carries its counts, each None where it is empty, or None for a row without counts."""
    if str(row.get("played", "")) == "":
        return None
    return int(row["played"]), int(row["planned"]) if str(row.get("planned", "")) != "" else None


def unscored_share(row):
    """The share of the played episodes of a row that the verifier did not score, or None where its table carries no counts."""
    played = as_float(row.get("played"))
    return as_float(row.get("unscored")) / played if played and as_float(row.get("unscored")) is not None else None


class Sheet:
    """The aggregate tables as read for one model, and the rows that its cells cite, which the results index lists."""

    def __init__(self, tables, model):
        self.tables, self.model, self.cited, self.judged = tables, model, [], {}
        # table label -> [(line, row)] cited, the lines counted, and the lines marked refused or incomplete in the order printed
        self.cited_rows, self.counted_lines, self.marked = {}, {}, {}
        # the fraction of a read below the fraction of record: one run of aggregate reads every table at one fraction and writes it
        # in the read cell of every row of AGENTS, so every line and every caption of the page carry it
        self.partial_read = next((read for read in map(row_read, tables.get("AGENTS", [])) if read), "")

    def find_all(self, table, label, server, preset=AGENTS_PRESET, clock=None, **cells):
        """The rows of a table for one of the six rows on a preset, its own clock mode unless clock names another, at the main level
        of every other setting with the agent as released."""
        mode = clock or aggregate.config_clock_mode(preset)
        return [row for row in self.tables[table] if row["agent"] == label and as_bool(row["diagnostics_server"]) is server and row["config"] == preset
                and row["clock_mode"] == mode and aggregate.at_main_settings(row) and all(row[k] == v for k, v in cells.items())]

    def find(self, table, label, server, preset=AGENTS_PRESET, clock=None, **cells):
        """The first row of find_all, or None."""
        return next(iter(self.find_all(table, label, server, preset, clock, **cells)), None)

    def first(self, table, **cells):
        """The first row of a table whose cells equal the given ones, for the tables whose rows hold no setting of a run, or None."""
        return next((row for row in self.tables[table] if all(str(row.get(k, "")) == str(v) for k, v in cells.items())), None)

    def cite(self, table_label, line, table, row):
        if row:
            self.cited.append({"model": self.model, "table": table_label, "line": line, "csv": f"{table}.csv", "row": row["row_number"]})
            self.cited_rows.setdefault(table_label, []).append((line, row))
        return row

    def counted(self, table_label, line, labels, cells):
        """The labels and cells of a line under the registered counting rule (counting.max_unscored_share_rule, counting.not_played
        and counting.fraction_of_record): a line one of whose rows is refused prints refused in every cell, a line with a row that
        misses planned episodes keeps its cells and marks its last label with the episodes the shortest row played and planned and
        the replicates it holds, every line of a read below the fraction of record marks its last label partial with the fraction,
        and any other line stays as it is. The list of marked lines, which the captions and the table of counts read, names a line by
        its name. The rows of a line are those cited under its name or under its name and a comma, the form of a line that reads
        one row per subset, stratum or policy. A line read twice, the lines of one row of a vertical table, is marked once, at its
        first line."""
        rows = [row for key, row in self.cited_rows.get(table_label, []) if key == line or key.startswith(f"{line}, ")]
        states = [(row, state) for row, state in zip(rows, map(row_state, rows)) if state]
        first = line not in self.counted_lines.setdefault(table_label, [])
        self.counted_lines[table_label].append(line)
        refused = [row for row, state in states if state[0] == table_cells.REFUSED]
        cells = [table_cells.plain(table_cells.REFUSED)] * len(cells) if refused else cells
        if not first:
            return labels, cells
        short = [(row, state[1]) for row, state in states if state[0] == table_cells.INCOMPLETE]
        if refused:
            shares = [share for share in map(unscored_share, refused) if share is not None]
            self.marked.setdefault(table_label, []).append((line, table_cells.REFUSED, episode_counts(refused[0]), max(shares) if shares else None, ""))
        elif short:
            row, (played, planned, held) = min(short, key=lambda pair: shortness(pair[1]))
            self.marked.setdefault(table_label, []).append((line, table_cells.INCOMPLETE, (played, planned), unscored_share(row), held))
            return [*labels[:-1], table_cells.Marked(labels[-1], played, planned, held, self.partial_read)], cells
        return ([*labels[:-1], table_cells.Marked(labels[-1], read=self.partial_read)] if self.partial_read else labels), cells

    def uncounted(self):
        """The cited lines of every table that no counted line covers, or that two counted lines cover, as (table label, line):
        empty when every row a table reads reaches the counting rule through the line it fills."""
        out = []
        for table_label, cited in self.cited_rows.items():
            lines = set(self.counted_lines.get(table_label, []))
            for key in dict.fromkeys(key for key, _ in cited):
                covering = [line for line in lines if key == line or key.startswith(f"{line}, ")]
                if len(covering) != 1:
                    out.append((table_label, key))
        return out

    def read(self, table_label, line, table, *key, **cells):
        """find, with the row recorded as the source of a line of a table."""
        return self.cite(table_label, line, table, self.find(table, *key, **cells))

    def hypothesis(self, table_label, line, name):
        """The row of AGENTS_pairs of a hypothesis on the preset of the six rows at the main level of every setting, cited."""
        mode = aggregate.config_clock_mode(AGENTS_PRESET)
        row = next((r for r in self.tables["AGENTS_pairs"] if r["hypothesis"] == name and r["model"] == self.model and r["config"] == AGENTS_PRESET
                    and r["clock_mode"] == mode and aggregate.at_main_settings(r)), None)
        return self.cite(table_label, line, "AGENTS_pairs", row)

    def judge_cells(self, table_label, line, shown, label, server, preset=AGENTS_PRESET):
        """The three scores of the judge of one line under the five call protocol. Episodes judged under another protocol
        fill no cell, which says so, and such a row enters no count of the sentence above the tables, and neither does a
        line that two judge models scored, whose cells say so."""
        if self.find("AGENTS", label, server, preset) is None:
            return [table_cells.plain(table_cells.NO_RUN)] * len(table_cells.JUDGED)
        found = self.find_all("JUDGE", label, server, preset, judge_protocol=PAPER_PROTOCOL)
        if len(found) > 1:
            return [table_cells.plain(table_cells.MIXED_JUDGES)] * len(table_cells.JUDGED)
        row = self.cite(table_label, line, "JUDGE", found[0] if found else None)
        if row is None:
            other = self.find("JUDGE", label, server, preset)
            return [table_cells.plain(f"protocol {other['judge_protocol']}" if other else table_cells.NO_JUDGE)] * len(table_cells.JUDGED)
        cells = [shown(row, metric) if row[f"{metric}_n"] == row["n"] else table_cells.plain(table_cells.MIXED) for _, metric in table_cells.JUDGED]
        if table_cells.plain(table_cells.MIXED) not in cells:
            self.judged.setdefault(table_label, []).append((line, row))
        return cells

    def judge_counts(self):
        """One sentence per table of the judge with a judged line on what counted as false, which the page states above its tables,
        with the judge calls without a parsed review and the declaring episodes of every row, so that a False of an
        outage of the judge can be told from a judged failure."""
        sentences = []
        for table_label, lines in self.judged.items():
            rows = [row for _, row in lines]
            failed, missing = (sum(int(row[column]) for row in rows) for column in ("failed_calls", "episodes_without_record"))
            per_row = ", ".join(f"{line} {row['failed_calls']} and {row['declared']}" for line, row in lines)
            sentences.append(f"In `tab:{table_label}`, {failed} judge {'call' if failed == 1 else 'calls'} without a parsed review and {missing} "
                             f"{'episode' if missing == 1 else 'episodes'} without a judge record count as false on every criterion. Per row, the judge "
                             f"calls without a parsed review and the declaring episodes, which the judge scores as failures: {per_row}.")
        return sentences

    def played(self, rows):
        """The model and the episodes per line of the rows that a table read, as its caption names them."""
        episodes = table_cells.counted(rows)
        return "".join(f", {part}" for part in (self.model, f"{episodes} episodes per line" if episodes else "") if part)

    def exact_baseline(self):
        """The majority-class baseline of exact match for a caption, the share of the episodes on the most frequent faulty set of the
        starts played: its value when the rows of the agents table that ran share it, as rows on the same starts do, else, and before
        any run, a pointer to the appendix table of exact match, which prints it per row."""
        rows = [self.find("AGENTS", row_label(row, self.model), row[1]) for row in AGENT_ROWS]
        values = {(row["faulty_set_right_baseline"], row["faulty_set_right_baseline_k"], row["faulty_set_right_baseline_n"]) for row in rows
                  if row and str(row.get("faulty_set_right_baseline", "")) != ""}
        if len(values) != 1:
            return "the majority-class baseline that the appendix table of exact match prints per row"
        baseline, k, n = values.pop()
        return f"a majority-class baseline of {as_float(baseline):.2f} ({k} of {n} episodes)"

    def agent_lines(self, table_label, cells_of):
        """One group per design of design.AGENT_ROWS with its lines of design.AGENT_LINES, the design named on its first line and the
        others under it, each from a function of (line, agent label, server level, preset)."""
        def entry(prefix, name, suffix, server, preset):
            line = " ".join(part for part in (name, suffix) if part)
            return self.counted(table_label, line, [suffix or name], cells_of(line, row_label((prefix, server), self.model), server, preset))
        return [[entry(prefix, name, suffix, server, preset) for suffix, server, preset in AGENT_LINES] for prefix, with_mcp, name in AGENT_ROWS if not with_mcp]

    def six(self, table_label, cells_of):
        """One group with a line per row of design.AGENT_ROWS, from a function of (line, agent label, server level)."""
        return [[self.counted(table_label, name, [name], cells_of(name, row_label((prefix, server), self.model), server)) for prefix, server, name in AGENT_ROWS]]

    def ladder(self, table_label, cells_of):
        """One group per row of design.LADDER_ROWS with a line per preset of design.LADDER, from a function of (line, agent label,
        server level, preset), the layout of Table 16 of AssetOpsBench."""
        return [[self.counted(table_label, f"{name}, {setting}", [name, setting], cells_of(f"{name}, {setting}", row_label((prefix, server), self.model), server, preset))
                 for preset, setting in zip(LADDER, LADDER_NAMES)] for prefix, server, name in LADDER_ROWS]
