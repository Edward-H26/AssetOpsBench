"""The ranking module, of the advice kind ranking (CausalDiagnosis/docs/general-agent-plan.md, section Advice modules): everything
that reads a ranking. It asks the ranking before a turn when the observations, the closed repairs or the clock moved, before a
commit write when it is stale, once after a batch, and for a standing step only when it is stale; its candidates take the first
ranks; it judges top_rank, top_mass_at_least and effective_at_most on an observe step and top_rank on a commit write with candidate
arguments, by default 1 for a commit by a ranked candidate; it prints the ranking, the advice, the lines of advice and the
candidate plans; its rules refuse a top_rank outside the ranked candidates and a commit by a ranked candidate before any ranking;
it owns the reserve for the repair and the advice lines.

The module never reads a tool name or writes an argument name of its own: it asks for the tool of its kind through the context,
whose key call maps its inputs, and every key of a row that it reads in an answer comes from the table. The series of a read is
the argument that the key series of the observe row names, its count the argument that the key minimum of that row names, a
faulty part the argument that the key target of the repair write names, and an answer the candidate arguments of the first commit
write that holds any. A word that a text prints comes from the terms of the vocabulary.

The module keeps the refuted sets that the loop hands it by record_refuted, and keeps the last of them out of the advice until a
ranking after a further observation names it first."""
import json
import re

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options

SLOT = re.compile(r"\{([a-z_]+)\}")
KIND = "ranking"
DETAIL_INPUT = "detail"  # the input of the module that the key call of the ranking row maps to an argument
COMPACT, SUMMARY = "compact", "summary"  # the form of the ranking: compact where the reads follow the advice, else the long form
READ_ARMS = {"gain": "the read batch by information gain per unit cost",
             "separating": "the separating test of the ranking",
             "ec2": "the one step edge cutting read over repair plan classes"}  # the advice of each value of the switch reads
FALLBACK_WORDS = {"time_unit": "unit", "series": "series", "target": "part", "repair": "repair"}

# the block of the system prompt, one sentence per item, so that a switch leaves out its sentence and edits none
BLOCK = (
    "Advice: the ranking.",
    "Before a planning turn in which the evidence or the clock moved, the controller asks {ranking_tool} and shows the answer as advice.",
    "The ranking lists the candidate answers now with their posterior mass, the effective number of hypotheses of the belief, and the "
    "decision fields of the server's rule: read_first, the separating test with its price and expected loss, and the expected loss of "
    "acting now.",
    "Per {series} it gives the information gain of one {time_unit} and that gain per unit cost, which ranks the reads, the server's "
    "read batch, and the lines of advice: the reads to take while the ranking says read first, the repair plan to act on and, after a "
    "refuted plan, the other plan.",
    "The table of candidate plans, when present, walks each ranked candidate forward: its confirm test, its repair plan with the warrant "
    "that every repair write of the plan cites, the {time_unit}s to a restored {short_noun}, the predicted recovery if acted, and the "
    "expected loss and the cost both ways.",
    "The ranked candidates take the first ranks and yours follow them.",
    "It is advice: you decide every step, and the controller makes only those your plan names.",
    "You never write a probability of your own: every number of the ranking comes from the board.",
    "The advised reads are the read the server advises.",
    "The advised repair plan is the rank the server advises to commit, with one repair write for each of its faulty {target}s.",
    'A step of the role observe can expect {"top_rank": <rank in the current ranking>, "top_mass_at_least": <mass>}: after the '
    "observation, the set of faulty {target}s of the candidate at that rank should be the set on top of the new ranking, with at least "
    "that mass.",
    'A step of {commit_tool} can expect {"top_rank": <the rank of the candidate on top when the write is made>}.',
    "The schedule on the board, when one is shown, lists the reads that the server names and the reads that several candidate plans "
    "share, each once with the plans it serves, and after a refuted plan what each remaining plan still needs.",
)
BRANCHES_SENTENCE, READS_SENTENCE, COMMIT_SENTENCE, SCHEDULE_SENTENCE = 4, 8, 11, 12  # the items that the switches branches, reads, the commit write of the table and the switch parallel own
READS_ARM = "The advised reads of this run are {arm} and not the read the server advises."
BOARD_SENTENCE = "It also shows the advice: the latest ranking, the lines of advice and, when present, the candidate plans."

# the shapes of the forms, as the line of a tool names them
SHAPE_TOP_RANK = "rank in the current ranking, at least 1"
SHAPE_TOP_RANK_COMMIT = "the rank of the candidate on top when the write is made, at least 1"
SHAPE_MASS = "mass from 0 to 1"
SHAPE_EFFECTIVE = "number of hypotheses, at least 1"

# the reasons of the rules of the module
REASON_RANK = "expect.top_rank must be one of the {n} ranked candidates, and {rank} is not"
REASON_NO_RANKING = "{tool} commits by a ranked candidate, and no ranking stands yet, so expect no top_rank on it or plan it after a ranking"

# the lines of the board
HEADING_RANKING, HEADING_ADVICE, HEADING_PLANS = "## Ranking", "## Advice", "## Candidate plans"
NO_RANKING = "No ranking yet."
FAILED_RANKING = "The ranking could not be read ({count} times); plan on the observations alone."
STALE = " (stale: evidence arrived after this ranking)"
RANKING_LINE = "{rank}. {answer}: mass {mass}{faults}"
FAULTS = ", faulty {targets}"
EFFECTIVE_LINE = "Effective hypotheses: {value}."
DECISION_LINE = "read_first {read_first}; expected loss of acting now {loss}."
SEPARATING_LINE = "Separating test: {read}, price {price}, expected loss after it {loss}."
GAIN_LINE = "Gain of one {time_unit} per {series}: {gains}."
PLAN_LINE = "Plan {rank}: faulty {faulty}; repair {repairs}; warrant {warrant}; predicted recovery if acted {recovery}; cost to restored act now {act_now}, test first {test_first}{refuted}."
PLAN_REFUTED = " (refuted, not advised until a later ranking names it first)"
REFUTED_LINE = "Refuted sets: {sets}."
LINE_SERVER_READ = ("Advised reads, the read of the server: {reads}. While the ranking says read first, read them, the separating test or "
                    "the confirm test of the plan you follow before a commitment.")
LINE_ARM_READS = ("Advised reads, {arm}: {reads}. While the ranking says read first, read them, the separating test or the confirm test "
                  "of the plan you follow before a commitment.")
LINE_NO_READ_FITS = ("No separating read fits the budget after the reserve, so the advice is to commit without one, after an "
                     "observation, and name the separating test, {read}.")
LINE_SERVER_DECLARE = "The advice of the server is to commit without separating the candidates, with the separating test, {read}, after an observation."
LINE_SERVER_COMMIT = "Advised repair plan, the advice of the server: act on rank {rank}{top}."
LINE_TOP = ", the top candidate"
LINE_RANK_ONE = "Advised repair plan, with no candidate plan shown: act on rank 1, the top candidate."
LINE_BACKTRACK = ("The last repair plan was refuted, so the advice is to repair rank {rank}, the cheapest other plan, and not the refuted "
                  "set until a later ranking names it first.")
LINE_STILL_FAULTY = ("{head}, so the advice is to order the repair of the {target}s that the ranking still holds as faulty, those of the "
                     "faulty set with the most mass that hold no open {repair}{outside}: {targets}.")
HEAD_REFUTED, HEAD_NONE_LEFT = "The last repair plan was refuted and no other plan is left", "No candidate plan is left"
OUTSIDE_REFUTED = " and lie outside the refuted set"
LINE_SERVER = "Advice of the server: {text}"


def fill(template, /, **slots):
    """The template with every slot that slots names filled; every other brace stays as it is."""
    return SLOT.sub(lambda m: str(slots[m.group(1)]) if m.group(1) in slots else m.group(0), template)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


def _fmt(value):
    number = _number(value)
    if number is None:
        return "unknown"
    return str(int(number)) if number == int(number) else f"{number:.3g}"


def _read(thing, name, default=None):
    """An attribute of the board or the ledger, called where it is a method without arguments; the default where it is missing or
    not built yet."""
    try:
        value = getattr(thing, name)
        return value() if callable(value) else value
    except Exception:  # noqa: BLE001 - a part that the board or the ledger does not serve yet reads as absent
        return default


class Form(str):
    """The shape of an expectation as the line of a tool prints it, and the reader of its value: a count of at least a minimum, or
    a number inside a range. Calling it returns (value, None) or (None, reason)."""

    def __new__(cls, shape, kind, low=None, high=None):
        form = super().__new__(cls, shape)
        form.kind, form.low, form.high = kind, low, high
        return form

    def __call__(self, value, name="value"):
        number = _number(value)
        if self.kind == "count":
            if number is None or number != int(number) or number < self.low:
                return None, f"{name} must be an integer of at least {_fmt(self.low)}"
            return int(number), None
        if number is None or (self.low is not None and number < self.low) or (self.high is not None and number > self.high):
            bounds = f"from {_fmt(self.low)} to {_fmt(self.high)}" if self.high is not None else f"of at least {_fmt(self.low)}"
            return None, f"{name} must be a number {bounds}"
        return number, None


class RankingModule:
    kinds = ("ranking",)
    switches = ("branches", "single_hypothesis", "reads")

    def __init__(self, tools, table, options):
        self.tool, self.record = tools[KIND]
        self.table, self.options = table, options if options is not None else Options()
        planned = {name: row for name, row in table.tools.items() if row.role != "advice"}
        observe = next((row for row in planned.values() if row.role == "observe" and row.series), None)
        repair = next((row for row in planned.values() if row.write == "repair" and row.target), None)
        commit = next((row for row in planned.values() if row.commits and row.candidate), None)
        self.observe_tool = observe.name if observe else None
        self.series_arg = observe.series if observe else None
        self.count_arg = next(iter(observe.minimum), None) if observe else None
        self.repair_tool, self.target_arg = (repair.name, repair.target) if repair else (None, None)
        self.commit_tool, self.answer_args = (commit.name, tuple(commit.candidate)) if commit else (None, ())
        self.ranking, self.previous, self.planned = None, None, None  # the latest ranking, the one before it, the one of the turn
        self.basis, self.calls, self.failures = None, 0, 0
        self.refuted, self.refuted_last = [], None

    # ----- what the ranking holds -----
    def _words(self):
        vocabulary, terms = self.table.declaration.vocabulary, dict(self.table.declaration.vocabulary.terms or {})
        words = {key: terms.get(key, value) for key, value in FALLBACK_WORDS.items()}
        return {**words, "short_noun": vocabulary.short_noun, "ranking_tool": self.tool, "commit_tool": self.commit_tool or "the commit write"}

    def ranked(self):
        """The candidates of the latest ranking, in its order, with their ranks."""
        out = []
        for rank, item in enumerate((self.ranking or {}).get("candidates") or [], 1):
            if isinstance(item, dict):
                out.append({**item, "rank": item.get("rank", rank) if isinstance(item.get("rank"), int) else rank})
        return out

    def faulty(self, candidate):
        """The faulty set of a candidate, sorted: the targets of its faults."""
        if not isinstance(candidate, dict):
            return ()
        faults = candidate.get("faults") or []
        return tuple(sorted({str(f.get(self.target_arg)) for f in faults if isinstance(f, dict) and f.get(self.target_arg) is not None}))

    def set_mass(self, faulty, ranking=None):
        """The mass of a faulty set: the sum over every combination of the set where the ranking states it (set_masses), else the
        sum over the listed candidates of the set."""
        ranking = self.ranking if ranking is None else ranking
        for row in (ranking or {}).get("set_masses") or []:
            if isinstance(row, dict) and tuple(sorted(row.get("faulty_set") or ())) == tuple(faulty):
                return _number(row.get("mass")) or 0.0
        saved, self.ranking = self.ranking, ranking
        try:
            return sum(_number(c.get("probability")) or 0.0 for c in self.ranked() if self.faulty(c) == tuple(faulty))
        finally:
            self.ranking = saved

    def top_set(self, ranking=None):
        """(faulty set, mass) of the set that holds the most mass, or None without a ranking."""
        ranking = self.ranking if ranking is None else ranking
        if not ranking:
            return None
        sets = [(tuple(sorted(r.get("faulty_set") or ())), _number(r.get("mass")) or 0.0) for r in ranking.get("set_masses") or [] if isinstance(r, dict)]
        if not sets:
            saved, self.ranking = self.ranking, ranking
            try:
                sets = [(self.faulty(c), self.set_mass(self.faulty(c), ranking)) for c in self.ranked()]
            finally:
                self.ranking = saved
        return max(sets, key=lambda s: s[1]) if sets else None

    def effective(self, ranking=None):
        return _number(((self.ranking if ranking is None else ranking) or {}).get("effective_hypotheses"))

    def _rule(self):
        """The block of the earlier rule where the long form carries it, else the ranking."""
        ranking = self.ranking or {}
        return ranking.get("e10_rule") if isinstance(ranking.get("e10_rule"), dict) else ranking

    def read_first(self):
        return bool(self._rule().get("read_first"))

    def _read_of(self, item):
        """(series, count) of a read that an answer names, as a text or an object."""
        if isinstance(item, str):
            return item, 1
        if isinstance(item, dict) and item.get(self.series_arg):
            count = _number(item.get(self.count_arg)) if self.count_arg else None
            return str(item[self.series_arg]), int(count) if count else 1
        return None

    def read_batch(self):
        return [read for read in map(self._read_of, self._rule().get("read_batch") or []) if read]

    def separating_test(self):
        test = self._rule().get("separating_test")
        return test if isinstance(test, dict) and test.get(self.series_arg) else None

    def server_advice(self):
        advice = (self.ranking or {}).get("advice")
        return advice if isinstance(advice, dict) else None

    def branches(self):
        """The candidate plans that the switches show: none with branches off, the first alone with single_hypothesis."""
        if not self.options.branches:
            return []
        plans = [b for b in (self.ranking or {}).get("branches") or [] if isinstance(b, dict)]
        return plans[:1] if self.options.single_hypothesis else plans

    def branch(self, rank):
        return next((b for b in self.branches() if rank in (b.get("ranks") or [b.get("rank")])), None)

    def plan_targets(self, branch):
        """The targets that the repair plan of a candidate plan repairs: those of its steps, else its faulty set."""
        plan = branch.get("repair_plan") or {}
        steps = plan.get("steps") if isinstance(plan, dict) else plan
        targets = [s.get(self.target_arg) for s in steps or [] if isinstance(s, dict) and s.get(self.target_arg)]
        return tuple(targets) or tuple(branch.get("faulty_set") or ())

    def plan_warrant(self, branch, target=None):
        """The warrant that every repair write of a plan cites: the plan's, else the one of the step on the target."""
        plan = branch.get("repair_plan") or {}
        if isinstance(plan, dict) and plan.get("plan_prediction_id"):
            return plan["plan_prediction_id"]
        steps = plan.get("steps") if isinstance(plan, dict) else plan
        return next((s.get("prediction_id") for s in steps or [] if isinstance(s, dict) and s.get(self.target_arg) == target
                     and s.get("prediction_id")), None)

    # ----- the reserve and the options of a repair -----
    def _repairs(self, targets):
        row = self.table.tools.get(self.repair_tool)
        return [(row, {self.target_arg: target}) for target in targets] if row is not None else []

    def completion(self, ledger, targets):
        """The Needs of completing the repair writes of the targets from the ledger as it stands, or None where the kind cannot
        tell."""
        try:
            return ledger.walk().needs(self._repairs(targets))
        except Exception:  # noqa: BLE001 - a kind without a walk prices no completion
            return None

    def _pays(self, ledger, targets):
        """Whether the budget left pays the completion of the repair of the targets; True where the episode has no budget."""
        if not targets:
            return True
        budget = _number((_read(ledger, "status") or {}).get("budget_left"))
        needs = self.completion(ledger, targets)
        return budget is None or needs is None or (_number(getattr(needs, "price", None)) or 0.0) <= budget

    def reserve_targets(self, ledger=None):
        """The targets that the reserve repairs: the repair plan of the first candidate plan, else the faulty set of the top
        candidate, without the targets whose repair is open."""
        branch = self.branch(1)
        targets = self.plan_targets(branch) if branch is not None else (self.faulty(self.ranked()[0]) if self.ranked() else ())
        opened = set(_read(ledger, "open_targets") or {}) if ledger is not None else set()
        return tuple(t for t in targets if t not in opened)

    def reserve(self, ledger):
        """What an observation leaves for the repair: the Needs of completing the repair of the reserve targets."""
        return self.completion(ledger, self.reserve_targets(ledger))

    def options_of_repair(self, ledger, after_commitment=False):
        """[(cost to restored acting now, rank)] of the candidate plans whose completion the budget left pays, cheapest first, ties
        to the higher rank, without the last refuted set; after a commitment, a plan with no faulty target is no option."""
        out = []
        for branch in self.branches():
            cost = _number((branch.get("cost_to_restored") or {}).get("act_now"))
            targets = self.plan_targets(branch)
            if cost is None or (self.refuted_last is not None and tuple(sorted(targets)) == self.refuted_last):
                continue
            if after_commitment and not targets:
                continue
            if self._pays(ledger, targets):
                out.append((round(cost, 6), branch.get("rank")))
        return sorted(out, key=lambda o: (o[0], o[1] if isinstance(o[1], int) else 0))

    # ----- the refuted sets -----
    def record_refuted(self, targets, observations=0):
        """A repair plan that the verification refuted: its set is kept out of the advice until a ranking after a further
        observation names it first."""
        entry = {"targets": tuple(sorted(targets)), "observations": observations}
        self.refuted.append(entry)
        self.refuted_last = entry["targets"]

    def _release(self, observations):
        if self.refuted_last is None or not self.refuted or observations <= self.refuted[-1]["observations"]:
            return
        top = self.top_set()
        if top is not None and top[0] == self.refuted_last:
            self.refuted_last = None

    # ----- the module interface -----
    def blocks(self):
        """Its block of the system prompt and its sentence of BOARD: the block for this run, with the sentences of the switches it
        owns and of the switch parallel left out or replaced where they are off or set."""
        words = self._words()
        sentences = []
        for index, sentence in enumerate(BLOCK):
            if index == BRANCHES_SENTENCE and not self.options.branches:
                continue
            if index == SCHEDULE_SENTENCE and not self.options.parallel:
                continue
            if index == READS_SENTENCE and self.options.reads != "gain":
                sentence = fill(READS_ARM, arm=READ_ARMS.get(self.options.reads, self.options.reads))
            if index == COMMIT_SENTENCE and self.commit_tool is None:
                continue
            sentences.append(fill(sentence, **words))
        return [" ".join(sentences), ("board", BOARD_SENTENCE)]

    def forms(self, row):
        """The expectation keys it judges for a step of this row, each a Form: top_rank, top_mass_at_least and effective_at_most on
        an observe step, top_rank on a commit write with candidate arguments."""
        if row.role == "observe":
            return {"top_rank": Form(SHAPE_TOP_RANK, "count", 1), "top_mass_at_least": Form(SHAPE_MASS, "number", 0.0, 1.0),
                    "effective_at_most": Form(SHAPE_EFFECTIVE, "number", 1.0)}
        if row.commits and row.candidate:
            return {"top_rank": Form(SHAPE_TOP_RANK_COMMIT, "count", 1)}
        return {}

    def step_problems(self, step, index, state):
        """Its rules of the check: a top_rank outside the ranked candidates, and a commit by a ranked candidate before any ranking."""
        row = self.table.tools.get(getattr(step, "tool", None))
        expect = getattr(step, "expect", None) or {}
        if row is None or "top_rank" not in expect:
            return []
        rank, n = expect.get("top_rank"), len(self.ranked())
        if row.commits and row.candidate and self.ranking is None:
            return [fill(REASON_NO_RANKING, tool=row.name)]
        if self.ranking is not None and isinstance(rank, int) and not isinstance(rank, bool) and not 1 <= rank <= n:
            return [fill(REASON_RANK, n=n, rank=rank)]
        return []

    def candidates(self):
        """The ranked candidates, which take the first ranks, each in the form of a candidate of a plan: its answer arguments, its
        rank, its mass and the identifiers it cites."""
        out = []
        for c in self.ranked():
            combination = c.get("current_combination") or c.get("combination") or {}
            answer = {arg: combination[arg] for arg in self.answer_args if isinstance(combination, dict) and arg in combination}
            out.append({**answer, "rank": c["rank"], "mass": _number(c.get("probability")), "cites": list(c.get("evidence_ids") or [])})
        return out

    def sections(self, board):
        """What it prints on the board, each (place, heading, lines): the ranking with its decision fields, the advice lines, and the
        candidate plans where the switch branches shows them."""
        words = self._words()
        if self.ranking is None:
            text = fill(FAILED_RANKING, count=self.failures) if self.failures else NO_RANKING
            return [("advice", HEADING_RANKING, [text])]
        stale = STALE if self.basis is not None and self.basis != self._basis(board) else ""
        ranking = [fill(RANKING_LINE, rank=c["rank"], mass=_fmt(c.get("probability")),
                        answer=json.dumps(c.get("current_combination") or c.get("combination") or {}, sort_keys=True),
                        faults=fill(FAULTS, targets=", ".join(self.faulty(c))) if self.faulty(c) else "") for c in self.ranked()]
        if self.effective() is not None:
            ranking.append(fill(EFFECTIVE_LINE, value=_fmt(self.effective())))
        ranking.append(fill(DECISION_LINE, read_first=self.read_first(), loss=_fmt(self._rule().get("expected_loss_act"))))
        test = self.separating_test()
        if test:
            ranking.append(fill(SEPARATING_LINE, read=self._read_text(self._read_of(test)), price=_fmt(test.get("price")),
                                loss=_fmt(test.get("expected_loss"))))
        gains = [e for e in (self.ranking.get("elimination") or []) if isinstance(e, dict) and e.get(self.series_arg)]
        if gains:
            ranking.append(fill(GAIN_LINE, gains=", ".join(f"{e[self.series_arg]} {_fmt(e.get('gain'))} ({_fmt(e.get('gain_per_cost'))} per unit cost)"
                                                           for e in gains), **words))
        out = [("advice", HEADING_RANKING + stale, ranking), ("advice", HEADING_ADVICE, self.advice_lines(board))]
        if self.branches():
            out.append(("advice", HEADING_PLANS, [self._plan_line(b) for b in self.branches()]))
        if self.refuted:
            out.append(("advice", HEADING_PLANS if not self.branches() else HEADING_ADVICE,
                        [fill(REFUTED_LINE, sets="; ".join(", ".join(e["targets"]) or "none" for e in self.refuted))]))
        return out

    def _plan_line(self, branch):
        targets = self.plan_targets(branch)
        outcome, cost = branch.get("predicted_outcome") or {}, branch.get("cost_to_restored") or {}
        refuted = PLAN_REFUTED if self.refuted_last is not None and tuple(sorted(targets)) == self.refuted_last else ""
        return fill(PLAN_LINE, rank=branch.get("rank"), faulty=", ".join(branch.get("faulty_set") or ()) or "none",
                    repairs=", ".join(targets) or "none", warrant=self.plan_warrant(branch, targets[0] if targets else None) or "none",
                    recovery=_fmt(outcome.get("recovery_if_acted")), act_now=_fmt(cost.get("act_now")), test_first=_fmt(cost.get("test_first")),
                    refuted=refuted)

    def _read_text(self, read):
        return f"{read[0]} x{read[1]}" if read else "none"

    def advised_reads(self, ledger=None):
        """The reads that the advice names while the ranking says read first: the read of the server under the default of the
        switch reads, else the read batch, the one step edge cutting read or the separating test of the arm, as far as they fit
        beside the reserve."""
        if self.ranking is None:
            return []
        advice = self.server_advice() or {}
        if self.options.reads == "gain" and advice:
            read = self._read_of(advice) if advice.get("action") == "read" else None
            return [read] if read else []
        if not self.read_first():
            return []
        reads = {"gain": self.read_batch(), "ec2": self._ec2_read()}.get(self.options.reads, [])
        if not reads and self.separating_test():
            reads = [self._read_of(self.separating_test())]
        return reads

    def _ec2_read(self):
        rows = [e for e in self.ranking.get("elimination") or [] if isinstance(e, dict) and (_number(e.get("ec2_per_cost")) or 0) > 0]
        if not rows:
            return []
        best = max(rows, key=lambda e: _number(e.get("ec2_per_cost")))
        read = self._read_of(best)
        return [read] if read else []

    def advice_lines(self, board):
        """The lines of advice: the reads to take while the ranking says read first, the repair plan to act on, after a refuted
        plan the cheapest other plan, and after a commitment with no plan left what the ranking still holds as faulty where the
        budget left pays its repair; then the advice of the server. None of it binds the model."""
        if self.ranking is None:
            return []
        words, ledger = self._words(), _read(board, "ledger")
        committed = _read(board, "commitment") is not None
        backtracking = getattr(self.options, "backtracking", True)
        advice = self.server_advice() or {}
        follow = self.options.reads == "gain" and bool(advice)
        lines = []
        if follow and advice.get("action") == "read" and self._read_of(advice):
            lines.append(fill(LINE_SERVER_READ, reads=self._read_text(self._read_of(advice))))
        elif not follow and not committed and self.read_first():
            reads = self.advised_reads(ledger)
            if reads:
                lines.append(fill(LINE_ARM_READS, arm=READ_ARMS.get(self.options.reads, self.options.reads),
                                  reads=", ".join(self._read_text(r) for r in reads)))
            elif self.separating_test():
                lines.append(fill(LINE_NO_READ_FITS, read=self._read_text(self._read_of(self.separating_test()))))
        refuted = bool(backtracking and self.refuted_last is not None)
        if advice and advice.get("action") == "declare":
            lines.append(fill(LINE_SERVER_DECLARE, read=self._read_text(self._read_of(advice))))
        elif advice and advice.get("action") in ("diagnose", "commit") and not committed:
            rank = advice.get("rank") if isinstance(advice.get("rank"), int) else 1
            lines.append(fill(LINE_SERVER_COMMIT, rank=rank, top=LINE_TOP if rank == 1 else ""))
        elif advice and not committed:
            pass  # the server advises a read first, and its repair plan follows the read
        elif not committed and not refuted and self.ranked():
            lines.append(LINE_RANK_ONE)
        elif committed or refuted:
            options = self.options_of_repair(ledger, after_commitment=True)
            if refuted and options:
                lines.append(fill(LINE_BACKTRACK, rank=options[0][1]))
            elif not options:
                top = self.top_set()
                opened = set(_read(ledger, "open_targets") or {})
                left_out = opened | set(self.refuted_last if refuted else ())
                still = [t for t in (top[0] if top else ()) if t not in left_out]
                if still and self._pays(ledger, still):
                    lines.append(fill(LINE_STILL_FAULTY, head=HEAD_REFUTED if refuted else HEAD_NONE_LEFT,
                                      outside=OUTSIDE_REFUTED if refuted else "", targets=", ".join(still), **words))
        if advice.get("text") or (self.ranking or {}).get("message"):
            lines.append(fill(LINE_SERVER, text=advice.get("text") or self.ranking.get("message")))
        return lines

    def known_ids(self):
        """The identifiers that the ranking puts on the board: those its candidates cite and the warrants of its candidate plans."""
        ids = {e for c in self.ranked() for e in c.get("evidence_ids") or [] if isinstance(e, str)}
        for branch in self.branches():
            plan = branch.get("repair_plan") or {}
            if isinstance(plan, dict) and plan.get("plan_prediction_id"):
                ids.add(plan["plan_prediction_id"])
            for s in (plan.get("steps") if isinstance(plan, dict) else plan) or []:
                if isinstance(s, dict) and s.get("prediction_id"):
                    ids.add(s["prediction_id"])
        return ids

    def evidence_for(self, step, row):
        """The identifiers it adds to the evidence of a commit write by a ranked candidate: those that candidate cites."""
        rank = getattr(step, "candidate", None)
        if not (row.commits and row.candidate) or not isinstance(rank, int):
            return []
        candidate = next((c for c in self.ranked() if c["rank"] == rank), None)
        return list((candidate or {}).get("evidence_ids") or [])

    # ----- the loop -----
    def _basis(self, board):
        """What a ranking stands on: the observations, the open repairs and the time of the board."""
        ledger = _read(board, "ledger")
        observations = _read(board, "observations") or ()
        status = _read(board, "status") or _read(ledger, "status") or {}
        return (len(observations) if hasattr(observations, "__len__") else 0, tuple(sorted(_read(ledger, "open_targets") or {})),
                status.get("time") if isinstance(status, dict) else None)

    def stale(self, board):
        return self.ranking is None or self.basis != self._basis(board)

    async def _ask(self, ctx):
        """One ranking call through the context, in the form that the switch reads asks for; a failed call counts and keeps the
        ranking before it."""
        self.calls += 1
        try:
            answer = await ctx.ask(KIND, {DETAIL_INPUT: COMPACT if self.options.reads == "gain" else SUMMARY})
        except Exception:  # noqa: BLE001 - a call without an answer is a failed ranking
            answer = None
        result = answer.get("result") if isinstance(answer, dict) and "refused" not in answer else None
        if not isinstance(result, dict):
            self.failures += 1
            return False
        self.previous, self.ranking, self.basis = self.ranking, result, self._basis(ctx.board)
        observations = self.basis[0]
        self._release(observations)
        return True

    async def before_turn(self, ctx):
        """Ask the ranking when the observations, the closed repairs or the clock moved, once the facts are on the board where the
        table has a facts tool."""
        if self.table.facts_tool() is not None and _read(ctx.board, "facts") is None:
            return
        if self.stale(ctx.board):
            await self._ask(ctx)
        self.planned = self.ranking

    async def before_step(self, step, row, ctx):
        """Before a commit write, the ranking again where it is stale; it writes no fill."""
        if row.commits and self.stale(ctx.board):
            await self._ask(ctx)
        return {}

    async def after_batch(self, steps, answers, ctx):
        """Once after a batch that made a call, the ranking again; then the judgment of its forms, one per step, each None where
        the step states none of its keys, else {"holds", "detail"}. A standing step comes alone with None as its answer and is
        judged on the ranking that the board holds, which is asked again only when it is stale."""
        before = self.ranking
        called = any(answer is not None for answer in answers)
        if called or self.stale(ctx.board):
            await self._ask(ctx)
        if not called:
            before = self.ranking
        return [self.judge(step, before) for step in steps]

    def judge(self, step, before):
        """The judgment of the forms of a step against the ranking before it and the ranking now."""
        row = self.table.tools.get(getattr(step, "tool", None))
        expect = dict(getattr(step, "expect", None) or {})
        if row is None:
            return None
        tolerance = getattr(self.options, "expectation_tolerance", 0.0) or 0.0
        if row.commits and row.candidate:
            if "top_rank" not in expect and isinstance(getattr(step, "candidate", None), int) and step.candidate <= len(self.ranked()):
                expect["top_rank"] = 1  # a commit by a ranked candidate expects the candidate on top by default
            if "top_rank" not in expect:
                return None
            planned = self.planned or before
            expected = self._faulty_at(expect["top_rank"], planned)
            top = self.top_set()
            holds = expected is not None and top is not None and top[0] == expected
            return {"holds": holds, "detail": f"top_rank {expect['top_rank']}: expected faulty set {list(expected or ())}, on top {list(top[0]) if top else []}"}
        if row.role != "observe" or not set(expect) & {"top_rank", "top_mass_at_least", "effective_at_most"}:
            return None
        failures = []
        top = self.top_set()
        if "top_rank" in expect or "top_mass_at_least" in expect:
            expected = self._faulty_at(expect.get("top_rank", 1), before)
            if expected is None or top is None or top[0] != expected:
                failures.append(f"the set on top is {list(top[0]) if top else []}, not {list(expected or ())}")
            elif "top_mass_at_least" in expect and top[1] < (_number(expect["top_mass_at_least"]) or 0.0) - tolerance:
                failures.append(f"the mass of the set on top is {_fmt(top[1])}, below {_fmt(expect['top_mass_at_least'])}")
        if "effective_at_most" in expect:
            effective = self.effective()
            if effective is None or effective > (_number(expect["effective_at_most"]) or 0.0) + tolerance:
                failures.append(f"the effective hypotheses are {_fmt(effective)}, above {_fmt(expect['effective_at_most'])}")
        return {"holds": not failures, "detail": "; ".join(failures)}

    def _faulty_at(self, rank, ranking):
        if not ranking or not isinstance(rank, int):
            return None
        saved, self.ranking = self.ranking, ranking
        try:
            candidate = next((c for c in self.ranked() if c["rank"] == rank), None)
            return self.faulty(candidate) if candidate is not None else None
        finally:
            self.ranking = saved

    async def on_pending(self, pending, observation, ctx):
        """Before a repair is verified, the ranking again where it is stale; the ranking states no verdict of its own."""
        if self.stale(ctx.board):
            await self._ask(ctx)
        return None

    def to_json(self):
        return {"kind": KIND, "tool": self.tool, "calls": self.calls, "failures": self.failures,
                "switches": {name: getattr(self.options, name) for name in self.switches},
                "refuted": [list(e["targets"]) for e in self.refuted], "top": list(self.top_set()[0]) if self.top_set() else None}
