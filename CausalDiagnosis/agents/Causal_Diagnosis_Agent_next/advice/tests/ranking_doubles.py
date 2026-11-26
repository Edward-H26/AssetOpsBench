"""Test doubles of the tests of the ranking module: a ledger with a status, open targets and a walk whose needs price a repair, a
board with the observations, the facts, the commitment and the ledger, a context whose ask answers from a queue, a ranking answer
built for any table from the names that the table gives, the declaration of the tiny asset of CausalDiagnosis/datasets/tests/ in
the shape of the rig's, and the stand-in model of the rule. The board, the ledger and the loop are stubs of the freeze until
their tasks build them, so the tests read them through these doubles."""
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.advice.ranking import RankingModule
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests import fixture_declarations as fixtures
from CausalDiagnosis.agents.utils.answer import answer
from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary

ROOT = Path(__file__).resolve().parents[5]
TINY_FILE = ROOT / "CausalDiagnosis" / "datasets" / "tests" / "tiny_asset.json"
RIG_FILE = ROOT / "CausalDiagnosis" / "datasets" / "hyd_1" / "asset.json"


@dataclass(frozen=True)
class Needs:
    price: float
    time: int
    calls: int


class Walk:
    def __init__(self, ledger):
        self.ledger = ledger

    def needs(self, repairs):
        """One visit charge for the visit and the price of each repair, one call per repair and one observation that verifies."""
        targets = [args[next(iter(args))] for _row, args in repairs]
        price = (self.ledger.visit if targets else 0) + sum(self.ledger.prices.get(t, 0) for t in targets)
        return Needs(price, len(targets), len(targets) + (1 if targets else 0))


class Ledger:
    def __init__(self, budget_left=None, time=0, open_targets=None, prices=None, visit=0):
        self.budget_left, self.time, self._open = budget_left, time, dict(open_targets or {})
        self.prices, self.visit = dict(prices or {}), visit

    def status(self):
        return {"time": self.time, "time_limit": None, "budget_left": self.budget_left, "steps_left": None, "over": False}

    def limits(self):
        return ("budget",) if self.budget_left is not None else ()

    def open_targets(self):
        return dict(self._open)

    def walk(self):
        return Walk(self)


class Board:
    def __init__(self, ledger=None, facts=None, observations=(), commitment=None):
        self.ledger, self.facts, self.commitment = ledger or Ledger(), facts, commitment
        self.observations = list(observations)

    @property
    def status(self):
        return self.ledger.status()


class Context:
    """The context of a module: the board, the ledger and ask, which answers from a queue of results (None for a failed call) and
    records each call."""

    def __init__(self, board, results=()):
        self.board, self.ledger = board, board.ledger
        self.results, self.asked = list(results), []

    async def ask(self, kind, inputs):
        self.asked.append((kind, dict(inputs)))
        result = self.results.pop(0) if self.results else None
        return answer(refused="the server did not answer") if result is None else answer(result=result)


def asset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def facts_of(data):
    return {"components": list(data["components"]), "levels": dict(data["levels"]), "healthy": dict(data["healthy"]),
            "sensor_groups": {name: list(kind["channels"]) for name, kind in data["observation_kinds"].items()}}


def _schema(properties, required):
    return {"type": "object", "properties": properties, "required": list(required)}


def tiny_declaration(terms=None):
    """The tiny asset in the shape of the rig on its own action server, with a ranking tool under a name of its own and the target
    term of the tiny asset; its listing holds neutral texts."""
    data = asset(TINY_FILE)
    components = tuple(data["components"])
    text = {"type": "string"}
    server = "tiny_bench"
    listing = {server: [
        {"name": "profile", "description": "The profile.", "inputSchema": _schema({}, [])},
        {"name": "read", "description": "Read one group.", "inputSchema": _schema({"group": text, "units": {"type": "integer"}}, ["group", "units"])},
        {"name": "commit_levels", "description": "Commit the levels.",
         "inputSchema": _schema({**{c: {"type": "integer"} for c in components}, "cites": text}, [*components, "cites"])},
        {"name": "fix", "description": "Fix one part.", "inputSchema": _schema({"part": text, "warrant": text}, ["part", "warrant"])},
        {"name": "end", "description": "End.", "inputSchema": _schema({"note": text}, ["note"])},
    ], fixtures.DIAGNOSTICS_SERVER: [{"name": "rank_causes", "description": "Rank.", "inputSchema": _schema({}, [])}]}
    rows = {
        "profile": ToolRow(name="profile", server=server, role="task", facts=True),
        "read": ToolRow(name="read", server=server, role="observe", series="group", values={"group": "sensor_groups"}, answer="records",
                        minimum={"units": 1}),
        "commit_levels": ToolRow(name="commit_levels", server=server, role="task", write="answer", commits=True, candidate=components,
                          fills={"cites": "evidence"}),
        "fix": ToolRow(name="fix", server=server, role="act", write="repair", target="part", fills={"warrant": "warrant"}),
        "end": ToolRow(name="end", server=server, role="task", write="answer", ends=True, fills={"note": "closing_summary"}),
        "rank_causes": ToolRow(name="rank_causes", server=fixtures.DIAGNOSTICS_SERVER, role="advice", advice="ranking"),
    }
    vocabulary = Vocabulary(noun=data["text"]["noun"], short_noun=data["text"]["short_noun"], example_field=data["text"]["example_channel"],
                            terms=terms if terms is not None else {"time_unit": "tick", "series": "channel group", "target": "part", "repair": "fix"})
    declaration = Declaration(asset_id=data["asset_id"], kind="recorded_levels", surface=None, vocabulary=vocabulary, rows=rows, listed_rows={},
                              server_roles={}, cards=(), advice={"rank_causes": fixtures._advice("ranking")}, events=(),
                              rules=fixtures.RecordedLevelsRules(), binding=None)
    return declaration, {s: [SimpleNamespace(**t) for t in tools] for s, tools in listing.items()}, facts_of(data)


def tables():
    """name -> (table, facts) of the two tables with a ranking tool: the rig on its own action server and the tiny asset."""
    rig, listing = fixtures.declarations()["rig"]
    tiny, tiny_listing, tiny_facts = tiny_declaration()
    return {"rig": (rig.table(listing), facts_of(asset(RIG_FILE))), "tiny": (tiny.table(tiny_listing), tiny_facts)}


def module(table, options=None):
    tool = next(name for name, row in table.tools.items() if row.advice == "ranking")
    return RankingModule({"ranking": (tool, table.declaration.advice[tool])}, table, options or Options())


def names(m):
    """The names that a ranking answer of the table keys its reads and its faults by."""
    return m.series_arg, m.count_arg, m.target_arg


def ranking(m, facts, sets, read_first=False, advice=None, branches=None, effective=None, read_batch=(), separating=None, elimination=()):
    """A ranking answer for the table of module m: sets is [(faulty targets, mass)], one candidate per set, best first; each
    candidate holds every other target at its healthy level and its faulty ones at the first other level."""
    series, count, target = names(m)
    healthy = facts["healthy"]
    candidates, set_masses = [], []
    for rank, (faulty, mass) in enumerate(sets, 1):
        combination = {c: (next(level for level in facts["levels"][c] if level != healthy[c]) if c in faulty else healthy[c]) for c in healthy}
        candidates.append({"rank": rank, "combination": combination, "probability": mass,
                           "faults": [{target: c, "level": combination[c], "code": f"F{rank}"} for c in faulty], "evidence_ids": [f"obs/{rank}/x"]})
        set_masses.append({"faulty_set": sorted(faulty), "mass": mass})
    result = {"message": "ranked", "candidates": candidates, "set_masses": set_masses, "read_first": read_first,
              "read_batch": [{series: g, count: n} for g, n in read_batch]}
    if effective is not None:
        result["effective_hypotheses"] = effective
    if advice is not None:
        result["advice"] = advice
    if separating is not None:
        result["separating_test"] = {series: separating[0], count: separating[1], "price": 1, "expected_loss": 0.5}
    if elimination:
        result["elimination"] = [{series: g, "gain": gain, "gain_per_cost": gain, "ec2_per_cost": ec2} for g, gain, ec2 in elimination]
    if branches is not None:
        result["branches"] = [{"rank": rank, "faulty_set": sorted(faulty), "cost_to_restored": {"act_now": cost, "test_first": cost + 1},
                               "predicted_outcome": {"recovery_if_acted": 0.9},
                               "repair_plan": {"plan_prediction_id": f"pred/b{rank}", "steps": [{target: c, "prediction_id": f"pred/{c}"} for c in faulty]}}
                              for rank, faulty, cost in branches]
    return result


class RuleModel:
    """The stand-in model of the rule: the rule that plays without a model, which writes its plan as the JSON object that a model
    would write, so that every plan of it passes through the reader and the check. It plans the opening reads before the facts;
    while the ranking says read first, the advised reads, one step each, with an expectation on the effective hypotheses where the
    ranking states them; else a commitment by the rank of the cheapest candidate plan (rank 1 without one), the repair write of
    each of its faulty targets, the observe step that verifies them, and the write that ends the episode."""

    model_id = "rule"

    def __init__(self, m, table):
        self.m, self.table = m, table

    def complete(self, board):
        table, m = self.table, self.m
        rows = {name: row for name, row in table.tools.items() if row.role != "advice"}
        if table.facts_tool() is not None and board.facts is None:
            steps = [{"tool": table.facts_tool(), "args": {}}]
            steps += [{"tool": name, "args": {}} for name, row in rows.items() if row.role == "knowledge"]
            return json.dumps({"follow": None, "why": "rule: read the opening", "steps": steps})
        reads = m.advised_reads(board.ledger) if m.read_first() else []
        if reads:
            expect = {"effective_at_most": m.effective()} if m.effective() is not None else {"top_rank": 1, "top_mass_at_least": 0.0}
            steps = [{"tool": m.observe_tool, "args": {m.series_arg: g, m.count_arg: n}, "expect": expect} for g, n in reads]
            return json.dumps({"follow": 1, "why": "rule: the ranking says read first", "steps": steps})
        options = m.options_of_repair(board.ledger)
        rank = options[0][1] if options else 1
        steps = []
        if m.commit_tool in rows:
            steps.append({"tool": m.commit_tool, "args": {"candidate": rank}, "expect": {"top_rank": 1}})
        branch = m.branch(rank)
        targets = m.plan_targets(branch) if branch else m._faulty_at(rank, m.ranking) or ()
        if m.repair_tool in rows:
            steps += [{"tool": m.repair_tool, "args": {m.target_arg: t}} for t in targets]
        if targets and m.observe_tool in rows:
            group = next(iter(board.facts.get("sensor_groups") or {}), None) if board.facts else None
            steps.append({"tool": m.observe_tool, "args": {m.series_arg: group, m.count_arg: 1}})
        if table.end_tool() in rows:
            steps.append({"tool": table.end_tool(), "args": {}})
        return json.dumps({"follow": rank, "why": f"rule: act on rank {rank}, the cheapest plan", "steps": steps})


def without(table, tool):
    """The table without one tool, as a row that does not serve it lists it."""
    return replace(table, tools={name: row for name, row in table.tools.items() if name != tool})
