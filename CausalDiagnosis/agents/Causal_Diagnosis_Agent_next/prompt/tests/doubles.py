"""Test doubles of the prompt tests: a board and a ledger that serve what the assembler reads through the documented interface
of working_memory/board.py and of the ledger of a kind, a module that gives a block, a sentence of BOARD and forms, the facts of an
asset read from its asset file, and a declaration of the tiny asset of CausalDiagnosis/datasets/tests/ in the shape of the rig's,
with a listing whose texts name nothing but the tiny asset's own words. The board and the ledger of the agent are stubs of the
freeze until their tasks build them, so the tests of the prompt read them through these doubles."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests import fixture_declarations as fixtures
from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary

ROOT = Path(__file__).resolve().parents[5]
DATASETS = ROOT / "CausalDiagnosis" / "datasets"
TINY_FILE = DATASETS / "tests" / "tiny_asset.json"
TASK = "Which level does each part hold?\n\nAn appended block of the episode: keep every read short."


class Ledger:
    """The limits and the status of an episode."""

    def __init__(self, limits=("budget", "time", "steps"), status=None):
        self._limits = tuple(limits)
        self._status = dict(status or {"time": 3, "time_limit": 100, "budget_left": 50, "steps_left": 9, "over": False})

    def limits(self):
        return self._limits

    def status(self):
        return dict(self._status)


class Board:
    """What the assembler reads of a board: the task message, the facts, the ledger and its status, the commitment, the writes
    and the verdicts, and the rendered board, which never repeats the task message."""

    def __init__(self, question=TASK, facts=None, ledger=None, writes=None, verdicts=(), text="The board rows."):
        self.question, self.facts, self.ledger = question, facts, ledger or Ledger()
        self.writes, self.verdicts, self.text = writes, tuple(verdicts), text
        self.commitment = None

    @property
    def status(self):
        return self.ledger.status()

    def render(self, modules):
        return "\n".join([self.text, *(f"Section of a module: {name}" for module in modules or () for name in getattr(module, "kinds", ()))])


class Module:
    """A module of one advice kind that gives one block, one sentence of BOARD, one form on the rows of one role and one
    identifier form."""

    def __init__(self, kind="ranking", block="Advice: a block of the module.", sentence="The board also shows the advice.",
                 form=("top_rank", "a rank of at least 1"), role="observe", identifier=("adv/<n>", "an answer of the module")):
        self.kinds, self.switches = (kind,), ()
        self._block, self._sentence, self._form, self._role = block, sentence, form, role
        self.identifier_forms = (identifier,)

    def blocks(self):
        return [self._block, ("board", self._sentence)]

    def forms(self, row):
        return {self._form[0]: self._form[1]} if row.role == self._role else {}


def asset_data(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registered_assets():
    """The asset files of the registered assets, read from the folders of the datasets."""
    return [asset_data(path) for path in sorted(DATASETS.glob("*/asset.json"))]


def asset_words(data):
    """The words of an asset that no text of another asset may hold: its nouns, its example channel, its id, its components, its
    sensor groups and its channels."""
    text = data.get("text") or {}
    nouns = {value for key, value in text.items() if key.endswith("noun") and isinstance(value, str)}
    example = {text["example_channel"]} if text.get("example_channel") else set()
    return nouns | example | {data["asset_id"], *data["components"], *data["observation_kinds"], *data["channels"]}


def facts_of(data):
    """The public facts that the facts tool of an asset of recorded levels would publish, in the paths that the rows read."""
    return {"components": list(data["components"]), "levels": dict(data["levels"]), "healthy": dict(data["healthy"]),
            "sensor_groups": {name: list(kind["channels"]) for name, kind in data["observation_kinds"].items()}}


def _schema(properties, required):
    return {"type": "object", "properties": properties, "required": list(required)}


def tiny_declaration(value="mean"):
    """The tiny asset in the shape of the rig on its own action server, with the diagnostics server: its rows hold the keys of the
    rig's rows, its listing names the tiny asset's components, and the descriptions of its tools are neutral texts; value is the
    term of the plain statistic that the asset serves."""
    data = asset_data(TINY_FILE)
    components = tuple(data["components"])
    text = {"type": "string"}
    listing = {"tiny_bench": [
        {"name": "get_asset_profile", "description": "The profile of the asset.", "inputSchema": _schema({}, [])},
        {"name": "read_sensors", "description": "Read one sensor group.",
         "inputSchema": _schema({"sensor_group": text, "cycles": {"type": "integer"}}, ["sensor_group", "cycles"])},
        {"name": "get_failure_modes", "description": "The failure modes of one component.", "inputSchema": _schema({"component": text}, ["component"])},
        {"name": "submit_diagnosis", "description": "Submit the diagnosis.",
         "inputSchema": _schema({**{c: {"type": "integer"} for c in components}, "evidence_ids": text}, [*components, "evidence_ids"])},
        {"name": "create_work_order", "description": "Repair one component.",
         "inputSchema": _schema({"component": text, "prediction_id": text, "evidence_ids": text}, ["component", "prediction_id", "evidence_ids"])},
        {"name": "finish", "description": "End the episode.", "inputSchema": _schema({"summary": text}, ["summary"])},
    ], fixtures.DIAGNOSTICS_SERVER: [
        {"name": "rank_root_causes", "description": "Rank the causes.", "inputSchema": _schema({}, [])},
    ]}
    server = "tiny_bench"
    keys = {
        "get_asset_profile": dict(role="task", facts=True),
        "read_sensors": dict(role="observe", values={"sensor_group": "sensor_groups"}, series="sensor_group", answer="records",
                             minimum={"cycles": 1}, omitted={"cycles": "fixed:1"}),
        "get_failure_modes": dict(role="knowledge", values={"component": "components"}, orients=True, each="component"),
        "submit_diagnosis": dict(role="task", write="answer", commits=True, candidate=components,
                                 values={c: f"levels.{c}" for c in components}, omitted={c: f"healthy.{c}" for c in components},
                                 fills={"evidence_ids": "evidence"}),
        "create_work_order": dict(role="act", write="repair", target="component", values={"component": "components"},
                                  after=("submit_diagnosis",), fills={"prediction_id": "warrant", "evidence_ids": "evidence"}),
        "finish": dict(role="task", write="answer", ends=True, fills={"summary": "closing_summary"}),
    }
    rows = {name: ToolRow(name=name, server=server, **row) for name, row in keys.items()}
    rows["rank_root_causes"] = ToolRow(name="rank_root_causes", server=fixtures.DIAGNOSTICS_SERVER, role="advice", advice="ranking")
    vocabulary = Vocabulary(noun=data["text"]["noun"], short_noun=data["text"]["short_noun"], example_field=data["text"]["example_channel"],
                            terms={**fixtures.RIG_VOCABULARY.terms, "value": value})
    declaration = Declaration(asset_id=data["asset_id"], kind="recorded_levels", surface=None, vocabulary=vocabulary, rows=rows,
                              listed_rows={}, server_roles={}, cards=(), advice={"rank_root_causes": fixtures._advice("ranking")},
                              events=fixtures.RIG_EVENTS, rules=fixtures.RecordedLevelsRules(), binding=None)
    return declaration, {s: [SimpleNamespace(**tool) for tool in tools] for s, tools in listing.items()}, facts_of(data)


def declarations():
    """name -> (declaration, listing, facts) of the fixture declarations of the freeze and of the tiny asset; facts None where
    the asset has no facts tool."""
    rig_facts = facts_of(asset_data(DATASETS / "hyd_1" / "asset.json"))
    out = {name: (declaration, listing, rig_facts if declaration.asset_id == "hyd_1" else None)
           for name, (declaration, listing) in fixtures.declarations().items()}
    out["tiny"] = tiny_declaration()
    return out


def without_example(declaration):
    return replace(declaration, vocabulary=replace(declaration.vocabulary, example_field=None))
